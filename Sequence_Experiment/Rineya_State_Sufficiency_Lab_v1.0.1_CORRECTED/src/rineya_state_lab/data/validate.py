from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils import atomic_write_json, iter_jsonl, normalize_casefold, sha256_file, timed, utc_now_iso
from .layout import DatasetLayout, discover_dataset_root


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


_REQUIRED_FULL = {
    "problem_id",
    "problem",
    "answer",
    "answer_type",
    "category",
    "domain",
    "difficulty",
    "messages",
    "reasoning_steps",
    "compact_state_trace",
    "source_pair_sha256",
    "split",
}
_REQUIRED_TRANSITION = {
    "transition_id",
    "problem_id",
    "problem",
    "answer_type",
    "current_state",
    "target_step",
    "next_state",
    "done",
    "step_index",
    "total_steps",
    "messages",
    "stateful_rollout_messages",
    "source_pair_sha256",
    "split",
}


def _message_shape_ok(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and len(messages) >= 2
        and all(isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"} and isinstance(m.get("content"), str) for m in messages)
        and messages[-1].get("role") == "assistant"
    )


def _record_missing(record: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required - record.keys())


def validate_dataset(data_root: str | Path, *, strict: bool = False, output: str | Path | None = None) -> dict[str, Any]:
    root = discover_dataset_root(data_root)
    layout = DatasetLayout(root)
    layout.assert_exists()
    issues: list[ValidationIssue] = []
    full: dict[str, dict[str, Any]] = {}
    problem_split: dict[str, str] = {}
    normalized_problem_owner: dict[str, str] = {}
    counts: dict[str, Any] = {"full": {}, "transitions": {}}
    category_counts: collections.Counter[str] = collections.Counter()
    domain_counts: collections.Counter[str] = collections.Counter()
    answer_type_counts: collections.Counter[str] = collections.Counter()

    def add(severity: str, code: str, message: str, **context: Any) -> None:
        issues.append(ValidationIssue(severity, code, message, context))

    with timed() as timing:
        for split in ("train", "dev", "test"):
            count = 0
            for line_number, record in enumerate(iter_jsonl(layout.full_file(split)), start=1):
                count += 1
                missing = _record_missing(record, _REQUIRED_FULL)
                if missing:
                    add("error", "full_missing_fields", f"Missing fields {missing}", split=split, line=line_number)
                    continue
                pid = str(record["problem_id"])
                if pid in full:
                    add("error", "duplicate_problem_id", f"Duplicate problem_id {pid}", split=split, line=line_number)
                    continue
                if record.get("split") != split:
                    add("error", "full_split_mismatch", f"Record {pid} says split={record.get('split')!r}", expected=split)
                if not _message_shape_ok(record.get("messages")):
                    add("error", "invalid_full_messages", f"Invalid messages for {pid}", split=split, line=line_number)
                steps = record.get("reasoning_steps")
                trace = record.get("compact_state_trace")
                if not isinstance(steps, list) or not steps:
                    add("error", "invalid_reasoning_steps", f"Invalid reasoning_steps for {pid}")
                if not isinstance(trace, list) or not isinstance(steps, list) or len(trace) != len(steps) + 1:
                    add(
                        "error",
                        "state_trace_length",
                        f"compact_state_trace must have one more item than reasoning_steps for {pid}",
                        trace_length=len(trace) if isinstance(trace, list) else None,
                        step_length=len(steps) if isinstance(steps, list) else None,
                    )
                normalized = normalize_casefold(record["problem"])
                owner = normalized_problem_owner.get(normalized)
                if owner is not None:
                    add("error", "duplicate_problem_text", f"Problem text duplicated between {owner} and {pid}")
                else:
                    normalized_problem_owner[normalized] = pid
                full[pid] = record
                problem_split[pid] = split
                category_counts[str(record["category"])] += 1
                domain_counts[str(record["domain"])] += 1
                answer_type_counts[str(record["answer_type"])] += 1
            counts["full"][split] = count

        chains: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
        transition_id_seen: set[str] = set()
        for split in ("train", "dev", "test"):
            count = 0
            for line_number, record in enumerate(iter_jsonl(layout.transition_file(split)), start=1):
                count += 1
                missing = _record_missing(record, _REQUIRED_TRANSITION)
                if missing:
                    add("error", "transition_missing_fields", f"Missing fields {missing}", split=split, line=line_number)
                    continue
                tid = str(record["transition_id"])
                pid = str(record["problem_id"])
                if tid in transition_id_seen:
                    add("error", "duplicate_transition_id", f"Duplicate transition_id {tid}", split=split, line=line_number)
                transition_id_seen.add(tid)
                if pid not in full:
                    add("error", "orphan_transition", f"Transition {tid} has no matching full record")
                    continue
                source = full[pid]
                if problem_split[pid] != split or record.get("split") != split:
                    add("error", "transition_split_mismatch", f"Transition {tid} is assigned to the wrong split")
                if str(record.get("problem")) != str(source.get("problem")):
                    add("error", "transition_problem_mismatch", f"Problem text mismatch for {tid}")
                if record.get("source_pair_sha256") != source.get("source_pair_sha256"):
                    add("error", "pair_hash_mismatch", f"source_pair_sha256 mismatch for {tid}")
                if record.get("answer_type") != source.get("answer_type"):
                    add("error", "answer_type_mismatch", f"answer_type mismatch for {tid}")
                if not _message_shape_ok(record.get("messages")) or not _message_shape_ok(record.get("stateful_rollout_messages")):
                    add("error", "invalid_transition_messages", f"Invalid messages for {tid}")
                try:
                    index = int(record["step_index"])
                    total = int(record["total_steps"])
                except Exception:
                    add("error", "invalid_step_index", f"Non-integer step index for {tid}")
                    continue
                if index < 1 or total < 1 or index > total:
                    add("error", "step_range", f"Invalid step range for {tid}", index=index, total=total)
                if index in chains[pid]:
                    add("error", "duplicate_step_index", f"Duplicate step {index} for {pid}")
                chains[pid][index] = record
            counts["transitions"][split] = count

        for pid, source in full.items():
            chain = chains.get(pid, {})
            expected_total = len(source.get("reasoning_steps", []))
            expected_indexes = set(range(1, expected_total + 1))
            actual_indexes = set(chain)
            if actual_indexes != expected_indexes:
                add(
                    "error",
                    "incomplete_chain",
                    f"Transition chain is incomplete for {pid}",
                    expected=sorted(expected_indexes),
                    actual=sorted(actual_indexes),
                )
                continue
            trace = source["compact_state_trace"]
            steps = source["reasoning_steps"]
            for index in sorted(chain):
                row = chain[index]
                if int(row["total_steps"]) != expected_total:
                    add("error", "total_steps_mismatch", f"total_steps mismatch for {row['transition_id']}")
                if normalize_casefold(row["current_state"]) != normalize_casefold(trace[index - 1]):
                    add("error", "current_state_mismatch", f"current_state mismatch for {row['transition_id']}")
                if normalize_casefold(row["next_state"]) != normalize_casefold(trace[index]):
                    add("error", "next_state_mismatch", f"next_state mismatch for {row['transition_id']}")
                if normalize_casefold(row["target_step"]) != normalize_casefold(steps[index - 1]):
                    add("error", "target_step_mismatch", f"target_step mismatch for {row['transition_id']}")
                should_done = index == expected_total
                if bool(row["done"]) != should_done:
                    add("error", "done_flag_mismatch", f"done mismatch for {row['transition_id']}")
                if should_done:
                    if normalize_casefold(row.get("final_answer")) != normalize_casefold(source.get("answer")):
                        add("error", "final_answer_mismatch", f"final answer mismatch for {row['transition_id']}")
                elif row.get("final_answer") is not None:
                    add("error", "premature_final_answer", f"Non-final transition has final_answer for {row['transition_id']}")
                target = row.get("stateful_rollout_target")
                if target is not None:
                    try:
                        parsed = json.loads(target)
                    except Exception:
                        add("error", "invalid_stateful_target_json", f"Invalid stateful target for {row['transition_id']}")
                    else:
                        expected_keys = {"next_reasoning_step", "updated_compact_state", "done", "final_answer"}
                        if set(parsed) != expected_keys:
                            add("error", "stateful_target_keys", f"Unexpected keys for {row['transition_id']}")

        # Detect the explicit non-sequential ordering property in each split file.
        adjacency: dict[str, dict[str, int | float]] = {}
        for split in ("train", "dev", "test"):
            previous: str | None = None
            adjacent_same = 0
            total_pairs = 0
            for row in iter_jsonl(layout.transition_file(split)):
                pid = str(row["problem_id"])
                if previous is not None:
                    total_pairs += 1
                    if pid == previous:
                        adjacent_same += 1
                previous = pid
            adjacency[split] = {
                "adjacent_same_problem_pairs": adjacent_same,
                "total_adjacent_pairs": total_pairs,
                "rate": adjacent_same / total_pairs if total_pairs else 0.0,
            }
            if adjacent_same > 0:
                add(
                    "warning",
                    "adjacent_transition_rows",
                    f"Split {split} contains {adjacent_same} adjacent rows from the same problem",
                )

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "validated_at": utc_now_iso(),
        "dataset_root": str(root),
        "passed": not errors,
        "strict": strict,
        "counts": counts,
        "unique_problems": len(full),
        "unique_transitions": len(transition_id_seen),
        "category_counts": dict(sorted(category_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "adjacency": adjacency,
        "file_hashes": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [
                *(layout.full_file(split) for split in ("train", "dev", "test")),
                *(layout.transition_file(split) for split in ("train", "dev", "test")),
            ]
        },
        "issue_summary": {"errors": len(errors), "warnings": len(warnings)},
        "issues": [issue.__dict__ for issue in issues[:1000]],
        "issues_truncated": len(issues) > 1000,
        "elapsed_seconds": timing["seconds"],
    }
    if output is not None:
        atomic_write_json(output, report)
    if strict and errors:
        examples = "\n".join(f"[{i.code}] {i.message}" for i in errors[:20])
        raise ValueError(f"Dataset validation failed with {len(errors)} errors:\n{examples}")
    return report
