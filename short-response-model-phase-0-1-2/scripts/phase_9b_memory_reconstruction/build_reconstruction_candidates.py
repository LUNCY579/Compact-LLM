from __future__ import annotations

"""Build Phase 9B reconstruction candidates from Phase 9 prompt records.

This stage does not silently certify model-generated labels. It creates a
reviewable packet with a conservative metadata baseline and, where available,
explicit author-provided reconstruction annotations. A teacher-generation step
may add another candidate, but human acceptance remains mandatory before
training data is materialized.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json
from scripts.phase_9_student_distillation.phase9_common import ABSTENTION_TEXT, CONFLICT_TEXT, validate_prompt_record
from scripts.phase_9_student_distillation.phase9_metrics import decision_class
from scripts.phase_9b_memory_reconstruction.phase9b_common import (
    PIPELINE_REVISION,
    deterministic_reconstruction,
    memory_from_phase9_passage,
    normalize_reconstruction_target,
    validate_memory_record,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build human-review candidates for Phase 9B.")
    parser.add_argument("--source-prompts", type=Path, required=True)
    parser.add_argument("--memory-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _answer(row: dict[str, Any], klass: str) -> str:
    for key in ("approved_answer", "reference_answer", "target_answer", "answer"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    messages = row.get("messages") or []
    if messages and messages[-1].get("role") == "assistant":
        value = str(messages[-1].get("content") or "").strip()
        if value:
            return value
    if klass == "unsupported":
        return ABSTENTION_TEXT
    if klass == "conflict":
        return CONFLICT_TEXT
    return ""


def _load_memory_catalog(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {row["memory_id"]: row for row in (validate_memory_record(item) for item in read_jsonl(path))}


def _memory_for_item(item: Any, catalog: dict[str, dict[str, Any]], *, default_type: str) -> dict[str, Any]:
    if isinstance(item, str):
        if item not in catalog:
            raise ValueError(f"Unknown memory ID {item!r}; provide it in --memory-file")
        return dict(catalog[item])
    if not isinstance(item, dict):
        raise ValueError(f"Memory/evidence item must be a string ID or object, got {type(item).__name__}")
    identifier = str(
        item.get("memory_id")
        or item.get("record_id")
        or item.get("document_id")
        or item.get("procedure_id")
        or item.get("id")
        or ""
    ).strip()
    if identifier and identifier in catalog:
        merged = {**catalog[identifier], **item}
        return validate_memory_record(merged)
    return memory_from_phase9_passage(item, default_type=default_type)


def _memories(row: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    raw = row.get("memories")
    if raw is not None:
        return [_memory_for_item(item, catalog, default_type="factual") for item in raw]
    result = [
        _memory_for_item(item, catalog, default_type="factual")
        for item in (row.get("evidence") or [])
    ]
    result.extend(
        _memory_for_item(item, catalog, default_type="procedure")
        for item in (row.get("procedures") or [])
    )
    return result


def _provisional_target(row: dict[str, Any], memories: list[dict[str, Any]], klass: str) -> dict[str, Any]:
    explicit = row.get("approved_reconstruction") or row.get("memory_reconstruction")
    if isinstance(explicit, dict):
        return normalize_reconstruction_target(explicit, record_id=str(row.get("id")))
    if klass == "unsupported":
        return normalize_reconstruction_target(
            {
                "decision": "insufficient",
                "source_ids": [],
                "applicable_facts": [],
                "invalid_assumptions": [],
                "guidance": "",
                "confidence": "medium",
            }
        )
    if klass == "conflict":
        return normalize_reconstruction_target(
            {
                "decision": "conflict",
                "source_ids": [memory["memory_id"] for memory in memories],
                "applicable_facts": [],
                "invalid_assumptions": [],
                "guidance": "The retrieved sources conflict and require clarification or abstention.",
                "confidence": "medium",
            }
        )
    baseline = deterministic_reconstruction(
        current_state=dict(row.get("current_state") or (row.get("metadata") or {}).get("current_state") or {}),
        memories=memories,
    )
    answer = _answer(row, klass)
    if baseline["decision"] in {"accept", "adapt"} and answer:
        # For the provisional review candidate only, the accepted answer is a
        # compact guidance seed. A human or teacher must verify that it is
        # actually supported before the row becomes training data.
        baseline = {**baseline, "guidance": answer}
    return normalize_reconstruction_target(baseline, record_id=str(row.get("id")))


def build_candidates(source_rows: list[dict[str, Any]], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for original in source_rows:
        row = validate_prompt_record(original)
        klass = decision_class(row)
        memories = _memories(row, catalog)
        # Social/user-memory rows are valuable Phase 9B material when they carry
        # explicit memory. Ordinary memory-free conversation stays in Phase 9.
        if klass == "social" and not memories:
            continue
        current_state = dict(row.get("current_state") or (row.get("metadata") or {}).get("current_state") or {})
        answer = _answer(row, klass)
        if not answer:
            # Training cannot proceed without a human-approved final target.
            answer = ABSTENTION_TEXT if klass in {"unsupported", "unknown"} else ""
        candidate = {
            "schema_version": "1.0",
            "pipeline_revision": PIPELINE_REVISION,
            "id": row["id"],
            "topic_id": row["topic_id"],
            "question": row["question"],
            "mode": row["mode"],
            "current_state": current_state,
            "memories": memories,
            "support_document_ids": list(row.get("support_document_ids") or []),
            "source_prompt_record": row,
            "provisional_reconstruction": _provisional_target(row, memories, klass),
            "teacher_reconstruction": None,
            "review_status": "pending",
            "selected_reconstruction_source": None,
            "approved_reconstruction": None,
            "approved_answer": answer,
            "review_notes": "",
        }
        output.append(candidate)

        # Explicit counterfactual variants are safe to expand because their
        # current states and expected targets are author-supplied, not invented.
        for index, counterfactual in enumerate(row.get("counterfactuals") or []):
            if not isinstance(counterfactual, dict):
                raise ValueError(f"{row['id']} counterfactual {index} must be an object")
            target = counterfactual.get("approved_reconstruction") or counterfactual.get("reconstruction")
            if not isinstance(target, dict):
                raise ValueError(f"{row['id']} counterfactual {index} requires a reconstruction target")
            output.append(
                {
                    **candidate,
                    "id": f"{row['id']}::counterfactual_{index}",
                    "topic_id": str(counterfactual.get("topic_id") or row["topic_id"]),
                    "question": str(counterfactual.get("question") or row["question"]),
                    "current_state": dict(counterfactual.get("current_state") or {}),
                    "provisional_reconstruction": normalize_reconstruction_target(
                        target, record_id=f"{row['id']}::counterfactual_{index}"
                    ),
                    "approved_answer": str(counterfactual.get("approved_answer") or answer),
                    "review_status": "pending",
                    "approved_reconstruction": None,
                }
            )
    return output


def review_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 9B memory-reconstruction review packet",
        "",
        "Every row is pending. Compare the current state with each memory's source state and validity conditions. "
        "Select or edit a reconstruction only when its final answer is fully supported.",
        "",
    ]
    for row in rows:
        provisional = row["provisional_reconstruction"]
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- Topic: `{row['topic_id']}`",
                f"- Mode: `{row['mode']}`",
                f"- Question: {row['question']}",
                f"- Current state: `{json.dumps(row['current_state'], ensure_ascii=False, sort_keys=True)}`",
                f"- Proposed decision: `{provisional['decision']}`",
                f"- Proposed guidance: {provisional['guidance'] or '(empty)'}",
                f"- Proposed answer: {row['approved_answer'] or '(missing; must be supplied)' }",
                "- Memory IDs: " + ", ".join(memory["memory_id"] for memory in row["memories"]),
                "",
                "Review action in JSONL: set `review_status` to `accepted`, set `approved_reconstruction`, "
                "and verify `approved_answer`; otherwise set `review_status` to `rejected`.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source_prompts)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    catalog = _load_memory_catalog(args.memory_file)
    candidates = build_candidates(source_rows, catalog)
    write_jsonl(args.output, candidates)
    packet = args.review_packet or args.output.with_suffix(".review.md")
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(review_markdown(candidates), encoding="utf-8")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "pipeline_revision": PIPELINE_REVISION,
            "source_prompts": str(args.source_prompts),
            "source_prompts_sha256": sha256(args.source_prompts),
            "memory_file": str(args.memory_file) if args.memory_file else None,
            "memory_file_sha256": sha256(args.memory_file) if args.memory_file else None,
            "candidate_count": len(candidates),
            "output": str(args.output),
            "review_packet": str(packet),
        },
    )
    print(args.output)
    print(packet)


if __name__ == "__main__":
    main()
