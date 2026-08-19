from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    full_response,
    normalize_problem,
    read_jsonl,
    sha256_text,
)
from generators_math import MATH_GENERATORS  # noqa: E402
from generators_reasoning import REASONING_GENERATORS  # noqa: E402

ALL_GENERATORS = {**MATH_GENERATORS, **REASONING_GENERATORS}


class ValidationError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate(root: Path, *, regenerate: bool = True) -> dict[str, Any]:
    stats_path = root / "data" / "manifests" / "dataset_stats.json"
    require(stats_path.exists(), f"Missing {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    full_map: dict[str, dict[str, Any]] = {}
    normalized_prompts: dict[str, str] = {}
    full_counts = Counter()
    category_split_counts = Counter()

    required_full_fields = {
        "problem_id", "split", "domain", "category", "difficulty", "problem", "answer",
        "full_response", "reasoning_steps", "compact_state_trace", "step_count", "messages",
        "source_pair_sha256", "generation_seed", "generator",
    }

    for split in ("train", "dev", "test"):
        path = root / "data" / "full_context" / f"{split}.jsonl"
        require(path.exists(), f"Missing {path}")
        for row_num, row in enumerate(read_jsonl(path), 1):
            missing = required_full_fields - row.keys()
            require(not missing, f"{path}:{row_num} missing fields {sorted(missing)}")
            pid = row["problem_id"]
            require(pid not in full_map, f"Duplicate problem_id {pid}")
            require(row["split"] == split, f"Split mismatch for {pid}")
            require(row["variant"] == "full_context", f"Variant mismatch for {pid}")
            require(isinstance(row["problem"], str) and row["problem"].strip(), f"Empty problem for {pid}")
            require(isinstance(row["answer"], str) and row["answer"].strip(), f"Empty answer for {pid}")
            require(len(row["reasoning_steps"]) == row["step_count"], f"step_count mismatch for {pid}")
            require(len(row["compact_state_trace"]) == row["step_count"] + 1, f"state trace mismatch for {pid}")
            require(row["full_response"] == "Reasoning:\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(row["reasoning_steps"], 1)), f"Full response assembly mismatch for {pid}")
            require(row["reasoning_steps"][-1] == f"Therefore, the answer is {row['answer']}.", f"Final reasoning step mismatch for {pid}")
            require(row["compact_state_trace"][-1] == f"Final answer: {row['answer']}", f"Final state mismatch for {pid}")
            require(row["messages"] == [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": row["full_response"]},
            ], f"Messages mismatch for {pid}")
            require(row["source_pair_sha256"] == sha256_text(row["problem"] + "\n" + row["full_response"]), f"Source hash mismatch for {pid}")
            normalized = normalize_problem(row["problem"])
            require(normalized not in normalized_prompts, f"Duplicate normalized problem text for {pid} and {normalized_prompts.get(normalized)}")
            normalized_prompts[normalized] = pid
            require(row["category"] in ALL_GENERATORS, f"Unknown category {row['category']} for {pid}")
            expected_domain, _ = ALL_GENERATORS[row["category"]]
            require(row["domain"] == expected_domain, f"Domain mismatch for {pid}")
            full_map[pid] = row
            full_counts[split] += 1
            category_split_counts[(row["category"], split)] += 1

            if regenerate:
                local_match = re.search(r"_(\d{4})$", pid)
                require(local_match is not None, f"Cannot parse local index from {pid}")
                local_idx = int(local_match.group(1))
                _, generator = ALL_GENERATORS[row["category"]]
                generated = generator(random.Random(row["generation_seed"]), local_idx)
                require(generated.problem == row["problem"], f"Regenerated problem mismatch for {pid}")
                require(generated.answer == row["answer"], f"Regenerated answer mismatch for {pid}")
                require(full_response(generated.steps) == row["full_response"], f"Regenerated response mismatch for {pid}")
                require([s.state_before for s in generated.steps] + [generated.steps[-1].state_after] == row["compact_state_trace"], f"Regenerated state trace mismatch for {pid}")

    require(len(full_map) == stats["total_unique_problems"], f"Problem total mismatch: {len(full_map)} vs {stats['total_unique_problems']}")
    require(dict(full_counts) == stats["problem_splits"], f"Problem split counts mismatch: {full_counts}")

    transition_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transition_ids: set[str] = set()
    transition_counts = Counter()
    required_transition_fields = {
        "transition_id", "problem_id", "split", "domain", "category", "step_index", "total_steps",
        "problem", "current_state", "target_step", "next_state", "done", "messages",
        "source_pair_sha256", "sequence_free_training", "state_construction",
        "stateful_rollout_messages", "stateful_rollout_target",
    }

    for split in ("train", "dev", "test"):
        path = root / "data" / "split_transitions" / f"{split}.jsonl"
        require(path.exists(), f"Missing {path}")
        previous_pid = None
        for row_num, row in enumerate(read_jsonl(path), 1):
            missing = required_transition_fields - row.keys()
            require(not missing, f"{path}:{row_num} missing fields {sorted(missing)}")
            tid = row["transition_id"]
            pid = row["problem_id"]
            require(tid not in transition_ids, f"Duplicate transition_id {tid}")
            transition_ids.add(tid)
            require(pid in full_map, f"Transition references unknown problem {pid}")
            full = full_map[pid]
            require(row["split"] == split == full["split"], f"Transition split mismatch for {tid}")
            require(row["variant"] == "split_transition", f"Transition variant mismatch for {tid}")
            require(row["problem"] == full["problem"], f"Transition problem mismatch for {tid}")
            require(row["category"] == full["category"] and row["domain"] == full["domain"], f"Transition metadata mismatch for {tid}")
            require(row["source_pair_sha256"] == full["source_pair_sha256"], f"Transition source hash mismatch for {tid}")
            require(row["sequence_free_training"] is True, f"sequence_free_training must be true for {tid}")
            require(row["state_construction"] == "derived_from_complete_canonical_trajectory", f"State construction mismatch for {tid}")
            require(previous_pid != pid, f"Adjacent transition rows share problem_id in {path}: {pid}")
            previous_pid = pid
            transition_groups[pid].append(row)
            transition_counts[split] += 1

    require(sum(transition_counts.values()) == stats["total_transition_records"], "Transition total mismatch")
    require(dict(transition_counts) == stats["transition_splits"], f"Transition split counts mismatch: {transition_counts}")

    for pid, full in full_map.items():
        rows = sorted(transition_groups[pid], key=lambda row: row["step_index"])
        require(len(rows) == full["step_count"], f"Transition count mismatch for {pid}")
        require([r["step_index"] for r in rows] == list(range(1, full["step_count"] + 1)), f"Non-contiguous step indexes for {pid}")
        for idx, row in enumerate(rows):
            require(row["total_steps"] == full["step_count"], f"total_steps mismatch for {row['transition_id']}")
            require(row["current_state"] == full["compact_state_trace"][idx], f"current_state mismatch for {row['transition_id']}")
            require(row["target_step"] == full["reasoning_steps"][idx], f"target_step mismatch for {row['transition_id']}")
            require(row["next_state"] == full["compact_state_trace"][idx + 1], f"next_state mismatch for {row['transition_id']}")
            require(row["done"] is (idx == len(rows) - 1), f"done flag mismatch for {row['transition_id']}")
            require(row["final_answer"] == (full["answer"] if row["done"] else None), f"final_answer mismatch for {row['transition_id']}")
            expected_user = (
                "Original problem:\n"
                f"{full['problem']}\n\n"
                "Current compact state:\n"
                f"{row['current_state']}\n\n"
                "Give exactly the next correct reasoning step. Do not replay the omitted reasoning history."
            )
            require(row["messages"] == [
                {"role": "user", "content": expected_user},
                {"role": "assistant", "content": row["target_step"]},
            ], f"Transition messages mismatch for {row['transition_id']}")
            expected_stateful = json.dumps({
                "done": row["done"],
                "final_answer": row["final_answer"],
                "next_reasoning_step": row["target_step"],
                "updated_compact_state": row["next_state"],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            require(row["stateful_rollout_target"] == expected_stateful, f"Stateful target mismatch for {row['transition_id']}")
            require(row["stateful_rollout_messages"] == [
                {"role": "user", "content": expected_user + "\nReturn a JSON object with next_reasoning_step, updated_compact_state, done, and final_answer."},
                {"role": "assistant", "content": expected_stateful},
            ], f"Stateful messages mismatch for {row['transition_id']}")

    # Exact 90/5/5 balance per category.
    per_category = stats["problems_per_category"]
    expected_per_category = {"train": per_category * 18 // 20, "dev": per_category // 20, "test": per_category // 20}
    for category in ALL_GENERATORS:
        for split, expected in expected_per_category.items():
            actual = category_split_counts[(category, split)]
            require(actual == expected, f"Category split count mismatch: {category}/{split}: {actual} vs {expected}")

    result = {
        "status": "PASS",
        "total_unique_problems": len(full_map),
        "total_transition_records": len(transition_ids),
        "unique_normalized_problems": len(normalized_prompts),
        "problem_split_counts": dict(full_counts),
        "transition_split_counts": dict(transition_counts),
        "categories": len(ALL_GENERATORS),
        "regeneration_check": regenerate,
        "adjacent_same_problem_transitions": 0,
        "full_split_alignment": True,
        "transition_chain_alignment": True,
    }
    report_path = root / "data" / "manifests" / "validation_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the generated paired reasoning datasets.")
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--skip-regeneration", action="store_true", help="Skip deterministic regeneration checks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(args.root.resolve(), regenerate=not args.skip_regeneration)
