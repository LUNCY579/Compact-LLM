from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rineya_state_lab.data.validate import validate_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _records(split: str, suffix: str) -> tuple[dict, dict]:
    pid = f"p-{suffix}"
    pair_hash = hashlib.sha256(pid.encode()).hexdigest()
    full = {
        "problem_id": pid,
        "problem": f"What is 1+1? {suffix}",
        "answer": "2",
        "answer_type": "integer",
        "category": "arithmetic",
        "domain": "mathematics",
        "difficulty": 1,
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "2"}],
        "reasoning_steps": ["Add 1 and 1 to get 2."],
        "compact_state_trace": ["No derived facts yet.", "Final answer: 2"],
        "source_pair_sha256": pair_hash,
        "split": split,
    }
    target = {
        "next_reasoning_step": "Add 1 and 1 to get 2.",
        "updated_compact_state": "Final answer: 2",
        "done": True,
        "final_answer": "2",
    }
    transition = {
        "transition_id": f"{pid}::step_01",
        "problem_id": pid,
        "problem": full["problem"],
        "answer_type": "integer",
        "category": "arithmetic",
        "domain": "mathematics",
        "difficulty": 1,
        "current_state": "No derived facts yet.",
        "target_step": "Add 1 and 1 to get 2.",
        "next_state": "Final answer: 2",
        "done": True,
        "final_answer": "2",
        "step_index": 1,
        "total_steps": 1,
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "stateful_rollout_messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": json.dumps(target)}],
        "stateful_rollout_target": json.dumps(target),
        "source_pair_sha256": pair_hash,
        "split": split,
    }
    return full, transition


def test_validator_accepts_aligned_tiny_dataset(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ["train", "dev", "test"]:
        full, transition = _records(split, split)
        _write_jsonl(root / f"data/full_context/{split}.jsonl", [full])
        _write_jsonl(root / f"data/split_transitions/{split}.jsonl", [transition])
    report = validate_dataset(root, strict=True)
    assert report["passed"]
    assert report["unique_problems"] == 3
