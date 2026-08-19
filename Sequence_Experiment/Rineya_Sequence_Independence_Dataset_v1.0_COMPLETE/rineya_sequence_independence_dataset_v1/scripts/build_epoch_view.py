from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_jsonl, stable_key, write_jsonl  # noqa: E402


def offset_for_problem(problem_id: str, count: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{problem_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def build_views(root: Path, split: str, epochs: int, seed: int, message_view: str) -> list[Path]:
    source = root / "data" / "split_transitions" / f"{split}.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(source):
        grouped[row["problem_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["step_index"])

    out_dir = root / "data" / "derived_views" / f"{split}_one_transition_per_problem"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for epoch in range(epochs):
        selected: list[dict[str, Any]] = []
        for problem_id, rows in grouped.items():
            offset = offset_for_problem(problem_id, len(rows), seed)
            chosen = rows[(offset + epoch) % len(rows)]
            messages = chosen["messages"] if message_view == "next_step" else chosen["stateful_rollout_messages"]
            selected.append(
                {
                    "problem_id": chosen["problem_id"],
                    "transition_id": chosen["transition_id"],
                    "category": chosen["category"],
                    "domain": chosen["domain"],
                    "step_index": chosen["step_index"],
                    "total_steps": chosen["total_steps"],
                    "messages": messages,
                    "view": message_view,
                    "epoch_view": epoch,
                }
            )
        selected.sort(key=lambda r: stable_key(r["transition_id"], f"epoch-view|{seed}|{epoch}"))
        out_path = out_dir / f"epoch_{epoch:03d}_{message_view}.jsonl"
        write_jsonl(out_path, selected)
        outputs.append(out_path)
        print(f"Wrote {len(selected):,} rows to {out_path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build problem-count-matched epoch views from the split-transition dataset.")
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--message-view", choices=["next_step", "stateful_rollout"], default="next_step")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_views(args.root.resolve(), args.split, args.epochs, args.seed, args.message_view)
