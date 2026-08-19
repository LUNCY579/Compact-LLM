from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_jsonl, write_jsonl  # noqa: E402


def export_view(root: Path, split: str, view: str, output: Path | None) -> Path:
    if view == "full_context":
        source = root / "data" / "full_context" / f"{split}.jsonl"
        message_field = "messages"
    elif view == "split_next_step":
        source = root / "data" / "split_transitions" / f"{split}.jsonl"
        message_field = "messages"
    else:
        source = root / "data" / "split_transitions" / f"{split}.jsonl"
        message_field = "stateful_rollout_messages"

    if output is None:
        output = root / "data" / "derived_views" / f"{view}_{split}_messages.jsonl"

    rows: list[dict[str, Any]] = []
    for row in read_jsonl(source):
        out = {
            "messages": row[message_field],
            "problem_id": row["problem_id"],
            "category": row["category"],
            "domain": row["domain"],
            "split": row["split"],
            "view": view,
        }
        if "transition_id" in row:
            out["transition_id"] = row["transition_id"]
            out["step_index"] = row["step_index"]
            out["total_steps"] = row["total_steps"]
        rows.append(out)
    write_jsonl(output, rows)
    print(f"Wrote {len(rows):,} rows to {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export minimal trainer-ready message JSONL views.")
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--view", choices=["full_context", "split_next_step", "split_stateful"], required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_view(args.root.resolve(), args.split, args.view, args.output)
