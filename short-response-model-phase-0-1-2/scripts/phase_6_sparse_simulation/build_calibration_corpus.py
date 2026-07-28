# Purpose: Builds a non-sealed, deduplicated activation-calibration corpus from approved development data.
from __future__ import annotations

"""Build a non-sealed, decode-oriented calibration corpus for Phase 6.

The builder combines the public pilot benchmark and development SFT records,
strips assistant targets to prevent label leakage, deduplicates prompts, and
preserves category metadata for coverage review.  It never reads data/sealed.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 6 calibration JSONL.")
    parser.add_argument("--source", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "data/calibration/phase_6_calibration.jsonl")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def normalize_prompt(row: dict[str, Any], source: Path, index: int) -> dict[str, Any] | None:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        prompt_messages = [dict(message) for message in messages]
        if prompt_messages[-1].get("role") == "assistant":
            prompt_messages.pop()
        if not prompt_messages:
            return None
        metadata = row.get("metadata") or {}
        return {
            "id": str(row.get("id", f"{source.stem}_{index}")),
            "messages": prompt_messages,
            "category": str(row.get("category") or metadata.get("reply_class") or "uncategorized"),
            "source": str(source),
        }
    text = row.get("prompt") or row.get("text")
    if isinstance(text, str) and text.strip():
        return {
            "id": str(row.get("id", f"{source.stem}_{index}")),
            "text": text,
            "category": str(row.get("category") or "uncategorized"),
            "source": str(source),
        }
    return None


def fingerprint(row: dict[str, Any]) -> str:
    if "messages" in row:
        value = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
    else:
        value = str(row.get("text", ""))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def main() -> None:
    args = parse_args()
    sources = args.source or [
        ROOT / "data/benchmark/shortchatbench_pilot.jsonl",
        ROOT / "data/dev/dev.jsonl",
        ROOT / "data/dev/dev_roleplay_v2.jsonl",
    ]
    for source in sources:
        if "sealed" in source.parts:
            raise SystemExit(f"Sealed data is forbidden for Phase 6 calibration: {source}")
        if not source.exists():
            raise SystemExit(f"Missing calibration source: {source}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for source in sources:
        for index, raw in enumerate(read_jsonl(source)):
            row = normalize_prompt(raw, source, index)
            if row is None:
                continue
            key = fingerprint(row)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(row)
            category = row["category"]
            counts[category] = counts.get(category, 0) + 1

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_records]
    if len(rows) < 50:
        raise RuntimeError(f"Calibration corpus is too small: {len(rows)} rows")
    write_jsonl(args.output, rows)
    write_json(
        args.output.with_suffix(".json"),
        {
            "rows": len(rows),
            "sources": [str(source) for source in sources],
            "category_counts_before_limit": counts,
            "sealed_data_used": False,
            "seed": args.seed,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
