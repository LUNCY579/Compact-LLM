from __future__ import annotations

"""Build a deterministic, category-balanced Phase 2B probe from development data.

The 50-case probe is intentionally small enough for every-layer iteration. It is
not a release benchmark and never reads the sealed split. Promoted frontier
models must still receive full phase-level evaluation and human review.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json, write_jsonl


def category_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(row.get("category") or metadata.get("reply_class") or "uncategorized")


def select_balanced(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin categories after deterministic per-category shuffling."""

    if limit < 1:
        raise ValueError("limit must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[category_of(row)].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)

    categories = sorted(buckets)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < min(limit, len(rows)):
        made_progress = False
        for category in categories:
            values = buckets[category]
            if cursor < len(values):
                selected.append(values[cursor])
                made_progress = True
                if len(selected) >= limit:
                    break
        if not made_progress:
            break
        cursor += 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 2B structural probe.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/dev/dev.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/phase_2_structural_reduction/probe_50.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/phase_2_structural_reduction/probe_50_manifest.json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    selected = select_balanced(rows, args.limit, args.seed)
    write_jsonl(args.output, selected)
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[category_of(row)] += 1
    write_json(
        args.manifest,
        {
            "source": str(args.input),
            "output": str(args.output),
            "seed": args.seed,
            "requested_limit": args.limit,
            "selected_count": len(selected),
            "category_counts": dict(sorted(counts.items())),
            "sealed_data_used": False,
            "purpose": "Fast trend detection for every-layer and progressive depth reduction.",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
