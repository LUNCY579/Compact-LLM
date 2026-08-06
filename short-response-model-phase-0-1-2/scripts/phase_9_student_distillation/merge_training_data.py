from __future__ import annotations

"""Deterministically mix real reviewed curriculum with synthetic foundation rows."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json, write_jsonl
from scripts.phase_9_student_distillation.phase9_metrics import decision_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix Phase 9 real and synthetic training rows.")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--primary-repeat", type=int, default=3)
    parser.add_argument("--foundation-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def replicated(rows: list[dict[str, Any]], *, source: str, count: int) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("Repeat counts must be non-negative")
    output: list[dict[str, Any]] = []
    for repeat in range(count):
        for row in rows:
            base_id = str(row.get("id") or "")
            if not base_id:
                raise ValueError(f"{source} row is missing id")
            copy = dict(row)
            copy["id"] = f"mix::{source}::{repeat}::{base_id}"
            copy["metadata"] = {
                **(row.get("metadata") or {}),
                "mixture_source": source,
                "mixture_repeat": repeat,
            }
            output.append(copy)
    return output


def main() -> None:
    args = parse_args()
    primary = read_jsonl(args.primary)
    foundation = read_jsonl(args.foundation)
    if not primary:
        raise ValueError("Primary human-reviewed curriculum is empty")
    if not foundation:
        raise ValueError("Synthetic foundation curriculum is empty")
    rows = [
        *replicated(primary, source="human_reviewed", count=args.primary_repeat),
        *replicated(foundation, source="synthetic_foundation", count=args.foundation_repeat),
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Mixed curriculum produced duplicate IDs")
    write_jsonl(args.output, rows)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "1.0",
        "phase": "9_student_distillation",
        "pipeline_revision": "phase9_behavior_curriculum_v2",
        "primary": str(args.primary),
        "primary_sha256": sha256(args.primary),
        "foundation": str(args.foundation),
        "foundation_sha256": sha256(args.foundation),
        "primary_repeat": args.primary_repeat,
        "foundation_repeat": args.foundation_repeat,
        "seed": args.seed,
        "row_count": len(rows),
        "source_counts": dict(Counter(str((row.get("metadata") or {}).get("mixture_source")) for row in rows)),
        "decision_class_counts": dict(Counter(decision_class(row) for row in rows)),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"rows": len(rows), "sources": manifest["source_counts"]}, indent=2))


if __name__ == "__main__":
    main()
