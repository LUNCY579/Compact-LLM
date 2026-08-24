#!/usr/bin/env python3
"""Audit small capability screens for degenerate answer baselines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(run_dir: Path, evaluation_path: Path) -> dict[str, Any]:
    run = run_dir.resolve(strict=True)
    cases = _jsonl(run / "cases.jsonl")
    evaluation = {str(row["case_id"]): row for row in _jsonl(evaluation_path.resolve(strict=True))}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_family[str(row["task_family"])].append(row)
    families: dict[str, Any] = {}
    invalid: list[str] = []
    for family, rows in sorted(by_family.items()):
        answers = [str(evaluation[str(row["case_id"])]["answer"]) for row in rows]
        distribution = Counter(answers)
        majority = max(distribution.values()) / len(answers)
        degenerate = len(distribution) < 2
        if degenerate:
            invalid.append(family)
        families[family] = {
            "clusters": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "answer_distribution": dict(sorted(distribution.items())),
            "majority_answer_baseline_accuracy": majority,
            "at_least_two_distinct_answers": not degenerate,
            "valid_for_capability_inference": not degenerate,
        }
    valid_rows = [row for row in cases if str(row["task_family"]) not in invalid]
    result = {
        "schema_version": "1.0",
        "status": "FAIL" if invalid else "PASS",
        "classification": "CAPABILITY_SCREEN_VALIDITY_AUDIT",
        "families": families,
        "invalid_families": invalid,
        "valid_rows": len(valid_rows),
        "valid_correct": sum(bool(row["correct"]) for row in valid_rows),
        "valid_accuracy": (
            sum(bool(row["correct"]) for row in valid_rows) / len(valid_rows)
            if valid_rows
            else None
        ),
        "source": {
            "cases_sha256": sha256_file(run / "cases.jsonl"),
            "evaluation_sha256": sha256_file(evaluation_path),
        },
        "manual_design_note": (
            "R7 v1 also exposes the decisive LOCKED fact directly and its rendered solver "
            "branches on that sentence; a balanced multi-rule R7 v2 is required."
        ),
        "claim_limit": "Raw predictions remain preserved; invalid family cells do not support capability claims.",
    }
    atomic_write_json(run / "validity_audit.json", result)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["validity_audit"] = {
        "status": result["status"],
        "invalid_families": invalid,
        "valid_rows": result["valid_rows"],
        "valid_correct": result["valid_correct"],
        "valid_accuracy": result["valid_accuracy"],
    }
    if invalid:
        summary["classification"] = "BOUNDED_FULL_CACHE_CAPABILITY_SCREEN_WITH_INVALID_FAMILY"
    atomic_write_json(summary_path, summary)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", {})["validity_audit"] = "validity_audit.json"
    note = "R7 raw 2/2 is invalid/uninformative because both selected answers are NO and the v1 rule task has a direct shortcut."
    if note not in manifest.setdefault("notes", []):
        manifest["notes"].append(note)
    atomic_write_json(manifest_path, manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=LAB_ROOT / "results" / "runs" / "halo_r5_r8_capability_screen_v1",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=LAB_ROOT / "results" / "benchmarks" / "halo_r1_r8_ladder_v1" / "evaluation.jsonl",
    )
    args = parser.parse_args()
    result = audit(args.run, args.evaluation)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
