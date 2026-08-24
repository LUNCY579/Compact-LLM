#!/usr/bin/env python3
"""Audit response balance and stop logic for the corrected R7 v2 B0 screen."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import atomic_write_json  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _predicted_label(text: str) -> str:
    values = re.findall(r"\b(?:yes|no)\b", text, flags=re.IGNORECASE)
    return values[-1].upper() if values else "OTHER"


def audit(run_dir: Path) -> dict[str, Any]:
    run = run_dir.resolve(strict=True)
    cases = _rows(run / "cases.jsonl")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if len(cases) != 8 or len({str(row["cluster_id"]) for row in cases}) != 8:
        raise RuntimeError("R7 v2 audit requires the frozen eight-cluster screen")
    true_counts = Counter(str(row["true_answer"]) for row in cases)
    prediction_counts = Counter(_predicted_label(str(row["prediction"])) for row in cases)
    if true_counts != {"YES": 4, "NO": 4}:
        raise RuntimeError("R7 v2 screen answer key is not balanced")
    correct = sum(bool(row["correct"]) for row in cases)
    if correct != int(summary["correct_clusters"]):
        raise RuntimeError("R7 v2 cases/summary accuracy mismatch")
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "POSTHOC_DESCRIPTIVE_RESPONSE_AUDIT",
        "run_id": run.name,
        "independent_clusters": len(cases),
        "true_answer_counts": dict(sorted(true_counts.items())),
        "prediction_counts": {
            label: prediction_counts.get(label, 0) for label in ("YES", "NO", "OTHER")
        },
        "correct_clusters": correct,
        "observed_accuracy": correct / len(cases),
        "constant_answer_baselines": {"always_YES": 0.5, "always_NO": 0.5},
        "observed_exceeds_best_constant_baseline": correct / len(cases) > 0.5,
        "response_pattern": (
            "ALL_YES_COLLAPSE" if prediction_counts == {"YES": len(cases)} else "NONCONSTANT"
        ),
        "advance_gate_passed": bool(summary["advance_gate"]["passed"]),
        "compression_decision": "STOP_NO_R7_V2_COMPRESSION_SCREEN",
        "interpretation": (
            "Halo matched the always-YES baseline: it solved all four YES cases and none of the four NO cases. "
            "This is a full-cache capability failure for the balanced R7 v2 task, not evidence about the "
            "Significance Matrix mechanism."
        ),
        "limitations": [
            "Only eight independent clusters were screened.",
            "The audit is descriptive and post-hoc; no inferential statistic is reported.",
            "Only Halo B0 full cache was executed; no Rineya/R2 or Aegis model and no compression condition ran.",
        ],
    }
    audit_path = run / "result_audit.json"
    atomic_write_json(audit_path, result)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", {})["result_audit"] = "result_audit.json"
    atomic_write_json(manifest_path, manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=LAB_ROOT / "results" / "runs" / "halo_r7_v2_capability_screen",
    )
    args = parser.parse_args()
    result = audit(args.run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
