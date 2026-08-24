#!/usr/bin/env python3
"""Apply multiplicity and specificity gates to adaptive-budget replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation.count_causal import holm_adjust  # noqa: E402
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def analyze(run_dir: Path) -> dict[str, Any]:
    run = run_dir.resolve(strict=True)
    statistics_path = run / "statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    comparisons = dict(statistics["comparisons"])
    adjusted = holm_adjust(
        {strategy: float(values["p_value_two_sided"]) for strategy, values in comparisons.items()}
    )
    for strategy, values in comparisons.items():
        differences = [float(value) for value in values["cluster_differences"].values()]
        values["wins"] = sum(value > 0 for value in differences)
        values["losses"] = sum(value < 0 for value in differences)
        values["ties"] = sum(value == 0 for value in differences)
        values["holm_adjusted_p_value"] = adjusted[strategy]
    entropy_delta = float(comparisons["entropy_proportional"]["mean_difference"])
    shuffled_delta = float(comparisons["shuffled_statistic"]["mean_difference"])
    specificity_passed = entropy_delta > shuffled_delta
    statistics["comparisons"] = comparisons
    statistics["specificity_gate"] = {
        "question": "Does significance-aligned entropy allocation beat its shuffled-statistic control?",
        "entropy_proportional_mean_delta": entropy_delta,
        "shuffled_statistic_mean_delta": shuffled_delta,
        "passed": specificity_passed,
        "decision": (
            "SIGNIFICANCE_SPECIFIC_MAPPING_SUPPORTED"
            if specificity_passed
            else "NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE"
        ),
    }
    statistics["multiplicity"] = {
        "method": "Holm",
        "family": sorted(comparisons),
        "adjusted_p_values": adjusted,
    }
    statistics["claim_limit"] = (
        "Boundary-local open-loop replay only. The shuffled allocation is at least as strong "
        "as the significance-aligned allocation, so no Cornerstone-specific adaptive-budget "
        "claim is supported."
    )
    atomic_write_json(statistics_path, statistics)

    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["primary_comparisons"] = comparisons
    summary["specificity_gate"] = statistics["specificity_gate"]
    summary["decision"] = statistics["specificity_gate"]["decision"]
    summary["claim_limit"] = statistics["claim_limit"]
    atomic_write_json(summary_path, summary)

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count_run = LAB_ROOT / "results" / "runs" / "halo_count_causal_controls_v1"
    evaluation = (
        LAB_ROOT / "results" / "benchmarks" / "halo_calibration_v3" / "evaluation.jsonl"
    )
    source_manifest = json.loads(
        (count_run / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["model"]["status"] = "HASH_RECORDED"
    manifest["model"]["hashes"] = dict(source_manifest["model"]["hashes"])
    manifest["runtime"]["model_loaded"] = False
    manifest["input_provenance"] = {
        "source_run_id": count_run.name,
        "source_run_manifest_sha256": sha256_file(count_run / "manifest.json"),
        "source_content_manifest_sha256": sha256_file(
            count_run / "content_manifest.json"
        ),
        "source_cases_sha256": sha256_file(count_run / "cases.jsonl"),
        "source_predictions_sha256": sha256_file(
            count_run / "predictions.unscored.jsonl"
        ),
        "source_boundary_index_sha256": sha256_file(
            count_run / "boundary_diagnostics.jsonl"
        ),
        "evaluation_sha256": sha256_file(evaluation),
    }
    note = "Shuffled-statistic allocation outperformed entropy-aligned allocation at the boundary-local endpoint; no significance-specific mapping claim is supported."
    if note not in manifest.setdefault("notes", []):
        manifest["notes"].append(note)
    atomic_write_json(manifest_path, manifest)
    return statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=LAB_ROOT / "results" / "runs" / "halo_adaptive_budget_screen_v1",
    )
    args = parser.parse_args()
    result = analyze(args.run)
    print(json.dumps({"status": result["status"], "specificity_gate": result["specificity_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
