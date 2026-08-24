#!/usr/bin/env python3
"""Cluster-aware descriptive analysis for the exploratory Halo retention pilot."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation import clustered_paired_analysis  # noqa: E402
from rineya_significance_lab.manifest import atomic_write_json  # noqa: E402


def analyze(run_dir: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in (run_dir / "cases.jsonl").read_text().splitlines()]
    indexed = {(str(row["case_id"]), str(row["condition"])): row for row in rows}
    cluster_ids = sorted({str(row["case_id"]) for row in rows})

    def values(condition: str, metric: str) -> list[float]:
        result: list[float] = []
        for cluster_id in cluster_ids:
            row = indexed[(cluster_id, condition)]
            value = row[metric] if metric == "correct" else row["evidence"][metric]
            result.append(float(value))
        return result

    metrics = (
        "correct",
        "minimum_per_layer_required_token_availability_fraction",
        "fraction_of_evaluated_layers_with_all_required_fact_tokens_available",
    )
    comparisons: dict[str, object] = {}
    for control in ("B2", "B4"):
        for metric in metrics:
            result = clustered_paired_analysis(
                cluster_ids,
                values(control, metric),
                values("B3", metric),
                minimum_clusters=8,
                bootstrap_samples=10_000,
                seed=17,
            )
            comparisons[f"B3_minus_{control}__{metric}"] = asdict(result)
    failures = Counter(
        label
        for row in rows
        for label in row.get("failure_labels", [])
    )
    payload = {
        "schema_version": "1.0",
        "status": "EXPLORATORY_POSTHOC",
        "selected_clusters": len(cluster_ids),
        "selection": "B0-eligible calibration cases, capped at four per R1/R3/R4 family",
        "comparisons": comparisons,
        "failure_label_counts": dict(sorted(failures.items())),
        "claim_limit": (
            "Evidence metrics and cluster analysis were not preregistered; p-values and intervals "
            "are exploratory diagnostics, not confirmatory evidence."
        ),
    }
    atomic_write_json(run_dir / "statistics.json", payload)
    return payload


def main() -> int:
    run_dir = LAB_ROOT / "results" / "runs" / "halo_retention_pilot_v2"
    payload = analyze(run_dir)
    print(json.dumps({"status": payload["status"], "output": str(run_dir / 'statistics.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
