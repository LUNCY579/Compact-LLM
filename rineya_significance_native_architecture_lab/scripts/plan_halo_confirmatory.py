#!/usr/bin/env python3
"""Write a model-free cluster/power planning artifact for Halo registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation import normal_approximation_mde  # noqa: E402
from rineya_significance_lab.manifest import atomic_write_json  # noqa: E402


def build_plan(config_path: Path) -> dict[str, object]:
    config_bytes = config_path.read_bytes()
    cluster_counts = (12, 16, 32, 64)
    standard_deviations = (0.10, 0.20, 0.30)
    mde = [
        {
            "independent_clusters": clusters,
            "assumed_cluster_difference_sd": sd,
            "normal_approximation_mde": normal_approximation_mde(
                independent_clusters=clusters,
                cluster_difference_sd=sd,
            ),
        }
        for clusters in cluster_counts
        for sd in standard_deviations
    ]
    return {
        "schema_version": "1.0",
        "status": "PLANNING_ONLY_NOT_REGISTERED",
        "target": "Rineya-Halo-1.2.0-devfreeze",
        "config": {
            "path": "configs/confirmatory.yaml",
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "calibration_stage": {
            "independent_clusters": 32,
            "families": ["R1", "R2", "R3", "R4"],
            "clusters_per_family": 8,
            "initial_depth": 0.5,
            "purpose": "estimate full-cache capability and between-cluster SD without treatment tuning",
            "advance_rule": "at least 12 B0-eligible clusters overall and at least 2 per family",
            "revise_rule": "if B0 capability is below 0.40, simplify task difficulty before registering compression arms",
        },
        "candidate_confirmatory_stage": {
            "independent_clusters": 64,
            "depths_per_cluster": [0.10, 0.50, 0.90],
            "conditions": ["B0", "B2", "B3", "B4", "B5", "B6", "B7"],
            "inference_unit": "cluster_id",
            "primary_test": "two-sided cluster sign-flip on cluster-mean B3-minus-B2 strict evidence outcome",
            "uncertainty": "cluster bootstrap percentile interval",
            "registration_gate": "pilot-derived cluster SD and exact evidence scorer must be frozen first",
        },
        "exact_sign_flip_resolution": [
            {
                "independent_clusters": clusters,
                "smallest_two_sided_p_when_all_nonzero_differences_agree": 2 / (2**clusters),
            }
            for clusters in cluster_counts
        ],
        "mde_sensitivity": mde,
        "limitations": [
            "MDE values use a normal planning approximation, not the registered randomization test.",
            "Cluster SD is unknown until capability calibration produces independent-cluster outcomes.",
            "No model outcome or treatment comparison is used by this planning artifact.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "confirmatory.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "results" / "statistics" / "halo_confirmatory_planning.json",
    )
    args = parser.parse_args()
    plan = build_plan(args.config.resolve())
    atomic_write_json(args.output.resolve(), plan)
    print(json.dumps({"status": plan["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
