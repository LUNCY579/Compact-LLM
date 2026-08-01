# Purpose: Collects available A-I ablation artifacts without inventing missing experiments.
from __future__ import annotations

"""Collect existing A-I ablation evidence without rerunning prior phases.

Phase 8 is a release audit, not a reason to silently retrain models.  This script
indexes the immutable summaries and manifests already produced by Phases 1-7,
marks missing controls explicitly, and writes a machine-readable matrix.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, sha256, write_json


def first_existing(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        for path in matches:
            if path.is_file():
                return path
    return None


def evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "MISSING", "path": None}
    payload: Any = None
    try:
        payload = read_json(path)
    except Exception:
        payload = None
    return {"status": "AVAILABLE", "path": str(path), "sha256": sha256(path), "summary": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect release ablation evidence.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/phase_8_release/ablation_matrix.json",
    )
    args = parser.parse_args()
    matrix = {
        "A_base": evidence(first_existing(["artifacts/base_evaluation/summary.json", "artifacts/baseline_evaluation/summary.json"])),
        "B_all_layer_sft": evidence(first_existing(["artifacts/dense_sft_evaluation/summary.json"])),
        "C_best_single": evidence(first_existing(["artifacts/phase_2*/**/*single*/**/summary.json"])),
        "D_best_2_4": evidence(first_existing(["artifacts/phase_2*/**/*pair*/**/summary.json", "artifacts/phase_2*/**/*top4*/**/summary.json"])),
        "E_random_control": evidence(first_existing(["artifacts/phase_2*/**/*random*/**/summary.json"])),
        "phase_2B_progressive_depth": evidence(
            first_existing(["artifacts/phase_2_progressive_reduction/model_catalog.json"])
        ),
        "F_selective_dpo": evidence(first_existing(["artifacts/phase_3_dpo_evaluation/summary.json"])),
        "G_phase_4_accepted": evidence(first_existing(["artifacts/phase_4_model_soups/accepted_evaluation/summary.json"])),
        "phase_4B_structural": evidence(first_existing(["artifacts/phase_4b_structural_reduction/candidate_registry.json"])),
        "H_sparse_quality": evidence(first_existing(["artifacts/phase_6_sparse/frontier_with_generations.json"])),
        "I_full_runtime": evidence(first_existing(["artifacts/phase_7_router/runtime_comparison.json"])),
    }
    stale_quality = [
        name for name, row in matrix.items()
        if row.get("status") == "AVAILABLE" and isinstance(row.get("summary"), dict)
        and "mean_heuristic_score" in row["summary"]
        and str(row["summary"].get("quality_evaluation_version")) != "3.0"
    ]
    write_json(
        args.out,
        {
            "matrix": matrix,
            "available": sum(1 for row in matrix.values() if row["status"] == "AVAILABLE"),
            "missing": [name for name, row in matrix.items() if row["status"] != "AVAILABLE"],
            "stale_quality_evidence": stale_quality,
            "interpretation": (
                "Missing research controls are reported honestly. They do not automatically block a commercial dense "
                "release when the final quality/safety gates pass, but they limit causal claims about which experiment helped."
            ),
        },
    )
    print(args.out)


if __name__ == "__main__":
    main()
