# Purpose: Select the Phase 6 ratio with the strongest positive relative opportunity margin.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.common.opportunity_evaluation import evaluate_opportunity, pareto_frontier
from scripts.common.quality_gate import assert_valid_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the most worthwhile Phase 6 active ratio.")
    parser.add_argument("--frontier", type=Path, default=ROOT / "artifacts/phase_6_sparse/frontier_with_generations.json")
    parser.add_argument("--quality-profile", choices=("balanced", "logic_oriented", "conversational_permissive"), default="balanced")
    parser.add_argument("--allow-none", action="store_true")
    # Historical floor flags parse but cannot affect the decision.
    for flag in ("--max-quality-loss", "--max-score-loss", "--max-semantic-pass-loss", "--min-complete-rate"):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    data = read_json(args.frontier)
    dense = data.get("dense_quality")
    if dense is None:
        raise SystemExit("Phase 6 frontier has no dense quality baseline.")
    assert_valid_reference(dense, label="Phase 6 dense reference")
    assessed = []
    for row in data.get("masked_generation_comparison", []):
        ratio = float(row["active_ratio"])
        if ratio >= 1.0:
            continue
        assessment = evaluate_opportunity(
            dense,
            row,
            resource_gain=max(0.0, 1.0 - ratio),
            resource_name="active_compute_reduction",
            profile=args.quality_profile,
        )
        assessed.append({**row, **assessment})
    worthwhile = [row for row in pareto_frontier(assessed) if row["eligible"]]
    selected = max(
        worthwhile,
        key=lambda row: (float(row["opportunity_margin"]), float(row["resource_gain_fraction"])),
        default=None,
    )
    if selected is None:
        if args.allow_none:
            print("NONE")
            return
        raise SystemExit("No sparse ratio has positive proportional opportunity margin.")
    print(selected["active_ratio"])


if __name__ == "__main__":
    main()
