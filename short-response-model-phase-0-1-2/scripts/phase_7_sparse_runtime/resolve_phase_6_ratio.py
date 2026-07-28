# Purpose: Reads the accepted Phase 6 frontier and returns the active ratio for Phase 7.
from __future__ import annotations

"""Resolve the Phase 6 recommended active ratio from its measured frontier."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the lowest eligible Phase 6 active ratio.")
    parser.add_argument(
        "--frontier",
        type=Path,
        default=ROOT / "artifacts/phase_6_sparse/frontier_with_generations.json",
    )
    parser.add_argument("--max-score-loss", type=float, default=0.01)
    parser.add_argument("--max-semantic-pass-loss", type=float, default=0.01)
    parser.add_argument("--min-complete-rate", type=float, default=0.99)
    parser.add_argument("--allow-none", action="store_true", help="Print NONE instead of failing when dense-only deployment was selected.")
    args = parser.parse_args()
    data = read_json(args.frontier)
    dense = data.get("dense_quality")
    if dense is None:
        raise SystemExit("Phase 6 frontier has no dense quality baseline.")
    eligible = []
    for row in data.get("masked_generation_comparison", []):
        ratio = float(row["active_ratio"])
        if ratio >= 0.999:
            continue
        if float(row["mean_score_loss_from_dense"]) > args.max_score_loss:
            continue
        if float(row["semantic_pass_loss_from_dense"]) > args.max_semantic_pass_loss:
            continue
        if float(row["complete_utterance_rate"]) < args.min_complete_rate:
            continue
        if int(row["critical_failure_count"]) > int(dense["critical_failure_count"]):
            continue
        eligible.append(ratio)
    if not eligible:
        if args.allow_none:
            print("NONE")
            return
        raise SystemExit("No sparse active ratio passed Phase 6 quality floors.")
    print(min(eligible))


if __name__ == "__main__":
    main()
