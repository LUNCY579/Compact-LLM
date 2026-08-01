# Purpose: Reads the Phase 6 frontier and returns the lowest ratio passing quality-v2 floors.
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json
from scripts.common.quality_gate import assert_valid_reference, compare_quality_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the lowest eligible Phase 6 active ratio.")
    parser.add_argument("--frontier", type=Path, default=ROOT / "artifacts/phase_6_sparse/frontier_with_generations.json")
    parser.add_argument("--max-quality-loss", "--max-score-loss", dest="max_quality_loss", type=float, default=0.01)
    parser.add_argument("--max-semantic-pass-loss", type=float, default=0.01)
    parser.add_argument("--min-complete-rate", type=float, default=0.99)
    parser.add_argument("--allow-none", action="store_true")
    args = parser.parse_args()
    data = read_json(args.frontier); dense = data.get("dense_quality")
    if dense is None: raise SystemExit("Phase 6 frontier has no dense quality baseline.")
    assert_valid_reference(dense, label="Phase 6 dense reference")
    eligible=[]
    for row in data.get("masked_generation_comparison", []):
        ratio=float(row["active_ratio"])
        if ratio >= .999: continue
        comparison=compare_quality_summaries(
            dense,row,max_mean_quality_loss=args.max_quality_loss,max_quality_index_loss=args.max_quality_loss+.005,
            max_worst_decile_loss=.03,max_category_quality_loss_allowed=.04,
            max_semantic_pass_loss=args.max_semantic_pass_loss,max_reference_alignment_loss=.05,
            min_quality_pass_rate=.98,min_intelligibility_pass_rate=1.0,min_semantic_pass_rate=.98,
            min_complete_utterance_rate=args.min_complete_rate,max_hard_failure_cases=0,max_critical_failures=0,
            max_major_issue_cases=0)
        if comparison["eligible"]: eligible.append(ratio)
    if not eligible:
        if args.allow_none: print("NONE"); return
        raise SystemExit("No sparse active ratio passed Phase 6 quality-v2 floors.")
    print(min(eligible))


if __name__ == "__main__": main()
