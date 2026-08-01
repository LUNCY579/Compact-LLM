from __future__ import annotations

"""Dry-run ranking helper for the relative-opportunity greedy soup."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.quality_evaluation import require_quality_v3, summary_quality_score
from scripts.common.quality_gate import quality_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_4_soup/greedy_plan.json")
    args = parser.parse_args()
    rows = []
    for summary_path in args.summaries:
        summary = read_json(summary_path)
        require_quality_v3(summary, str(summary_path))
        rows.append({
            "summary": str(summary_path),
            "mean_quality": summary_quality_score(summary),
            "quality_index": quality_index(summary),
            "worst_decile": summary.get("worst_decile_mean_quality_score", 0.0),
            "diagnostic_issue_free_rate": summary.get("quality_pass_rate", 0.0),
            "severe_issue_case_rate": summary.get("severe_issue_case_rate", 0.0),
            "mean_issue_cost": summary.get("mean_issue_cost", 0.0),
        })
    rows.sort(key=lambda row: (-row["quality_index"], -row["worst_decile"], -row["mean_quality"]))
    write_json(args.out, {
        "quality_evaluation_version": "3.0",
        "ranked_ingredients": rows,
        "procedure": (
            "Start with the highest continuous-quality ingredient. Add one compatible ingredient, "
            "merge in FP32 delta space, evaluate, and retain only when weighted proportional quality "
            "gains exceed weighted proportional quality losses. No absolute quality floor applies."
        ),
    })
    print(args.out)


if __name__ == "__main__":
    main()
