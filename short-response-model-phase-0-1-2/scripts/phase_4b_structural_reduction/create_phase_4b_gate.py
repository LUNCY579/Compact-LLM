# Purpose: Turns Phase 4B candidate evidence into a pending human stop-gate report.
from __future__ import annotations

"""Create a review-ready Phase 4B stop-gate report from measured candidates."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Phase 4B human gate template.")
    parser.add_argument(
        "--recommendation",
        type=Path,
        default=ROOT / "artifacts/phase_4b_structural_reduction/candidate_recommendation.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/PHASE_4B_STOP_GATE.md",
    )
    args = parser.parse_args()
    payload = read_json(args.recommendation)
    recommended = payload["recommended_candidate"]
    assessment = next(row for row in payload["assessments"] if row["name"] == recommended)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(
            [
                "# PHASE 4B STOP GATE",
                "",
                "## Automated recommendation",
                "",
                f"- Candidate: `{recommended}`",
                f"- Model: `{assessment['model']}`",
                f"- Eligible under configured floors: `{assessment['eligible']}`",
                f"- Parameter reduction: `{assessment['parameter_reduction_fraction']:.3%}`",
                f"- Mean-score delta: `{assessment['mean_score_delta']:+.6f}`",
                "",
                "## Required review",
                "",
                "Inspect candidate_registry.json, every MODEL_EVALUATION.md, raw replies, reload checks, "
                "and measured latency/VRAM. You may choose dense_reference even when a reduced candidate is eligible.",
                "",
                "## Human decision",
                "",
                "- Selected candidate: `PENDING`",
                "- Human decision: `PENDING`",
                "- Reviewer:",
                "- Date:",
                "- Reason:",
                "",
                "After changing the human decision to PASS, run finalize_reference_selection.py "
                "with the exact selected candidate name.",
            ]
        ),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
