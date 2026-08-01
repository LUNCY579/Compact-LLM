from __future__ import annotations

"""Create the pending Phase 4B human gate from opportunity evidence."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Phase 4B human gate template.")
    parser.add_argument("--recommendation", type=Path, default=ROOT / "artifacts/phase_4b_structural_reduction/candidate_recommendation.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/PHASE_4B_STOP_GATE.md")
    args = parser.parse_args()
    payload = read_json(args.recommendation)
    recommended = payload["recommended_candidate"]
    if isinstance(recommended, str):
        assessment = next(row for row in payload["assessments"] if row["name"] == recommended)
    else:
        assessment = recommended
        recommended = assessment["name"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join([
        "# PHASE 4B STOP GATE", "", "## Automated recommendation", "",
        f"- Candidate: `{recommended}`", f"- Model: `{assessment['model']}`",
        f"- Proportionally worthwhile: `{assessment['eligible']}`",
        f"- Resource gain: `{assessment['resource_gain_fraction']:.3%}`",
        f"- Quality opportunity loss: `{assessment['quality_opportunity_loss_fraction']:.3%}`",
        f"- Opportunity margin: `{assessment['opportunity_margin']:+.3%}`", "",
        "No absolute quality or parameter-reduction floor controls this recommendation.", "",
        "## Required review", "",
        "Inspect the candidate registry, raw replies, reload checks, opportunity dimensions, and measured runtime evidence. The dense reference remains a valid choice.", "",
        "## Human decision", "", "- Selected candidate: `PENDING`", "- Human decision: `PENDING`",
        "- Reviewer:", "- Date:", "- Reason:", "",
        "After an explicit human PASS, run finalize_reference_selection.py with the selected candidate name.",
    ]), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
