# Purpose: Creates the final pending human release gate from the generated audit.
from __future__ import annotations

"""Create the final human release gate from the measured release audit."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PHASE_8_STOP_GATE.md.")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/RELEASE_CANDIDATE_AUDIT.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/PHASE_8_STOP_GATE.md")
    args = parser.parse_args()
    audit = read_json(args.audit)
    status = audit.get("status", "BLOCKED")
    args.out.write_text(
        "\n".join(
            [
                "# PHASE 8 STOP GATE",
                "",
                "## Automated Gate Status",
                "",
                f"`{status}`",
                "",
                f"- Deployment mode: `{audit.get('deployment_mode')}`",
                f"- Selected model: `{audit.get('model_dir')}`",
                f"- Blocking reasons: `{audit.get('blocking_reasons') or 'None'}`",
                "",
                "## Human commercial release decision",
                "",
                "- Human decision: `PENDING`",
                "- Reviewer:",
                "- Date:",
                "- Release version:",
                "- Rollback tested: `PENDING`",
                "- Notes:",
            ]
        ),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
