from __future__ import annotations

"""Finalize the checkpoint that downstream phases must use.

This command is deliberately separate from automated recommendation. It writes
the canonical selected-reference manifest only after the Phase 4B report has an
explicit human PASS and the requested candidate exists in the measured registry.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.check_human_gate import has_human_pass
from scripts.common.experiment_utils import read_json
from scripts.common.reference_selection import (
    DEFAULT_REFERENCE_MANIFEST,
    write_selected_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the human-approved Phase 4B selected-reference manifest."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate", required=True, help="Candidate name in the registry.")
    parser.add_argument(
        "--decision-report",
        type=Path,
        default=ROOT / "reports" / "PHASE_4B_STOP_GATE.md",
    )
    parser.add_argument("--reason", required=True, help="Short human-readable decision reason.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not has_human_pass(args.decision_report):
        raise SystemExit(
            f"{args.decision_report} must contain an explicit human PASS before finalization."
        )

    registry = read_json(args.registry)
    by_name = {row["name"]: row for row in registry["candidates"]}
    if args.candidate not in by_name:
        raise SystemExit(
            f"Unknown candidate {args.candidate!r}. Available: {sorted(by_name)}"
        )

    chosen = by_name[args.candidate]
    summary_path = Path(chosen["summary"])
    footprint_path = Path(chosen["footprint"])
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    if not footprint_path.is_absolute():
        footprint_path = ROOT / footprint_path
    summary = read_json(summary_path)
    footprint = read_json(footprint_path)
    write_selected_reference(
        output=args.output,
        source_phase_4_model=registry["dense_reference_model"],
        selected_name=args.candidate,
        selected_model=chosen["model"],
        registry=args.registry,
        decision_report=args.decision_report,
        selection_reason=args.reason,
        candidate_metrics={"summary": summary, "footprint": footprint},
    )
    print(args.output)


if __name__ == "__main__":
    main()
