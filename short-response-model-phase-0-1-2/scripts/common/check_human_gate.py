from __future__ import annotations

"""Require an explicit human PASS before advancing the pipeline.

The official phases use numeric labels, while structural studies are inserted
as Phases 2B and 4B. Accepting a string label keeps all runners on one gate checker and
prevents a second, slightly different approval implementation from drifting.
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def has_human_pass(path: Path) -> bool:
    """Return True only for the repository's exact approval line."""

    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return bool(
        re.search(
            r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$",
            text,
        )
    )


def automated_gate_failed(path: Path) -> bool:
    """Detect the special Phase 2 automated FAIL status."""

    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return bool(
        re.search(
            r"(?is)##\s+Automated Gate Status\s*\n+\s*`?FAIL`?",
            text,
        )
    )


def normalize_phase_label(value: str) -> str:
    """Normalize supported labels such as ``4b`` to the filename form ``4B``."""

    phase = value.strip().upper()
    if not re.fullmatch(r"(?:[0-9]+|2B|4B)", phase):
        raise argparse.ArgumentTypeError(
            "Phase must be a number such as 3, or an inserted structural phase such as 2B/4B."
        )
    return phase


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require an explicit human PASS before advancing phases."
    )
    parser.add_argument("--phase", type=normalize_phase_label, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional explicit stop-gate report for a named model-profile branch.",
    )
    parser.add_argument(
        "--allow-deviation",
        action="store_true",
        help=(
            "For Phase 2 only, allow an approved PHASE_2_DEVIATION_APPROVAL.md "
            "when the automated Phase 2 gate failed."
        ),
    )
    args = parser.parse_args()

    report = args.report or (ROOT / "reports" / f"PHASE_{args.phase}_STOP_GATE.md")

    # Preserve the repository's existing Phase 2 negative-result/deviation
    # workflow. No other phase may bypass its own stop-gate report.
    if args.phase == "2" and args.allow_deviation and args.report is None and automated_gate_failed(report):
        deviation = ROOT / "reports" / "PHASE_2_DEVIATION_APPROVAL.md"
        if not deviation.exists():
            raise SystemExit(
                "Phase 2 automated gate failed and the deviation approval is missing: "
                f"{deviation}"
            )
        if not has_human_pass(deviation):
            raise SystemExit(
                "Phase 2 automated gate failed and the deviation report lacks an "
                f"explicit human PASS: {deviation}"
            )
        print(
            "Phase 2 deviation gate: PASS "
            f"(approved by {deviation.name}; Phase 2 itself remains a negative result)"
        )
        return

    if not report.exists():
        raise SystemExit(f"Missing {report}")
    if not has_human_pass(report):
        raise SystemExit(
            f"Phase {args.phase} lacks explicit human PASS in {report}"
        )

    print(f"Phase {args.phase} human gate: PASS")


if __name__ == "__main__":
    main()
