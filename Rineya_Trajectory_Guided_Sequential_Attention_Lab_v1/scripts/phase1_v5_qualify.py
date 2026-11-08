#!/usr/bin/env python3
"""Execute one resumable TGSA Phase 1 v5 apparatus-qualification cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase1_v5_runner import run_v5_stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one v5 prerequisite-probe rung or one qualification exposure. "
            "A scientific NO_GO is a valid completed run and therefore exits 0."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("probe", "qualification"), required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument(
        "--exposure",
        help="registered exposure id; required only for qualification",
    )
    parser.add_argument("--frozen-band", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="refuse any existing cell instead of resuming or reusing it",
    )
    args = parser.parse_args()
    if args.stage == "probe" and args.exposure is not None:
        parser.error("--exposure is forbidden for the fixed prerequisite probe")
    if args.stage == "qualification" and args.exposure is None:
        parser.error("--exposure is required for qualification")

    report = run_v5_stage(
        plan_path=args.plan.resolve(),
        stage=args.stage,
        rung_id=args.rung,
        exposure_id=args.exposure,
        frozen_band_path=(
            None if args.frozen_band is None else args.frozen_band.resolve()
        ),
        output_dir=args.output_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        resume=not args.no_resume,
    )
    summary = {
        "status": report["status"],
        "stage": report["stage_execution"],
        "rung_id": report["rung_id"],
        "exposure_id": report["exposure_id"],
        "verdict": report["verdict"],
        "decision": report["decision"],
        "output_dir": str(args.output_dir.resolve()),
        "note": (
            "exit code reports software completion, not scientific GO/NO_GO"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
