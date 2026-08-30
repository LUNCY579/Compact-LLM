#!/usr/bin/env python3
"""Verify a real-checkpoint TGSA + MTP adapter warm-start bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rineya_tgsa.run_integrity import verify_training_integrity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_training_integrity(
        output_dir=args.output_dir.resolve(),
        report_path=args.report.resolve() if args.report else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
