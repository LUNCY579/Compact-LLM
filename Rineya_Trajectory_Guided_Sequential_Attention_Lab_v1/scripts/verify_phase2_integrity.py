#!/usr/bin/env python3
"""Verify the Phase-2 MTP integrity and anti-laundering contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase2_integrity import verify_phase2_integrity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_phase2_integrity(
        config_path=args.config.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        report_path=args.report.resolve() if args.report else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
