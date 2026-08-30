#!/usr/bin/env python3
"""Verify the original Phase-2 evidence chain plus the integrity contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from tgsa.phase2_integrity import verify_phase2_integrity
from tgsa.phase2_verify import verify_phase2_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    original = verify_phase2_evidence(
        config_path=args.config.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        report_path=args.report.resolve() if args.report else None,
    )
    integrity = verify_phase2_integrity(
        config_path=args.config.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    passed = bool(original["passed"] and integrity["passed"])
    print(
        "Phase-2 evidence verification "
        f"passed={passed} original={original['passed']} "
        f"integrity={integrity['passed']}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
