#!/usr/bin/env python3
"""Run the hardened competence-gated TGSA x MTP development workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase2_hardened import run_phase2_study


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.force_data:
        parser.error("--overwrite and --force-data cannot be combined")

    report = run_phase2_study(
        config_path=args.config.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        overwrite=args.overwrite,
        force_data=args.force_data,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
