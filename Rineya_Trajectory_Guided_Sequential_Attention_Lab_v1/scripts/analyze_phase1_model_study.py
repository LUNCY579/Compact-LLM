#!/usr/bin/env python3
"""Apply the frozen paired analysis to TGSA Phase-1 artifacts."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
from tgsa.phase1_analysis import analyze_study


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    analysis = analyze_study(
        study_config_path=args.config.resolve(),
        output_root=args.output_dir.resolve(),
        dataset_dir=None if args.dataset_dir is None else args.dataset_dir.resolve(),
        repo_root=None if args.repo_root is None else args.repo_root.resolve(),
        allow_partial=args.allow_partial,
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
