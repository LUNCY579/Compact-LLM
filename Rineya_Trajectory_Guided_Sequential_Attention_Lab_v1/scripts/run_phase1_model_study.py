#!/usr/bin/env python3
"""Run all or a registered subset of the TGSA Phase-1 paired matrix."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
from tgsa.phase1_runner import run_study_matrix
from tgsa.phase1_runner import validate_completed_run_reuse as _validate_completed_run_reuse

__all__ = ["main", "_validate_completed_run_reuse"]


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("comma-separated selection cannot be empty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--arms", help="comma-separated registered arms")
    parser.add_argument("--seeds", help="comma-separated registered integer seeds")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.force_data:
        parser.error("--overwrite and --force-data cannot be combined")
    arm_values = _csv(args.arms)
    seed_values = _csv(args.seeds)
    matrix = run_study_matrix(
        study_config_path=args.config.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        output_root=args.output_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        arms=None if arm_values is None else [item for item in arm_values],  # type: ignore[arg-type]
        seeds=None if seed_values is None else [int(item) for item in seed_values],
        resume=not args.no_resume,
        overwrite=args.overwrite,
        force_data=args.force_data,
    )
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0 if matrix["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
