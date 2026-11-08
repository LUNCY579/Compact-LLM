#!/usr/bin/env python3
"""Materialize and validate the registered TGSA Phase-1 dataset."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
from tgsa.phase1_runner import load_study_config, prepare_study_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    study = load_study_config(args.config.resolve())
    manifest = prepare_study_dataset(study, dataset_dir=args.output_dir.resolve(), force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
