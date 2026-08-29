#!/usr/bin/env python3
"""Freeze the outcome-blind task-difficulty and shortcut-baseline report."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
from tgsa.phase1_difficulty import difficulty_report, render_markdown
from tgsa.phase1_runner import atomic_write_json, load_study_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = load_study_config(args.config.resolve())
    report = difficulty_report(
        dataset_dir=args.dataset_dir.resolve(),
        dataset_config=study["dataset"],
    )
    atomic_write_json(args.output.resolve(), report)
    args.output.resolve().with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
