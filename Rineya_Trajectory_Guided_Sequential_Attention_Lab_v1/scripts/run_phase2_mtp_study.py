#!/usr/bin/env python3
"""Run the hardened TGSA x MTP factorial study."""

from __future__ import annotations

import argparse
from pathlib import Path

from tgsa.phase2_hardened import run_phase2_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    matrix = run_phase2_study(
        config_path=args.config.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        force_data=args.force_data,
    )
    print(
        f"Phase-2 matrix complete: {matrix['study_id']} "
        f"({len(matrix['runs'])} runs, "
        f"effective_epochs={matrix['training_dose']['effective_epochs']:.4f})"
    )


if __name__ == "__main__":
    main()
