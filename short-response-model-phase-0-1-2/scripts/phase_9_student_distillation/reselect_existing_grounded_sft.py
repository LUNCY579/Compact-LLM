from __future__ import annotations

"""Recompute Phase 9 Stage-D SFT selection from an existing run root.

This utility is intentionally training-free. It is useful when selector policy is
updated after expensive SFT/evaluation artifacts already exist.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.phase_9_student_distillation.select_grounded_sft_candidate import (
    build_selection_packet,
    render_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh warm-SFT selection from existing grounded-recovery summaries without retraining."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    init_path = run_root / "evaluation_warm_initialization_oracle" / "summary.json"
    sweep_root = run_root / "warm_sft_sweep"
    out_path = sweep_root / "selection.json"
    report_path = sweep_root / "SELECTION_REPORT.md"

    if not init_path.exists():
        raise FileNotFoundError(f"Missing Stage-B initialization summary: {init_path}")
    if not sweep_root.exists():
        raise FileNotFoundError(f"Missing warm SFT sweep root: {sweep_root}")

    candidate_paths = sorted(sweep_root.glob("*/evaluation_oracle/summary.json"))
    if not candidate_paths:
        raise FileNotFoundError(f"No Stage-D candidate summaries found under: {sweep_root}")

    initialization = read_json(init_path)
    candidates = [
        (path.parents[1].name, path, read_json(path))
        for path in candidate_paths
    ]
    packet = build_selection_packet(
        initialization,
        candidates,
        initialization_summary=str(init_path),
    )
    write_json(out_path, packet)
    report_path.write_text(render_report(packet), encoding="utf-8")

    print(json.dumps(packet, indent=2))
    if packet.get("selected") is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
