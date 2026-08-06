from __future__ import annotations

"""Print the full checkpoint selected and approved in Phase 10."""

import argparse
from pathlib import Path

from scripts.common.experiment_utils import read_json

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "references" / "phase_10_student_reference.json",
    )
    args = parser.parse_args()
    if not args.manifest.exists():
        raise SystemExit(
            f"Missing Phase 10 selected-student manifest: {args.manifest}. "
            "Run finalize_phase_10_selection.py after human approval."
        )
    payload = read_json(args.manifest)
    selected = Path(str(payload.get("selected_model") or ""))
    if not selected.is_absolute():
        selected = ROOT / selected
    if not selected.exists() or not (selected / "config.json").exists():
        raise SystemExit(f"Selected Phase 10 checkpoint is unavailable: {selected}")
    print(selected)


if __name__ == "__main__":
    main()
