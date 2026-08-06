from __future__ import annotations

"""Print the full checkpoint selected at the Phase 9 human handoff."""

import argparse
from pathlib import Path

from scripts.common.experiment_utils import read_json

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "references" / "phase_9_student_reference.json",
    )
    args = parser.parse_args()
    if not args.manifest.exists():
        raise SystemExit(
            f"Missing Phase 9 selected-student manifest: {args.manifest}. "
            "Run finalize_phase_9_selection.py after human approval."
        )
    payload = read_json(args.manifest)
    selected = Path(str(payload.get("selected_model") or ""))
    if not selected.is_absolute():
        selected = ROOT / selected
    if not selected.exists() or not (selected / "config.json").exists():
        raise SystemExit(f"Selected Phase 9 student checkpoint is unavailable: {selected}")
    print(selected)


if __name__ == "__main__":
    main()
