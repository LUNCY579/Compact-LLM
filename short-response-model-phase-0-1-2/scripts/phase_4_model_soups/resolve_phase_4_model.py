# Purpose: Reads the Phase 4 reference manifest and prints the accepted local checkpoint path.
from __future__ import annotations

"""Print the full checkpoint accepted by the Phase 4 soup workflow."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json

DEFAULT_MANIFEST = ROOT / "artifacts/references/phase_4_reference.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the Phase 4 accepted full model.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Missing Phase 4 reference manifest: {args.manifest}")
    payload = read_json(args.manifest)
    selected = payload.get("selected_model")
    if not isinstance(selected, str) or not selected:
        raise SystemExit(f"Malformed Phase 4 reference manifest: {args.manifest}")
    print(selected)


if __name__ == "__main__":
    main()
