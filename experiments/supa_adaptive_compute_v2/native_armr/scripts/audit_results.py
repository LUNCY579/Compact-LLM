"""Audit complete Track A evidence and save a new immutable decision receipt."""
from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROGRAM))

from native_armr.scripts.aggregate import main


if __name__ == "__main__":
    raise SystemExit(main())
