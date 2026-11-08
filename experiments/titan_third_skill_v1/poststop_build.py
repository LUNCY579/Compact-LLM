"""Materialize the one POSTSTOP-001 random arm with the unchanged TTR builder."""
from __future__ import annotations

from pathlib import Path

import build_arm

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    build_arm.ARMS_DIR = HERE / "poststop_null" / "arms"
    raise SystemExit(build_arm.build("TTR", 0.25))
