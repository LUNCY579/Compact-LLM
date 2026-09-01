#!/usr/bin/env python3
"""Run unit tests, then optional GPU α=0 check."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
candidates = [
    REPO / ".venv" / "Scripts" / "python.exe",
    REPO.parent / "Chat-Bubble" / ".venv" / "Scripts" / "python.exe",
]
python = str(next((p for p in candidates if p.is_file()), Path(sys.executable)))

cmd = [
    python,
    "-m",
    "pytest",
    str(LAB / "tests"),
    "-q",
    "--ignore",
    str(LAB / "tests" / "test_model_alpha_zero.py"),
]
env_pythonpath = str(LAB / "src")
print(" ".join(cmd), flush=True)
env = os.environ.copy()
env["PYTHONPATH"] = str(LAB / "src")
rc = subprocess.call(cmd, cwd=str(LAB), env=env)
if rc != 0:
    raise SystemExit(rc)
print("unit tests passed")
