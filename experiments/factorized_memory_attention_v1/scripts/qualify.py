#!/usr/bin/env python
"""Run the bounded FMA mechanical qualification suite.

The runner does not discover datasets, download checkpoints, or access sealed
artifacts. It reports the subprocess result so a receipt can bind the exact
command and environment used by the caller.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit a JSON result")
    args = parser.parse_args()
    command = [sys.executable, "-m", "pytest", "-q", str(LAB / "tests")]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    result = {
        "kind": "FMA-QUAL-001",
        "command": command,
        "returncode": completed.returncode,
        "python": platform.python_version(),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "sealed_data_access": False,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
