from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_package_imports_from_normal_repository_execution() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rineya_pcsm; print(rineya_pcsm.__name__)",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "rineya_pcsm"
