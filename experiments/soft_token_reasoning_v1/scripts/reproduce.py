"""Reproducibility entry point for the no-training qualification path."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-out", type=Path, default=Path("experiments/soft_token_reasoning_v1/evidence/resource_probe.json"))
    parser.add_argument("--receipt-out", type=Path, default=Path("experiments/soft_token_reasoning_v1/QUALIFICATION_RECEIPT.json"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    commands = [
        [sys.executable, "-m", "pytest", "-q", "experiments/soft_token_reasoning_v1/tests"],
        [sys.executable, "-m", "experiments.soft_token_reasoning_v1.scripts.probe_resources", "--out", str(args.resource_out)],
        [sys.executable, "-m", "experiments.soft_token_reasoning_v1.scripts.run_qualification", "--out", str(args.receipt_out), "--resource-probe", str(args.resource_out)],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
