#!/usr/bin/env python3
"""Verify the frozen TGSA v5.1 programme before any evidence is created."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.v5_1_programme import atomic_write_json, verify_repository_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_repository_contract(
        repo_root=args.repo_root.resolve(),
        contract_path=args.contract.resolve(),
    )
    if args.output is not None:
        atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
