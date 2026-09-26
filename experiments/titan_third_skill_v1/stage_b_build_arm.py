"""Rebuild only the Stage A selected arms in a separate Stage B output directory.

The original builder's construction and canaries are used unchanged. Its Stage A
arm receipts remain in place after Stage A deletes the temporary weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import build_arm
from stage_b_guard import selected

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()
    composite, solo, beta = selected()
    if args.arm not in (composite, solo, "TTR"):
        raise SystemExit(f"Stage B refused: {args.arm} is not a registered selected arm")
    build_arm.ARMS_DIR = HERE / "stage_b" / "arms"
    return build_arm.build(args.arm, beta if args.arm == "TTR" else None)


if __name__ == "__main__":
    raise SystemExit(main())
