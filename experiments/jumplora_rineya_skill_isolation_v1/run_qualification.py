"""CLI entry point for JLR-QUAL-001."""
from __future__ import annotations

import argparse
from pathlib import Path

from .qualification import run_qualification


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded JumpLoRA qualification")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("QUALIFICATION_RECEIPT.json"))
    args = parser.parse_args()
    receipt = run_qualification(args.output)
    print(f"{receipt['qualification_id']}: {receipt['status']}")
    print(f"receipt: {args.output}")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
