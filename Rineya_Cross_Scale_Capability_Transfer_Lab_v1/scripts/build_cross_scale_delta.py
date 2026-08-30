from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.operations import build_delta  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build projected-donor minus Qwen3-0.6B.")
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_delta(
        projected_path=args.projected,
        reference_path=args.reference,
        destination=args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
