from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.operations import merge_consensus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject a 4B/8B consensus delta into Supa.")
    parser.add_argument("--supa", type=Path, required=True)
    parser.add_argument("--delta4", type=Path, required=True)
    parser.add_argument("--delta8", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--method", choices=("mean", "sign_agree", "tensor_cosine"), default="mean")
    parser.add_argument("--cosine-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = merge_consensus(
        base_path=args.supa,
        delta4_path=args.delta4,
        delta8_path=args.delta8,
        destination=args.output,
        alpha=args.alpha,
        method=args.method,
        cosine_threshold=args.cosine_threshold,
        output_dtype=args.output_dtype,
        max_shard_bytes=int(1.5 * 1024**3),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
