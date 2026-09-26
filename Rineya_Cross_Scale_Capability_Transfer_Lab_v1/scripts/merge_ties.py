from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.advanced import merge_ties_candidate  # noqa: E402
from rineya_cross_scale.hashing import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="TIES consensus injection into Supa.")
    parser.add_argument("--supa", type=Path, required=True)
    parser.add_argument("--delta", type=Path, action="append", required=True)
    parser.add_argument("--delta-manifest", type=Path, action="append", required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--keep-fraction", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.delta) != len(args.delta_manifest):
        raise SystemExit("--delta and --delta-manifest counts must match")
    sources = {"base_validation_sha256": sha256_file(args.base_validation)}
    sources.update(
        {
            f"delta_manifest_{index}_sha256": sha256_file(path)
            for index, path in enumerate(args.delta_manifest)
        }
    )
    result = merge_ties_candidate(
        base_path=args.supa,
        delta_paths=args.delta,
        destination=args.output,
        alpha=args.alpha,
        keep_fraction=args.keep_fraction,
        output_dtype=args.output_dtype,
        overwrite=args.overwrite,
        source_fingerprints=sources,
    )
    print(json.dumps({"operation": result["operation"], "coefficient": result["coefficient"], "operation_parameters": result["operation_parameters"], "storage_summary": result["storage_summary"], "output_fingerprint": result["output_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
