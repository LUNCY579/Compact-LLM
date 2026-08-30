from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.advanced import merge_fisher_candidate  # noqa: E402
from rineya_cross_scale.hashing import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fisher-weighted target/donor merge.")
    parser.add_argument("--supa", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--fisher-supa", type=Path, required=True)
    parser.add_argument("--fisher-donor", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--donor-lambda", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--fisher-clip", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sources = {
        f"source_manifest_{index}_sha256": sha256_file(path)
        for index, path in enumerate(args.source_manifest)
    }
    result = merge_fisher_candidate(
        base_path=args.supa,
        donor_path=args.donor,
        fisher_base_path=args.fisher_supa,
        fisher_donor_path=args.fisher_donor,
        destination=args.output,
        donor_lambda=args.donor_lambda,
        epsilon=args.epsilon,
        fisher_clip=args.fisher_clip,
        output_dtype=args.output_dtype,
        overwrite=args.overwrite,
        source_fingerprints=sources,
    )
    print(json.dumps({"operation": result["operation"], "operation_parameters": result["operation_parameters"], "storage_summary": result["storage_summary"], "output_fingerprint": result["output_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
