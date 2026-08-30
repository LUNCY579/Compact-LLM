from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import sha256_file  # noqa: E402
from rineya_cross_scale.interventions import merge_masked_delta  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject a delta into selected target layers.")
    parser.add_argument("--supa", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--delta-manifest", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = merge_masked_delta(
        base_path=args.supa,
        delta_path=args.delta,
        destination=args.output,
        alpha=args.alpha,
        layers=set(args.layer),
        overwrite=args.overwrite,
        source_fingerprints={"delta_manifest_sha256": sha256_file(args.delta_manifest)},
    )
    print(json.dumps({"selected_tensor_count": result["selected_tensor_count"], "output_fingerprint": result["output_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
