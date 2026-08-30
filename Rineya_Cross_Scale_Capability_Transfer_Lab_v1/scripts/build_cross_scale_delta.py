from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.operations import build_delta  # noqa: E402
from rineya_cross_scale.hashing import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build projected-donor minus Qwen3-0.6B.")
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projected-manifest", type=Path, required=True)
    parser.add_argument(
        "--reference-audit",
        type=Path,
        default=LAB_ROOT / "cross_scale_evidence" / "02_architecture" / "qwen06.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    projected_manifest = json.loads(
        args.projected_manifest.read_text(encoding="utf-8")
    )
    reference_audit = json.loads(args.reference_audit.read_text(encoding="utf-8"))
    result = build_delta(
        projected_path=args.projected,
        reference_path=args.reference,
        destination=args.output,
        overwrite=args.overwrite,
        source_fingerprints={
            "projected_manifest_sha256": sha256_file(args.projected_manifest),
            "projected_output_fingerprint": projected_manifest[
                "output_fingerprint"
            ],
            "reference_revision": reference_audit["revision"],
            "reference_weight_files": reference_audit["weights"]["files"],
        },
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
