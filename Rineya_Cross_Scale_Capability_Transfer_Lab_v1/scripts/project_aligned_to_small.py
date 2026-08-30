from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from project_large_to_small import audit_identity  # noqa: E402
from rineya_cross_scale.aligned_projection import project_aligned_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding-anchored W5 donor projection.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-name", choices=("qwen4b", "qwen8b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", choices=("D1", "D2", "D3"), default="D2")
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--heldout-count", type=int, default=128)
    parser.add_argument("--ridge-relative", type=float, default=1e-3)
    parser.add_argument(
        "--alignment-method", choices=("ridge", "procrustes"), default="ridge"
    )
    parser.add_argument("--compute-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--architecture-evidence",
        type=Path,
        default=LAB_ROOT / "cross_scale_evidence" / "02_architecture",
    )
    args = parser.parse_args()
    sources = {
        name: audit_identity(args.architecture_evidence / f"{name}.json")
        for name in ("supa", "qwen06", args.donor_name)
    }
    manifest = project_aligned_checkpoint(
        template_path=args.template,
        reference_path=args.reference,
        donor_path=args.donor,
        destination=args.output,
        depth_strategy=args.depth,
        sample_count=args.sample_count,
        heldout_count=args.heldout_count,
        ridge_relative=args.ridge_relative,
        alignment_method=args.alignment_method,
        compute_device=args.compute_device,
        overwrite=args.overwrite,
        source_fingerprints=sources,
    )
    print(
        json.dumps(
            {
                "strategy": manifest["strategy"],
                "width_strategy": manifest["width_strategy"],
                "depth_strategy": manifest["depth_strategy"],
                "calibration": manifest["calibration"],
                "output_fingerprint": manifest["output_fingerprint"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
