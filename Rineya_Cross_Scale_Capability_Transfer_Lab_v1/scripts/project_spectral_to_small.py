from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from project_large_to_small import audit_identity  # noqa: E402
from rineya_cross_scale.spectral_projection import project_spectral_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Target-basis spectral donor projection.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-name", choices=("qwen4b", "qwen8b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1)
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
    manifest = project_spectral_checkpoint(
        template_path=args.template,
        reference_path=args.reference,
        donor_path=args.donor,
        destination=args.output,
        rank=args.rank,
        strength=args.strength,
        iterations=args.iterations,
        overwrite=args.overwrite,
        source_fingerprints=sources,
    )
    print(
        json.dumps(
            {
                "strategy": manifest["strategy"],
                "rank": manifest["rank"],
                "strength": manifest["strength"],
                "tensor_count": manifest["tensor_count"],
                "output_fingerprint": manifest["output_fingerprint"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
