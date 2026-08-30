from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.projection import project_checkpoint  # noqa: E402


def audit_identity(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    return {
        "name": audit["name"],
        "source_id": audit["source_id"],
        "revision": audit["revision"],
        "config_sha256": audit["config"]["sha256"],
        "tokenizer_bundle_sha256": audit["tokenizer"]["bundle_sha256"],
        "weight_files": audit["weights"]["files"],
        "state_dict_shape_layout_sha256": audit["weights"][
            "state_dict_shape_layout_sha256"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project a larger Qwen3 checkpoint into Qwen3-0.6B tensor shapes."
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--metadata-template",
        type=Path,
        help="sidecar/config source; defaults to the tensor-shape template",
    )
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-name", choices=("qwen4b", "qwen8b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", choices=("D1", "D2", "D3"), default="D2")
    parser.add_argument("--width", choices=("W1", "W2", "W3"), default="W2")
    parser.add_argument(
        "--tied-projection",
        choices=("embedding", "output_head", "mean"),
        default="embedding",
        help="collapse policy when an untied donor maps into the tied target",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--max-shard-gib", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--architecture-evidence",
        type=Path,
        default=LAB_ROOT / "cross_scale_evidence" / "02_architecture",
    )
    args = parser.parse_args()
    source_fingerprints = {
        name: audit_identity(args.architecture_evidence / f"{name}.json")
        for name in ("supa", "qwen06", args.donor_name)
    }
    manifest = project_checkpoint(
        template_path=args.template,
        donor_path=args.donor,
        destination=args.output,
        metadata_template_path=args.metadata_template,
        depth_strategy=args.depth,
        width_strategy=args.width,
        tied_projection=args.tied_projection,
        output_dtype=args.output_dtype,
        max_shard_bytes=int(args.max_shard_gib * 1024**3),
        overwrite=args.overwrite,
        source_fingerprints=source_fingerprints,
    )
    print(
        json.dumps(
            {
                "plan_id": manifest["plan"]["id"],
                "depth_strategy": manifest["plan"]["depth_strategy"],
                "width_strategy": manifest["plan"]["width_strategy"],
                "tied_projection": manifest["plan"]["tied_projection"],
                "tensor_count": manifest["tensor_count"],
                "output_fingerprint": manifest["output_fingerprint"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
