"""Exploratory residual direction-transfer projection into the 0.6B basis."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.architecture import uniform_depth_map  # noqa: E402
from rineya_cross_scale.checkpoint import (  # noqa: E402
    CheckpointError,
    CheckpointReader,
    StreamingSafetensorsWriter,
    copy_sidecars,
    validate_safe_destination,
)
from rineya_cross_scale.direction_transfer import spectral_direction_transfer  # noqa: E402
from rineya_cross_scale.hashing import atomic_write_json  # noqa: E402


LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _donor_key(key: str, mapping: list[int]) -> str:
    match = LAYER_KEY.match(key)
    if not match:
        return key
    layer = int(match.group(1))
    if layer >= len(mapping):
        raise CheckpointError(f"target layer {layer} exceeds D2 map")
    return f"model.layers.{mapping[layer]}.{match.group(2)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = validate_safe_destination(args.output, [args.target, args.donor])
    if destination.exists():
        if not args.overwrite:
            raise SystemExit(f"destination exists: {destination}")
        import shutil

        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    writer = StreamingSafetensorsWriter(destination, 1536 * 1024 * 1024)
    stats = {}
    with CheckpointReader(args.target) as target, CheckpointReader(args.donor) as donor:
        target_config = target.config()
        target_layers = int(target_config["num_hidden_layers"])
        donor_layers = int(donor.config()["num_hidden_layers"])
        mapping = uniform_depth_map(target_layers, donor_layers)
        donor_keys = set(donor.keys)
        tied = bool(target_config.get("tie_word_embeddings", False))
        embed_value = None
        for key in target.keys:
            if tied and key == "lm_head.weight":
                continue
            value = target.get_tensor(key)
            mapped = _donor_key(key, mapping)
            output = value
            if mapped in donor_keys:
                donor_value = donor.get_tensor(mapped)
                if (
                    value.ndim == 2
                    and donor_value.ndim == 2
                    and donor_value.shape[0] >= value.shape[0]
                    and donor_value.shape[1] >= value.shape[1]
                ):
                    output, record = spectral_direction_transfer(
                        value, donor_value, rank=args.rank, strength=args.strength
                    )
                    stats[key] = {"donor_key": mapped, **record}
            if key == "model.embed_tokens.weight":
                embed_value = output
            writer.add(key, output.half() if output.is_floating_point() else output)
        if tied and embed_value is not None and "lm_head.weight" in target.keys:
            writer.add("lm_head.weight", embed_value.half())
            stats["lm_head.weight"] = {"tied_to": "model.embed_tokens.weight"}
    writer.finalize()
    copy_sidecars(args.target, destination)
    atomic_write_json(
        destination / "direction_transfer_stats.json",
        {"depth_mapping": mapping, "tensors": stats},
    )
    print(json.dumps({"tensors": len(stats), "output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
