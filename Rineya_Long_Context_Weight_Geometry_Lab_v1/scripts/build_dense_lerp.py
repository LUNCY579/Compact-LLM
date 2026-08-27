#!/usr/bin/env python3
"""Stream a shape-compatible two-checkpoint LERP through the existing soup engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOUP_SRC = REPO / "Rineya_Layerwise_Soup_Lab_v1.0" / "src"
if str(SOUP_SRC) not in sys.path:
    sys.path.insert(0, str(SOUP_SRC))

from rineya_soup_lab.model.layer_map import MergeRecipe
from rineya_soup_lab.model.merge import merge_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--greedy", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, required=True)
    parser.add_argument(
        "--layer-key-regex", default=r"(?:^|\.)layers\.(\d+)\."
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float16", "bfloat16", "preserve_anchor"),
        default="bfloat16",
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must lie in [0, 1]")
    if args.expected_layers <= 0:
        parser.error("--expected-layers must be positive")

    anchor = args.anchor.resolve(strict=True)
    greedy = args.greedy.resolve(strict=True)
    output = args.output.resolve()
    for source in (anchor, greedy):
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if int(config.get("num_hidden_layers", 0)) != args.expected_layers:
            raise SystemExit(
                f"{source} has {config.get('num_hidden_layers')} layers; "
                f"expected {args.expected_layers}"
            )

    recipe = MergeRecipe(
        global_alpha=args.alpha,
        label=f"dense_lerp_alpha_{args.alpha:g}",
        metadata={"experiment": "long_context_published_core_search"},
    )
    manifest = merge_checkpoints(
        anchor,
        greedy,
        output,
        recipe,
        layer_key_regex=args.layer_key_regex,
        accumulation_dtype="float32",
        output_dtype=args.output_dtype,
        max_shard_size_gb=1.5,
        require_non_floating_equal=True,
        copy_non_floating_from="anchor",
        overwrite=False,
        allow_tied_lm_head_alias=True,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_id": manifest["candidate_id"],
                "tensor_count": manifest["tensor_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
