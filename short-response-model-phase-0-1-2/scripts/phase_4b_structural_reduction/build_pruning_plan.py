# Purpose: Converts attribution traces into auditable structured-pruning plans and equal-size controls.
from __future__ import annotations

"""Convert Phase 4B attribution statistics into a reloadable pruning plan.

The plan keeps the same retained MLP width and grouped-attention topology in
every decoder layer so standard Transformers checkpoint loading can reconstruct
the modified architecture. This file selects indices only; it never edits model
weights.
"""

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import write_json


def parse_args() -> argparse.Namespace:
    """Configure profile input, structural budgets, and control policy."""

    parser = argparse.ArgumentParser(
        description="Build a uniform-width MLP/GQA pruning plan for Phase 4B."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "phase_4b_structural_reduction"
        / "pruning_plan.json",
    )
    parser.add_argument("--mlp-prune-ratio", type=float, default=0.10)
    parser.add_argument("--attention-group-prune-ratio", type=float, default=0.0)
    parser.add_argument(
        "--mlp-multiple",
        type=int,
        default=64,
        help="Round retained MLP width down to this hardware-friendly multiple.",
    )
    parser.add_argument(
        "--short-importance-quantile",
        type=float,
        default=0.60,
        help="Prefer pruning channels below this short-use quantile.",
    )
    parser.add_argument(
        "--selection-method",
        choices=("long-specific", "lowest-short", "random"),
        default="long-specific",
        help=(
            "long-specific uses the attribution hypothesis; lowest-short and "
            "random are controls."
        ),
    )
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def choose_long_specific(
    short: torch.Tensor,
    specificity: torch.Tensor,
    prune_count: int,
    quantile: float,
) -> list[int]:
    """Prefer late-specific components that are less important to short replies."""

    if prune_count <= 0:
        return []

    # Restrict the primary ranking to components below the chosen short-answer
    # importance threshold. If too few qualify, extend with the global
    # late-specific ranking rather than changing the requested tensor width.
    threshold = torch.quantile(short.float(), quantile)
    eligible = torch.nonzero(short <= threshold, as_tuple=False).flatten()
    ranked = eligible[torch.argsort(specificity[eligible], descending=True)]
    if ranked.numel() < prune_count:
        fallback = torch.argsort(specificity, descending=True)
        merged: list[int] = ranked.tolist()
        seen = set(merged)
        for index in fallback.tolist():
            if index not in seen:
                merged.append(index)
                seen.add(index)
            if len(merged) >= prune_count:
                break
        return sorted(merged[:prune_count])
    return sorted(ranked[:prune_count].tolist())


def choose_indices(
    short: torch.Tensor,
    specificity: torch.Tensor,
    prune_count: int,
    quantile: float,
    method: str,
    rng: random.Random,
) -> list[int]:
    """Dispatch the main attribution policy or one equal-size control."""

    if method == "long-specific":
        return choose_long_specific(short, specificity, prune_count, quantile)
    if method == "lowest-short":
        return sorted(torch.argsort(short)[:prune_count].tolist())
    population = list(range(short.numel()))
    return sorted(rng.sample(population, prune_count)) if prune_count else []


def main() -> None:
    args = parse_args()
    if not 0 <= args.mlp_prune_ratio < 1:
        raise ValueError("MLP prune ratio must be in [0, 1).")
    if not 0 <= args.attention_group_prune_ratio < 1:
        raise ValueError("Attention-group prune ratio must be in [0, 1).")
    if not 0 < args.short_importance_quantile <= 1:
        raise ValueError("Short-importance quantile must be in (0, 1].")
    if args.mlp_multiple < 1:
        raise ValueError("MLP multiple must be positive.")

    profile = torch.load(args.profile, map_location="cpu", weights_only=False)
    rng = random.Random(args.seed)
    q_heads = int(profile["num_attention_heads"])
    kv_heads = int(profile["num_key_value_heads"])
    if q_heads % kv_heads != 0:
        raise ValueError(
            "Query heads must be divisible by key/value heads for grouped pruning."
        )
    group_size = q_heads // kv_heads

    plan_layers: dict[str, Any] = {}
    retained_widths: set[int] = set()
    retained_kv_counts: set[int] = set()

    for layer_id, values in profile["layers"].items():
        # Gated MLP pruning must remove matching indices from gate/up rows and
        # down-projection columns. The plan records one shared index set.
        mlp_short = values["mlp"]["short_mean"].float()
        mlp_specific = values["mlp"]["long_specific_score"].float()
        original_width = int(mlp_short.numel())
        desired_keep = int(math.floor(original_width * (1 - args.mlp_prune_ratio)))
        desired_keep = max(
            min(args.mlp_multiple, original_width),
            (desired_keep // args.mlp_multiple) * args.mlp_multiple,
        )
        desired_keep = min(original_width, desired_keep)
        mlp_prune_count = original_width - desired_keep
        prune_channels = choose_indices(
            mlp_short,
            mlp_specific,
            mlp_prune_count,
            args.short_importance_quantile,
            args.selection_method,
            rng,
        )
        prune_set = set(prune_channels)
        keep_channels = [
            index for index in range(original_width) if index not in prune_set
        ]
        retained_widths.add(len(keep_channels))

        # For grouped-query attention, prune complete KV groups together with
        # all query heads assigned to that group. Individual head slicing would
        # break the GQA relationship.
        head_short = values["attention"]["short_mean"].float()
        head_specific = values["attention"]["long_specific_score"].float()
        grouped_short = head_short.reshape(kv_heads, group_size).mean(dim=1)
        grouped_specific = head_specific.reshape(kv_heads, group_size).mean(dim=1)
        keep_kv = max(
            1, int(round(kv_heads * (1 - args.attention_group_prune_ratio)))
        )
        prune_kv_count = kv_heads - keep_kv
        prune_groups = choose_indices(
            grouped_short,
            grouped_specific,
            prune_kv_count,
            args.short_importance_quantile,
            args.selection_method,
            rng,
        )
        prune_group_set = set(prune_groups)
        keep_groups = [
            index for index in range(kv_heads) if index not in prune_group_set
        ]
        retained_kv_counts.add(len(keep_groups))

        plan_layers[layer_id] = {
            "mlp_original_width": original_width,
            "mlp_keep_indices": keep_channels,
            "mlp_pruned_indices": prune_channels,
            "attention_keep_kv_groups": keep_groups,
            "attention_pruned_kv_groups": prune_groups,
        }

    # Standard model configuration stores one global intermediate size and head
    # count, so all layers must retain compatible shapes.
    if len(retained_widths) != 1 or len(retained_kv_counts) != 1:
        raise RuntimeError(
            "The plan must retain the same tensor dimensions in every layer."
        )

    payload = {
        "version": 1,
        "phase": "4B",
        "source_profile": str(args.profile),
        "method": args.selection_method,
        "seed": args.seed,
        "mlp_prune_ratio_requested": args.mlp_prune_ratio,
        "attention_group_prune_ratio_requested": args.attention_group_prune_ratio,
        "original_num_attention_heads": q_heads,
        "original_num_key_value_heads": kv_heads,
        "query_heads_per_kv_group": group_size,
        "retained_mlp_width": next(iter(retained_widths)),
        "retained_num_key_value_heads": next(iter(retained_kv_counts)),
        "retained_num_attention_heads": next(iter(retained_kv_counts)) * group_size,
        "layers": plan_layers,
        "warning": (
            "A pruning plan is not a quality result. Apply it to a copy, reload "
            "the checkpoint, and evaluate every product-critical category."
        ),
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
