from __future__ import annotations

"""Physically slice MLP channels and optional GQA head groups.

Unlike masks, this creates smaller tensors and therefore can reduce checkpoint
size and weight memory. It supports Qwen/Llama-style gate/up/down MLPs and
q/k/v/o attention. Unsupported models stop before saving anything.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.model_introspection import (
    count_parameters,
    find_decoder_stack,
    get_gated_mlp,
    get_self_attention,
    linear_from_slices,
    update_config_field,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an experimental structured-pruning plan.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase_4b_structural_reduction/pruned_model")
    parser.add_argument("--smoke-text", default="What is photosynthesis?")
    parser.add_argument("--skip-attention", action="store_true", help="Apply only the lower-risk MLP pruning part of the plan.")
    parser.add_argument("--skip-reload-check", action="store_true", help="Skip the fresh-process-equivalent reload smoke check.")
    return parser.parse_args()


def head_channel_indices(head_indices: list[int], head_dim: int, device: torch.device) -> torch.Tensor:
    channels: list[int] = []
    for head in head_indices:
        channels.extend(range(head * head_dim, (head + 1) * head_dim))
    return torch.tensor(channels, dtype=torch.long, device=device)


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    stack = find_decoder_stack(model)
    if len(stack.layers) != len(plan["layers"]):
        raise ValueError("Plan layer count does not match the loaded model.")

    before = count_parameters(model)
    original_q_heads = int(plan["original_num_attention_heads"])
    original_kv_heads = int(plan["original_num_key_value_heads"])
    group_size = int(plan["query_heads_per_kv_group"])
    first_attention = get_self_attention(stack.layers[0])
    original_head_dim = first_attention.q_proj.out_features // original_q_heads

    for layer_index, layer in enumerate(stack.layers):
        layer_plan = plan["layers"][str(layer_index)]
        mlp = get_gated_mlp(layer)
        keep = torch.tensor(layer_plan["mlp_keep_indices"], dtype=torch.long, device=mlp.gate_proj.weight.device)
        if mlp.gate_proj.out_features != int(layer_plan["mlp_original_width"]):
            raise ValueError(f"Layer {layer_index} MLP width does not match plan.")
        mlp.gate_proj = linear_from_slices(mlp.gate_proj, row_indices=keep)
        mlp.up_proj = linear_from_slices(mlp.up_proj, row_indices=keep)
        mlp.down_proj = linear_from_slices(mlp.down_proj, column_indices=keep)
        for attr in ("intermediate_size", "intermediate_dim"):
            if hasattr(mlp, attr):
                setattr(mlp, attr, int(keep.numel()))

        if args.skip_attention:
            continue
        attention = get_self_attention(layer)
        keep_kv_groups = [int(value) for value in layer_plan["attention_keep_kv_groups"]]
        if len(keep_kv_groups) == original_kv_heads:
            continue
        head_dim = attention.q_proj.out_features // original_q_heads
        if head_dim != original_head_dim:
            raise ValueError("Per-layer attention head dimensions differ; uniform structured pruning is unsupported.")
        query_heads = [
            group * group_size + offset
            for group in keep_kv_groups
            for offset in range(group_size)
        ]
        q_channels = head_channel_indices(query_heads, head_dim, attention.q_proj.weight.device)
        kv_channels = head_channel_indices(keep_kv_groups, head_dim, attention.k_proj.weight.device)
        attention.q_proj = linear_from_slices(attention.q_proj, row_indices=q_channels)
        attention.k_proj = linear_from_slices(attention.k_proj, row_indices=kv_channels)
        attention.v_proj = linear_from_slices(attention.v_proj, row_indices=kv_channels)
        attention.o_proj = linear_from_slices(attention.o_proj, column_indices=q_channels)
        new_q_heads = len(query_heads)
        new_kv_heads = len(keep_kv_groups)
        for attr, value in (
            ("num_heads", new_q_heads),
            ("num_attention_heads", new_q_heads),
            ("num_key_value_heads", new_kv_heads),
            ("num_key_value_groups", new_q_heads // new_kv_heads),
        ):
            if hasattr(attention, attr):
                setattr(attention, attr, value)

    retained_width = int(plan["retained_mlp_width"])
    update_config_field(model.config, "intermediate_size", retained_width)
    if not args.skip_attention:
        update_config_field(model.config, "num_attention_heads", int(plan["retained_num_attention_heads"]))
        update_config_field(model.config, "num_key_value_heads", int(plan["retained_num_key_value_heads"]))
        # Preserve the original per-head width. Without an explicit head_dim,
        # some model constructors derive hidden_size / new_num_heads and build
        # projection shapes that cannot reload the sliced checkpoint.
        update_config_field(model.config, "head_dim", int(original_head_dim))

    # A real forward pass catches most incompatible shape/config assumptions
    # before an unusable checkpoint is written.
    encoded = tokenizer(args.smoke_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Smoke forward produced non-finite logits.")

    after = count_parameters(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    reload_logits_shape = None
    if not args.skip_reload_check:
        reloaded = AutoModelForCausalLM.from_pretrained(
            args.output_dir, torch_dtype="auto", low_cpu_mem_usage=True
        )
        reloaded.eval()
        with torch.no_grad():
            reload_output = reloaded(**encoded)
        if not torch.isfinite(reload_output.logits).all():
            raise RuntimeError("Reloaded pruned checkpoint produced non-finite logits.")
        reload_logits_shape = list(reload_output.logits.shape)

    manifest = {
        "source_model": args.model,
        "plan": str(args.plan),
        "skip_attention": args.skip_attention,
        "parameters_before": before,
        "parameters_after": after,
        "parameter_reduction_fraction": 1 - after["total"] / before["total"],
        "smoke_text": args.smoke_text,
        "smoke_logits_shape": list(output.logits.shape),
        "reload_check_performed": not args.skip_reload_check,
        "reload_smoke_logits_shape": reload_logits_shape,
        "warning": "Passing a smoke forward is not a quality gate. Evaluate all benchmark categories before use.",
    }
    write_json(args.output_dir / "structured_pruning_manifest.json", manifest)
    print(args.output_dir)


if __name__ == "__main__":
    main()
