from __future__ import annotations

"""Shared helpers for Phase 10 student architecture reduction.

The dependency-light pieces in this module are intentionally separated from
Transformers model loading so candidate planning, registry construction, and
unit tests can run without a downloaded checkpoint.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from scripts.common.model_introspection import (
    find_decoder_stack,
    get_gated_mlp,
    get_self_attention,
    linear_from_slices,
    update_config_field,
)

PHASE_10_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    description: str
    depth_drop: int = 0
    depth_mode: str = "uniform-drop"
    hidden_prune_ratio: float = 0.0
    mlp_prune_ratio: float = 0.0
    attention_group_prune_ratio: float = 0.0
    recovery_eligible: bool = True
    production_eligible: bool = True


def parse_candidate_spec(row: dict[str, Any]) -> CandidateSpec:
    name = str(row.get("name") or "").strip()
    if not name:
        raise ValueError("Phase 10 candidate is missing name")
    depth_mode = str(row.get("depth_mode") or "uniform-drop")
    if depth_mode not in {"uniform-drop", "drop-last"}:
        raise ValueError(f"Unsupported depth mode for {name}: {depth_mode}")
    ratios = {
        "hidden_prune_ratio": float(row.get("hidden_prune_ratio", 0.0)),
        "mlp_prune_ratio": float(row.get("mlp_prune_ratio", 0.0)),
        "attention_group_prune_ratio": float(row.get("attention_group_prune_ratio", 0.0)),
    }
    for key, value in ratios.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} {key} must be in [0,1)")
    depth_drop = int(row.get("depth_drop", 0))
    if depth_drop < 0:
        raise ValueError(f"{name} depth_drop cannot be negative")
    if depth_drop == 0 and not any(value > 0 for value in ratios.values()):
        raise ValueError(f"{name} performs no architecture reduction")
    return CandidateSpec(
        name=name,
        description=str(row.get("description") or name),
        depth_drop=depth_drop,
        depth_mode=depth_mode,
        hidden_prune_ratio=ratios["hidden_prune_ratio"],
        mlp_prune_ratio=ratios["mlp_prune_ratio"],
        attention_group_prune_ratio=ratios["attention_group_prune_ratio"],
        recovery_eligible=bool(row.get("recovery_eligible", True)),
        production_eligible=bool(row.get("production_eligible", True)),
    )


def load_candidate_specs(config: dict[str, Any], mode: str) -> list[CandidateSpec]:
    normalized = mode.strip().lower()
    if normalized not in {"pilot", "full"}:
        raise ValueError("Phase 10 mode must be pilot or full")
    section = (config.get("candidate_sets") or {}).get(normalized)
    if not isinstance(section, list) or not section:
        raise ValueError(f"No Phase 10 candidate set configured for {normalized}")
    specs = [parse_candidate_spec(dict(row)) for row in section]
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Phase 10 candidate names must be unique")
    return specs


def round_retained_width(original: int, prune_ratio: float, multiple: int, *, minimum: int | None = None) -> int:
    if original <= 0 or multiple <= 0:
        raise ValueError("original and multiple must be positive")
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError("prune_ratio must be in [0,1)")
    minimum = multiple if minimum is None else int(minimum)
    raw = int(math.floor(original * (1.0 - prune_ratio)))
    retained = max(minimum, (raw // multiple) * multiple)
    return min(original, retained)


def uniform_drop_indices(layer_count: int, drop_count: int) -> list[int]:
    if drop_count <= 0:
        return []
    if drop_count >= layer_count:
        raise ValueError("Cannot drop every decoder layer")
    if drop_count == 1:
        return [layer_count // 2]
    raw = [round(index * (layer_count - 1) / (drop_count + 1)) for index in range(1, drop_count + 1)]
    selected: list[int] = []
    for value in raw:
        candidate = max(0, min(layer_count - 1, int(value)))
        while candidate in selected and candidate + 1 < layer_count:
            candidate += 1
        while candidate in selected and candidate - 1 >= 0:
            candidate -= 1
        if candidate in selected:
            raise RuntimeError("Could not construct unique uniform layer drops")
        selected.append(candidate)
    return sorted(selected)


def retained_layer_indices(layer_count: int, *, drop_count: int, mode: str) -> list[int]:
    if drop_count == 0:
        return list(range(layer_count))
    if mode == "drop-last":
        dropped = set(range(layer_count - drop_count, layer_count))
    elif mode == "uniform-drop":
        dropped = set(uniform_drop_indices(layer_count, drop_count))
    else:
        raise ValueError(f"Unsupported depth mode: {mode}")
    retained = [index for index in range(layer_count) if index not in dropped]
    if not retained:
        raise ValueError("Depth candidate retained no decoder layers")
    return retained


def top_indices(values: torch.Tensor, count: int) -> torch.Tensor:
    flat = values.detach().float().flatten()
    if count <= 0 or count > flat.numel():
        raise ValueError(f"Invalid retained index count {count} for {flat.numel()} values")
    # Sort selected coordinates to preserve original tensor order after ranking.
    return torch.topk(flat, k=count, largest=True, sorted=False).indices.sort().values


def replace_embedding_columns(source: nn.Embedding, indices: torch.Tensor) -> nn.Embedding:
    indices = indices.to(device=source.weight.device, dtype=torch.long)
    replacement = nn.Embedding(
        source.num_embeddings,
        int(indices.numel()),
        padding_idx=source.padding_idx,
        max_norm=source.max_norm,
        norm_type=source.norm_type,
        scale_grad_by_freq=source.scale_grad_by_freq,
        sparse=source.sparse,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(source.weight.index_select(1, indices))
    return replacement


def slice_norm_parameter(module: nn.Module, indices: torch.Tensor, original_hidden_size: int) -> bool:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter) or weight.ndim != 1 or weight.numel() != original_hidden_size:
        return False
    selected = indices.to(device=weight.device, dtype=torch.long)
    module.weight = nn.Parameter(weight.detach().index_select(0, selected).clone(), requires_grad=weight.requires_grad)
    bias = getattr(module, "bias", None)
    if isinstance(bias, nn.Parameter) and bias.ndim == 1 and bias.numel() == original_hidden_size:
        module.bias = nn.Parameter(bias.detach().index_select(0, selected).clone(), requires_grad=bias.requires_grad)
    if hasattr(module, "normalized_shape"):
        module.normalized_shape = (int(selected.numel()),)
    return True


def apply_mlp_channel_slicing(
    model: nn.Module,
    *,
    importance_by_original_layer: dict[int, torch.Tensor],
    current_original_layer_ids: list[int],
    prune_ratio: float,
    multiple: int,
) -> dict[str, Any]:
    if prune_ratio <= 0.0:
        stack = find_decoder_stack(model)
        width = int(get_gated_mlp(stack.layers[0]).gate_proj.out_features)
        return {"original_width": width, "retained_width": width, "selected_indices": {}}
    stack = find_decoder_stack(model)
    if len(stack.layers) != len(current_original_layer_ids):
        raise ValueError("current_original_layer_ids length does not match decoder stack")
    original_width = int(get_gated_mlp(stack.layers[0]).gate_proj.out_features)
    retained_width = round_retained_width(original_width, prune_ratio, multiple)
    if retained_width >= original_width:
        raise ValueError("MLP prune ratio rounded to no structural reduction")
    selected_manifest: dict[str, list[int]] = {}
    for current_index, layer in enumerate(stack.layers):
        original_index = int(current_original_layer_ids[current_index])
        importance = importance_by_original_layer.get(original_index)
        if importance is None or int(importance.numel()) != original_width:
            raise ValueError(f"Missing compatible MLP importance for original layer {original_index}")
        keep = top_indices(importance, retained_width)
        mlp = get_gated_mlp(layer)
        mlp.gate_proj = linear_from_slices(mlp.gate_proj, row_indices=keep)
        mlp.up_proj = linear_from_slices(mlp.up_proj, row_indices=keep)
        mlp.down_proj = linear_from_slices(mlp.down_proj, column_indices=keep)
        for attr in ("intermediate_size", "intermediate_dim"):
            if hasattr(mlp, attr):
                setattr(mlp, attr, retained_width)
        selected_manifest[str(original_index)] = [int(value) for value in keep.tolist()]
    update_config_field(model.config, "intermediate_size", retained_width)
    return {
        "original_width": original_width,
        "retained_width": retained_width,
        "selected_indices": selected_manifest,
    }


def _head_channels(head_indices: Iterable[int], head_dim: int, device: torch.device) -> torch.Tensor:
    values: list[int] = []
    for head in head_indices:
        values.extend(range(int(head) * head_dim, (int(head) + 1) * head_dim))
    return torch.tensor(values, dtype=torch.long, device=device)


def apply_attention_group_slicing(
    model: nn.Module,
    *,
    importance_by_original_layer: dict[int, torch.Tensor],
    current_original_layer_ids: list[int],
    prune_ratio: float,
) -> dict[str, Any]:
    stack = find_decoder_stack(model)
    first = get_self_attention(stack.layers[0])
    q_heads = int(getattr(first, "num_heads", getattr(model.config, "num_attention_heads")))
    kv_heads = int(getattr(first, "num_key_value_heads", getattr(model.config, "num_key_value_heads", q_heads)))
    if q_heads % kv_heads != 0:
        raise ValueError("Grouped-query attention topology is invalid")
    group_size = q_heads // kv_heads
    head_dim = int(first.q_proj.out_features // q_heads)
    if prune_ratio <= 0.0:
        return {
            "original_query_heads": q_heads,
            "original_kv_heads": kv_heads,
            "retained_query_heads": q_heads,
            "retained_kv_heads": kv_heads,
            "selected_kv_groups": {},
        }
    retained_kv = max(1, int(math.floor(kv_heads * (1.0 - prune_ratio))))
    if retained_kv >= kv_heads:
        raise ValueError("Attention prune ratio rounded to no structural reduction")
    retained_q = retained_kv * group_size
    selected_manifest: dict[str, list[int]] = {}
    for current_index, layer in enumerate(stack.layers):
        original_index = int(current_original_layer_ids[current_index])
        head_importance = importance_by_original_layer.get(original_index)
        if head_importance is None or int(head_importance.numel()) != q_heads:
            raise ValueError(f"Missing compatible attention-head importance for original layer {original_index}")
        group_scores = head_importance.detach().float().reshape(kv_heads, group_size).sum(dim=1)
        keep_groups = top_indices(group_scores, retained_kv)
        keep_group_list = [int(value) for value in keep_groups.tolist()]
        keep_query_heads = [group * group_size + offset for group in keep_group_list for offset in range(group_size)]
        attention = get_self_attention(layer)
        q_channels = _head_channels(keep_query_heads, head_dim, attention.q_proj.weight.device)
        kv_channels = _head_channels(keep_group_list, head_dim, attention.k_proj.weight.device)
        attention.q_proj = linear_from_slices(attention.q_proj, row_indices=q_channels)
        attention.k_proj = linear_from_slices(attention.k_proj, row_indices=kv_channels)
        attention.v_proj = linear_from_slices(attention.v_proj, row_indices=kv_channels)
        attention.o_proj = linear_from_slices(attention.o_proj, column_indices=q_channels)
        for attr, value in (
            ("num_heads", retained_q),
            ("num_attention_heads", retained_q),
            ("num_key_value_heads", retained_kv),
            ("num_key_value_groups", group_size),
        ):
            if hasattr(attention, attr):
                setattr(attention, attr, value)
        selected_manifest[str(original_index)] = keep_group_list
    update_config_field(model.config, "num_attention_heads", retained_q)
    update_config_field(model.config, "num_key_value_heads", retained_kv)
    update_config_field(model.config, "head_dim", head_dim)
    return {
        "original_query_heads": q_heads,
        "original_kv_heads": kv_heads,
        "retained_query_heads": retained_q,
        "retained_kv_heads": retained_kv,
        "query_heads_per_kv_group": group_size,
        "head_dim": head_dim,
        "selected_kv_groups": selected_manifest,
    }


def apply_hidden_coordinate_slicing(
    model: nn.Module,
    *,
    hidden_importance: torch.Tensor,
    prune_ratio: float,
    multiple: int,
) -> dict[str, Any]:
    input_embeddings = model.get_input_embeddings()
    if not isinstance(input_embeddings, nn.Embedding):
        raise RuntimeError("Phase 10 hidden slicing requires nn.Embedding input embeddings")
    original_hidden = int(input_embeddings.embedding_dim)
    if hidden_importance.numel() != original_hidden:
        raise ValueError(
            f"Hidden importance has {hidden_importance.numel()} entries, expected {original_hidden}"
        )
    if prune_ratio <= 0.0:
        return {
            "original_hidden_size": original_hidden,
            "retained_hidden_size": original_hidden,
            "selected_indices": list(range(original_hidden)),
        }
    retained_hidden = round_retained_width(original_hidden, prune_ratio, multiple, minimum=multiple)
    if retained_hidden >= original_hidden:
        raise ValueError("Hidden prune ratio rounded to no structural reduction")
    keep = top_indices(hidden_importance, retained_hidden)

    # Slice every residual-stream entry/exit point explicitly. Projection output
    # dimensions for q/k/v and MLP expansion remain unchanged here; separate
    # head/MLP pruning owns those dimensions.
    model.set_input_embeddings(replace_embedding_columns(input_embeddings, keep))
    stack = find_decoder_stack(model)
    for layer in stack.layers:
        attention = get_self_attention(layer)
        attention.q_proj = linear_from_slices(attention.q_proj, column_indices=keep)
        attention.k_proj = linear_from_slices(attention.k_proj, column_indices=keep)
        attention.v_proj = linear_from_slices(attention.v_proj, column_indices=keep)
        attention.o_proj = linear_from_slices(attention.o_proj, row_indices=keep)
        mlp = get_gated_mlp(layer)
        mlp.gate_proj = linear_from_slices(mlp.gate_proj, column_indices=keep)
        mlp.up_proj = linear_from_slices(mlp.up_proj, column_indices=keep)
        mlp.down_proj = linear_from_slices(mlp.down_proj, row_indices=keep)
        for module in layer.modules():
            slice_norm_parameter(module, keep, original_hidden)
        for attr in ("hidden_size", "embed_dim"):
            if hasattr(attention, attr):
                setattr(attention, attr, retained_hidden)
            if hasattr(mlp, attr):
                setattr(mlp, attr, retained_hidden)

    # Slice final/global norms outside the decoder blocks.
    layer_module_ids = {id(module) for layer in stack.layers for module in layer.modules()}
    for module in model.modules():
        if id(module) not in layer_module_ids:
            slice_norm_parameter(module, keep, original_hidden)

    output_embeddings = model.get_output_embeddings()
    if not isinstance(output_embeddings, nn.Linear):
        raise RuntimeError("Phase 10 hidden slicing requires nn.Linear output embeddings")
    model.set_output_embeddings(linear_from_slices(output_embeddings, column_indices=keep))
    update_config_field(model.config, "hidden_size", retained_hidden)
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        model.tie_weights()
    return {
        "original_hidden_size": original_hidden,
        "retained_hidden_size": retained_hidden,
        "selected_indices": [int(value) for value in keep.tolist()],
    }


def candidate_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Candidate must use name=summary.json format")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not path.strip():
        raise ValueError("Candidate name and path cannot be empty")
    return name, Path(path)
