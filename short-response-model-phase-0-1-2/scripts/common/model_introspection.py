from __future__ import annotations

"""Architecture helpers shared by the experimental reduction/sparsity scripts.

The project primarily targets Qwen/Llama-style decoder-only models, but these
helpers deliberately discover modules from the loaded object instead of relying
on one hard-coded checkpoint path. Unsupported architectures fail loudly rather
than silently pruning the wrong tensors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class DecoderStack:
    """A resolved decoder layer container and its dotted module path."""

    layers: nn.ModuleList
    path: str


KNOWN_LAYER_PATHS = (
    "model.layers",          # Llama, Mistral, Qwen2/Qwen3 families
    "transformer.h",         # GPT-2 style
    "gpt_neox.layers",       # GPT-NeoX/Pythia style
    "model.decoder.layers",  # some encoder-decoder wrappers
)


def _get_dotted(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(path)
        current = getattr(current, part)
    return current


def _set_dotted(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def find_decoder_stack(model: nn.Module) -> DecoderStack:
    """Locate the transformer block ModuleList on a loaded causal LM."""

    for path in KNOWN_LAYER_PATHS:
        try:
            candidate = _get_dotted(model, path)
        except AttributeError:
            continue
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return DecoderStack(candidate, path)
    raise RuntimeError(
        "Could not locate decoder layers. Supported common paths are: "
        + ", ".join(KNOWN_LAYER_PATHS)
    )


def replace_decoder_layers(model: nn.Module, layers: Iterable[nn.Module]) -> DecoderStack:
    """Replace the decoder stack while retaining the original dotted path."""

    stack = find_decoder_stack(model)
    replacement = nn.ModuleList(list(layers))
    _set_dotted(model, stack.path, replacement)
    return DecoderStack(replacement, stack.path)


def get_gated_mlp(layer: nn.Module) -> nn.Module:
    """Return a gate/up/down MLP or fail with an actionable message."""

    mlp = getattr(layer, "mlp", None)
    required = ("gate_proj", "up_proj", "down_proj")
    if mlp is None or not all(hasattr(mlp, name) for name in required):
        raise RuntimeError(
            f"Layer {type(layer).__name__} is not a supported gated MLP. "
            "Expected layer.mlp.gate_proj/up_proj/down_proj."
        )
    if not all(isinstance(getattr(mlp, name), nn.Linear) for name in required):
        raise RuntimeError("Structured pruning currently requires torch.nn.Linear MLP projections.")
    if not hasattr(mlp, "act_fn"):
        raise RuntimeError("Supported gated MLPs must expose an act_fn callable.")
    return mlp


def get_self_attention(layer: nn.Module) -> nn.Module:
    """Return a Qwen/Llama-style self-attention module."""

    attention = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    required = ("q_proj", "k_proj", "v_proj", "o_proj")
    if attention is None or not all(hasattr(attention, name) for name in required):
        raise RuntimeError(
            f"Layer {type(layer).__name__} lacks q/k/v/o attention projections."
        )
    if not all(isinstance(getattr(attention, name), nn.Linear) for name in required):
        raise RuntimeError("Structured head pruning requires torch.nn.Linear projections.")
    return attention


def layer_id_from_module_name(name: str) -> int:
    """Extract the integer following a '.layers.' path component."""

    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    raise ValueError(f"Cannot derive decoder layer ID from module name: {name}")


def linear_from_slices(
    source: nn.Linear,
    *,
    row_indices: torch.Tensor | None = None,
    column_indices: torch.Tensor | None = None,
) -> nn.Linear:
    """Create a new Linear layer by structurally slicing rows and/or columns.

    Rows correspond to output features; columns correspond to input features.
    The returned layer has real smaller tensors, so saved checkpoints shrink.
    """

    device = source.weight.device
    row_indices = (
        torch.arange(source.out_features, device=device)
        if row_indices is None
        else row_indices.to(device=device, dtype=torch.long)
    )
    column_indices = (
        torch.arange(source.in_features, device=device)
        if column_indices is None
        else column_indices.to(device=device, dtype=torch.long)
    )
    new_layer = nn.Linear(
        int(column_indices.numel()),
        int(row_indices.numel()),
        bias=source.bias is not None,
        device=device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        new_layer.weight.copy_(source.weight.index_select(0, row_indices).index_select(1, column_indices))
        if source.bias is not None:
            new_layer.bias.copy_(source.bias.index_select(0, row_indices))
    return new_layer


def update_config_field(config: Any, name: str, value: Any) -> None:
    """Update a field on the root config and common nested text configs."""

    # PretrainedConfig preserves arbitrary public attributes when saved.
    # Setting the root unconditionally is important for fields such as
    # head_dim that some architectures derive unless explicitly provided.
    setattr(config, name, value)
    for nested_name in ("text_config", "decoder", "language_config"):
        nested = getattr(config, nested_name, None)
        if nested is not None and hasattr(nested, name):
            setattr(nested, name, value)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def first_parameter_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters.") from exc


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
