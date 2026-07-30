from __future__ import annotations

"""Reusable decoder-depth reduction helpers.

Phase 2B and Phase 4B both create physically shallower checkpoints. Keeping the
mutation logic here prevents the two phases from silently using different layer
selection or cache-reindexing rules.
"""

from dataclasses import dataclass
from typing import Any, Iterable

from torch import nn

from scripts.common.model_introspection import (
    count_parameters,
    find_decoder_stack,
    replace_decoder_layers,
    update_config_field,
)


@dataclass(frozen=True)
class DepthReductionResult:
    """Description of one in-memory decoder-stack reduction."""

    original_layer_count: int
    retained_current_layer_ids: list[int]
    retained_original_layer_ids: list[int]
    dropped_original_layer_ids: list[int]
    parameters_before: dict[str, int]
    parameters_after: dict[str, int]

    @property
    def parameter_reduction_fraction(self) -> float:
        before = float(self.parameters_before["total"])
        after = float(self.parameters_after["total"])
        return 1.0 - after / before


def choose_layers(total: int, mode: str, count: int, explicit: str = "") -> list[int]:
    """Return current decoder indices retained by a depth-reduction policy."""

    if total < 1:
        raise ValueError("Model has no decoder layers.")

    if mode == "drop-last":
        keep_count = total - count
        if keep_count < 1:
            raise ValueError("Cannot drop every layer.")
        return list(range(keep_count))

    if mode == "keep-first":
        if not 1 <= count <= total:
            raise ValueError("keep-first count is out of range.")
        return list(range(count))

    if mode == "uniform-drop":
        keep_count = total - count
        if keep_count < 1:
            raise ValueError("Cannot drop every layer.")
        return choose_layers(total, "uniform", keep_count, explicit)

    if mode == "uniform":
        if not 1 <= count <= total:
            raise ValueError("uniform keep count is out of range.")
        if count == 1:
            return [total // 2]
        indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
        unique = sorted(set(indices))
        if len(unique) != count:
            raise RuntimeError("Uniform layer selection produced duplicate indices.")
        return unique

    if mode != "explicit":
        raise ValueError(f"Unsupported depth-reduction mode: {mode}")

    indices = sorted({int(part.strip()) for part in explicit.split(",") if part.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= total:
        raise ValueError("Explicit layer list is empty or out of range.")
    return indices


def _config_objects(config: Any) -> list[Any]:
    """Return the root and common nested text configs without duplicates."""

    configs: list[Any] = []
    seen: set[int] = set()
    for candidate in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "decoder", None),
        getattr(config, "language_config", None),
    ):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        configs.append(candidate)
    return configs


def _retain_layer_indexed_config_metadata(
    config: Any,
    *,
    retained_current_layer_ids: list[int],
    original_layer_count: int,
) -> None:
    """Keep per-layer configuration arrays aligned with the reduced stack.

    Recent Transformers Qwen3 configs carry ``layer_types`` with one entry per
    decoder layer. Updating only ``num_hidden_layers`` leaves that list at the
    old length, so strict config validation refuses to save or reload the
    reduced checkpoint. Preserve exactly the entries belonging to retained
    layers, including any full/sliding/linear-attention pattern.
    """

    new_count = len(retained_current_layer_ids)
    for current_config in _config_objects(config):
        if not hasattr(current_config, "layer_types"):
            continue
        layer_types = getattr(current_config, "layer_types")
        if layer_types is None:
            continue
        if not isinstance(layer_types, (list, tuple)):
            raise TypeError(
                "Expected config.layer_types to be a list or tuple, "
                f"got {type(layer_types).__name__}."
            )
        if len(layer_types) == original_layer_count:
            retained = [layer_types[index] for index in retained_current_layer_ids]
            setattr(
                current_config,
                "layer_types",
                tuple(retained) if isinstance(layer_types, tuple) else retained,
            )
        elif len(layer_types) != new_count:
            raise ValueError(
                "Cannot safely reduce decoder depth because config.layer_types "
                f"has {len(layer_types)} entries, expected {original_layer_count} "
                f"before reduction or {new_count} after reduction."
            )


def reindex_decoder_layers(
    model: nn.Module,
    *,
    retained_current_layer_ids: list[int],
    original_layer_count: int,
) -> None:
    """Reindex caches and synchronize layer-indexed config after deletion."""

    stack = find_decoder_stack(model)
    if len(stack.layers) != len(retained_current_layer_ids):
        raise ValueError(
            "Reduced decoder stack length does not match retained layer IDs: "
            f"{len(stack.layers)} != {len(retained_current_layer_ids)}."
        )

    for new_index, layer in enumerate(stack.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_index
        attention = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = new_index

    _retain_layer_indexed_config_metadata(
        model.config,
        retained_current_layer_ids=retained_current_layer_ids,
        original_layer_count=original_layer_count,
    )
    update_config_field(model.config, "num_hidden_layers", len(stack.layers))


def reduce_model_depth(
    model: nn.Module,
    *,
    retained_current_layer_ids: Iterable[int],
    current_original_layer_ids: list[int] | None = None,
) -> DepthReductionResult:
    """Physically retain selected current blocks and preserve original IDs.

    ``current_original_layer_ids`` maps current positions to the source model's
    original layer numbering. It allows progressive deletion reports to keep
    saying "original layer 14" even after earlier deletions have reindexed the
    checkpoint.
    """

    stack = find_decoder_stack(model)
    total = len(stack.layers)
    keep = sorted({int(index) for index in retained_current_layer_ids})
    if not keep or keep[0] < 0 or keep[-1] >= total:
        raise ValueError("Retained current layer IDs are empty or out of range.")
    if len(keep) >= total:
        raise ValueError("Depth reduction must remove at least one layer.")

    original_map = current_original_layer_ids or list(range(total))
    if len(original_map) != total:
        raise ValueError(
            "current_original_layer_ids must contain one original ID per current layer."
        )

    before = count_parameters(model)
    retained_original = [int(original_map[index]) for index in keep]
    dropped_original = [
        int(original_map[index]) for index in range(total) if index not in set(keep)
    ]
    replace_decoder_layers(model, [stack.layers[index] for index in keep])
    reindex_decoder_layers(
        model,
        retained_current_layer_ids=keep,
        original_layer_count=total,
    )
    after = count_parameters(model)

    return DepthReductionResult(
        original_layer_count=total,
        retained_current_layer_ids=keep,
        retained_original_layer_ids=retained_original,
        dropped_original_layer_ids=dropped_original,
        parameters_before=before,
        parameters_after=after,
    )
