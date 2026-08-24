"""SDPA adapter for exact padded heterogeneous KV caches."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch

from .heterogeneous_cache import (
    HeterogeneousSignificanceCache,
    HeterogeneousSignificanceLayer,
)
from .masks import bottom_right_causal_allowed
from .scoring import post_rope_gqa_significance
from .transformers_runtime import _PATCH_LOCK

_ACTIVE_HETEROGENEOUS_CACHE: ContextVar[HeterogeneousSignificanceCache | None] = ContextVar(
    "rineya_active_heterogeneous_cache", default=None
)


def _rank4_attention_mask(mask: torch.Tensor, *, query_length: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.view(1, 1, *mask.shape)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"attention mask must have rank 2, 3, or 4; got {mask.ndim}")
    if mask.shape[-2] < query_length:
        raise ValueError(
            f"attention mask has {mask.shape[-2]} query rows, expected at least {query_length}"
        )
    return mask[..., -query_length:, :]


def _remap_layer_slots(
    mask: torch.Tensor,
    *,
    query_length: int,
    key_length: int,
) -> torch.Tensor:
    """Map a max-layer mask to ``old-prefix + this-layer-new-suffix`` slots."""

    source_length = int(mask.shape[-1])
    if source_length == key_length:
        return mask
    if source_length < key_length:
        raise ValueError(
            f"attention mask key width {source_length} is shorter than layer width {key_length}"
        )
    old_length = key_length - query_length
    source_old_length = source_length - query_length
    if old_length < 0 or source_old_length < old_length:
        raise ValueError(
            f"cannot remap mask width {source_length} to key/query {(key_length, query_length)}"
        )
    # Runtime masks are laid out as [resident max-layer prefix, new query
    # suffix].  A shorter layer owns only its old prefix but shares that exact
    # new suffix.  A naive ``[..., :key_length]`` would accidentally replace
    # new-token columns with padding columns from a longer layer.
    return torch.cat(
        [mask[..., :old_length], mask[..., source_old_length:]],
        dim=-1,
    )


def heterogeneous_bottom_right_mask(
    layer: HeterogeneousSignificanceLayer,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return a causal, validity-aware mask aligned to one padded layer.

    The result has one head when no padding requires head-specific validity;
    otherwise it has one row per query head, with each KV-head validity row
    repeated across its contiguous GQA query-head group.
    """

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query/key must have [batch, heads, positions, dim] shape")
    batch, query_heads, query_length, _ = query.shape
    key_batch, kv_heads, key_length, _ = key.shape
    if batch != key_batch or query_heads % kv_heads:
        raise ValueError("query/key batch and GQA head grouping are incompatible")
    if layer.validity is None or layer.validity.shape != (batch, kv_heads, key_length):
        raise RuntimeError(
            f"layer {layer.layer_idx}: validity does not match live K shape "
            f"{(batch, kv_heads, key_length)}"
        )

    if attention_mask is None:
        base = bottom_right_causal_allowed(
            query_length,
            key_length,
            device=query.device,
        ).view(1, 1, query_length, key_length)
    else:
        base = _rank4_attention_mask(attention_mask, query_length=query_length)
        base = _remap_layer_slots(
            base,
            query_length=query_length,
            key_length=key_length,
        )
        if base.shape[0] not in (1, batch):
            raise ValueError(f"attention-mask batch {base.shape[0]} does not broadcast to {batch}")

    mask_heads = int(base.shape[1])
    groups = query_heads // kv_heads
    if mask_heads == kv_heads and query_heads != kv_heads:
        base = base.repeat_interleave(groups, dim=1)
        mask_heads = query_heads
    if mask_heads not in (1, query_heads):
        raise ValueError(
            f"attention-mask heads {mask_heads} must be 1, {kv_heads}, or {query_heads}"
        )

    if bool(layer.validity.all().item()):
        # Uniform allocations take this identity path: the original mask object
        # and head broadcast are preserved whenever no remapping was required.
        return base

    validity = layer.validity.repeat_interleave(groups, dim=1).unsqueeze(2)
    validity = validity.expand(batch, query_heads, query_length, key_length)
    if base.shape[0] == 1 and batch != 1:
        base = base.expand(batch, -1, -1, -1)
    if base.shape[1] == 1:
        base = base.expand(-1, query_heads, -1, -1)
    if base.dtype == torch.bool:
        return (base & validity).contiguous()
    if not base.is_floating_point():
        raise TypeError(f"attention mask must be boolean or floating point, got {base.dtype}")
    return base.masked_fill(~validity, torch.finfo(base.dtype).min).contiguous()


@contextmanager
def heterogeneous_transformers_sdpa(
    cache: HeterogeneousSignificanceCache,
    *,
    recording_window: int | None = None,
    force_repeat_kv: bool = True,
) -> Iterator[None]:
    """Adapt every Qwen SDPA layer and optionally record significance.

    Unlike the rectangular runtime's recorder, this context is required for
    *all* forwards after heterogeneous compression, including ordinary decode.
    It remaps the common model-level mask to each layer's physical width and
    masks every padded head slot before SDPA and before auxiliary scoring.
    """

    if recording_window is not None and recording_window <= 0:
        raise ValueError("recording_window must be positive when provided")
    if _ACTIVE_HETEROGENEOUS_CACHE.get() is not None:
        raise RuntimeError("nested heterogeneous SDPA contexts are not supported")
    from transformers.integrations import sdpa_attention as sdpa_module
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    with _PATCH_LOCK:
        original = ALL_ATTENTION_FUNCTIONS.get("sdpa")
        if original is None:
            raise RuntimeError("Transformers has no registered `sdpa` attention interface")
        original_gqa_selector = getattr(sdpa_module, "use_gqa_in_sdpa", None)

        def wrapped(
            module,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0,
            scaling=None,
            is_causal=None,
            position_bias=None,
            **kwargs,
        ):
            active = _ACTIVE_HETEROGENEOUS_CACHE.get()
            if active is not cache:
                return original(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=attention_mask,
                    dropout=dropout,
                    scaling=scaling,
                    is_causal=is_causal,
                    position_bias=position_bias,
                    **kwargs,
                )
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None or not 0 <= int(layer_idx) < len(cache.layers):
                raise RuntimeError(f"attention module has invalid layer_idx={layer_idx!r}")
            if float(dropout) != 0.0:
                raise RuntimeError("heterogeneous significance runtime is inference-only")
            if position_bias is not None:
                raise RuntimeError(
                    "the heterogeneous prototype targets Qwen3 attention without position bias"
                )
            layer = cache.layers[int(layer_idx)]
            adjusted_mask = heterogeneous_bottom_right_mask(
                layer,
                query,
                key,
                attention_mask,
            )
            contribution = None
            if cache.recording_enabled:
                contribution = post_rope_gqa_significance(
                    query,
                    key,
                    window=cache.recording_window,
                    scaling=scaling,
                    attention_mask=adjusted_mask,
                    position_bias=position_bias,
                )
            result = original(
                module,
                query,
                key,
                value,
                attention_mask=adjusted_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=False,
                position_bias=position_bias,
                **kwargs,
            )
            if contribution is not None:
                cache.record_attention(int(layer_idx), contribution)
            return result

        token = _ACTIVE_HETEROGENEOUS_CACHE.set(cache)
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
        if force_repeat_kv and original_gqa_selector is not None:
            # A head-specific validity mask necessarily disables fused GQA.
            # Explicit repeat_kv is exact and matches the existing audited CPU
            # path, although it has a real transient-memory cost.
            sdpa_module.use_gqa_in_sdpa = lambda attention_mask, key, value: False
        recording = (
            cache.recording(window=recording_window)
            if recording_window is not None
            else nullcontext()
        )
        try:
            with cache.attention_adapter():
                with recording:
                    yield
        finally:
            registry_changed = ALL_ATTENTION_FUNCTIONS.get("sdpa") is not wrapped
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original
            if force_repeat_kv and original_gqa_selector is not None:
                sdpa_module.use_gqa_in_sdpa = original_gqa_selector
            _ACTIVE_HETEROGENEOUS_CACHE.reset(token)
            if registry_changed:
                raise RuntimeError(
                    "Transformers SDPA registry changed while heterogeneous adapter was active"
                )
