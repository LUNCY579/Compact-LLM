"""Closed-loop Qwen/Transformers adapter for fixed significance memory slots.

This module is intentionally separate from the ordinary rectangular cache,
heterogeneous-budget runtime, and attention-prior experiments.  Memory slots
are prepended only inside the attention call; they are never appended to the
ordinary cache or its aligned S/C state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

from .cache import CompressionReport, SignificanceCache
from .masks import bottom_right_causal_allowed, bottom_right_causal_mask
from .memory import MemorySlotState, MemoryWriteReport, SignificanceMemory
from .policy import SignificancePolicy, select_keep_indices
from .scoring import post_rope_gqa_significance
from .transformers_runtime import (
    _ACTIVE_CACHE,
    _PATCH_LOCK,
    ChunkedPrefillResult,
    _model_dtype,
    _validate_model_kwargs,
    chunked_prefill,
)


@dataclass
class MemoryAttentionAccounting:
    """Non-persistent diagnostics proving when slots entered attention."""

    attention_calls: int = 0
    calls_with_valid_slots: int = 0
    max_valid_slots_per_batch_head: int = 0
    augmented_key_lengths: list[int] = field(default_factory=list)

    def observe(self, slots: MemorySlotState, *, key_length: int) -> None:
        self.attention_calls += 1
        if slots.valid.numel():
            per_head = slots.valid.sum(dim=-1)
            maximum = int(per_head.max().item())
        else:
            maximum = 0
        if maximum > 0:
            self.calls_with_valid_slots += 1
        self.max_valid_slots_per_batch_head = max(
            self.max_valid_slots_per_batch_head,
            maximum,
        )
        self.augmented_key_lengths.append(int(key_length))


@dataclass(frozen=True)
class MemoryCompressionResult:
    compression: CompressionReport
    writes: tuple[MemoryWriteReport, ...]


@dataclass
class MemoryChunkedPrefillResult:
    output: Any
    compression_reports: list[CompressionReport]
    memory_write_reports: list[tuple[MemoryWriteReport, ...]]
    attention_accounting: MemoryAttentionAccounting
    logits: torch.Tensor | None = None


@dataclass
class MemoryDecodeAppendResult:
    output: Any
    attention_accounting: MemoryAttentionAccounting


def _expand_mask_heads(
    mask: torch.Tensor,
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_length: int,
    key_length: int,
) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.view(1, 1, *mask.shape)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"attention mask must have rank 2, 3, or 4; got {mask.ndim}")
    if mask.shape[0] not in (1, batch):
        raise ValueError(f"attention mask batch {mask.shape[0]} does not broadcast to {batch}")
    if mask.shape[-2] < query_length or mask.shape[-1] != key_length:
        raise ValueError(
            f"attention mask tail {tuple(mask.shape[-2:])} is incompatible with "
            f"query/key lengths {(query_length, key_length)}"
        )
    mask = mask[..., -query_length:, :].expand(batch, -1, -1, -1)
    mask_heads = int(mask.shape[1])
    if mask_heads == 1:
        return mask.expand(batch, query_heads, query_length, key_length)
    if mask_heads == kv_heads:
        groups = query_heads // kv_heads
        return mask.repeat_interleave(groups, dim=1)
    if mask_heads == query_heads:
        return mask
    raise ValueError(
        f"attention mask head dimension {mask_heads} must be 1, {kv_heads}, or {query_heads}"
    )


def augment_attention_with_memory(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    slots: MemorySlotState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepend fixed memory slots and a per-query-head validity mask.

    Invalid physical slots remain present but are completely masked.  Slots are
    historical summaries, so every valid slot is visible to every query row;
    the ordinary cache tail retains its original bottom-right causal mask.
    """

    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query/K/V must have [batch, heads, positions, dim] shapes")
    batch, query_heads, query_length, head_dim = query.shape
    key_batch, kv_heads, key_length, key_dim = key.shape
    if batch != key_batch or head_dim != key_dim:
        raise ValueError("query and cache K/V batch/head dimensions are incompatible")
    if query_heads % kv_heads:
        raise ValueError("query-head count must be divisible by KV-head count")
    expected_slots = (batch, kv_heads, slots.valid.shape[-1], head_dim)
    if tuple(slots.keys.shape) != expected_slots or tuple(slots.values.shape) != expected_slots:
        raise ValueError(
            f"memory K/V must have shape {expected_slots}; got "
            f"{tuple(slots.keys.shape)}/{tuple(slots.values.shape)}"
        )
    if tuple(slots.valid.shape) != expected_slots[:-1]:
        raise ValueError("memory validity shape is not aligned with memory K/V")
    if slots.keys.device != key.device or slots.values.device != value.device:
        raise ValueError("memory and ordinary K/V must share a device")
    if slots.keys.dtype != key.dtype or slots.values.dtype != value.dtype:
        raise TypeError("memory and ordinary K/V must share a dtype")

    memory_length = int(slots.valid.shape[-1])
    if memory_length == 0:
        return key, value, attention_mask

    augmented_key = torch.cat((slots.keys, key), dim=2)
    augmented_value = torch.cat((slots.values, value), dim=2)
    groups = query_heads // kv_heads
    memory_allowed = slots.valid.repeat_interleave(groups, dim=1)
    memory_allowed = memory_allowed.unsqueeze(2).expand(batch, query_heads, query_length, -1)

    if attention_mask is None:
        ordinary_allowed = bottom_right_causal_allowed(
            query_length,
            key_length,
            device=query.device,
        ).view(1, 1, query_length, key_length)
        ordinary_allowed = ordinary_allowed.expand(batch, query_heads, -1, -1)
        augmented_mask: torch.Tensor = torch.cat((memory_allowed, ordinary_allowed), dim=-1)
    else:
        ordinary_mask = _expand_mask_heads(
            attention_mask,
            batch=batch,
            query_heads=query_heads,
            kv_heads=kv_heads,
            query_length=query_length,
            key_length=key_length,
        )
        if ordinary_mask.dtype == torch.bool:
            augmented_mask = torch.cat((memory_allowed, ordinary_mask), dim=-1)
        elif ordinary_mask.is_floating_point():
            memory_mask = torch.zeros(
                memory_allowed.shape,
                device=ordinary_mask.device,
                dtype=ordinary_mask.dtype,
            )
            memory_mask.masked_fill_(~memory_allowed, torch.finfo(ordinary_mask.dtype).min)
            augmented_mask = torch.cat((memory_mask, ordinary_mask), dim=-1)
        else:
            raise TypeError(
                f"attention mask must be boolean or floating point, got {ordinary_mask.dtype}"
            )
    return augmented_key, augmented_value, augmented_mask


_ACTIVE_MEMORY: ContextVar[
    tuple[SignificanceCache, SignificanceMemory, MemoryAttentionAccounting] | None
] = ContextVar("rineya_active_significance_memory", default=None)


@contextmanager
def transformers_sdpa_with_memory(
    cache: SignificanceCache,
    memory: SignificanceMemory,
    *,
    record_window: int | None,
    accounting: MemoryAttentionAccounting | None = None,
    force_repeat_kv: bool = True,
) -> Iterator[MemoryAttentionAccounting]:
    """Expose fixed memory slots to one model forward and optionally score S/C."""

    if record_window is not None and record_window <= 0:
        raise ValueError("record_window must be positive when recording is enabled")
    if _ACTIVE_CACHE.get() is not None or _ACTIVE_MEMORY.get() is not None:
        raise RuntimeError("nested Transformers significance adapters are not supported")
    if memory.num_layers != len(cache.layers):
        raise ValueError("memory and cache layer counts differ")
    memory.assert_valid()
    counters = accounting if accounting is not None else MemoryAttentionAccounting()

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
            active = _ACTIVE_MEMORY.get()
            if active is None or active[0] is not cache or active[1] is not memory:
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
            layer_index = int(layer_idx)
            if float(dropout) != 0.0:
                raise RuntimeError("significance memory is inference-only and requires dropout 0")
            if position_bias is not None:
                raise RuntimeError(
                    "the bounded prototype supports post-RoPE Qwen attention without a separate position bias"
                )
            query_start = cache.layers[layer_index].true_length - int(query.shape[-2])
            memory.assert_visible(layer_index, query_start_position=query_start)
            slots = memory.retrieve_slots(layer_index)
            augmented_key, augmented_value, augmented_mask = augment_attention_with_memory(
                query,
                key,
                value,
                attention_mask,
                slots,
            )
            counters.observe(slots, key_length=int(augmented_key.shape[-2]))
            contribution = None
            if record_window is not None:
                contribution = post_rope_gqa_significance(
                    query,
                    augmented_key,
                    window=record_window,
                    scaling=scaling,
                    attention_mask=augmented_mask,
                )[..., memory.num_slots :]
            result = original(
                module,
                query,
                augmented_key,
                augmented_value,
                attention_mask=augmented_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=False,
                position_bias=None,
                **kwargs,
            )
            if contribution is not None:
                cache.record_attention(layer_index, contribution)
            return result

        cache_token = _ACTIVE_CACHE.set(cache)
        memory_token = _ACTIVE_MEMORY.set((cache, memory, counters))
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
        if force_repeat_kv and original_gqa_selector is not None:
            sdpa_module.use_gqa_in_sdpa = lambda attention_mask, key, value: False
        recording_context = (
            cache.recording(window=int(record_window))
            if record_window is not None
            else nullcontext()
        )
        try:
            with recording_context:
                yield counters
        finally:
            registry_changed = ALL_ATTENTION_FUNCTIONS.get("sdpa") is not wrapped
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original
            if force_repeat_kv and original_gqa_selector is not None:
                sdpa_module.use_gqa_in_sdpa = original_gqa_selector
            _ACTIVE_MEMORY.reset(memory_token)
            _ACTIVE_CACHE.reset(cache_token)
            if registry_changed:
                raise RuntimeError("Transformers SDPA registry changed during memory attention")


def _keep_indices(layer: Any, policy: SignificancePolicy) -> torch.Tensor:
    layer.assert_aligned()
    stored = layer.stored_length()
    if layer.state is None:
        raise RuntimeError(f"layer {layer.layer_idx}: no significance state")
    if stored <= policy.budget:
        return (
            torch.arange(stored, device=layer.keys.device, dtype=torch.long)
            .view(1, 1, -1)
            .expand(layer.keys.shape[0], layer.keys.shape[1], -1)
            .clone()
        )
    if (layer.state.counts <= 0).any():
        raise RuntimeError(
            f"layer {layer.layer_idx}: memory compression cannot consume unscored entries"
        )
    ranking = (
        layer.state.positions.to(dtype=torch.float32)
        if policy.ranking_mode == "position"
        else layer.state.ranked(policy.count_exponent)
    )
    return select_keep_indices(
        ranking,
        policy,
        positions=layer.state.positions,
        entry_ids=layer.state.entry_ids,
    )


def _gather_evicted(tensor: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    batch, heads, entries = tensor.shape[:3]
    keep_mask = torch.zeros((batch, heads, entries), dtype=torch.bool, device=tensor.device)
    keep_mask.scatter_(2, keep, True)
    indices = (
        torch.arange(entries, device=tensor.device, dtype=torch.long)
        .view(1, 1, entries)
        .expand(batch, heads, -1)
    )
    evicted_count = entries - int(keep.shape[-1])
    evicted = indices.masked_select(~keep_mask).view(batch, heads, evicted_count)
    if tensor.ndim == 4:
        return tensor.gather(2, evicted.unsqueeze(-1).expand(-1, -1, -1, tensor.shape[-1]))
    return tensor.gather(2, evicted)


def compress_cache_with_memory(
    cache: SignificanceCache,
    policy: SignificancePolicy,
    memory: SignificanceMemory,
) -> MemoryCompressionResult:
    """Apply ordinary selection and write only its evicted complement to memory."""

    cache.assert_forward_complete()
    if memory.num_layers != len(cache.layers):
        raise ValueError("memory and cache layer counts differ")
    if memory.num_slots == 0:
        # This direct delegation is the zero-slot identity contract.
        return MemoryCompressionResult(cache.compress(policy), ())
    initialized = [layer for layer in cache.layers if layer.is_initialized]
    if len(initialized) != len(cache.layers):
        raise RuntimeError("memory compression requires every model layer to be initialized")

    prepared: list[tuple[Any, torch.Tensor, int]] = []
    writes: list[MemoryWriteReport] = []
    memory_snapshot = memory.snapshot()
    before = sum(layer.stored_length() for layer in cache.layers)
    try:
        for layer_index, layer in enumerate(cache.layers):
            old_length = layer.stored_length()
            keep = _keep_indices(layer, policy)
            prepared.append((layer, keep, old_length))
            if int(keep.shape[-1]) == old_length:
                continue
            state = layer.state
            writes.append(
                memory.compress_evicted(
                    layer_index=layer_index,
                    keys=_gather_evicted(layer.keys, keep),
                    values=_gather_evicted(layer.values, keep),
                    scores=_gather_evicted(state.scores, keep),
                    counts=_gather_evicted(state.counts, keep),
                    positions=_gather_evicted(state.positions, keep),
                    entry_ids=_gather_evicted(state.entry_ids, keep),
                    boundary_position=layer.true_length,
                )
            )
    except Exception:
        memory.restore(memory_snapshot)
        raise

    compressed_layers = 0
    evictions = 0
    for layer, keep, old_length in prepared:
        if int(keep.shape[-1]) == old_length:
            continue
        gather = keep.unsqueeze(-1).expand(-1, -1, -1, layer.keys.shape[-1])
        layer.keys = layer.keys.gather(2, gather).contiguous()
        layer.values = layer.values.gather(2, gather).contiguous()
        layer.state.gather(keep, cache_length=old_length)
        layer.evictions += 1
        layer.assert_aligned()
        compressed_layers += 1
        evictions += old_length - layer.stored_length()
    cache.assert_aligned()
    memory.assert_valid()
    after = sum(layer.stored_length() for layer in cache.layers)
    return MemoryCompressionResult(
        CompressionReport(compressed_layers, before, after, evictions),
        tuple(writes),
    )


@torch.no_grad()
def score_with_query_memory(
    model: Any,
    cache: SignificanceCache,
    memory: SignificanceMemory,
    query_ids: torch.Tensor,
    *,
    accounting: MemoryAttentionAccounting | None = None,
    verify_rollback: bool = True,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> None:
    """Query-condition ordinary S/C while proving memory remains read-only."""

    cache.assert_forward_complete()
    if query_ids.ndim != 2 or query_ids.shape[-1] <= 0:
        raise ValueError("query_ids must have shape [batch, q>0]")
    first_layer = next(layer for layer in cache.layers if layer.is_initialized)
    cache_batch = int(first_layer.keys.shape[0])
    if query_ids.shape[0] == 1 and cache_batch > 1:
        query_ids = query_ids.expand(cache_batch, -1)
    elif query_ids.shape[0] != cache_batch:
        raise ValueError("query batch does not match cache batch")
    kwargs = _validate_model_kwargs(model_kwargs)
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = int(logits_to_keep)

    snapshot = cache.snapshot_structure(exact_tensors=verify_rollback)
    memory_before = memory.state_sha256()
    query_length = int(query_ids.shape[-1])
    positions = (
        torch.arange(
            cache.true_length,
            cache.true_length + query_length,
            device=query_ids.device,
            dtype=torch.long,
        )
        .view(1, -1)
        .expand(cache_batch, -1)
    )
    mask = bottom_right_causal_mask(
        query_length,
        cache.stored_length() + query_length,
        device=query_ids.device,
        dtype=_model_dtype(model),
    )
    before_events = cache.scoring_events()
    try:
        cache.prepare_append(positions)
        with transformers_sdpa_with_memory(
            cache,
            memory,
            record_window=query_length,
            accounting=accounting,
            force_repeat_kv=force_repeat_kv,
        ):
            model(
                query_ids,
                past_key_values=cache,
                position_ids=positions,
                attention_mask={"full_attention": mask},
                use_cache=True,
                **kwargs,
            )
        cache.assert_forward_complete()
        cache.require_one_score_event(before_events)
    except Exception:
        cache.rollback_structure(snapshot, keep_score_updates=False, verify_exact=verify_rollback)
        raise
    cache.rollback_structure(snapshot, keep_score_updates=True, verify_exact=verify_rollback)
    if memory.state_sha256() != memory_before:
        raise RuntimeError("query probe mutated persistent significance memory")


@torch.no_grad()
def memory_decode_append(
    model: Any,
    token_ids: torch.Tensor,
    cache: SignificanceCache,
    memory: SignificanceMemory,
    *,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> MemoryDecodeAppendResult:
    """Append decode tokens while exposing memory read-only to closed-loop attention."""

    if token_ids.ndim != 2 or token_ids.shape[-1] <= 0:
        raise ValueError("token_ids must have shape [batch, q>0]")
    if memory.num_slots == 0:
        baseline = chunked_prefill(
            model,
            token_ids,
            cache,
            chunk_size=int(token_ids.shape[-1]),
            policy=None,
            collect_logits=False,
            force_repeat_kv=force_repeat_kv,
            logits_to_keep=logits_to_keep,
            model_kwargs=model_kwargs,
        )
        return MemoryDecodeAppendResult(
            output=baseline.output,
            attention_accounting=MemoryAttentionAccounting(),
        )
    if memory.batch_size != int(token_ids.shape[0]):
        raise ValueError("memory batch size does not match decode batch")
    if memory.num_layers != len(cache.layers):
        raise ValueError("memory and cache layer counts differ")
    cache.assert_forward_complete()
    before_true = cache.true_length
    before_stored = cache.stored_length()
    memory_before = memory.state_sha256()
    query_length = int(token_ids.shape[-1])
    positions = (
        torch.arange(
            before_true,
            before_true + query_length,
            device=token_ids.device,
            dtype=torch.long,
        )
        .view(1, -1)
        .expand(token_ids.shape[0], -1)
    )
    mask = bottom_right_causal_mask(
        query_length,
        before_stored + query_length,
        device=token_ids.device,
        dtype=_model_dtype(model),
    )
    kwargs = _validate_model_kwargs(model_kwargs)
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = int(logits_to_keep)
    accounting = MemoryAttentionAccounting()
    cache.prepare_append(positions)
    with transformers_sdpa_with_memory(
        cache,
        memory,
        record_window=None,
        accounting=accounting,
        force_repeat_kv=force_repeat_kv,
    ):
        output = model(
            token_ids,
            past_key_values=cache,
            position_ids=positions,
            attention_mask={"full_attention": mask},
            use_cache=True,
            **kwargs,
        )
    cache.assert_forward_complete()
    if cache.true_length != before_true + query_length:
        raise RuntimeError("memory decode append changed semantic length incorrectly")
    if cache.stored_length() != before_stored + query_length:
        raise RuntimeError("memory decode append changed ordinary resident length incorrectly")
    if memory.state_sha256() != memory_before:
        raise RuntimeError("decode append mutated persistent significance memory")
    return MemoryDecodeAppendResult(output=output, attention_accounting=accounting)


@torch.no_grad()
def chunked_prefill_with_memory(
    model: Any,
    input_ids: torch.Tensor,
    cache: SignificanceCache,
    memory: SignificanceMemory,
    *,
    chunk_size: int,
    policy: SignificancePolicy,
    query_ids: torch.Tensor | None = None,
    observation_window: int = 16,
    verify_query_rollback: bool = True,
    collect_logits: bool = False,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> MemoryChunkedPrefillResult:
    """Run closed-loop chunked prefill with fixed memory available after writes."""

    if memory.num_slots == 0:
        baseline: ChunkedPrefillResult = chunked_prefill(
            model,
            input_ids,
            cache,
            chunk_size=chunk_size,
            policy=policy,
            query_ids=query_ids,
            observation_window=observation_window,
            verify_query_rollback=verify_query_rollback,
            collect_logits=collect_logits,
            force_repeat_kv=force_repeat_kv,
            logits_to_keep=logits_to_keep,
            model_kwargs=model_kwargs,
        )
        return MemoryChunkedPrefillResult(
            output=baseline.output,
            compression_reports=baseline.compression_reports,
            memory_write_reports=[],
            attention_accounting=MemoryAttentionAccounting(),
            logits=baseline.logits,
        )
    if input_ids.ndim != 2 or input_ids.shape[-1] <= 0:
        raise ValueError("input_ids must have shape [batch, n>0]")
    if chunk_size <= 0 or observation_window <= 0:
        raise ValueError("chunk_size and observation_window must be positive")
    if memory.batch_size != int(input_ids.shape[0]):
        raise ValueError("memory batch size does not match input batch")
    if memory.num_layers != len(cache.layers):
        raise ValueError("memory and cache layer counts differ")
    kwargs_base = _validate_model_kwargs(model_kwargs)
    total = int(input_ids.shape[-1])
    n_chunks = (total + chunk_size - 1) // chunk_size
    output: Any | None = None
    reports: list[CompressionReport] = []
    write_reports: list[tuple[MemoryWriteReport, ...]] = []
    collected: list[torch.Tensor] = []
    accounting = MemoryAttentionAccounting()

    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        piece = input_ids[:, start : start + chunk_size]
        query_length = int(piece.shape[-1])
        positions = (
            torch.arange(
                cache.true_length,
                cache.true_length + query_length,
                device=input_ids.device,
                dtype=torch.long,
            )
            .view(1, -1)
            .expand(input_ids.shape[0], -1)
        )
        mask = bottom_right_causal_mask(
            query_length,
            cache.stored_length() + query_length,
            device=input_ids.device,
            dtype=_model_dtype(model),
        )
        kwargs = dict(kwargs_base)
        effective_logits = 0 if collect_logits else logits_to_keep
        if effective_logits is not None:
            kwargs["logits_to_keep"] = int(effective_logits)
        cache.prepare_append(positions)
        before_events = cache.scoring_events()
        record_window = observation_window if query_ids is None else None
        with transformers_sdpa_with_memory(
            cache,
            memory,
            record_window=record_window,
            accounting=accounting,
            force_repeat_kv=force_repeat_kv,
        ):
            output = model(
                piece,
                past_key_values=cache,
                position_ids=positions,
                attention_mask={"full_attention": mask},
                use_cache=True,
                **kwargs,
            )
        cache.assert_forward_complete()
        if record_window is not None:
            cache.require_one_score_event(before_events)
        if collect_logits:
            if not hasattr(output, "logits"):
                raise RuntimeError("collect_logits requires causal-LM logits")
            collected.append(output.logits.float())

        if chunk_index < n_chunks - 1:
            if query_ids is not None:
                score_with_query_memory(
                    model,
                    cache,
                    memory,
                    query_ids,
                    accounting=accounting,
                    verify_rollback=verify_query_rollback,
                    force_repeat_kv=force_repeat_kv,
                    logits_to_keep=1,
                    model_kwargs=model_kwargs,
                )
            compressed = compress_cache_with_memory(cache, policy, memory)
            reports.append(compressed.compression)
            write_reports.append(compressed.writes)

    if output is None:
        raise RuntimeError("memory prefill consumed no input")
    cache.assert_aligned()
    memory.assert_valid()
    return MemoryChunkedPrefillResult(
        output=output,
        compression_reports=reports,
        memory_write_reports=write_reports,
        attention_accounting=accounting,
        logits=torch.cat(collected, dim=1) if collected else None,
    )
