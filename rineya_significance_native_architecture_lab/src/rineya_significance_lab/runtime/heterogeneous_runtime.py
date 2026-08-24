"""Closed-loop Transformers execution for heterogeneous Significance budgets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ..core.allocation import AllocationStrategy
from .heterogeneous_attention import heterogeneous_transformers_sdpa
from .heterogeneous_budget import LiveBudgetDecision, allocate_live_budget
from .heterogeneous_cache import (
    HeterogeneousCompressionReport,
    HeterogeneousSignificanceCache,
)
from .masks import bottom_right_causal_mask
from .policy import SignificancePolicy
from .transformers_runtime import _model_dtype, _validate_model_kwargs

AllocationCallback = Callable[
    [HeterogeneousSignificanceCache, SignificancePolicy, int],
    torch.Tensor,
]


@dataclass
class HeterogeneousPrefillResult:
    output: Any
    compression_reports: list[HeterogeneousCompressionReport]
    allocation_decisions: list[LiveBudgetDecision | None]
    logits: torch.Tensor | None = None


@dataclass
class HeterogeneousDecodeResult:
    output: Any
    logits: torch.Tensor


def _append_positions(
    cache: HeterogeneousSignificanceCache,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    query_length = int(input_ids.shape[-1])
    return torch.arange(
        cache.true_length,
        cache.true_length + query_length,
        device=input_ids.device,
        dtype=torch.long,
    ).view(1, -1).expand(input_ids.shape[0], -1)


def _max_layer_mask(
    model,
    cache: HeterogeneousSignificanceCache,
    query_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return bottom_right_causal_mask(
        query_length,
        cache.max_stored_length() + query_length,
        device=device,
        dtype=_model_dtype(model),
    )


@torch.no_grad()
def heterogeneous_score_with_query(
    model,
    cache: HeterogeneousSignificanceCache,
    query_ids: torch.Tensor,
    *,
    verify_rollback: bool = True,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> None:
    """Score with a known query and exactly rollback layer-specific suffixes."""

    cache.assert_forward_complete()
    if query_ids.ndim != 2 or query_ids.shape[-1] <= 0:
        raise ValueError(f"query_ids must have shape [batch, q>0], got {tuple(query_ids.shape)}")
    if cache.true_length <= 0:
        raise RuntimeError("cannot query-condition an empty cache")
    first = next(layer for layer in cache.layers if layer.is_initialized)
    assert first.keys is not None
    cache_batch = int(first.keys.shape[0])
    if query_ids.shape[0] == 1 and cache_batch > 1:
        query_ids = query_ids.expand(cache_batch, -1)
    elif query_ids.shape[0] != cache_batch:
        raise ValueError(f"query batch {query_ids.shape[0]} does not match cache batch {cache_batch}")
    kwargs = _validate_model_kwargs(model_kwargs)
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = int(logits_to_keep)

    snapshot = cache.snapshot_structure(exact_tensors=verify_rollback)
    positions = _append_positions(cache, query_ids)
    query_length = int(query_ids.shape[-1])
    mask = _max_layer_mask(model, cache, query_length, device=query_ids.device)
    before_events = cache.scoring_events()
    try:
        cache.prepare_append(positions)
        with heterogeneous_transformers_sdpa(
            cache,
            recording_window=query_length,
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
        cache.rollback_structure(
            snapshot,
            keep_score_updates=False,
            verify_exact=verify_rollback,
        )
        raise
    cache.rollback_structure(
        snapshot,
        keep_score_updates=True,
        verify_exact=verify_rollback,
    )


@torch.no_grad()
def heterogeneous_chunked_prefill(
    model,
    input_ids: torch.Tensor,
    cache: HeterogeneousSignificanceCache,
    *,
    chunk_size: int,
    policy: SignificancePolicy,
    global_budget: int,
    allocation_strategy: AllocationStrategy = "uniform",
    minimum_per_unit: int = 1,
    maximum_per_unit: int | None = None,
    allocation_exponent: float = 1.0,
    allocation_seed: int = 0,
    allocation_callback: AllocationCallback | None = None,
    query_ids: torch.Tensor | None = None,
    observation_window: int = 16,
    verify_query_rollback: bool = True,
    collect_logits: bool = False,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> HeterogeneousPrefillResult:
    """Run closed-loop prefill with one conserved logical budget per boundary.

    If ``allocation_callback`` is absent, diagnostics are computed from the live
    S/C tensors after each scoring pass and the requested adaptive strategy is
    applied. A callback may instead supply a deliberate allocation matrix for
    identity and mechanism tests; its sum is still checked against
    ``global_budget`` by the cache.
    """

    if input_ids.ndim != 2 or input_ids.shape[-1] <= 0:
        raise ValueError(f"input_ids must have shape [batch, n>0], got {tuple(input_ids.shape)}")
    if chunk_size <= 0 or observation_window <= 0:
        raise ValueError("chunk_size and observation_window must be positive")
    if global_budget < 0:
        raise ValueError("global_budget must be non-negative")
    kwargs_base = _validate_model_kwargs(model_kwargs)
    total = int(input_ids.shape[-1])
    n_chunks = (total + chunk_size - 1) // chunk_size
    output = None
    reports: list[HeterogeneousCompressionReport] = []
    decisions: list[LiveBudgetDecision | None] = []
    collected: list[torch.Tensor] = []

    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        piece = input_ids[:, start : start + chunk_size]
        query_length = int(piece.shape[-1])
        positions = _append_positions(cache, piece)
        mask = _max_layer_mask(model, cache, query_length, device=input_ids.device)
        kwargs = dict(kwargs_base)
        effective_logits = 0 if collect_logits else logits_to_keep
        if effective_logits is not None:
            kwargs["logits_to_keep"] = int(effective_logits)
        cache.prepare_append(positions)
        before_events = cache.scoring_events()
        recording_window = observation_window if query_ids is None else None
        with heterogeneous_transformers_sdpa(
            cache,
            recording_window=recording_window,
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
        if query_ids is None:
            cache.require_one_score_event(before_events)

        if collect_logits:
            if not hasattr(output, "logits"):
                raise RuntimeError("collect_logits requires a causal-LM output with logits")
            collected.append(output.logits.float())

        if chunk_index < n_chunks - 1:
            if query_ids is not None:
                heterogeneous_score_with_query(
                    model,
                    cache,
                    query_ids,
                    verify_rollback=verify_query_rollback,
                    force_repeat_kv=force_repeat_kv,
                    logits_to_keep=1,
                    model_kwargs=model_kwargs,
                )
            if allocation_callback is None:
                decision = allocate_live_budget(
                    cache,
                    strategy=allocation_strategy,
                    global_budget=global_budget,
                    minimum_per_unit=minimum_per_unit,
                    maximum_per_unit=maximum_per_unit,
                    exponent=allocation_exponent,
                    seed=allocation_seed + chunk_index,
                )
                first = cache.layers[0]
                assert first.keys is not None
                allocations = decision.torch_allocations(device=first.keys.device)
                decisions.append(decision)
            else:
                allocations = allocation_callback(cache, policy, chunk_index)
                decisions.append(None)
            reports.append(
                cache.compress(
                    policy,
                    allocations,
                    global_budget=global_budget,
                )
            )

    logits = torch.cat(collected, dim=1) if collected else None
    return HeterogeneousPrefillResult(
        output=output,
        compression_reports=reports,
        allocation_decisions=decisions,
        logits=logits,
    )


@torch.no_grad()
def heterogeneous_decode_append(
    model,
    input_ids: torch.Tensor,
    cache: HeterogeneousSignificanceCache,
    *,
    recording_window: int | None = None,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 0,
    model_kwargs: dict[str, Any] | None = None,
) -> HeterogeneousDecodeResult:
    """Append one or more decode tokens through layer/head validity masks."""

    cache.assert_forward_complete()
    if input_ids.ndim != 2 or input_ids.shape[-1] <= 0:
        raise ValueError(f"input_ids must have shape [batch, q>0], got {tuple(input_ids.shape)}")
    kwargs = _validate_model_kwargs(model_kwargs)
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = int(logits_to_keep)
    positions = _append_positions(cache, input_ids)
    query_length = int(input_ids.shape[-1])
    mask = _max_layer_mask(model, cache, query_length, device=input_ids.device)
    before_events = cache.scoring_events()
    cache.prepare_append(positions)
    with heterogeneous_transformers_sdpa(
        cache,
        recording_window=recording_window,
        force_repeat_kv=force_repeat_kv,
    ):
        output = model(
            input_ids,
            past_key_values=cache,
            position_ids=positions,
            attention_mask={"full_attention": mask},
            use_cache=True,
            **kwargs,
        )
    cache.assert_forward_complete()
    if recording_window is not None:
        cache.require_one_score_event(before_events)
    if not hasattr(output, "logits"):
        raise RuntimeError("decode requires a causal-LM output with logits")
    return HeterogeneousDecodeResult(output=output, logits=output.logits.float())
