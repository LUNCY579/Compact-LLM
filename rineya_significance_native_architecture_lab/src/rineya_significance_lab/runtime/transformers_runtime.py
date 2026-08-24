"""Inference-only Transformers 5.14/5.15 integration for Cornerstone 1.

The integration is process-local: it temporarily wraps the registered SDPA
attention function and restores it afterward. No site-package or model file is
modified.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

import torch

from .attention_prior import (
    AttentionPriorTrace,
    SignificanceAttentionPrior,
    merge_attention_prior_mask,
)
from .cache import CompressionReport, SignificanceCache
from .masks import bottom_right_causal_mask
from .policy import SignificancePolicy
from .scoring import post_rope_gqa_significance_diagnostics

_ACTIVE_CACHE: ContextVar[SignificanceCache | None] = ContextVar("rineya_active_significance_cache", default=None)
_PATCH_LOCK = threading.RLock()

CompressionCallback = Callable[
    [SignificanceCache, SignificancePolicy, int],
    CompressionReport,
]


def _model_dtype(model) -> torch.dtype:
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration as exc:
        raise ValueError("model has no parameters from which to infer attention-mask dtype") from exc
    if not dtype.is_floating_point:
        raise TypeError(f"model parameter dtype must be floating point, got {dtype}")
    return dtype


@contextmanager
def record_transformers_sdpa(
    cache: SignificanceCache,
    *,
    window: int,
    force_repeat_kv: bool = True,
    attention_prior: SignificanceAttentionPrior | None = None,
    prior_trace: AttentionPriorTrace | None = None,
    record_scores: bool = True,
    trace_role: str = "scoring_forward",
) -> Iterator[None]:
    """Record post-RoPE SDPA significance into ``cache`` for one forward.

    This targets the `AttentionInterface` used by Transformers 5.14+ Qwen3.
    Replacing the registry entry is global, so a lock intentionally serializes
    the short recording context. A ``ContextVar`` prevents unrelated forwards in
    the same process from writing into this cache.
    """

    if _ACTIVE_CACHE.get() is not None:
        raise RuntimeError("nested Transformers significance recorders are not supported")
    if prior_trace is not None and attention_prior is None:
        raise ValueError("prior_trace requires an attention_prior")
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
            active = _ACTIVE_CACHE.get()
            contribution = None
            prior_result = None
            layer_idx = getattr(module, "layer_idx", None)
            effective_attention_mask = attention_mask
            if active is cache:
                if layer_idx is None or not 0 <= int(layer_idx) < len(active.layers):
                    raise RuntimeError(f"attention module has invalid layer_idx={layer_idx!r}")
                if float(dropout) != 0.0:
                    raise RuntimeError("significance attention is inference-only and requires attention dropout 0")
                layer = active.layers[int(layer_idx)]
                if layer.state is None:
                    raise RuntimeError(f"layer {layer_idx}: cache state was not initialized before attention")
                if attention_prior is not None:
                    causal = (
                        bool(is_causal)
                        if is_causal is not None
                        else bool(getattr(module, "is_causal", True))
                    )
                    prior_result = attention_prior.bias(
                        layer.state,
                        layer_idx=int(layer_idx),
                        query_heads=int(query.shape[1]),
                        query_length=int(query.shape[2]),
                        key_length=int(key.shape[2]),
                        attention_mask=attention_mask,
                        causal=causal,
                        forward_role=trace_role,
                    )
                    # A literal bypass is required for C0: merely adding a zero
                    # tensor can change the SDPA kernel/mask path.
                    if prior_result.diagnostics.nonzero_bias_applied:
                        effective_attention_mask = merge_attention_prior_mask(
                            attention_mask,
                            prior_result.bias,
                            kv_heads=int(key.shape[1]),
                            causal=causal,
                            dtype=query.dtype,
                        )
                if active.recording_enabled or prior_trace is not None:
                    scored = post_rope_gqa_significance_diagnostics(
                        query,
                        key,
                        window=active.recording_window if active.recording_enabled else window,
                        scaling=scaling,
                        attention_mask=effective_attention_mask,
                        position_bias=position_bias,
                    )
                    if active.recording_enabled:
                        contribution = scored.contribution
                    if prior_trace is not None and prior_result is not None:
                        prior_trace.record(
                            replace(
                                prior_result.diagnostics,
                                attention_entropy_nats_mean=scored.attention_entropy_nats_mean,
                                attention_entropy_normalized_mean=(
                                    scored.attention_entropy_normalized_mean
                                ),
                            )
                        )
            result = original(
                module,
                query,
                key,
                value,
                attention_mask=effective_attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=is_causal,
                position_bias=position_bias,
                **kwargs,
            )
            if contribution is not None:
                cache.record_attention(int(layer_idx), contribution)
            return result

        token = _ACTIVE_CACHE.set(cache)
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
        if force_repeat_kv and original_gqa_selector is not None:
            # Avoid the audited enable_gqa math-kernel fallback on consumer GPUs.
            sdpa_module.use_gqa_in_sdpa = lambda attention_mask, key, value: False
        try:
            if record_scores:
                with cache.recording(window=window):
                    yield
            else:
                yield
        finally:
            registry_changed = ALL_ATTENTION_FUNCTIONS.get("sdpa") is not wrapped
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original
            if force_repeat_kv and original_gqa_selector is not None:
                sdpa_module.use_gqa_in_sdpa = original_gqa_selector
            _ACTIVE_CACHE.reset(token)
            if registry_changed:
                raise RuntimeError("Transformers SDPA registry changed while significance recorder was active")


def _validate_model_kwargs(model_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    kwargs = dict(model_kwargs or {})
    reserved = {"past_key_values", "position_ids", "attention_mask", "use_cache", "input_ids"}
    overlap = reserved.intersection(kwargs)
    if overlap:
        raise ValueError(f"model_kwargs cannot override runtime-owned arguments: {sorted(overlap)}")
    return kwargs


@torch.no_grad()
def attention_prior_decode_append(
    model,
    cache: SignificanceCache,
    token_ids: torch.Tensor,
    *,
    attention_prior: SignificanceAttentionPrior,
    prior_trace: AttentionPriorTrace | None = None,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
):
    """Append one decode token while applying, but not updating, the prior."""

    cache.assert_forward_complete()
    if token_ids.ndim != 2 or token_ids.shape[-1] != 1:
        raise ValueError(f"token_ids must have shape [batch, 1], got {tuple(token_ids.shape)}")
    if cache.stored_length() <= 0:
        raise RuntimeError("cannot decode-append to an empty cache")
    first_layer = next(layer for layer in cache.layers if layer.is_initialized)
    cache_batch = int(first_layer.keys.shape[0])
    if int(token_ids.shape[0]) != cache_batch:
        raise ValueError(f"decode batch {token_ids.shape[0]} does not match cache batch {cache_batch}")
    kwargs = _validate_model_kwargs(model_kwargs)
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = int(logits_to_keep)
    positions = torch.full(
        (cache_batch, 1),
        cache.true_length,
        dtype=torch.long,
        device=token_ids.device,
    )
    mask = bottom_right_causal_mask(
        1,
        cache.stored_length() + 1,
        device=token_ids.device,
        dtype=_model_dtype(model),
    )
    cache.prepare_append(positions)
    with record_transformers_sdpa(
        cache,
        window=1,
        force_repeat_kv=force_repeat_kv,
        attention_prior=attention_prior,
        prior_trace=prior_trace,
        record_scores=False,
        trace_role="decode",
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
    return output


@torch.no_grad()
def score_with_query(
    model,
    cache: SignificanceCache,
    query_ids: torch.Tensor,
    *,
    verify_rollback: bool = True,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
    attention_prior: SignificanceAttentionPrior | None = None,
    prior_trace: AttentionPriorTrace | None = None,
) -> None:
    """Condition significance on a known query, then structurally undo it.

    Existing entries retain the query's contribution to S/C. Temporary query K/V,
    positions, and entry IDs are removed, and absolute/semantic counters return
    to their exact pre-probe values. With ``verify_rollback=True``, K/V are cloned
    and checked byte-for-byte; disable it only when the extra verification memory
    is prohibitive.
    """

    cache.assert_forward_complete()
    if query_ids.ndim != 2 or query_ids.shape[-1] <= 0:
        raise ValueError(f"query_ids must have shape [batch, q>0], got {tuple(query_ids.shape)}")
    if cache.stored_length() <= 0:
        raise RuntimeError("cannot query-condition an empty cache")
    if attention_prior is not None and attention_prior.access.query_access != "conditioned":
        raise ValueError(
            "query probing requires attention_prior.access.query_access='conditioned' so future-question "
            "access is explicit"
        )
    first_layer = next(layer for layer in cache.layers if layer.is_initialized)
    cache_batch = int(first_layer.keys.shape[0])
    if query_ids.shape[0] == 1 and cache_batch > 1:
        query_ids = query_ids.expand(cache_batch, -1)
    elif query_ids.shape[0] != cache_batch:
        raise ValueError(f"query batch {query_ids.shape[0]} does not match cache batch {cache_batch}")
    kwargs = _validate_model_kwargs(model_kwargs)
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = int(logits_to_keep)

    snapshot = cache.snapshot_structure(exact_tensors=verify_rollback)
    query_length = int(query_ids.shape[-1])
    positions = torch.arange(
        cache.true_length,
        cache.true_length + query_length,
        device=query_ids.device,
        dtype=torch.long,
    ).view(1, -1).expand(cache_batch, -1)
    mask = bottom_right_causal_mask(
        query_length,
        cache.stored_length() + query_length,
        device=query_ids.device,
        dtype=_model_dtype(model),
    )
    before_events = cache.scoring_events()
    try:
        cache.prepare_append(positions)
        with record_transformers_sdpa(
            cache,
            window=query_length,
            force_repeat_kv=force_repeat_kv,
            attention_prior=attention_prior,
            prior_trace=prior_trace,
            trace_role="query_probe",
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


@dataclass
class ChunkedPrefillResult:
    output: Any
    compression_reports: list[CompressionReport]
    logits: torch.Tensor | None = None


@torch.no_grad()
def chunked_prefill(
    model,
    input_ids: torch.Tensor,
    cache: SignificanceCache,
    *,
    chunk_size: int,
    policy: SignificancePolicy | None = None,
    query_ids: torch.Tensor | None = None,
    observation_window: int = 16,
    verify_query_rollback: bool = True,
    collect_logits: bool = False,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
    compression_callback: CompressionCallback | None = None,
    attention_prior: SignificanceAttentionPrior | None = None,
    prior_trace: AttentionPriorTrace | None = None,
) -> ChunkedPrefillResult:
    """Prefill with explicit bottom-right masks and between-chunk compression."""

    if input_ids.ndim != 2 or input_ids.shape[-1] <= 0:
        raise ValueError(f"input_ids must have shape [batch, n>0], got {tuple(input_ids.shape)}")
    if chunk_size <= 0 or observation_window <= 0:
        raise ValueError("chunk_size and observation_window must be positive")
    if query_ids is not None and policy is None:
        raise ValueError("query-conditioned scoring requires a compression policy")
    if compression_callback is not None and policy is None:
        raise ValueError("a compression callback requires a compression policy")
    if attention_prior is not None and policy is None:
        raise ValueError("a significance attention prior requires significance recording and a policy")
    if prior_trace is not None and attention_prior is None:
        raise ValueError("prior_trace requires an attention_prior")
    if (
        query_ids is not None
        and attention_prior is not None
        and attention_prior.access.query_access != "conditioned"
    ):
        raise ValueError(
            "query-conditioned state requires attention_prior.access.query_access='conditioned'"
        )
    kwargs_base = _validate_model_kwargs(model_kwargs)
    total = int(input_ids.shape[-1])
    n_chunks = (total + chunk_size - 1) // chunk_size
    output = None
    reports: list[CompressionReport] = []
    collected: list[torch.Tensor] = []

    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        piece = input_ids[:, start : start + chunk_size]
        query_length = int(piece.shape[-1])
        positions = torch.arange(
            cache.true_length,
            cache.true_length + query_length,
            device=input_ids.device,
            dtype=torch.long,
        ).view(1, -1).expand(input_ids.shape[0], -1)
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
        if policy is not None and query_ids is None:
            before_events = cache.scoring_events()
            with record_transformers_sdpa(
                cache,
                window=observation_window,
                force_repeat_kv=force_repeat_kv,
                attention_prior=attention_prior,
                prior_trace=prior_trace,
                trace_role="context_scoring",
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
            cache.require_one_score_event(before_events)
        else:
            if attention_prior is None:
                output = model(
                    piece,
                    past_key_values=cache,
                    position_ids=positions,
                    attention_mask={"full_attention": mask},
                    use_cache=True,
                    **kwargs,
                )
            else:
                # Query-conditioned runs score only through the temporary query
                # probe, but the prior still affects ordinary context forwards.
                with record_transformers_sdpa(
                    cache,
                    window=observation_window,
                    force_repeat_kv=force_repeat_kv,
                    attention_prior=attention_prior,
                    prior_trace=prior_trace,
                    record_scores=False,
                    trace_role="context_prefill",
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

        if collect_logits:
            if not hasattr(output, "logits"):
                raise RuntimeError("collect_logits requires a causal-LM output with a logits tensor")
            collected.append(output.logits.float())

        if policy is not None and chunk_index < n_chunks - 1:
            if query_ids is not None:
                score_with_query(
                    model,
                    cache,
                    query_ids,
                    verify_rollback=verify_query_rollback,
                    force_repeat_kv=force_repeat_kv,
                    logits_to_keep=1,
                    model_kwargs=model_kwargs,
                    attention_prior=attention_prior,
                    prior_trace=prior_trace,
                )
            reports.append(
                cache.compress(policy)
                if compression_callback is None
                else compression_callback(cache, policy, chunk_index)
            )

    logits = torch.cat(collected, dim=1) if collected else None
    return ChunkedPrefillResult(output=output, compression_reports=reports, logits=logits)
