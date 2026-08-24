"""Explicit query-head to KV-head score/exposure reduction for GQA traces."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Reduction = Literal["sum", "mean", "max"]


def _reduce(values: NDArray[np.float64], axis: int, mode: Reduction) -> NDArray[np.float64]:
    if mode == "sum":
        return np.asarray(np.sum(values, axis=axis, dtype=np.float64), dtype=np.float64)
    if mode == "mean":
        return np.asarray(np.mean(values, axis=axis, dtype=np.float64), dtype=np.float64)
    if mode == "max":
        return np.asarray(np.max(values, axis=axis), dtype=np.float64)
    raise ValueError(f"unsupported reduction: {mode}")


def reduce_gqa_attention(
    attention: NDArray[Any],
    *,
    kv_heads: int,
    query_reduce: Reduction = "sum",
    group_reduce: Reduction = "sum",
) -> NDArray[np.float32]:
    """Reduce ``[batch, query_heads, query_tokens, keys]`` to KV-head scores.

    Qwen3 assigns contiguous groups of query heads to each KV head. For the
    local 16Q/8KV geometry, heads ``(0,1)`` map to KV head 0, ``(2,3)`` to KV
    head 1, and so on. The reduction choices are trace semantics and therefore
    must be frozen in each run manifest.
    """

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 4 or kv_heads <= 0:
        raise ValueError("attention must be [batch, query_heads, query_tokens, keys]")
    batch, query_heads, _, keys = values.shape
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    groups = query_heads // kv_heads
    grouped = values.reshape(batch, kv_heads, groups, values.shape[2], keys)
    by_query = _reduce(grouped, axis=3, mode=query_reduce)
    by_group = _reduce(by_query, axis=2, mode=group_reduce)
    return np.asarray(by_group, dtype=np.float32)


def realized_exposure(
    visibility: NDArray[Any],
    *,
    query_heads: int,
    kv_heads: int,
    mode: Literal["event", "visible_queries"] = "event",
) -> NDArray[np.float32]:
    """Convert visibility ``[batch, query_tokens, keys]`` into KV-head counts."""

    visible = np.asarray(visibility, dtype=np.bool_)
    if visible.ndim != 3 or min(query_heads, kv_heads) <= 0 or query_heads % kv_heads:
        raise ValueError("invalid visibility/head geometry")
    if mode == "event":
        exposure = np.any(visible, axis=1, keepdims=False).astype(np.float32)
    elif mode == "visible_queries":
        exposure = np.sum(visible, axis=1, dtype=np.float32)
    else:
        raise ValueError(f"unsupported exposure mode: {mode}")
    return np.broadcast_to(exposure[:, None, :], (visible.shape[0], kv_heads, visible.shape[2])).copy()
