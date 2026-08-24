"""Cornerstone-1 ranking and deterministic selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .state import SignificanceState, StateInvariantError


@dataclass(frozen=True, slots=True)
class Selection:
    indices: NDArray[np.int64]
    mask: NDArray[np.bool_]
    budget: NDArray[np.int64]
    policy_id: str


def normalized_rank(
    state: SignificanceState,
    exponent: float,
    *,
    counts: NDArray[Any] | None = None,
) -> NDArray[np.float64]:
    if not np.isfinite(exponent) or exponent < 0:
        raise ValueError("significance exponent must be finite and non-negative")
    effective = state.counts if counts is None else np.asarray(counts, dtype=np.float64)
    if effective.shape != state.shape or not np.isfinite(effective).all() or np.any(effective < 0):
        raise StateInvariantError("ranking counts must match state and be finite/non-negative")
    denominator = np.power(np.maximum(effective, 1.0), exponent)
    rank = np.divide(state.scores, denominator, dtype=np.float64)
    return np.where(state.active, rank, -np.inf)


def _budgets(value: int | NDArray[Any], state: SignificanceState) -> NDArray[np.int64]:
    budget = np.asarray(value, dtype=np.int64)
    if budget.ndim == 0:
        budget = np.full(state.shape[:2], int(budget), dtype=np.int64)
    if budget.shape != state.shape[:2] or np.any(budget < 0):
        raise ValueError("budget must be non-negative scalar or [layers, kv_heads] array")
    return budget


def stable_topk(
    state: SignificanceState,
    rank: NDArray[Any],
    budget: int | NDArray[Any],
    *,
    eligible: NDArray[Any] | None = None,
    protected: NDArray[Any] | None = None,
    policy_id: str = "significance",
) -> Selection:
    """Select by rank with deterministic ties, then return cache-order indices.

    Decision ties are broken by older absolute position, stable entry ID, and
    original slot. Returned indices are sorted by absolute position so gathered
    cache entries retain causal order.
    """

    values = np.asarray(rank, dtype=np.float64)
    if values.shape != state.shape or np.isnan(values).any():
        raise StateInvariantError("rank must match state and contain no NaN")
    allowed = state.active.copy() if eligible is None else state.active & np.asarray(eligible, dtype=np.bool_)
    fixed = np.zeros(state.shape, dtype=np.bool_) if protected is None else np.asarray(protected, dtype=np.bool_)
    if allowed.shape != state.shape or fixed.shape != state.shape:
        raise StateInvariantError("eligible/protected masks must match state")
    if np.any(fixed & ~allowed):
        raise StateInvariantError("protected entries must also be eligible")
    budgets = _budgets(budget, state)
    width = int(budgets.max(initial=0))
    indices = np.full((*state.shape[:2], width), -1, dtype=np.int64)
    mask = np.zeros(state.shape, dtype=np.bool_)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            row_budget = int(budgets[layer, head])
            protected_slots = np.flatnonzero(fixed[layer, head])
            if protected_slots.size > row_budget:
                raise StateInvariantError(
                    f"protected entries ({protected_slots.size}) exceed budget ({row_budget}) "
                    f"at layer={layer}, kv_head={head}"
                )
            candidates = np.flatnonzero(allowed[layer, head] & ~fixed[layer, head])
            ordered = sorted(
                candidates.tolist(),
                key=lambda slot: (
                    -float(values[layer, head, slot]),
                    int(state.positions[layer, head, slot]),
                    int(state.entry_ids[layer, head, slot]),
                    int(slot),
                ),
            )
            selected = protected_slots.tolist() + ordered[: max(0, row_budget - protected_slots.size)]
            selected.sort(
                key=lambda slot: (
                    int(state.positions[layer, head, slot]),
                    int(state.entry_ids[layer, head, slot]),
                    int(slot),
                )
            )
            if selected:
                indices[layer, head, : len(selected)] = selected
                mask[layer, head, selected] = True
    return Selection(indices=indices, mask=mask, budget=budgets, policy_id=policy_id)
