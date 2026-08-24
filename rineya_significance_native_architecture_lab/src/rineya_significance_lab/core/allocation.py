"""Matched-global-budget allocation from Cornerstone-1 diagnostics.

This module is open-loop infrastructure for Architecture Experiment B. It does
not claim that a heterogeneous allocation changes Rineya outputs; the current
Transformers runtime intentionally keeps equal rectangular cache lengths until
per-layer masks pass a separate closed-loop identity gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .randomness import stable_rng
from .ranking import normalized_rank
from .state import SignificanceState

AllocationStrategy = Literal[
    "uniform",
    "entropy_proportional",
    "inverse_concentration",
    "inverse_stability",
    "shuffled_statistic",
]


@dataclass(frozen=True, slots=True)
class SignificanceDiagnostics:
    entropy: NDArray[np.float64]
    gini: NDArray[np.float64]
    top_mass: NDArray[np.float64]
    stability: NDArray[np.float64]
    capacity: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class AdaptiveBudget:
    allocations: NDArray[np.int64]
    strategy: AllocationStrategy
    global_budget: int
    minimum_per_unit: int

    def validate(self) -> None:
        if self.allocations.ndim != 2 or np.any(self.allocations < self.minimum_per_unit):
            raise ValueError("adaptive allocations violate their shape/minimum contract")
        if int(self.allocations.sum()) != self.global_budget:
            raise ValueError("adaptive allocations do not conserve the global budget")


def _gini(probabilities: NDArray[np.float64]) -> float:
    # Zero-mass entries are part of the distribution. Dropping them would call
    # a one-hot head perfectly equal (Gini 0) instead of highly concentrated.
    values = np.sort(np.maximum(probabilities, 0.0))
    if values.size <= 1 or float(values.sum()) == 0.0:
        return 0.0
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(indices * values) / (values.size * values.sum())) - (values.size + 1) / values.size)


def significance_diagnostics(
    state: SignificanceState,
    *,
    exponent: float = 1.0,
    top_fraction: float = 0.1,
    previous_selected: NDArray[np.bool_] | None = None,
) -> SignificanceDiagnostics:
    """Measure entropy, Gini, top mass, and retained-set stability per unit."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if previous_selected is not None and np.asarray(previous_selected).shape != state.shape:
        raise ValueError("previous_selected must match state shape")
    rank = np.maximum(normalized_rank(state, exponent), 0.0)
    shape = state.shape[:2]
    entropy = np.zeros(shape, dtype=np.float64)
    gini = np.zeros(shape, dtype=np.float64)
    top_mass = np.zeros(shape, dtype=np.float64)
    stability = np.zeros(shape, dtype=np.float64)
    capacity = state.active.sum(axis=-1, dtype=np.int64)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            slots = np.flatnonzero(state.active[layer, head])
            if not slots.size:
                continue
            values = rank[layer, head, slots]
            total = float(values.sum())
            probabilities = (
                values / total if total > 0 else np.full(slots.size, 1.0 / slots.size, dtype=np.float64)
            )
            raw_entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30))))
            entropy[layer, head] = raw_entropy / np.log(slots.size) if slots.size > 1 else 0.0
            gini[layer, head] = _gini(probabilities)
            top_count = max(1, int(np.ceil(slots.size * top_fraction)))
            top_mass[layer, head] = float(np.sort(probabilities)[-top_count:].sum())
            if previous_selected is not None:
                old = np.asarray(previous_selected, dtype=np.bool_)[layer, head, slots]
                current = np.zeros(slots.size, dtype=np.bool_)
                current[np.argsort(values, kind="stable")[-int(max(1, old.sum())) :]] = True
                union = int(np.logical_or(old, current).sum())
                stability[layer, head] = float(np.logical_and(old, current).sum() / union) if union else 1.0
    return SignificanceDiagnostics(
        entropy=entropy,
        gini=gini,
        top_mass=top_mass,
        stability=stability,
        capacity=capacity,
    )


def _conserved_allocation(
    weights: NDArray[np.float64],
    global_budget: int,
    minimum_per_unit: int,
    capacity: NDArray[np.int64],
) -> NDArray[np.int64]:
    if weights.ndim != 2 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("allocation weights must be a finite non-negative [layers, heads] matrix")
    units = weights.size
    if capacity.shape != weights.shape or np.any(capacity < 0):
        raise ValueError("per-unit capacity must match weights and be non-negative")
    if np.any(capacity < minimum_per_unit):
        raise ValueError("a per-unit capacity cannot satisfy the declared minimum")
    if global_budget < units * minimum_per_unit:
        raise ValueError("global budget cannot satisfy the per-unit minimum")
    if global_budget > int(capacity.sum()):
        raise ValueError("global budget exceeds active per-unit capacity")
    result = np.full(weights.shape, minimum_per_unit, dtype=np.int64)
    remaining_capacity = capacity.astype(np.int64, copy=True) - minimum_per_unit
    remaining = global_budget - int(result.sum())
    while remaining:
        eligible = remaining_capacity > 0
        if not eligible.any():
            raise ValueError("no capacity remains before the global budget is allocated")
        effective = np.where(eligible, weights, 0.0)
        if float(effective.sum()) == 0:
            effective = eligible.astype(np.float64)
        shares = effective / effective.sum() * remaining
        grants = np.minimum(np.floor(shares).astype(np.int64), remaining_capacity)
        granted = int(grants.sum())
        if granted:
            result += grants
            remaining_capacity -= grants
            remaining -= granted
            continue
        fractions = (shares - np.floor(shares)).ravel()
        order = sorted(range(units), key=lambda index: (-fractions[index], index))
        for index in order:
            row = np.unravel_index(index, weights.shape)
            if remaining_capacity[row] <= 0:
                continue
            result[row] += 1
            remaining_capacity[row] -= 1
            remaining -= 1
            if remaining == 0:
                break
    return result


def allocate_global_budget(
    diagnostics: SignificanceDiagnostics,
    *,
    strategy: AllocationStrategy,
    global_budget: int,
    minimum_per_unit: int = 0,
    maximum_per_unit: int | None = None,
    seed: int = 0,
) -> AdaptiveBudget:
    """Allocate exactly one fixed global slot budget across layer/KV-head units."""

    if minimum_per_unit < 0 or (maximum_per_unit is not None and maximum_per_unit < minimum_per_unit):
        raise ValueError("invalid adaptive-budget bounds")
    if strategy == "uniform":
        weights = np.ones_like(diagnostics.entropy)
    elif strategy == "entropy_proportional":
        weights = diagnostics.entropy + 1e-12
    elif strategy == "inverse_concentration":
        weights = 1.0 / np.maximum(diagnostics.top_mass, 1e-12)
    elif strategy == "inverse_stability":
        weights = 1.0 - diagnostics.stability + 1e-12
    elif strategy == "shuffled_statistic":
        weights = diagnostics.entropy.ravel().copy()
        stable_rng(seed, "adaptive-budget-shuffle").shuffle(weights)
        weights = weights.reshape(diagnostics.entropy.shape)
    else:
        raise ValueError(f"unknown adaptive-budget strategy: {strategy}")
    capacity = diagnostics.capacity.astype(np.int64, copy=True)
    if maximum_per_unit is not None:
        capacity = np.minimum(capacity, int(maximum_per_unit))
    allocation = AdaptiveBudget(
        allocations=_conserved_allocation(weights, global_budget, minimum_per_unit, capacity),
        strategy=strategy,
        global_budget=global_budget,
        minimum_per_unit=minimum_per_unit,
    )
    allocation.validate()
    return allocation
