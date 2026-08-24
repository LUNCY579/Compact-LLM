"""Boundary-local open-loop screen for matched global KV allocations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.allocation import (
    AdaptiveBudget,
    AllocationStrategy,
    SignificanceDiagnostics,
    allocate_global_budget,
    significance_diagnostics,
)
from ..core.ranking import Selection, normalized_rank, stable_topk
from ..core.state import SignificanceState


@dataclass(frozen=True, slots=True)
class BoundaryBudgetScreen:
    state: SignificanceState
    diagnostics: SignificanceDiagnostics
    budget: AdaptiveBudget
    selection: Selection
    protected: NDArray[np.bool_]


def state_from_count_boundary(
    arrays: Mapping[str, NDArray[Any]],
    *,
    boundary_index: int,
    layer_count: int,
) -> SignificanceState:
    """Reconstruct the complete pre-selection state recorded by count controls."""

    if boundary_index < 0 or layer_count <= 0:
        raise ValueError("boundary_index must be non-negative and layer_count positive")

    def stacked(suffix: str, dtype: Any) -> NDArray[Any]:
        rows = []
        for layer in range(layer_count):
            name = f"b{boundary_index:03d}_l{layer:03d}_{suffix}"
            if name not in arrays:
                raise KeyError(f"boundary trace is missing {name}")
            value = np.asarray(arrays[name], dtype=dtype)
            if value.ndim != 3 or value.shape[0] != 1:
                raise ValueError(f"{name} must have shape [1, KV heads, positions]")
            rows.append(value[0])
        shapes = {row.shape for row in rows}
        if len(shapes) != 1:
            raise ValueError(f"layer rows disagree for boundary suffix {suffix!r}: {shapes}")
        return np.stack(rows, axis=0)

    positions = stacked("positions", np.int64)
    entry_ids = stacked("entry_ids", np.int64)
    scores = stacked("scores", np.float32)
    counts = stacked("true_counts", np.float32)
    # Token IDs are not required for ranking or evidence-position scoring. The
    # stable semantic IDs are used as an explicit surrogate rather than
    # fabricating content labels.
    return SignificanceState(
        entry_ids=entry_ids,
        positions=positions,
        token_ids=entry_ids.copy(),
        scores=scores,
        counts=counts,
        active=np.ones(entry_ids.shape, dtype=np.bool_),
    )


def sink_recent_protection(
    state: SignificanceState,
    *,
    sink_tokens: int,
    recent_tokens: int,
) -> NDArray[np.bool_]:
    if sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("sink/recent protection must be non-negative")
    protected = np.zeros(state.shape, dtype=np.bool_)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            slots = np.flatnonzero(state.active[layer, head])
            ordered = sorted(
                slots.tolist(),
                key=lambda slot: (
                    int(state.positions[layer, head, slot]),
                    int(state.entry_ids[layer, head, slot]),
                    slot,
                ),
            )
            for slot in ordered[:sink_tokens]:
                protected[layer, head, slot] = True
            if recent_tokens:
                for slot in ordered[-recent_tokens:]:
                    protected[layer, head, slot] = True
    return protected


def screen_adaptive_budget(
    state: SignificanceState,
    *,
    strategy: AllocationStrategy,
    average_budget_per_unit: int,
    minimum_per_unit: int,
    maximum_per_unit: int | None,
    sink_tokens: int,
    recent_tokens: int,
    exponent: float = 1.0,
    seed: int = 0,
) -> BoundaryBudgetScreen:
    """Allocate and select one boundary under an exactly conserved slot sum."""

    if average_budget_per_unit <= 0:
        raise ValueError("average_budget_per_unit must be positive")
    diagnostics = significance_diagnostics(state, exponent=exponent)
    global_budget = state.layers * state.kv_heads * average_budget_per_unit
    budget = allocate_global_budget(
        diagnostics,
        strategy=strategy,
        global_budget=global_budget,
        minimum_per_unit=minimum_per_unit,
        maximum_per_unit=maximum_per_unit,
        seed=seed,
    )
    protected = sink_recent_protection(
        state,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
    )
    protected_counts = protected.sum(axis=-1)
    if np.any(protected_counts > budget.allocations):
        raise ValueError("adaptive allocation cannot hold the declared sink/recent protection")
    selection = stable_topk(
        state,
        normalized_rank(state, exponent),
        budget.allocations,
        protected=protected,
        policy_id=f"adaptive:{strategy}",
    )
    if int(selection.mask.sum()) != global_budget:
        raise RuntimeError("adaptive selection did not consume the exact global slot budget")
    return BoundaryBudgetScreen(
        state=state,
        diagnostics=diagnostics,
        budget=budget,
        selection=selection,
        protected=protected,
    )


def selected_resident_positions(screen: BoundaryBudgetScreen) -> list[list[list[int]]]:
    """Convert a variable per-unit selection to exact-evidence scorer input."""

    result: list[list[list[int]]] = []
    for layer in range(screen.state.layers):
        heads: list[list[int]] = []
        for head in range(screen.state.kv_heads):
            slots = np.flatnonzero(screen.selection.mask[layer, head])
            heads.append(
                sorted(int(screen.state.positions[layer, head, slot]) for slot in slots)
            )
        result.append(heads)
    return result
