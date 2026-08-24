from __future__ import annotations

import numpy as np
import pytest

from rineya_significance_lab.core import (
    SignificanceState,
    allocate_global_budget,
    significance_diagnostics,
    stable_topk,
)


def diagnostic_state() -> SignificanceState:
    state = SignificanceState.from_entries(
        np.arange(8),
        np.arange(8),
        np.arange(8),
        layers=1,
        kv_heads=2,
        counts=np.ones(8, dtype=np.float32),
    )
    state.scores[0, 0] = np.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    state.scores[0, 1] = np.ones(8, dtype=np.float32)
    return state


def test_entropy_allocator_gives_diffuse_head_more_without_extra_slots() -> None:
    diagnostics = significance_diagnostics(diagnostic_state())
    assert diagnostics.gini[0, 0] > diagnostics.gini[0, 1]
    allocation = allocate_global_budget(
        diagnostics,
        strategy="entropy_proportional",
        global_budget=12,
        minimum_per_unit=2,
    )
    assert allocation.allocations.sum() == 12
    assert allocation.allocations[0, 1] > allocation.allocations[0, 0]
    assert np.all(allocation.allocations <= diagnostics.capacity)
    selection = stable_topk(
        diagnostic_state(),
        diagnostic_state().scores,
        allocation.allocations,
    )
    assert np.array_equal(selection.mask.sum(axis=-1), allocation.allocations)


@pytest.mark.parametrize(
    "strategy",
    ["uniform", "entropy_proportional", "inverse_concentration", "inverse_stability", "shuffled_statistic"],
)
def test_every_adaptive_strategy_is_deterministic_and_conserves_budget(strategy: str) -> None:
    diagnostics = significance_diagnostics(diagnostic_state())
    first = allocate_global_budget(
        diagnostics,
        strategy=strategy,
        global_budget=14,
        minimum_per_unit=3,
        maximum_per_unit=9,
        seed=17,
    )
    second = allocate_global_budget(
        diagnostics,
        strategy=strategy,
        global_budget=14,
        minimum_per_unit=3,
        maximum_per_unit=9,
        seed=17,
    )
    assert np.array_equal(first.allocations, second.allocations)
    assert int(first.allocations.sum()) == 14
    assert np.all((first.allocations >= 3) & (first.allocations <= 9))


def test_impossible_global_budget_fails_loudly() -> None:
    diagnostics = significance_diagnostics(diagnostic_state())
    with pytest.raises(ValueError, match="minimum"):
        allocate_global_budget(diagnostics, strategy="uniform", global_budget=3, minimum_per_unit=2)
