"""Closed-loop allocation from the live padded Significance Matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.allocation import (
    AdaptiveBudget,
    AllocationStrategy,
    SignificanceDiagnostics,
    allocate_global_budget,
    significance_diagnostics,
)
from ..core.state import SignificanceState
from .heterogeneous_cache import HeterogeneousSignificanceCache


@dataclass(frozen=True, slots=True)
class LiveBudgetDecision:
    """Diagnostics and exactly conserved allocation for one live boundary."""

    diagnostics: SignificanceDiagnostics
    budget: AdaptiveBudget

    def torch_allocations(self, *, device: torch.device | str) -> torch.Tensor:
        return torch.as_tensor(self.budget.allocations, dtype=torch.long, device=device)


def live_significance_state(cache: HeterogeneousSignificanceCache) -> SignificanceState:
    """Copy all live layer/head rows into a Torch-free padded state."""

    cache.assert_forward_complete()
    if any(layer.state is None or layer.validity is None for layer in cache.layers):
        raise RuntimeError("live allocation requires every layer to be initialized")
    first = cache.layers[0]
    assert first.state is not None
    if first.state.batch_size != 1:
        raise RuntimeError(
            "closed-loop allocation currently requires batch size 1 so one global budget "
            "cannot ambiguously mix sequence-specific diagnostics"
        )
    layers = len(cache.layers)
    heads = first.state.kv_heads
    width = max(layer.physical_length() for layer in cache.layers)
    state = SignificanceState.empty(layers, heads, width)
    for layer_idx, layer in enumerate(cache.layers):
        assert layer.state is not None and layer.validity is not None
        if layer.state.batch_size != 1 or layer.state.kv_heads != heads:
            raise RuntimeError("cache layers disagree on batch size or KV-head count")
        length = layer.physical_length()
        state.entry_ids[layer_idx, :, :length] = (
            layer.state.entry_ids[0].detach().cpu().numpy().astype(np.int64, copy=False)
        )
        state.positions[layer_idx, :, :length] = (
            layer.state.positions[0].detach().cpu().numpy().astype(np.int64, copy=False)
        )
        # Stable semantic entry IDs are an explicit surrogate because token IDs
        # are not stored in the runtime cache and are irrelevant to allocation.
        state.token_ids[layer_idx, :, :length] = state.entry_ids[layer_idx, :, :length]
        state.scores[layer_idx, :, :length] = (
            layer.state.scores[0].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        state.counts[layer_idx, :, :length] = (
            layer.state.counts[0].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        state.active[layer_idx, :, :length] = layer.validity[0].detach().cpu().numpy()
    state.validate()
    return state


def allocate_live_budget(
    cache: HeterogeneousSignificanceCache,
    *,
    strategy: AllocationStrategy,
    global_budget: int,
    minimum_per_unit: int,
    maximum_per_unit: int | None,
    exponent: float = 1.0,
    seed: int = 0,
) -> LiveBudgetDecision:
    """Allocate a fixed global budget from current S/C diagnostics."""

    state = live_significance_state(cache)
    diagnostics = significance_diagnostics(state, exponent=exponent)
    budget = allocate_global_budget(
        diagnostics,
        strategy=strategy,
        global_budget=global_budget,
        minimum_per_unit=minimum_per_unit,
        maximum_per_unit=maximum_per_unit,
        seed=seed,
    )
    if int(budget.allocations.sum()) != global_budget:
        raise RuntimeError("allocator failed exact global-budget conservation")
    return LiveBudgetDecision(diagnostics=diagnostics, budget=budget)
