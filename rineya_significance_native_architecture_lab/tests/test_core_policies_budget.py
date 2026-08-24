from __future__ import annotations

import numpy as np
import pytest

from rineya_significance_lab.core import BudgetLedger, CacheGeometry, SignificanceState
from rineya_significance_lab.policies import (
    PolicyContext,
    PolicyError,
    available_baselines,
    policy_from_baseline,
)


def make_state() -> SignificanceState:
    return SignificanceState.from_entries(
        np.arange(8, dtype=np.int64),
        np.arange(8, dtype=np.int64),
        np.arange(100, 108, dtype=np.int64),
        layers=2,
        kv_heads=2,
        scores=np.array([8, 7, 6, 5, 4, 3, 2, 9], dtype=np.float32),
        counts=np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32),
    )


def test_every_b0_b8_policy_is_available_and_deterministic() -> None:
    state = make_state()
    evidence = np.zeros(state.shape, dtype=np.bool_)
    evidence[:, :, -1] = True
    context = PolicyContext(
        budget=4,
        sink_tokens=1,
        recent_tokens=1,
        current_position=8,
        chunk_size=1,
        seed=5,
        event_id="event-x",
        oracle_evidence=evidence,
        allow_oracle=True,
        source_semantics_verified=True,
    )
    assert available_baselines() == tuple(f"B{index}" for index in range(9))
    for baseline in available_baselines():
        first = policy_from_baseline(baseline).select(state, context)
        second = policy_from_baseline(baseline).select(state, context)
        assert np.array_equal(first.indices, second.indices)
        expected = state.slots if baseline == "B0" else 4
        assert np.all(first.mask.sum(axis=-1) == expected)


def test_sink_recent_protection_and_oracle_guard() -> None:
    state = make_state()
    context = PolicyContext(budget=3, sink_tokens=1, recent_tokens=1)
    selection = policy_from_baseline("B3").select(state, context)
    assert selection.mask[:, :, 0].all()
    assert selection.mask[:, :, -1].all()
    with pytest.raises(PolicyError, match="allow_oracle"):
        policy_from_baseline("B8").select(state, context)


def test_budget_ledger_reports_slot_and_total_byte_matching() -> None:
    ledger = BudgetLedger(CacheGeometry(layers=28, kv_heads=8, head_dim=128))
    no_sig = ledger.report(1024, significance=False)
    sig_same_slots = ledger.report(1024, significance=True)
    assert no_sig.kv_bytes == sig_same_slots.kv_bytes
    assert sig_same_slots.total_persistent_bytes > no_sig.total_persistent_bytes
    baseline, candidate = ledger.match_total_bytes(
        1024,
        baseline_significance=False,
        candidate_significance=True,
    )
    assert candidate.slots_per_layer_head < baseline.slots_per_layer_head
    assert candidate.total_persistent_bytes <= baseline.total_persistent_bytes
    assert baseline.kv_bytes == no_sig.kv_bytes
