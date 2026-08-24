from __future__ import annotations

import numpy as np
import pytest

from rineya_significance_lab.core import SignificanceState, StateInvariantError, atomic_gather


def make_state() -> SignificanceState:
    return SignificanceState.from_entries(
        np.array([101, 102, 103, 104]),
        np.array([5, 9, 17, 22]),
        np.array([11, 12, 13, 14]),
        layers=2,
        kv_heads=2,
        scores=np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float32),
        counts=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )


def test_state_persists_fp32_scores_and_counts() -> None:
    state = make_state()
    assert state.scores.dtype == np.float32
    assert state.counts.dtype == np.float32
    delta = np.ones(state.shape, dtype=np.float32)
    state.update(delta, delta)
    assert state.scores.dtype == np.float32
    assert state.counts.dtype == np.float32


def test_atomic_gather_keeps_every_field_and_payload_aligned() -> None:
    state = make_state()
    indices = np.array([[[2, 0], [1, 3]], [[3, 1], [0, 2]]], dtype=np.int64)
    key = np.zeros((*state.shape, 3), dtype=np.float32)
    value = np.zeros((*state.shape, 3), dtype=np.float32)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            for slot in range(state.slots):
                semantic = state.entry_ids[layer, head, slot]
                key[layer, head, slot] = semantic
                value[layer, head, slot] = semantic * 10
    gathered, payloads = atomic_gather(state, indices, {"K": key, "V": value})
    for layer in range(gathered.layers):
        for head in range(gathered.kv_heads):
            for slot in range(gathered.slots):
                semantic = gathered.entry_ids[layer, head, slot]
                assert np.all(payloads["K"][layer, head, slot] == semantic)
                assert np.all(payloads["V"][layer, head, slot] == semantic * 10)
    gathered.validate()


def test_alignment_invariants_reject_duplicate_or_inactive_metadata() -> None:
    state = make_state()
    state.entry_ids[0, 0, 1] = state.entry_ids[0, 0, 0]
    with pytest.raises(StateInvariantError, match="duplicate"):
        state.validate()

    state = make_state()
    state.active[0, 0, 0] = False
    with pytest.raises(StateInvariantError, match="inactive"):
        state.validate()


def test_atomic_gather_rejects_misaligned_payload_and_duplicate_selection() -> None:
    state = make_state()
    with pytest.raises(StateInvariantError, match="payload"):
        atomic_gather(state, np.zeros((2, 2, 1), dtype=np.int64), {"K": np.zeros((2, 2, 3, 4))})
    duplicate = np.zeros((2, 2, 2), dtype=np.int64)
    with pytest.raises(StateInvariantError, match="duplicate"):
        atomic_gather(state, duplicate)
