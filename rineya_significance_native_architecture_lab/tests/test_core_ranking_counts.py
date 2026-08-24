from __future__ import annotations

import numpy as np
import pytest

from rineya_significance_lab.core import (
    CountTransformContext,
    SignificanceState,
    normalized_rank,
    realized_exposure,
    reduce_gqa_attention,
    stable_topk,
    transform_counts,
)
from rineya_significance_lab.policies import PolicyContext, PolicyError, policy_from_baseline


def make_state() -> SignificanceState:
    return SignificanceState.from_entries(
        np.array([10, 11, 12, 13]),
        np.array([0, 1, 2, 3]),
        np.array([100, 101, 102, 103]),
        layers=1,
        kv_heads=2,
        scores=np.array([6.0, 4.0, 2.4, 1.25], dtype=np.float32),
        counts=np.array([6.0, 4.0, 2.0, 1.0], dtype=np.float32),
    )


def test_observation_normalization_removes_pure_exposure_factor() -> None:
    state = make_state()
    raw = normalized_rank(state, 0.0)
    normalized = normalized_rank(state, 1.0)
    assert raw[0, 0, 0] > raw[0, 0, 3]
    assert normalized[0, 0, 3] > normalized[0, 0, 0]


def test_ties_use_position_then_entry_id_and_output_cache_order() -> None:
    state = make_state()
    rank = np.ones(state.shape, dtype=np.float64)
    selected = stable_topk(state, rank, 2)
    assert state.selected_entry_ids(selected.indices)[0, 0].tolist() == [10, 11]


def test_all_count_transforms_are_deterministic_and_aligned() -> None:
    state = make_state()
    context = CountTransformContext(
        current_position=8,
        chunk_size=2,
        previous_counts=np.maximum(state.counts - 1, 0),
        noise_std=0.1,
        seed=9,
        event_id="boundary-3",
    )
    names = ("true", "constant", "age", "shuffled", "circular", "delayed", "noisy", "head_averaged")
    for name in names:
        first = transform_counts(state, name, context)
        second = transform_counts(state, name, context)
        assert first.shape == state.shape
        assert np.array_equal(first, second)
        assert np.all(first >= 0)


def test_age_count_uses_one_past_boundary_without_extra_first_token_exposure() -> None:
    state = make_state()
    counts = transform_counts(
        state,
        "age",
        CountTransformContext(current_position=4, chunk_size=2),
    )
    assert counts[0, 0].tolist() == [2.0, 2.0, 1.0, 1.0]


def test_head_average_joins_by_stable_entry_id_not_slot() -> None:
    state = make_state()
    state.entry_ids[0, 1] = state.entry_ids[0, 1, ::-1]
    state.positions[0, 1] = state.positions[0, 1, ::-1]
    state.token_ids[0, 1] = state.token_ids[0, 1, ::-1]
    state.counts[0, 1] = np.array([40, 30, 20, 10], dtype=np.float32)
    state.validate()
    averaged = transform_counts(state, "head_averaged")
    # Entry 10 has count 6 in head 0 and 10 in its moved slot in head 1.
    assert averaged[0, 0, 0] == pytest.approx(8.0)


def test_b1_is_gated_and_matches_b3_at_verified_e1() -> None:
    state = make_state()
    context = PolicyContext(budget=2, exponent=1.0)
    with pytest.raises(PolicyError, match="authoritative"):
        policy_from_baseline("B1").select(state, context)
    verified = PolicyContext(budget=2, exponent=1.0, source_semantics_verified=True)
    b1 = policy_from_baseline("B1").select(state, verified)
    b3 = policy_from_baseline("B3").select(state, verified)
    assert np.array_equal(b1.indices, b3.indices)


def test_explicit_qwen_gqa_reduction_maps_two_query_heads_per_kv_head() -> None:
    attention = np.zeros((1, 16, 2, 3), dtype=np.float32)
    for query_head in range(16):
        attention[:, query_head, :, :] = query_head
    reduced = reduce_gqa_attention(attention, kv_heads=8, query_reduce="mean", group_reduce="mean")
    audited_sum = reduce_gqa_attention(attention, kv_heads=8)
    assert reduced.shape == (1, 8, 3)
    assert np.allclose(reduced[0, 0], 0.5)
    assert np.allclose(reduced[0, 7], 14.5)
    assert np.allclose(audited_sum[0, 0], 2.0)
    assert np.allclose(audited_sum[0, 7], 58.0)
    visibility = np.array([[[True, False, True], [True, True, False]]])
    event = realized_exposure(visibility, query_heads=16, kv_heads=8, mode="event")
    queries = realized_exposure(visibility, query_heads=16, kv_heads=8, mode="visible_queries")
    assert event.shape == (1, 8, 3)
    assert event[0, 0].tolist() == [1, 1, 1]
    assert queries[0, 0].tolist() == [2, 1, 1]
