from __future__ import annotations

import pytest
import torch

from rineya_significance_lab.runtime import (
    SignificanceCache,
    SignificancePolicy,
    bottom_right_causal_allowed,
    bottom_right_causal_mask,
    post_rope_gqa_significance,
    post_rope_gqa_significance_diagnostics,
    select_keep_indices,
)
from rineya_significance_lab.torch_state import TorchSignificanceState, gather_counts


def test_bottom_right_mask_allows_resident_prefix_and_causal_chunk() -> None:
    allowed = bottom_right_causal_allowed(3, 7)
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(allowed, expected)
    additive = bottom_right_causal_mask(3, 7, device="cpu", dtype=torch.float32)
    assert additive.shape == (1, 1, 3, 7)
    assert torch.equal(additive[0, 0] == 0, expected)
    assert (additive[0, 0][~expected] < -1e30).all()


def test_post_rope_gqa_reduces_16_query_heads_to_8_kv_heads() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn((1, 16, 3, 4), generator=generator)
    key = torch.randn((1, 8, 5, 4), generator=generator)
    got = post_rope_gqa_significance(query, key, window=2)

    grouped_query = query[:, :, -2:, :].reshape(1, 8, 2, 2, 4).float()
    logits = torch.einsum("bgcwd,bgnd->bgcwn", grouped_query, key.float()) / 2.0
    allowed = bottom_right_causal_allowed(2, 5).view(1, 1, 1, 2, 5)
    expected = logits.masked_fill(~allowed, float("-inf")).softmax(-1).sum(dim=(2, 3))
    assert got.shape == (1, 8, 5)
    torch.testing.assert_close(got, expected)
    # Two query heads x two observed rows each contribute unit probability mass.
    torch.testing.assert_close(got.sum(-1), torch.full((1, 8), 4.0))
    diagnostics = post_rope_gqa_significance_diagnostics(query, key, window=2)
    torch.testing.assert_close(diagnostics.contribution, got)
    assert 0.0 <= diagnostics.attention_entropy_normalized_mean <= 1.0
    assert diagnostics.attention_entropy_nats_minimum >= 0.0
    assert diagnostics.attention_entropy_nats_maximum >= diagnostics.attention_entropy_nats_minimum


def test_persistent_scores_counts_and_exponent_ranking() -> None:
    state = TorchSignificanceState.empty(1, 1, device="cpu")
    state.append(torch.tensor([0, 1]), torch.tensor([10, 11]))
    state.update(torch.tensor([[[2.0, 4.0]]]))
    state.append(torch.tensor([2]), torch.tensor([12]))
    state.update(torch.tensor([[[1.0, 1.0, 9.0]]]))

    torch.testing.assert_close(state.scores, torch.tensor([[[3.0, 5.0, 9.0]]]))
    torch.testing.assert_close(state.counts, torch.tensor([[[2.0, 2.0, 1.0]]]))
    torch.testing.assert_close(state.ranked(0.0), state.scores)
    torch.testing.assert_close(state.ranked(1.0), torch.tensor([[[1.5, 2.5, 9.0]]]))
    assert state.matrix_bytes() == 3 * 4 * 2
    assert state.metadata_bytes() == 3 * 8 * 2


def test_corrected_count_gather_pads_unscored_tail_and_rejects_overrun() -> None:
    counts = torch.tensor([[[4.0, 2.0]]])
    indices = torch.tensor([[[0, 2]]])
    gathered = gather_counts(counts, indices, cache_length=3)
    torch.testing.assert_close(gathered, torch.tensor([[[4.0, 0.0]]]))
    with pytest.raises(RuntimeError, match="counts cover 3 entries but KV covers 2"):
        gather_counts(torch.ones(1, 1, 3), torch.tensor([[[0]]]), cache_length=2)


def test_sink_heavy_recent_selection_and_pooling_default_off() -> None:
    scores = torch.tensor([[[0.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 5.0, 6.0]]])
    policy = SignificancePolicy(budget=5, sinks=1, recent=1, count_exponent=1.0)
    assert policy.pool_width == 1
    selected = select_keep_indices(scores, policy)
    assert selected.tolist() == [[[0, 2, 4, 6, 9]]]


def test_runtime_ties_use_position_then_entry_id_deterministically() -> None:
    scores = torch.ones((1, 1, 6), dtype=torch.float32)
    positions = torch.tensor([[[0, 3, 1, 2, 4, 5]]])
    entry_ids = torch.tensor([[[0, 30, 10, 20, 40, 50]]])
    policy = SignificancePolicy(budget=4, sinks=1, recent=1)
    selected = select_keep_indices(
        scores,
        policy,
        positions=positions,
        entry_ids=entry_ids,
    )
    assert selected.tolist() == [[[0, 2, 3, 5]]]


def test_runtime_position_mode_selects_recent_content_with_same_budget() -> None:
    cache = SignificanceCache(num_hidden_layers=1)
    keys = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
    _append_cache(cache, keys, torch.arange(8))
    cache.layers[0].record(torch.ones(1, 1, 8))
    cache.compress(
        SignificancePolicy(
            budget=4,
            sinks=1,
            recent=1,
            ranking_mode="position",
        )
    )
    assert cache.layers[0].state.positions.tolist() == [[[0, 5, 6, 7]]]


def _append_cache(cache: SignificanceCache, keys: torch.Tensor, positions: torch.Tensor) -> None:
    cache.prepare_append(positions)
    cache.update(keys, keys + 1000, 0)
    cache.assert_forward_complete()


def test_cache_gather_keeps_kv_scores_counts_positions_and_ids_aligned() -> None:
    cache = SignificanceCache(num_hidden_layers=1)
    keys = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).expand(1, 2, 6, 1).clone()
    _append_cache(cache, keys, torch.arange(6))
    layer = cache.layers[0]
    layer.record(
        torch.tensor(
            [
                [
                    [0.0, 1.0, 10.0, 2.0, 3.0, 0.0],
                    [0.0, 1.0, 2.0, 10.0, 3.0, 0.0],
                ]
            ]
        )
    )
    cache.compress(SignificancePolicy(budget=3, sinks=1, recent=1, count_exponent=1.0))

    assert layer.state.entry_ids.tolist() == [[[0, 2, 5], [0, 3, 5]]]
    assert layer.state.positions.tolist() == [[[0, 2, 5], [0, 3, 5]]]
    assert layer.keys[..., 0].to(torch.long).tolist() == [[[0, 2, 5], [0, 3, 5]]]
    assert (layer.values[..., 0] - 1000).to(torch.long).tolist() == [[[0, 2, 5], [0, 3, 5]]]
    torch.testing.assert_close(layer.state.counts, torch.ones_like(layer.state.counts))
    assert cache.true_length == 6
    assert cache.stored_length() == 3

    new_keys = torch.tensor([6.0, 7.0]).view(1, 1, 2, 1).expand(1, 2, 2, 1).clone()
    _append_cache(cache, new_keys, torch.tensor([6, 7]))
    layer.record(torch.ones(1, 2, 5))
    assert layer.state.counts.tolist() == [[[2, 2, 2, 1, 1], [2, 2, 2, 1, 1]]]
    assert cache.true_length == 8
    assert cache.stored_length() == 5


def test_query_probe_structural_rollback_retains_only_prefix_score_update() -> None:
    cache = SignificanceCache(num_hidden_layers=1)
    keys = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)
    _append_cache(cache, keys, torch.arange(4))
    layer = cache.layers[0]
    layer.record(torch.ones(1, 1, 4))
    snapshot = cache.snapshot_structure(exact_tensors=True)

    query_keys = torch.tensor([4.0, 5.0]).view(1, 1, 2, 1)
    _append_cache(cache, query_keys, torch.tensor([4, 5]))
    layer.record(torch.tensor([[[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]]))
    cache.rollback_structure(snapshot, keep_score_updates=True, verify_exact=True)

    torch.testing.assert_close(layer.keys, keys)
    torch.testing.assert_close(layer.values, keys + 1000)
    torch.testing.assert_close(layer.state.scores, torch.tensor([[[11.0, 21.0, 31.0, 41.0]]]))
    torch.testing.assert_close(layer.state.counts, torch.full((1, 1, 4), 2.0))
    assert layer.state.positions.tolist() == [[[0, 1, 2, 3]]]
    assert layer.state.entry_ids.tolist() == [[[0, 1, 2, 3]]]
    assert cache.true_length == 4
    assert cache.stored_length() == 4
    assert layer.next_entry_id == 4


def test_fail_loudly_on_unaligned_state_and_unscored_compression() -> None:
    cache = SignificanceCache(num_hidden_layers=1)
    keys = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
    _append_cache(cache, keys, torch.arange(5))
    with pytest.raises(RuntimeError, match="compression requested with one or more unscored entries"):
        cache.compress(SignificancePolicy(budget=3, sinks=1, recent=1))

    cache.layers[0].state.positions = cache.layers[0].state.positions[..., :-1]
    with pytest.raises(RuntimeError, match="unaligned significance state"):
        cache.layers[0].assert_aligned()


def test_transformers_batch_lifecycle_keeps_semantic_state_aligned() -> None:
    cache = SignificanceCache(num_hidden_layers=1)
    keys = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1)
    _append_cache(cache, keys, torch.arange(3))
    cache.layers[0].record(torch.ones(1, 1, 3))

    cache.batch_repeat_interleave(2)
    assert cache.layers[0].keys.shape[0] == 2
    assert cache.layers[0].state.scores.shape[0] == 2
    cache.batch_select_indices(torch.tensor([1]))
    assert cache.layers[0].keys.shape[0] == 1
    cache.assert_aligned()
    with pytest.raises(RuntimeError, match=r"generic cache\.crop"):
        cache.crop(-1)

    cache.reset()
    assert cache.get_seq_length() == 0
    assert cache.stored_length() == 0
    assert cache.layers[0].state is None
