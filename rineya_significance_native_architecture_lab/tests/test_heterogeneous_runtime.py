from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from rineya_significance_lab.runtime import (
    HeterogeneousSignificanceCache,
    SignificanceCache,
    SignificancePolicy,
    allocate_live_budget,
    bottom_right_causal_mask,
    chunked_prefill,
    heterogeneous_bottom_right_mask,
    heterogeneous_chunked_prefill,
    heterogeneous_decode_append,
    heterogeneous_score_with_query,
)


def _append_all_layers(
    cache: HeterogeneousSignificanceCache,
    keys_by_layer: list[torch.Tensor],
    positions: torch.Tensor,
) -> None:
    cache.prepare_append(positions)
    adapter = cache.attention_adapter() if cache.attention_adapter_required() else nullcontext()
    with adapter:
        for layer_idx, keys in enumerate(keys_by_layer):
            cache.update(keys, keys + 1000, layer_idx)
    cache.assert_forward_complete()


def _scored_two_layer_cache() -> HeterogeneousSignificanceCache:
    cache = HeterogeneousSignificanceCache(num_hidden_layers=2)
    base = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).expand(1, 2, 6, 1)
    _append_all_layers(cache, [base.clone(), (base + 100).clone()], torch.arange(6))
    cache.layers[0].record(
        torch.tensor(
            [
                [
                    [1.0, 9.0, 2.0, 8.0, 3.0, 7.0],
                    [1.0, 2.0, 9.0, 3.0, 8.0, 7.0],
                ]
            ]
        )
    )
    cache.layers[1].record(
        torch.tensor(
            [
                [
                    [9.0, 1.0, 8.0, 2.0, 7.0, 3.0],
                    [1.0, 8.0, 2.0, 7.0, 3.0, 9.0],
                ]
            ]
        )
    )
    return cache


def test_fixed_global_unequal_layer_head_storage_and_exact_bytes() -> None:
    cache = _scored_two_layer_cache()
    policy = SignificancePolicy(budget=3, sinks=0, recent=0, count_exponent=1.0)
    allocations = torch.tensor([[2, 3], [3, 4]])
    report = cache.compress(policy, allocations, global_budget=12)

    assert report.allocations == ((2, 3), (3, 4))
    assert report.before_logical_entries == 24
    assert report.after_logical_entries == 12
    assert report.evicted_logical_entries == 12
    assert cache.layer_physical_lengths() == (3, 4)
    assert cache.logical_entries_per_sequence() == 12
    assert cache.physical_slots_per_sequence() == 14
    assert cache.layers[0].logical_counts().tolist() == [[2, 3]]
    assert cache.layers[1].logical_counts().tolist() == [[3, 4]]
    assert cache.layers[0].validity.tolist() == [[[True, True, False], [True] * 3]]
    cache.assert_aligned()

    ledger = cache.byte_accounting()
    assert ledger.logical_entries_per_sequence == 12
    assert ledger.logical_kv_bytes == 96
    assert ledger.physical_kv_bytes == 112
    assert ledger.logical_matrix_bytes == 96
    assert ledger.physical_matrix_bytes == 112
    assert ledger.logical_metadata_bytes == 192
    assert ledger.physical_metadata_bytes == 224
    assert ledger.validity_mask_bytes == 14
    assert ledger.logical_payload_bytes == 384
    assert ledger.physical_total_bytes == 462
    assert ledger.representation_overhead_bytes == 78

    cache.prepare_append(torch.tensor([6]))
    next_key = torch.ones((1, 2, 1, 1))
    with pytest.raises(RuntimeError, match="require heterogeneous_transformers_sdpa"):
        cache.update(next_key, next_key, 0)


def test_live_entropy_allocation_is_closed_loop_unequal_and_conserved() -> None:
    cache = HeterogeneousSignificanceCache(num_hidden_layers=2)
    base = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).expand(1, 2, 6, 1)
    _append_all_layers(cache, [base.clone(), base.clone()], torch.arange(6))
    cache.layers[0].record(
        torch.tensor([[[1.0] * 6, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    )
    cache.layers[1].record(
        torch.tensor([[[1.0] * 6, [90.0, 1.0, 0.0, 0.0, 0.0, 0.0]]])
    )

    decision = allocate_live_budget(
        cache,
        strategy="entropy_proportional",
        global_budget=12,
        minimum_per_unit=2,
        maximum_per_unit=4,
        exponent=1.0,
    )
    assert int(decision.budget.allocations.sum()) == 12
    assert len(set(decision.budget.allocations.ravel().tolist())) > 1
    report = cache.compress(
        SignificancePolicy(budget=3, sinks=1, recent=1, count_exponent=1.0),
        decision.torch_allocations(device="cpu"),
        global_budget=12,
    )
    assert report.after_logical_entries == 12
    cache.assert_aligned()


def test_validity_mask_covers_gqa_padding_causality_and_layer_remap() -> None:
    cache = _scored_two_layer_cache()
    cache.compress(
        SignificancePolicy(budget=3, sinks=0, recent=0),
        torch.tensor([[2, 3], [3, 4]]),
        global_budget=12,
    )
    new = torch.tensor([6.0, 7.0]).view(1, 1, 2, 1).expand(1, 2, 2, 1)
    _append_all_layers(cache, [new.clone(), new.clone()], torch.tensor([6, 7]))
    query = torch.zeros((1, 4, 2, 1))
    common = bottom_right_causal_mask(2, 6, device="cpu", dtype=torch.float32)

    layer0 = cache.layers[0]
    mask0 = heterogeneous_bottom_right_mask(layer0, query, layer0.keys, common)
    assert mask0.shape == (1, 4, 2, 5)
    allowed0 = mask0 == 0
    # Query heads 0/1 share KV head 0, whose old padding slots 2/3 are absent.
    assert allowed0[0, 0].tolist() == [
        [True, True, False, True, False],
        [True, True, False, True, True],
    ]
    assert torch.equal(allowed0[0, 0], allowed0[0, 1])
    # Query heads 2/3 share the fully populated KV head 1.
    assert allowed0[0, 2].tolist() == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]
    assert torch.equal(allowed0[0, 2], allowed0[0, 3])

    layer1 = cache.layers[1]
    mask1 = heterogeneous_bottom_right_mask(layer1, query, layer1.keys, common)
    # Layer 1 has the common physical width, but KV head 0 contains one padded
    # old slot. Its GQA group must exclude that slot without affecting head 1.
    assert mask1.shape == (1, 4, 2, 6)
    assert (mask1[0, 0] == 0).tolist() == [
        [True, True, True, False, True, False],
        [True, True, True, False, True, True],
    ]
    assert (mask1[0, 2] == 0).tolist() == [
        [True, True, True, True, True, False],
        [True, True, True, True, True, True],
    ]


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(29)
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.model
def test_tiny_qwen_uniform_allocation_is_exact_rectangular_identity() -> None:
    model = _tiny_qwen3()
    generator = torch.Generator().manual_seed(31)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 9), generator=generator)
    policy = SignificancePolicy(
        budget=3,
        sinks=1,
        recent=1,
        count_exponent=1.0,
    )
    rectangular = SignificanceCache(config=model.config)
    reference = chunked_prefill(
        model,
        input_ids,
        rectangular,
        chunk_size=3,
        policy=policy,
        observation_window=2,
        collect_logits=True,
    )
    heterogeneous = HeterogeneousSignificanceCache(config=model.config)
    candidate = heterogeneous_chunked_prefill(
        model,
        input_ids,
        heterogeneous,
        chunk_size=3,
        policy=policy,
        global_budget=12,
        allocation_strategy="uniform",
        minimum_per_unit=1,
        maximum_per_unit=3,
        observation_window=2,
        collect_logits=True,
    )

    assert reference.logits is not None and candidate.logits is not None
    assert torch.equal(candidate.logits, reference.logits)
    assert [report.allocations for report in candidate.compression_reports] == [
        ((3, 3), (3, 3)),
        ((3, 3), (3, 3)),
    ]
    for rectangular_layer, heterogeneous_layer in zip(
        rectangular.layers, heterogeneous.layers, strict=True
    ):
        assert torch.equal(heterogeneous_layer.keys, rectangular_layer.keys)
        assert torch.equal(heterogeneous_layer.values, rectangular_layer.values)
        assert torch.equal(heterogeneous_layer.state.scores, rectangular_layer.state.scores)
        assert torch.equal(heterogeneous_layer.state.counts, rectangular_layer.state.counts)
        assert torch.equal(heterogeneous_layer.state.positions, rectangular_layer.state.positions)
        assert torch.equal(heterogeneous_layer.state.entry_ids, rectangular_layer.state.entry_ids)
        assert bool(heterogeneous_layer.validity.all().item())


@pytest.mark.model
def test_tiny_qwen_unequal_heads_layers_query_rollback_and_decode_append() -> None:
    model = _tiny_qwen3()
    generator = torch.Generator().manual_seed(37)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 10), generator=generator)
    cache = HeterogeneousSignificanceCache(config=model.config)

    def unequal_callback(cache, policy, chunk_index):
        assert chunk_index == 0
        return torch.tensor([[2, 3], [3, 4]])

    result = heterogeneous_chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=5,
        policy=SignificancePolicy(budget=3, sinks=1, recent=1, count_exponent=1.0),
        global_budget=12,
        allocation_callback=unequal_callback,
        observation_window=2,
        collect_logits=True,
    )
    assert result.logits is not None and result.logits.shape == (1, 10, 128)
    assert cache.true_length == 10
    assert cache.layer_physical_lengths() == (8, 9)
    assert cache.logical_entries_per_sequence() == 32
    assert cache.physical_slots_per_sequence() == 34
    cache.assert_aligned()

    before_keys = [layer.keys.clone() for layer in cache.layers]
    before_values = [layer.values.clone() for layer in cache.layers]
    before_lengths = cache.layer_physical_lengths()
    before_true = cache.true_length
    before_events = cache.scoring_events()
    query = torch.randint(0, model.config.vocab_size, (1, 2), generator=generator)
    heterogeneous_score_with_query(model, cache, query, verify_rollback=True)
    assert cache.layer_physical_lengths() == before_lengths
    assert cache.true_length == before_true
    assert cache.scoring_events() == [value + 1 for value in before_events]
    for layer_idx, layer in enumerate(cache.layers):
        assert torch.equal(layer.keys, before_keys[layer_idx])
        assert torch.equal(layer.values, before_values[layer_idx])
    cache.assert_aligned()

    decode_id = torch.randint(0, model.config.vocab_size, (1, 1), generator=generator)
    decoded = heterogeneous_decode_append(
        model,
        decode_id,
        cache,
        recording_window=1,
        logits_to_keep=1,
    )
    assert decoded.logits.shape == (1, 1, 128)
    assert cache.true_length == 11
    assert cache.layer_physical_lengths() == (9, 10)
    assert cache.logical_entries_per_sequence() == 36
    assert cache.physical_slots_per_sequence() == 38
    cache.assert_aligned()
