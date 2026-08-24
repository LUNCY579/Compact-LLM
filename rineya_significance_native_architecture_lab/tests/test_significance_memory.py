from __future__ import annotations

import inspect
import math

import pytest
import torch

from rineya_significance_lab.runtime import (
    MEMORY_VARIANTS,
    SignificanceCache,
    SignificanceMemory,
    SignificancePolicy,
    augment_attention_with_memory,
    chunked_prefill,
    chunked_prefill_with_memory,
    compress_cache_with_memory,
    match_memory_total_budget,
    memory_decode_append,
    rectangular_position_bytes,
    score_with_query_memory,
)


def _memory(
    *,
    slots: int,
    variant: str = "mean",
    layers: int = 1,
    batch: int = 1,
    heads: int = 1,
    dim: int = 1,
    dtype: torch.dtype = torch.float32,
    seed: int = 7,
) -> SignificanceMemory:
    return SignificanceMemory(
        num_layers=layers,
        batch_size=batch,
        num_kv_heads=heads,
        num_slots=slots,
        head_dim=dim,
        variant=variant,  # type: ignore[arg-type]
        device="cpu",
        kv_dtype=dtype,
        recency_tau=2.0,
        random_seed=seed,
    )


def _write_two(memory: SignificanceMemory) -> None:
    keys = torch.tensor([[[[1.0], [3.0]]]], dtype=memory.kv_dtype)
    memory.compress_evicted(
        layer_index=0,
        keys=keys,
        values=keys + 10,
        scores=torch.tensor([[[1.0, 3.0]]]),
        counts=torch.tensor([[[1.0, 3.0]]]),
        positions=torch.tensor([[[0, 3]]]),
        entry_ids=torch.tensor([[[0, 1]]]),
        boundary_position=4,
    )


def test_zero_slots_are_exactly_empty_and_have_zero_bytes() -> None:
    memory = _memory(slots=0, layers=2, heads=2, dim=3)
    ledger = memory.byte_ledger()
    assert ledger.slot_capacity == 0
    assert ledger.valid_slots == 0
    assert ledger.total_bytes == 0
    for layer in range(2):
        state = memory.retrieve_slots(layer)
        assert state.keys.shape == (1, 2, 0, 3)
        assert state.valid.shape == (1, 2, 0)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("mean", 2.0),
        (
            "recency",
            (math.exp(-2.0) * 1.0 + math.exp(-0.5) * 3.0) / (math.exp(-2.0) + math.exp(-0.5)),
        ),
        ("raw_significance", 2.5),
        ("normalized_significance", 2.0),
    ],
)
def test_declared_weighting_variants_aggregate_post_rope_kv(
    variant: str,
    expected: float,
) -> None:
    memory = _memory(slots=1, variant=variant)
    _write_two(memory)
    state = memory.retrieve_slots(0)
    torch.testing.assert_close(state.keys[0, 0, 0, 0], torch.tensor(expected), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        state.values[0, 0, 0, 0],
        torch.tensor(expected + 10),
        atol=1e-6,
        rtol=1e-6,
    )
    assert state.valid.tolist() == [[[True]]]
    assert state.source_positions.tolist() == [[[3]]]
    assert state.created_at.tolist() == [[[4]]]
    assert state.last_updated_at.tolist() == [[[4]]]
    assert state.source_counts.tolist() == [[[2]]]


def test_random_control_is_deterministic_and_seeded() -> None:
    first = _memory(slots=1, variant="random", seed=19)
    replay = _memory(slots=1, variant="random", seed=19)
    different = _memory(slots=1, variant="random", seed=20)
    for memory in (first, replay, different):
        _write_two(memory)
    assert first.state_sha256() == replay.state_sha256()
    assert first.state_sha256() != different.state_sha256()
    assert torch.isfinite(first.retrieve_slots(0).keys).all()


def test_all_required_unlearned_variants_are_exposed() -> None:
    assert set(MEMORY_VARIANTS) == {
        "mean",
        "recency",
        "raw_significance",
        "normalized_significance",
        "random",
    }


def test_stable_entry_ids_assign_sources_to_exact_fixed_slots() -> None:
    memory = _memory(slots=2)
    keys = torch.tensor([[[[0.0], [1.0], [2.0], [3.0]]]])
    memory.compress_evicted(
        layer_index=0,
        keys=keys,
        values=keys + 10,
        scores=torch.ones(1, 1, 4),
        counts=torch.ones(1, 1, 4),
        positions=torch.tensor([[[0, 1, 2, 3]]]),
        entry_ids=torch.tensor([[[0, 1, 2, 3]]]),
        boundary_position=4,
    )
    state = memory.retrieve_slots(0)
    assert state.keys[..., 0].tolist() == [[[1.0, 2.0]]]
    assert state.source_counts.tolist() == [[[2, 2]]]
    assert state.valid.tolist() == [[[True, True]]]


def test_mean_memory_merges_new_evictions_without_growing_slot_count() -> None:
    memory = _memory(slots=1)
    _write_two(memory)
    memory.compress_evicted(
        layer_index=0,
        keys=torch.tensor([[[[5.0]]]]),
        values=torch.tensor([[[[15.0]]]]),
        scores=torch.ones(1, 1, 1),
        counts=torch.ones(1, 1, 1),
        positions=torch.tensor([[[5]]]),
        entry_ids=torch.tensor([[[2]]]),
        boundary_position=6,
    )
    state = memory.retrieve_slots(0)
    assert state.keys.shape == (1, 1, 1, 1)
    torch.testing.assert_close(state.keys, torch.tensor([[[[3.0]]]]))
    assert state.created_at.tolist() == [[[4]]]
    assert state.last_updated_at.tolist() == [[[6]]]
    assert state.source_positions.tolist() == [[[5]]]
    assert state.source_counts.tolist() == [[[3]]]


def test_byte_ledger_counts_every_persistent_tensor() -> None:
    memory = _memory(slots=2, layers=2, heads=2, dim=3)
    ledger = memory.byte_ledger()
    assert ledger.slot_capacity == 8
    assert ledger.key_bytes == 8 * 3 * 4
    assert ledger.value_bytes == 8 * 3 * 4
    assert ledger.validity_bytes == 8
    assert ledger.source_position_bytes == 8 * 8
    assert ledger.created_at_bytes == 8 * 8
    assert ledger.last_updated_at_bytes == 8 * 8
    assert ledger.source_count_bytes == 8 * 8
    assert ledger.weight_mass_bytes == 8 * 4
    assert ledger.total_bytes == 488


def test_total_byte_match_reduces_halo_m4_budget_by_five_with_discrete_slack() -> None:
    unit_bytes = rectangular_position_bytes(
        num_layers=28,
        batch_size=1,
        num_kv_heads=8,
        head_dim=128,
        kv_dtype=torch.bfloat16,
    )
    assert unit_bytes == 120_064
    matched = match_memory_total_budget(
        baseline_ordinary_budget=128,
        ordinary_bytes_per_budget_unit=unit_bytes,
        memory_bytes=491_904,
        minimum_ordinary_budget=37,
    )
    assert matched.baseline_ordinary_budget == 128
    assert matched.candidate_ordinary_budget == 123
    assert matched.ordinary_budget_reduction == 5
    assert matched.baseline_total_bytes == 15_368_192
    assert matched.candidate_total_bytes == 15_259_776
    assert matched.discrete_slack_bytes == 108_416
    assert matched.candidate_total_bytes <= matched.baseline_total_bytes
    assert matched.discrete_slack_bytes < unit_bytes


def test_write_api_has_no_oracle_evidence_or_answer_channel() -> None:
    parameters = set(inspect.signature(SignificanceMemory.compress_evicted).parameters)
    assert parameters == {
        "self",
        "layer_index",
        "keys",
        "values",
        "scores",
        "counts",
        "positions",
        "entry_ids",
        "boundary_position",
    }
    memory = _memory(slots=1)
    with pytest.raises(TypeError, match="unexpected keyword argument 'evidence_labels'"):
        memory.compress_evicted(  # type: ignore[call-arg]
            layer_index=0,
            keys=torch.ones(1, 1, 1, 1),
            values=torch.ones(1, 1, 1, 1),
            scores=torch.ones(1, 1, 1),
            counts=torch.ones(1, 1, 1),
            positions=torch.zeros(1, 1, 1, dtype=torch.long),
            entry_ids=torch.zeros(1, 1, 1, dtype=torch.long),
            boundary_position=1,
            evidence_labels=[0],
        )


def test_future_or_unscored_inputs_are_rejected() -> None:
    memory = _memory(slots=1)
    common = {
        "layer_index": 0,
        "keys": torch.ones(1, 1, 1, 1),
        "values": torch.ones(1, 1, 1, 1),
        "scores": torch.ones(1, 1, 1),
        "counts": torch.ones(1, 1, 1),
        "entry_ids": torch.zeros(1, 1, 1, dtype=torch.long),
        "boundary_position": 1,
    }
    with pytest.raises(ValueError, match="precede the write boundary"):
        memory.compress_evicted(
            **common,
            positions=torch.ones(1, 1, 1, dtype=torch.long),
        )
    common["positions"] = torch.zeros(1, 1, 1, dtype=torch.long)
    common["counts"] = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="counts strictly positive"):
        memory.compress_evicted(**common)


def _append(cache: SignificanceCache, keys: torch.Tensor, positions: torch.Tensor) -> None:
    cache.prepare_append(positions)
    cache.update(keys, keys + 100, 0)
    cache.assert_forward_complete()


def _prepared_cache() -> SignificanceCache:
    cache = SignificanceCache(num_hidden_layers=1)
    keys = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).expand(1, 2, 6, 1).clone()
    _append(cache, keys, torch.arange(6))
    cache.layers[0].record(
        torch.tensor([[[0.0, 1.0, 10.0, 2.0, 3.0, 0.0], [0.0, 1.0, 2.0, 10.0, 3.0, 0.0]]])
    )
    return cache


def _assert_cache_exact(left: SignificanceCache, right: SignificanceCache) -> None:
    assert left.true_length == right.true_length
    assert left.stored_length() == right.stored_length()
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        for name in ("keys", "values"):
            assert torch.equal(getattr(left_layer, name), getattr(right_layer, name))
        for name in ("scores", "counts", "positions", "entry_ids"):
            assert torch.equal(getattr(left_layer.state, name), getattr(right_layer.state, name))
        assert left_layer.evictions == right_layer.evictions


def test_m0_compression_is_exact_hard_eviction_identity() -> None:
    hard = _prepared_cache()
    zero = _prepared_cache()
    policy = SignificancePolicy(budget=3, sinks=1, recent=1, count_exponent=1.0)
    hard_report = hard.compress(policy)
    memory = _memory(slots=0, layers=1, heads=2)
    zero_report = compress_cache_with_memory(zero, policy, memory)
    assert zero_report.compression == hard_report
    assert zero_report.writes == ()
    assert memory.byte_ledger().total_bytes == 0
    _assert_cache_exact(hard, zero)


def test_evicted_complement_writes_then_ordinary_append_stays_aligned() -> None:
    cache = _prepared_cache()
    memory = _memory(slots=2, layers=1, heads=2)
    result = compress_cache_with_memory(
        cache,
        SignificancePolicy(budget=3, sinks=1, recent=1, count_exponent=1.0),
        memory,
    )
    assert result.compression.evictions == 3
    # Head 0 retains IDs 0,2,5 and head 1 retains 0,3,5.  Only the
    # complementary model-state IDs enter the two memory heads.
    assert memory.retrieve_slots(0).source_counts.tolist() == [[[1, 2], [2, 1]]]
    digest = memory.state_sha256()
    new_keys = torch.tensor([6.0, 7.0]).view(1, 1, 2, 1).expand(1, 2, 2, 1).clone()
    _append(cache, new_keys, torch.tensor([6, 7]))
    assert cache.true_length == 8
    assert cache.stored_length() == 5
    assert memory.state_sha256() == digest
    cache.assert_aligned()
    memory.assert_valid()


def test_gqa_mask_repeats_per_kv_head_validity_and_preserves_causal_tail() -> None:
    memory = _memory(slots=2, heads=2, dim=4)
    keys = torch.arange(8, dtype=torch.float32).view(1, 2, 1, 4)
    memory.compress_evicted(
        layer_index=0,
        keys=keys,
        values=keys + 10,
        scores=torch.ones(1, 2, 1),
        counts=torch.ones(1, 2, 1),
        positions=torch.zeros(1, 2, 1, dtype=torch.long),
        entry_ids=torch.tensor([[[0], [1]]]),
        boundary_position=1,
    )
    query = torch.randn(1, 4, 2, 4)
    ordinary_key = torch.randn(1, 2, 3, 4)
    ordinary_value = torch.randn(1, 2, 3, 4)
    ordinary_mask = torch.tensor([[[[0.0, 0.0, torch.finfo(torch.float32).min], [0.0, 0.0, 0.0]]]])
    key, value, mask = augment_attention_with_memory(
        query,
        ordinary_key,
        ordinary_value,
        ordinary_mask,
        memory.retrieve_slots(0),
    )
    assert key.shape == value.shape == (1, 2, 5, 4)
    assert mask is not None and mask.shape == (1, 4, 2, 5)
    allowed = mask == 0
    # Q heads 0/1 share KV head 0; Q heads 2/3 share KV head 1.
    assert allowed[0, 0, 0, :2].tolist() == [True, False]
    assert allowed[0, 1, 0, :2].tolist() == [True, False]
    assert allowed[0, 2, 0, :2].tolist() == [False, True]
    assert allowed[0, 3, 0, :2].tolist() == [False, True]
    assert torch.equal(mask[..., 2:], ordinary_mask.expand(1, 4, 2, 3))


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
    torch.manual_seed(11)
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.model
def test_tiny_qwen_m0_runtime_exactly_matches_hard_eviction() -> None:
    model = _tiny_qwen3()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    policy = SignificancePolicy(budget=4, sinks=1, recent=1, count_exponent=1.0)
    hard_cache = SignificanceCache(config=model.config)
    hard = chunked_prefill(
        model,
        input_ids,
        hard_cache,
        chunk_size=4,
        policy=policy,
        observation_window=2,
        collect_logits=True,
    )
    zero_cache = SignificanceCache(config=model.config)
    zero_memory = _memory(slots=0, layers=2, heads=2, dim=16)
    zero = chunked_prefill_with_memory(
        model,
        input_ids,
        zero_cache,
        zero_memory,
        chunk_size=4,
        policy=policy,
        observation_window=2,
        collect_logits=True,
    )
    assert hard.logits is not None and zero.logits is not None
    assert torch.equal(hard.logits, zero.logits)
    _assert_cache_exact(hard_cache, zero_cache)
    assert zero.attention_accounting.attention_calls == 0
    assert zero_memory.byte_ledger().total_bytes == 0


@pytest.mark.model
def test_tiny_qwen_m4_is_finite_closed_loop_and_probe_is_read_only() -> None:
    model = _tiny_qwen3()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    query_ids = torch.randint(0, model.config.vocab_size, (1, 2))
    cache = SignificanceCache(config=model.config)
    memory = _memory(
        slots=4,
        variant="normalized_significance",
        layers=2,
        heads=2,
        dim=16,
    )
    result = chunked_prefill_with_memory(
        model,
        input_ids,
        cache,
        memory,
        chunk_size=4,
        policy=SignificancePolicy(budget=4, sinks=1, recent=1, count_exponent=1.0),
        query_ids=query_ids,
        verify_query_rollback=True,
        collect_logits=True,
    )
    assert result.logits is not None and torch.isfinite(result.logits).all()
    assert cache.true_length == 12
    assert cache.stored_length() == 8
    assert memory.byte_ledger().slot_capacity == 16
    assert memory.byte_ledger().valid_slots > 0
    assert result.attention_accounting.calls_with_valid_slots > 0
    assert result.attention_accounting.max_valid_slots_per_batch_head <= 4
    before = memory.state_sha256()
    cache_true = cache.true_length
    cache_stored = cache.stored_length()
    score_with_query_memory(
        model,
        cache,
        memory,
        query_ids,
        verify_rollback=True,
    )
    assert memory.state_sha256() == before
    assert cache.true_length == cache_true
    assert cache.stored_length() == cache_stored
    next_id = result.output.logits[:, -1].argmax(dim=-1, keepdim=True)
    decode = memory_decode_append(model, next_id, cache, memory, logits_to_keep=1)
    assert torch.isfinite(decode.output.logits).all()
    assert decode.attention_accounting.calls_with_valid_slots == 2
    assert memory.state_sha256() == before
    assert cache.true_length == cache_true + 1
    assert cache.stored_length() == cache_stored + 1
    cache.assert_aligned()
    memory.assert_valid()
