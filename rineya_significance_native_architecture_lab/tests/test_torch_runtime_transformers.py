from __future__ import annotations

import pytest
import torch

from rineya_significance_lab.runtime import (
    SignificanceCache,
    SignificancePolicy,
    chunked_prefill,
)


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
def test_tiny_qwen3_chunked_noop_matches_single_shot_all_positions() -> None:
    """Required B0 gate: chunking alone must not change the model trajectory."""

    model = _tiny_qwen3()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 11))
    with torch.no_grad():
        reference = model(input_ids, use_cache=True, logits_to_keep=0).logits.float()
    cache = SignificanceCache(config=model.config)
    chunked = chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=3,
        policy=None,
        collect_logits=True,
        logits_to_keep=0,
    ).logits

    assert chunked is not None and chunked.shape == reference.shape
    delta = (chunked - reference).abs()
    assert float(delta.max()) < 1e-4
    assert torch.equal(chunked.argmax(-1), reference.argmax(-1))


@pytest.mark.model
def test_tiny_qwen3_chunked_prefill_records_and_compresses() -> None:
    model = _tiny_qwen3()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    cache = SignificanceCache(config=model.config)
    result = chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=4,
        policy=SignificancePolicy(
            budget=4,
            sinks=1,
            recent=1,
            count_exponent=1.0,
        ),
        observation_window=2,
        collect_logits=True,
    )

    assert result.logits is not None and result.logits.shape == (1, 12, model.config.vocab_size)
    assert cache.true_length == 12
    # Two between-chunk compressions reduce the prefix to B; the final chunk is
    # deliberately left resident, matching the audited B + final-chunk contract.
    assert cache.stored_length() == 8
    assert len(result.compression_reports) == 2
    assert cache.scoring_events() == [3, 3]
    cache.assert_aligned()


@pytest.mark.model
def test_tiny_qwen3_query_probe_rolls_back_before_compression() -> None:
    model = _tiny_qwen3()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    query_ids = torch.randint(0, model.config.vocab_size, (1, 2))
    cache = SignificanceCache(config=model.config)
    chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=4,
        policy=SignificancePolicy(
            budget=3,
            sinks=1,
            recent=1,
            count_exponent=1.0,
        ),
        query_ids=query_ids,
        verify_query_rollback=True,
    )

    assert cache.true_length == 8
    assert cache.stored_length() == 7  # compressed B plus the final uncompressed chunk
    assert cache.scoring_events() == [1, 1]
    for layer in cache.layers:
        assert layer.next_entry_id == 8
        assert layer.state.entry_ids.max().item() == 7
    cache.assert_aligned()
