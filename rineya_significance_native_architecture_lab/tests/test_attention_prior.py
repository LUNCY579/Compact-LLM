from __future__ import annotations

import math

import pytest
import torch

from rineya_significance_lab.runtime import (
    AttentionPriorAccessDeclaration,
    AttentionPriorTrace,
    SignificanceAttentionPrior,
    SignificanceCache,
    SignificancePolicy,
    attention_prior_decode_append,
    chunked_prefill,
    merge_attention_prior_mask,
)
from rineya_significance_lab.torch_state import TorchSignificanceState


def _state_with_current_tail(*, batch: int = 1) -> TorchSignificanceState:
    state = TorchSignificanceState.empty(batch, 2, device="cpu")
    historical_positions = torch.arange(4).view(1, -1).expand(batch, -1)
    state.append(historical_positions, historical_positions)
    contribution = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 3.0, 2.0]]]
    ).expand(batch, -1, -1)
    state.update(contribution)
    current_positions = torch.arange(4, 6).view(1, -1).expand(batch, -1)
    state.append(current_positions, current_positions)
    return state


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
    torch.manual_seed(211)
    return Qwen3ForCausalLM(config).eval()


def _run_tiny(model, input_ids: torch.Tensor, prior: SignificanceAttentionPrior | None):
    cache = SignificanceCache(config=model.config)
    trace = AttentionPriorTrace() if prior is not None else None
    output = chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=4,
        policy=SignificancePolicy(budget=64, sinks=0, recent=0, count_exponent=1.0),
        observation_window=4,
        collect_logits=True,
        attention_prior=prior,
        prior_trace=trace,
    )
    assert output.logits is not None
    return output.logits, cache, trace


def test_prior_conditions_and_layer_lambdas_are_bounded() -> None:
    state = _state_with_current_tail()
    # Make S/C constant while raw S remains non-constant.
    state.scores[..., :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state.counts[..., :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = SignificanceAttentionPrior(mode="normalized", global_lambda=0.02).bias(
        state,
        layer_idx=0,
        query_heads=4,
        query_length=2,
        key_length=6,
    )
    raw = SignificanceAttentionPrior(mode="raw", global_lambda=0.02).bias(
        state,
        layer_idx=0,
        query_heads=4,
        query_length=2,
        key_length=6,
    )
    assert torch.count_nonzero(normalized.bias) == 0
    assert torch.count_nonzero(raw.bias) > 0

    control_state = _state_with_current_tail()
    control = SignificanceAttentionPrior(mode="normalized", global_lambda=0.01).bias(
        control_state,
        layer_idx=0,
        query_heads=4,
        query_length=2,
        key_length=6,
    )
    shuffled = SignificanceAttentionPrior(
        mode="shuffled",
        global_lambda=0.01,
        shuffle_seed=19,
    ).bias(
        control_state,
        layer_idx=0,
        query_heads=4,
        query_length=2,
        key_length=6,
    )
    # The deterministic control breaks token identity while preserving each
    # head's transformed-bias distribution exactly.
    assert torch.equal(
        control.bias[:, :, 0, :4].sort(dim=-1).values,
        shuffled.bias[:, :, 0, :4].sort(dim=-1).values,
    )
    age = SignificanceAttentionPrior(mode="age", global_lambda=0.01).bias(
        control_state,
        layer_idx=0,
        query_heads=4,
        query_length=2,
        key_length=6,
    )
    for result in (shuffled, age):
        assert torch.isfinite(result.bias).all()
        assert float(result.bias.abs().max()) <= 0.02 + 1.0e-7

    per_layer = SignificanceAttentionPrior(
        global_lambda=0.01,
        layer_lambdas=[0.0, 0.02],
    )
    assert per_layer.lambda_for_layer(0) == 0.0
    assert per_layer.lambda_for_layer(1) == 0.02
    assert per_layer.lambda_for_layer(5) == 0.01
    mapped = SignificanceAttentionPrior(
        global_lambda=0.01,
        layer_lambdas={"1": 0.03},  # type: ignore[dict-item]
    )
    assert mapped.lambda_for_layer(1) == 0.03
    with pytest.raises(ValueError, match="within"):
        SignificanceAttentionPrior(global_lambda=0.5)


def test_prior_gqa_padding_and_current_tail_are_never_biased() -> None:
    state = _state_with_current_tail()
    allowed = torch.tensor(
        [[[[True, False, True, True, True, False], [True, False, True, True, True, True]]]]
    )
    prior = SignificanceAttentionPrior(mode="normalized", global_lambda=0.02)
    result = prior.bias(
        state,
        layer_idx=0,
        query_heads=4,
        query_length=2,
        key_length=6,
        attention_mask=allowed,
        causal=True,
    )
    assert result.bias.shape == (1, 4, 2, 6)
    # GQA query-head groups inherit the matching KV-head prior.
    assert torch.equal(result.bias[:, 0], result.bias[:, 1])
    assert torch.equal(result.bias[:, 2], result.bias[:, 3])
    # Padding and every key appended by this forward stay at literal zero.
    assert torch.count_nonzero(result.bias[..., 1]) == 0
    assert torch.count_nonzero(result.bias[..., 4:]) == 0
    assert result.diagnostics.current_key_bias_abs_max == 0.0
    assert result.diagnostics.invalid_key_bias_abs_max == 0.0

    additive = torch.where(
        allowed,
        torch.tensor(0.0),
        torch.tensor(float("-inf")),
    )
    additive_result = prior.bias(
        state,
        layer_idx=0,
        query_heads=4,
        query_length=2,
        key_length=6,
        attention_mask=additive,
        causal=True,
    )
    assert torch.equal(additive_result.bias, result.bias)

    merged = merge_attention_prior_mask(
        allowed,
        result.bias,
        kv_heads=2,
        causal=True,
        dtype=torch.float32,
    )
    assert torch.isneginf(merged[..., 1]).all()
    assert torch.isneginf(merged[:, :, 0, 5]).all()
    assert torch.isfinite(merged[:, :, 1, 0]).all()
    implicit_causal = merge_attention_prior_mask(
        None,
        result.bias,
        kv_heads=2,
        causal=True,
        dtype=torch.float32,
    )
    assert torch.isneginf(implicit_causal[:, :, 0, 5]).all()
    assert torch.isfinite(implicit_causal[:, :, 1, 5]).all()


def test_access_declaration_prohibits_answer_or_future_information() -> None:
    declaration = AttentionPriorAccessDeclaration(query_access="conditioned")
    assert declaration.to_dict() == {
        "query_access": "conditioned",
        "answer_access": "none",
        "oracle_evidence_access": "none",
        "future_token_access": "none",
        "future_hidden_state_access": "none",
    }
    with pytest.raises(ValueError, match="forbids"):
        AttentionPriorAccessDeclaration(query_access="agnostic", answer_access="tokens")  # type: ignore[arg-type]


def test_query_probe_rejects_an_agnostic_prior_declaration_before_execution() -> None:
    cache = SignificanceCache(num_hidden_layers=1)
    prior = SignificanceAttentionPrior(
        global_lambda=0.01,
        access=AttentionPriorAccessDeclaration(query_access="agnostic"),
    )
    with pytest.raises(ValueError, match="query-conditioned state"):
        chunked_prefill(
            object(),
            torch.tensor([[1, 2]]),
            cache,
            chunk_size=1,
            policy=SignificancePolicy(budget=2),
            query_ids=torch.tensor([[3]]),
            attention_prior=prior,
        )


@pytest.mark.model
def test_tiny_qwen_zero_lambda_is_bit_exact_identity() -> None:
    model = _tiny_qwen3()
    input_ids = torch.tensor([[3, 7, 11, 19, 23, 29, 31, 37]])
    with torch.no_grad():
        standard = model(input_ids, use_cache=False, logits_to_keep=0).logits.float()
    plain_cache = SignificanceCache(config=model.config)
    plain = chunked_prefill(
        model,
        input_ids,
        plain_cache,
        chunk_size=4,
        policy=None,
        collect_logits=True,
    ).logits
    assert plain is not None
    baseline, baseline_cache, _ = _run_tiny(model, input_ids, None)
    c0 = SignificanceAttentionPrior(mode="normalized", global_lambda=0.0)
    zero_logits, zero_cache, trace = _run_tiny(model, input_ids, c0)

    assert torch.equal(zero_logits, baseline)
    assert torch.equal(zero_logits, plain)
    assert torch.equal(zero_logits, standard)
    for base_layer, zero_layer in zip(baseline_cache.layers, zero_cache.layers, strict=True):
        assert torch.equal(base_layer.keys, zero_layer.keys)
        assert torch.equal(base_layer.values, zero_layer.values)
        assert torch.equal(base_layer.state.scores, zero_layer.state.scores)
        assert torch.equal(base_layer.state.counts, zero_layer.state.counts)
    for plain_layer, zero_layer in zip(plain_cache.layers, zero_cache.layers, strict=True):
        assert torch.equal(plain_layer.keys, zero_layer.keys)
        assert torch.equal(plain_layer.values, zero_layer.values)
        assert torch.equal(plain_layer.state.positions, zero_layer.state.positions)
        assert torch.equal(plain_layer.state.entry_ids, zero_layer.state.entry_ids)
    assert trace is not None
    assert all(row.zero_lambda_bypass for row in trace.applications)
    assert all(not row.nonzero_bias_applied for row in trace.applications)


@pytest.mark.model
def test_tiny_qwen_nonzero_prior_has_no_future_access_and_finite_feedback_diagnostics() -> None:
    model = _tiny_qwen3()
    first = torch.tensor([[3, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53]])
    changed_future = first.clone()
    changed_future[:, 5:8] = torch.tensor([59, 61, 67])
    prior = SignificanceAttentionPrior(mode="normalized", global_lambda=0.02)

    _, baseline_cache, _ = _run_tiny(model, first, None)
    logits_a, cache_a, trace_a = _run_tiny(model, first, prior)
    logits_b, _, trace_b = _run_tiny(model, changed_future, prior)
    # The first row of chunk two can read positions <=4 only.  Altering later
    # tokens cannot affect it even though the prior turns the SDPA mask explicit.
    assert torch.equal(logits_a[:, 4], logits_b[:, 4])
    assert torch.isfinite(logits_a).all() and torch.isfinite(logits_b).all()
    assert cache_a.scoring_events() == [3, 3]
    assert any(
        not torch.equal(baseline_layer.state.scores, prior_layer.state.scores)
        for baseline_layer, prior_layer in zip(
            baseline_cache.layers,
            cache_a.layers,
            strict=True,
        )
    )
    assert trace_a is not None and trace_b is not None
    assert any(row.nonzero_bias_applied for row in trace_a.applications)
    assert all(row.attention_entropy_normalized_mean is not None for row in trace_a.applications)
    assert {row.forward_role for row in trace_a.applications} == {"context_scoring"}
    assert all(row.bias_abs_max <= 0.04 + 1.0e-7 for row in trace_a.applications)
    summary = trace_a.to_dict()
    assert set(summary) == {"applications", "layers"}
    assert set(summary["layers"]) == {"0", "1"}  # type: ignore[arg-type]
    for row in trace_a.applications:
        for value in (
            row.bias_min,
            row.bias_max,
            row.bias_abs_max,
            row.current_key_bias_abs_max,
            row.invalid_key_bias_abs_max,
            row.clipped_fraction,
            row.source_entropy_normalized,
            row.prior_entropy_normalized,
            row.old_minus_recent_bias,
        ):
            assert math.isfinite(value)


@pytest.mark.model
def test_tiny_qwen_per_layer_lambdas_control_the_sdpa_integration() -> None:
    model = _tiny_qwen3()
    prior = SignificanceAttentionPrior(
        global_lambda=0.0,
        layer_lambdas=[0.0, 0.02],
    )
    _, _, trace = _run_tiny(
        model,
        torch.tensor([[5, 7, 11, 13, 17, 19, 23, 29]]),
        prior,
    )
    assert trace is not None
    layer_zero = [row for row in trace.applications if row.layer_idx == 0]
    layer_one = [row for row in trace.applications if row.layer_idx == 1]
    assert layer_zero and all(row.zero_lambda_bypass for row in layer_zero)
    assert layer_one and any(row.nonzero_bias_applied for row in layer_one)


@pytest.mark.model
def test_tiny_qwen_declared_query_conditioning_applies_after_exact_probe_rollback() -> None:
    model = _tiny_qwen3()
    cache = SignificanceCache(config=model.config)
    trace = AttentionPriorTrace()
    prior = SignificanceAttentionPrior(
        global_lambda=0.02,
        access=AttentionPriorAccessDeclaration(query_access="conditioned"),
    )
    chunked_prefill(
        model,
        torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19]]),
        cache,
        chunk_size=4,
        policy=SignificancePolicy(budget=16, sinks=0, recent=0, count_exponent=1.0),
        query_ids=torch.tensor([[23, 29]]),
        verify_query_rollback=True,
        attention_prior=prior,
        prior_trace=trace,
    )
    assert cache.true_length == 8
    assert cache.stored_length() == 8
    assert cache.scoring_events() == [1, 1]
    assert any(row.nonzero_bias_applied for row in trace.applications)
    assert all(row.current_key_bias_abs_max == 0.0 for row in trace.applications)
    for layer in cache.layers:
        assert layer.state.entry_ids.max().item() == 7
    output = attention_prior_decode_append(
        model,
        cache,
        torch.tensor([[31]]),
        attention_prior=prior,
        prior_trace=trace,
    )
    assert torch.isfinite(output.logits).all()
    assert cache.true_length == 9
    assert cache.scoring_events() == [1, 1]
    assert any(row.forward_role == "decode" for row in trace.applications)
