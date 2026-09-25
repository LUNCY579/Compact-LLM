from __future__ import annotations

import math

import pytest
import torch

from experiments.gflowrl_reasoning_v1.objective import (
    GFlowRLConfig,
    RolloutBatch,
    compute_gflowrl_loss,
    compute_objective,
)


def make_batch(
    current=None,
    old=None,
    reference=None,
    rewards=None,
    lengths=None,
    valid=None,
):
    current = torch.tensor(current or [[-0.4, -0.2, -0.1, -0.3]], dtype=torch.float64)
    current.requires_grad_(True)
    shape = current.shape
    return RolloutBatch(
        current,
        torch.tensor(old or [[-0.5, -0.4, -0.2, -0.25]], dtype=torch.float64),
        torch.tensor(reference or [[-0.6, -0.1, -0.3, -0.5]], dtype=torch.float64),
        torch.tensor(rewards or [[0.0, 1.0, 0.5, -0.5]], dtype=torch.float64),
        torch.tensor(lengths or [[2.0, 4.0, 1.0, 3.0]], dtype=torch.float64),
        torch.ones(shape, dtype=torch.bool) if valid is None else torch.tensor(valid, dtype=torch.bool),
    )


def test_disabled_gflowrl_keeps_r0_output_unchanged():
    batch = make_batch()
    result_a = compute_objective(batch, GFlowRLConfig(enabled=False, epsilon_low=0.01))
    result_b = compute_objective(batch, GFlowRLConfig(enabled=False, epsilon_low=9.0))
    assert result_a.mode == result_b.mode == "R0"
    assert result_a.loss.item() == pytest.approx(result_b.loss.item())


def test_loss_matches_explicit_eq4_to_eq8_calculation():
    batch = make_batch()
    config = GFlowRLConfig(beta=0.7, epsilon_low=0.2, epsilon_high=0.28, importance_clip=2.0)
    result = compute_gflowrl_loss(batch, config)
    current = batch.current_logprob.detach()[0]
    old = batch.old_logprob[0]
    ref = batch.reference_logprob[0]
    rewards = batch.rewards[0]
    lengths = batch.response_lengths[0]
    log_z = (config.beta * rewards + ref - old).mean()
    raw_gap = log_z + (old - ref) / lengths - config.beta * rewards
    clipped = torch.clamp(raw_gap, -config.epsilon_low, config.epsilon_high)
    residual = clipped + (current - old) / lengths
    weight = torch.minimum(torch.exp(current - old), torch.tensor(config.importance_clip, dtype=current.dtype))
    expected = (weight * residual.square()).mean()
    assert result.loss.item() == pytest.approx(expected.item())


def test_partition_estimator_is_group_mean_and_uses_reference_minus_old():
    batch = make_batch()
    result = compute_gflowrl_loss(batch, GFlowRLConfig(beta=0.7))
    target = 0.7 * batch.rewards[0] + batch.reference_logprob[0] - batch.old_logprob[0]
    assert result.diagnostics["partition_targets"][0].tolist() == pytest.approx(target.tolist())
    assert result.diagnostics["partition_log_z"].item() == pytest.approx(target.mean().item())


def test_importance_weight_known_ratios_and_upper_clip():
    old = torch.zeros((1, 4), dtype=torch.float64)
    current = torch.tensor([[math.log(0.5), math.log(1.5), math.log(3.0), math.log(0.01)]], dtype=torch.float64)
    result = compute_gflowrl_loss(
        make_batch(current=current.tolist(), old=old.tolist()), GFlowRLConfig(importance_clip=2.0)
    )
    assert result.diagnostics["importance_weight"][0].tolist() == pytest.approx([0.5, 1.5, 2.0, 0.01])


def test_asymmetric_clipping_matches_paper_bounds():
    result = compute_gflowrl_loss(
        make_batch(
            current=[[0.0, 0.0, 0.0, 0.0]],
            old=[[0.0, 0.0, 0.0, 0.0]],
            reference=[[0.0, 0.0, 0.0, 0.0]],
            rewards=[[0.0, 1.0, 2.0, 3.0]],
            lengths=[[1.0, 1.0, 1.0, 1.0]],
        ),
        GFlowRLConfig(beta=1.0, epsilon_low=0.2, epsilon_high=0.28),
    )
    assert result.diagnostics["flow_gap"][0].tolist() == pytest.approx([1.5, 0.5, -0.5, -1.5])
    assert result.diagnostics["clipped_flow_gap"][0].tolist() == pytest.approx([0.28, 0.28, -0.2, -0.2])


def test_zero_and_constant_rewards_are_finite():
    batch = make_batch(
        current=[[0.0, 0.0, 0.0, 0.0]],
        old=[[0.0, 0.0, 0.0, 0.0]],
        reference=[[0.0, 0.0, 0.0, 0.0]],
        rewards=[[2.0, 2.0, 2.0, 2.0]],
    )
    result = compute_gflowrl_loss(batch, GFlowRLConfig(beta=1.0))
    assert result.loss.item() == pytest.approx(0.0)
    assert torch.isfinite(result.loss)


def test_all_invalid_group_is_masked_to_zero():
    batch = make_batch(valid=[[False, False, False, False]])
    result = compute_gflowrl_loss(batch, GFlowRLConfig())
    result.loss.backward()
    assert result.loss.item() == pytest.approx(0.0)
    assert result.diagnostics["valid_counts"].tolist() == [0]
    assert torch.isfinite(batch.current_logprob.grad).all()


def test_frozen_reference_old_reward_and_partition_have_no_gradient():
    batch = make_batch()
    result = compute_gflowrl_loss(batch, GFlowRLConfig())
    assert not result.diagnostics["partition_log_z"].requires_grad
    assert not result.diagnostics["flow_gap"].requires_grad
    result.loss.backward()
    assert batch.current_logprob.grad is not None


def test_policy_reference_logprob_convention_is_old_minus_reference_in_gap():
    batch = make_batch(
        current=[[0.0, 0.0, 0.0, 0.0]],
        old=[[2.0, 2.0, 2.0, 2.0]],
        reference=[[1.0, 1.0, 1.0, 1.0]],
        rewards=[[0.0, 0.0, 0.0, 0.0]],
        lengths=[[2.0, 2.0, 2.0, 2.0]],
    )
    result = compute_gflowrl_loss(batch, GFlowRLConfig())
    # Eq. 6 is sg[Z] + log(pi_old/pi_ref)/|y| - beta*r.
    assert result.diagnostics["flow_gap"][0, 0].item() == pytest.approx(-0.5)


def test_eos_and_truncation_are_represented_by_masked_contract_in_tests():
    from experiments.gflowrl_reasoning_v1.contracts import Termination, validate_termination

    validate_termination(Termination(eos_seen=True, truncated=False, valid=True))
    validate_termination(Termination(eos_seen=False, truncated=True, valid=False))
    with pytest.raises(ValueError):
        validate_termination(Termination(eos_seen=True, truncated=True, valid=False))
    with pytest.raises(ValueError):
        validate_termination(Termination(eos_seen=False, truncated=True, valid=True))


def test_extreme_logprob_ratios_remain_finite():
    batch = make_batch(
        current=[[-1000.0, 1000.0, 0.0, 0.0]],
        old=[[0.0, 0.0, 0.0, 0.0]],
    )
    result = compute_gflowrl_loss(batch, GFlowRLConfig())
    assert torch.isfinite(result.diagnostics["importance_weight"]).all()
    assert torch.isfinite(result.loss)


def test_nan_inf_guard_rejects_nonfinite_inputs():
    with pytest.raises(FloatingPointError, match="nonfinite current_logprob"):
        make_batch(current=[[float("nan"), 0.0, 0.0, 0.0]])


def test_fixed_seed_micro_example_is_deterministic():
    torch.manual_seed(123)
    a = make_batch(
        current=torch.randn(1, 4).tolist(),
        old=torch.randn(1, 4).tolist(),
        reference=torch.randn(1, 4).tolist(),
        rewards=torch.randn(1, 4).tolist(),
        lengths=[[1.0, 2.0, 3.0, 4.0]],
    )
    value_a = compute_gflowrl_loss(a, GFlowRLConfig()).loss.item()
    torch.manual_seed(123)
    b = make_batch(
        current=torch.randn(1, 4).tolist(),
        old=torch.randn(1, 4).tolist(),
        reference=torch.randn(1, 4).tolist(),
        rewards=torch.randn(1, 4).tolist(),
        lengths=[[1.0, 2.0, 3.0, 4.0]],
    )
    value_b = compute_gflowrl_loss(b, GFlowRLConfig()).loss.item()
    assert value_a == pytest.approx(value_b, abs=1e-12)


def test_group_size_is_fixed_for_qualification():
    with pytest.raises(ValueError, match="group size"):
        compute_gflowrl_loss(
            make_batch(
                current=[[-1.0, -1.0, -1.0]],
                old=[[-1.0, -1.0, -1.0]],
                reference=[[-1.0, -1.0, -1.0]],
                rewards=[[0.0, 0.0, 0.0]],
                lengths=[[1.0, 1.0, 1.0]],
            ),
            GFlowRLConfig(),
        )
