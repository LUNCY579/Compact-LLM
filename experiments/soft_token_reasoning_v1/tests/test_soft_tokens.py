from __future__ import annotations

import pytest
import torch

from experiments.soft_token_reasoning_v1.implementation.rollout import (
    hard_inference_rollout,
    hard_reasoning_rollout,
    soft_reasoning_rollout,
)
from experiments.soft_token_reasoning_v1.implementation.soft_tokens import (
    SoftTokenConfig,
    embedding_rms,
    gaussian_log_probability,
    noisy_soft_embedding,
    reinforce_loss,
    rloo_advantages,
    soft_embedding,
    soft_probabilities,
)


def toy_embedding() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)


def test_hand_computed_soft_embedding():
    result = soft_embedding(torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64), toy_embedding())
    assert torch.equal(result, torch.tensor([1.2, 1.8], dtype=torch.float64))


@pytest.mark.parametrize("token", [0, 1, 2])
def test_one_hot_recovers_token_embedding(token: int):
    embedding = toy_embedding()
    probabilities = torch.nn.functional.one_hot(torch.tensor(token), num_classes=3).to(torch.float64)
    assert torch.equal(soft_embedding(probabilities, embedding), embedding[token])


def test_gradient_flows_through_soft_distribution_and_embedding():
    logits = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
    embedding = toy_embedding().requires_grad_()
    state = soft_embedding(soft_probabilities(logits, 0.5), embedding)
    state.square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
    assert logits.grad.abs().sum() > 0


def test_noise_path_is_deterministic_under_fixed_seed():
    embedding = toy_embedding()
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    first = noisy_soft_embedding(probabilities, embedding, 0.33, torch.Generator().manual_seed(7))
    second = noisy_soft_embedding(probabilities, embedding, 0.33, torch.Generator().manual_seed(7))
    third = noisy_soft_embedding(probabilities, embedding, 0.33, torch.Generator().manual_seed(8))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[1], third[1])
    assert torch.equal(first[2], 0.33 * embedding_rms(embedding))


def test_gaussian_density_matches_zero_residual_constant():
    sample = torch.zeros(2, 3, dtype=torch.float64)
    mean = torch.zeros_like(sample)
    sigma = torch.tensor(2.0, dtype=torch.float64)
    expected = torch.full((2,), float(-3 * 0.5 * torch.log(2 * torch.pi * sigma.square())), dtype=torch.float64)
    assert torch.allclose(gaussian_log_probability(sample, mean, sigma), expected)


def test_rloo_and_reinforce_stop_gradient_on_advantage():
    rewards = torch.tensor([[0.0, 1.0, 2.0], [2.0, 2.0, 0.0]], dtype=torch.float64)
    advantages = rloo_advantages(rewards).requires_grad_()
    soft_logp = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    loss = reinforce_loss(advantages, soft_logp)
    loss.backward()
    assert advantages.grad is None
    assert soft_logp.grad is not None and torch.isfinite(soft_logp.grad).all()
    assert torch.equal(rloo_advantages(torch.tensor([[1.0, 1.0]])), torch.zeros(1, 2))


def test_soft_mode_disabled_is_exactly_hard_rollout():
    embedding = torch.arange(15, dtype=torch.float64).reshape(5, 3) / 7
    prompt = torch.ones(2, 3, dtype=torch.float64)

    def next_logits(prefix: torch.Tensor) -> torch.Tensor:
        return torch.arange(5, dtype=prefix.dtype) + prefix.sum() * 0.01

    hard = hard_reasoning_rollout(next_logits, prompt, embedding, 4, torch.Generator().manual_seed(19), 1.0)
    disabled = soft_reasoning_rollout(
        next_logits,
        prompt,
        embedding,
        4,
        SoftTokenConfig(temperature=1.0, train_soft=False, add_noise=False),
        torch.Generator().manual_seed(19),
    )
    assert disabled.mode == "hard"
    assert disabled.token_ids == hard.token_ids
    assert torch.equal(torch.stack(disabled.states), torch.stack(hard.states))


def test_soft_train_hard_infer_switch_is_explicit():
    embedding = torch.eye(4, dtype=torch.float64)
    prompt = torch.zeros(1, 4, dtype=torch.float64)

    def next_logits(prefix: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.0, 0.5, 1.0, -0.5], dtype=prefix.dtype)

    trained = soft_reasoning_rollout(
        next_logits,
        prompt,
        embedding,
        3,
        SoftTokenConfig(temperature=0.5, noise_scale=0.33, train_soft=True, add_noise=True),
        torch.Generator().manual_seed(3),
    )
    deployed = hard_inference_rollout(next_logits, prompt, embedding, 3, torch.Generator().manual_seed(3))
    assert trained.mode == "soft"
    assert deployed.mode == "hard"
    assert all(torch.equal(state, embedding[token]) for state, token in zip(deployed.states, deployed.token_ids))
    assert any(not torch.equal(state, embedding[token]) for state, token in zip(trained.states, trained.token_ids))


def test_tiny_optimization_decreases_loss():
    torch.manual_seed(4)
    embedding = torch.eye(3, dtype=torch.float64)
    target = embedding[2]
    logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = torch.optim.SGD([logits], lr=0.8)
    initial = None
    for step in range(40):
        optimizer.zero_grad()
        state = soft_embedding(soft_probabilities(logits, 0.5), embedding)
        loss = torch.nn.functional.mse_loss(state, target)
        if initial is None:
            initial = float(loss)
        loss.backward()
        optimizer.step()
    assert float(loss) < initial
    assert torch.isfinite(logits).all()


def test_nonfinite_inputs_are_rejected():
    with pytest.raises(ValueError):
        soft_probabilities(torch.tensor([float("nan"), 0.0]), 0.5)
    with pytest.raises(ValueError):
        soft_embedding(torch.tensor([0.5, 0.6]), torch.ones(2, 2))
