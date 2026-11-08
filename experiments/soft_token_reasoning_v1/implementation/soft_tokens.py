"""Faithful soft-token mechanics from Butt et al., arXiv:2509.19170.

This module deliberately contains tensor mechanics only.  It does not load a
checkpoint, construct an optimizer, run a training job, or access evaluation
data.  The bounded qualification tests use it with tiny tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SoftTokenConfig:
    """Explicit train/inference contract.

    ``temperature`` is the paper's probability temperature.  ``noise_scale``
    is a multiplier on the RMS norm of the embedding matrix, matching the
    paper's sigma = 0.33 * RMS(E) default.  Noise is present only on the
    training-time soft path unless an evaluation arm explicitly opts in.
    """

    temperature: float = 0.5
    noise_scale: float = 0.33
    train_soft: bool = True
    infer_hard: bool = True
    add_noise: bool = True

    def validate(self) -> None:
        if self.temperature <= 0 or not torch.isfinite(torch.tensor(self.temperature)):
            raise ValueError("temperature must be finite and > 0")
        if self.noise_scale < 0 or not torch.isfinite(torch.tensor(self.noise_scale)):
            raise ValueError("noise_scale must be finite and >= 0")
        if not self.train_soft and self.add_noise:
            raise ValueError("noise cannot be enabled when soft training is disabled")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_embedding(embedding: torch.Tensor) -> None:
    if embedding.ndim != 2 or embedding.shape[0] == 0 or embedding.shape[1] == 0:
        raise ValueError("embedding must have shape [vocabulary, width]")
    if not embedding.is_floating_point():
        raise ValueError("embedding must be floating point")
    _require_finite("embedding", embedding)


def _validate_probabilities(probabilities: torch.Tensor, vocabulary: int) -> None:
    if probabilities.ndim < 1 or probabilities.shape[-1] != vocabulary:
        raise ValueError("probabilities must end with the embedding vocabulary")
    if not probabilities.is_floating_point():
        raise ValueError("probabilities must be floating point")
    _require_finite("probabilities", probabilities)
    if (probabilities < 0).any():
        raise ValueError("probabilities must be non-negative")
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("probabilities must sum to one")


def embedding_rms(embedding: torch.Tensor) -> torch.Tensor:
    """Return the scalar RMS norm used to set Gaussian exploration scale."""

    _validate_embedding(embedding)
    return torch.sqrt(torch.mean(embedding.square()))


def soft_probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute the paper's p_t = softmax(logits / tau)."""

    if logits.ndim < 1 or not logits.is_floating_point():
        raise ValueError("logits must be floating point with a vocabulary axis")
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("temperature must be finite and > 0")
    _require_finite("logits", logits)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    _require_finite("soft probabilities", probabilities)
    return probabilities


def soft_embedding(probabilities: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    """Compute z_t = sum_v p_t(v) E(v), preserving gradient to p_t and E."""

    _validate_embedding(embedding)
    _validate_probabilities(probabilities, embedding.shape[0])
    if probabilities.device != embedding.device:
        raise ValueError("probabilities and embedding must share a device")
    return probabilities @ embedding


def noisy_soft_embedding(
    probabilities: torch.Tensor,
    embedding: torch.Tensor,
    noise_scale: float,
    generator: Optional[torch.Generator] = None,
    add_noise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mean, noisy_state, sigma)`` for one soft reasoning state."""

    if noise_scale < 0 or not torch.isfinite(torch.tensor(noise_scale)):
        raise ValueError("noise_scale must be finite and >= 0")
    mean = soft_embedding(probabilities, embedding)
    sigma = noise_scale * embedding_rms(embedding).to(mean)
    if add_noise and sigma.item() > 0:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        ) * sigma
        state = mean + noise
    else:
        state = mean.clone()
    _require_finite("soft state", state)
    return mean, state, sigma


def gaussian_log_probability(
    sample: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor | float
) -> torch.Tensor:
    """Log density of the isotropic Gaussian used for exploration.

    The result is summed over the embedding width, yielding one log density
    per leading sample/state index.  The mean remains differentiable; callers
    must detach the reward/advantage, not this density.
    """

    if sample.shape != mean.shape or not sample.is_floating_point():
        raise ValueError("sample and mean must be same-shaped floating tensors")
    _require_finite("sample", sample)
    _require_finite("mean", mean)
    sigma_tensor = torch.as_tensor(sigma, dtype=sample.dtype, device=sample.device)
    if sigma_tensor.ndim != 0 or sigma_tensor.item() <= 0 or not torch.isfinite(sigma_tensor):
        raise ValueError("sigma must be finite and > 0")
    width = sample.shape[-1]
    normalized = (sample - mean) / sigma_tensor
    value = -0.5 * (normalized.square() + torch.log(2 * torch.pi * sigma_tensor.square()))
    return value.sum(dim=-1)


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Per-prompt leave-one-out rewards from the paper's Appendix A."""

    if rewards.ndim != 2 or rewards.shape[1] < 2 or not rewards.is_floating_point():
        raise ValueError("rewards must have shape [prompts, group>=2]")
    _require_finite("rewards", rewards)
    group = rewards.shape[1]
    return rewards - (rewards.sum(dim=1, keepdim=True) - rewards) / (group - 1)


def reinforce_loss(
    advantages: torch.Tensor,
    soft_log_probability: torch.Tensor,
    answer_log_probability: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Paper Eq. (8), with a stop-gradient on the advantage.

    ``soft_log_probability`` is the sum over noisy soft states.  The optional
    answer term is ordinary sampled-token log-probability.  No clipping, KL,
    GFlowRL term, straight-through estimator, or other objective is included.
    """

    if advantages.shape != soft_log_probability.shape:
        raise ValueError("advantage and soft log-probability shapes differ")
    if answer_log_probability is not None and answer_log_probability.shape != advantages.shape:
        raise ValueError("answer log-probability shape differs")
    _require_finite("advantages", advantages)
    _require_finite("soft log-probability", soft_log_probability)
    total = soft_log_probability
    if answer_log_probability is not None:
        _require_finite("answer log-probability", answer_log_probability)
        total = total + answer_log_probability
    return -(advantages.detach() * total).mean()
