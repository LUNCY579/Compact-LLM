"""Tiny, model-agnostic rollout adapter for mechanical qualification.

The adapter accepts a callable returning next-token logits from the current
embedding prefix.  A production adapter can wrap a Qwen3 ``inputs_embeds``
forward; this qualification layer intentionally does not load a checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .soft_tokens import SoftTokenConfig, noisy_soft_embedding, soft_probabilities


LogitFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class RolloutTrace:
    token_ids: list[int]
    states: list[torch.Tensor]
    means: list[torch.Tensor]
    soft_log_probability: torch.Tensor
    mode: str


def _sample_categorical(probabilities: torch.Tensor, generator: Optional[torch.Generator]) -> int:
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def hard_reasoning_rollout(
    next_logits: LogitFn,
    prompt_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    steps: int,
    generator: Optional[torch.Generator] = None,
    temperature: float = 1.0,
) -> RolloutTrace:
    """Ordinary discrete autoregressive reasoning, including the prompt prefix."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if prompt_embeddings.ndim != 2 or token_embeddings.ndim != 2:
        raise ValueError("prompt and token embeddings must be rank two")
    prefix = prompt_embeddings
    states: list[torch.Tensor] = []
    token_ids: list[int] = []
    means: list[torch.Tensor] = []
    log_probability = torch.zeros((), dtype=prefix.dtype, device=prefix.device)
    for _ in range(steps):
        logits = next_logits(prefix)
        probabilities = soft_probabilities(logits, temperature)
        token_id = _sample_categorical(probabilities, generator)
        state = token_embeddings[token_id]
        prefix = torch.cat((prefix, state.unsqueeze(0)), dim=0)
        token_ids.append(token_id)
        states.append(state)
        means.append(state)
        log_probability = log_probability + torch.log(probabilities[token_id])
    return RolloutTrace(token_ids, states, means, log_probability, "hard")


def soft_reasoning_rollout(
    next_logits: LogitFn,
    prompt_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    steps: int,
    config: SoftTokenConfig,
    generator: Optional[torch.Generator] = None,
) -> RolloutTrace:
    """Paper-faithful noisy soft-token reasoning for a bounded rollout."""

    config.validate()
    if not config.train_soft:
        return hard_reasoning_rollout(
            next_logits,
            prompt_embeddings,
            token_embeddings,
            steps,
            generator=generator,
            temperature=config.temperature,
        )
    if steps <= 0:
        raise ValueError("steps must be positive")
    prefix = prompt_embeddings
    states: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    token_ids: list[int] = []
    log_probability = torch.zeros((), dtype=prefix.dtype, device=prefix.device)
    for _ in range(steps):
        logits = next_logits(prefix)
        probabilities = soft_probabilities(logits, config.temperature)
        mean, state, sigma = noisy_soft_embedding(
            probabilities,
            token_embeddings,
            config.noise_scale,
            generator=generator,
            add_noise=config.add_noise,
        )
        prefix = torch.cat((prefix, state.unsqueeze(0)), dim=0)
        means.append(mean)
        states.append(state)
        token_ids.append(int(probabilities.argmax().item()))
        if config.add_noise and sigma.item() > 0:
            from .soft_tokens import gaussian_log_probability

            log_probability = log_probability + gaussian_log_probability(state, mean, sigma)
    return RolloutTrace(token_ids, states, means, log_probability, "soft")


def reasoning_rollout(
    next_logits: LogitFn,
    prompt_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    steps: int,
    *,
    train_soft: bool,
    infer_hard: bool,
    config: SoftTokenConfig,
    generator: Optional[torch.Generator] = None,
) -> RolloutTrace:
    """Explicit switch: training may be soft; inference must be hard."""

    if not train_soft or infer_hard is False and not config.train_soft:
        return hard_reasoning_rollout(
            next_logits,
            prompt_embeddings,
            token_embeddings,
            steps,
            generator=generator,
            temperature=1.0,
        )
    return soft_reasoning_rollout(
        next_logits,
        prompt_embeddings,
        token_embeddings,
        steps,
        config,
        generator=generator,
    )


def hard_inference_rollout(
    next_logits: LogitFn,
    prompt_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    steps: int,
    generator: Optional[torch.Generator] = None,
) -> RolloutTrace:
    """Deployment path: ordinary discrete autoregressive generation only."""

    return hard_reasoning_rollout(
        next_logits,
        prompt_embeddings,
        token_embeddings,
        steps,
        generator=generator,
        temperature=1.0,
    )
