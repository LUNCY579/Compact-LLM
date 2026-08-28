#!/usr/bin/env python3
"""Inference-time forms of the abstention gate, for use with ``model.generate``.

Two gates, two costs:

``LogitOffsetGate``
    ``z_UNKNOWN + b >= z_top1``. The softmax normaliser cancels, so this is a
    constant added to one logit. It needs no processor at all -- it can live in
    the weights or the model class (see the shipped checkpoint's
    ``modeling_abstain.py``). The class here exists so the baked checkpoint can
    be reproduced and unit-tested without loading remote code.

``NormalisedAbstentionGate``
    ``logsumexp(z_U) - logsumexp(z) >= threshold``, i.e. the gate selected as the
    confirmatory primary (S1). The normaliser does *not* cancel, so no constant
    offset expresses it and a processor is required. It is still one forward
    pass; the cost is six lines at the call site, not a second decode.

Both fire on the first generated token only, matching the gate definition. When
they fire they force the argmax to ``UNKNOWN``; they never suppress it.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import LogitsProcessor


class _FirstStepGate(LogitsProcessor):
    """Shared plumbing: act once, on the first generated token, then stand down."""

    def __init__(self, *, prompt_length: int, unknown_token_id: int) -> None:
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        self.prompt_length = int(prompt_length)
        self.unknown_token_id = int(unknown_token_id)

    def fires(self, scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if int(input_ids.shape[-1]) != self.prompt_length:
            return scores
        fires = self.fires(scores)
        if not bool(fires.any()):
            return scores
        scores = scores.clone()
        # Force the argmax without touching any other ordering.
        ceiling = scores.max(dim=-1).values + 1.0
        scores[fires, self.unknown_token_id] = ceiling[fires].to(scores.dtype)
        return scores


class LogitOffsetGate(_FirstStepGate):
    """S4: abstain iff ``z_UNKNOWN + bias >= max(z)``."""

    def __init__(self, *, prompt_length: int, unknown_token_id: int, bias: float) -> None:
        super().__init__(prompt_length=prompt_length, unknown_token_id=unknown_token_id)
        self.bias = float(bias)

    def fires(self, scores: torch.Tensor) -> torch.Tensor:
        return scores[:, self.unknown_token_id] + self.bias >= scores.max(dim=-1).values


class NormalisedAbstentionGate(_FirstStepGate):
    """S1: abstain iff ``log sum_{i in U} P(i) >= threshold``."""

    def __init__(
        self,
        *,
        prompt_length: int,
        unknown_token_id: int,
        abstention_token_ids: Sequence[int],
        threshold: float,
    ) -> None:
        super().__init__(prompt_length=prompt_length, unknown_token_id=unknown_token_id)
        if not abstention_token_ids:
            raise ValueError("At least one abstention token id is required")
        self.abstention_token_ids = [int(value) for value in abstention_token_ids]
        self.threshold = float(threshold)

    def fires(self, scores: torch.Tensor) -> torch.Tensor:
        logprobs = torch.log_softmax(scores.float(), dim=-1)
        mass = torch.logsumexp(logprobs[:, self.abstention_token_ids], dim=-1)
        return mass >= self.threshold

