"""Auditable GRPO primitives; torch is only imported by explicit training callers.

Sequence-mean clipped objective follows the existing RoLA/NGU branch's
objective.py (blob 9ba6b17307bb84af9e0980ab9d9362a863e9f048).
The curriculum does not enable RoLA, NGU, sparse pruning, or another RL objective.
"""
from __future__ import annotations

import math


def advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2 or not all(math.isfinite(r) and 0 <= r <= 1 for r in rewards):
        raise ValueError("need >=2 finite rewards in [0,1]")
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean)**2 for r in rewards) / len(rewards)
    if variance <= 1e-12:
        return [0.0] * len(rewards)
    return [(r - mean) / math.sqrt(variance) for r in rewards]


def policy_loss(current, behavior, reference, advantage, mask, *, epsilon=.2, beta=.02):
    import torch
    if current.ndim != 2 or any(x.shape != current.shape for x in (behavior, reference, mask)):
        raise ValueError("expected matching [completion,token] tensors")
    if mask.dtype != torch.bool or advantage.shape != (current.shape[0],) or current.shape[0] == 0:
        raise ValueError("explicit boolean response mask and one advantage per completion required")
    if current.dtype not in (torch.float32, torch.float64):
        raise ValueError("FP32/64 loss arithmetic required")
    if any(x.device != current.device for x in (behavior, reference, advantage, mask)):
        raise ValueError("device mismatch")
    if any(x.dtype != current.dtype for x in (behavior, reference, advantage)):
        raise ValueError("dtype mismatch")
    if not (0 <= epsilon < 1 and math.isfinite(beta) and beta >= 0):
        raise ValueError("invalid loss constants")
    if not bool(mask.any(1).all()) or not bool(torch.isfinite(advantage).all()):
        raise ValueError("empty completion or nonfinite advantage")
    for value in (current, behavior, reference):
        if not bool(torch.isfinite(value[mask]).all()) or bool((value[mask] > 1e-6).any()):
            raise ValueError("invalid log probabilities")
    new = torch.where(mask, current, torch.zeros_like(current))
    old = torch.where(mask, behavior.detach(), torch.zeros_like(behavior))
    ref = torch.where(mask, reference.detach(), torch.zeros_like(reference))
    delta, ref_delta = new - old, ref - new
    if bool((delta.abs() > 20).any()) or bool((ref_delta.abs() > 20).any()):
        raise FloatingPointError("importance/KL log ratio exceeds qualified numeric range")
    ratio = delta.exp()
    a = advantage.detach()[:, None]
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - epsilon, 1 + epsilon) * a)
    kl = ref_delta.exp() - ref_delta - 1
    return (((-surrogate + beta * kl) * mask).sum(1) / mask.sum(1)).mean()
