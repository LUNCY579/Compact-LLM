"""Shared clipped, sequence-mean policy loss for an eventual qualified trainer."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def policy_loss(current: Tensor, behavior: Tensor, reference: Tensor,
                advantages: Tensor, mask: Tensor, *, clip_low: float = .2,
                clip_high: float = .28, beta: float = .01) -> Tensor:
    if current.ndim != 2 or behavior.shape != current.shape or reference.shape != current.shape:
        raise ValueError('logprob tensors must share [completion, token] shape')
    if mask.shape != current.shape or mask.dtype != torch.bool:
        raise ValueError('explicit boolean completion-token mask required')
    if advantages.shape != (current.shape[0],) or current.shape[0] == 0:
        raise ValueError('one advantage per completion required')
    if any(x.device != current.device for x in (behavior, reference, advantages, mask)):
        raise ValueError('device mismatch')
    if current.dtype not in (torch.float32, torch.float64):
        raise ValueError('explicit FP32/FP64 loss arithmetic required')
    if any(x.dtype != current.dtype for x in (behavior, reference, advantages)):
        raise ValueError('dtype mismatch')
    if not bool(mask.any(dim=1).all()):
        raise ValueError('every completion needs at least one loss-bearing token')
    if not all(math.isfinite(x) for x in (clip_low, clip_high, beta)):
        raise ValueError('invalid loss constants')
    if not 0 <= clip_low < 1 or clip_high < 0 or beta < 0:
        raise ValueError('invalid clipping/KL coefficients')
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError('nonfinite advantages')
    for x in (current, behavior, reference):
        if not bool(torch.isfinite(x[mask]).all()) or bool((x[mask] > 0).any()):
            raise ValueError('invalid log probabilities')
    # Sanitize masked positions *before* nonlinear operations: nan*0 is still nan.
    new = torch.where(mask, current, torch.zeros_like(current))
    old = torch.where(mask, behavior.detach(), torch.zeros_like(behavior))
    ref = torch.where(mask, reference.detach(), torch.zeros_like(reference))
    log_ratio, log_ref = new - old, ref - new
    if bool((log_ratio.abs() > 20).any()) or bool((log_ref.abs() > 20).any()):
        raise FloatingPointError('importance/KL log-ratio outside qualified range')
    ratio = log_ratio.exp()
    advantage = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)
    kl = log_ref.exp() - log_ref - 1
    loss = ((-surrogate + beta * kl) * mask).sum(1) / mask.sum(1)
    return loss.mean()
