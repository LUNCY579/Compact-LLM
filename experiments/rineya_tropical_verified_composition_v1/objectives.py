"""Differentiable fixed-bank losses; these are NOT on-policy MaxRL estimators."""
from __future__ import annotations

import math
import torch
from torch import Tensor

ARMS = ("uniform", "hard_max", "soft", "coverage", "paper_token_mean")


def path_loss(logps: Tensor, lengths: Tensor, arm: str, temperature: float = 1.0,
              selected: tuple[int, ...] | None = None) -> Tensor:
    """One problem's loss. Caller averages equally over nonempty problems.

    logps are current-policy SUMMED log probabilities of distinct complete paths.
    lengths count supervised response tokens (or symbolic actions in toy runs).
    The primary uniform/hard/soft/coverage contrasts use sequence normalization.
    paper_token_mean is an explicitly different, secondary imitation treatment.
    """
    if logps.ndim != 1 or lengths.shape != logps.shape or not logps.numel():
        raise ValueError("Expected nonempty aligned path vectors")
    if not logps.is_floating_point() or not torch.isfinite(logps).all() or (logps > 1e-6).any():
        raise ValueError("Invalid path log probabilities")
    if torch.logsumexp(logps.detach(), 0) > 1e-5:
        raise ValueError("Candidate mass exceeds one: check duplicates and sequence normalization")
    if not torch.isfinite(lengths).all() or (lengths <= 0).any() or (lengths.remainder(1) != 0).any():
        raise ValueError("Empty or invalid supervised path")
    if arm not in ARMS:
        raise ValueError("Unknown objective")
    if arm == "uniform":
        return -logps.mean()
    if arm == "hard_max":
        # Stable first-index tie rule; do not spread a tie's gradient implicitly.
        return -logps[torch.argmax(logps.detach())]
    if arm == "soft":
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        if temperature < torch.finfo(logps.dtype).tiny:
            raise ValueError("Temperature underflows score dtype; use hard_max or float64")
        # Center BEFORE division to avoid overflow at very small temperatures.
        maximum = logps.max().detach()
        return -(maximum + temperature * torch.logsumexp(
            (logps - maximum) / temperature, dim=0))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Coverage selection must contain unique indices")
    if any(type(i) is not int or not 0 <= i < len(logps) for i in selected):
        raise ValueError("Coverage index out of range")
    indices = torch.tensor(selected, device=logps.device)
    scores = logps[indices]
    if arm == "paper_token_mean":
        scores = scores / lengths[indices]
    return -scores.mean()


def causal_response_logps(logits: Tensor, input_ids: Tensor,
                         attention_mask: Tensor, response_mask: Tensor
                         ) -> tuple[Tensor, Tensor]:
    """Score response tokens with exactly one causal shift.

    EOS must be marked by response_mask even when EOS == padding token ID.
    This function never guesses boundaries from token IDs. Inputs are already
    tokenized by a separately qualified model/tokenizer adapter.
    """
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Expected logits [batch,time,vocab] and IDs [batch,time]")
    if input_ids.dtype != torch.long or attention_mask.dtype != torch.bool or response_mask.dtype != torch.bool:
        raise ValueError("IDs must be int64 and masks explicitly boolean")
    if attention_mask.shape != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError("Mask shape mismatch")
    if input_ids.shape[1] < 2 or response_mask[:, 0].any():
        raise ValueError("First token has no causal predecessor")
    if (response_mask & ~attention_mask).any():
        raise ValueError("Padding cannot be supervised")
    if (response_mask[:, 1:] & ~attention_mask[:, :-1]).any():
        raise ValueError("Supervised token has a padded predecessor")
    if (input_ids < 0).any() or (input_ids >= logits.shape[-1]).any():
        raise ValueError("Out-of-vocabulary token")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    mask = response_mask[:, 1:]
    counts = mask.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("Empty response mask; truncation is not success")
    token_logps = logits[:, :-1].float().log_softmax(-1).gather(
        -1, input_ids[:, 1:, None]).squeeze(-1)
    return (token_logps * mask).sum(-1), counts


def reference_kl(student_logits: Tensor, reference_logits: Tensor, mask: Tensor) -> Tensor:
    """KL(reference || student); reference is detached, protected TRAIN inputs only."""
    if student_logits.shape != reference_logits.shape or student_logits.ndim != 3:
        raise ValueError("Reference/student logits must align")
    if mask.shape != student_logits.shape[:2] or mask.dtype != torch.bool or not mask.any():
        raise ValueError("Nonempty aligned boolean KL mask required")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(reference_logits).all():
        raise ValueError("Nonfinite KL logits")
    logq = reference_logits.detach().float().log_softmax(-1)
    logp = student_logits.float().log_softmax(-1)
    return ((logq.exp() * (logq - logp)).sum(-1))[mask].mean()


def protected_loss(new: Tensor, acquired: Tensor | None = None,
                   legacy: Tensor | None = None, acquired_weight: float = 0.0,
                   legacy_weight: float = 0.0) -> Tensor:
    total = new
    for loss, weight in ((acquired, acquired_weight), (legacy, legacy_weight)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Replay weights must be finite and nonnegative")
        if weight and loss is None:
            raise ValueError("Requested protection has no training reservoir")
        if loss is not None:
            total = total + weight * loss
    if total.ndim != 0 or not torch.isfinite(total):
        raise ValueError("Invalid optimization loss")
    return total
