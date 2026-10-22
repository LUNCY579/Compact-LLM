"""Rineya LM adaptation, NOT the original FlexLoop RL training implementation.

logits: [depth, batch, supervised_position, vocabulary]. labels: [batch, position]
are already aligned next-token targets. Caller performs the causal shift ONCE.
All depths MUST use identical prompt/answer prefixes and tokenization. No free-
running trajectories of different prefixes may be mixed as distillation pairs.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

ARMS = ("full", "sampled", "all", "adjacent", "deepest")


def objective(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
              *, arm: str = "adjacent", coefficient: float = 0.1,
              temperature: float = 1.0, sampled_depth: int | None = None
              ) -> dict[str, torch.Tensor]:
    """Masked task loss plus teacher-detached KL, normalized by active positions.

    full: full-depth CE; sampled: explicit one-based sampled-depth CE;
    all: mean CE over depths; adjacent/deepest: full-depth CE + lambda*KL.
    This function does not sample depths, run a model, or perform optimization.
    Shared weights still change teacher outputs after subsequent optimizer steps.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if (isinstance(coefficient, bool) or not math.isfinite(coefficient)
            or coefficient < 0 or isinstance(temperature, bool)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("invalid coefficient or temperature")
    if logits.ndim != 4 or min(logits.shape) < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [R,B,T,V], V>=2")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    if labels.shape != logits.shape[1:3] or mask.shape != labels.shape:
        raise ValueError("labels/mask shape mismatch")
    if labels.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError("labels must be int64 and mask must be bool")
    if labels.device != logits.device or mask.device != logits.device:
        raise ValueError("all tensors must be on the same device")
    if not mask.any():
        raise ValueError("at least one active target is required")
    active_labels = labels[mask]
    if ((active_labels < 0) | (active_labels >= logits.shape[-1])).any():
        raise ValueError("active target outside vocabulary")
    r = logits.shape[0]
    if arm in {"adjacent", "deepest"} and r < 2:
        raise ValueError("distillation requires at least two depths")
    if arm == "sampled":
        if type(sampled_depth) is not int or not 1 <= sampled_depth <= r:
            raise ValueError("sampled arm requires explicit one-based sampled_depth")
    elif sampled_depth is not None:
        raise ValueError("sampled_depth only belongs to the sampled arm")
    # Ignore padded labels (including -100) before CE; accumulation is in fp32.
    active = logits[:, mask, :].float()
    if arm == "sampled":
        task = F.cross_entropy(active[sampled_depth - 1], active_labels)
    elif arm == "all":
        task = torch.stack([F.cross_entropy(x, active_labels) for x in active]).mean()
    else:
        task = F.cross_entropy(active[-1], active_labels)
    kl = task.new_zeros(())
    if arm in {"adjacent", "deepest"}:
        log_student = F.log_softmax(active[:-1] / temperature, dim=-1)
        teacher = active[1:] if arm == "adjacent" else active[-1:].expand_as(active[:-1])
        log_teacher = F.log_softmax(teacher.detach() / temperature, dim=-1)
        # Mean over depth and active positions, sum over vocabulary. KL(teacher||student).
        kl = F.kl_div(log_student, log_teacher, log_target=True, reduction="none").sum(-1).mean()
        kl = kl * temperature ** 2
    return {"loss": task + coefficient * kl, "task_loss": task, "depth_kl": kl}
