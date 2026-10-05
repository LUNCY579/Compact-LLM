"""Eq. 2: fixed trajectories, detached teacher, exact full-vocabulary KL."""
from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.nn import functional as F


class _ExactKL(torch.autograd.Function):
    """Analytical logit derivatives avoid cancelling two unit-mass terms.

    At identical represented distributions the gradient is exactly zero, rather
    than an Adam-amplified roundoff residual. Backward recomputation retains
    higher-order derivatives with respect to the student, not the teacher.
    """
    @staticmethod
    def forward(ctx, student, teacher, direction):
        ctx.save_for_backward(student, teacher)
        ctx.direction = direction
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        lp, lq = (F.log_softmax(x.to(dtype), -1) for x in (student, teacher))
        return ((lq.exp() * (lq - lp)).sum(-1) if direction == "forward"
                else (lp.exp() * (lp - lq)).sum(-1))

    @staticmethod
    def backward(ctx, incoming):
        student, teacher = ctx.saved_tensors
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        lp, lq = (F.log_softmax(x.to(dtype), -1) for x in (student, teacher))
        p = lp.exp()
        if ctx.direction == "forward":
            gradient = p - lq.exp()
        else:
            difference = lp - lq
            gradient = p * (difference - (p * difference).sum(-1, keepdim=True))
        return (incoming.unsqueeze(-1) * gradient).to(student.dtype), None, None



def token_kl(student: Tensor, teacher: Tensor, direction: str) -> Tensor:
    """Return one KL per position. Teacher is always stop-gradient.

    forward = KL(q_teacher || p_student); reverse = KL(p_student || q_teacher).
    This is neither sampled-token NLL nor a top-k approximation. No T**2 scaling.
    """
    if direction not in {"forward", "reverse"}:
        raise ValueError("KL direction must be forward or reverse")
    if student.shape != teacher.shape or student.ndim < 2 or student.shape[-1] < 2:
        raise ValueError("Matching [..., vocabulary] logits with vocabulary >= 2 required")
    if student.device != teacher.device:
        raise ValueError("Logits must share a device")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError("Floating logits required")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Nonfinite logits")
    return _ExactKL.apply(student, teacher.detach(), direction)


def sequence_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean within each response, then across examples (not pooled token mean)."""
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != torch.bool:
        raise ValueError("Expected [batch, response] values and boolean mask")
    if mask.device != values.device or not torch.isfinite(values).all():
        raise ValueError("Invalid loss values/device")
    lengths = mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("Empty supervised response")
    return ((values * mask).sum(-1) / lengths).mean()


def response_logits(logits: Tensor, prefix_length: int, response_length: int) -> Tensor:
    """Causal shift: first response token is predicted at prefix_length - 1."""
    if logits.ndim != 3 or min(prefix_length, response_length) < 1:
        raise ValueError("Nonempty prefix/response and [batch, tokens, vocab] required")
    if logits.shape[1] != prefix_length + response_length:
        raise ValueError("Exact prefix/response alignment required; no hidden truncation")
    return logits[:, prefix_length - 1:prefix_length + response_length - 1]


def reference_nll(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("Reference targets must align with response logits")
    if not torch.isfinite(logits).all() or targets.numel() == 0:
        raise ValueError("Invalid supervised reference")
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def validate_rate(rate: float) -> None:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("ema_rate must be finite in [0,1], not a KL mixing parameter")


@torch.no_grad()
def ema_update(teacher: dict[str, Tensor], student: dict[str, Tensor], rate: float) -> None:
    """Eq. 3, called AFTER the optimizer: teacher <- (1-a) teacher + a student."""
    validate_rate(rate)
    if teacher.keys() != student.keys():
        raise ValueError("EMA parameter names differ")
    for name, t in teacher.items():
        s = student[name]
        if t.shape != s.shape or not t.is_floating_point() or not torch.isfinite(s).all() or not torch.isfinite(t).all():
            raise ValueError("Invalid EMA state")
    if rate == 0:
        return  # exact frozen endpoint, including storage bytes
    for name, t in teacher.items():
        s = student[name].detach().to(t)
        if rate == 1:
            t.copy_(s)  # exact synchronized endpoint
        else:
            t.mul_(1 - rate).add_(s, alpha=rate)
