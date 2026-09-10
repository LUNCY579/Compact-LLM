"""Shifted CE plus exact forward KL; all-masked/packed boundaries fail safely."""
from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def recovery_loss(student: Tensor, teacher: Tensor, input_ids: Tensor, *,
                  valid: Tensor | None = None, segments: Tensor | None = None,
                  alpha: float = 0.5, temperature: float = 1.0, chunk_tokens: int = 32) -> Tensor:
    if not (math.isfinite(alpha) and 0 <= alpha <= 1 and math.isfinite(temperature) and temperature > 0):
        raise ValueError('invalid CE/KL coefficient or temperature')
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError('chunk_tokens must be positive')
    if student.ndim != 3 or teacher.shape != student.shape or input_ids.shape != student.shape[:2] or input_ids.shape[1] < 2:
        raise ValueError('logit/token shapes do not support shifted loss')
    if input_ids.dtype != torch.long or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError('invalid token dtype or nonfinite logits')
    if valid is not None and not bool(((valid == 0) | (valid == 1)).all()):
        raise ValueError('valid mask must be binary')
    mask = torch.ones_like(input_ids, dtype=torch.bool) if valid is None else valid.bool()
    if mask.shape != input_ids.shape:
        raise ValueError('mask shape mismatch')
    active = mask[:, :-1] & mask[:, 1:] & (input_ids[:, 1:] != -100)
    if segments is not None:
        if segments.shape != input_ids.shape or segments.dtype != torch.long:
            raise ValueError('segment shape mismatch')
        active &= segments[:, :-1] == segments[:, 1:]
    s = student[:, :-1][active]
    t = teacher.detach()[:, :-1][active]
    y = input_ids[:, 1:][active]
    if y.numel() == 0 or (y < 0).any() or (y >= student.shape[-1]).any():
        raise ValueError('no usable shifted targets or targets outside vocabulary')
    total = student.new_zeros((), dtype=torch.float32)
    for start in range(0, y.numel(), chunk_tokens):
        ss, tt, yy = s[start:start+chunk_tokens].float(), t[start:start+chunk_tokens].float(), y[start:start+chunk_tokens]
        ce = F.cross_entropy(ss, yy, reduction='sum')
        # KL(teacher || student), not the reverse; include T^2 exactly once.
        kl = F.kl_div(F.log_softmax(ss/temperature, -1), F.log_softmax(tt/temperature, -1),
                      log_target=True, reduction='sum') * temperature**2
        total = total + (1-alpha)*ce + alpha*kl
    return total / y.numel()
