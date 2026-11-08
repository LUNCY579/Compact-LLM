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


def recovery_loss_from_hidden(student_hidden: Tensor, student_head: Tensor,
                              teacher_hidden: Tensor, teacher_head: Tensor, input_ids: Tensor, *,
                              valid: Tensor | None = None, segments: Tensor | None = None,
                              alpha: float = 0.5, temperature: float = 1.0,
                              chunk_tokens: int = 8, checkpoint_chunks: bool = True,
                              work_counter: dict[str, int] | None = None) -> Tensor:
    """Exact full-vocabulary CE + forward KL without resident [B,T,V] logits.

    Inputs are the final normalized states, before the tied LM head. Student and
    teacher widths may differ. Teacher features/head may remain on CPU while the
    student is on CUDA; only a chunk of teacher logits is transferred. Teacher
    targets are detached. Non-reentrant checkpointing recomputes student-head
    chunks during backward instead of retaining all softmax activations.

    This is not top-K distillation, feature matching, or a new training objective.
    A changed projection batch shape can still introduce ordinary rounding error.
    """
    from torch.utils.checkpoint import checkpoint
    if not (math.isfinite(alpha) and 0 <= alpha <= 1 and math.isfinite(temperature) and temperature > 0):
        raise ValueError('invalid CE/KL coefficient or temperature')
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError('chunk_tokens must be positive')
    if (student_hidden.ndim != 3 or teacher_hidden.ndim != 3
            or student_hidden.shape[:2] != teacher_hidden.shape[:2]
            or input_ids.shape != student_hidden.shape[:2] or input_ids.shape[1] < 2
            or input_ids.dtype != torch.long or student_head.ndim != 2 or teacher_head.ndim != 2
            or student_head.shape[0] != teacher_head.shape[0]
            or student_head.shape[1] != student_hidden.shape[-1]
            or teacher_head.shape[1] != teacher_hidden.shape[-1]):
        raise ValueError('incompatible final-state, vocabulary-head, or shifted-token shapes')
    if input_ids.device != student_hidden.device or student_head.device != student_hidden.device:
        raise ValueError('student tokens/states/head must share a device')
    if teacher_hidden.device != teacher_head.device:
        raise ValueError('teacher states and head must share a device')
    if not torch.isfinite(student_hidden).all() or not torch.isfinite(teacher_hidden).all():
        raise ValueError('nonfinite final hidden states')
    if valid is not None and (valid.shape != input_ids.shape or valid.device != input_ids.device
                             or not bool(((valid == 0) | (valid == 1)).all())):
        raise ValueError('valid mask must be matching and binary')
    mask = torch.ones_like(input_ids, dtype=torch.bool) if valid is None else valid.bool()
    active = mask[:, :-1] & mask[:, 1:] & (input_ids[:, 1:] != -100)
    if segments is not None:
        if segments.shape != input_ids.shape or segments.dtype != torch.long or segments.device != input_ids.device:
            raise ValueError('segment shape/dtype/device mismatch')
        active &= segments[:, :-1] == segments[:, 1:]
    y = input_ids[:, 1:][active]
    if not y.numel() or (y < 0).any() or (y >= student_head.shape[0]).any():
        raise ValueError('no usable shifted targets or targets outside vocabulary')
    s = student_hidden[:, :-1][active]
    t = teacher_hidden.detach()[:, :-1][active.to(teacher_hidden.device)]
    frozen_head = teacher_head.detach()

    def chunk_loss(sh, weight, th, labels):
        if work_counter is not None:
            work_counter['student_head_positions'] = work_counter.get('student_head_positions', 0) + sh.shape[0]
        ss = F.linear(sh, weight).float()
        with torch.no_grad():
            if work_counter is not None:
                work_counter['teacher_head_positions'] = work_counter.get('teacher_head_positions', 0) + th.shape[0]
            tt = F.linear(th, frozen_head).float().to(ss.device)
        if not torch.isfinite(ss).all() or not torch.isfinite(tt).all():
            raise ValueError('nonfinite projected logits')
        ce = F.cross_entropy(ss, labels, reduction='sum')
        kl = F.kl_div(F.log_softmax(ss/temperature, -1), F.log_softmax(tt/temperature, -1),
                      log_target=True, reduction='sum') * temperature**2
        return (1-alpha)*ce + alpha*kl

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, y.numel(), chunk_tokens):
        args = (s[start:start+chunk_tokens], student_head, t[start:start+chunk_tokens], y[start:start+chunk_tokens])
        if checkpoint_chunks and torch.is_grad_enabled() and (s.requires_grad or student_head.requires_grad):
            total = total + checkpoint(chunk_loss, *args, use_reentrant=False)
        else:
            total = total + chunk_loss(*args)
    return total / y.numel()
