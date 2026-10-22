"""Exact full-vocabulary objectives with a frozen output projection.

The custom first-order backward stores O(T*H), not O(T*V), state. Every chunk
normalizes over ALL vocabulary entries. Not top-k distillation. No higher-order
autograd is supported. Equivalence and gradcheck tests use independent references.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .contracts import require


def _check(hidden, weight, chunk):
    require(hidden.ndim == weight.ndim == 2 and hidden.shape[1] == weight.shape[1]
            and hidden.shape[0] > 0 and weight.shape[0] > 1, "projection shapes")
    require(hidden.dtype == weight.dtype and hidden.device == weight.device, "projection dtype/device")
    require(hidden.is_floating_point() and not weight.requires_grad, "output head must be frozen")
    require(type(chunk) is int and chunk > 0, "positive chunk required")
    require(bool(torch.isfinite(hidden).all()) , "nonfinite projection")


def _work(logits):
    return logits if logits.dtype == torch.float64 else logits.float()


class _ForwardKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, weight, chunk):
        _check(student, weight, chunk)
        require(teacher.shape == student.shape and teacher.dtype == student.dtype
                and teacher.device == student.device and bool(torch.isfinite(teacher).all()), "teacher hidden contract")
        total = torch.zeros((), dtype=torch.float64 if student.dtype == torch.float64 else torch.float32,
                            device=student.device)
        n = len(student)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            lp = _work(torch.nn.functional.linear(student[start:stop], weight)).log_softmax(-1)
            lq = _work(torch.nn.functional.linear(teacher[start:stop], weight)).log_softmax(-1)
            p, q = lp.exp(), lq.exp()
            total += (q * (lq - lp)).sum()
        ctx.save_for_backward(student, teacher, weight)
        ctx.chunk = chunk
        return total / n

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        student, teacher, weight = ctx.saved_tensors
        grad = torch.empty_like(student)
        for start in range(0, len(student), ctx.chunk):
            stop = min(len(student), start + ctx.chunk)
            lp = _work(torch.nn.functional.linear(student[start:stop], weight)).log_softmax(-1)
            lq = _work(torch.nn.functional.linear(teacher[start:stop], weight)).log_softmax(-1)
            residual = (lp.exp() - lq.exp()) * (scale / len(student))
            grad[start:stop] = residual.to(weight.dtype) @ weight
        return grad, None, None, None


class _TokenCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        _check(hidden, weight, chunk)
        require(targets.shape == (len(hidden),) and targets.dtype == torch.long
                and targets.device == hidden.device, "CE target contract")
        require(bool(((targets >= 0) & (targets < len(weight))).all()), "CE token range")
        total = torch.zeros((), dtype=torch.float64 if hidden.dtype == torch.float64 else torch.float32,
                            device=hidden.device)
        for start in range(0, len(hidden), chunk):
            stop = min(len(hidden), start + chunk)
            lp = _work(torch.nn.functional.linear(hidden[start:stop], weight)).log_softmax(-1)
            y = targets[start:stop]
            total -= lp.gather(1, y[:, None]).sum()
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk = chunk
        return total / len(hidden)

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        hidden, weight, targets = ctx.saved_tensors
        grad = torch.empty_like(hidden)
        for start in range(0, len(hidden), ctx.chunk):
            stop = min(len(hidden), start + ctx.chunk)
            lp = _work(torch.nn.functional.linear(hidden[start:stop], weight)).log_softmax(-1)
            y = targets[start:stop]
            residual = lp.exp()
            residual.scatter_add_(1, y[:, None], -torch.ones_like(y[:, None], dtype=residual.dtype))
            grad[start:stop] = (residual * (scale / len(hidden))).to(weight.dtype) @ weight
        return grad, None, None, None


def guidance_kl(student, teacher, weight, chunk: int = 8):
    """Sequence-mean FKL(q||p); teacher gradients are stopped unconditionally."""
    return _ForwardKL.apply(student, teacher.detach(), weight, chunk)


def token_ce(hidden, weight, targets, chunk: int = 8):
    return _TokenCE.apply(hidden, weight, targets, chunk)
