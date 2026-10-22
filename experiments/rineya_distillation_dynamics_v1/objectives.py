"""Exact-vocabulary first-order KL/CE with frozen, potentially DIFFERENT heads.

Only response positions enter these functions. Chunks span tokens, never a top-k
vocabulary approximation. The teacher may stay on CPU. Backward recomputes full
vocabulary probabilities, storing O(T*(Hs+Ht)), not O(T*V), activations here.
This does not eliminate transformer activations, model weights, or compute cost.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .contracts import require


def work(t):
    return t if t.dtype == torch.float64 else t.float()


def logprobs(hidden, weight):
    z = work(torch.nn.functional.linear(hidden, weight))
    require(bool(torch.isfinite(z).all()), "nonfinite logits")
    return z.log_softmax(-1)


def check(hidden, weight, chunk):
    require(hidden.ndim == weight.ndim == 2 and hidden.shape[1] == weight.shape[1]
            and hidden.shape[0] > 0 and weight.shape[0] > 1, "projection dimensions")
    require(hidden.device == weight.device and hidden.dtype == weight.dtype and hidden.is_floating_point(),
            "projection dtype/device")
    require(not weight.requires_grad, "output projection must remain frozen")
    require(type(chunk) is int and chunk > 0, "positive token chunk")


class _KL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sh, sw, th, tw, direction, chunk):
        check(sh, sw, chunk)
        check(th, tw, chunk)
        require(len(sh) == len(th) and len(sw) == len(tw), "teacher/student response or vocabulary mismatch")
        require(direction in {"forward", "reverse"}, "KL direction")
        total = torch.zeros((), dtype=work(sh).dtype, device=sh.device)
        for a in range(0, len(sh), chunk):
            b = a + chunk
            lp = logprobs(sh[a:b], sw)
            lq = logprobs(th[a:b], tw).to(device=sh.device, dtype=lp.dtype)
            total += ((lq.exp() * (lq - lp)) if direction == "forward" else (lp.exp() * (lp - lq))).sum()
        require(bool(torch.isfinite(total)), "nonfinite KL")
        ctx.save_for_backward(sh, sw, th, tw)
        ctx.direction, ctx.chunk = direction, chunk
        return total / len(sh)

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        sh, sw, th, tw = ctx.saved_tensors
        grad = torch.empty_like(sh)
        for a in range(0, len(sh), ctx.chunk):
            b = a + ctx.chunk
            lp = logprobs(sh[a:b], sw)
            lq = logprobs(th[a:b], tw).to(device=sh.device, dtype=lp.dtype)
            p = lp.exp()
            if ctx.direction == "forward":
                dz = p - lq.exp()
            else:
                ratio = lp - lq
                dz = p * (ratio - (p * ratio).sum(-1, keepdim=True))
            grad[a:b] = (dz * (scale / len(sh))).to(sw.dtype) @ sw
        return grad, None, None, None, None, None


def projected_kl(student_hidden, student_weight, teacher_hidden, teacher_weight,
                 direction="forward", chunk=8):
    """Sequence-token mean KL at identical prefixes. No gradient to teacher or heads."""
    return _KL.apply(student_hidden, student_weight, teacher_hidden.detach(), teacher_weight, direction, chunk)


class _CE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, targets, chunk):
        check(h, w, chunk)
        require(targets.shape == (len(h),) and targets.dtype == torch.long and targets.device == h.device,
                "CE target dimensions")
        require(bool(((targets >= 0) & (targets < len(w))).all()), "CE token range")
        total = torch.zeros((), dtype=work(h).dtype, device=h.device)
        for a in range(0, len(h), chunk):
            total -= logprobs(h[a:a+chunk], w).gather(1, targets[a:a+chunk, None]).sum()
        ctx.save_for_backward(h, w, targets)
        ctx.chunk = chunk
        return total / len(h)

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        h, w, y = ctx.saved_tensors
        grad = torch.empty_like(h)
        for a in range(0, len(h), ctx.chunk):
            b = a + ctx.chunk
            dz = logprobs(h[a:b], w).exp()
            dz.scatter_add_(1, y[a:b, None], -torch.ones_like(y[a:b, None], dtype=dz.dtype))
            grad[a:b] = (dz * (scale / len(h))).to(w.dtype) @ w
        return grad, None, None, None


def projected_ce(h, w, targets, chunk=8):
    return _CE.apply(h, w, targets, chunk)


def dense_kl(student_logits, teacher_logits, direction="forward"):
    """Independent autograd reference used ONLY by bounded qualification/tests."""
    require(student_logits.shape == teacher_logits.shape and student_logits.ndim == 2
            and student_logits.shape[0] > 0, "reference logits shape")
    require(direction in {"forward", "reverse"}, "KL direction")
    p = work(student_logits).log_softmax(-1)
    q = work(teacher_logits.detach()).log_softmax(-1)
    return ((q.exp() * (q-p)) if direction == "forward" else (p.exp() * (p-q))).sum(-1).mean()
