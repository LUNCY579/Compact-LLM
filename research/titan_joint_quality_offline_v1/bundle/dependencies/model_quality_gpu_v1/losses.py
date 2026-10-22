"""Sum-loss chunks with one denominator per update, not a mean of row means."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def chunked_sum(hidden, head, labels, chunk_size: int, *, recompute=True):
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("positive CE chunk required")
    if (hidden.ndim != 3 or labels.shape != hidden.shape[:2]
            or labels.dtype != torch.long or head.ndim != 2
            or hidden.shape[-1] != head.shape[-1] or hidden.shape[1] < 2
            or hidden.device != head.device or labels.device != hidden.device):
        raise ValueError("invalid hidden/head/label layout")
    target = labels[:, 1:].reshape(-1)
    valid = target != -100
    if not bool(valid.any()) or bool(((target[valid] < 0) | (target[valid] >= head.shape[0])).any()):
        raise ValueError("empty or invalid next-token targets")
    states = hidden[:, :-1].reshape(-1, hidden.shape[-1])[valid]
    target = target[valid]

    def part_loss(x, w, y):
        return F.cross_entropy(F.linear(x, w).float(), y, reduction="sum")

    total = None
    for start in range(0, target.numel(), chunk_size):
        args = (states[start:start + chunk_size], head, target[start:start + chunk_size])
        if recompute and torch.is_grad_enabled():
            part = checkpoint(part_loss, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            part = part_loss(*args)
        total = part if total is None else total + part
    return total, target.numel()
