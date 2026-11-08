"""Single-device FP32 AdamW with CPU moments and bounded per-tensor transfers.

No disk spill, foreach temporaries or implicit CUDA precision changes. Matched against
PyTorch AdamW in tests. A failure during a step invalidates that in-memory run; resume
only from a COMPLETE boundary checkpoint in a NEW attempt directory.
"""
from __future__ import annotations

import math
import torch
from .config import Design


class CpuAdamW:
    def __init__(self, named_parameters, design: Design):
        self.named = list(named_parameters)
        if not self.named or len({n for n, _ in self.named}) != len(self.named):
            raise ValueError("unique named trainable parameters required")
        if len({id(p) for _, p in self.named}) != len(self.named):
            raise ValueError("aliased trainable parameter")
        if any(p.dtype != torch.float32 or not p.requires_grad for _, p in self.named):
            raise ValueError("optimizer accepts trainable FP32 parameters only")
        self.design = design
        self.step_count = 0
        self.state = {name: (torch.zeros_like(p, device="cpu"), torch.zeros_like(p, device="cpu"))
                      for name, p in self.named}

    def zero_grad(self):
        for _, p in self.named:
            p.grad = None

    @torch.no_grad()
    def step(self, lr: float):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("invalid learning rate")
        # Validate all gradients before mutating any state or parameter.
        for name, p in self.named:
            if p.grad is None or p.grad.is_sparse or not torch.isfinite(p.grad).all():
                raise ValueError(f"missing/sparse/nonfinite gradient: {name}")
        d = self.design
        step = self.step_count + 1
        for name, p in self.named:
            grad = p.grad.detach().to("cpu", copy=True)
            weight = p.detach().to("cpu", copy=True)
            m, v = self.state[name]
            weight.mul_(1.0 - lr * d.weight_decay)
            m.lerp_(grad, 1.0 - d.beta1)
            v.mul_(d.beta2).addcmul_(grad, grad, value=1.0 - d.beta2)
            denominator = (v.sqrt() / math.sqrt(1.0 - d.beta2 ** step)).add_(d.epsilon)
            weight.addcdiv_(m, denominator, value=-lr / (1.0 - d.beta1 ** step))
            if not torch.isfinite(weight).all():
                raise FloatingPointError("nonfinite AdamW update; discard uncheckpointed state")
            p.copy_(weight)
        self.step_count = step
