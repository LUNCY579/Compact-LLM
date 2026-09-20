"""Eq. (3) of Ortiz-Jimenez et al. (2305.12827v3), using torch.func.

The anchor stays fixed. Deltas are direct parameters, NOT a LoRA product with
one zero factor (which would have a different tangent space). No Jacobian is
materialized; unsupported forward-AD operators fail rather than approximate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp

Delta = dict[str, Tensor]


def tensor_digest(tensors: Mapping[str, Tensor]) -> str:
    """Canonical value hash, independent of safetensors metadata ordering."""
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        t = value.detach().cpu().contiguous()
        h.update(f"{name}\0{t.dtype}\0{tuple(t.shape)}\0".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass
class CallBudget:
    """Logical API invocations, NOT GPU forward-equivalent cost estimates."""
    limit: int = 100000
    nonlinear: int = 0
    jvp: int = 0

    def consume(self, mode: str) -> None:
        if self.nonlinear + self.jvp >= self.limit:
            raise RuntimeError("logical model-call budget exhausted")
        if mode == "jvp":
            self.jvp += 1
        elif mode == "nonlinear":
            self.nonlinear += 1
        else:
            raise ValueError(mode)


class FrozenAnchor:
    """Functional module with unique tied parameters and an immutable snapshot.

    Supply a deterministic eval-mode module returning a Tensor or .logits.
    Keep all arrays on the same device. Full snapshot memory is intentional:
    host qualification must account for the module AND this copy.
    """
    def __init__(self, model: nn.Module, scope: list[str] | None = None,
                 budget: CallBudget | None = None):
        self.model = model.eval()
        named = dict(model.named_parameters())  # removes tied aliases
        if not named:
            raise ValueError("model has no parameters")
        self.scope = tuple(sorted(named if scope is None else scope))
        if not self.scope or len(set(self.scope)) != len(self.scope):
            raise ValueError("scope must be nonempty and unique")
        if set(self.scope) - named.keys():
            raise ValueError("scope contains unknown or noncanonical tied parameter")
        if any(p.dtype not in (torch.float32, torch.float64) for p in named.values()):
            raise ValueError("anchor must be FP32 or FP64; no silent low-precision merge")
        if any(not torch.isfinite(p).all() for p in named.values()):
            raise ValueError("nonfinite anchor")
        self.params = {k: v.detach().clone() for k, v in named.items()}
        self.buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        self.budget = budget or CallBudget()
        self.initial_digest = tensor_digest(self.params)
        self.initial_buffers_digest = tensor_digest(self.buffers)

    def zero(self, *, trainable: bool = False) -> Delta:
        return {k: torch.zeros_like(self.params[k], requires_grad=trainable)
                for k in self.scope}

    def validate(self, delta: Mapping[str, Tensor]) -> None:
        if set(delta) != set(self.scope):
            raise ValueError("delta keys must exactly match canonical scope")
        for k in self.scope:
            p, d = self.params[k], delta[k]
            if d.shape != p.shape or d.dtype != p.dtype or d.device != p.device:
                raise ValueError(f"shape/dtype/device mismatch: {k}")
            if not torch.isfinite(d).all():
                raise ValueError(f"nonfinite delta: {k}")

    def _function(self, scoped: tuple[Tensor, ...], args: tuple,
                  kwargs: dict) -> Tensor:
        state = dict(self.params)
        state.update(zip(self.scope, scoped))
        # Prevent model buffer updates from modifying the anchor snapshot.
        buffers = {k: v.clone() for k, v in self.buffers.items()}
        out = functional_call(self.model, (state, buffers), args, kwargs,
                              tie_weights=True, strict=True)
        return out if isinstance(out, Tensor) else out.logits

    def evaluate(self, delta: Mapping[str, Tensor], *args,
                 mode: str = "nonlinear", **kwargs) -> Tensor:
        if self.model.training:
            raise RuntimeError("stochastic training mode is not eligible")
        self.validate(delta)
        p0 = tuple(self.params[k] for k in self.scope)
        d = tuple(delta[k] for k in self.scope)
        fn = lambda p: self._function(p, args, kwargs)
        if mode == "nonlinear":
            self.budget.consume("nonlinear")
            out = fn(tuple(p + v for p, v in zip(p0, d)))
        elif mode == "linear":
            self.budget.consume("jvp")
            base, response = jvp(fn, (p0,), (d,))
            out = base + response  # response MUST retain the gradient w.r.t. d
        else:
            raise ValueError("mode must be nonlinear or linear")
        if not torch.isfinite(out).all():
            raise FloatingPointError("nonfinite model output")
        return out

    def assert_unchanged(self) -> None:
        if (tensor_digest(self.params) != self.initial_digest or
                tensor_digest(self.buffers) != self.initial_buffers_digest):
            raise RuntimeError("frozen anchor mutated")


def combine(*terms: tuple[float, Mapping[str, Tensor]]) -> Delta:
    if not terms:
        raise ValueError("no task vectors")
    keys = set(terms[0][1])
    if any(not math.isfinite(a) or set(d) != keys for a, d in terms):
        raise ValueError("incompatible vectors or nonfinite coefficient")
    out = {k: torch.zeros_like(terms[0][1][k]) for k in sorted(keys)}
    for a, d in terms:
        for k in out:
            if (d[k].shape, d[k].dtype, d[k].device) != (out[k].shape, out[k].dtype, out[k].device):
                raise ValueError("task vectors must share a coordinate system")
            out[k] = out[k] + a * d[k]
    return out
