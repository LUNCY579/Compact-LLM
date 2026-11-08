"""Matched injection interfaces and finite recurrence; independent implementation.

The orthogonal formula follows xLLM-Loop Part II. Low-rank anchor adaptation and
zero-start interpolation are Rineya adaptations, not exact paper reproduction.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def finite_tensor(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be a finite floating tensor")


def orthogonal_injection(retained: Tensor, anchor: Tensor, eps: float = 1e-8) -> Tensor:
    """Per-token Q_anchor(retained) + anchor; preserve float64 for gradcheck."""
    finite_tensor(retained, "retained")
    finite_tensor(anchor, "anchor")
    if retained.shape != anchor.shape or retained.ndim < 1:
        raise ValueError("matching non-scalar shapes required")
    if retained.dtype != anchor.dtype or retained.device != anchor.device:
        raise ValueError("dtype and device must match")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    dtype = torch.float64 if retained.dtype == torch.float64 else torch.float32
    v, q = retained.to(dtype), anchor.to(dtype)
    coefficient = (v * q).sum(-1, keepdim=True) / (q.square().sum(-1, keepdim=True) + eps)
    result = (v - coefficient * q).to(retained.dtype) + anchor
    finite_tensor(result, "injected output")
    return result


class InjectionInterface(nn.Module):
    """Identical parameterization for additive and orthogonal arms.

    q = softplus(dt) * (x + U V x)
    v = exp(-softplus(dt) * exp(a)) * h
    target = v+q or Q_q(v)+q
    output = h + tanh(blend) * (target-h)

    blend=0 preserves the un-injected extra application, NOT the intact model.
    Only blend generally gets a gradient on the first step. Up starts at zero;
    after blend and Up open, Down can learn. Log these quantities explicitly.
    """
    def __init__(self, width: int, rank: int = 8, mode: str = "orthogonal", eps: float = 1e-8):
        super().__init__()
        if type(width) is not int or type(rank) is not int or not 1 <= rank <= width:
            raise ValueError("1 <= rank <= width required")
        if mode not in {"additive", "orthogonal"}:
            raise ValueError("unknown injection mode")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("invalid epsilon")
        self.mode, self.eps = mode, eps
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.dt = nn.Parameter(torch.full((width,), math.log(math.expm1(0.1))))
        self.a = nn.Parameter(torch.full((width,), math.log(0.1)))
        self.blend = nn.Parameter(torch.zeros(()))

    def forward(self, state: Tensor, anchor: Tensor) -> Tensor:
        finite_tensor(state, "state")
        finite_tensor(anchor, "anchor")
        if state.shape != anchor.shape or state.shape[-1] != self.down.in_features:
            raise ValueError("interface shape mismatch")
        if state.dtype != self.dt.dtype or anchor.dtype != state.dtype or anchor.device != state.device or state.device != self.dt.device:
            raise ValueError("explicit interface dtype/device conversion required")
        dt = F.softplus(self.dt)
        q = dt * (anchor + self.up(self.down(anchor)))
        v = torch.exp(-dt * self.a.exp()) * state
        target = v + q if self.mode == "additive" else orthogonal_injection(v, q, self.eps)
        result = state + self.blend.tanh() * (target - state)
        finite_tensor(result, "interface output")
        return result


def finite_recurrence(core: Callable[[Tensor], Tensor], state: Tensor, anchor: Tensor,
                      *, steps: int, interface: InjectionInterface | None = None,
                      alpha: float = 1.0, grad_window: int | None = None) -> Tensor:
    """Reference EXTRA-step engine. Does not license additional native depths.

    Early state history is detached, but the anchor is re-injected with its live
    gradient in each retained step. The caller owns the ordinary first pass.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if grad_window is not None and (type(grad_window) is not int or grad_window < 1):
        raise ValueError("grad_window must be positive or None")
    if not math.isfinite(alpha) or not 0 <= alpha <= 2:
        raise ValueError("alpha outside [0,2]")
    finite_tensor(state, "state")
    for r in range(steps):
        keep = grad_window is None or r >= steps - grad_window
        if not keep:
            with torch.no_grad():
                injected = state if interface is None else interface(state, anchor)
                proposal = core(injected)
                state = state + alpha * (proposal - state)
            state = state.detach()
        else:
            injected = state if interface is None else interface(state, anchor)
            proposal = core(injected)
            state = state + alpha * (proposal - state)
        finite_tensor(state, "recurrent state")
    return state


def kv_geometry(ordinary_layers: int, recurrent_layers: int, applications: int) -> dict:
    """Hypothetical equal-geometry bank counts, NOT measured VRAM or speed."""
    if any(type(v) is not int or v < 1 for v in (ordinary_layers, recurrent_layers, applications)):
        raise ValueError("positive integer geometry required")
    if recurrent_layers > ordinary_layers:
        raise ValueError("recurrent block exceeds model")
    full = ordinary_layers + recurrent_layers * (applications - 1)
    return {"full_banks": full, "terminal_banks": ordinary_layers,
            "kv_fraction_saved": 1 - ordinary_layers / full,
            "measured": False, "assumption": "equal KV geometry; a qualified cached host exists"}
