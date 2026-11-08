"""Titan-specific interventions. These are extensions, not the paper's method."""
from __future__ import annotations
import math
from collections.abc import Mapping
import torch
from torch import Tensor
from .tangent import Delta, combine


def _check(delta: Mapping[str, Tensor]) -> None:
    if not delta or any(d.dtype not in (torch.float32, torch.float64)
                        or not torch.isfinite(d).all() for d in delta.values()):
        raise ValueError("finite FP32/FP64 deltas required")


def head_tail(delta: Mapping[str, Tensor], keep: float) -> tuple[Delta, Delta]:
    _check(delta)
    if not math.isfinite(keep) or not 0 <= keep <= 1:
        raise ValueError("keep must be in [0,1]")
    head, tail = {}, {}
    for k, d in delta.items():
        flat = d.reshape(-1)
        order = torch.argsort(flat.abs(), descending=True, stable=True)
        mask = torch.zeros_like(flat)
        mask[order[:math.floor(keep * flat.numel())]] = 1
        head[k] = (flat * mask).reshape_as(d)
        tail[k] = d - head[k]
    return head, tail


def norm_match(direction: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> Delta:
    """Per-tensor norm control: retain direction, match target Frobenius norm."""
    _check(direction); _check(target)
    combine((0.0, direction), (0.0, target))  # exact compatibility check
    out = {}
    for k, d in direction.items():
        a, b = d.double().norm().item(), target[k].double().norm().item()
        if a == 0 and b != 0:
            raise ValueError(f"zero direction cannot match nonzero norm: {k}")
        out[k] = d * (b / a if a else 0.0)
    return out


def composition_controls(a: Mapping[str, Tensor], b: Mapping[str, Tensor]) -> dict[str, Delta]:
    """Match ||a+b|| to ||a||, AND provide a-only with the identical scale.

    Compare matched_sum against attenuated_a, NOT full a, to isolate adding b
    at fixed a-component strength. Reject cancellation; never hide a fallback.
    """
    _check(a); _check(b)
    raw = combine((1.0, a), (1.0, b))
    matched, only_a, only_b = {}, {}, {}
    for k in raw:
        n, target = raw[k].double().norm().item(), a[k].double().norm().item()
        if n == 0 and (target != 0 or b[k].count_nonzero()):
            raise ValueError(f"sum cancellation: {k}")
        s = target / n if n else 0.0
        matched[k], only_a[k], only_b[k] = s * raw[k], s * a[k], s * b[k]
    return {"raw_sum": raw, "matched_sum": matched,
            "attenuated_a": only_a, "attenuated_b": only_b}


def rank_truncate(delta: Mapping[str, Tensor], rank: int) -> Delta:
    _check(delta)
    if type(rank) is not int or rank < 1:
        raise ValueError("rank must be a positive integer")
    out = {}
    for k, d in delta.items():
        if d.ndim == 1:
            out[k] = d.clone()  # Titan norm-vector convention
        elif d.ndim == 2:
            u, s, vh = torch.linalg.svd(d, full_matrices=False)
            r = min(rank, s.numel())
            out[k] = (u[:, :r] * s[:r]) @ vh[:r]
        else:
            raise ValueError("rank experiment requires vectors or matrices")
    return out


def effective_export_delta(base: Mapping[str, Tensor], delta: Mapping[str, Tensor]) -> Delta:
    combine((0.0, base), (0.0, delta))
    return {k: (base[k] + delta[k]).to(torch.bfloat16).to(base[k].dtype) - base[k]
            for k in delta}
