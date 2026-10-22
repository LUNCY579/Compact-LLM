"""Detached trajectory diagnostics: stationarity is never called correctness."""
from __future__ import annotations

import math
import torch
from torch import Tensor
from .core import finite_tensor


@torch.no_grad()
def trajectory_metrics(before: Tensor, injected: Tensor, proposal: Tensor, after: Tensor,
                       anchor: Tensor, normal: Tensor, mask: Tensor | None = None) -> dict:
    tensors = (before, injected, proposal, after, anchor, normal)
    for value in tensors:
        finite_tensor(value, "trajectory")
        if value.shape != before.shape or value.ndim < 2:
            raise ValueError("matching token-by-hidden tensors required")
    selected = torch.ones(before.shape[:-1], device=before.device, dtype=torch.bool) if mask is None else mask
    if selected.shape != before.shape[:-1] or selected.dtype != torch.bool or not selected.any():
        raise ValueError("nonempty boolean token mask required")
    h, c, f, y, q, n = [t.detach().double()[selected] for t in tensors]
    norm = lambda t: torch.linalg.vector_norm(t, dim=-1)
    eps = 1e-12
    raw, applied, operator = f - h, y - h, f - c
    hn, qn = norm(h), norm(q)
    denom = (hn * qn).clamp_min(eps)
    alignment = (h * q).sum(-1) / denom
    raw_norm, applied_norm = norm(raw), norm(applied)
    ratio = applied_norm / raw_norm.clamp_min(eps)
    result = {
        "tokens": int(h.shape[0]),
        "raw_update_norm": float(raw_norm.mean()),
        "applied_update_norm": float(applied_norm.mean()),
        "operator_residual_norm": float(norm(operator).mean()),
        "injection_change_norm": float(norm(c - h).mean()),
        "relative_state_change": float((applied_norm / hn.clamp_min(eps)).mean()),
        "relative_change_max": float((applied_norm / hn.clamp_min(eps)).max()),
        "applied_to_raw_ratio": float(ratio.mean()),
        "suppressed_update_fraction": float(((raw_norm > 1e-8) & (ratio < 0.01)).double().mean()),
        "stationary_fraction": float((applied_norm / hn.clamp_min(eps) < 0.02).double().mean()),
        "anchor_cosine": float(alignment.mean()),
        "anchor_zero_fraction": float((qn < eps).double().mean()),
        "state_rms": float(y.square().mean().sqrt()),
        "rms_ratio_to_normal": float(y.square().mean().sqrt() / n.square().mean().sqrt().clamp_min(eps)),
        "correctness_inferred": False,
    }
    if any(not math.isfinite(v) for v in result.values() if type(v) is float):
        raise FloatingPointError("nonfinite diagnostic")
    return result


@torch.no_grad()
def logit_metrics(reference: Tensor, candidate: Tensor) -> dict:
    finite_tensor(reference, "reference logits")
    finite_tensor(candidate, "candidate logits")
    if reference.shape != candidate.shape or reference.ndim < 1:
        raise ValueError("matching logits required")
    p, q = reference.double().log_softmax(-1), candidate.double().log_softmax(-1)
    return {"kl_reference_to_candidate": float((p.exp() * (p - q)).sum(-1).mean()),
            "top1_agreement": float((p.argmax(-1) == q.argmax(-1)).double().mean()),
            "max_abs_logit_difference": float((reference.double() - candidate.double()).abs().max())}
