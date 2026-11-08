"""Dense numerical oracles, NOT a FlashAttention implementation.

Equations: arXiv:2609.34272v1, sections 3--4 and Appendix B. BF16 operands
are explicitly formed, then promoted for contractions. This tests the algebra,
not the accumulation order, LSE reconstruction or throughput of a fused kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import Tensor


@dataclass
class Case:
    q: Tensor
    k: Tensor
    v: Tensor
    u: Tensor
    mask: Tensor | None = None
    causal: bool = False
    scale: float | None = None
    saved_output: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, max_score_elements: int = 2_000_000) -> None:
        if any(x.ndim != 4 for x in (self.q, self.k, self.v, self.u)):
            raise ValueError("Expected [batch, heads, tokens, width] tensors")
        b, h, n, d = self.q.shape
        bk, hk, m, dk = self.k.shape
        if min(b, h, n, d, hk, m, self.v.shape[-1]) < 1:
            raise ValueError("Empty tensor dimension")
        if bk != b or dk != d or h % hk or self.v.shape[:3] != (b, hk, m):
            raise ValueError("Incompatible Q/K/V shapes or GQA head ratio")
        if self.u.shape != (b, h, n, self.v.shape[-1]):
            raise ValueError("Incoming derivative does not match attention output")
        if b * h * n * m > max_score_elements:
            raise ValueError("Dense audit score-element budget exceeded; do not truncate captures")
        if len({x.device for x in (self.q, self.k, self.v, self.u)}) != 1:
            raise ValueError("Operands must share a device")
        if len({x.dtype for x in (self.q, self.k, self.v, self.u)}) != 1:
            raise ValueError("Operands must share their represented dtype")
        for x in (self.q, self.k, self.v, self.u):
            if not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError("Non-floating or nonfinite operand")
        if self.scale is not None and (not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("Scale must be finite and positive")
        if self.mask is not None:
            if self.mask.device != self.q.device:
                raise ValueError("Mask device mismatch")
            try:
                shape = torch.broadcast_shapes(self.mask.shape, (b, h, n, m))
            except RuntimeError as exc:
                raise ValueError("Mask cannot broadcast to score shape") from exc
            if shape != (b, h, n, m):
                raise ValueError("Mask expands the score shape")
            if self.mask.dtype != torch.bool:
                if not self.mask.is_floating_point():
                    raise ValueError("Mask must be boolean or additive floating point")
                if torch.isnan(self.mask).any() or torch.isposinf(self.mask).any():
                    raise ValueError("Mask contains NaN or positive infinity")
        if self.saved_output is not None:
            if self.saved_output.shape != self.u.shape or not torch.isfinite(self.saved_output).all():
                raise ValueError("Invalid saved output")
            if self.saved_output.device != self.q.device:
                raise ValueError("Saved output device mismatch")

    @property
    def alpha(self) -> float:
        return self.scale if self.scale is not None else self.q.shape[-1] ** -0.5


def expand_kv(x: Tensor, query_heads: int) -> Tensor:
    return x.repeat_interleave(query_heads // x.shape[1], dim=1)


def reduce_kv(x: Tensor, kv_heads: int) -> Tensor:
    b, h, n, d = x.shape
    return x.reshape(b, kv_heads, h // kv_heads, n, d).sum(dim=2)


def effective_mask(case: Case, dtype: torch.dtype) -> Tensor | None:
    """PyTorch upper-left rectangular causal convention; True means allowed."""
    n, m = case.q.shape[-2], case.k.shape[-2]
    mask = case.mask
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(dtype)
    if case.causal:
        support = torch.ones(n, m, dtype=torch.bool, device=case.q.device).tril()
        if mask is None:
            mask = support
        elif mask.dtype == torch.bool:
            mask = mask & support
        else:
            mask = mask.masked_fill(~support, -torch.inf)
    return mask


def probabilities(case: Case, dtype: torch.dtype = torch.float64) -> Tensor:
    """Stable dense softmax with explicit zero output for all-masked rows."""
    q = case.q.to(dtype)
    k = expand_kv(case.k.to(dtype), q.shape[1])
    logits = q @ k.transpose(-1, -2)
    mask = effective_mask(case, dtype)
    # Subtract before scale; additive bias is represented in unscaled units.
    if mask is not None:
        logits = (logits.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool
                  else logits + mask / case.alpha)
    maximum = logits.amax(dim=-1, keepdim=True)
    live = torch.isfinite(maximum)
    scores = (logits - torch.where(live, maximum, 0.0)) * case.alpha
    scores = torch.where(live, scores, 0.0)
    return torch.softmax(scores, dim=-1) * live.to(dtype)


def reference(case: Case, dtype: torch.dtype = torch.float64) -> dict[str, Tensor]:
    case.validate()
    p = probabilities(case, dtype)
    q, u = case.q.to(dtype), case.u.to(dtype)
    k, v = (expand_kv(x.to(dtype), q.shape[1]) for x in (case.k, case.v))
    a = u @ v.transpose(-1, -2)
    g = p * (a - (p * a).sum(dim=-1, keepdim=True))
    return {
        "output": p @ v,
        "dq": case.alpha * (g @ k),
        "dk": reduce_kv(case.alpha * (g.transpose(-1, -2) @ q), case.k.shape[1]),
        "dv": reduce_kv(p.transpose(-1, -2) @ u, case.v.shape[1]),
        "p": p, "g": g,
    }


def project_contracts(t: Tensor, r: Tensor, q: Tensor, k: Tensor,
                      alpha: float, mode: str = "gproj",
                      round_key_operand: bool = False) -> dict[str, Tensor]:
    """Correct contractions AFTER the cast, before the final gradient cast.

    Inputs t/r must already be the consumed, promoted BF16 operands.
    Projection is along r, not a uniform mean subtraction. Division uses the
    actual r mass. Do not cast the projected score matrix back to BF16.
    """
    if mode not in {"cast_only", "gproj_q", "gproj"}:
        raise ValueError("Unknown correction mode")
    rho, mass = t.sum(-1, keepdim=True), r.sum(-1, keepdim=True)
    lam = torch.where(mass > 0, rho / mass.clamp_min(torch.finfo(r.dtype).tiny), 0.0)
    dq = t @ k
    dk = t.transpose(-1, -2) @ q
    if mode != "cast_only":
        dq = dq - lam * (r @ k)
    if mode == "gproj":
        z = lam * q
        if round_key_operand:
            z = z.to(torch.bfloat16).to(q.dtype)
        dk = dk - r.transpose(-1, -2) @ z
    return {"dq": alpha * dq, "dk": alpha * dk, "rho": rho,
            "mass": mass, "lambda": lam, "projected_mass": (t - lam * r).sum(-1)}


def emulate(case: Case, mode: str = "gproj", center: str = "saved",
            accumulation_dtype: torch.dtype = torch.float32,
            round_key_operand: bool = True) -> dict[str, Tensor]:
    """Dense BF16-cast model with a shared repaired forward across modes.

    center='exact' recomputes the probability-weighted center in the chosen
    accumulation dtype; use FP64 to isolate the cast. center='saved' also
    includes the saved-output reduction channel. A supplied saved_output may be native, but
    probabilities are still dense reconstructions, NOT native internal LSE.
    """
    case.validate()
    if center not in {"saved", "exact"}:
        raise ValueError("Unknown centering contract")
    if accumulation_dtype not in {torch.float32, torch.float64}:
        raise ValueError("Accumulate only in FP32 or FP64")
    dtype = accumulation_dtype
    q, u = case.q.to(dtype), case.u.to(dtype)
    k, v = (expand_kv(x.to(dtype), q.shape[1]) for x in (case.k, case.v))
    p = probabilities(case, dtype)
    output = (p @ v).to(case.q.dtype)
    a = u @ v.transpose(-1, -2)
    saved = output if case.saved_output is None else case.saved_output
    delta = ((p * a).sum(-1, keepdim=True) if center == "exact"
             else (u * saved.to(dtype)).sum(-1, keepdim=True))
    t = (p * (a - delta)).to(torch.bfloat16).to(dtype)
    r = p.to(torch.bfloat16).to(dtype)
    result = project_contracts(t, r, q, k, case.alpha, mode, round_key_operand)
    result["dk"] = reduce_kv(result["dk"], case.k.shape[1])
    result.update(output=output, dv=reduce_kv(r.transpose(-1, -2) @ u, case.v.shape[1]),
                  t=t, r=r)
    return result


def error_metrics(actual: Tensor, expected: Tensor, zero_atol: float = 1e-12) -> dict:
    if actual.shape != expected.shape:
        raise ValueError("Metric shape mismatch")
    a, b = actual.detach().double().reshape(-1), expected.detach().double().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"finite": False, "relative_l2": None, "cosine": None}
    na, nb, ne = (float(torch.linalg.vector_norm(x)) for x in (a, b, a - b))
    zero = nb <= zero_atol
    return {"finite": True, "reference_norm": nb, "actual_norm": na,
            "absolute_l2": ne, "max_absolute": float((a - b).abs().max()),
            "relative_l2": None if zero else ne / nb,
            "cosine": None if zero or na <= zero_atol else float((a @ b) / (na * nb)),
            "near_zero_reference": zero}


def witness(width: int = 64, offset: float = 65536.0) -> Case:
    """Appendix A's exact BF16 four-key witness (dq[0] = -24 without repair)."""
    if width < 2:
        raise ValueError("Witness needs width >= 2")
    q = torch.zeros(1, 1, 1, width, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 4, width, dtype=torch.bfloat16)
    k[..., 0] = offset
    k[0, 0, :, 1] = torch.arange(4)
    v = torch.zeros_like(k)
    eta = 3 / 1024
    v[0, 0, 0, :2] = torch.tensor([4.0, 4 * eta])
    v[0, 0, 1, 0], v[0, 0, 2, 0] = -4.0, -4 * eta
    u = torch.zeros_like(q)
    u[..., :2] = 1
    return Case(q, k, v, u, scale=1 / 8, metadata={"origin": "synthetic_witness"})
