"""Explicit SDPA dispatch and opt-in dense backward replacements."""
from __future__ import annotations

from contextlib import contextmanager
import platform
import time

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.autograd.function import once_differentiable

from .core import Case, effective_mask, emulate, probabilities, expand_kv

# Preserve the unwrapped operator so recorder/replacement contexts cannot recurse.
SDPA = F.scaled_dot_product_attention
BACKENDS = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}


@contextmanager
def precision_guard():
    """Restore global flags even on failure; use only in single-threaded experiments."""
    precision = torch.get_float32_matmul_precision()
    reduced = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduced)
        torch.set_float32_matmul_precision(precision)


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "math_reduced_precision": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}


def native_vjp(case: Case, backend: str = "math", fp32_attention: bool = False) -> dict:
    """No silent backend fallback; profiler evidence distinguishes requested/observed."""
    case.validate()
    if backend not in BACKENDS:
        raise ValueError("Choose an explicit backend; auto is not a qualification")
    if backend != "math" and case.q.device.type != "cuda":
        return {"status": "UNAVAILABLE", "backend_requested": backend,
                "reason": "CUDA backend requires CUDA tensors"}
    dtype = torch.float32 if fp32_attention else case.q.dtype
    q, k, v = (x.detach().to(dtype).requires_grad_(True) for x in (case.q, case.k, case.v))
    mask = effective_mask(case, dtype)
    if case.q.is_cuda:
        torch.cuda.synchronize(case.q.device)
    start = time.perf_counter()
    try:
        with precision_guard(), torch.autocast(case.q.device.type, enabled=False):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                with sdpa_kernel(BACKENDS[backend]):
                    out = SDPA(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
                               scale=case.alpha, enable_gqa=q.shape[1] != k.shape[1])
                    # FP32 attention keeps the same represented output boundary.
                    represented_out = out.to(case.q.dtype)
                    grads = torch.autograd.grad(represented_out, (q, k, v), case.u)
        if case.q.is_cuda:
            torch.cuda.synchronize(case.q.device)
    except (RuntimeError, NotImplementedError) as exc:
        if "out of memory" in str(exc).lower():
            raise
        return {"status": "UNAVAILABLE", "backend_requested": backend,
                "reason": str(exc)}
    events = sorted({e.key for e in prof.key_averages() if "scaled_dot_product" in e.key})
    return {"status": "MEASURED", "backend_requested": backend,
            "backend_events": events, "fp32_attention": fp32_attention,
            "output": represented_out.detach(),
            **{name: grad.detach().to(case.q.dtype) for name, grad in zip(("dq", "dk", "dv"), grads)},
            "profiled_seconds_not_benchmark": time.perf_counter() - start}


class _DenseBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, causal, scale, mode):
        # Explicit BF16 boundary: reject accidentally changing a FP32 model to BF16.
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("Dense BF16-cast arms require BF16 Q/K/V")
        u = torch.zeros((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
        case = Case(q, k, v, u, mask, causal, scale)
        case.validate()
        with torch.autocast(q.device.type, enabled=False):
            p = probabilities(case, torch.float32)
            out = (p @ expand_kv(v.float(), q.shape[1])).to(q.dtype)
        ctx.save_for_backward(q, k, v, out)
        ctx.mask = None if mask is None else mask.detach().clone()
        ctx.causal, ctx.scale, ctx.mode = causal, scale, mode
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, u):
        q, k, v, out = ctx.saved_tensors
        case = Case(q, k, v, u.to(q.dtype), ctx.mask, ctx.causal, ctx.scale, out)
        with torch.autocast(q.device.type, enabled=False):
            result = emulate(case, mode=ctx.mode)
        return (*(result[name].to(q.dtype) for name in ("dq", "dk", "dv")),
                None, None, None, None)


def dense_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None,
                    causal: bool = False, scale: float | None = None,
                    mode: str = "gproj") -> Tensor:
    if mode not in {"cast_only", "gproj_q", "gproj"}:
        raise ValueError("Invalid dense backward mode")
    if mask is not None and mask.requires_grad:
        raise ValueError("Learnable attention bias is unsupported")
    return _DenseBackward.apply(q, k, v, mask, causal, scale, mode)
