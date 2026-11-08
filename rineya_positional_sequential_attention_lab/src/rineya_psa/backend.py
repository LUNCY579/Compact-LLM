"""SDPA backend selection. Import this and use it around every forward.

Why this module exists
----------------------
Left to itself, PyTorch's SDPA dispatcher picks the **math** backend on this machine,
and the math backend materialises the full ``[B, H, T, T]`` attention matrix. Measured
on the RTX 4070 Laptop with Qwen3-0.6B, bf16, batch 1, ``logits_to_keep=8``:

===========  ==========================  ============================
sequence T   default dispatch            forced cuDNN/mem-efficient
===========  ==========================  ============================
2048         0.64 s / 1.94 GB            0.119 s / 1.26 GB
4096         2.30 s / 3.91 GB            0.288 s / 1.31 GB
8192         54.2 s / 11.6 GB (spills)   0.834 s / 1.42 GB
16384        out of memory               2.27 s / 1.65 GB
32768        out of memory               7.06 s / 2.10 GB
===========  ==========================  ============================

Eight times faster and three times lighter, and it is the difference between "16K
contexts are impossible on this card" and "16K contexts cost two seconds". Backend
availability here, measured on a ``[1, 16, 4096, 128]`` problem:

* ``FLASH_ATTENTION`` — **unavailable** (no kernel; this is a Windows wheel)
* ``CUDNN_ATTENTION`` — 1.8 ms, 0.09 GB
* ``EFFICIENT_ATTENTION`` — 4.5 ms, 0.08 GB
* ``MATH`` — 82 ms, 2.69 GB

Because FlashAttention is missing, the dispatcher falls through to MATH rather than to
the two good backends, and it does so silently. Hence the explicit preference list.

One trap worth naming: ``scaled_dot_product_attention(..., enable_gqa=True)`` also
forces the MATH path here (106 ms, 2.73 GB). Expand KV heads with ``repeat_kv``
instead — that is what ``hooks/registry.py`` does.

Passing an explicit additive ``attn_mask`` does **not** cost the fast path as long as
the backend is forced (0.36 s / 1.31 GB at T=4096), which is what makes the Mode-1
logit-bias intervention affordable.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

# Preference order. cuDNN first (fastest measured), mem-efficient as the fallback that
# is always present on CUDA builds. MATH is deliberately excluded: including it lets the
# dispatcher choose it, which is the failure this module exists to prevent.
_PREFERRED = ("CUDNN_ATTENTION", "EFFICIENT_ATTENTION")

_ENV_DISABLE = "PSA_NO_BACKEND_PIN"


def _backends() -> list[Any]:
    from torch.nn.attention import SDPBackend

    return [getattr(SDPBackend, name) for name in _PREFERRED if hasattr(SDPBackend, name)]


@contextlib.contextmanager
def efficient_attention(*, required: bool = False, active: bool | None = None) -> Iterator[bool]:
    """Pin SDPA to the fast backends for the duration of the block.

    ``active`` must be True only when the model being run is **on CUDA**. Pinning is
    process-global, and cuDNN/mem-efficient have no CPU implementation, so pinning
    around a CPU forward leaves SDPA with no viable backend at all and raises. Pass
    ``active=<model is on cuda>``; the default (``None``) falls back to "is there a
    CUDA device", which is right for a script that has just loaded onto the GPU and
    wrong for a test running a tiny model on the CPU of a CUDA machine.

    Yields whether the pin is actually in force. Set ``PSA_NO_BACKEND_PIN=1`` to
    disable it, which exists so the cost of the pin can itself be measured rather than
    assumed. With ``required=True`` a machine that offers neither backend raises
    instead of silently running eight times slower — use that on long sweeps, where a
    silent fallback would waste hours.

    CPU-only runs are a no-op: the context manager is CUDA-specific and the tiny models
    in ``tests/`` do not need it.
    """
    import torch

    if active is None:
        active = torch.cuda.is_available()
    if os.environ.get(_ENV_DISABLE) == "1" or not active or not torch.cuda.is_available():
        yield False
        return

    backends = _backends()
    if not backends:
        if required:
            raise RuntimeError(
                "neither CUDNN_ATTENTION nor EFFICIENT_ATTENTION is available; refusing "
                "to fall through to the MATH backend on a long run "
                f"(set {_ENV_DISABLE}=1 to override)"
            )
        yield False
        return

    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(backends):
        yield True


def probe_backends(seq_len: int = 4096, n_q_heads: int = 16, head_dim: int = 128) -> dict[str, Any]:
    """Measure which backends this machine actually offers, and how they compare.

    Recorded into run provenance so a result from another machine can be read against
    the backend it was produced on rather than against an assumption about it.
    """
    import time

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        return {"cuda": False}

    q = torch.randn(1, n_q_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out: dict[str, Any] = {"cuda": True, "device": torch.cuda.get_device_name(0)}
    for name in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        be = getattr(SDPBackend, name, None)
        if be is None:
            out[name] = {"available": False, "reason": "not in this torch build"}
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with sdpa_kernel(be), torch.no_grad():
                for _ in range(3):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
            out[name] = {
                "available": True,
                "ms": round(dt * 1e3, 3),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4),
            }
        except Exception as exc:  # noqa: BLE001 - availability probe, any failure means "no"
            out[name] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    del q, k, v
    torch.cuda.empty_cache()
    out["pinned"] = [n for n in _PREFERRED if out.get(n, {}).get("available")]
    return out


__all__ = ["efficient_attention", "probe_backends"]
