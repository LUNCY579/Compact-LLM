"""Measured resource accounting. Do not substitute estimates for peak VRAM."""

from __future__ import annotations

import time
from typing import Any

import torch

from rineya_fwa.arms import count_fwa_tensors, count_trainable
from rineya_fwa.wrapper import InstalledFWA


def kv_cache_bytes(seq_len: int, *, layers: int = 28, kv_heads: int = 8, head_dim: int = 128) -> int:
    # K and V, fp16, all layers
    return layers * 2 * kv_heads * seq_len * head_dim * 2


def recurrent_state_bytes(handle: InstalledFWA | None, batch: int = 1) -> int:
    if handle is None:
        return 0
    return sum(block.state_bytes(batch) for block in handle.blocks())


def profile_forward(
    model: Any,
    tokenizer: Any,
    handle: InstalledFWA | None,
    *,
    seq_len: int,
    device: str,
) -> dict[str, Any]:
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((1, seq_len), pad_id, dtype=torch.long, device=device)
    input_ids[0, -8:] = tokenizer.eos_token_id
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids[:, : min(seq_len, 64)],
            max_new_tokens=8,
            do_sample=False,
            use_cache=False,
            pad_token_id=pad_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_dt = time.perf_counter() - t1
    return {
        "seq_len": seq_len,
        "forward_seconds": dt,
        "tokens_per_sec": seq_len / dt if dt > 0 else None,
        "generate_8_seconds": gen_dt,
        "ttft_proxy_seconds": dt,
        "peak_vram_bytes": peak,
        "trainable_parameters": count_trainable(handle),
        "fwa_parameters": count_fwa_tensors(handle),
        "recurrent_state_bytes": recurrent_state_bytes(handle),
        "kv_cache_bytes_if_used": kv_cache_bytes(seq_len),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
