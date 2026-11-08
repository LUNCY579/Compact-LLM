"""Inspectable NSA reference, not a fused kernel or a KV-eviction algorithm.

Implements the aligned-block special case l = stride = selection block length
of arXiv:2502.11089v2, equations 5, 7, 8, 10-12. Selection is shared within
KV groups. Hard indices are detached; compression learns through its own path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NSAConfig:
    block_size: int = 32
    selected_blocks: int = 8
    window_size: int = 128
    compression_hidden: int = 32
    gate_hidden: int = 32
    query_chunk_size: int = 16
    force_initial_blocks: int = 1
    force_recent_blocks: int = 1
    kv_mode: str = "independent"
    selection_policy: str = "learned"
    control_seed: int = 17
    max_reference_tokens: int = 4096
    max_grad_reference_tokens: int = 256
    max_scratch_bytes: int = 268435456

    def __post_init__(self):
        positive = ("block_size", "selected_blocks", "window_size", "compression_hidden",
                    "gate_hidden", "query_chunk_size", "max_reference_tokens", "max_grad_reference_tokens", "max_scratch_bytes")
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("force_initial_blocks", "force_recent_blocks", "control_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.force_initial_blocks + self.force_recent_blocks > self.selected_blocks:
            raise ValueError("mandatory blocks must fit inside, not be added to, the budget")
        if self.max_grad_reference_tokens > self.max_reference_tokens:
            raise ValueError("gradient ceiling must not exceed the reference token ceiling")
        if self.control_seed > 2147483647:
            raise ValueError("control_seed must fit a signed 31-bit integer")
        if self.kv_mode not in {"independent", "shared"}:
            raise ValueError("kv_mode must be independent or shared")
        if self.selection_policy not in {"learned", "hash_control"}:
            raise ValueError("unknown selection policy")


def check_valid(valid: Tensor, batch: int, length: int) -> None:
    if valid.shape != (batch, length) or valid.dtype != torch.bool:
        raise ValueError("valid_tokens must be Boolean [batch, sequence]")
    if length > 1 and torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("reference v1 supports right padding only; no left padding or packing")


class BlockCompressor(nn.Module):
    """Head-shared intra-block positions + bottleneck MLP + mean residual.

The residual mean/bottleneck parameterization is a declared Rineya adaptation,
not the exact compressor parameterization used for the paper's speed results.
Only complete, entirely valid blocks can participate in attention.
"""
    def __init__(self, block_size: int, head_dim: int, hidden: int):
        super().__init__()
        self.block_size = block_size
        self.head_dim = head_dim
        self.position = nn.Parameter(torch.zeros(block_size, head_dim))
        self.down = nn.Linear(block_size * head_dim, hidden)
        self.up = nn.Linear(hidden, head_dim)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, g, t, d = x.shape
        if d != self.head_dim:
            raise ValueError("compressor head dimension mismatch")
        n = t // self.block_size
        if n == 0:
            return x[:, :, :0]
        blocks = x[:, :, :n * self.block_size].reshape(b, g, n, self.block_size, d)
        return blocks.mean(-2) + self.up(F.silu(self.down((blocks + self.position).flatten(-2))))


def masked_softmax(logits: Tensor, allowed: Tensor) -> Tensor:
    """True masked softmax with an exactly zero, finite all-masked row."""
    if logits.shape[-1] == 0:
        return logits
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    masked = logits.to(dtype).masked_fill(~allowed, -torch.inf)
    safe = torch.where(allowed.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return safe.softmax(-1).masked_fill(~allowed, 0)


def _gather_attention(q: Tensor, k: Tensor, v: Tensor, indices: Tensor,
                      allowed: Tensor, dropout: float) -> Tensor:
    # q [B,G,C,R,D]; indices [B,G,C,S]. KV are gathered ONCE per group,
    # rather than repeated for every query head or masked in a dense T x T map.
    b, g = k.shape[:2]
    bi = torch.arange(b, device=k.device)[:, None, None, None]
    gi = torch.arange(g, device=k.device)[None, :, None, None]
    kk, vv = k[bi, gi, indices], v[bi, gi, indices]
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    logits = torch.einsum("bgcrd,bgcsd->bgcrs", q.to(acc), kk.to(acc)) / math.sqrt(q.shape[-1])
    p = masked_softmax(logits, allowed.unsqueeze(-2))
    p = F.dropout(p, p=dropout, training=dropout > 0)
    return torch.einsum("bgcrs,bgcsd->bgcrd", p, vv.to(acc)).to(q.dtype)


def sparse_attention(q: Tensor, selected_k: Tensor, selected_v: Tensor,
                     window_k: Tensor, window_v: Tensor,
                     compressed_k: Tensor, compressed_v: Tensor,
                     gates: Tensor, valid_tokens: Tensor, config: NSAConfig,
                     *, mode: str = "nsa", dropout: float = 0.0,
                     return_debug: bool = False) -> tuple[Tensor, dict]:
    """All Q/K are already normalized/positioned; V is never rotary-rotated.

q: [B,H,T,D]; raw KV: [B,G,T,D]; compressed KV: [B,G,T//L,D].
Gates: [B,H,T,3] independent sigmoid values, in cmp/selected/window order.
Supports full-prefix training/prefill only. No cache mutation or generation.
"""
    if mode not in {"nsa", "local", "compressed_local"}:
        raise ValueError("unknown active mode")
    if not q.is_floating_point():
        raise ValueError("floating-point Q/K/V are required")
    if q.ndim != 4 or selected_k.ndim != 4:
        raise ValueError("Q and KV must be rank four")
    b, h, t, d = q.shape
    g = selected_k.shape[1]
    if min(b, h, t, d, g) <= 0 or h % g:
        raise ValueError("invalid dimensions or nonintegral GQA grouping")
    if t > config.max_reference_tokens:
        raise ValueError("reference token ceiling exceeded; qualify a larger budget first")
    if (torch.is_grad_enabled() and t > config.max_grad_reference_tokens
            and any(x.requires_grad for x in (q, selected_k, selected_v, window_k, window_v, compressed_k, compressed_v, gates))):
        raise ValueError("reference autograd token ceiling exceeded; chunking does not bound saved activations")
    check_valid(valid_tokens, b, t)
    if valid_tokens.device != q.device:
        raise ValueError("mask and Q must be on the same device")
    n = t // config.block_size
    for x in (selected_k, selected_v, window_k, window_v):
        if x.shape != (b, g, t, d) or x.device != q.device or x.dtype != q.dtype:
            raise ValueError("raw KV shape/device/dtype mismatch")
    for x in (compressed_k, compressed_v):
        if x.shape != (b, g, n, d) or x.device != q.device or x.dtype != q.dtype:
            raise ValueError("compressed KV shape/device/dtype mismatch")
    if gates.shape != (b, h, t, 3) or gates.device != q.device or gates.dtype != q.dtype:
        raise ValueError("gate shape/device mismatch")
    if not torch.isfinite(gates).all() or torch.any((gates < 0) | (gates > 1)):
        raise ValueError("gates must be finite independent probabilities")
    if not 0 <= dropout < 1:
        raise ValueError("invalid dropout")
    c = min(t, config.query_chunk_size)
    # Scratch estimate, NOT a peak-training-memory or allocator guarantee.
    s = min(config.selected_blocks, (t + config.block_size - 1) // config.block_size) * config.block_size
    scratch = 2 * b * g * c * (s + config.window_size) * d * max(4, q.element_size())
    scratch += b * h * c * (s + config.window_size + n) * 8
    if scratch > config.max_scratch_bytes:
        raise ValueError("reference gather scratch ceiling exceeded")
    r = h // g
    qg = q.reshape(b, g, r, t, d).permute(0, 1, 3, 2, 4)
    gg = gates.reshape(b, g, r, t, 3).permute(0, 1, 3, 2, 4)
    block_ids = torch.arange((t + config.block_size - 1) // config.block_size, device=q.device)
    ends = (torch.arange(n, device=q.device) + 1) * config.block_size - 1
    block_valid = valid_tokens[:, :n * config.block_size].reshape(b, n, config.block_size).all(-1)
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    outputs, selections, validity = [], [], []
    for start in range(0, t, c):
        stop = min(start + c, t)
        pos = torch.arange(start, stop, device=q.device)
        qc = qg[:, :, start:stop]
        width = stop - start
        local_idx = pos[:, None] - torch.arange(config.window_size - 1, -1, -1, device=q.device)[None]
        local_allowed = local_idx >= 0
        local_idx = local_idx.clamp(0, t - 1)[None, None].expand(b, g, width, -1)
        local_allowed = local_allowed[None, None] & valid_tokens[:, None, None, :].expand(b, g, width, t).gather(-1, local_idx)
        local_out = _gather_attention(qc, window_k, window_v, local_idx, local_allowed, dropout)
        if mode == "local":
            out = local_out
        else:
            if n:
                logits = torch.einsum("bgcrd,bgnd->bgcrn", qc.to(acc), compressed_k.to(acc)) / math.sqrt(d)
                permitted = (ends[None, :] <= pos[:, None])[None, None, :, None, :] & block_valid[:, None, None, None, :]
                probabilities = masked_softmax(logits, permitted)
                p = F.dropout(probabilities, p=dropout, training=dropout > 0)
                coarse_out = torch.einsum("bgcrn,bgnd->bgcrd", p, compressed_v.to(acc)).to(q.dtype)
                importance = probabilities.detach().sum(-2)  # same selection for R query heads
            else:
                coarse_out = torch.zeros_like(qc)
                importance = q.new_zeros((b, g, width, 0), dtype=acc)
            out = gg[:, :, start:stop, :, 0, None] * coarse_out
            out = out + gg[:, :, start:stop, :, 2, None] * local_out
            if mode == "nsa":
                scores = q.new_full((b, g, width, len(block_ids)), -torch.inf, dtype=acc)
                if n:
                    eligible = (ends[None] <= pos[:, None])[None, None] & block_valid[:, None, None]
                    scores[..., :n] = importance.masked_fill(~eligible, -torch.inf)
                if config.selection_policy == "hash_control":
                    # Stateless, content-blind ranks: independent of future sequence length.
                    ids = block_ids[None, None, None]
                    groups = torch.arange(g, device=q.device)[None, :, None, None]
                    ranks = ((ids + 1) * 0x45d9f3b ^ (pos[None, None, :, None] + 1) * 0x27d4eb2d
                             ^ (groups + 1) * 0x165667b1 ^ config.control_seed) & 2147483647
                    ranks = ((ranks ^ (ranks >> 16)) * 0x45d9f3b) & 2147483647
                    scores = torch.where(torch.isfinite(scores), ranks.to(acc), scores)
                current = pos // config.block_size
                mandatory = (block_ids[None] < config.force_initial_blocks) | (
                    (block_ids[None] <= current[:, None]) &
                    (block_ids[None] > current[:, None] - config.force_recent_blocks))
                visible = block_ids[None] * config.block_size <= pos[:, None]
                mandatory = mandatory & visible
                scores = scores.masked_fill(mandatory[None, None], torch.inf)
                count = min(config.selected_blocks, len(block_ids))
                # Stable ties: older blocks first. Sorting is an inspectable reference cost.
                chosen = scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
                chosen_valid = scores.gather(-1, chosen) > -torch.inf
                token_idx = chosen[..., None] * config.block_size + torch.arange(config.block_size, device=q.device)
                token_ok = chosen_valid[..., None] & (token_idx <= pos[None, None, :, None, None]) & (token_idx < t)
                token_idx = token_idx.flatten(-2).clamp(0, t - 1)
                token_ok = token_ok.flatten(-2) & valid_tokens[:, None, None, :].expand(b, g, width, t).gather(-1, token_idx)
                exact_out = _gather_attention(qc, selected_k, selected_v, token_idx, token_ok, dropout)
                out = out + gg[:, :, start:stop, :, 1, None] * exact_out
                if return_debug:
                    selections.append(chosen.detach())
                    validity.append(chosen_valid.detach())
        out = out * valid_tokens[:, None, start:stop, None, None]
        outputs.append(out)
    merged = torch.cat(outputs, 2).permute(0, 1, 3, 2, 4).reshape(b, h, t, d)
    debug = {"scratch_estimate_bytes": scratch, "fused_kernel": False}
    if return_debug and selections:
        debug.update(selected_blocks=torch.cat(selections, 2), selected_valid=torch.cat(validity, 2))
    return merged, debug
