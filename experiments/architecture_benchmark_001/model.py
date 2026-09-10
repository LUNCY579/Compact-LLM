"""Small, readable dense causal encoder-decoder reference, extended from the supplied CED40 bundle.

Implements GQA, Q/K per-head RMSNorm, half-split RoPE, SwiGLU, tied embeddings,
explicit causal/padding/segment masks, and request-local inference caches.
The CED decoder REPLACES self-attention with reads of one encoder-derived KV bank.
It intentionally has no decoder local SWA, MoE, CSA, MLA, vision, or custom kernels.

The reference is for mechanical reasoning/tests. It is not HF generate-compatible,
optimized for serving, or qualified against pretrained Qwen/Rineya weights.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from experiments.architecture_benchmark_001.budget import Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
        return z.to(x.dtype) * self.weight


def rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """x: [batch, heads, length, head_dim]; absolute positions: [batch, length]."""
    h = x.shape[-1]
    inv = theta ** (-torch.arange(0, h, 2, device=x.device, dtype=torch.float32) / h)
    angles = positions.float()[..., None] * inv
    angles = torch.cat((angles, angles), dim=-1)[:, None]
    a, b = x.chunk(2, dim=-1)
    rotated = torch.cat((-b, a), dim=-1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


def dense_attention(q: Tensor, k: Tensor, v: Tensor, qp: Tensor, kp: Tensor,
                    qvalid: Tensor, kvalid: Tensor, qseg: Tensor, kseg: Tensor) -> Tensor:
    """Slow explicit attention with safe all-masked rows; no sparse speed claims."""
    group = q.shape[1] // k.shape[1]
    # Expand only temporarily for computation; persistent KV retains KV-head count.
    kk, vv = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    allowed = ((kp[:, None, :] <= qp[:, :, None])
               & kvalid[:, None, :] & qvalid[:, :, None]
               & (qseg[:, :, None] == kseg[:, None, :]))
    # Absolute/RoPE positions can repeat; physical sequence order still must be causal.
    # Current-chunk queries (or its final position) occupy the suffix of the key bank.
    qslots = torch.arange(k.shape[2] - q.shape[2], k.shape[2], device=q.device)
    kslots = torch.arange(k.shape[2], device=q.device)
    allowed = allowed & (kslots[None, None, :] <= qslots[None, :, None])
    scores = (q.float() @ kk.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
    prob = scores.softmax(-1) * allowed[:, None]
    prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-20)
    return prob.to(vv.dtype) @ vv


class SwiGLU(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        d, m = c.hidden_size, c.intermediate_size
        self.gate_proj, self.up_proj = nn.Linear(d, m, False), nn.Linear(d, m, False)
        self.down_proj = nn.Linear(m, d, False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


KV = tuple[Tensor, Tensor]


class SelfBlock(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        d, h = c.hidden_size, c.head_dim
        self.input_norm, self.post_norm = RMSNorm(d, c.rms_norm_eps), RMSNorm(d, c.rms_norm_eps)
        self.q_norm, self.k_norm = RMSNorm(h, c.rms_norm_eps), RMSNorm(h, c.rms_norm_eps)
        self.q_proj = nn.Linear(d, c.num_attention_heads * h, False)
        self.k_proj = nn.Linear(d, c.num_key_value_heads * h, False)
        self.v_proj = nn.Linear(d, c.num_key_value_heads * h, False)
        self.o_proj = nn.Linear(c.num_attention_heads * h, d, False)
        self.mlp = SwiGLU(c)

    def forward(self, x: Tensor, qp: Tensor, kp: Tensor, qv: Tensor, kv: Tensor,
                qs: Tensor, ks: Tensor, past: Optional[KV]) -> tuple[Tensor, KV]:
        b, t, _ = x.shape
        c = self.c
        n = self.input_norm(x)
        shape = lambda z, heads: z.view(b, t, heads, c.head_dim).transpose(1, 2)
        q = rope(self.q_norm(shape(self.q_proj(n), c.num_attention_heads)), qp, c.rope_theta)
        k = rope(self.k_norm(shape(self.k_proj(n), c.num_key_value_heads)), qp, c.rope_theta)
        v = shape(self.v_proj(n), c.num_key_value_heads)
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        a = dense_attention(q, k, v, qp, kp, qv, kv, qs, ks)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, t, -1))
        return x + self.mlp(self.post_norm(x)), (k, v)


class SharedMemory(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, False)

    def forward(self, e: Tensor, pos: Tensor, past: Optional[KV]) -> KV:
        b, t, _ = e.shape
        c = self.c
        n = self.norm(e)
        shape = lambda z: z.view(b, t, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        k = rope(self.k_norm(shape(self.k_proj(n))), pos, c.rope_theta)
        v = shape(self.v_proj(n))
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        return k, v


class ReadBlock(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.input_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, False)
        self.mlp = SwiGLU(c)

    def forward(self, x: Tensor, memory: KV, qp: Tensor, kp: Tensor,
                qv: Tensor, kv: Tensor, qs: Tensor, ks: Tensor) -> Tensor:
        b, t, _ = x.shape
        c = self.c
        q = self.q_proj(self.input_norm(x)).view(b, t, c.num_attention_heads, c.head_dim).transpose(1, 2)
        q = rope(self.q_norm(q), qp, c.rope_theta)
        a = dense_attention(q, *memory, qp, kp, qv, kv, qs, ks)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, t, -1))
        return x + self.mlp(self.post_norm(x))


@dataclass(frozen=True)
class Cache:
    owner: int
    signature: tuple
    encoder: tuple[KV, ...]
    memory: Optional[KV]
    positions: Tensor
    valid: Tensor
    segments: Tensor


@dataclass
class Output:
    logits: Tensor
    cache: Optional[Cache]


class DenseReference(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.encoder = nn.ModuleList(SelfBlock(c) for _ in range(c.encoder_layers))
        self.memory = SharedMemory(c) if c.architecture == "ced" else None
        self.decoder = nn.ModuleList(ReadBlock(c) for _ in range(c.decoder_layers))
        self.final_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _signature(self) -> tuple:
        # Detect stale caches after load_state_dict, optimizer steps, or dtype/device moves.
        # This is a readable correctness reference, not a serving-optimized cache API.
        return tuple((p.data_ptr(), p._version, str(p.dtype), str(p.device))
                     for p in self.parameters())

    def _validate_cache(self, past: Cache, batch: int, device: torch.device) -> None:
        if not isinstance(past, Cache) or past.owner != id(self) or past.signature != self._signature():
            raise ValueError("cache belongs to another model or stale parameter revision")
        if past.positions.ndim != 2 or past.positions.shape[0] != batch or past.positions.shape[1] == 0:
            raise ValueError("invalid cache metadata shape")
        for value, dtype in ((past.positions, torch.long), (past.valid, torch.bool),
                             (past.segments, torch.long)):
            if value.shape != past.positions.shape or value.dtype != dtype or value.device != device:
                raise ValueError("cache metadata shape/dtype/device mismatch")
        if (past.positions < 0).any() or (past.positions >= self.config.max_position_embeddings).any():
            raise ValueError("cache positions outside supported context")
        if len(past.encoder) != len(self.encoder) or (past.memory is None) != (self.memory is None):
            raise ValueError("cache architecture/bank mismatch")
        pairs = list(past.encoder) + ([] if past.memory is None else [past.memory])
        expected = (batch, self.config.num_key_value_heads, past.positions.shape[1], self.config.head_dim)
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("cache requires K/V pairs")
            for value in pair:
                if (value.shape != expected or value.dtype != self.embed_tokens.weight.dtype
                        or value.device != device):
                    raise ValueError("cache tensor shape/dtype/device mismatch")

    def forward(self, input_ids: Tensor, *, attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None,
                past: Optional[Cache] = None, use_cache: bool = False,
                last_position_only: bool = False) -> Output:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be nonempty [batch,length] int64")
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError("cache path is inference-only: use eval() and inference_mode()")
        if past is not None and not use_cache:
            raise ValueError("past requires use_cache=True")
        if last_position_only and self.training:
            raise ValueError("tail-only path is not a training objective")
        b, t = input_ids.shape
        device = input_ids.device
        if b == 0 or device != self.embed_tokens.weight.device:
            raise ValueError("input batch/device mismatch")
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError("token IDs outside vocabulary")
        if attention_mask is not None and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must contain only zero and one")
        valid = torch.ones((b, t), dtype=torch.bool, device=device) if attention_mask is None else attention_mask.bool()
        seg = torch.zeros((b, t), dtype=torch.long, device=device) if segment_ids is None else segment_ids
        for label, value in (("attention_mask", valid), ("segment_ids", seg)):
            if value.shape != input_ids.shape or value.device != device:
                raise ValueError(f"{label} shape/device mismatch")
        if seg.dtype != torch.long:
            raise ValueError("segment_ids must be int64; use unique IDs per packed document")
        if past is not None:
            self._validate_cache(past, b, device)
        if position_ids is None:
            start = 0
            if past is not None:
                slots = torch.arange(past.valid.shape[1], device=device).expand_as(past.valid)
                last = slots.masked_fill(~past.valid, -1).max(-1).values
                start = past.positions.gather(1, last.clamp_min(0)[:, None]) + 1
                start = torch.where(last[:, None] >= 0, start, 0)
            pos = (valid.long().cumsum(-1) - 1).clamp_min(0) + start
        else:
            pos = position_ids
        if pos.shape != input_ids.shape or pos.dtype != torch.long or pos.device != device:
            raise ValueError("position_ids must be matching [batch,length] int64")
        if (pos < 0).any() or (pos >= self.config.max_position_embeddings).any():
            raise ValueError("positions outside supported context")
        if t + (0 if past is None else past.positions.shape[1]) > self.config.max_position_embeddings:
            raise ValueError("physical cache/context limit exceeded; eviction is not implemented")
        if last_position_only and not bool(valid[:, -1].all()):
            raise ValueError("tail-only requires a valid last physical token; left-pad batches")
        kp = pos if past is None else torch.cat((past.positions, pos), -1)
        kv = valid if past is None else torch.cat((past.valid, valid), -1)
        ks = seg if past is None else torch.cat((past.segments, seg), -1)
        h = self.embed_tokens(input_ids)
        layer_kv = []
        for i, block in enumerate(self.encoder):
            h, pair = block(h, pos, kp, valid, kv, seg, ks, None if past is None else past.encoder[i])
            layer_kv.append(pair)
        memory = None
        if self.memory is not None:
            memory = self.memory(h, pos, None if past is None else past.memory)
            qp, qv, qs = pos, valid, seg
            if last_position_only:
                h, qp, qv, qs = h[:, -1:], pos[:, -1:], valid[:, -1:], seg[:, -1:]
            for block in self.decoder:
                h = block(h, memory, qp, kp, qv, kv, qs, ks)
        elif last_position_only:
            # For an ordinary model this saves only final-logit computation.
            h = h[:, -1:]
        # Direct use of the input embedding guarantees one tied Parameter object.
        logits = F.linear(self.final_norm(h), self.embed_tokens.weight)
        cache = None
        if use_cache:
            cache = Cache(id(self), self._signature(), tuple(layer_kv), memory, kp.clone(), kv.clone(), ks.clone())
        return Output(logits, cache)
