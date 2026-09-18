"""Causal, dense-residual RoLA port. No token sparsification or pretrained loader.

Inputs are normalized *pre-RoPE* Q/K and unrotated V, [batch, heads, time, dim].
The supplied schedule must be the backbone's actual inverse frequencies. Qwen's
split-half rotary convention is explicit; other layouts need a qualified adapter.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PrefixState:
    memory: Tensor
    last_position: Tensor
    owner: int
    versions: tuple[int, ...]
    sequence_id: str


class CausalRoLA(nn.Module):
    def __init__(self, heads: int, head_dim: int, rank: int, inv_freq: Tensor):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (heads, head_dim, rank)):
            raise ValueError('positive integer dimensions required')
        if head_dim % 2 or rank % 2 or rank > head_dim:
            raise ValueError('rank must contain complete rotary pairs')
        if inv_freq.ndim != 1 or inv_freq.numel() != head_dim // 2:
            raise ValueError('supply the full native inverse-frequency schedule')
        if not bool(torch.isfinite(inv_freq).all()) or not bool((inv_freq > 0).all()):
            raise ValueError('invalid frequency schedule')
        self.heads, self.head_dim, self.rank = heads, head_dim, rank
        self.register_buffer('inv_freq', inv_freq[:rank // 2].detach().clone())
        self.pq = nn.Parameter(torch.empty(heads, rank, head_dim))
        self.pk = nn.Parameter(torch.empty(heads, rank, head_dim))
        for weight in (self.pq, self.pk):
            nn.init.uniform_(weight, -1 / math.sqrt(head_dim), 1 / math.sqrt(head_dim))
        # Exact no-op is a deliberate language-model-port deviation from the paper.
        # Alpha learns first; the projections receive gradients once it is nonzero.
        self.alpha = nn.Parameter(torch.zeros(()))

    def _versions(self) -> tuple[int, ...]:
        return tuple(x._version for x in (self.pq, self.pk, self.alpha, self.inv_freq))

    def _rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        angle = positions.to(x.dtype)[:, None, :, None] * self.inv_freq.to(x.dtype)
        a, b = x.chunk(2, dim=-1)
        return torch.cat((a * angle.cos() - b * angle.sin(),
                          b * angle.cos() + a * angle.sin()), dim=-1)

    def features(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return (self._rotate(F.silu(torch.einsum('bhtd,hrd->bhtr', q, self.pq)), positions),
                self._rotate(F.silu(torch.einsum('bhtd,hrd->bhtr', k, self.pk)), positions))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, dense: Tensor,
                positions: Tensor, valid: Tensor, *, sequence_id: str,
                state: PrefixState | None = None) -> tuple[Tensor, PrefixState]:
        if not sequence_id or not isinstance(sequence_id, str):
            raise ValueError('explicit sequence identity required')
        if q.ndim != 4 or q.shape[1] != self.heads or q.shape[-1] != self.head_dim:
            raise ValueError('Q must be [batch, query_heads, time, head_dim]')
        b, h, t, d = q.shape
        if min(b, t) <= 0 or k.shape != v.shape or k.ndim != 4:
            raise ValueError('invalid KV shape')
        if k.shape[0] != b or k.shape[2:] != (t, d) or k.shape[1] < 1 or h % k.shape[1]:
            raise ValueError('invalid grouped-query attention layout')
        if dense.shape != q.shape or positions.shape != (b, t) or valid.shape != (b, t):
            raise ValueError('shape mismatch')
        if positions.dtype != torch.int64 or valid.dtype != torch.bool:
            raise ValueError('positions=int64 and valid=bool required')
        tensors = (q, k, v, dense)
        if q.dtype not in (torch.float32, torch.float64) or q.dtype != self.pq.dtype:
            raise ValueError('mechanics supports explicit FP32/FP64 only')
        if any(x.dtype != q.dtype or x.device != q.device for x in tensors):
            raise ValueError('inconsistent floating tensors')
        if positions.device != q.device or valid.device != q.device or self.pq.device != q.device:
            raise ValueError('inconsistent devices')
        if not all(bool(torch.isfinite(x).all()) for x in (*tensors, *self.parameters(), self.inv_freq)):
            raise ValueError('nonfinite input or parameter')
        if bool((positions[valid] < 0).any()):
            raise ValueError('negative valid position')
        versions = self._versions()
        if state is None:
            memory = q.new_zeros(b, h, self.rank, d)
            last = positions.new_full((b,), -1)
        else:
            if state.owner != id(self) or state.versions != versions or state.sequence_id != sequence_id:
                raise ValueError('foreign sequence, module, or stale parameter cache')
            if state.memory.shape != (b, h, self.rank, d) or state.last_position.shape != (b,):
                raise ValueError('invalid cache shape')
            if state.memory.dtype != q.dtype or state.memory.device != q.device:
                raise ValueError('invalid cache dtype/device')
            if state.last_position.dtype != torch.int64 or state.last_position.device != q.device:
                raise ValueError('invalid position cache')
            if not bool(torch.isfinite(state.memory).all()):
                raise ValueError('nonfinite cache')
            memory, last = state.memory, state.last_position
        k = k.repeat_interleave(h // k.shape[1], dim=1)
        v = v.repeat_interleave(h // v.shape[1], dim=1)
        qf, kf = self.features(q, k, positions)
        reads = []
        for i in range(t):
            active = valid[:, i]
            if bool((active & (positions[:, i] <= last)).any()):
                raise ValueError('positions must strictly increase within a sequence')
            write = torch.einsum('bhr,bhd->bhrd', kf[:, :, i], v[:, :, i])
            memory = memory + write * active[:, None, None, None]
            read = torch.einsum('bhr,bhrd->bhd', qf[:, :, i], memory)
            read = read / torch.sqrt(read.square().mean(-1, keepdim=True) + 1e-6)
            reads.append(read * active[:, None, None])
            last = torch.where(active, positions[:, i], last)
        correction = torch.stack(reads, dim=2)
        scale = torch.sqrt(dense.square().mean(-1, keepdim=True) + 1e-6)
        output = (dense + self.alpha * scale * correction) * valid[:, None, :, None]
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(memory).all()):
            raise FloatingPointError('nonfinite readout/state')
        return output, PrefixState(memory, last, id(self), versions, sequence_id)
