"""Differentiable reference equations. No fused-kernel or throughput claims.

Layout: state [batch, heads, slices, key_dim, value_dim]. A SINGLE joint
residual is used across slices; this is not a bank of independent DeltaNets.
"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class MemoryConfig:
    d_model: int = 1024
    heads: int = 4
    key_dim: int = 64
    value_dim: int = 64
    slices: int = 4
    conv_width: int = 4
    gate_init: float = 0.0
    max_state_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in ("d_model", "heads", "key_dim", "value_dim", "slices", "conv_width", "max_state_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.gate_init):
            raise ValueError("gate_init must be finite")

    @property
    def channels(self) -> int:
        return self.heads * (2 * self.key_dim + self.value_dim + (2 * self.slices if self.slices > 1 else 0))

    def state_bytes(self, batch: int = 1, itemsize: int = 4) -> int:
        if batch <= 0 or itemsize <= 0:
            raise ValueError("batch and itemsize must be positive")
        recurrent = batch * self.heads * self.slices * self.key_dim * self.value_dim
        history = batch * self.channels * (self.conv_width - 1)
        return (recurrent + history) * itemsize


@dataclass
class MemoryState:
    tensor: Tensor
    history: Tensor

    def detach(self):
        return MemoryState(self.tensor.detach(), self.history.detach())

    def clone(self):
        return MemoryState(self.tensor.clone(), self.history.clone())

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.tensor, self.history))


def normalize(x: Tensor) -> Tensor:
    """Preserve FP64 tests; native reference recurrence is explicitly FP32."""
    return x / x.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()


def _check(state, k, k2, v, alpha, beta):
    if state.ndim != 5:
        raise ValueError("state must be [B,H,E,K,V]")
    b, h, e, d, dv = state.shape
    expected = [(b, h, d), (b, h, e), (b, h, dv), (b, h, e), (b, h)]
    for name, x, shape in zip(("key", "second key", "value", "decay", "write"),
                              (k, k2, v, alpha, beta), expected):
        if tuple(x.shape) != shape or x.dtype != state.dtype or x.device != state.device:
            raise ValueError(f"invalid {name}: require {shape}, matching dtype/device")


def read_state(state: Tensor, q: Tensor, q2: Tensor) -> Tensor:
    b, h, e, d, _ = state.shape
    if q.shape != (b, h, d) or q2.shape != (b, h, e):
        raise ValueError("query shape mismatch")
    return torch.einsum("bhedv,bhd,bhe->bhv", state, q, q2)


def delta_step(state: Tensor, k: Tensor, k2: Tensor, v: Tensor,
               alpha: Tensor, beta: Tensor) -> Tensor:
    """Out-of-place update; caller supplies normalized keys and bounded gates."""
    _check(state, k, k2, v, alpha, beta)
    decayed = state * alpha[..., None, None]
    residual = v - read_state(decayed, k, k2)
    correction = torch.einsum("bhe,bhd,bhv->bhedv", k2, k, residual)
    return decayed + beta[..., None, None, None] * correction


def flattened_delta_step(state: Tensor, k: Tensor, k2: Tensor, v: Tensor,
                         alpha: Tensor, beta: Tensor) -> Tensor:
    """Independent matrix formulation, returned in tensor layout for auditing."""
    _check(state, k, k2, v, alpha, beta)
    b, h, e, d, dv = state.shape
    flat = (state * alpha[..., None, None]).reshape(b, h, e * d, dv)
    joint = (k2.unsqueeze(-1) * k.unsqueeze(-2)).reshape(b, h, e * d)
    residual = v - torch.matmul(joint.unsqueeze(-2), flat).squeeze(-2)
    flat = flat + beta[..., None, None] * joint.unsqueeze(-1) * residual.unsqueeze(-2)
    return flat.reshape_as(state)


def shuffle_second_keys(k2: Tensor, reset: Tensor, seed: int = 0) -> Tensor:
    """OFFLINE evidence-replay ablation only, not a causal streaming operation.

    Shuffle token addresses within each document, never across documents/batches.
    The final answer/query positions must NOT be included in this evidence tensor.
    """
    if k2.ndim != 4 or reset.shape != k2.shape[:2] or reset.dtype != torch.bool:
        raise ValueError("expected k2 [B,T,H,E] and boolean document starts [B,T]")
    result = k2.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for b in range(k2.shape[0]):
        starts = sorted(set([0] + reset[b].nonzero().flatten().tolist()))
        ends = starts[1:] + [k2.shape[1]]
        for start, end in zip(starts, ends):
            order = torch.randperm(end - start, generator=generator).to(k2.device) + start
            result[b, start:end] = k2[b, order]
    return result
