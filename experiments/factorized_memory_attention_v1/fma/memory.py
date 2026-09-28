"""Factorized token memory and gated value-path substitution.

Paper MA: M = Norm(E[s]); V = K + M, with K taken before RoPE.  This lab replaces
per-layer E with shared U[N,r] and per-layer P_l[r,d_v], then applies head-wise
RMSNorm.  The transition gate is a buffer, not an added trainable parameter.
"""
from __future__ import annotations

import hashlib
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import FMAConfig, ModelShape


class HeadwiseRMSNorm(nn.Module):
    """RMS-normalize each KV head independently, sharing one learned d_h scale.

    For x[..., H_KV*d_h], reshape to x[..., H_KV, d_h], compute the RMS over
    d_h independently for every token and head, then multiply by a learned
    d_h-vector broadcast over heads.  This follows the paper's explicit
    normalization axis while keeping the Rineya variant compact.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float, affine: bool = True) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.eps = float(eps)
        if affine:
            self.weight = nn.Parameter(torch.ones(head_dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.num_heads * self.head_dim:
            raise ValueError(
                f"expected trailing width {self.num_heads * self.head_dim}, got {value.shape[-1]}"
            )
        shape = value.shape
        x = value.reshape(*shape[:-1], self.num_heads, self.head_dim)
        # Accumulate norm in float32 for BF16/FP16 stability, then return input dtype.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        variance = work.square().mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        if self.weight is not None:
            y = y * self.weight.to(dtype=x.dtype)
        return y.reshape(shape)


class FactorizedMemoryBank(nn.Module):
    """Shared U plus per-layer P_l and head-wise RMSNorm."""

    def __init__(self, shape: ModelShape, cfg: FMAConfig) -> None:
        super().__init__()
        self.shape = shape
        self.config = cfg
        self.layers = cfg.resolved_layers(shape)
        self.layer_to_slot = {layer: slot for slot, layer in enumerate(self.layers)}
        self.token_factors = nn.Embedding(shape.vocab_size, cfg.rank)
        self.projections = nn.ModuleList(
            nn.Linear(cfg.rank, shape.value_width, bias=False) for _ in self.layers
        )
        self.norms = nn.ModuleList(
            HeadwiseRMSNorm(
                shape.num_key_value_heads,
                shape.head_dim,
                shape.rms_norm_eps,
                affine=cfg.affine_head_rmsnorm,
            )
            for _ in self.layers
        )
        self.register_buffer(
            "gates",
            torch.full((len(self.layers),), float(cfg.gate), dtype=torch.float32),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_factors.weight, mean=0.0, std=self.config.init_std)
        for projection in self.projections:
            nn.init.normal_(projection.weight, mean=0.0, std=self.config.init_std)
        for norm in self.norms:
            if norm.weight is not None:
                nn.init.ones_(norm.weight)

    def slot(self, layer_idx: int) -> int:
        try:
            return self.layer_to_slot[int(layer_idx)]
        except KeyError as exc:
            raise KeyError(f"layer {layer_idx} is not FMA-enabled") from exc

    def set_gate(self, layer_idx: int, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("gate must be in [0,1]")
        self.gates[self.slot(layer_idx)] = float(value)

    def set_all_gates(self, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("gate must be in [0,1]")
        self.gates.fill_(float(value))

    def raw_memory(self, token_ids: Tensor, layer_idx: int) -> Tensor:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must contain integer vocabulary IDs")
        if token_ids.numel() and (int(token_ids.min()) < 0 or int(token_ids.max()) >= self.shape.vocab_size):
            raise IndexError("token ID outside configured vocabulary")
        slot = self.slot(layer_idx)
        return self.projections[slot](self.token_factors(token_ids))

    def memory(self, token_ids: Tensor, layer_idx: int) -> Tensor:
        slot = self.slot(layer_idx)
        return self.norms[slot](self.raw_memory(token_ids, layer_idx))

    def memory_path(self, token_ids: Tensor, layer_idx: int, key_pre_rope: Tensor) -> Tensor:
        memory = self.memory(token_ids, layer_idx)
        if memory.shape != key_pre_rope.shape:
            raise ValueError(f"memory shape {tuple(memory.shape)} != key shape {tuple(key_pre_rope.shape)}")
        return key_pre_rope + memory

    def value(
        self,
        *,
        hidden_states: Tensor,
        token_ids: Tensor,
        layer_idx: int,
        key_pre_rope: Tensor,
        standard_value_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Apply the transition gate with hard structural short-circuits.

        g=0 never evaluates memory. g=1 never calls standard_value_fn.  The latter
        makes W_V bypass testable rather than a claim inferred from arithmetic.
        """
        gate = float(self.gates[self.slot(layer_idx)].item())
        if gate == 0.0:
            return standard_value_fn(hidden_states)
        if gate == 1.0:
            return self.memory_path(token_ids, layer_idx, key_pre_rope)
        standard = standard_value_fn(hidden_states)
        memory = self.memory_path(token_ids, layer_idx, key_pre_rope)
        return (1.0 - gate) * standard + gate * memory

    @torch.no_grad()
    def folded_rows(self, token_ids: Tensor, layer_idx: int) -> Tensor:
        """Fold normalization into only the requested rows; never materialize N*d_v by default."""
        return self.memory(token_ids, layer_idx)

    @torch.no_grad()
    def materialize_dense_layer(self, layer_idx: int, *, max_elements: int = 2_000_000) -> Tensor:
        """Materialize U@P only for tiny/toy verification.

        The default guard deliberately rejects the real Rineya N*d_v matrix.
        """
        elements = self.shape.vocab_size * self.shape.value_width
        if elements > max_elements:
            raise MemoryError(
                f"refusing to materialize {elements:,} elements; use indexed lookup or raise max_elements explicitly"
            )
        slot = self.slot(layer_idx)
        return self.token_factors.weight @ self.projections[slot].weight.T

    def added_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def teacher_residual(standard_value: Tensor, key_pre_rope: Tensor) -> Tensor:
    """Teacher target R_l = XW_V,l - K_l, with Rineya K taken pre-RoPE."""
    if standard_value.shape != key_pre_rope.shape:
        raise ValueError("teacher residual requires matching value/key shapes")
    return standard_value - key_pre_rope


def calibration_loss(
    bank: FactorizedMemoryBank,
    token_ids: Tensor,
    layer_idx: int,
    standard_value: Tensor,
    key_pre_rope: Tensor,
) -> Tensor:
    """Fit the *normalized* factorized memory to the teacher residual."""
    target = teacher_residual(standard_value.detach(), key_pre_rope.detach())
    predicted = bank.memory(token_ids, layer_idx)
    return F.mse_loss(predicted, target)


def freeze_backbone_for_calibration(backbone: nn.Module, bank: FactorizedMemoryBank) -> dict[str, int]:
    """Freeze every backbone parameter and leave only FMA memory parameters trainable."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in bank.parameters():
        p.requires_grad_(True)
    # Gate is a buffer and therefore cannot accidentally train.
    return {
        "backbone_trainable": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
        "memory_trainable": sum(p.numel() for p in bank.parameters() if p.requires_grad),
    }


def module_digest(module: nn.Module) -> str:
    """Stable SHA-256 over tensor names/shapes/dtypes/bytes for mutation checks."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
