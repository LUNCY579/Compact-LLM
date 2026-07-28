from __future__ import annotations

"""Reference block-sparse wrappers for Qwen/Llama-style gated MLPs.

`OracleMaskedGatedMLP` is a scientific simulation. It still computes the full
up/gate projections and therefore MUST NOT be used for speed claims.

`RoutedSparseGatedMLP` gathers selected projection rows/columns for single-token
decode. It performs less matrix arithmetic, but ordinary PyTorch gather overhead
may make it slower than the dense kernel. It exists as a correctness reference
before Triton/custom kernels.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SparseStats:
    calls: int = 0
    sparse_calls: int = 0
    dense_fallback_calls: int = 0
    selected_blocks: int = 0
    total_blocks: int = 0

    def as_dict(self) -> dict[str, float | int]:
        ratio = self.selected_blocks / self.total_blocks if self.total_blocks else 1.0
        fallback = self.dense_fallback_calls / self.calls if self.calls else 0.0
        return {
            "calls": self.calls,
            "sparse_calls": self.sparse_calls,
            "dense_fallback_calls": self.dense_fallback_calls,
            "mean_active_block_ratio": ratio,
            "dense_fallback_rate": fallback,
        }


def block_scores_from_intermediate(
    intermediate: torch.Tensor,
    down_weight: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute |z_j| * ||W_down[:,j]|| and sum contiguous channels into blocks."""

    if intermediate.shape[-1] != down_weight.shape[1]:
        raise ValueError("Intermediate width does not match down projection input width.")
    column_norm = down_weight.detach().float().norm(dim=0).to(intermediate.device)
    score = intermediate.detach().float().abs() * column_norm
    pad = (-score.shape[-1]) % block_size
    if pad:
        score = F.pad(score, (0, pad))
    return score.reshape(*score.shape[:-1], -1, block_size).sum(dim=-1)


def top_block_mask(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1].")
    blocks = scores.shape[-1]
    keep = max(1, min(blocks, int(round(blocks * keep_ratio))))
    indices = torch.topk(scores, k=keep, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(-1, indices, True)


def expand_block_mask(mask: torch.Tensor, block_size: int, width: int) -> torch.Tensor:
    return mask.repeat_interleave(block_size, dim=-1)[..., :width]


class TinyBlockRouter(nn.Module):
    def __init__(self, hidden_size: int, router_width: int, num_blocks: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(hidden_size, router_width, bias=False)
        self.proj_out = nn.Linear(router_width, num_blocks, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj_out(F.silu(self.proj_in(hidden)))


class OracleMaskedGatedMLP(nn.Module):
    """Dense-compute oracle masking used only to measure quality tolerance."""

    def __init__(self, dense_mlp: nn.Module, block_size: int, keep_ratio: float) -> None:
        super().__init__()
        self.dense_mlp = dense_mlp
        self.block_size = int(block_size)
        self.keep_ratio = float(keep_ratio)
        self.stats = SparseStats()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = self.dense_mlp.act_fn(self.dense_mlp.gate_proj(hidden))
        up = self.dense_mlp.up_proj(hidden)
        intermediate = gate * up
        scores = block_scores_from_intermediate(
            intermediate,
            self.dense_mlp.down_proj.weight,
            self.block_size,
        )
        block_mask = top_block_mask(scores, self.keep_ratio)
        channel_mask = expand_block_mask(block_mask, self.block_size, intermediate.shape[-1])
        self.stats.calls += 1
        self.stats.sparse_calls += 1
        self.stats.selected_blocks += int(block_mask.sum().item())
        self.stats.total_blocks += int(block_mask.numel())
        return self.dense_mlp.down_proj(intermediate * channel_mask.to(intermediate.dtype))


class RoutedSparseGatedMLP(nn.Module):
    """Single-token PyTorch reference that gathers selected channel blocks.

    Prefill and multi-token batches use the original dense MLP by default. This
    avoids pretending that a token-wise Python gather implementation is an
    efficient prefill kernel.
    """

    def __init__(
        self,
        dense_mlp: nn.Module,
        router: TinyBlockRouter,
        *,
        block_size: int,
        keep_ratio: float,
        confidence_threshold: float,
        dense_prefill: bool = True,
    ) -> None:
        super().__init__()
        self.dense_mlp = dense_mlp
        self.router = router
        self.block_size = int(block_size)
        self.keep_ratio = float(keep_ratio)
        self.confidence_threshold = float(confidence_threshold)
        self.dense_prefill = bool(dense_prefill)
        self.stats = SparseStats()

    @staticmethod
    def confidence(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-6))
            + (1 - probabilities) * torch.log((1 - probabilities).clamp_min(1e-6))
        )
        return 1.0 - entropy.mean(dim=-1) / 0.6931471805599453

    def _dense(self, hidden: torch.Tensor) -> torch.Tensor:
        self.stats.dense_fallback_calls += 1
        return self.dense_mlp(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.stats.calls += 1
        # The reference sparse path is intentionally limited to batch=1,
        # sequence=1 decode, where one mask can be applied without unions.
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != 1:
            return self._dense(hidden)
        if self.dense_prefill and hidden.shape[1] > 1:
            return self._dense(hidden)

        vector = hidden[:, -1, :]
        logits = self.router(vector)
        confidence = float(self.confidence(logits).item())
        if confidence < self.confidence_threshold:
            return self._dense(hidden)

        num_blocks = logits.shape[-1]
        keep = max(1, min(num_blocks, int(round(num_blocks * self.keep_ratio))))
        selected_blocks = torch.topk(logits, k=keep, dim=-1).indices[0]
        channels = []
        width = self.dense_mlp.gate_proj.out_features
        for block in selected_blocks.tolist():
            start = block * self.block_size
            channels.extend(range(start, min(start + self.block_size, width)))
        channel_index = torch.tensor(channels, device=hidden.device, dtype=torch.long)

        gate_proj = self.dense_mlp.gate_proj
        up_proj = self.dense_mlp.up_proj
        down_proj = self.dense_mlp.down_proj
        gate_bias = gate_proj.bias.index_select(0, channel_index) if gate_proj.bias is not None else None
        up_bias = up_proj.bias.index_select(0, channel_index) if up_proj.bias is not None else None
        gate = F.linear(hidden, gate_proj.weight.index_select(0, channel_index), gate_bias)
        up = F.linear(hidden, up_proj.weight.index_select(0, channel_index), up_bias)
        intermediate = self.dense_mlp.act_fn(gate) * up
        output = F.linear(
            intermediate,
            down_proj.weight.index_select(1, channel_index),
            down_proj.bias,
        )
        self.stats.sparse_calls += 1
        self.stats.selected_blocks += keep
        self.stats.total_blocks += num_blocks
        return output
