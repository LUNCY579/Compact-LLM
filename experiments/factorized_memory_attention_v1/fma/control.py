"""Parameter-matched conventional added-capacity control (M2)."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .budget import parameter_budget
from .config import FMAConfig, ModelShape


class LowRankValueAdapter(nn.Module):
    def __init__(self, hidden_size: int, value_width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, value_width, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.up(self.down(hidden_states))


class ParameterMatchedValueAdapter(nn.Module):
    """Conventional residual low-rank adapters on V, within 0.2% of M1 budget."""

    def __init__(self, shape: ModelShape, fma_cfg: FMAConfig) -> None:
        super().__init__()
        self.shape = shape
        self.layers = fma_cfg.resolved_layers(shape)
        self.layer_to_slot = {layer: slot for slot, layer in enumerate(self.layers)}
        budget = parameter_budget(shape, fma_cfg)
        self.rank = budget.control_rank
        self.adapters = nn.ModuleList(
            LowRankValueAdapter(shape.hidden_size, shape.value_width, self.rank)
            for _ in self.layers
        )

    def forward(self, standard_value: Tensor, hidden_states: Tensor, layer_idx: int) -> Tensor:
        slot = self.layer_to_slot[int(layer_idx)]
        return standard_value + self.adapters[slot](hidden_states)

    def added_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
