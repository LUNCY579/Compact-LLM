from __future__ import annotations

import unittest

import torch
from torch import nn

from scripts.common.sparse_mlp import (
    OracleMaskedGatedMLP,
    RoutedSparseGatedMLP,
    TinyBlockRouter,
    block_scores_from_intermediate,
)


class FakeMLP(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


class SparseMLPTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_block_scores_shape(self) -> None:
        z = torch.randn(2, 3, 16)
        down = torch.randn(8, 16)
        scores = block_scores_from_intermediate(z, down, block_size=4)
        self.assertEqual(tuple(scores.shape), (2, 3, 4))
        self.assertTrue(torch.all(scores >= 0))

    def test_oracle_ratio_one_matches_dense(self) -> None:
        dense = FakeMLP()
        wrapper = OracleMaskedGatedMLP(dense, block_size=4, keep_ratio=1.0)
        hidden = torch.randn(2, 5, 8)
        torch.testing.assert_close(wrapper(hidden), dense(hidden), rtol=1e-5, atol=1e-6)

    def test_routed_reference_returns_correct_shape(self) -> None:
        dense = FakeMLP()
        router = TinyBlockRouter(hidden_size=8, router_width=4, num_blocks=4)
        wrapper = RoutedSparseGatedMLP(
            dense,
            router,
            block_size=4,
            keep_ratio=0.5,
            confidence_threshold=0.0,
        )
        hidden = torch.randn(1, 1, 8)
        output = wrapper(hidden)
        self.assertEqual(tuple(output.shape), (1, 1, 8))
        self.assertEqual(wrapper.stats.sparse_calls, 1)

    def test_routed_reference_falls_back_for_prefill(self) -> None:
        dense = FakeMLP()
        router = TinyBlockRouter(hidden_size=8, router_width=4, num_blocks=4)
        wrapper = RoutedSparseGatedMLP(
            dense,
            router,
            block_size=4,
            keep_ratio=0.5,
            confidence_threshold=0.0,
        )
        hidden = torch.randn(1, 6, 8)
        torch.testing.assert_close(wrapper(hidden), dense(hidden))
        self.assertEqual(wrapper.stats.dense_fallback_calls, 1)


if __name__ == "__main__":
    unittest.main()
