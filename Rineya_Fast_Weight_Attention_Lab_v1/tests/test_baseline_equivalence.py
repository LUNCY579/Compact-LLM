"""Baseline equivalence and arm parameter matching."""

from __future__ import annotations

import torch
from torch import nn

from rineya_fwa.falcon2 import FWAConfig, FastWeightBlock
from rineya_fwa.wrapper import FWAAttentionWrapper


class _ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("C", (), {"hidden_size": 16, "num_key_value_heads": 2})()
        self.inner = nn.Linear(16, 16, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        return self.inner(hidden_states)


def test_eval_alpha_zero_matches_unwrapped_logits():
    toy = _ToyLM()
    cfg = FWAConfig(hidden_size=16, n_kv_heads=2, rank=4, init_alpha=0.0, recurrent=True)
    block = FastWeightBlock(cfg)
    wrapped = FWAAttentionWrapper(toy.inner, block)
    wrapped.eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        a = toy.inner(x)
        b = wrapped(x)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_arm_b_and_c_parameter_counts_match():
    b = FastWeightBlock(FWAConfig(hidden_size=1024, n_kv_heads=8, rank=16, recurrent=False))
    c = FastWeightBlock(FWAConfig(hidden_size=1024, n_kv_heads=8, rank=16, recurrent=True))
    nb = sum(p.numel() for p in b.parameters())
    nc = sum(p.numel() for p in c.parameters())
    assert nb == nc
    d = FastWeightBlock(
        FWAConfig(hidden_size=1024, n_kv_heads=8, rank=16, recurrent=True, significance_gate=True)
    )
    nd = sum(p.numel() for p in d.parameters())
    assert nd > nc
