from __future__ import annotations

import torch

from experiments.factorized_memory_attention_v1.fma.config import FMAConfig
from experiments.factorized_memory_attention_v1.fma.toy import ToyLM, tiny_shape


def make_model(gate: float = 0.0) -> ToyLM:
    torch.manual_seed(20260925)
    return ToyLM(tiny_shape(), FMAConfig(rank=4, layers=(0,), gate=gate)).eval()


def test_causal_behavior_has_no_future_token_leakage() -> None:
    model = make_model(gate=1.0)
    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 9, 4]])
    first, _ = model(prefix)
    second, _ = model(changed_future)
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6, rtol=1e-6)


def test_cache_continuation_matches_full_causal_forward() -> None:
    model = make_model(gate=1.0)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    full, _ = model(ids)
    _, cache = model(ids[:, :4], use_cache=True)
    continued, _ = model(ids[:, 4:], past=cache, use_cache=True)
    assert torch.allclose(full[:, 4], continued[:, 0], atol=1e-6, rtol=1e-6)


def test_micro_generation_is_finite_and_deterministic() -> None:
    ids = torch.tensor([[1, 2, 3]])
    first = make_model(gate=1.0).generate(ids, max_new_tokens=4)
    second = make_model(gate=1.0).generate(ids, max_new_tokens=4)
    assert torch.equal(first, second)
    assert first.shape == (1, 7)
    assert torch.isfinite(first.float()).all()


def test_gate_zero_toy_path_is_finite() -> None:
    model = make_model(gate=0.0)
    logits, _ = model(torch.tensor([[1, 2, 3]]))
    assert torch.isfinite(logits).all()
