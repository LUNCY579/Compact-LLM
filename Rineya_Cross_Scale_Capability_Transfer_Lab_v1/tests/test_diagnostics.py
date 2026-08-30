from __future__ import annotations

import pytest
import torch

from rineya_cross_scale.diagnostics import classify_tensor, tensor_delta_stats


def test_classification_and_pair_stats() -> None:
    key = "model.layers.3.self_attn.q_proj.weight"
    assert classify_tensor(key) == {"layer": 3, "module": "attention.q"}
    delta4 = torch.tensor([1.0, -2.0, 0.0, 4.0])
    delta8 = torch.tensor([2.0, -1.0, 0.0, -4.0])
    stats = tensor_delta_stats(key, delta4, base=torch.ones(4), other=delta8)
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0
    assert stats["pair_sign_agreement"] == pytest.approx(0.75)
    assert stats["relative_delta_norm"] > 0
