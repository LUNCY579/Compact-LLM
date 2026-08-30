from __future__ import annotations

import pytest
import torch

from rineya_cross_scale.delta_analysis import Accumulator


def test_accumulator_exact_alignment() -> None:
    accumulator = Accumulator()
    x = torch.tensor([1.0, -2.0, 3.0])
    accumulator.add(x, x * 2.0, torch.ones(3))
    result = accumulator.result()
    assert result["cosine"] == pytest.approx(1.0)
    assert result["sign_agreement"] == 1.0
    assert result["norm_ratio_delta4_over_delta8"] == pytest.approx(0.5)
