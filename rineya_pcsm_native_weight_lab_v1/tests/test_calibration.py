from __future__ import annotations

import pytest
import torch

from rineya_pcsm.calibration import first_order_loss_change, normalized_alignment


def test_first_order_change_respects_support_and_stabilizer_ratio() -> None:
    meta = torch.ones(4)
    cal = torch.ones(4)
    support = torch.tensor([True, True, False, False])
    stabilizer = torch.tensor([0.25, 1.0, 1.0, 1.0], dtype=torch.float32)
    change = first_order_loss_change(meta, cal, support, stabilizer, learning_rate=1.0)
    assert change.item() == pytest.approx(-1.25)


def test_normalized_alignment_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        normalized_alignment(torch.zeros(3), torch.ones(3))


def test_first_order_change_preserves_ste_meta_gradient() -> None:
    support = torch.tensor([0.0, 1.0], requires_grad=True)
    change = first_order_loss_change(
        torch.ones(2),
        torch.ones(2),
        support,
        torch.ones(2, dtype=torch.float32),
        learning_rate=1.0,
    )
    change.backward()
    assert support.grad is not None
    assert torch.equal(support.grad, torch.tensor([-1.0, -1.0]))


def test_missing_calibration_gradient_fails_closed() -> None:
    with pytest.raises(TypeError, match="calibration_gradient"):
        first_order_loss_change(
            torch.ones(2),
            None,  # type: ignore[arg-type]
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.float32),
            learning_rate=1.0,
        )
