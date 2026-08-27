from __future__ import annotations

import pytest
import torch
from torch import nn

from rineya_pcsm.application import (
    NativeUpdatePlan,
    TensorUpdatePlan,
    apply_update_plan,
)
from rineya_pcsm.optimizer import ExactSupportAdamW


class TinyWeights(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1, 7, dtype=dtype).reshape(2, 3))
        self.bias = nn.Parameter(torch.tensor([7.0, 8.0], dtype=dtype))


def _element_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .reshape(tensor.numel(), tensor.element_size())
    )


def test_bf16_step_writes_only_support_and_keeps_sparse_fp32_state() -> None:
    model = TinyWeights(dtype=torch.bfloat16)
    # Deliberately unsorted: TensorUpdatePlan canonicalizes support and keeps A
    # paired with its original coordinate.
    plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=6,
                shape=(2, 3),
                flat_indices=[4, 1],
                stabilizer=[1.0, 0.5],
            ),
        )
    )
    applied = apply_update_plan(model, plan)
    optimizer = ExactSupportAdamW(
        applied,
        lr=0.25,
        betas=(0.0, 0.0),
        eps=0.0,
        weight_decay=0.2,
    )
    before = model.weight.detach().clone()

    # The autograd guard zeros every coordinate not selected by the plan.
    coefficients = torch.tensor(
        [[11.0, 2.0, 13.0], [17.0, 200.0, 19.0]], dtype=torch.float32
    )
    (model.weight.float() * coefficients).sum().backward()
    assert model.weight.grad is not None
    assert model.weight.grad.reshape(-1).tolist() == [0.0, 2.0, 0.0, 0.0, 200.0, 0.0]
    optimizer.step()

    after = model.weight.detach()
    outside = torch.tensor([0, 2, 3, 5])
    assert torch.equal(
        _element_bytes(before).index_select(0, outside),
        _element_bytes(after).index_select(0, outside),
    )
    # beta=(0, 0), eps=0 normalizes both positive gradients to 1.  A is
    # applied to the complete selected AdamW direction, not to the raw gradient.
    expected = before.reshape(-1).index_select(0, torch.tensor([1, 4])).float()
    scales = torch.tensor([0.5, 1.0])
    expected.add_(-0.25 * scales * (torch.ones(2) + 0.2 * expected))
    assert torch.equal(
        after.reshape(-1).index_select(0, torch.tensor([1, 4])),
        expected.to(torch.bfloat16),
    )
    assert not model.bias.requires_grad
    assert model.bias.grad is None

    state = optimizer.state[model.weight]
    assert state["flat_indices"].numel() == 2
    assert state["exp_avg"].shape == (2,)
    assert state["exp_avg_sq"].shape == (2,)
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert optimizer.selected_coordinate_count == 2


def test_selected_only_weight_decay_never_touches_frozen_values() -> None:
    model = TinyWeights()
    plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=6,
                flat_indices=[0],
                stabilizer=[0.5],
            ),
        )
    )
    optimizer = ExactSupportAdamW(
        apply_update_plan(model, plan),
        lr=0.5,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.4,
    )
    before_weight = model.weight.detach().clone()
    before_bias = model.bias.detach().clone()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()

    # Gradient direction is 1 and selected decay direction is 0.4; both are
    # scaled by A=0.5 after normalization.
    assert model.weight.reshape(-1)[0].item() == pytest.approx(0.65)
    assert torch.equal(model.weight.reshape(-1)[1:], before_weight.reshape(-1)[1:])
    assert torch.equal(model.bias, before_bias)


def test_parameter_without_gradient_does_not_advance_sparse_state() -> None:
    model = TinyWeights()
    plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=6,
                flat_indices=[2],
                stabilizer=[1.0],
            ),
        )
    )
    optimizer = ExactSupportAdamW(apply_update_plan(model, plan))
    before = model.weight.detach().clone()
    optimizer.step()
    assert optimizer.state[model.weight]["step"] == 0
    assert torch.equal(model.weight, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_native_weights_keep_fp32_moments(dtype: torch.dtype) -> None:
    model = TinyWeights(dtype=dtype)
    plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=6,
                flat_indices=[1, 4],
                stabilizer=[0.25, 1.0],
            ),
        )
    )
    optimizer = ExactSupportAdamW(apply_update_plan(model, plan), weight_decay=0.0)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    state = optimizer.state[model.weight]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_step_scale_controls_realized_first_adam_step_in_one_to_four_ratio() -> None:
    model = TinyWeights()
    plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=6,
                flat_indices=[0, 1],
                stabilizer=[0.25, 1.0],
            ),
        )
    )
    optimizer = ExactSupportAdamW(
        apply_update_plan(model, plan),
        lr=0.2,
        betas=(0.0, 0.0),
        eps=0.0,
        weight_decay=0.0,
    )
    before = model.weight.detach().clone()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    realized = (before - model.weight.detach()).reshape(-1)[:2]
    assert realized[0].item() == pytest.approx(0.05)
    assert realized[1].item() == pytest.approx(0.2)
    assert (realized[1] / realized[0]).item() == pytest.approx(4.0)
