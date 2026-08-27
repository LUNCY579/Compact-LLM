from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from rineya_pcsm.application import (
    NativeUpdatePlan,
    TensorUpdatePlan,
    apply_update_plan,
)
from rineya_pcsm.optimizer import ExactSupportAdamW


class ResumeWeights(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[0.5, -1.0], [1.5, -2.0]], dtype=dtype)
        )


def _plan(*, stabilizer: tuple[float, float] = (0.3, 0.9)) -> NativeUpdatePlan:
    return NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=4,
                shape=(2, 2),
                flat_indices=[0, 3],
                stabilizer=stabilizer,
            ),
        )
    )


def _set_gradient(model: ResumeWeights, values: tuple[float, ...]) -> None:
    model.weight.grad = torch.tensor(values, dtype=model.weight.dtype).reshape(2, 2)


def _optimizer(model: ResumeWeights, plan: NativeUpdatePlan) -> ExactSupportAdamW:
    return ExactSupportAdamW(
        apply_update_plan(model, plan),
        lr=0.07,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=0.03,
    )


def _element_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .reshape(tensor.numel(), tensor.element_size())
    )


def _assert_frozen_coordinates_bitwise_equal(
    model: ResumeWeights, initial: torch.Tensor
) -> None:
    outside = torch.tensor([1, 2])
    assert torch.equal(
        _element_bytes(model.weight).index_select(0, outside),
        _element_bytes(initial).index_select(0, outside),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_resume_matches_uninterrupted_update_bitwise(dtype: torch.dtype) -> None:
    plan = _plan()
    reference = ResumeWeights(dtype=dtype)
    optimizer = _optimizer(reference, plan)
    gradients = (
        (1.0, 99.0, -99.0, -2.0),
        (-0.5, 44.0, 55.0, 3.0),
        (2.5, -77.0, 88.0, -1.5),
    )
    for gradient in gradients[:2]:
        _set_gradient(reference, gradient)
        optimizer.step()
    checkpoint_weight = reference.weight.detach().clone()
    checkpoint_optimizer = deepcopy(optimizer.state_dict())

    _set_gradient(reference, gradients[2])
    optimizer.step()

    resumed = ResumeWeights(dtype=dtype)
    with torch.no_grad():
        resumed.weight.copy_(checkpoint_weight)
    resumed_optimizer = _optimizer(resumed, plan)
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    _set_gradient(resumed, gradients[2])
    resumed_optimizer.step()

    assert torch.equal(resumed.weight, reference.weight)
    resumed_state = resumed_optimizer.state[resumed.weight]
    reference_state = optimizer.state[reference.weight]
    assert resumed_state["step"] == reference_state["step"] == 3
    assert resumed_state["exp_avg"].dtype == torch.float32
    assert resumed_state["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(resumed_state["exp_avg"], reference_state["exp_avg"])
    assert torch.equal(resumed_state["exp_avg_sq"], reference_state["exp_avg_sq"])


def test_load_rejects_support_and_plan_hash_mismatches() -> None:
    model = ResumeWeights()
    optimizer = _optimizer(model, _plan())
    checkpoint = optimizer.state_dict()

    different_a_model = ResumeWeights()
    different_a = _optimizer(different_a_model, _plan(stabilizer=(0.3, 0.8)))
    with pytest.raises(ValueError, match="plan_hash mismatch"):
        different_a.load_state_dict(checkpoint)

    different_support_plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=4,
                flat_indices=[0, 2],
                stabilizer=[0.3, 0.9],
            ),
        )
    )
    different_support_model = ResumeWeights()
    different_support = _optimizer(different_support_model, different_support_plan)
    with pytest.raises(ValueError, match="support_hash mismatch"):
        different_support.load_state_dict(checkpoint)


def test_load_rejects_tampered_indices_even_with_untouched_hash_metadata() -> None:
    model = ResumeWeights()
    optimizer = _optimizer(model, _plan())
    checkpoint = deepcopy(optimizer.state_dict())
    checkpoint["state"][0]["flat_indices"][1] = 2

    fresh = ResumeWeights()
    fresh_optimizer = _optimizer(fresh, _plan())
    with pytest.raises(ValueError, match="exact support mismatch"):
        fresh_optimizer.load_state_dict(checkpoint)


def test_checkpoint_contains_no_dense_optimizer_state() -> None:
    model = ResumeWeights(dtype=torch.bfloat16)
    optimizer = _optimizer(model, _plan())
    state = optimizer.state_dict()["state"][0]
    assert set(state) == {"step", "flat_indices", "exp_avg", "exp_avg_sq"}
    assert state["flat_indices"].numel() == 2
    assert state["exp_avg"].numel() == 2
    assert state["exp_avg_sq"].numel() == 2
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_many_step_momentum_decay_and_resume_keep_frozen_coordinates_bitwise() -> None:
    plan = _plan()
    reference = ResumeWeights()
    initial = reference.weight.detach().clone()
    optimizer = _optimizer(reference, plan)
    gradients = tuple(
        (
            0.25 + (step % 7) * 0.3,
            100.0 + step,
            -100.0 - step,
            -0.5 - (step % 5) * 0.4,
        )
        for step in range(64)
    )

    for gradient in gradients[:32]:
        _set_gradient(reference, gradient)
        optimizer.step()
        _assert_frozen_coordinates_bitwise_equal(reference, initial)
    checkpoint_weight = reference.weight.detach().clone()
    checkpoint_optimizer = deepcopy(optimizer.state_dict())

    for gradient in gradients[32:]:
        _set_gradient(reference, gradient)
        optimizer.step()
        _assert_frozen_coordinates_bitwise_equal(reference, initial)

    resumed = ResumeWeights()
    with torch.no_grad():
        resumed.weight.copy_(checkpoint_weight)
    resumed_optimizer = _optimizer(resumed, plan)
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    for gradient in gradients[32:]:
        _set_gradient(resumed, gradient)
        resumed_optimizer.step()
        _assert_frozen_coordinates_bitwise_equal(resumed, initial)

    assert torch.equal(_element_bytes(resumed.weight), _element_bytes(reference.weight))
    resumed_state = resumed_optimizer.state[resumed.weight]
    reference_state = optimizer.state[reference.weight]
    assert resumed_state["step"] == reference_state["step"] == 64
    assert resumed_state["flat_indices"].numel() == 2
    assert torch.equal(resumed_state["exp_avg"], reference_state["exp_avg"])
    assert torch.equal(resumed_state["exp_avg_sq"], reference_state["exp_avg_sq"])
