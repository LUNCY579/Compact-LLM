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


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 2)
        self.second = nn.Linear(2, 1)


def _plan() -> NativeUpdatePlan:
    return NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="first.weight",
                numel=6,
                shape=(2, 3),
                flat_indices=[5, 1],
                stabilizer=[0.8, 0.4],
            ),
        )
    )


def test_application_freezes_unplanned_tensors_and_grad_coordinates() -> None:
    model = TwoLayer()
    plan = _plan()
    applied = apply_update_plan(model, plan)

    assert model.first.weight.requires_grad
    assert not model.first.bias.requires_grad
    assert not model.second.weight.requires_grad
    assert not model.second.bias.requires_grad
    model.first.weight.sum().backward()
    assert model.first.weight.grad is not None
    expected = torch.zeros(6)
    expected[[1, 5]] = 1.0
    assert torch.equal(model.first.weight.grad.reshape(-1), expected)
    assert applied.k == plan.k == 2
    assert applied.support_hash == plan.support_hash
    assert applied.plan_hash == plan.plan_hash

    applied.restore()
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_plan_artifact_round_trip_and_content_hash_validation() -> None:
    plan = _plan()
    artifact = plan.to_dict()
    loaded = NativeUpdatePlan.from_dict(artifact)
    assert loaded.support_hash == plan.support_hash
    assert loaded.plan_hash == plan.plan_hash
    assert loaded.entries[0].flat_indices.tolist() == [1, 5]
    # A followed the deliberately unsorted source indices.
    assert torch.equal(loaded.entries[0].stabilizer, torch.tensor([0.4, 0.8]))

    tampered = deepcopy(artifact)
    tampered["entries"][0]["flat_indices"][0] = 2
    with pytest.raises(ValueError, match="declared support_hash"):
        NativeUpdatePlan.from_dict(tampered)


def test_stale_plan_application_is_transactional() -> None:
    model = TwoLayer()
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    stale = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="first.weight",
                numel=7,
                flat_indices=[1],
                stabilizer=[1.0],
            ),
        )
    )
    with pytest.raises(ValueError, match="plan numel"):
        apply_update_plan(model, stale)
    current_flags = [parameter.requires_grad for parameter in model.parameters()]
    assert current_flags == original_flags


@pytest.mark.parametrize(
    ("indices", "stabilizer", "message"),
    [
        ([1, 1], [0.5, 0.5], "unique"),
        ([6], [0.5], "out of range"),
        ([1], [float("nan")], "non-finite"),
        ([1], [-0.1], "strictly positive"),
    ],
)
def test_plan_rejects_invalid_exact_support(
    indices: list[int], stabilizer: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TensorUpdatePlan(
            name="first.weight",
            numel=6,
            flat_indices=indices,
            stabilizer=stabilizer,
        )
