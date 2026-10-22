from __future__ import annotations

import math

import torch

from rineya_pcsm.application import NativeUpdatePlan, TensorUpdatePlan
from rineya_pcsm.delta_audit import ParameterSnapshot, audit_parameter_delta


def _snapshot() -> ParameterSnapshot:
    return ParameterSnapshot(
        {
            "weight": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16),
            "bias": torch.tensor([5.0, 6.0], dtype=torch.bfloat16),
        }
    )


def _plan() -> NativeUpdatePlan:
    return NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=4,
                flat_indices=[1, 3],
                stabilizer=[0.5, 0.75],
            ),
        )
    )


def test_selected_only_delta_is_valid_and_reports_k_union_and_norms() -> None:
    before = _snapshot()
    after = {name: tensor.clone() for name, tensor in before.parameters.items()}
    after["weight"][1] += 1.0
    after["weight"][3] -= 2.0

    report = audit_parameter_delta(before, after, _plan())
    assert report.valid
    assert report.planned_k == 2
    assert report.support_union_size == 2
    assert report.total_native_coordinates == 6
    assert report.support_union_fraction == 1 / 3
    assert report.changed_union_size == 2
    assert report.selected_changed_count == 2
    assert report.outside_support_changed_count == 0
    assert report.selected_delta_l2 == math.sqrt(5.0)
    assert report.outside_support_delta_l2 == 0.0
    assert report.total_delta_l2 == math.sqrt(5.0)
    assert report.selected_delta_max_abs == 2.0
    assert report.leakage_examples == ()
    serialized = report.to_dict()
    assert serialized["total_native_coordinates"] == 6
    assert serialized["support_union_fraction"] == 1 / 3


def test_delta_audit_accepts_full_model_coordinate_denominator() -> None:
    before = _snapshot()
    report = audit_parameter_delta(
        before,
        before,
        _plan(),
        native_coordinate_count_override=100,
    )
    assert report.total_native_coordinates == 100
    assert report.support_union_fraction == 2 / 100


def test_bitwise_audit_detects_planned_and_unplanned_tensor_leakage() -> None:
    before = _snapshot()
    after = {name: tensor.clone() for name, tensor in before.parameters.items()}
    after["weight"][0] += 0.5
    after["bias"][1] -= 1.0

    report = audit_parameter_delta(before, after, _plan())
    assert not report.valid
    assert report.outside_support_changed_count == 2
    assert report.changed_union_size == 2
    assert report.outside_support_delta_l2 == math.sqrt(1.25)
    assert report.outside_support_delta_max_abs == 1.0
    assert report.leakage_examples == (("bias", 1), ("weight", 0))
    tensor_reports = {item.name: item for item in report.tensors}
    assert tensor_reports["bias"].selected_k == 0
    assert tensor_reports["bias"].outside_support_changed_count == 1
    assert tensor_reports["weight"].selected_k == 2


def test_bitwise_audit_catches_signed_zero_with_zero_numerical_norm() -> None:
    before = ParameterSnapshot(
        {
            "weight": torch.tensor([1.0, 2.0]),
            "bias": torch.tensor([0.0]),
        }
    )
    after = {name: tensor.clone() for name, tensor in before.parameters.items()}
    after["bias"][0] = -0.0
    plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=2,
                flat_indices=[0],
                stabilizer=[1.0],
            ),
        )
    )

    report = audit_parameter_delta(before, after, plan)
    assert not report.valid
    assert report.outside_support_changed_count == 1
    assert report.outside_support_delta_l2 == 0.0
    assert report.leakage_examples == (("bias", 0),)


def test_delta_audit_rejects_parameter_set_or_dtype_changes() -> None:
    before = _snapshot()
    missing = {"weight": before.parameters["weight"].clone()}
    try:
        audit_parameter_delta(before, missing, _plan())
    except ValueError as error:
        assert "parameter name set changed" in str(error)
    else:
        raise AssertionError("missing parameter was not rejected")

    wrong_dtype = {name: tensor.clone() for name, tensor in before.parameters.items()}
    wrong_dtype["bias"] = wrong_dtype["bias"].float()
    try:
        audit_parameter_delta(before, wrong_dtype, _plan())
    except ValueError as error:
        assert "identical before/after dtypes" in str(error)
    else:
        raise AssertionError("dtype change was not rejected")
