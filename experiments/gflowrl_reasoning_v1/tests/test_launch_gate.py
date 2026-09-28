from __future__ import annotations

import pytest

from experiments.gflowrl_reasoning_v1.launch_gate import (
    NativeLaunchBlocked,
    assess_native_launch,
    assert_current_branch_launch_allowed,
    assert_native_launch_allowed,
    load_current_contracts,
)


def test_checked_in_contracts_refuse_native_training():
    binding, fidelity, manifest = load_current_contracts()
    result = assess_native_launch(binding, fidelity, manifest)
    assert result["allowed"] is False
    assert result["training_authorized"] is False
    assert result["sealed_access"] == "DENIED"
    assert any("canonical R0" in reason for reason in result["reasons"])
    assert any("runtime fidelity" in reason for reason in result["reasons"])
    assert any("dataset" in reason for reason in result["reasons"])
    with pytest.raises(NativeLaunchBlocked):
        assert_current_branch_launch_allowed()


def test_gate_can_only_open_with_all_explicit_bindings():
    binding = {"requested_r0": {"status": "QUALIFIED_RL_BASELINE"}}
    fidelity = {"full_runtime_status": "QUALIFIED"}
    manifest = {
        "data": {"dataset_status": "BOUND_TRAIN_DEV_ONLY"},
        "sealed_access": {"denied": True},
    }
    result = assert_native_launch_allowed(binding, fidelity, manifest)
    assert result["allowed"] is True
    assert result["training_authorized"] is True


def test_sealed_denial_is_a_required_launch_invariant():
    binding = {"requested_r0": {"status": "QUALIFIED_RL_BASELINE"}}
    fidelity = {"full_runtime_status": "QUALIFIED"}
    manifest = {
        "data": {"dataset_status": "BOUND_TRAIN_DEV_ONLY"},
        "sealed_access": {"denied": False},
    }
    with pytest.raises(NativeLaunchBlocked, match="sealed"):
        assert_native_launch_allowed(binding, fidelity, manifest)
