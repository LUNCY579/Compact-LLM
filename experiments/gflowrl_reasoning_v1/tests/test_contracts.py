from __future__ import annotations

import pytest

from experiments.gflowrl_reasoning_v1.contracts import (
    DatasetBinding,
    Termination,
    assert_sealed_asset_denied,
    runtime_fingerprint,
    validate_termination,
)


def test_dataset_split_binding_allows_train_and_denies_confirmation():
    DatasetBinding("qualified-train", "train", "data/train.jsonl", "sha").assert_optimization_allowed()
    with pytest.raises(PermissionError):
        DatasetBinding("confirmation", "confirmation", "data/confirmation.jsonl", "sha", sealed=True).assert_optimization_allowed()


def test_sealed_asset_denial_does_not_read_asset():
    with pytest.raises(PermissionError, match="sealed"):
        assert_sealed_asset_denied("/nonexistent/confirmation/answers.json")
    with pytest.raises(PermissionError, match="sealed"):
        assert_sealed_asset_denied("/nonexistent/sealed/answers.json")


def test_termination_contract():
    validate_termination(Termination(eos_seen=True, truncated=False, valid=True))
    with pytest.raises(ValueError):
        validate_termination(Termination(eos_seen=False, truncated=True, valid=True))


def test_runtime_fingerprint_is_explicit():
    value = runtime_fingerprint()
    assert value["python"]
    assert value["torch"]
    assert "cuda_available" in value


def test_bounded_resource_probe_is_objective_only():
    from experiments.gflowrl_reasoning_v1.resource_probe import run_probe

    value = run_probe(seconds=0.01, seed=7)
    assert value["probe"] == "objective_only"
    assert value["model_loaded"] is False
    assert value["training_occurred"] is False
    assert value["steps"] > 0
