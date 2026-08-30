from __future__ import annotations

import pytest

from rineya_cross_scale.registry import (
    ExperimentRegistry,
    ExperimentSpec,
    RegistryError,
    alpha_label,
)


def spec(alpha: float = 0.005) -> ExperimentSpec:
    return ExperimentSpec(
        donor="4B",
        projection_strategy="P1",
        depth_mapping="D2",
        width_mapping="W2",
        merge_family="TASKVEC",
        coefficient=alpha,
        seed=17,
        output_dtype="float16",
        source_hashes={"supa": "a" * 64, "delta": "b" * 64},
    )


def test_alpha_labels() -> None:
    assert alpha_label(0.005) == "A0050"
    assert alpha_label(-0.02) == "AN0200"


def test_registry_is_append_only_and_idempotent(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.jsonl")
    experiment_id = registry.create(spec())
    assert registry.create(spec()) == experiment_id
    registry.status(experiment_id, "building")
    registry.status(experiment_id, "failed", error="intentional")
    events = registry.events()
    assert [event["status"] for event in events] == [
        "registered",
        "building",
        "failed",
    ]


def test_registry_rejects_unknown_status_target(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.jsonl")
    with pytest.raises(RegistryError, match="unknown experiment"):
        registry.status("missing", "failed")
