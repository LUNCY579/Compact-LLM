from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_abstention_scale_ensemble.py"
SPEC = importlib.util.spec_from_file_location("build_abstention_scale_ensemble", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(row_id: str, prediction: str, scale: float) -> dict:
    return {
        "id": row_id,
        "status": "ok",
        "prediction": prediction,
        "adapter_scale": scale,
    }


def test_high_arm_is_used_only_for_normalized_unknown() -> None:
    low = [row("a", "Talia Frost", 0.75), row("b", "Sunday", 0.75)]
    high = [row("a", "Different", 1.25), row("b", "Answer: UNKNOWN.", 1.25)]

    result, abstained = MODULE.build_ensemble(low, high)

    assert [item["prediction"] for item in result] == ["Talia Frost", "Answer: UNKNOWN."]
    assert [item["selected_arm"] for item in result] == ["low", "high"]
    assert abstained == 1


def test_mismatched_coverage_is_rejected() -> None:
    with pytest.raises(ValueError, match="same ids"):
        MODULE.build_ensemble([row("a", "x", 0.75)], [row("b", "x", 1.25)])
