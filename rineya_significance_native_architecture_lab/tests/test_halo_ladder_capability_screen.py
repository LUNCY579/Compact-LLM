from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]


def _module():
    path = LAB / "scripts" / "run_halo_ladder_capability_screen.py"
    spec = importlib.util.spec_from_file_location("run_halo_ladder_capability_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("family", "prediction", "answer", "expected"),
    [
        ("R5", "The final code is 4812.", "4812", True),
        ("R6", "A+B-C gives 7, then RESULT = 14.", "14", True),
        ("R7", "Reasoning omitted. YES", "YES", True),
        ("R7", "NO", "YES", False),
        ("R8", "Final stack: (73, 4)", "73,4", True),
        ("R8", "Final stack: (73, 5)", "73,4", False),
    ],
)
def test_task_specific_scoring(family: str, prediction: str, answer: str, expected: bool) -> None:
    assert _module()._task_correct(family, prediction, [answer]) is expected


def test_screen_selection_is_eight_independent_clusters() -> None:
    import yaml

    module = _module()
    config = yaml.safe_load(
        (LAB / "configs" / "halo_r5_r8_capability_screen.yaml").read_text(encoding="utf-8")
    )
    inference = module._rows(
        LAB / "results" / "benchmarks" / "halo_r1_r8_ladder_v1" / "inference.jsonl"
    )
    selected = module._select(inference, dict(config["selection"]))
    assert len(selected) == 8
    assert {row["family"] for row in selected} == {"R5", "R6", "R7", "R8"}
    assert {row["seed"] for row in selected} == {300, 301}
    assert len({row["cluster_id"] for row in selected}) == 8


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _module()._task_correct("R2", "0,0", ["0,0"])
