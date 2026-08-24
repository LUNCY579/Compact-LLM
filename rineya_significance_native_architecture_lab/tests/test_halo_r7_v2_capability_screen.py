from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]


def _module():
    path = LAB / "scripts" / "run_halo_r7_v2_capability_screen.py"
    spec = importlib.util.spec_from_file_location("run_halo_r7_v2_capability_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("prediction", "answer", "expected"),
    [
        ("YES", "YES", True),
        ("Final answer: NO.", "NO", True),
        ("The base says NO, but after the override: YES", "YES", True),
        ("NO", "YES", False),
        ("READY", "YES", False),
    ],
)
def test_r7_v2_task_scoring(prediction: str, answer: str, expected: bool) -> None:
    assert _module()._task_correct(prediction, [answer]) is expected


def test_r7_v2_screen_selects_eight_versioned_independent_clusters() -> None:
    import yaml

    root = LAB / "results" / "benchmarks" / "halo_r7_v2"
    assert root.is_dir()
    module = _module()
    config = yaml.safe_load(
        (LAB / "configs" / "halo_r7_v2_capability_screen.yaml").read_text(encoding="utf-8")
    )
    rows = [json.loads(line) for line in (root / "inference.jsonl").read_text().splitlines()]
    selected = module._select(rows, dict(config["selection"]))
    assert len(selected) == 8
    assert len({row["cluster_id"] for row in selected}) == 8
    assert {row["seed"] for row in selected} == set(range(410, 418))
    assert {row["benchmark_version"] for row in selected} == {"r7-v2.0.0"}


def test_r7_v2_screen_rejects_unversioned_rows() -> None:
    module = _module()
    selection = {
        "benchmark_version": "r7-v2.0.0",
        "seeds": [410],
        "context_layout": "clustered",
        "requested_evidence_depth": 0.5,
        "expected_independent_clusters": 1,
    }
    row = {
        "benchmark_version": "r7-v1",
        "family": "R7",
        "seed": 410,
        "context_layout": "clustered",
        "requested_evidence_depth": 0.5,
        "cluster_id": "R7v2-s000410",
    }
    with pytest.raises(RuntimeError, match="selection"):
        module._select([row], selection)


def test_committed_r7_v2_screen_records_all_yes_collapse_and_stop() -> None:
    run = LAB / "results" / "runs" / "halo_r7_v2_capability_screen"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "result_audit.json").read_text(encoding="utf-8"))
    assert summary["correct_clusters"] == 4
    assert summary["by_true_answer"] == {
        "YES": {"clusters": 4, "correct": 4},
        "NO": {"clusters": 4, "correct": 0},
    }
    assert summary["advance_gate"]["passed"] is False
    assert audit["prediction_counts"] == {"YES": 8, "NO": 0, "OTHER": 0}
    assert audit["observed_exceeds_best_constant_baseline"] is False
    assert audit["compression_decision"] == "STOP_NO_R7_V2_COMPRESSION_SCREEN"
