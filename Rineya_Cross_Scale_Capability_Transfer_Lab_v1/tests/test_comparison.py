from __future__ import annotations

import json

import pytest

from rineya_cross_scale.comparison import (
    compare_prediction_files,
    mcnemar_exact_two_sided,
    scenario_bootstrap,
)


def write_rows(path, scores, responses):
    with path.open("w", encoding="utf-8") as handle:
        for index, (score, response) in enumerate(zip(scores, responses, strict=True)):
            row = {
                "case_id": f"c{index}",
                "scenario_id": f"s{index // 2}",
                "family": "f",
                "prompt_sha256": str(index),
                "response_sha256": response,
                "score": {"score": score},
            }
            handle.write(json.dumps(row) + "\n")


def test_mcnemar_exact() -> None:
    assert mcnemar_exact_two_sided(0, 0) == 1.0
    assert mcnemar_exact_two_sided(0, 5) == pytest.approx(0.0625)


def test_cluster_bootstrap_is_deterministic() -> None:
    rows = [("s1", 1.0), ("s1", -1.0), ("s2", 1.0)]
    assert scenario_bootstrap(rows, seed=17, replicates=100) == scenario_bootstrap(
        rows, seed=17, replicates=100
    )


def test_compare_prediction_files(tmp_path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_rows(a, [1, 0, 1, 0], ["a", "b", "c", "d"])
    write_rows(b, [1, 1, 0, 0], ["a", "x", "y", "d"])
    report = compare_prediction_files(
        a, b, label_a="A", label_b="B", seed=17, replicates=100
    )
    assert report["paired"]["a_only"] == 1
    assert report["paired"]["b_only"] == 1
    assert report["response_changes"] == 2
