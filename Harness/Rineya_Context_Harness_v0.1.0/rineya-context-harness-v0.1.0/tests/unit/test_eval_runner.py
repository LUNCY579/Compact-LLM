from __future__ import annotations

import json

import pytest

from rineya_harness.evals.runner import load_trials, run_eval, write_report


def test_eval_runner_round_trip(tmp_path) -> None:
    source = tmp_path / "trials.jsonl"
    source.write_text(
        json.dumps(
            {
                "case_id": "x",
                "before_active_ids": ["a", "b"],
                "after_active_ids": ["b"],
                "expected_evicted_ids": ["a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(load_trials(source)) == 1
    metrics = run_eval(source)
    target = write_report(metrics, tmp_path / "report.json")
    assert target.exists()


def test_eval_runner_reports_line_number(tmp_path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1"):
        load_trials(source)
