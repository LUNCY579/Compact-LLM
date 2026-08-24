from __future__ import annotations

import json
from pathlib import Path


def test_committed_calibration_views_are_separate_and_clustered() -> None:
    lab = Path(__file__).resolve().parents[1]
    root = lab / "results" / "benchmarks" / "halo_calibration_v3"
    inference = [json.loads(line) for line in (root / "inference.jsonl").read_text().splitlines()]
    evaluation = [json.loads(line) for line in (root / "evaluation.jsonl").read_text().splitlines()]
    assert len(inference) == len(evaluation) == 32
    assert len({row["cluster_id"] for row in inference}) == 32
    assert not any("answer" in row or "evidence_annotation" in row for row in inference)
    assert all(row["evidence_annotation"]["facts"] for row in evaluation)
    assert {row["case_id"] for row in inference} == {row["case_id"] for row in evaluation}
