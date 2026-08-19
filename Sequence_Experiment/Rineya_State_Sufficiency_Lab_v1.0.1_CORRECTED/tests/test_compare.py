from __future__ import annotations

import json
from pathlib import Path

from rineya_state_lab.evaluation.compare import compare_runs


def _write(path: Path, protocol: str, values: list[bool]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(values):
            handle.write(
                json.dumps(
                    {
                        "request_id": f"{protocol}::p{index}",
                        "protocol": protocol,
                        "problem_id": f"p{index}",
                        "category": "cat",
                        "domain": "domain",
                        "difficulty": 1,
                        "raw_output": str(value),
                        "score": {"correct_strict": value},
                    }
                )
                + "\n"
            )


def test_compare_pairs_by_problem_and_reports_superiority(tmp_path: Path) -> None:
    baseline = tmp_path / "base.jsonl"
    candidate = tmp_path / "cand.jsonl"
    _write(baseline, "direct", [False] * 20)
    _write(candidate, "rollout", [True] * 20)
    result = compare_runs(
        baseline,
        candidate,
        tmp_path / "out",
        baseline_protocol="direct",
        candidate_protocol="rollout",
        bootstrap_samples=500,
        randomization_samples=500,
        seed=1,
    )
    assert result["decision"]["verdict"] == "SUPERIOR"
    assert (tmp_path / "out/comparison.json").is_file()
