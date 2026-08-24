from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rineya_significance_lab.benchmarks import (
    generate_case,
    generate_suite,
    read_cases_jsonl,
    solve_case,
    solve_case_from_segments,
    validate_case,
    write_cases_jsonl,
)


@pytest.mark.parametrize("family", ["R1", "R2", "R3", "R4"])
@pytest.mark.parametrize("depth", [0.05, 0.5, 0.95])
def test_r1_r4_cases_are_deterministic_and_independently_solved(family: str, depth: float) -> None:
    first = generate_case(family, 17, depth)
    second = generate_case(family, 17, depth)
    assert first.to_dict() == second.to_dict()
    assert solve_case(first) == first.answer
    assert solve_case_from_segments(first) == first.answer
    assert validate_case(first)["status"] == "PASS"
    assert abs(first.actual_evidence_depth - depth) < 0.12
    assert first.metadata["cluster_id"] == f"{family}-s000017"


def test_case_validator_detects_answer_corruption() -> None:
    case = generate_case("R1", 17, 0.5)
    with pytest.raises(ValueError, match="solver mismatch"):
        validate_case(replace(case, answer="not-the-answer"))


def test_suite_jsonl_round_trip(tmp_path: Path) -> None:
    cases = generate_suite([17, 23], [0.1, 0.9])
    path = write_cases_jsonl(tmp_path / "cases.jsonl", cases)
    loaded = read_cases_jsonl(path)
    assert [case.to_dict() for case in loaded] == [case.to_dict() for case in cases]
    assert len(loaded) == 16
