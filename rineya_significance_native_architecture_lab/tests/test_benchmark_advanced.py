from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rineya_significance_lab.benchmarks import (
    ADVANCED_FAMILIES,
    CONTEXT_LAYOUTS,
    R1_CARDINALITIES,
    generate_advanced_case,
    generate_advanced_suite,
    generate_r1_cardinality_case,
    read_advanced_cases_jsonl,
    relayout_case,
    solve_advanced_case,
    solve_advanced_case_from_segments,
    validate_advanced_case,
    write_advanced_cases_jsonl,
)
from rineya_significance_lab.benchmarks.synthetic import generate_case


@pytest.mark.parametrize("family", ADVANCED_FAMILIES)
@pytest.mark.parametrize("layout", CONTEXT_LAYOUTS)
def test_r5_r8_are_deterministic_and_independently_solved(family: str, layout: str) -> None:
    first = generate_advanced_case(family, 41, 0.5, filler_count=12, layout=layout)  # type: ignore[arg-type]
    second = generate_advanced_case(family, 41, 0.5, filler_count=12, layout=layout)  # type: ignore[arg-type]
    assert first.to_dict() == second.to_dict()
    assert solve_advanced_case(first) == first.answer
    assert solve_advanced_case_from_segments(first) == first.answer
    assert validate_advanced_case(first)["status"] == "PASS"
    assert first.metadata["context_layout"] == layout
    assert first.metadata["cluster_id"] == f"{family}-s000041"


def test_context_layouts_change_evidence_geometry() -> None:
    early = generate_advanced_case("R6", 17, 0.5, filler_count=24, layout="all_early")
    middle = generate_advanced_case("R6", 17, 0.5, filler_count=24, layout="all_middle")
    late = generate_advanced_case("R6", 17, 0.5, filler_count=24, layout="all_late")
    distributed = generate_advanced_case(
        "R6",
        17,
        0.5,
        filler_count=24,
        layout="uniformly_distributed",
    )

    assert early.actual_evidence_depth < 0.25
    assert 0.35 < middle.actual_evidence_depth < 0.65
    assert late.actual_evidence_depth > 0.75
    positions = [
        index
        for index, segment in enumerate(distributed.segments)
        if segment.segment_id in distributed.evidence_segment_ids
    ]
    assert positions[-1] - positions[0] > len(distributed.segments) // 2


def test_frozen_r2_r4_cases_can_be_relaid_out_without_answer_drift() -> None:
    original = generate_case("R2", 17, 0.5, filler_count=12)
    updated = relayout_case(original, layout="early_middle_late", evidence_depth=0.5)
    assert updated.answer == original.answer
    assert updated.oracle == original.oracle
    assert updated.metadata["context_layout"] == "early_middle_late"
    assert updated.case_id != original.case_id


@pytest.mark.parametrize("key_count", R1_CARDINALITIES)
@pytest.mark.parametrize("layout", ["clustered", "early_late", "uniformly_distributed"])
def test_r1_required_cardinality_matrix(key_count: int, layout: str) -> None:
    case = generate_r1_cardinality_case(
        23,
        0.5,
        key_count=key_count,
        filler_count=10,
        layout=layout,  # type: ignore[arg-type]
    )
    assert len(case.evidence_segment_ids) == key_count
    assert case.metadata["required_fact_count"] == key_count
    assert case.answer


def test_advanced_validator_detects_corruption() -> None:
    case = generate_advanced_case("R8", 19, 0.5)
    with pytest.raises(ValueError, match="solver mismatch"):
        validate_advanced_case(replace(case, answer="corrupt"))


def test_advanced_suite_jsonl_round_trip(tmp_path: Path) -> None:
    cases = generate_advanced_suite(
        [17, 23],
        [0.1, 0.9],
        layouts=["clustered", "random"],
        filler_count=8,
    )
    path = write_advanced_cases_jsonl(tmp_path / "advanced.jsonl", cases)
    loaded = read_advanced_cases_jsonl(path)
    assert [case.to_dict() for case in loaded] == [case.to_dict() for case in cases]
    assert len(loaded) == 32


def test_r8_generator_never_emits_stack_underflow() -> None:
    for seed in range(64):
        case = generate_advanced_case("R8", seed, 0.5, filler_count=0)
        assert solve_advanced_case_from_segments(case) == case.answer
