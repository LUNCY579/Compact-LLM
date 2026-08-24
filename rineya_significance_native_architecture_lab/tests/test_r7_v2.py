from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import replace

import pytest

from rineya_significance_lab.benchmarks import (
    CONTEXT_LAYOUTS,
    R7_V2_GENERATOR_VERSION,
    generate_r7_v2_case,
    interpret_r7_v2_segments,
    solve_r7_v2_from_segments,
    solve_r7_v2_oracle,
    validate_r7_v2_case,
)


@pytest.mark.parametrize("layout", CONTEXT_LAYOUTS)
@pytest.mark.parametrize("seed", [410, 411, 412, 413])
def test_r7_v2_is_deterministic_across_layouts(seed: int, layout: str) -> None:
    first = generate_r7_v2_case(seed, 0.5, filler_count=12, layout=layout)  # type: ignore[arg-type]
    second = generate_r7_v2_case(seed, 0.5, filler_count=12, layout=layout)  # type: ignore[arg-type]
    assert first.to_dict() == second.to_dict()
    assert first.generator_version == R7_V2_GENERATOR_VERSION
    assert solve_r7_v2_oracle(first) == solve_r7_v2_from_segments(first) == first.answer
    assert validate_r7_v2_case(first)["status"] == "PASS"


def test_r7_v2_selected_seed_schedule_is_jointly_balanced() -> None:
    cases = [generate_r7_v2_case(seed, 0.5, filler_count=4) for seed in range(410, 418)]
    assert Counter(case.answer for case in cases) == {"YES": 4, "NO": 4}
    assert Counter((case.answer, str(case.metadata["active_branch"])) for case in cases) == {
        ("YES", "A"): 2,
        ("YES", "B"): 2,
        ("NO", "A"): 2,
        ("NO", "B"): 2,
    }
    assert max(Counter(case.answer for case in cases).values()) / len(cases) == 0.5


def test_r7_v2_has_no_role_id_or_status_frequency_shortcut() -> None:
    for seed in range(410, 418):
        case = generate_r7_v2_case(seed, 0.5, filler_count=0)
        assert not any(
            segment_id.startswith(("E", "D")) for segment_id in case.evidence_segment_ids
        )
        # Count exact conclusion phrases instead of substring occurrences.
        assert sum(" is READY." in segment.text for segment in case.segments) == 2
        assert sum(" is NOT READY." in segment.text for segment in case.segments) == 2


def test_r7_v2_rendered_solver_is_text_only_and_counterfactually_sensitive() -> None:
    source = inspect.getsource(interpret_r7_v2_segments)
    assert "case.oracle" not in source
    assert "case.answer" not in source
    assert "seed" not in inspect.signature(interpret_r7_v2_segments).parameters

    case = generate_r7_v2_case(410, 0.5, filler_count=0)
    target = str(case.oracle["target_unit"])
    decoy = str(case.oracle["decoy_unit"])
    active = str(case.oracle["active_branch"])
    active_fact_id, inactive_fact_id = ("P06", "P10") if active == "A" else ("P10", "P06")
    counterfactual = []
    for segment in case.segments:
        if segment.segment_id == active_fact_id:
            counterfactual.append(replace(segment, text=segment.text.replace(target, decoy)))
        elif segment.segment_id == inactive_fact_id:
            counterfactual.append(replace(segment, text=segment.text.replace(decoy, target)))
        else:
            counterfactual.append(segment)
    changed = interpret_r7_v2_segments(counterfactual, case.question)
    assert changed.answer != case.answer
    assert (
        changed.fired_override_ids
        != interpret_r7_v2_segments(case.segments, case.question).fired_override_ids
    )


def test_r7_v2_exact_evidence_is_outcome_specific_and_deletion_minimal() -> None:
    branch_a = generate_r7_v2_case(412, 0.5, filler_count=6)
    branch_b = generate_r7_v2_case(410, 0.5, filler_count=6)
    assert branch_a.evidence_segment_ids != branch_b.evidence_segment_ids
    for case in (branch_a, branch_b):
        by_id = {segment.segment_id: segment for segment in case.segments}
        proof = [by_id[segment_id] for segment_id in case.evidence_segment_ids]
        assert interpret_r7_v2_segments(proof, case.question).answer == case.answer
        for removed in case.evidence_segment_ids:
            ablated = [segment for segment in proof if segment.segment_id != removed]
            try:
                ablated_answer = interpret_r7_v2_segments(ablated, case.question).answer
            except ValueError:
                continue
            assert ablated_answer != case.answer


def test_r7_v2_layouts_cover_early_middle_late_and_distributed_geometry() -> None:
    early = generate_r7_v2_case(410, 0.5, filler_count=24, layout="all_early")
    middle = generate_r7_v2_case(410, 0.5, filler_count=24, layout="all_middle")
    late = generate_r7_v2_case(410, 0.5, filler_count=24, layout="all_late")
    distributed = generate_r7_v2_case(
        410,
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
