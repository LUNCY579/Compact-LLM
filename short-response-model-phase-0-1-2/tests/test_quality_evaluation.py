from __future__ import annotations

from scripts.common.quality_evaluation import (
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)
from scripts.common.quality_gate import compare_quality_summaries


def record(case_id: str = "case") -> dict:
    return {
        "id": case_id,
        "category": "concise_clarification",
        "messages": [
            {"role": "user", "content": "Which version are you using?"},
            {"role": "assistant", "content": "Which version do you mean?"},
        ],
        "reference": "Which version do you mean?",
        "expected": {"min_tokens": 2, "max_tokens": 12, "should_ask_question": True},
    }


def assess(text: str, tokens: int = 8) -> dict:
    row = record()
    return evaluate_response(
        row,
        text,
        tokens,
        visible_model_tokens=tokens,
        config=load_quality_config(),
        corpus_profile=build_corpus_profile([row]),
    )


def test_repetition_collapse_is_zero_quality() -> None:
    result = assess("Here here here here here here here?", 7)
    assert result["quality_score"] == 0.0
    assert result["quality_hard_fail"] is True
    assert result["quality_pass"] is False


def test_unicode_and_word_salad_are_zero_quality() -> None:
    result = assess("� Tigo politistas impressionoon Sm-elect Tsunlook", 12)
    assert result["quality_score"] == 0.0
    assert "unicode_replacement_corruption" in result["critical_failures"]


def test_instruction_or_reasoning_leakage_fails_semantics() -> None:
    result = assess("The user is asking which version, let me think.", 10)
    assert "instruction_or_reasoning_leakage" in result["major_issues"]
    assert result["quality_pass"] is False


def test_healthy_clarification_passes() -> None:
    result = assess("Which version do you mean?", 6)
    assert result["quality_score"] >= 0.80
    assert result["quality_pass"] is True
    assert result["intelligibility_pass"] is True


def test_tail_metrics_and_gate_reject_one_collapsed_case() -> None:
    healthy = assess("Which version do you mean?", 6)
    reference_rows = [
        {"id": f"r{i}", "category": "concise_clarification", **healthy}
        for i in range(10)
    ]
    bad = assess("Here here here here here here here?", 7)
    candidate_rows = reference_rows[:-1] + [
        {"id": "bad", "category": "concise_clarification", **bad}
    ]
    reference = aggregate_quality_rows(reference_rows)
    candidate = aggregate_quality_rows(candidate_rows)
    comparison = compare_quality_summaries(
        reference,
        candidate,
        max_mean_quality_loss=1.0,
        max_quality_index_loss=1.0,
        max_worst_decile_loss=1.0,
        max_category_quality_loss_allowed=1.0,
        max_semantic_pass_loss=1.0,
        max_reference_alignment_loss=1.0,
        min_quality_pass_rate=0.0,
        min_intelligibility_pass_rate=1.0,
        min_semantic_pass_rate=0.0,
        min_complete_utterance_rate=0.0,
        max_hard_failure_cases=0,
        max_critical_failures=0,
        max_major_issue_cases=10,
    )
    assert comparison["eligible"] is False
    assert comparison["hard_failure_case_count"] == 1


def test_fragmented_micro_lines_are_rejected() -> None:
    result = assess("F\n\nS\n\nP\n\nR\n\nG", 5)
    assert "fragmented_micro_line_output" in result["major_issues"]
    assert result["quality_pass"] is False


def test_missing_explicit_required_content_is_major() -> None:
    row = record("required")
    row["expected"]["must_include_any"] = ["version", "option"]
    result = evaluate_response(
        row,
        "Tell me more?",
        4,
        visible_model_tokens=4,
        config=load_quality_config(),
        corpus_profile=build_corpus_profile([row]),
    )
    assert "required_content_missing" in result["major_issues"]
    assert result["quality_pass"] is False


def test_category_semantic_collapse_cannot_hide_in_global_mean() -> None:
    healthy = assess("Which version do you mean?", 6)
    reference_rows = [
        {"id": f"r{i}", "category": "concise_clarification", **healthy}
        for i in range(20)
    ]
    candidate_rows = [dict(row) for row in reference_rows]
    candidate_rows[0] = dict(candidate_rows[0])
    candidate_rows[0]["major_issues"] = ["failed_required_clarification"]
    candidate_rows[0]["semantic_pass"] = False
    candidate_rows[0]["quality_pass"] = False
    reference = aggregate_quality_rows(reference_rows)
    candidate = aggregate_quality_rows(candidate_rows)
    comparison = compare_quality_summaries(
        reference,
        candidate,
        max_mean_quality_loss=1.0,
        max_quality_index_loss=1.0,
        max_worst_decile_loss=1.0,
        max_category_quality_loss_allowed=0.04,
        max_semantic_pass_loss=1.0,
        max_reference_alignment_loss=1.0,
        min_quality_pass_rate=0.0,
        min_intelligibility_pass_rate=0.0,
        min_semantic_pass_rate=0.0,
        min_complete_utterance_rate=0.0,
        max_hard_failure_cases=10,
        max_critical_failures=10,
        max_major_issue_cases=10,
    )
    assert comparison["eligible"] is False
    assert comparison["maximum_category_semantic_pass_loss"] >= 0.05
