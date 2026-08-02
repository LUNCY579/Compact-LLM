from __future__ import annotations

from scripts.common.opportunity_evaluation import evaluate_opportunity
from scripts.common.quality_evaluation import (
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)
from scripts.common.quality_gate import compare_quality_summaries


def record(case_id: str = "case", *, category: str = "concise_clarification", expected: dict | None = None) -> dict:
    return {
        "id": case_id,
        "category": category,
        "messages": [
            {"role": "user", "content": "Which version are you using?"},
            {"role": "assistant", "content": "Which version do you mean?"},
        ],
        "reference": "Which version do you mean?",
        "expected": expected or {"min_tokens": 2, "max_tokens": 12, "should_ask_question": True},
    }


def assess(text: str, tokens: int = 8, *, row: dict | None = None) -> dict:
    row = row or record()
    return evaluate_response(
        row,
        text,
        tokens,
        visible_model_tokens=tokens,
        config=load_quality_config(),
        corpus_profile=build_corpus_profile([row]),
    )


def summary(score: float, *, severe_rate: float = 0.0, issue_cost: float = 0.0) -> dict:
    dims = {
        "comprehensibility": score,
        "semantic_coherence": score,
        "prompt_relevance": score,
        "intent_fulfillment": score,
        "task_adequacy": score,
        "logic_factual_consistency": score,
        "safety_boundary": score,
        "context_consistency": score,
        "response_completeness": score,
        "brevity_appropriateness": score,
        "conversation_naturalness": score,
        "content_anchor_preservation": score,
    }
    return {
        "quality_evaluation_version": "3.1",
        "case_count": 100,
        "mean_quality_score": score,
        "worst_decile_mean_quality_score": score,
        "mean_quality_dimensions": dims,
        "category_quality_scores": {"general": score},
        "mean_issue_cost": issue_cost,
        "severe_issue_case_rate": severe_rate,
    }


def test_repetition_collapse_is_zero_quality() -> None:
    result = assess("Here here here here here here here?", 7)
    assert result["quality_score"] == 0.0
    assert result["quality_hard_fail"] is True


def test_unicode_and_word_salad_are_zero_quality() -> None:
    result = assess("� Tigo politistas impressionoon Sm-elect Tsunlook", 12)
    assert result["quality_score"] == 0.0
    assert "unicode_replacement_corruption" in result["critical_failures"]


def test_instruction_or_reasoning_leakage_is_costly_but_not_a_phase_veto() -> None:
    result = assess("The user is asking which version, let me think.", 10)
    assert "instruction_or_reasoning_leakage" in result["major_issues"]
    assert result["case_issue_cost"] > 0.0


def test_healthy_clarification_is_understandable() -> None:
    result = assess("Which version do you mean?", 6)
    assert result["quality_score"] > 0.0
    assert result["intelligibility_pass"] is True


def test_informal_grammar_and_natural_followups_are_permitted() -> None:
    row = record(
        "casual",
        category="acknowledgment",
        expected={"min_tokens": 1, "max_tokens": 20},
    )
    row["messages"] = [
        {"role": "user", "content": "I backed up the checkpoint."},
        {"role": "assistant", "content": "Good call."},
    ]
    row["reference"] = "Good call."
    examples = [
        "Did it complete successfully?",
        "How did the metrics look?",
        "Did it succeed?",
        "How the other adapter looks?",
        "That layer you are targeting?",
        "Probably not tbh.",
        "Yeah, that sounds rough.",
        "Nope, I would leave it.",
    ]
    for reply in examples:
        result = assess(reply, len(reply.split()), row=row)
        assert result["quality_hard_fail"] is False, reply
        assert "unwanted_follow_up_question" not in result["major_issues"], reply
        assert result["quality_score"] > 0.0, reply


def test_question_is_not_unsupported_certainty() -> None:
    row = record("boundary", category="knowledge_boundary", expected={"min_tokens": 1, "max_tokens": 20})
    row["messages"] = [
        {"role": "user", "content": "What happened in that sector?"},
        {"role": "assistant", "content": "Which sector are you referring to?"},
    ]
    row["reference"] = "Which sector are you referring to?"
    result = assess("Which sector you are referring to?", 6, row=row)
    assert "unsupported_certainty" not in result["major_issues"]
    assert result["quality_hard_fail"] is False


def test_question_forbidden_only_when_explicit_contract_says_so() -> None:
    open_row = record("open", category="acknowledgment", expected={"min_tokens": 1, "max_tokens": 20})
    explicit_row = record(
        "explicit",
        category="acknowledgment",
        expected={"min_tokens": 1, "max_tokens": 20, "should_not_ask_question": True},
    )
    assert "unwanted_follow_up_question" not in assess("Did it work?", 4, row=open_row)["major_issues"]
    assert "violated_explicit_question_prohibition" in assess("Did it work?", 4, row=explicit_row)["major_issues"]


def test_aggregate_records_collapse_proportionally() -> None:
    healthy = assess("Which version do you mean?", 6)
    reference_rows = [{"id": f"r{i}", "category": "concise_clarification", **healthy} for i in range(10)]
    bad = assess("Here here here here here here here?", 7)
    candidate_rows = reference_rows[:-1] + [{"id": "bad", "category": "concise_clarification", **bad}]
    reference = aggregate_quality_rows(reference_rows)
    candidate = aggregate_quality_rows(candidate_rows)
    descriptive = compare_quality_summaries(reference, candidate)
    assert descriptive["eligible"] is True  # technical comparability only
    assert candidate["severe_issue_case_rate"] > reference["severe_issue_case_rate"]
    decision = evaluate_opportunity(
        reference,
        candidate,
        resource_gain=0.02,
        resource_name="test_resource_reduction",
    )
    assert decision["quality_opportunity_loss_fraction"] > 0.0


def test_opportunity_rule_is_strictly_proportional() -> None:
    reference = summary(1.0)
    slightly_lower = summary(0.99)
    worthwhile = evaluate_opportunity(reference, slightly_lower, resource_gain=0.03, resource_name="memory_reduction")
    not_worthwhile = evaluate_opportunity(reference, slightly_lower, resource_gain=0.001, resource_name="memory_reduction")
    assert worthwhile["eligible"] is True
    assert worthwhile["resource_gain_fraction"] > worthwhile["quality_opportunity_loss_fraction"]
    assert not_worthwhile["eligible"] is False


def test_missing_explicit_required_content_has_continuous_cost() -> None:
    row = record("required")
    row["expected"]["must_include_any"] = ["version", "option"]
    result = assess("Tell me more?", 4, row=row)
    assert "required_content_missing" in result["major_issues"]
    assert result["case_issue_cost"] > 0.0


def semantic_case(case_id: str, prompt: str, reference: str, category: str) -> dict:
    return {
        "id": case_id,
        "category": category,
        "messages": [{"role": "user", "content": prompt}],
        "reference": reference,
        "response_contract": {"open_ended": category not in {"factual_compression", "safety_boundary"}, "min_tokens": 1, "max_tokens": 32},
    }


def test_actual_phase2b_word_salad_no_longer_scores_high() -> None:
    cases = [
        (
            semantic_case("actual01", "Explain all of model compression in ten words.", "Choose one method.", "complex_escalation_in_character"),
            "You creates. you is you. that is not not—you. (— you, would find the system- result. ( share to the platform. (",
        ),
        (
            semantic_case("actual02", "I reduced the context window.", "How did latency and context accuracy change?", "contextual_follow_up"),
            "You gives. you gives. You can you is the duplicate.",
        ),
        (
            semantic_case("actual03", "Can I claim a speedup from theoretical FLOPs alone?", "No—measure wall-clock latency and memory.", "safety_boundary"),
            "I: the is you's size, and you, the amount.",
        ),
        (
            semantic_case("actual04", "Why is the output weird?", "Can you share one prompt and reply?", "clarification"),
            "I: you can you the works. I would you all-+ the work. First the work. You. The. you. i. if you",
        ),
        (
            semantic_case("actual05", "I am nervous about calling them.", "Write the first sentence you want to say.", "initiative_without_overreach"),
            "You gives the default. you convert to format.",
        ),
    ]
    for row, reply in cases:
        result = assess(reply, max(1, len(reply.split())), row=row)
        assert result["quality_score"] < 0.40, (row["id"], result)
        assert result["semantic_severity"] in {"critical", "major"}, (row["id"], result)
        assert (
            "unrecoverable_semantic_word_salad" in result["critical_failures"]
            or "semantically_unresponsive_output" in result["major_issues"]
            or "required_meaning_not_preserved" in result["major_issues"]
        ), (row["id"], result)


def test_nonempty_but_irrelevant_reply_does_not_receive_perfect_task_score() -> None:
    row = semantic_case(
        "irrelevant",
        "Can I claim a speedup from theoretical FLOPs alone?",
        "No—measure wall-clock latency and memory.",
        "safety_boundary",
    )
    result = assess("I like bananas.", 4, row=row)
    assert result["quality_dimensions"]["task_adequacy"] < 0.5
    assert result["prompt_relevance_score"] < 0.5
    assert "semantically_unresponsive_output" in result["major_issues"]


def test_semantic_summary_exposes_new_rates_and_dimensions() -> None:
    good_row = semantic_case("good", "I backed up the checkpoint.", "Good call.", "acknowledgment")
    bad_row = semantic_case("bad", "I reduced the context window.", "How did latency change?", "contextual_follow_up")
    good = assess("Good call.", 3, row=good_row)
    bad = assess("You gives. you gives. You can you is the duplicate.", 10, row=bad_row)
    aggregate = aggregate_quality_rows([
        {"id": "good", "category": "acknowledgment", **good},
        {"id": "bad", "category": "contextual_follow_up", **bad},
    ])
    assert aggregate["semantic_collapse_case_rate"] > 0.0
    assert aggregate["semantic_unresponsive_case_rate"] > 0.0
    assert "semantic_coherence" in aggregate["mean_quality_dimensions"]
    assert "prompt_relevance" in aggregate["mean_quality_dimensions"]
    assert "intent_fulfillment" in aggregate["mean_quality_dimensions"]
