from __future__ import annotations

from scripts.common.opportunity_evaluation import evaluate_opportunity
from scripts.common.quality_evaluation import (
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)


FORBIDDEN_CATCH_ALL_LABELS = {
    "semantically_unresponsive_output",
    "weak_prompt_reply_relevance",
    "weak_content_anchor_preservation",
    "high_out_of_benchmark_vocabulary",
    "answer_polarity_contradiction",
    "unrecoverable_semantic_word_salad",
    "probable_semantic_word_salad",
    "unintelligible_word_salad",
    "probable_word_salad",
}


def record(
    case_id: str = "case",
    *,
    category: str = "conversation",
    prompt: str = "I backed up the checkpoint.",
    reference: str = "Good call.",
    contract: dict | None = None,
) -> dict:
    return {
        "id": case_id,
        "category": category,
        "messages": [{"role": "user", "content": prompt}],
        "reference": reference,
        "response_contract": contract
        or {
            "schema_version": "4.3",
            "open_ended": True,
            "min_tokens": 1,
            "max_tokens": 32,
            "expected_language": "en",
            "requirements": [],
            "prohibitions": [],
        },
    }


def requirement_contract(label: str, description: str, pattern: str, *, severity: str = "major") -> dict:
    return {
        "schema_version": "4.3",
        "open_ended": False,
        "min_tokens": 1,
        "max_tokens": 32,
        "expected_language": "en",
        "requirements": [
            {
                "id": label,
                "label": label,
                "severity": severity,
                "description": description,
                "failure_evidence": description,
                "match": {"any_regex": [pattern]},
            }
        ],
        "prohibitions": [],
        "domains": {"factual": [label]},
    }


def assess(text: str, *, row: dict | None = None, tokens: int | None = None) -> dict:
    row = row or record()
    token_count = tokens if tokens is not None else max(1, len(text.split()))
    return evaluate_response(
        row,
        text,
        token_count,
        visible_model_tokens=token_count,
        config=load_quality_config(),
        corpus_profile=build_corpus_profile([row]),
    )


def summary(score: float, *, critical_rate: float = 0.0, defect_rate: float = 0.0, issue_cost: float = 0.0) -> dict:
    dims = {
        "comprehensibility": score,
        "mechanical_integrity": score,
        "contract_fulfillment": score,
        "prompt_function_alignment": score,
        "task_adequacy": score,
        "factual_correctness": score,
        "logic_factual_consistency": score,
        "safety_boundary": score,
        "context_consistency": score,
        "response_completeness": score,
        "brevity_appropriateness": score,
        "conversation_naturalness": score,
    }
    return {
        "quality_evaluation_version": "4.3",
        "case_count": 100,
        "mean_quality_score": score,
        "worst_decile_mean_quality_score": score,
        "mean_quality_dimensions": dims,
        "category_quality_scores": {"general": score},
        "mean_issue_cost": issue_cost,
        "critical_defect_case_rate": critical_rate,
        "severe_issue_case_rate": critical_rate,
        "observable_defect_case_rate": defect_rate,
        "contract_violation_case_rate": defect_rate,
        "mechanical_failure_case_rate": critical_rate,
    }


def labels(result: dict) -> set[str]:
    return {
        *(result.get("critical_failures") or []),
        *(result.get("major_issues") or []),
        *(result.get("minor_issues") or []),
    }


def test_repetition_collapse_has_concrete_zero_quality_label() -> None:
    result = assess("Here here here here here here here", tokens=7)
    assert result["quality_score"] == 0.0
    assert result["quality_hard_fail"] is True
    assert "repeated_token_loop" in result["critical_failures"]


def test_encoding_corruption_has_concrete_zero_quality_label() -> None:
    result = assess("� Tigo politistas impressionoon Sm-elect Tsunlook", tokens=12)
    assert result["quality_score"] == 0.0
    assert "encoding_replacement_character" in result["critical_failures"]


def test_real_word_salad_is_labeled_by_visible_structure_not_subjective_semantics() -> None:
    examples = [
        "You gives. you gives. You can you is the duplicate.",
        "I: the is you's size, and you, the amount.",
        "You creates. you is you. that is not not—you.",
    ]
    for reply in examples:
        result = assess(reply)
        assert result["quality_score"] == 0.0, (reply, result)
        assert "broken_clause_sequence" in result["critical_failures"], (reply, result)
        assert not (labels(result) & FORBIDDEN_CATCH_ALL_LABELS), (reply, result)


def test_internal_reasoning_leakage_is_precisely_named() -> None:
    result = assess("The user is asking which version, let me think.")
    assert "internal_reasoning_leakage" in result["major_issues"]
    assert result["case_issue_cost"] > 0.0


def test_informal_grammar_followups_and_alternate_social_replies_are_permitted() -> None:
    examples = [
        "Did it complete successfully?",
        "How did the metrics look?",
        "Did it succeed?",
        "How the other adapter looks?",
        "That layer you are targeting?",
        "Probably not tbh.",
        "Yeah, that sounds rough.",
        "Nope, I would leave it.",
        "That is not something I can take lightly.",
        "Perfect.",
        "That is impressive!",
    ]
    for reply in examples:
        result = assess(reply)
        assert result["quality_hard_fail"] is False, (reply, result)
        assert not result["major_issues"], (reply, result)
        assert not (labels(result) & FORBIDDEN_CATCH_ALL_LABELS), (reply, result)


def test_unusual_or_foreign_vocabulary_is_not_a_scored_defect() -> None:
    unusual = assess("I cannot verify real-time events without a sensor network.")
    chinese = assess("我不能确认实时结果。")
    assert not labels(unusual)
    assert not labels(chinese)
    assert unusual["case_issue_cost"] == 0.0
    assert chinese["case_issue_cost"] == 0.0
    assert any(item["label"] == "expected_language_review" for item in chinese["review_diagnostics"])


def test_reference_overlap_is_review_only_and_zero_cost() -> None:
    row = record(reference="Good call.")
    result = assess("Did it complete successfully?", row=row)
    assert result["reference_importance"] == 0.0
    assert result["case_issue_cost"] == 0.0
    assert not any(item["label"] == "low_reference_wording_overlap" for item in result["review_diagnostics"])


def test_question_rule_exists_only_when_explicitly_declared() -> None:
    open_result = assess("Did it work?")
    strict_row = record(
        contract={
            "schema_version": "4.3",
            "open_ended": False,
            "min_tokens": 1,
            "max_tokens": 20,
            "question_forbidden": True,
            "question_forbidden_label": "forbidden_question_present",
            "requirements": [],
            "prohibitions": [],
        }
    )
    strict_result = assess("Did it work?", row=strict_row)
    assert "forbidden_question_present" not in labels(open_result)
    assert "forbidden_question_present" in strict_result["major_issues"]


def test_explicit_required_fact_produces_specific_contract_label() -> None:
    row = record(
        prompt="What was the peak VRAM?",
        reference="6.4 GB.",
        contract=requirement_contract(
            "peak_vram_fact_missing_or_wrong",
            "The reply must preserve the stated peak VRAM value of 6.4 GB.",
            r"\b6\.4\s*gb\b",
        ),
    )
    good = assess("6.4 GB.", row=row)
    bad = assess("About 8 GB.", row=row)
    assert not labels(good)
    assert "peak_vram_fact_missing_or_wrong" in bad["major_issues"]
    assert bad["contract_fulfillment_score"] == 0.0


def test_irrelevant_answer_is_not_invented_as_a_defect_without_contract() -> None:
    # v4 deliberately refuses to invent a subjective relevance failure. The
    # benchmark author must state the actual required evidence or fact.
    result = assess("I like bananas.")
    assert not labels(result)
    assert result["task_adequacy"] if "task_adequacy" in result else True


def test_factual_inadequacy_is_caught_by_explicit_observable_requirement() -> None:
    row = record(
        prompt="Can I claim a speedup from theoretical FLOPs alone?",
        reference="No—measure wall-clock latency and memory.",
        contract=requirement_contract(
            "measured_runtime_evidence_missing",
            "The reply must require measured wall-clock/runtime/latency evidence.",
            r"\b(?:wall[- ]?clock|runtime|latency|benchmark|measure)\b",
        ),
    )
    good = assess("No—measure wall-clock latency.", row=row)
    bad = assess("No—quality and hardware are required.", row=row)
    assert not labels(good)
    assert "measured_runtime_evidence_missing" in bad["major_issues"]


def test_aggregate_reports_observable_and_contract_rates() -> None:
    good = assess("Good call.")
    collapse = assess("Here here here here here here")
    strict_row = record(
        contract=requirement_contract("privacy_assurance_missing", "The reply must promise not to share the disclosure.", r"\b(?:not share|keep .* private|safe with me)\b")
    )
    contract_failure = assess("That is understandable.", row=strict_row)
    aggregate = aggregate_quality_rows(
        [
            {"id": "good", "category": "general", **good},
            {"id": "collapse", "category": "general", **collapse},
            {"id": "contract", "category": "privacy", **contract_failure},
        ]
    )
    assert aggregate["critical_defect_case_rate"] == 1 / 3
    assert aggregate["observable_defect_case_rate"] == 2 / 3
    assert aggregate["mechanical_failure_case_rate"] == 1 / 3
    assert aggregate["contract_violation_case_rate"] == 1 / 3
    assert "mechanical_integrity" in aggregate["mean_quality_dimensions"]
    assert "contract_fulfillment" in aggregate["mean_quality_dimensions"]


def test_opportunity_rule_remains_strictly_proportional() -> None:
    reference = summary(1.0)
    candidate = summary(0.99, defect_rate=0.01, issue_cost=0.002)
    worthwhile = evaluate_opportunity(reference, candidate, resource_gain=0.03, resource_name="memory_reduction")
    not_worthwhile = evaluate_opportunity(reference, candidate, resource_gain=0.0001, resource_name="memory_reduction")
    assert worthwhile["eligible"] is True
    assert worthwhile["resource_gain_fraction"] > worthwhile["quality_opportunity_loss_fraction"]
    assert not_worthwhile["eligible"] is False


def test_no_forbidden_catch_all_label_is_ever_emitted_across_representative_cases() -> None:
    rows_and_replies = [
        (record(), "Did it improve?"),
        (record(), "I cannot verify real-time events without a sensor network."),
        (record(), "You gives. you gives. You can you is the duplicate."),
        (
            record(contract=requirement_contract("consent_requirement_missing", "Consent is required.", r"\bconsent\b")),
            "Publish it now.",
        ),
    ]
    for row, reply in rows_and_replies:
        result = assess(reply, row=row)
        assert not (labels(result) & FORBIDDEN_CATCH_ALL_LABELS), result


def test_undocumented_scored_contract_label_is_rejected() -> None:
    row = record(
        contract=requirement_contract(
            "made_up_subjective_label",
            "This deliberately undocumented label must not enter official scoring.",
            r"\bnever-match-this\b",
        )
    )
    import pytest

    with pytest.raises(ValueError, match="undocumented scored label"):
        assess("Ordinary reply.", row=row)


def test_category_is_strictly_reporting_only() -> None:
    base = record(
        category="acknowledgment",
        prompt="So, I backed up the checkpoint.",
        reference="Good call.",
    )
    reply = "Nice—good call."
    variants = []
    for category in ("acknowledgment", "safety_boundary", "totally_wrong_tag", None):
        row = dict(base)
        if category is None:
            row.pop("category", None)
        else:
            row["category"] = category
        variants.append(assess(reply, row=row))
    fields = (
        "quality_score", "quality_dimensions", "quality_weights", "quality_defects",
        "critical_failures", "major_issues", "minor_issues",
        "contract_fulfillment_score", "prompt_function_alignment_score",
    )
    for candidate in variants[1:]:
        for field in fields:
            assert candidate[field] == variants[0][field], field
        assert candidate["category_scoring_used"] is False
        assert candidate["category_contract_inference_used"] is False


def test_legacy_expected_modes_do_not_limit_valid_responses() -> None:
    base = record(prompt="So, I backed up the checkpoint.")
    constrained = dict(base)
    constrained["response_contract"] = dict(base["response_contract"])
    constrained["response_contract"]["expected_modes"] = ["privacy_assurance"]
    plain = assess("Did it save correctly?", row=base)
    legacy = assess("Did it save correctly?", row=constrained)
    assert legacy["quality_score"] == plain["quality_score"]
    assert legacy["quality_dimensions"] == plain["quality_dimensions"]
    assert any(item["label"] == "legacy_expected_modes_ignored" for item in legacy["review_diagnostics"])


def test_real_decoding_corruption_is_category_independent_zero_quality() -> None:
    corrupted = "议9999999\n99\n1\n1\national fromihilation或者:normal\"\nsteel\napatkan tray with\nbí"
    for category in ("acknowledgment", "factual_compression", "safety_boundary", None):
        row = record(category=category or "temporary")
        if category is None:
            row.pop("category", None)
        result = assess(corrupted, row=row, tokens=30)
        assert result["quality_score"] == 0.0
        assert "compound_decoding_corruption" in result["critical_failures"]
