from __future__ import annotations

from rineya_soup_lab.vendor.official_v45.quality_evaluation import (
    QUALITY_EVALUATION_VERSION,
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


def _record(*, contract: dict | None = None) -> dict:
    return {
        "id": "case",
        "category": "conversation",
        "messages": [{"role": "user", "content": "I backed up the checkpoint."}],
        "reference": "Good call.",
        "response_contract": contract
        or {
            "schema_version": "4.5",
            "open_ended": True,
            "min_tokens": 1,
            "max_tokens": 32,
            "expected_language": "en",
            "requirements": [],
            "prohibitions": [],
        },
    }


def _assess(text: str, *, row: dict | None = None, tokens: int | None = None) -> dict:
    selected = row or _record()
    count = tokens if tokens is not None else max(1, len(text.split()))
    return evaluate_response(
        selected,
        text,
        count,
        visible_model_tokens=count,
        config=load_quality_config(),
        corpus_profile=build_corpus_profile([selected]),
    )


def _labels(result: dict) -> set[str]:
    return {
        *(result.get("critical_failures") or []),
        *(result.get("major_issues") or []),
        *(result.get("minor_issues") or []),
    }


def test_bundled_quality_config_is_v45() -> None:
    config = load_quality_config()
    assert QUALITY_EVALUATION_VERSION == "4.5"
    assert config.raw["schema_version"] == "4.5"
    assert config.issue_taxonomy["schema_version"] == "4.5"


def test_repetition_and_encoding_collapse_are_critical() -> None:
    repeated = _assess("Here here here here here here here", tokens=7)
    corrupted = _assess("� Tigo politistas impressionoon Sm-elect Tsunlook", tokens=12)
    assert repeated["quality_score"] == 0.0
    assert "repeated_token_loop" in repeated["critical_failures"]
    assert corrupted["quality_score"] == 0.0
    assert "encoding_replacement_character" in corrupted["critical_failures"]


def test_informal_followups_and_alternate_social_replies_are_permitted() -> None:
    replies = [
        "Did it complete successfully?",
        "How the other adapter looks?",
        "That layer you are targeting?",
        "Probably not tbh.",
        "Yeah, that sounds rough.",
        "Nope, I would leave it.",
        "Perfect.",
    ]
    for reply in replies:
        result = _assess(reply)
        assert result["quality_hard_fail"] is False, (reply, result)
        assert not result["major_issues"], (reply, result)
        assert not (_labels(result) & FORBIDDEN_CATCH_ALL_LABELS), (reply, result)


def test_question_is_restricted_only_by_explicit_contract() -> None:
    open_result = _assess("Did it work?")
    strict = _record(
        contract={
            "schema_version": "4.5",
            "open_ended": False,
            "min_tokens": 1,
            "max_tokens": 20,
            "question_forbidden": True,
            "question_forbidden_label": "forbidden_question_present",
            "requirements": [],
            "prohibitions": [],
        }
    )
    strict_result = _assess("Did it work?", row=strict)
    assert "forbidden_question_present" not in _labels(open_result)
    assert "forbidden_question_present" in strict_result["major_issues"]


def test_internal_reasoning_leakage_has_specific_label() -> None:
    result = _assess("The user is asking which version, let me think.")
    assert "internal_reasoning_leakage" in result["major_issues"]
    assert result["case_issue_cost"] > 0.0


def test_aggregate_contract_matches_v45_schema() -> None:
    first = _assess("Good call.")
    second = _assess("Did it work?")
    first.update({"id": "a", "category": "conversation", "reference_token_f1": 1.0})
    second.update({"id": "b", "category": "conversation", "reference_token_f1": 0.0})
    summary = aggregate_quality_rows([first, second])
    assert summary["quality_evaluation_version"] == "4.5"
    assert summary["case_count"] == 2
    assert "mean_quality_dimensions" in summary
    assert "worst_decile_mean_quality_score" in summary
