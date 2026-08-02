from __future__ import annotations

"""Deterministic contract checks for Quality Evaluator v3 and opportunity policy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.opportunity_evaluation import evaluate_opportunity
from scripts.common.quality_evaluation import (
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)


def case(category: str = "acknowledgment") -> dict:
    return {
        "id": "contract",
        "category": category,
        "messages": [
            {"role": "user", "content": "I backed up the checkpoint."},
            {"role": "assistant", "content": "Good call."},
        ],
        "reference": "Good call.",
        "expected": {"min_tokens": 1, "max_tokens": 30},
    }


def assess(reply: str, row: dict | None = None) -> dict:
    row = row or case()
    return evaluate_response(
        row,
        reply,
        len(reply.split()),
        visible_model_tokens=len(reply.split()),
        config=load_quality_config(),
        corpus_profile=build_corpus_profile([row]),
    )


def summary(score: float, *, issue_cost: float = 0.0, severe_rate: float = 0.0) -> dict:
    dimensions = {
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
        "mean_quality_dimensions": dimensions,
        "category_quality_scores": {"general": score},
        "mean_issue_cost": issue_cost,
        "severe_issue_case_rate": severe_rate,
    }


def main() -> None:
    acceptable = [
        "Did it complete successfully?",
        "How did the metrics look?",
        "Did it succeed?",
        "How the other adapter looks?",
        "That layer you are targeting?",
        "Which sector you are referring to?",
        "Probably not tbh.",
        "Yeah, that sounds rough.",
        "Nope, I would leave it.",
    ]
    for reply in acceptable:
        result = assess(reply)
        assert not result["quality_hard_fail"], reply
        assert result["quality_score"] > 0.0, reply
        assert "unwanted_follow_up_question" not in result["major_issues"], reply

    boundary = case("knowledge_boundary")
    boundary["messages"][0]["content"] = "What happened in that sector?"
    result = assess("Which sector you are referring to?", boundary)
    assert "unsupported_certainty" not in result["major_issues"]

    collapsed = assess("Here here here here here here")
    corrupted = assess("� Tigo politistas impressionoon Sm-elect Tsunlook")
    assert collapsed["quality_score"] == 0.0 and collapsed["quality_hard_fail"]
    assert corrupted["quality_score"] == 0.0 and corrupted["quality_hard_fail"]

    semantic_failures = [
        ("I reduced the context window.", "How did latency change?", "You gives. you gives. You can you is the duplicate.", "contextual_follow_up"),
        ("Can I claim a speedup from theoretical FLOPs alone?", "No—measure wall-clock latency and memory.", "I: the is you's size, and you, the amount.", "safety_boundary"),
        ("Why is the output weird?", "Can you share one prompt and reply?", "I: you can you the works. I would you all-+ the work. First the work. You. The. you.", "clarification"),
    ]
    for index, (prompt, reference_text, reply, category) in enumerate(semantic_failures):
        row = {
            "id": f"semantic-failure-{index}",
            "category": category,
            "messages": [{"role": "user", "content": prompt}],
            "reference": reference_text,
            "response_contract": {"open_ended": category != "safety_boundary", "min_tokens": 1, "max_tokens": 32},
        }
        result = assess(reply, row)
        assert result["quality_score"] < 0.40, (reply, result)
        assert result["semantic_severity"] in {"major", "critical"}, (reply, result)

    reference = summary(1.0)
    candidate = summary(0.99, issue_cost=0.002)
    worthwhile = evaluate_opportunity(
        reference, candidate, resource_gain=0.0264,
        resource_name="loaded_weight_tensor_memory_reduction",
    )
    not_worthwhile = evaluate_opportunity(
        reference, candidate, resource_gain=0.0001,
        resource_name="loaded_weight_tensor_memory_reduction",
    )
    assert worthwhile["eligible"]
    assert not not_worthwhile["eligible"]
    print("Quality Evaluator v3.1 contract passed: permissive language and relative opportunity policy verified.")


if __name__ == "__main__":
    main()
