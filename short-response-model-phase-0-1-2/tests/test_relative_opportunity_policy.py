from __future__ import annotations

from scripts.common.audit_relative_opportunity_policy import audit
from scripts.common.opportunity_evaluation import evaluate_opportunity


def summary(score: float, *, issue_cost: float = 0.0) -> dict:
    dimensions = {
        "comprehensibility": score,
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
        "mean_heuristic_score": score,
        "worst_decile_mean_quality_score": score,
        "mean_quality_dimensions": dimensions,
        "category_quality_scores": {"general": score},
        "mean_issue_cost": issue_cost,
        "severe_issue_case_rate": 0.0,
    }


def test_static_policy_audit_has_no_findings() -> None:
    assert audit() == []


def test_profile_sensitivity_is_reported_for_all_profiles() -> None:
    result = evaluate_opportunity(
        summary(1.0), summary(0.99), resource_gain=0.02, resource_name="memory_reduction"
    )
    assert set(result["profile_sensitivity"]) == {
        "balanced", "logic_oriented", "conversational_permissive"
    }
    assert result["eligible_profile_count"] >= 1
