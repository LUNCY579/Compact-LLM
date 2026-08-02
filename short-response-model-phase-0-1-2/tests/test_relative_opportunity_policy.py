from __future__ import annotations

from scripts.common.audit_relative_opportunity_policy import audit
from scripts.common.opportunity_evaluation import evaluate_opportunity


def summary(score: float, *, issue_cost: float = 0.0) -> dict:
    dimensions = {
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
        "mean_heuristic_score": score,
        "worst_decile_mean_quality_score": score,
        "mean_quality_dimensions": dimensions,
        "category_quality_scores": {"general": score},
        "mean_issue_cost": issue_cost,
        "critical_defect_case_rate": 0.0,
        "severe_issue_case_rate": 0.0,
        "observable_defect_case_rate": 0.0,
        "contract_violation_case_rate": 0.0,
        "mechanical_failure_case_rate": 0.0,
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


def test_category_statistics_cannot_change_combined_loss_or_eligibility() -> None:
    reference = summary(1.0)
    candidate_a = summary(0.99)
    candidate_b = summary(0.99)
    candidate_a["category_quality_scores"] = {"accurate_tag": 0.99}
    candidate_b["category_quality_scores"] = {
        "wrong_tag": 0.01,
        "another_wrong_tag": 1.0,
    }
    first = evaluate_opportunity(reference, candidate_a, resource_gain=0.03, resource_name="memory_reduction")
    second = evaluate_opportunity(reference, candidate_b, resource_gain=0.03, resource_name="memory_reduction")
    assert first["quality_opportunity_loss_fraction"] == second["quality_opportunity_loss_fraction"]
    assert first["eligible"] == second["eligible"]
    assert first["category_statistics_contribute_to_combined_loss"] is False
    assert second["category_statistics_contribute_to_combined_loss"] is False
