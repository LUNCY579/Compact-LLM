from __future__ import annotations

"""Compatibility facade for the relative-opportunity policy.

The old repository routed absolute floors through this module. Evaluator v4
keeps the public function names so existing phase scripts and user commands do
not break, but all numerical floor/ceiling keyword arguments are deprecated and
ignored. Official decisions must combine the returned relative quality loss
with a phase-specific proportional resource benefit.
"""

from typing import Any

from scripts.common.opportunity_evaluation import (
    OPPORTUNITY_POLICY_VERSION,
    evaluate_opportunity,
    higher_is_better_loss,
    quality_opportunity_components,
    validate_technical_comparability,
)
from scripts.common.quality_evaluation import require_current_quality, summary_quality_score


def quality_index(summary: dict[str, Any]) -> float:
    """Continuous descriptive quality index with no pass-rate thresholds."""

    require_current_quality(summary)
    mean = summary_quality_score(summary)
    worst = float(summary.get("worst_decile_mean_quality_score", mean))
    dimensions = summary.get("mean_quality_dimensions") or summary.get("quality_dimension_means") or {}
    comprehensibility = float(dimensions.get("comprehensibility", mean))
    task = float(dimensions.get("task_adequacy", mean))
    logic = float(dimensions.get("logic_factual_consistency", mean))
    safety = float(dimensions.get("safety_boundary", mean))
    return 0.40 * mean + 0.20 * worst + 0.12 * comprehensibility + 0.10 * task + 0.10 * logic + 0.08 * safety


def relative_loss(reference: float, candidate: float) -> float:
    return float(higher_is_better_loss(reference, candidate) or 0.0)


def absolute_loss(reference: float, candidate: float) -> float:
    return max(0.0, float(reference) - float(candidate))


def category_metric_losses(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    field: str,
) -> tuple[float, dict[str, float]]:
    ref_values = {str(k): float(v) for k, v in (reference.get(field) or {}).items() if v is not None}
    cand_values = {str(k): float(v) for k, v in (candidate.get(field) or {}).items() if v is not None}
    losses: dict[str, float] = {}
    for category, ref_value in ref_values.items():
        candidate_value = cand_values.get(category)
        losses[category] = 1.0 if candidate_value is None else relative_loss(ref_value, candidate_value)
    return (max(losses.values()) if losses else 0.0), losses


def compare_quality_summaries(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    **deprecated_thresholds: Any,
) -> dict[str, Any]:
    """Return relative continuous comparison; absolute threshold args are ignored."""

    technical_reasons = validate_technical_comparability(reference, candidate)
    components = quality_opportunity_components(reference, candidate) if not technical_reasons else {
        "combined_quality_opportunity_loss": float("inf"),
        "combined_quality_opportunity_gain": 0.0,
        "net_quality_change": float("-inf"),
        "quality_dimension_losses": {},
        "quality_dimension_gains": {},
        "category_quality_relative_losses": {},
        "maximum_category_relative_loss": float("inf"),
        "quality_profile": "balanced",
        "opportunity_policy_version": OPPORTUNITY_POLICY_VERSION,
    }
    mean_loss = relative_loss(summary_quality_score(reference), summary_quality_score(candidate)) if not technical_reasons else float("inf")
    ref_worst = float(reference.get("worst_decile_mean_quality_score", summary_quality_score(reference)))
    cand_worst = float(candidate.get("worst_decile_mean_quality_score", summary_quality_score(candidate)))
    worst_loss = relative_loss(ref_worst, cand_worst)
    observable_pass_loss = relative_loss(
        float(reference.get("observable_quality_pass_rate", reference.get("semantic_pass_rate", 1.0))),
        float(candidate.get("observable_quality_pass_rate", candidate.get("semantic_pass_rate", 1.0))),
    )
    alignment_loss = relative_loss(
        float(reference.get("mean_reference_alignment_score", 1.0) or 1.0),
        float(candidate.get("mean_reference_alignment_score", 1.0) or 1.0),
    )
    category_quality_losses = dict(components.get("category_quality_relative_losses", {}))
    category_observable_loss, category_observable_losses = category_metric_losses(
        reference, candidate, "category_observable_quality_pass_rates"
    )
    if not category_observable_losses:
        category_observable_loss, category_observable_losses = category_metric_losses(
            reference, candidate, "category_semantic_pass_rates"
        )
    category_completion_loss, category_completion_losses = category_metric_losses(reference, candidate, "category_complete_utterance_rates")
    category_quality_pass_loss, category_quality_pass_losses = category_metric_losses(reference, candidate, "category_quality_pass_rates")
    category_alignment_loss, category_alignment_losses = category_metric_losses(reference, candidate, "category_reference_alignment_scores")
    return {
        "eligible": not technical_reasons,
        "reasons": technical_reasons,
        "technical_valid": not technical_reasons,
        "technical_invalidity_reasons": technical_reasons,
        "deprecated_threshold_arguments_ignored": sorted(deprecated_thresholds),
        "reference_mean_quality_score": summary_quality_score(reference),
        "candidate_mean_quality_score": summary_quality_score(candidate),
        "mean_quality_loss": mean_loss,
        "reference_quality_index": quality_index(reference),
        "candidate_quality_index": quality_index(candidate),
        "quality_index_loss": relative_loss(quality_index(reference), quality_index(candidate)),
        "worst_decile_quality_loss": worst_loss,
        "maximum_category_quality_loss": float(components.get("maximum_category_relative_loss", 0.0)),
        "category_quality_losses": category_quality_losses,
        "maximum_category_observable_quality_pass_loss": category_observable_loss,
        "category_observable_quality_pass_losses": category_observable_losses,
        "maximum_category_completion_loss": category_completion_loss,
        "category_completion_losses": category_completion_losses,
        "maximum_category_quality_pass_loss": category_quality_pass_loss,
        "category_quality_pass_losses": category_quality_pass_losses,
        "maximum_category_reference_alignment_loss": category_alignment_loss,
        "category_reference_alignment_losses": category_alignment_losses,
        "observable_quality_pass_loss": observable_pass_loss,
        "reference_alignment_loss": alignment_loss,
        "quality_pass_rate": float(candidate.get("quality_pass_rate", 0.0)),
        "intelligibility_pass_rate": float(candidate.get("intelligibility_pass_rate", 0.0)),
        "observable_quality_pass_rate": float(candidate.get("observable_quality_pass_rate", candidate.get("semantic_pass_rate", 0.0))),
        "complete_utterance_rate": float(candidate.get("complete_utterance_rate", 0.0)),
        "hard_failure_case_count": int(candidate.get("hard_failure_case_count", 0)),
        "critical_failure_count": int(candidate.get("critical_failure_count", 0)),
        "major_case_count": int(candidate.get("major_case_count", 0)),
        "combined_quality_opportunity_loss": float(components["combined_quality_opportunity_loss"]),
        "combined_quality_opportunity_gain": float(components["combined_quality_opportunity_gain"]),
        "net_quality_change": float(components["net_quality_change"]),
        "quality_dimension_losses": components.get("quality_dimension_losses", {}),
        "quality_dimension_gains": components.get("quality_dimension_gains", {}),
        "policy_note": "No absolute threshold controls eligibility; combine this loss with the phase primary resource gain.",
    }


def assert_valid_reference(summary: dict[str, Any], *, label: str = "reference") -> None:
    """Require only valid, reproducible current-schema evidence.

    Low quality, issue counts, or imperfect language do not invalidate a
    reference. Missing/non-finite evidence still does.
    """

    require_current_quality(summary, f"{label} summary")


__all__ = [
    "assert_valid_reference",
    "compare_quality_summaries",
    "evaluate_opportunity",
    "quality_index",
    "relative_loss",
    "absolute_loss",
    "category_metric_losses",
    "summary_quality_score",
]
