from __future__ import annotations

"""Shared quality-gate helpers used across Phases 1-8 and playground.

All official gates require evaluator schema v2.0. Existing summaries created by
the legacy three-check scorer are intentionally rejected until re-evaluated;
otherwise stale, overly permissive evidence could silently pass later phases.
"""

from typing import Any

from scripts.common.quality_evaluation import (
    max_category_quality_loss,
    require_quality_v2,
    summary_quality_score,
)


def quality_index(summary: dict[str, Any]) -> float:
    """Tail-aware aggregate used only for ranking; hard gates remain separate."""

    require_quality_v2(summary)
    mean = summary_quality_score(summary)
    worst = float(summary.get("worst_decile_mean_quality_score", mean))
    pass_rate = float(summary.get("quality_pass_rate", 0.0))
    semantic = float(summary.get("semantic_pass_rate", 0.0))
    alignment = summary.get("mean_reference_alignment_score")
    reference = mean if alignment is None else float(alignment)
    return 0.45 * mean + 0.20 * worst + 0.15 * pass_rate + 0.10 * semantic + 0.10 * reference


def relative_loss(reference: float, candidate: float) -> float:
    return max(0.0, reference - candidate) / max(abs(reference), 1e-12)


def absolute_loss(reference: float, candidate: float) -> float:
    return max(0.0, reference - candidate)


def category_metric_losses(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    field: str,
) -> tuple[float, dict[str, float]]:
    """Measure category-specific degradation for an independent metric.

    A mean score can hide one damaged behavior class. Missing candidate categories
    are therefore treated as complete loss rather than silently ignored.
    """

    ref_values = {str(k): float(v) for k, v in (reference.get(field) or {}).items() if v is not None}
    cand_values = {str(k): float(v) for k, v in (candidate.get(field) or {}).items() if v is not None}
    losses: dict[str, float] = {}
    for category, ref_value in ref_values.items():
        candidate_value = cand_values.get(category)
        losses[category] = 1.0 if candidate_value is None else max(0.0, ref_value - candidate_value)
    return (max(losses.values()) if losses else 0.0), losses


def compare_quality_summaries(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_mean_quality_loss: float,
    max_quality_index_loss: float | None = None,
    max_worst_decile_loss: float = 0.05,
    max_category_quality_loss_allowed: float = 0.08,
    max_category_semantic_loss_allowed: float | None = None,
    max_category_completion_loss_allowed: float | None = None,
    max_category_quality_pass_loss_allowed: float | None = None,
    max_category_reference_alignment_loss_allowed: float | None = None,
    max_semantic_pass_loss: float = 0.02,
    max_reference_alignment_loss: float = 0.08,
    min_quality_pass_rate: float = 0.95,
    min_intelligibility_pass_rate: float = 1.0,
    min_semantic_pass_rate: float = 0.98,
    min_complete_utterance_rate: float = 0.99,
    max_hard_failure_cases: int = 0,
    max_critical_failures: int = 0,
    max_major_issue_cases: int | None = None,
) -> dict[str, Any]:
    """Compare a candidate against a reference with independent quality floors."""

    require_quality_v2(reference, "reference summary")
    require_quality_v2(candidate, "candidate summary")
    ref_mean = summary_quality_score(reference)
    cand_mean = summary_quality_score(candidate)
    mean_loss = max(0.0, ref_mean - cand_mean)
    ref_index = quality_index(reference)
    cand_index = quality_index(candidate)
    index_loss = max(0.0, ref_index - cand_index)
    ref_worst = float(reference.get("worst_decile_mean_quality_score", ref_mean))
    cand_worst = float(candidate.get("worst_decile_mean_quality_score", cand_mean))
    worst_loss = max(0.0, ref_worst - cand_worst)
    category_loss, category_losses = max_category_quality_loss(reference, candidate)
    category_semantic_loss, category_semantic_losses = category_metric_losses(
        reference, candidate, "category_semantic_pass_rates"
    )
    category_completion_loss, category_completion_losses = category_metric_losses(
        reference, candidate, "category_complete_utterance_rates"
    )
    category_quality_pass_loss, category_quality_pass_losses = category_metric_losses(
        reference, candidate, "category_quality_pass_rates"
    )
    category_alignment_loss, category_alignment_losses = category_metric_losses(
        reference, candidate, "category_reference_alignment_scores"
    )
    semantic_category_ceiling = (
        max_category_quality_loss_allowed
        if max_category_semantic_loss_allowed is None
        else max_category_semantic_loss_allowed
    )
    completion_category_ceiling = (
        max_category_quality_loss_allowed
        if max_category_completion_loss_allowed is None
        else max_category_completion_loss_allowed
    )
    pass_category_ceiling = (
        max_category_quality_loss_allowed
        if max_category_quality_pass_loss_allowed is None
        else max_category_quality_pass_loss_allowed
    )
    alignment_category_ceiling = (
        max_reference_alignment_loss
        if max_category_reference_alignment_loss_allowed is None
        else max_category_reference_alignment_loss_allowed
    )
    semantic_loss = max(
        0.0,
        float(reference.get("semantic_pass_rate", 0.0))
        - float(candidate.get("semantic_pass_rate", 0.0)),
    )
    ref_alignment = reference.get("mean_reference_alignment_score")
    cand_alignment = candidate.get("mean_reference_alignment_score")
    alignment_loss = (
        0.0
        if ref_alignment is None or cand_alignment is None
        else max(0.0, float(ref_alignment) - float(cand_alignment))
    )

    reasons: list[str] = []
    if int(candidate.get("hard_failure_case_count", 0)) > max_hard_failure_cases:
        reasons.append("hard-failure case count exceeds the zero-tolerance ceiling")
    if int(candidate.get("critical_failure_count", 0)) > max_critical_failures:
        reasons.append("critical failure count exceeds the configured ceiling")
    if max_major_issue_cases is not None and int(candidate.get("major_case_count", 0)) > max_major_issue_cases:
        reasons.append("major-issue case count exceeds the configured ceiling")
    if mean_loss > max_mean_quality_loss:
        reasons.append(f"mean quality loss {mean_loss:.4f} exceeds {max_mean_quality_loss:.4f}")
    if max_quality_index_loss is not None and index_loss > max_quality_index_loss:
        reasons.append(f"quality-index loss {index_loss:.4f} exceeds {max_quality_index_loss:.4f}")
    if worst_loss > max_worst_decile_loss:
        reasons.append(f"worst-decile quality loss {worst_loss:.4f} exceeds {max_worst_decile_loss:.4f}")
    if category_loss > max_category_quality_loss_allowed:
        reasons.append(
            f"worst category quality loss {category_loss:.4f} exceeds "
            f"{max_category_quality_loss_allowed:.4f}"
        )
    if category_semantic_loss > semantic_category_ceiling:
        reasons.append(
            f"worst category semantic-pass loss {category_semantic_loss:.4f} exceeds "
            f"{semantic_category_ceiling:.4f}"
        )
    if category_completion_loss > completion_category_ceiling:
        reasons.append(
            f"worst category completion loss {category_completion_loss:.4f} exceeds "
            f"{completion_category_ceiling:.4f}"
        )
    if category_quality_pass_loss > pass_category_ceiling:
        reasons.append(
            f"worst category quality-pass loss {category_quality_pass_loss:.4f} exceeds "
            f"{pass_category_ceiling:.4f}"
        )
    if category_alignment_loss > alignment_category_ceiling:
        reasons.append(
            f"worst category reference-alignment loss {category_alignment_loss:.4f} exceeds "
            f"{alignment_category_ceiling:.4f}"
        )
    if semantic_loss > max_semantic_pass_loss:
        reasons.append(f"semantic-pass loss {semantic_loss:.4f} exceeds {max_semantic_pass_loss:.4f}")
    if alignment_loss > max_reference_alignment_loss:
        reasons.append(
            f"reference-alignment loss {alignment_loss:.4f} exceeds "
            f"{max_reference_alignment_loss:.4f}"
        )
    if float(candidate.get("quality_pass_rate", 0.0)) < min_quality_pass_rate:
        reasons.append("quality-pass rate is below the configured floor")
    if float(candidate.get("intelligibility_pass_rate", 0.0)) < min_intelligibility_pass_rate:
        reasons.append("intelligibility-pass rate is below the zero-tolerance floor")
    if float(candidate.get("semantic_pass_rate", 0.0)) < min_semantic_pass_rate:
        reasons.append("semantic-pass rate is below the configured floor")
    if float(candidate.get("complete_utterance_rate", 0.0)) < min_complete_utterance_rate:
        reasons.append("complete-utterance rate is below the configured floor")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "reference_mean_quality_score": ref_mean,
        "candidate_mean_quality_score": cand_mean,
        "mean_quality_loss": mean_loss,
        "reference_quality_index": ref_index,
        "candidate_quality_index": cand_index,
        "quality_index_loss": index_loss,
        "worst_decile_quality_loss": worst_loss,
        "maximum_category_quality_loss": category_loss,
        "category_quality_losses": category_losses,
        "maximum_category_semantic_pass_loss": category_semantic_loss,
        "category_semantic_pass_losses": category_semantic_losses,
        "maximum_category_completion_loss": category_completion_loss,
        "category_completion_losses": category_completion_losses,
        "maximum_category_quality_pass_loss": category_quality_pass_loss,
        "category_quality_pass_losses": category_quality_pass_losses,
        "maximum_category_reference_alignment_loss": category_alignment_loss,
        "category_reference_alignment_losses": category_alignment_losses,
        "semantic_pass_loss": semantic_loss,
        "reference_alignment_loss": alignment_loss,
        "quality_pass_rate": float(candidate.get("quality_pass_rate", 0.0)),
        "intelligibility_pass_rate": float(candidate.get("intelligibility_pass_rate", 0.0)),
        "semantic_pass_rate": float(candidate.get("semantic_pass_rate", 0.0)),
        "complete_utterance_rate": float(candidate.get("complete_utterance_rate", 0.0)),
        "hard_failure_case_count": int(candidate.get("hard_failure_case_count", 0)),
        "critical_failure_count": int(candidate.get("critical_failure_count", 0)),
        "major_case_count": int(candidate.get("major_case_count", 0)),
    }


def assert_valid_reference(summary: dict[str, Any], *, label: str = "reference") -> None:
    """Abort experiments anchored to a collapsed or stale reference model."""

    require_quality_v2(summary, f"{label} summary")
    reasons: list[str] = []
    if int(summary.get("hard_failure_case_count", 0)) != 0:
        reasons.append("contains hard-failure outputs")
    if float(summary.get("intelligibility_pass_rate", 0.0)) < 1.0:
        reasons.append("does not have 100% intelligibility pass rate")
    if int(summary.get("critical_failure_count", 0)) != 0:
        reasons.append("contains critical failures")
    if reasons:
        raise ValueError(
            f"Invalid {label} evaluation: {'; '.join(reasons)}. Stop the experiment and "
            "repair model/tokenizer/checkpoint loading before comparing candidates."
        )
