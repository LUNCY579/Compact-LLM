from __future__ import annotations

"""Relative opportunity-cost comparison shared by every official phase.

No absolute quality score, pass-rate, category floor, NLL ceiling, issue-count, or category-tail
veto appears here. A candidate is technically comparable first, then its
proportional primary resource benefit is compared with its proportional combined
quality opportunity loss. The strict decision is:

    eligible = resource_gain > quality_opportunity_loss

For phases whose purpose is pure quality improvement at unchanged inference
cost, use :func:`evaluate_quality_improvement` instead.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.common.quality_evaluation import require_current_quality, summary_quality_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "configs" / "relative_opportunity_evaluation.json"
OPPORTUNITY_POLICY_VERSION = "1.4"
EPSILON = 1e-12


@dataclass(frozen=True)
class OpportunityPolicy:
    raw: dict[str, Any]

    @property
    def default_profile(self) -> str:
        return str(self.raw.get("default_profile", "balanced"))

    def profile(self, name: str | None = None) -> dict[str, float]:
        selected = name or self.default_profile
        profiles = self.raw.get("profiles") or {}
        if selected not in profiles:
            raise KeyError(f"Unknown opportunity profile {selected!r}; available: {sorted(profiles)}")
        weights = {str(k): max(0.0, float(v)) for k, v in profiles[selected].items()}
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError(f"Opportunity profile {selected!r} has no positive weights.")
        return {key: value / total for key, value in weights.items()}


def load_opportunity_policy(path: Path | None = None) -> OpportunityPolicy:
    source = path or DEFAULT_POLICY_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != OPPORTUNITY_POLICY_VERSION:
        raise ValueError(
            f"Unsupported opportunity policy version {payload.get('schema_version')!r}; "
            f"expected {OPPORTUNITY_POLICY_VERSION!r}."
        )
    return OpportunityPolicy(payload)


def higher_is_better_loss(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    reference, candidate = float(reference), float(candidate)
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return None
    return max(0.0, reference - candidate) / max(abs(reference), EPSILON)


def higher_is_better_gain(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    reference, candidate = float(reference), float(candidate)
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return None
    return max(0.0, candidate - reference) / max(abs(reference), EPSILON)


def lower_is_better_loss(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    reference, candidate = float(reference), float(candidate)
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return None
    return max(0.0, candidate - reference) / max(abs(reference), EPSILON)


def bounded_rate_increase(reference: float | None, candidate: float | None) -> float | None:
    """Additive increase for naturally bounded [0,1] costs, safe when reference=0."""

    if reference is None or candidate is None:
        return None
    reference, candidate = float(reference), float(candidate)
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return None
    return max(0.0, candidate - reference)


def resource_reduction(reference_usage: float | int | None, candidate_usage: float | int | None) -> float | None:
    if reference_usage is None or candidate_usage is None:
        return None
    reference, candidate = float(reference_usage), float(candidate_usage)
    if not math.isfinite(reference) or not math.isfinite(candidate) or reference <= 0.0:
        return None
    return max(0.0, 1.0 - candidate / reference)


def relative_improvement(reference_cost: float | int | None, candidate_cost: float | int | None) -> float | None:
    """Alias for lower-is-better resource improvement."""

    return resource_reduction(reference_cost, candidate_cost)


def _dimension(summary: dict[str, Any], name: str) -> float | None:
    values = summary.get("mean_quality_dimensions") or summary.get("quality_dimension_means") or {}
    value = values.get(name)
    return None if value is None else float(value)


def _category_tail_loss(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float | None, dict[str, float]]:
    ref = reference.get("category_quality_scores") or reference.get("category_scores") or {}
    cand = candidate.get("category_quality_scores") or candidate.get("category_scores") or {}
    losses: dict[str, float] = {}
    for category, ref_value in ref.items():
        cand_value = cand.get(category)
        losses[str(category)] = 1.0 if cand_value is None else float(
            higher_is_better_loss(float(ref_value), float(cand_value)) or 0.0
        )
    return (max(losses.values()) if losses else None), losses


def quality_opportunity_components(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    profile: str | None = None,
    reference_target_nll: float | None = None,
    candidate_target_nll: float | None = None,
    policy: OpportunityPolicy | None = None,
) -> dict[str, Any]:
    """Compute normalized quality losses and gains without making a decision."""

    require_current_quality(reference, "reference quality summary")
    require_current_quality(candidate, "candidate quality summary")
    policy = policy or load_opportunity_policy()
    selected_profile = profile or policy.default_profile
    configured_weights = policy.profile(selected_profile)

    category_tail, category_losses = _category_tail_loss(reference, candidate)
    losses: dict[str, float | None] = {
        "mean_quality": higher_is_better_loss(summary_quality_score(reference), summary_quality_score(candidate)),
        "worst_decile_quality": higher_is_better_loss(
            reference.get("worst_decile_mean_quality_score"),
            candidate.get("worst_decile_mean_quality_score"),
        ),
        "comprehensibility": higher_is_better_loss(_dimension(reference, "comprehensibility"), _dimension(candidate, "comprehensibility")),
        "mechanical_integrity": higher_is_better_loss(_dimension(reference, "mechanical_integrity"), _dimension(candidate, "mechanical_integrity")),
        "contract_fulfillment": higher_is_better_loss(_dimension(reference, "contract_fulfillment"), _dimension(candidate, "contract_fulfillment")),
        "prompt_function_alignment": higher_is_better_loss(_dimension(reference, "prompt_function_alignment"), _dimension(candidate, "prompt_function_alignment")),
        "task_adequacy": higher_is_better_loss(_dimension(reference, "task_adequacy"), _dimension(candidate, "task_adequacy")),
        "factual_correctness": higher_is_better_loss(_dimension(reference, "factual_correctness"), _dimension(candidate, "factual_correctness")),
        "logic_factual_consistency": higher_is_better_loss(_dimension(reference, "logic_factual_consistency"), _dimension(candidate, "logic_factual_consistency")),
        "safety_boundary": higher_is_better_loss(_dimension(reference, "safety_boundary"), _dimension(candidate, "safety_boundary")),
        "context_consistency": higher_is_better_loss(_dimension(reference, "context_consistency"), _dimension(candidate, "context_consistency")),
        "response_completeness": higher_is_better_loss(_dimension(reference, "response_completeness"), _dimension(candidate, "response_completeness")),
        "brevity_appropriateness": higher_is_better_loss(_dimension(reference, "brevity_appropriateness"), _dimension(candidate, "brevity_appropriateness")),
        "conversation_naturalness": higher_is_better_loss(_dimension(reference, "conversation_naturalness"), _dimension(candidate, "conversation_naturalness")),
        "category_tail": category_tail,  # diagnostic-only; no v1.4 profile assigns weight
        "issue_cost": bounded_rate_increase(reference.get("mean_issue_cost", 0.0), candidate.get("mean_issue_cost", 0.0)),
        "critical_defect_rate": bounded_rate_increase(reference.get("critical_defect_case_rate", 0.0), candidate.get("critical_defect_case_rate", 0.0)),
        "observable_defect_rate": bounded_rate_increase(reference.get("observable_defect_case_rate", 0.0), candidate.get("observable_defect_case_rate", 0.0)),
        "contract_violation_rate": bounded_rate_increase(reference.get("contract_violation_case_rate", 0.0), candidate.get("contract_violation_case_rate", 0.0)),
        "mechanical_failure_rate": bounded_rate_increase(reference.get("mechanical_failure_case_rate", 0.0), candidate.get("mechanical_failure_case_rate", 0.0)),
        "target_nll": lower_is_better_loss(reference_target_nll, candidate_target_nll),
    }
    gains: dict[str, float | None] = {
        "mean_quality": higher_is_better_gain(summary_quality_score(reference), summary_quality_score(candidate)),
        "worst_decile_quality": higher_is_better_gain(
            reference.get("worst_decile_mean_quality_score"),
            candidate.get("worst_decile_mean_quality_score"),
        ),
    }
    for name in (
        "comprehensibility", "mechanical_integrity", "contract_fulfillment", "prompt_function_alignment",
        "task_adequacy", "factual_correctness", "logic_factual_consistency",
        "safety_boundary", "context_consistency", "response_completeness",
        "brevity_appropriateness", "conversation_naturalness",
    ):
        gains[name] = higher_is_better_gain(_dimension(reference, name), _dimension(candidate, name))

    available_weights = {
        key: configured_weights.get(key, 0.0)
        for key, value in losses.items()
        if value is not None and configured_weights.get(key, 0.0) > 0.0
    }
    total = sum(available_weights.values())
    normalized_weights = {
        key: value / total for key, value in available_weights.items()
    } if total > 0.0 else {}
    combined_loss = sum(normalized_weights[key] * float(losses[key] or 0.0) for key in normalized_weights)

    gain_weights = {
        key: configured_weights.get(key, 0.0)
        for key, value in gains.items()
        if value is not None and configured_weights.get(key, 0.0) > 0.0
    }
    gain_total = sum(gain_weights.values())
    normalized_gain_weights = {
        key: value / gain_total for key, value in gain_weights.items()
    } if gain_total > 0.0 else {}
    combined_gain = sum(normalized_gain_weights[key] * float(gains[key] or 0.0) for key in normalized_gain_weights)

    return {
        "opportunity_policy_version": OPPORTUNITY_POLICY_VERSION,
        "quality_profile": selected_profile,
        "quality_dimension_losses": {key: value for key, value in losses.items() if value is not None},
        "quality_dimension_gains": {key: value for key, value in gains.items() if value is not None},
        "effective_loss_weights": normalized_weights,
        "effective_gain_weights": normalized_gain_weights,
        "combined_quality_opportunity_loss": combined_loss,
        "combined_quality_opportunity_gain": combined_gain,
        "net_quality_change": combined_gain - combined_loss,
        "category_quality_relative_losses": category_losses,
        "maximum_category_relative_loss": category_tail or 0.0,
        "category_statistics_are_diagnostic_only": True,
        "category_statistics_contribute_to_combined_loss": False,
    }


def validate_technical_comparability(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for label, summary in (("reference", reference), ("candidate", candidate)):
        try:
            require_current_quality(summary, f"{label} summary")
        except (ValueError, TypeError) as exc:
            reasons.append(str(exc))
    if reference.get("case_count") and candidate.get("case_count") and int(reference["case_count"]) != int(candidate["case_count"]):
        reasons.append("reference and candidate case counts differ")
    return reasons


def _profile_sensitivity(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    resource_gain: float,
    reference_target_nll: float | None,
    candidate_target_nll: float | None,
    policy: OpportunityPolicy,
    quality_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate every configured profile for sensitivity reporting.

    Profile weights are not hidden thresholds. This packet exposes whether a
    ranking depends strongly on a particular weighting perspective.
    """

    packet: dict[str, dict[str, Any]] = {}
    for profile_name in sorted((policy.raw.get("profiles") or {}).keys()):
        components = quality_opportunity_components(
            reference,
            candidate,
            profile=profile_name,
            reference_target_nll=reference_target_nll,
            candidate_target_nll=candidate_target_nll,
            policy=policy,
        )
        loss = float(components["combined_quality_opportunity_loss"])
        if quality_only:
            gain = float(components["combined_quality_opportunity_gain"])
            margin = gain - loss
            decision = "NET_QUALITY_IMPROVEMENT" if margin > 0.0 else "NO_NET_QUALITY_IMPROVEMENT"
        else:
            gain = float(resource_gain)
            margin = gain - loss
            decision = "PROPORTIONALLY_WORTHWHILE" if margin > 0.0 else "NOT_PROPORTIONALLY_WORTHWHILE"
        packet[str(profile_name)] = {
            "resource_or_quality_gain_fraction": gain,
            "quality_opportunity_loss_fraction": loss,
            "opportunity_margin": margin,
            "eligible": margin > 0.0,
            "decision": decision,
        }
    return packet


def evaluate_opportunity(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    resource_gain: float | None,
    resource_name: str,
    profile: str | None = None,
    reference_target_nll: float | None = None,
    candidate_target_nll: float | None = None,
    policy: OpportunityPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or load_opportunity_policy()
    technical_reasons = validate_technical_comparability(reference, candidate)
    components = quality_opportunity_components(
        reference,
        candidate,
        profile=profile,
        reference_target_nll=reference_target_nll,
        candidate_target_nll=candidate_target_nll,
        policy=policy,
    ) if not technical_reasons else {
        "opportunity_policy_version": OPPORTUNITY_POLICY_VERSION,
        "quality_profile": profile or "balanced",
        "quality_dimension_losses": {},
        "quality_dimension_gains": {},
        "combined_quality_opportunity_loss": math.inf,
        "combined_quality_opportunity_gain": 0.0,
        "net_quality_change": -math.inf,
        "category_quality_relative_losses": {},
        "maximum_category_relative_loss": math.inf,
    }
    valid_resource = resource_gain is not None and math.isfinite(float(resource_gain)) and float(resource_gain) >= 0.0
    if not valid_resource:
        technical_reasons.append(f"missing or invalid primary resource gain: {resource_name}")
    normalized_gain = 0.0 if resource_gain is None else max(0.0, float(resource_gain))
    quality_loss = float(components["combined_quality_opportunity_loss"])
    margin = normalized_gain - quality_loss
    eligible = not technical_reasons and margin > 0.0
    sensitivity = (
        _profile_sensitivity(
            reference,
            candidate,
            resource_gain=normalized_gain,
            reference_target_nll=reference_target_nll,
            candidate_target_nll=candidate_target_nll,
            policy=policy,
        )
        if not technical_reasons else {}
    )
    return {
        **components,
        "primary_resource": resource_name,
        "resource_gain_fraction": normalized_gain,
        "quality_opportunity_loss_fraction": quality_loss,
        "opportunity_margin": margin,
        "eligible": eligible,
        "profile_sensitivity": sensitivity,
        "decision_robust_across_profiles": bool(sensitivity) and len({row["eligible"] for row in sensitivity.values()}) == 1,
        "eligible_profile_count": sum(1 for row in sensitivity.values() if row["eligible"]),
        "decision": "PROPORTIONALLY_WORTHWHILE" if eligible else "NOT_PROPORTIONALLY_WORTHWHILE",
        "technical_valid": not technical_reasons,
        "technical_invalidity_reasons": technical_reasons,
        "decision_explanation": (
            f"{resource_name} gain {normalized_gain:.6f} is greater than quality opportunity loss {quality_loss:.6f}"
            if eligible else
            f"{resource_name} gain {normalized_gain:.6f} is not greater than quality opportunity loss {quality_loss:.6f}"
        ),
    }


def evaluate_quality_improvement(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    profile: str | None = None,
    policy: OpportunityPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or load_opportunity_policy()
    technical_reasons = validate_technical_comparability(reference, candidate)
    components = quality_opportunity_components(reference, candidate, profile=profile, policy=policy) if not technical_reasons else {
        "opportunity_policy_version": OPPORTUNITY_POLICY_VERSION,
        "quality_profile": profile or "balanced",
        "quality_dimension_losses": {},
        "quality_dimension_gains": {},
        "combined_quality_opportunity_loss": math.inf,
        "combined_quality_opportunity_gain": 0.0,
        "net_quality_change": -math.inf,
        "category_quality_relative_losses": {},
        "maximum_category_relative_loss": math.inf,
    }
    net = float(components["net_quality_change"])
    eligible = not technical_reasons and net > 0.0
    sensitivity = (
        _profile_sensitivity(
            reference,
            candidate,
            resource_gain=0.0,
            reference_target_nll=None,
            candidate_target_nll=None,
            policy=policy,
            quality_only=True,
        )
        if not technical_reasons else {}
    )
    return {
        **components,
        "primary_resource": "net_relative_quality_improvement",
        "resource_gain_fraction": max(0.0, float(components["combined_quality_opportunity_gain"])),
        "quality_opportunity_loss_fraction": float(components["combined_quality_opportunity_loss"]),
        "opportunity_margin": net,
        "eligible": eligible,
        "profile_sensitivity": sensitivity,
        "decision_robust_across_profiles": bool(sensitivity) and len({row["eligible"] for row in sensitivity.values()}) == 1,
        "eligible_profile_count": sum(1 for row in sensitivity.values() if row["eligible"]),
        "decision": "NET_QUALITY_IMPROVEMENT" if eligible else "NO_NET_QUALITY_IMPROVEMENT",
        "technical_valid": not technical_reasons,
        "technical_invalidity_reasons": technical_reasons,
        "decision_explanation": (
            f"relative quality gains exceed relative quality losses by {net:.6f}"
            if eligible else f"relative quality gains do not exceed relative quality losses; net {net:.6f}"
        ),
    }


def pareto_frontier(rows: Iterable[dict[str, Any]], *, benefit_key: str = "resource_gain_fraction", loss_key: str = "quality_opportunity_loss_fraction") -> list[dict[str, Any]]:
    items = list(rows)
    frontier: list[dict[str, Any]] = []
    for row in items:
        benefit = float(row.get(benefit_key, 0.0))
        loss = float(row.get(loss_key, math.inf))
        dominated = False
        for other in items:
            if other is row:
                continue
            other_benefit = float(other.get(benefit_key, 0.0))
            other_loss = float(other.get(loss_key, math.inf))
            if other_benefit >= benefit and other_loss <= loss and (other_benefit > benefit or other_loss < loss):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier
