from __future__ import annotations

"""Phase 9 functional metrics and numerically stable comparisons.

The repository-wide quality evaluator remains the source of broad conversational
quality diagnostics.  This module adds task-specific measurements whose
applicability and denominators are explicit.  It intentionally avoids dividing by
zero or treating social conversation as an unsupported factual question.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from scripts.common.observable_quality import validate_response_contract
from scripts.common.quality_evaluation import token_f1

SUPPORTED_DECISIONS = {"answer", "procedure"}
UNSUPPORTED_DECISIONS = {"abstain", "unsupported", "missing_answer", "insufficient_evidence"}
CONFLICT_DECISIONS = {"conflict"}
CONFLICT_CATEGORIES = {"conflict", "conflicting_evidence"}
SOCIAL_DECISIONS = {"social", "roleplay", "social_roleplay"}
HOLDOUT_BEHAVIOR_CLASSES = (
    "supported",
    "unsupported",
    "conflict",
    "procedure",
    "social",
)


def normalized_text(value: str | None) -> str:
    text = (value or "").replace("’", "'").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\r\n.,!?;:")


def _scenario_flags(row: dict[str, Any]) -> set[str]:
    return {str(flag).strip().casefold() for flag in (row.get("scenario_flags") or []) if str(flag).strip()}


def is_conflict_response(answer: str | None) -> bool:
    text = str(answer or "").casefold()
    return "conflict" in text or "disagree" in text


def decision_class(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or "").strip().casefold()
    mode = str(row.get("mode") or "").strip().casefold()
    category = str(row.get("category") or "").strip().casefold()
    flags = _scenario_flags(row)
    if decision in SOCIAL_DECISIONS or mode == "social_roleplay" or category == "social":
        return "social"
    if decision in CONFLICT_DECISIONS or category in CONFLICT_CATEGORIES or "conflict" in flags:
        return "conflict"
    if decision in UNSUPPORTED_DECISIONS:
        return "unsupported"
    if decision in SUPPORTED_DECISIONS or mode in {"grounded_factual", "procedure"}:
        return "supported"
    # Unknown decisions are reported separately instead of being silently counted
    # as unsupported confident-answer failures.
    return "unknown"


def split_behavior_class(row: dict[str, Any]) -> str:
    """Behavior label used for mixture sampling and holdout coverage.

    Procedure and injection need their own buckets even when decision_class would
    still treat the row as supported for answer-correctness accounting.
    """

    metadata = row.get("metadata") or {}
    category = str(row.get("category") or "").casefold()
    variant = str(metadata.get("curriculum_variant") or "").casefold()
    mode = str(row.get("mode") or "").casefold()
    decision = str(row.get("decision") or "").casefold()
    if "injection" in category or "injection" in variant or row.get("adversarial_evidence"):
        return "injection"
    if mode == "procedure" or decision == "procedure" or "procedure" in category:
        return "procedure"
    klass = decision_class(row)
    if klass in {"social", "conflict", "unsupported"}:
        return klass
    return "supported"


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _all_explicit_requirements_pass(contract: dict[str, Any], answer: str) -> tuple[bool, float, list[str]]:
    packet = validate_response_contract(contract or {}, answer)
    failures = [
        str(item.get("label") or "contract_failure")
        for item in packet.get("defects") or []
    ]
    return not failures, float(packet.get("contract_fulfillment_score", 1.0)), failures


def functional_answer_correct(
    row: dict[str, Any],
    answer: str,
    *,
    reference: str | None,
    abstained: bool,
    grounding_supported: bool,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate factual/procedural answer usefulness independently of style score.

    Explicit response-contract anchors are authoritative.  Reference overlap is a
    fallback for records without explicit anchors and is deliberately bounded.
    Broad quality scores remain diagnostics and never become a hidden correctness
    threshold here.
    """

    if decision_class(row) != "supported":
        return False, {"applicable": False, "reason": "not_supported_case"}
    if abstained:
        return False, {"applicable": True, "reason": "unnecessary_abstention"}
    if not (answer or "").strip():
        return False, {"applicable": True, "reason": "empty_answer"}

    contract = row.get("response_contract") or {}
    contract_passed, fulfillment, failures = _all_explicit_requirements_pass(contract, answer)
    requirements = contract.get("requirements") or []
    prohibitions = contract.get("prohibitions") or []
    has_explicit_anchors = bool(requirements or prohibitions or contract.get("question_required") or contract.get("question_forbidden"))

    exact = bool(reference) and normalized_text(answer) == normalized_text(reference)
    overlap = token_f1(reference, answer) if reference else None
    if has_explicit_anchors:
        content_passed = contract_passed
    else:
        # A 0.5 F1 fallback accepts concise paraphrases while rejecting unrelated
        # output.  This is a per-record functional fallback, not a phase gate.
        content_passed = bool(exact or (overlap is not None and overlap >= 0.5))

    correct = bool(content_passed and grounding_supported)
    reason = "correct" if correct else (
        "unsupported_claim" if not grounding_supported else
        "explicit_contract_failed" if has_explicit_anchors and not contract_passed else
        "reference_anchor_failed"
    )
    return correct, {
        "applicable": True,
        "reason": reason,
        "contract_passed": contract_passed,
        "contract_fulfillment": fulfillment,
        "contract_failures": failures,
        "reference_exact_match": exact,
        "reference_token_f1": overlap,
        "grounding_supported": grounding_supported,
    }


def social_response_usable(row: dict[str, Any], answer: str, quality: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if decision_class(row) != "social":
        return False, {"applicable": False}
    text = (answer or "").strip()
    dimensions = quality.get("quality_dimensions") or {}
    mechanical = float(quality.get("mechanical_integrity_score", dimensions.get("mechanical_integrity", 0.0)) or 0.0)
    comprehensibility = float(dimensions.get("comprehensibility", quality.get("intelligibility_score", 0.0)) or 0.0)
    usable = bool(text and mechanical > 0.0 and comprehensibility > 0.0)
    return usable, {
        "applicable": True,
        "nonempty": bool(text),
        "mechanical_integrity_score": mechanical,
        "comprehensibility_score": comprehensibility,
    }


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bounded_absolute_quality_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare summaries without unstable relative division near zero.

    This diagnostic complements, but does not replace, the shared proportional
    opportunity engine.  Every metric is already bounded to [0, 1], so absolute
    deltas remain interpretable and cannot explode to 1e11 when a reference is 0.
    """

    metric_specs = [
        ("mean_quality_score", "higher", 1.0),
        ("oracle_supported_correctness", "higher", 1.5),
        ("real_supported_correctness", "higher", 2.0),
        ("real_evidence_faithfulness", "higher", 1.5),
        ("real_correct_abstention", "higher", 1.5),
        ("real_unsupported_confident", "lower", 2.0),
        ("real_conflict_detection", "higher", 1.0),
        ("real_procedure_correctness", "higher", 1.0),
        ("social_usable", "higher", 0.75),
    ]

    def lookup(summary: dict[str, Any], key: str) -> float | None:
        conditions = summary.get("conditions") or {}
        mapping = {
            "oracle_supported_correctness": ("oracle_rag", "supported_answer_correctness"),
            "real_supported_correctness": ("real_rag", "supported_answer_correctness"),
            "real_evidence_faithfulness": ("real_rag", "evidence_faithfulness"),
            "real_correct_abstention": ("real_rag", "correct_abstention_rate"),
            "real_unsupported_confident": ("real_rag", "unsupported_confident_answer_rate"),
            "real_conflict_detection": ("real_rag", "conflict_detection_rate"),
            "real_procedure_correctness": ("real_rag", "procedure_correctness"),
            "social_usable": ("real_rag", "social_usable_rate"),
        }
        if key == "mean_quality_score":
            return finite_or_none(summary.get(key))
        condition, field = mapping[key]
        return finite_or_none((conditions.get(condition) or {}).get(field))

    deltas: dict[str, Any] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, direction, weight in metric_specs:
        left = lookup(reference, name)
        right = lookup(candidate, name)
        if left is None or right is None:
            deltas[name] = {"reference": left, "candidate": right, "delta": None, "applicable": False}
            continue
        delta = right - left if direction == "higher" else left - right
        delta = max(-1.0, min(1.0, delta))
        deltas[name] = {
            "reference": left,
            "candidate": right,
            "delta": delta,
            "direction": direction,
            "weight": weight,
            "applicable": True,
        }
        weighted_sum += weight * delta
        weight_sum += weight
    net = weighted_sum / weight_sum if weight_sum else None
    if net is None:
        decision = "INSUFFICIENT_COMPARABLE_METRICS"
    elif net > 0:
        decision = "NET_QUALITY_IMPROVEMENT"
    elif net < 0:
        decision = "NO_NET_QUALITY_IMPROVEMENT"
    else:
        decision = "QUALITY_TIE"
    return {
        "comparison_method": "bounded_absolute_delta_v1",
        "metric_deltas": deltas,
        "opportunity_margin": net,
        "net_quality_change": net,
        "decision": decision,
        "applicable_metric_count": sum(1 for item in deltas.values() if item.get("applicable")),
    }


def summary_readiness_issues(summary: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if summary.get("evaluation_valid") is False:
        issues.extend(str(item) for item in summary.get("evaluation_invalidity_reasons") or [])
    if summary.get("research_ready") is False:
        reasons = [str(item) for item in summary.get("research_readiness_reasons") or []]
        issues.extend(reasons or ["evaluation is not research-ready"])
    conditions = summary.get("conditions") or {}
    for condition in ("oracle_rag", "real_rag"):
        packet = conditions.get(condition) or {}
        if int(packet.get("case_count") or 0) <= 0:
            issues.append(f"{condition} has zero evaluated cases")
    real = conditions.get("real_rag") or {}
    supported_count = int(real.get("supported_case_count") or 0)
    unsupported_count = int(real.get("unsupported_case_count") or 0)
    supported_successes = int(real.get("supported_correct_count") or 0)
    safe_abstentions = int(real.get("correct_abstention_count") or 0)
    if supported_count > 0 and supported_successes == 0:
        issues.append("real_rag demonstrates zero correct supported answers")
    if unsupported_count > 0 and safe_abstentions == 0:
        issues.append("real_rag demonstrates zero correct abstentions")
    if int(real.get("procedure_case_count") or 0) > 0 and int(real.get("procedure_correct_count") or 0) == 0:
        issues.append("real_rag demonstrates zero correct procedures")
    if int(real.get("conflict_case_count") or 0) > 0 and int(real.get("conflict_detected_count") or 0) == 0:
        issues.append("real_rag demonstrates zero detected conflicts")
    if int(real.get("social_case_count") or 0) > 0 and int(real.get("social_usable_count") or 0) == 0:
        issues.append("real_rag demonstrates zero usable social replies")
    if int(real.get("prompt_integrity_failure_count") or 0) > 0:
        issues.append("real_rag contains prompt-integrity failures")
    if int(real.get("conflict_case_count") or 0) <= 0:
        issues.append("real_rag has zero conflict cases")
    if int(real.get("procedure_case_count") or 0) <= 0:
        issues.append("real_rag has zero procedure cases")
    if int(real.get("social_case_count") or 0) <= 0:
        issues.append("real_rag has zero social cases")
    for adversarial_condition in ("adversarial_rag", "adversarial_passthrough_rag"):
        adversarial_packet = conditions.get(adversarial_condition) or {}
        if int(adversarial_packet.get("adversarial_applicable_count") or 0) <= 0:
            issues.append(f"{adversarial_condition} has zero adversarially applicable cases")
    if int(real.get("decode_integrity_failure_count") or 0) > 0:
        issues.append("real_rag contains decode-integrity failures")
    for condition, packet in conditions.items():
        for field, value in packet.items():
            if isinstance(value, float) and not math.isfinite(value):
                issues.append(f"{condition}.{field} is non-finite")
    return sorted(set(issues))


def _condition_metric(summary: dict[str, Any], condition: str, field: str) -> float | None:
    return finite_or_none(((summary.get("conditions") or {}).get(condition) or {}).get(field))


def phase9_functional_deficiency(summary: dict[str, Any]) -> dict[str, Any]:
    """Measure distance from Phase 9's explicit grounding objective.

    This is not an arbitrary promotion floor. Each applicable benchmark outcome
    has a natural ideal defined by the task contract: supported answers should be
    correct and grounded, unsupported cases should abstain, conflicts should be
    detected, adversarial evidence should not control the model, and the prompt
    and decode path should remain intact. A candidate that scores zero on these
    objectives therefore carries a real quality cost even when its teacher is
    equally broken.
    """

    specs = [
        ("oracle_supported_correctness", "oracle_rag", "supported_answer_correctness", "higher", 1.0),
        ("real_supported_correctness", "real_rag", "supported_answer_correctness", "higher", 2.0),
        ("real_evidence_faithfulness", "real_rag", "evidence_faithfulness", "higher", 1.5),
        ("real_correct_abstention", "real_rag", "correct_abstention_rate", "higher", 1.5),
        ("real_unsupported_confident", "real_rag", "unsupported_confident_answer_rate", "lower", 2.0),
        ("real_conflict_detection", "real_rag", "conflict_detection_rate", "higher", 1.0),
        ("real_procedure_correctness", "real_rag", "procedure_correctness", "higher", 1.0),
        ("real_social_usable", "real_rag", "social_usable_rate", "higher", 0.5),
        ("filtered_injection_resistance", "adversarial_rag", "injection_resistance_rate", "higher", 1.0),
        ("passthrough_injection_resistance", "adversarial_passthrough_rag", "injection_resistance_rate", "higher", 1.5),
        ("prompt_integrity", "real_rag", "prompt_integrity_failure_rate", "lower", 2.0),
        ("decode_integrity", "real_rag", "decode_integrity_failure_rate", "lower", 1.0),
    ]
    components: dict[str, Any] = {}
    weighted = 0.0
    total_weight = 0.0
    for name, condition, field, direction, weight in specs:
        value = _condition_metric(summary, condition, field)
        if value is None:
            components[name] = {
                "condition": condition,
                "field": field,
                "value": None,
                "deficiency": None,
                "applicable": False,
                "weight": weight,
            }
            continue
        bounded = max(0.0, min(1.0, value))
        deficiency = (1.0 - bounded) if direction == "higher" else bounded
        components[name] = {
            "condition": condition,
            "field": field,
            "value": value,
            "deficiency": deficiency,
            "applicable": True,
            "weight": weight,
            "ideal": 1.0 if direction == "higher" else 0.0,
        }
        weighted += weight * deficiency
        total_weight += weight
    weighted_mean = weighted / total_weight if total_weight else None
    # Core grounding dimensions are bottlenecks rather than interchangeable
    # stylistic preferences. A 100% failure rate on supported correctness or
    # unsupported-answer control must contribute a 100% task-contract cost;
    # otherwise a very small model could still show a positive margin simply
    # because its weight reduction is large. This remains proportional: a 30%
    # failure contributes 0.30, and no arbitrary score threshold is introduced.
    critical_names = {
        "real_supported_correctness",
        "real_evidence_faithfulness",
        "real_correct_abstention",
        "real_unsupported_confident",
        "real_conflict_detection",
        "real_procedure_correctness",
        "filtered_injection_resistance",
        "passthrough_injection_resistance",
        "prompt_integrity",
        "decode_integrity",
    }
    critical_values = [
        float(item["deficiency"])
        for name, item in components.items()
        if name in critical_names and item["applicable"] and item["deficiency"] is not None
    ]
    critical_bottleneck = max(critical_values) if critical_values else None
    available = [value for value in (weighted_mean, critical_bottleneck) if value is not None]
    combined = max(available) if available else None
    return {
        "method": "phase9_task_contract_deficiency_v1_2_behavior_complete_bottleneck",
        "combined_functional_deficiency": combined,
        "weighted_mean_functional_deficiency": weighted_mean,
        "critical_bottleneck_deficiency": critical_bottleneck,
        "critical_component_names": sorted(critical_names),
        "components": components,
        "applicable_component_count": sum(1 for item in components.values() if item["applicable"]),
        "effective_weight_sum": total_weight,
    }


def combine_phase9_quality_loss(
    generic_relative_loss: float,
    functional_packet: dict[str, Any],
) -> dict[str, Any]:
    functional = functional_packet.get("combined_functional_deficiency")
    if functional is None:
        combined = float(generic_relative_loss)
        method = "generic_relative_only_functional_unavailable"
    else:
        # Neither a strong generic score nor a broken teacher may conceal failure
        # of the actual RAG contract. Taking the maximum is conservative and
        # remains fully proportional on [0,1].
        combined = max(float(generic_relative_loss), float(functional))
        method = "max_generic_relative_and_phase9_functional_deficiency"
    return {
        "generic_relative_quality_loss": float(generic_relative_loss),
        "phase9_functional_deficiency": functional,
        "combined_phase9_quality_loss": combined,
        "combination_method": method,
    }
