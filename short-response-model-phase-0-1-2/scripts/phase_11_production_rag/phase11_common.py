from __future__ import annotations

"""Dependency-light production RAG policy and calibration helpers."""

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable

from scripts.phase_9_student_distillation.phase9_common import (
    ABSTENTION_TEXT,
    CONFLICT_TEXT,
    GroundingResult,
    detect_structured_conflict,
    grounding_check,
    normalized_tokens,
    route_mode,
)

PHASE_11_SCHEMA_VERSION = "1.0"

_HIGH_RISK_PATTERNS = (
    re.compile(r"\b(?:dose|dosage|overdose|suicide|self[- ]?harm|kill myself)\b", re.I),
    re.compile(r"\b(?:legal advice|lawsuit|arrest|criminal charge)\b", re.I),
    re.compile(r"\b(?:invest all|guaranteed return|tax evasion)\b", re.I),
)
_COMPLEX_PATTERNS = (
    re.compile(r"\b(?:write|build|design|implement|analyze|compare)\b.*\b(?:complete|comprehensive|detailed|step[- ]by[- ]step)\b", re.I),
    re.compile(r"\b(?:prove|derive|debug)\b", re.I),
)
_PROPER_NOUN_RE = re.compile(r"(?<![.!?]\s)\b[A-Z][A-Za-z0-9_-]{2,}\b")
_ENTITY_STOPWORDS = {"the", "this", "that", "these", "those", "it", "its", "based", "available", "version"}
_NEGATION_RE = re.compile(r"\b(?:not|never|no|cannot|can't|isn't|wasn't|won't|without)\b", re.I)


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    reason: str
    requires_fallback: bool = False


@dataclass(frozen=True)
class AnswerabilityFeatures:
    best_retrieval_score: float
    question_coverage: float
    evidence_count_score: float
    conflict_detected: bool


@dataclass(frozen=True)
class AnswerabilityDecision:
    should_answer: bool
    reason: str
    score: float
    threshold: float
    features: AnswerabilityFeatures
    conflict_detected: bool = False


@dataclass(frozen=True)
class ProductionGroundingResult:
    supported: bool
    hard_failures: tuple[str, ...]
    diagnostics: tuple[str, ...]
    base: GroundingResult


def route_request(question: str, explicit_mode: str | None = None) -> RouteDecision:
    if explicit_mode:
        normalized = explicit_mode.strip().lower()
        aliases = {
            "high-risk": "high_risk",
            "highrisk": "high_risk",
            "complex_task": "complex",
            "factual": "grounded_factual",
            "social": "social_roleplay",
            "roleplay": "social_roleplay",
            "task": "procedure",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {"high_risk", "complex"}:
            return RouteDecision(normalized, "explicit_mode", requires_fallback=True)
        return RouteDecision(route_mode(question, normalized), "explicit_mode")
    for pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(question or ""):
            return RouteDecision("high_risk", f"matched:{pattern.pattern}", requires_fallback=True)
    for pattern in _COMPLEX_PATTERNS:
        if pattern.search(question or ""):
            return RouteDecision("complex", f"matched:{pattern.pattern}", requires_fallback=True)
    return RouteDecision(route_mode(question), "phase9_auditable_router")


def passage_id(passage: dict[str, Any]) -> str:
    return str(passage.get("record_id") or passage.get("document_id") or passage.get("procedure_id") or "")


def question_coverage(question: str, passages: Iterable[dict[str, Any]]) -> float:
    question_tokens = set(normalized_tokens(question))
    if not question_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for passage in passages:
        evidence_tokens.update(normalized_tokens(str(passage.get("text") or "")))
    return len(question_tokens & evidence_tokens) / len(question_tokens)


def answerability_features(question: str, passages: list[dict[str, Any]]) -> AnswerabilityFeatures:
    best = max((float(item.get("normalized_score", 0.0)) for item in passages), default=0.0)
    coverage = question_coverage(question, passages)
    count_score = min(1.0, len(passages) / 3.0)
    return AnswerabilityFeatures(
        best_retrieval_score=max(0.0, min(1.0, best)),
        question_coverage=max(0.0, min(1.0, coverage)),
        evidence_count_score=count_score,
        conflict_detected=detect_structured_conflict(passages),
    )


def score_answerability(features: AnswerabilityFeatures, weights: dict[str, float]) -> float:
    values = {
        "best_retrieval_score": features.best_retrieval_score,
        "question_coverage": features.question_coverage,
        "evidence_count_score": features.evidence_count_score,
    }
    total_weight = sum(max(0.0, float(weights.get(name, 0.0))) for name in values)
    if total_weight <= 0:
        raise ValueError("Answerability weights must contain positive mass")
    return sum(max(0.0, float(weights.get(name, 0.0))) * value for name, value in values.items()) / total_weight


def answerability_gate(
    question: str,
    passages: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> AnswerabilityDecision:
    features = answerability_features(question, passages)
    if not passages:
        return AnswerabilityDecision(False, "no_retrieval_results", 0.0, float(calibration["threshold"]), features)
    if features.conflict_detected:
        return AnswerabilityDecision(False, "conflicting_evidence", 0.0, float(calibration["threshold"]), features, True)
    weights = dict(calibration.get("weights") or {})
    threshold = float(calibration["threshold"])
    score = score_answerability(features, weights)
    if score < threshold:
        return AnswerabilityDecision(False, "answerability_score_below_calibrated_threshold", score, threshold, features)
    return AnswerabilityDecision(True, "answerability_gate_passed", score, threshold, features)


def threshold_metrics(rows: list[dict[str, Any]], threshold: float, *, false_answer_cost: float, false_abstention_cost: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = float(row["score"]) >= threshold
        answerable = bool(row["answerable"])
        if predicted and answerable:
            tp += 1
        elif predicted and not answerable:
            fp += 1
        elif not predicted and not answerable:
            tn += 1
        else:
            fn += 1
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "answer_precision": tp / max(1, tp + fp),
        "supported_recall": tp / max(1, tp + fn),
        "unsupported_rejection": tn / max(1, tn + fp),
        "false_answer_rate": fp / max(1, fp + tn),
        "unnecessary_abstention_rate": fn / max(1, fn + tp),
        "weighted_cost": (false_answer_cost * fp + false_abstention_cost * fn) / max(1, len(rows)),
    }


def calibrate_answerability(
    feature_rows: list[dict[str, Any]],
    *,
    weight_candidates: list[dict[str, float]],
    false_answer_cost: float,
    false_abstention_cost: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not feature_rows:
        raise ValueError("Cannot calibrate answerability on an empty development set")
    all_packets: list[dict[str, Any]] = []
    for weights in weight_candidates:
        scored = [
            {
                **row,
                "score": (
                    0.0
                    if bool((row.get("features") or {}).get("conflict_detected"))
                    else score_answerability(
                        AnswerabilityFeatures(**row["features"]),
                        weights,
                    )
                ),
            }
            for row in feature_rows
        ]
        thresholds = sorted({0.0, 1.0, *(float(row["score"]) for row in scored)})
        for threshold in thresholds:
            packet = threshold_metrics(
                scored,
                threshold,
                false_answer_cost=false_answer_cost,
                false_abstention_cost=false_abstention_cost,
            )
            packet["weights"] = weights
            all_packets.append(packet)
    best = min(
        all_packets,
        key=lambda item: (
            float(item["weighted_cost"]),
            int(item["false_positive"]),
            -float(item["supported_recall"]),
            -float(item["threshold"]),
        ),
    )
    return best, all_packets


def production_grounding_check(answer: str, evidence: list[dict[str, Any]]) -> ProductionGroundingResult:
    base = grounding_check(answer, evidence)
    evidence_text = " ".join(str(item.get("text") or "") for item in evidence)
    hard = list(base.hard_failures)
    hard.extend(f"unsupported_number:{value}" for value in base.unsupported_numbers)
    diagnostics = list(base.diagnostics)

    answer_entities = {
        value.casefold()
        for value in _PROPER_NOUN_RE.findall(answer or "")
        if value.casefold() not in _ENTITY_STOPWORDS
    }
    evidence_entities = {
        value.casefold()
        for value in _PROPER_NOUN_RE.findall(evidence_text)
        if value.casefold() not in _ENTITY_STOPWORDS
    }
    unsupported_entities = sorted(answer_entities - evidence_entities)
    if unsupported_entities:
        hard.append("unsupported_named_entities:" + ",".join(unsupported_entities))

    answer_negated = bool(_NEGATION_RE.search(answer or ""))
    evidence_negated = bool(_NEGATION_RE.search(evidence_text))
    if answer_negated != evidence_negated and base.lexical_support_fraction >= 0.5:
        diagnostics.append("possible_negation_or_polarity_mismatch")

    supported = not hard
    return ProductionGroundingResult(
        supported=supported,
        hard_failures=tuple(sorted(set(hard))),
        diagnostics=tuple(sorted(set(diagnostics))),
        base=base,
    )
