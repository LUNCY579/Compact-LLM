from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import EvaluationError
from ..utils import get_nested


@dataclass(frozen=True)
class MetricContribution:
    path: str
    raw_value: float
    transformed_value: float
    normalized_weight: float
    contribution: float


@dataclass(frozen=True)
class ObjectiveResult:
    score: float
    contributions: tuple[MetricContribution, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "contributions": [item.__dict__ for item in self.contributions],
        }


def transform_metric(value: float, spec: Mapping[str, Any]) -> float:
    transform = str(spec.get("transform", "maximize_01"))
    if transform == "maximize_01":
        return max(0.0, min(1.0, value))
    if transform == "minimize_01":
        return max(0.0, min(1.0, 1.0 - value))
    if transform == "identity":
        return value
    if transform == "negative":
        return -value
    if transform == "target_range":
        minimum = float(spec.get("minimum", 0.0))
        maximum = float(spec.get("maximum", minimum))
        soft_upper = float(spec.get("soft_upper", max(maximum + 1.0, maximum * 2.0)))
        soft_lower = float(spec.get("soft_lower", 0.0))
        if minimum <= value <= maximum:
            return 1.0
        if value < minimum:
            denominator = max(1e-12, minimum - soft_lower)
            return max(0.0, 1.0 - (minimum - value) / denominator)
        denominator = max(1e-12, soft_upper - maximum)
        return max(0.0, 1.0 - (value - maximum) / denominator)
    if transform == "log_minimize":
        scale = float(spec.get("scale", 1.0))
        return 1.0 / (1.0 + math.log1p(max(0.0, value) / max(1e-12, scale)))
    raise EvaluationError(f"Unknown objective transform: {transform}")


def score_summary(summary: Mapping[str, Any], objective_config: Mapping[str, Any]) -> ObjectiveResult:
    specs = objective_config.get("metrics") or []
    weights = [max(0.0, float(spec.get("weight", 0.0))) for spec in specs]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise EvaluationError("Objective weights sum to zero.")
    contributions: list[MetricContribution] = []
    for spec, weight in zip(specs, weights):
        path = str(spec["path"])
        value = get_nested(summary, path)
        if value is None:
            raise EvaluationError(f"Summary lacks objective metric: {path}")
        try:
            raw = float(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"Objective metric {path} is not numeric: {value!r}") from exc
        if not math.isfinite(raw):
            raise EvaluationError(f"Objective metric {path} is non-finite: {raw}")
        normalized_weight = weight / total_weight
        transformed = transform_metric(raw, spec)
        contributions.append(
            MetricContribution(
                path=path,
                raw_value=raw,
                transformed_value=transformed,
                normalized_weight=normalized_weight,
                contribution=normalized_weight * transformed,
            )
        )
    return ObjectiveResult(
        score=sum(item.contribution for item in contributions),
        contributions=tuple(contributions),
    )


def brevity_utility(token_count: float, config: Mapping[str, Any]) -> float:
    minimum = float(config.get("brevity_target_min", 1))
    maximum = float(config.get("brevity_target_max", 12))
    soft_upper = float(config.get("brevity_soft_upper", 24))
    if minimum <= token_count <= maximum:
        return 1.0
    if token_count < minimum:
        return max(0.0, token_count / max(1.0, minimum))
    return max(0.0, 1.0 - (token_count - maximum) / max(1.0, soft_upper - maximum))


def case_utility(row: Mapping[str, Any], objective_config: Mapping[str, Any]) -> float:
    cfg = objective_config.get("case_utility") or {}
    quality = max(0.0, min(1.0, float(row.get("quality_score", 0.0))))
    complete = 1.0 if row.get("complete_utterance", row.get("complete", False)) else 0.0
    observable = 1.0 if row.get("observable_quality_pass", row.get("semantic_pass", False)) else 0.0
    tokens = float(row.get("visible_generated_tokens", row.get("generated_tokens", 0.0)))
    components = {
        "quality": (float(cfg.get("quality_weight", 0.72)), quality),
        "complete": (float(cfg.get("completeness_weight", 0.10)), complete),
        "observable": (float(cfg.get("observable_pass_weight", 0.08)), observable),
        "brevity": (float(cfg.get("brevity_weight", 0.10)), brevity_utility(tokens, cfg)),
    }
    total = sum(max(0.0, weight) for weight, _ in components.values())
    if total <= 0:
        return quality
    return sum(max(0.0, weight) * value for weight, value in components.values()) / total


def objective_summary_from_rows(
    rows: Sequence[Mapping[str, Any]],
    objective_config: Mapping[str, Any],
    *,
    duplicate_rate_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise EvaluationError("Cannot build an objective summary from zero rows.")
    qualities = [float(row.get("quality_score", 0.0)) for row in rows]
    tail_fraction = float(objective_config.get("tail_fraction", 0.10))
    tail_n = max(1, math.ceil(len(rows) * tail_fraction))
    ordered = sorted(qualities)
    complete = [
        1.0 if row.get("complete_utterance", row.get("complete", False)) else 0.0 for row in rows
    ]
    observable = [
        1.0 if row.get("observable_quality_pass", row.get("semantic_pass", False)) else 0.0
        for row in rows
    ]
    token_counts = [float(row.get("visible_generated_tokens", row.get("generated_tokens", 0.0))) for row in rows]
    # Bootstrap resampling intentionally repeats cases. Those repeated draws must not be
    # mistaken for model-level duplicate replies. Callers can therefore provide one copy
    # of each sampled case for this non-additive diagnostic while retaining multiplicity
    # for means, tails, and other bootstrap statistics.
    duplicate_rows = duplicate_rate_rows if duplicate_rate_rows is not None else rows
    normalized_replies = [str(row.get("reply", "")).strip().lower() for row in duplicate_rows]
    duplicates = len(normalized_replies) - len(set(normalized_replies))
    dimension_names = sorted(
        {
            str(key)
            for row in rows
            for key in (row.get("quality_dimensions") or {}).keys()
        }
    )
    dimensions: dict[str, float] = {}
    for name in dimension_names:
        values = [
            float((row.get("quality_dimensions") or {}).get(name, 0.0))
            for row in rows
        ]
        dimensions[name] = statistics.mean(values)
    return {
        "case_count": len(rows),
        "mean_quality_score": statistics.mean(qualities),
        "worst_decile_mean_quality_score": statistics.mean(ordered[:tail_n]),
        "p10_quality_score": ordered[tail_n - 1],
        "complete_utterance_rate": statistics.mean(complete),
        "observable_quality_pass_rate": statistics.mean(observable),
        "semantic_pass_rate": statistics.mean(observable),
        "median_reply_tokens": statistics.median(token_counts),
        "p90_reply_tokens": _percentile(token_counts, 0.90),
        "duplicate_reply_rate": duplicates / max(1, len(normalized_replies)),
        "mean_quality_dimensions": dimensions,
    }


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return float(ordered[index])


def align_rows(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    first_map = {str(row["id"]): row for row in first}
    second_map = {str(row["id"]): row for row in second}
    if set(first_map) != set(second_map):
        raise EvaluationError("Paired result rows have different stable IDs.")
    ids = [str(row["id"]) for row in first]
    return [first_map[value] for value in ids], [second_map[value] for value in ids]
