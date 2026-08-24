"""Pure analysis helpers for the exploratory Experiment-C quality screen."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

ATTENTION_PRIOR_QUALITY_CONDITION_IDS = (
    "C0_B3_NO_PRIOR",
    "C1_NORMALIZED_SC",
    "C5_RAW_S",
    "C4_SHUFFLED_SC",
    "C6_AGE_ONLY",
)


@dataclass(frozen=True, slots=True)
class AttentionPriorQualityCondition:
    condition_id: str
    source: Literal["normalized", "raw", "shuffled", "age"]
    lambda_value: float


def validate_attention_prior_conditions(
    rows: Sequence[Mapping[str, object]],
    *,
    frozen_lambda: float,
) -> tuple[AttentionPriorQualityCondition, ...]:
    if len(rows) != len(ATTENTION_PRIOR_QUALITY_CONDITION_IDS):
        raise ValueError("quality screen requires exactly five conditions")
    result: list[AttentionPriorQualityCondition] = []
    expected_sources = ("normalized", "normalized", "raw", "shuffled", "age")
    for index, (row, expected_id, expected_source) in enumerate(
        zip(
            rows,
            ATTENTION_PRIOR_QUALITY_CONDITION_IDS,
            expected_sources,
            strict=True,
        )
    ):
        condition_id = str(row.get("id", ""))
        source = str(row.get("source", ""))
        lambda_value = float(cast(float | int, row.get("lambda", float("nan"))))
        if condition_id != expected_id or source != expected_source:
            raise ValueError(f"condition {index} must be {expected_id}/{expected_source}")
        expected_lambda = 0.0 if index == 0 else float(frozen_lambda)
        if not math.isfinite(lambda_value) or lambda_value != expected_lambda:
            raise ValueError(
                f"condition {condition_id} must use frozen lambda {expected_lambda}"
            )
        result.append(
            AttentionPriorQualityCondition(
                condition_id=condition_id,
                source=cast(Literal["normalized", "raw", "shuffled", "age"], source),
                lambda_value=lambda_value,
            )
        )
    return tuple(result)


def expected_cell_order(
    case_ids: Sequence[str],
    condition_ids: Sequence[str] = ATTENTION_PRIOR_QUALITY_CONDITION_IDS,
) -> tuple[tuple[str, str], ...]:
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must be non-empty and unique")
    if not condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition IDs must be non-empty and unique")
    return tuple((str(case_id), str(condition)) for case_id in case_ids for condition in condition_ids)


def recover_prediction_prefix(
    rows: Sequence[Mapping[str, object]],
    expected_order: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    """Validate and return an exact durable prefix of the frozen cell order."""

    if len(rows) > len(expected_order):
        raise RuntimeError("partial checkpoint has more rows than the frozen plan")
    recovered: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        pair = (str(row.get("case_id", "")), str(row.get("condition", "")))
        if pair in seen:
            raise RuntimeError(f"partial checkpoint repeats cell {pair}")
        if pair != tuple(expected_order[index]):
            raise RuntimeError("partial prediction checkpoint is not an exact frozen-order prefix")
        seen.add(pair)
        recovered.append(dict(row))
    return recovered


def _jaccard_distance(left: set[int], right: set[int]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def retained_set_diagnostics(
    resident_positions: Sequence[Sequence[Sequence[int]]],
    *,
    input_tokens: int,
    previous_positions: Sequence[Sequence[Sequence[int]]] | None = None,
) -> dict[str, float | int | None]:
    """Summarize cross-head diversity, boundary churn, and position capture."""

    if input_tokens <= 0 or not resident_positions:
        raise ValueError("input_tokens and resident positions must be non-empty")
    layer_count = len(resident_positions)
    head_count = len(resident_positions[0])
    if head_count <= 0 or any(len(layer) != head_count for layer in resident_positions):
        raise ValueError("resident positions must form a rectangular layer/head grid")
    if previous_positions is not None:
        if len(previous_positions) != layer_count or any(
            len(layer) != head_count for layer in previous_positions
        ):
            raise ValueError("previous resident grid shape changed")

    diversity: list[float] = []
    union_efficiency: list[float] = []
    churn: list[float] = []
    all_positions: list[int] = []
    for layer_idx, layer in enumerate(resident_positions):
        sets = [{int(value) for value in head} for head in layer]
        if any(not values for values in sets):
            raise ValueError("every retained head set must be non-empty")
        all_positions.extend(value for values in sets for value in values)
        distances = [
            _jaccard_distance(sets[left], sets[right])
            for left in range(head_count)
            for right in range(left + 1, head_count)
        ]
        diversity.append(float(np.mean(distances)) if distances else 0.0)
        total_slots = sum(len(values) for values in sets)
        union_efficiency.append(len(set.union(*sets)) / total_slots)
        if previous_positions is not None:
            previous_sets = [
                {int(value) for value in head}
                for head in previous_positions[layer_idx]
            ]
            churn.extend(
                _jaccard_distance(previous_sets[head], sets[head])
                for head in range(head_count)
            )

    early_cut = input_tokens / 3.0
    late_cut = 2.0 * input_tokens / 3.0
    count = len(all_positions)
    return {
        "layers": layer_count,
        "kv_heads_per_layer": head_count,
        "retained_entries_with_head_multiplicity": count,
        "mean_cross_head_jaccard_distance": float(np.mean(diversity)),
        "mean_unique_position_fraction_across_heads": float(np.mean(union_efficiency)),
        "mean_boundary_churn_jaccard_distance": float(np.mean(churn)) if churn else None,
        "early_position_fraction": sum(value < early_cut for value in all_positions) / count,
        "middle_position_fraction": sum(
            early_cut <= value < late_cut for value in all_positions
        )
        / count,
        "late_position_fraction": sum(value >= late_cut for value in all_positions) / count,
        "mean_normalized_absolute_position": float(np.mean(all_positions))
        / max(1, input_tokens - 1),
    }


def specificity_decision(
    normalized_minus_controls: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen exploratory specificity rule to endpoint contrasts."""

    required = {"C5_RAW_S", "C4_SHUFFLED_SC", "C6_AGE_ONLY"}
    if set(normalized_minus_controls) != required:
        raise ValueError(f"specificity decision requires controls {sorted(required)}")
    means: dict[str, float] = {}
    lower_bounds: dict[str, float] = {}
    for control in sorted(required):
        row = normalized_minus_controls[control]
        mean = float(cast(float | int, row["mean_difference"]))
        interval = cast(Sequence[float], row["confidence_interval"])
        if len(interval) != 2 or not all(math.isfinite(float(value)) for value in interval):
            raise ValueError("specificity contrast has an invalid confidence interval")
        means[control] = mean
        lower_bounds[control] = float(interval[0])
    if any(value <= 0 for value in means.values()):
        decision = "NOT_SUPPORTED"
        reason = "normalized S/C did not exceed every raw, shuffled, and age control"
    elif all(value > 0 for value in lower_bounds.values()):
        decision = "SUPPORTED_EXPLORATORY_SIGNAL"
        reason = "all three endpoint deltas were positive with bootstrap lower bounds above zero"
    else:
        decision = "DIRECTIONAL_BUT_INCONCLUSIVE"
        reason = "all three endpoint deltas were positive, but at least one interval included zero"
    return {
        "decision": decision,
        "reason": reason,
        "mean_endpoint_differences": means,
        "confidence_interval_lower_bounds": lower_bounds,
        "classification": "EXPLORATORY_NOT_REGISTERED",
    }
