"""Cluster-aware analysis for the closed-loop exposure-count controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .clustered import clustered_paired_analysis

REFERENCE_COUNT_CONDITION = "true_realized_count"
COUNT_CONDITIONS = (
    REFERENCE_COUNT_CONDITION,
    "constant_count",
    "age_derived_count",
    "shuffled_count",
    "wrong_token_count",
    "delayed_count",
    "noisy_count",
    "head_averaged_count",
)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values with monotonicity enforcement."""

    if not p_values:
        raise ValueError("at least one p-value is required")
    ordered = sorted((float(value), str(name)) for name, value in p_values.items())
    if any(not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("p-values must be in [0, 1]")
    total = len(ordered)
    previous = 0.0
    result: dict[str, float] = {}
    for index, (value, name) in enumerate(ordered):
        adjusted = min(1.0, max(previous, (total - index) * value))
        result[name] = adjusted
        previous = adjusted
    return dict(sorted(result.items()))


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _arrays_equal(
    left: Mapping[str, np.ndarray[Any, Any]],
    right: Mapping[str, np.ndarray[Any, Any]],
    predicate: Callable[[str], bool],
) -> bool:
    names = sorted(name for name in left if predicate(name))
    if not names or names != sorted(name for name in right if predicate(name)):
        return False
    return all(
        left[name].shape == right[name].shape and np.array_equal(left[name], right[name])
        for name in names
    )


def _trace_equivalence(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, bool]:
    baseline_paths = [artifact_root / str(value) for value in baseline["trace_ids"]]
    candidate_paths = [artifact_root / str(value) for value in candidate["trace_ids"]]
    if len(baseline_paths) != 2 or len(candidate_paths) != 2:
        raise ValueError("count-control rows must reference resident and boundary NPZ traces")
    baseline_resident = _load_npz(baseline_paths[0])
    candidate_resident = _load_npz(candidate_paths[0])
    baseline_boundary = _load_npz(baseline_paths[1])
    candidate_boundary = _load_npz(candidate_paths[1])
    return {
        "prediction_exact": baseline["prediction"] == candidate["prediction"],
        "correctness_exact": bool(baseline["correct"]) == bool(candidate["correct"]),
        "primary_endpoint_exact": (
            float(baseline["evidence"]["layer_fact_exact_completion_rate"])
            == float(candidate["evidence"]["layer_fact_exact_completion_rate"])
        ),
        "final_resident_positions_exact": _arrays_equal(
            baseline_resident,
            candidate_resident,
            lambda _name: True,
        ),
        "boundary_selection_trajectory_exact": _arrays_equal(
            baseline_boundary,
            candidate_boundary,
            lambda name: name.endswith("_selected") or name.endswith("_retained_positions"),
        ),
        "effective_count_trajectory_exact": _arrays_equal(
            baseline_boundary,
            candidate_boundary,
            lambda name: name.endswith("_effective_counts"),
        ),
        "true_count_trajectory_exact": _arrays_equal(
            baseline_boundary,
            candidate_boundary,
            lambda name: name.endswith("_true_counts"),
        ),
        "score_trajectory_exact": _arrays_equal(
            baseline_boundary,
            candidate_boundary,
            lambda name: name.endswith("_scores"),
        ),
        "complete_boundary_diagnostic_exact": _arrays_equal(
            baseline_boundary,
            candidate_boundary,
            lambda _name: True,
        ),
    }


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[tuple[str, str], Mapping[str, Any]]]:
    if not rows:
        raise ValueError("count-control rows must be non-empty")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    cluster_by_case: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        condition = str(row.get("condition", ""))
        cluster_id = str(row.get("scenario_id", ""))
        if not case_id or not cluster_id or condition not in COUNT_CONDITIONS:
            raise ValueError("invalid count-control case/cluster/condition")
        if (case_id, condition) in by_key:
            raise ValueError(f"duplicate count-control cell {(case_id, condition)}")
        by_key[(case_id, condition)] = row
        previous = cluster_by_case.setdefault(case_id, cluster_id)
        if previous != cluster_id:
            raise ValueError(f"case {case_id!r} maps to multiple clusters")
    cases = sorted(cluster_by_case)
    for case_id in cases:
        observed = {condition for observed_case, condition in by_key if observed_case == case_id}
        if observed != set(COUNT_CONDITIONS):
            raise ValueError(f"case {case_id!r} does not contain all count controls")
    return cases, by_key


def analyze_count_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_root: str | Path,
    minimum_clusters: int = 8,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
    seed: int = 17,
) -> dict[str, Any]:
    """Compare every intervention to true realized ``C`` on cluster units."""

    cases, by_key = _validate_rows(rows)
    root = Path(artifact_root).resolve(strict=True)
    metrics: dict[str, Callable[[Mapping[str, Any]], float]] = {
        "layer_fact_exact_completion_rate": lambda row: float(
            row["evidence"]["layer_fact_exact_completion_rate"]
        ),
        "minimum_per_layer_required_token_availability_fraction": lambda row: float(
            row["evidence"]["minimum_per_layer_required_token_availability_fraction"]
        ),
        "answer_correct": lambda row: float(bool(row["correct"])),
    }
    comparisons: dict[str, Any] = {}
    primary_p_values: dict[str, float] = {}
    for condition_index, condition in enumerate(COUNT_CONDITIONS[1:]):
        paired_metrics: dict[str, Any] = {}
        for metric_index, (metric_name, getter) in enumerate(metrics.items()):
            baseline_values = [getter(by_key[(case_id, REFERENCE_COUNT_CONDITION)]) for case_id in cases]
            candidate_values = [getter(by_key[(case_id, condition)]) for case_id in cases]
            result = clustered_paired_analysis(
                cases,
                baseline_values,
                candidate_values,
                minimum_clusters=minimum_clusters,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                randomization_samples=randomization_samples,
                seed=seed + 100 * condition_index + metric_index,
            )
            paired_metrics[metric_name] = {
                **asdict(result),
                "wins": sum(right > left for left, right in zip(baseline_values, candidate_values, strict=True)),
                "losses": sum(right < left for left, right in zip(baseline_values, candidate_values, strict=True)),
                "ties": sum(right == left for left, right in zip(baseline_values, candidate_values, strict=True)),
            }
            if metric_name == "layer_fact_exact_completion_rate":
                primary_p_values[condition] = result.p_value_two_sided

        equivalence = [
            _trace_equivalence(
                by_key[(case_id, REFERENCE_COUNT_CONDITION)],
                by_key[(case_id, condition)],
                artifact_root=root,
            )
            for case_id in cases
        ]
        comparisons[condition] = {
            "paired_metrics": paired_metrics,
            "exact_equivalence_case_counts": {
                name: sum(bool(row[name]) for row in equivalence)
                for name in equivalence[0]
            },
            "case_count": len(cases),
        }

    adjusted = holm_adjust(primary_p_values)
    for condition, value in adjusted.items():
        comparisons[condition]["paired_metrics"]["layer_fact_exact_completion_rate"][
            "holm_adjusted_p_value_across_count_controls"
        ] = value

    age_counts = comparisons["age_derived_count"]["exact_equivalence_case_counts"]
    head_counts = comparisons["head_averaged_count"]["exact_equivalence_case_counts"]
    age_exact = all(value == len(cases) for value in age_counts.values())
    head_exact = all(value == len(cases) for value in head_counts.values())
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_CLUSTER_AWARE_COUNT_CAUSAL_ANALYSIS",
        "reference_condition": REFERENCE_COUNT_CONDITION,
        "independent_clusters": len(cases),
        "comparisons": comparisons,
        "primary_multiplicity_family": list(COUNT_CONDITIONS[1:]),
        "mechanism_diagnostics": {
            "age_derived_count_exactly_reconstructs_true_realized_count_trajectory": age_exact,
            "head_averaged_count_exactly_reconstructs_true_realized_count_trajectory": head_exact,
            "explicit_count_state_adds_information_beyond_deterministic_age_in_this_regime": (
                False if age_exact else None
            ),
            "scope": (
                "fixed 128-token chunks, one query-scoring event at every non-final boundary, "
                "B0-selected R1/R3/R4 Halo clusters"
            ),
        },
        "claim_limit": (
            "Exploratory calibration-selected analysis. Exact trajectory equivalence is valid for "
            "this event schedule only; statistical contrasts are not registered confirmatory evidence."
        ),
    }
