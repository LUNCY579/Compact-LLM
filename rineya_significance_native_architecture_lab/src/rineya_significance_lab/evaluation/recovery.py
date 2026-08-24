"""Answer-blind Cornerstone risk features and whole-cluster recovery curves.

This module is intentionally open-loop.  It may replay exact state recorded by
an on-policy compressed Halo run, but it does not claim that a counterfactual
recovery decision changes earlier hidden states.  The only closed-loop claim is
reserved for a live runtime action smoke.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..recovery import RecoverySignal

RISK_COMPONENT_NAMES: tuple[str, ...] = (
    "entropy",
    "concentration",
    "margin",
    "churn",
    "count",
    "query_delta",
)

FORBIDDEN_PRE_ANSWER_FIELDS = frozenset(
    {
        "accepted_answers",
        "answer",
        "correct",
        "evidence",
        "failure_labels",
        "full_cache_correct",
        "gold",
        "label",
        "prediction",
        "reference_answer",
        "target",
    }
)


@dataclass(frozen=True, slots=True)
class CornerstoneRiskFeatures:
    """Final-boundary features available before answer decoding."""

    case_id: str
    cluster_id: str
    task_family: str
    boundary_index: int
    layer_count: int
    kv_head_count: int
    candidate_count_per_unit: int
    entropy_normalized: float
    top_mass: float
    cutoff_margin_strength: float
    retained_set_churn: float
    count_cv_bounded: float
    query_delta_fraction: float
    risk_score: float
    weights: Mapping[str, float]

    @property
    def risk_components(self) -> dict[str, float]:
        return {
            "entropy": self.entropy_normalized,
            "concentration": 1.0 - self.top_mass,
            "margin": 1.0 - self.cutoff_margin_strength,
            "churn": self.retained_set_churn,
            "count": self.count_cv_bounded,
            "query_delta": 1.0 - self.query_delta_fraction,
        }

    def signal(self) -> RecoverySignal:
        return RecoverySignal(
            case_id=self.case_id,
            cluster_id=self.cluster_id,
            risk_score=self.risk_score,
            components=self.risk_components,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "case_id": self.case_id,
            "cluster_id": self.cluster_id,
            "task_family": self.task_family,
            "boundary_index": self.boundary_index,
            "layer_count": self.layer_count,
            "kv_head_count": self.kv_head_count,
            "candidate_count_per_unit": self.candidate_count_per_unit,
            "feature_provenance": "PRE_ANSWER_ON_POLICY_CORNERSTONE_STATE",
            "entropy_normalized": self.entropy_normalized,
            "top_mass": self.top_mass,
            "cutoff_margin_strength": self.cutoff_margin_strength,
            "retained_set_churn": self.retained_set_churn,
            "count_cv_bounded": self.count_cv_bounded,
            "query_delta_fraction": self.query_delta_fraction,
            "query_delta_derivation": (
                "cumulative-score difference aligned by persistent semantic entry_id"
            ),
            "risk_components": self.risk_components,
            "risk_weights": dict(self.weights),
            "risk_score": self.risk_score,
        }


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("risk diagnostic is non-finite")
    return float(np.clip(value, 0.0, 1.0))


def _boundary_array(
    arrays: Mapping[str, NDArray[Any]],
    *,
    boundary: int,
    layer: int,
    suffix: str,
    dtype: Any,
) -> NDArray[Any]:
    name = f"b{boundary:03d}_l{layer:03d}_{suffix}"
    if name not in arrays:
        raise KeyError(f"boundary trace is missing {name}")
    value = np.asarray(arrays[name], dtype=dtype)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] <= 0:
        raise ValueError(f"{name} must have shape [1, KV heads, positive positions]")
    result: NDArray[Any] = value[0]
    return result


def _normalized_entropy(values: NDArray[np.float64]) -> float:
    nonnegative = np.maximum(values, 0.0)
    if nonnegative.size <= 1:
        return 0.0
    total = float(nonnegative.sum())
    if total <= 0.0:
        return 1.0
    probabilities = nonnegative / total
    nonzero = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return _bounded(entropy / math.log(nonnegative.size))


def _top_mass(values: NDArray[np.float64], fraction: float) -> float:
    nonnegative = np.maximum(values, 0.0)
    total = float(nonnegative.sum())
    if total <= 0.0:
        return 0.0
    count = max(1, min(nonnegative.size, math.ceil(nonnegative.size * fraction)))
    top = np.partition(nonnegative, nonnegative.size - count)[-count:]
    return _bounded(float(top.sum()) / total)


def _cutoff_margin_strength(
    ranking: NDArray[np.float64],
    selected: NDArray[np.bool_],
) -> float:
    chosen = ranking[selected]
    rejected = ranking[~selected]
    if chosen.size == 0:
        raise ValueError("boundary selection retained no entries")
    if rejected.size == 0:
        # No eviction means the ranking boundary supplies no confidence signal.
        return 0.0
    gap = max(0.0, float(chosen.min()) - float(rejected.max()))
    scale = float(np.std(ranking))
    if scale <= 0.0:
        return 0.0
    return _bounded(gap / (gap + scale))


def _jaccard_distance(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return _bounded(1.0 - len(left & right) / len(union))


def _query_delta_fraction(
    *,
    current_ids: NDArray[np.int64],
    current_scores: NDArray[np.float64],
    previous_ids: NDArray[np.int64],
    previous_scores: NDArray[np.float64],
    previous_selected: NDArray[np.bool_],
) -> float:
    previous_all = {int(entry_id) for entry_id in previous_ids.tolist()}
    retained_map: dict[int, float] = {}
    for entry_id, score, retained in zip(
        previous_ids.tolist(),
        previous_scores.tolist(),
        previous_selected.tolist(),
        strict=True,
    ):
        semantic_id = int(entry_id)
        if semantic_id in retained_map:
            raise RuntimeError("duplicate semantic entry_id in one previous KV head")
        if retained:
            retained_map[semantic_id] = float(score)

    baseline = np.zeros(current_scores.shape, dtype=np.float64)
    seen: set[int] = set()
    for index, entry_id in enumerate(current_ids.tolist()):
        semantic_id = int(entry_id)
        if semantic_id in seen:
            raise RuntimeError("duplicate semantic entry_id in one current KV head")
        seen.add(semantic_id)
        if semantic_id in previous_all and semantic_id not in retained_map:
            raise RuntimeError("an entry evicted at the prior boundary reappeared in current state")
        baseline[index] = retained_map.get(semantic_id, 0.0)

    delta = current_scores - baseline
    tolerance = max(1e-6, 1e-5 * float(np.max(np.abs(current_scores), initial=0.0)))
    if np.any(delta < -tolerance):
        raise RuntimeError("cumulative significance decreased for a surviving semantic entry")
    positive_delta = np.maximum(delta, 0.0)
    total = float(np.maximum(current_scores, 0.0).sum())
    if total <= 0.0:
        return 0.0
    return _bounded(float(positive_delta.sum()) / total)


def extract_cornerstone_risk_features(
    arrays: Mapping[str, NDArray[Any]],
    *,
    case_id: str,
    cluster_id: str,
    task_family: str,
    boundary_index: int,
    layer_count: int,
    top_mass_fraction: float,
    weights: Mapping[str, float],
) -> CornerstoneRiskFeatures:
    """Extract a fixed pre-answer risk signal from an on-policy boundary trace."""

    if not case_id or not cluster_id or not task_family:
        raise ValueError("case, cluster, and family identifiers must be non-empty")
    if boundary_index <= 0:
        raise ValueError("query delta and churn require a boundary after boundary zero")
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if not 0.0 < top_mass_fraction <= 1.0:
        raise ValueError("top_mass_fraction must lie in (0, 1]")
    if set(weights) != set(RISK_COMPONENT_NAMES):
        raise ValueError(f"risk weights must cover exactly {RISK_COMPONENT_NAMES}")
    numeric_weights = {name: float(value) for name, value in weights.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in numeric_weights.values()):
        raise ValueError("risk weights must be finite and non-negative")
    total_weight = sum(numeric_weights.values())
    if total_weight <= 0.0:
        raise ValueError("at least one risk weight must be positive")

    entropy_rows: list[float] = []
    top_mass_rows: list[float] = []
    margin_rows: list[float] = []
    churn_rows: list[float] = []
    count_rows: list[float] = []
    query_delta_rows: list[float] = []
    kv_heads: int | None = None
    candidate_count: int | None = None

    for layer in range(layer_count):
        ranking = _boundary_array(
            arrays,
            boundary=boundary_index,
            layer=layer,
            suffix="ranking",
            dtype=np.float64,
        )
        counts = _boundary_array(
            arrays,
            boundary=boundary_index,
            layer=layer,
            suffix="true_counts",
            dtype=np.float64,
        )
        current_ids = _boundary_array(
            arrays,
            boundary=boundary_index,
            layer=layer,
            suffix="entry_ids",
            dtype=np.int64,
        )
        current_scores = _boundary_array(
            arrays,
            boundary=boundary_index,
            layer=layer,
            suffix="scores",
            dtype=np.float64,
        )
        current_selected = _boundary_array(
            arrays,
            boundary=boundary_index,
            layer=layer,
            suffix="selected",
            dtype=np.uint8,
        ).astype(np.bool_, copy=False)
        previous_ids = _boundary_array(
            arrays,
            boundary=boundary_index - 1,
            layer=layer,
            suffix="entry_ids",
            dtype=np.int64,
        )
        previous_scores = _boundary_array(
            arrays,
            boundary=boundary_index - 1,
            layer=layer,
            suffix="scores",
            dtype=np.float64,
        )
        previous_selected = _boundary_array(
            arrays,
            boundary=boundary_index - 1,
            layer=layer,
            suffix="selected",
            dtype=np.uint8,
        ).astype(np.bool_, copy=False)

        current_shapes = {
            ranking.shape,
            counts.shape,
            current_ids.shape,
            current_scores.shape,
            current_selected.shape,
        }
        previous_shapes = {previous_ids.shape, previous_scores.shape, previous_selected.shape}
        if len(current_shapes) != 1 or len(previous_shapes) != 1:
            raise ValueError("boundary arrays disagree on layer/head/position shape")
        if kv_heads is None:
            kv_heads = int(ranking.shape[0])
            candidate_count = int(ranking.shape[-1])
        elif ranking.shape[0] != kv_heads or ranking.shape[-1] != candidate_count:
            raise ValueError("layer geometry changed within one boundary")

        for head in range(ranking.shape[0]):
            unit_ranking = ranking[head]
            unit_counts = counts[head]
            unit_selected = current_selected[head]
            entropy_rows.append(_normalized_entropy(unit_ranking))
            top_mass_rows.append(_top_mass(unit_ranking, top_mass_fraction))
            margin_rows.append(_cutoff_margin_strength(unit_ranking, unit_selected))

            previous_set = {
                int(entry_id)
                for entry_id, retained in zip(
                    previous_ids[head].tolist(), previous_selected[head].tolist(), strict=True
                )
                if retained
            }
            current_set = {
                int(entry_id)
                for entry_id, retained in zip(
                    current_ids[head].tolist(), current_selected[head].tolist(), strict=True
                )
                if retained
            }
            churn_rows.append(_jaccard_distance(previous_set, current_set))

            mean_count = float(np.mean(unit_counts))
            count_cv = 0.0 if mean_count <= 0.0 else float(np.std(unit_counts)) / mean_count
            count_rows.append(_bounded(count_cv / (1.0 + count_cv)))
            query_delta_rows.append(
                _query_delta_fraction(
                    current_ids=current_ids[head],
                    current_scores=current_scores[head],
                    previous_ids=previous_ids[head],
                    previous_scores=previous_scores[head],
                    previous_selected=previous_selected[head],
                )
            )

    if kv_heads is None or candidate_count is None:
        raise RuntimeError("no risk features were extracted")
    raw = {
        "entropy_normalized": float(np.mean(entropy_rows)),
        "top_mass": float(np.mean(top_mass_rows)),
        "cutoff_margin_strength": float(np.mean(margin_rows)),
        "retained_set_churn": float(np.mean(churn_rows)),
        "count_cv_bounded": float(np.mean(count_rows)),
        "query_delta_fraction": float(np.mean(query_delta_rows)),
    }
    components = {
        "entropy": raw["entropy_normalized"],
        "concentration": 1.0 - raw["top_mass"],
        "margin": 1.0 - raw["cutoff_margin_strength"],
        "churn": raw["retained_set_churn"],
        "count": raw["count_cv_bounded"],
        "query_delta": 1.0 - raw["query_delta_fraction"],
    }
    risk = sum(numeric_weights[name] * components[name] for name in RISK_COMPONENT_NAMES)
    risk /= total_weight
    return CornerstoneRiskFeatures(
        case_id=case_id,
        cluster_id=cluster_id,
        task_family=task_family,
        boundary_index=boundary_index,
        layer_count=layer_count,
        kv_head_count=kv_heads,
        candidate_count_per_unit=candidate_count,
        entropy_normalized=_bounded(raw["entropy_normalized"]),
        top_mass=_bounded(raw["top_mass"]),
        cutoff_margin_strength=_bounded(raw["cutoff_margin_strength"]),
        retained_set_churn=_bounded(raw["retained_set_churn"]),
        count_cv_bounded=_bounded(raw["count_cv_bounded"]),
        query_delta_fraction=_bounded(raw["query_delta_fraction"]),
        risk_score=_bounded(risk),
        weights=numeric_weights,
    )


def assert_pre_answer_feature_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject outcome/oracle leakage in the durable feature view."""

    seen: set[str] = set()
    for row in rows:
        exposed = FORBIDDEN_PRE_ANSWER_FIELDS.intersection(row)
        if exposed:
            raise RuntimeError(f"pre-answer feature row exposes forbidden fields: {sorted(exposed)}")
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen:
            raise RuntimeError("pre-answer feature case IDs must be non-empty and unique")
        seen.add(case_id)
        if row.get("feature_provenance") != "PRE_ANSWER_ON_POLICY_CORNERSTONE_STATE":
            raise RuntimeError("pre-answer feature provenance is missing or invalid")


@dataclass(frozen=True, slots=True)
class RecoveryCaseOutcome:
    case_id: str
    cluster_id: str
    task_family: str
    risk_score: float
    compressed_correct: bool
    full_cache_correct: bool
    compressed_wall_seconds: float
    fallback_wall_seconds: float
    compressed_persistent_bytes: int
    fallback_persistent_bytes: int
    recompute_input_bytes: int


@dataclass(frozen=True, slots=True)
class RecoveryClusterOutcome:
    cluster_id: str
    task_family: str
    case_ids: tuple[str, ...]
    risk_score: float
    compressed_correct: bool
    full_cache_correct: bool
    compressed_wall_seconds: float
    fallback_wall_seconds: float
    compressed_peak_persistent_bytes: int
    fallback_peak_persistent_bytes: int
    recompute_input_bytes: int


def aggregate_recovery_clusters(
    rows: Sequence[RecoveryCaseOutcome],
) -> tuple[RecoveryClusterOutcome, ...]:
    """Make one indivisible gate decision per independent cluster."""

    if not rows:
        raise ValueError("at least one case outcome is required")
    grouped: defaultdict[str, list[RecoveryCaseOutcome]] = defaultdict(list)
    seen_cases: set[str] = set()
    for row in rows:
        if row.case_id in seen_cases:
            raise ValueError(f"duplicate recovery case: {row.case_id}")
        seen_cases.add(row.case_id)
        grouped[row.cluster_id].append(row)

    clusters: list[RecoveryClusterOutcome] = []
    for cluster_id in sorted(grouped):
        cases = grouped[cluster_id]
        families = {row.task_family for row in cases}
        if len(families) != 1:
            raise ValueError(f"cluster {cluster_id} spans multiple task families")
        clusters.append(
            RecoveryClusterOutcome(
                cluster_id=cluster_id,
                task_family=next(iter(families)),
                case_ids=tuple(sorted(row.case_id for row in cases)),
                # A cluster recovers if any member is risky; no within-cluster cherry-picking.
                risk_score=max(row.risk_score for row in cases),
                compressed_correct=all(row.compressed_correct for row in cases),
                full_cache_correct=all(row.full_cache_correct for row in cases),
                compressed_wall_seconds=sum(row.compressed_wall_seconds for row in cases),
                fallback_wall_seconds=sum(row.fallback_wall_seconds for row in cases),
                compressed_peak_persistent_bytes=max(
                    row.compressed_persistent_bytes for row in cases
                ),
                fallback_peak_persistent_bytes=max(
                    row.fallback_persistent_bytes for row in cases
                ),
                recompute_input_bytes=sum(row.recompute_input_bytes for row in cases),
            )
        )
    return tuple(clusters)


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    threshold: float
    accepted_clusters: int
    recovered_clusters: int
    coverage: float
    selective_risk: float | None
    system_accuracy: float
    rescued_errors: int
    harmed_correct: int
    recovery_precision: float | None
    total_wall_seconds: float
    mean_wall_seconds_per_cluster: float
    wall_multiplier_vs_compressed: float
    recompute_input_bytes: int
    peak_persistent_bytes: int
    incremental_peak_persistent_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "accepted_clusters": self.accepted_clusters,
            "recovered_clusters": self.recovered_clusters,
            "coverage": self.coverage,
            "selective_risk": self.selective_risk,
            "system_accuracy": self.system_accuracy,
            "rescued_errors": self.rescued_errors,
            "harmed_correct": self.harmed_correct,
            "recovery_precision": self.recovery_precision,
            "total_wall_seconds": self.total_wall_seconds,
            "mean_wall_seconds_per_cluster": self.mean_wall_seconds_per_cluster,
            "wall_multiplier_vs_compressed": self.wall_multiplier_vs_compressed,
            "recompute_input_bytes": self.recompute_input_bytes,
            "peak_persistent_bytes": self.peak_persistent_bytes,
            "incremental_peak_persistent_bytes": self.incremental_peak_persistent_bytes,
        }


def risk_coverage_curve(
    clusters: Sequence[RecoveryClusterOutcome],
) -> tuple[RiskCoveragePoint, ...]:
    """Sweep every tie-safe fixed threshold without fitting on outcomes."""

    if not clusters:
        raise ValueError("at least one cluster is required")
    scores = [float(row.risk_score) for row in clusters]
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("cluster risks must be finite and lie in [0, 1]")
    unique = sorted(set(scores), reverse=True)
    thresholds = [float(np.nextafter(max(unique), math.inf)), *unique]
    baseline_wall = sum(row.compressed_wall_seconds for row in clusters)
    if baseline_wall <= 0.0:
        raise ValueError("compressed wall time must be positive")
    baseline_peak = max(row.compressed_peak_persistent_bytes for row in clusters)
    total = len(clusters)
    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        recovered = [row for row in clusters if row.risk_score >= threshold]
        accepted = [row for row in clusters if row.risk_score < threshold]
        accepted_errors = sum(not row.compressed_correct for row in accepted)
        system_correct = sum(
            row.full_cache_correct if row in recovered else row.compressed_correct
            for row in clusters
        )
        rescued = sum(
            (not row.compressed_correct) and row.full_cache_correct for row in recovered
        )
        harmed = sum(
            row.compressed_correct and (not row.full_cache_correct) for row in recovered
        )
        fallback_wall = sum(row.fallback_wall_seconds for row in recovered)
        total_wall = baseline_wall + fallback_wall
        peak = max(
            baseline_peak,
            max((row.fallback_peak_persistent_bytes for row in recovered), default=0),
        )
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                accepted_clusters=len(accepted),
                recovered_clusters=len(recovered),
                coverage=len(accepted) / total,
                selective_risk=(accepted_errors / len(accepted) if accepted else None),
                system_accuracy=system_correct / total,
                rescued_errors=rescued,
                harmed_correct=harmed,
                recovery_precision=(rescued / len(recovered) if recovered else None),
                total_wall_seconds=total_wall,
                mean_wall_seconds_per_cluster=total_wall / total,
                wall_multiplier_vs_compressed=total_wall / baseline_wall,
                recompute_input_bytes=sum(row.recompute_input_bytes for row in recovered),
                peak_persistent_bytes=peak,
                incremental_peak_persistent_bytes=max(0, peak - baseline_peak),
            )
        )
    return tuple(points)


def descriptive_error_roc_auc(clusters: Sequence[RecoveryClusterOutcome]) -> float | None:
    """Pairwise descriptive AUC for ranking compressed errors above successes."""

    failures = [row.risk_score for row in clusters if not row.compressed_correct]
    successes = [row.risk_score for row in clusters if row.compressed_correct]
    if not failures or not successes:
        return None
    credit = 0.0
    for failure in failures:
        for success in successes:
            credit += 1.0 if failure > success else (0.5 if failure == success else 0.0)
    return credit / (len(failures) * len(successes))
