"""Leakage-safe primitives for Architecture Experiment F.

The module is deliberately independent of the Halo runtime.  It consumes only
the six durable, pre-answer Cornerstone features produced by Experiment E and
fits tiny deterministic CPU controllers.  It contains no answer-file I/O; the
runner owns and records the phase boundary between feature/split freezing and
label access.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES: tuple[str, ...] = (
    "entropy",
    "concentration",
    "margin",
    "churn",
    "count",
    "query_delta",
)
SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "evaluation")

FORBIDDEN_PRE_ANSWER_FIELDS = frozenset(
    {
        "accepted_answers",
        "answer",
        "compressed_correct",
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

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _as_float_array(values: Sequence[Sequence[float]] | NDArray[np.float64]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("features must be finite")
    return array


def _as_bool_array(values: Sequence[bool] | NDArray[np.bool_], *, size: int) -> BoolArray:
    array = np.asarray(values, dtype=np.bool_)
    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError("labels must be a one-dimensional vector aligned to features")
    return array


def assert_pre_answer_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject outcome/oracle leakage and malformed six-feature rows."""

    if not rows:
        raise ValueError("at least one pre-answer feature row is required")
    seen_cases: set[str] = set()
    seen_clusters: set[str] = set()
    for row in rows:
        exposed = FORBIDDEN_PRE_ANSWER_FIELDS.intersection(row)
        if exposed:
            raise RuntimeError(f"pre-answer row exposes forbidden fields: {sorted(exposed)}")
        case_id = str(row.get("case_id", ""))
        cluster_id = str(row.get("cluster_id", ""))
        if not case_id or case_id in seen_cases:
            raise RuntimeError("pre-answer case IDs must be non-empty and unique")
        if not cluster_id or cluster_id in seen_clusters:
            raise RuntimeError(
                "Experiment F v1 requires exactly one durable feature row per cluster"
            )
        seen_cases.add(case_id)
        seen_clusters.add(cluster_id)
        if row.get("feature_provenance") != "PRE_ANSWER_ON_POLICY_CORNERSTONE_STATE":
            raise RuntimeError("pre-answer feature provenance is missing or invalid")
        components = row.get("risk_components")
        if not isinstance(components, Mapping) or set(components) != set(FEATURE_NAMES):
            raise RuntimeError(f"risk_components must cover exactly {FEATURE_NAMES}")
        numeric = np.asarray([float(components[name]) for name in FEATURE_NAMES], dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0) or np.any(numeric > 1.0):
            raise RuntimeError("pre-answer features must be finite and lie in [0, 1]")
        equal_weight = float(np.mean(numeric))
        if not math.isclose(equal_weight, float(row.get("risk_score", math.nan)), abs_tol=1e-12):
            raise RuntimeError("saved deterministic risk score is not the equal-weight feature mean")


def feature_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    components = row.get("risk_components")
    if not isinstance(components, Mapping):
        raise ValueError("row has no risk_components mapping")
    values = tuple(float(components[name]) for name in FEATURE_NAMES)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("feature vector contains a non-finite value")
    return values


def cluster_digest(cluster_id: str, *, salt: str) -> str:
    if not cluster_id or not salt:
        raise ValueError("cluster_id and split salt must be non-empty")
    return hashlib.sha256(f"{salt}\0{cluster_id}".encode()).hexdigest()


def deterministic_cluster_split(
    cluster_ids: Sequence[str],
    *,
    salt: str,
    train_count: int,
    calibration_count: int,
    evaluation_count: int,
) -> dict[str, str]:
    """Assign exact split sizes using only a salted cluster-ID hash rank.

    Labels are intentionally not accepted by this function.  There is no
    retry, stratification, or alternate-salt path.
    """

    ids = list(cluster_ids)
    if any(not cluster_id for cluster_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("cluster IDs must be non-empty and unique")
    counts = (train_count, calibration_count, evaluation_count)
    if any(count <= 0 for count in counts) or sum(counts) != len(ids):
        raise ValueError("positive split counts must sum to the cluster count")
    ranked = sorted(ids, key=lambda cluster_id: (cluster_digest(cluster_id, salt=salt), cluster_id))
    result: dict[str, str] = {}
    train_end = train_count
    calibration_end = train_end + calibration_count
    for index, cluster_id in enumerate(ranked):
        if index < train_end:
            split = "train"
        elif index < calibration_end:
            split = "calibration"
        else:
            split = "evaluation"
        result[cluster_id] = split
    return result


def assert_each_split_has_both_classes(
    labels_by_cluster: Mapping[str, bool],
    split_by_cluster: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    """Fail rather than alter a frozen split when binary support is absent."""

    if set(labels_by_cluster) != set(split_by_cluster):
        missing_labels = sorted(set(split_by_cluster) - set(labels_by_cluster))
        extra_labels = sorted(set(labels_by_cluster) - set(split_by_cluster))
        raise RuntimeError(
            f"label/split cluster mismatch; missing={missing_labels}, extra={extra_labels}"
        )
    support: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        labels = [labels_by_cluster[key] for key, value in split_by_cluster.items() if value == split]
        negatives = sum(not label for label in labels)
        positives = sum(labels)
        support[split] = {"compressed_success": negatives, "compressed_error": positives}
        if negatives == 0 or positives == 0:
            raise RuntimeError(
                f"frozen {split} split lacks both classes: successes={negatives}, errors={positives}; "
                "outcome-based reassignment is forbidden"
            )
    return support


@dataclass(frozen=True, slots=True)
class Standardization:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    zero_variance_features: tuple[int, ...]

    def transform(self, values: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
        array = _as_float_array(values)
        if array.shape[1] != len(self.mean):
            raise ValueError("feature dimension does not match fitted standardization")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        return (array - mean) / scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "zero_variance_feature_indices": list(self.zero_variance_features),
            "fit_partition": "train_only",
        }


def fit_standardization(values: Sequence[Sequence[float]] | FloatArray) -> Standardization:
    array = _as_float_array(values)
    mean = np.mean(array, axis=0)
    raw_scale = np.std(array, axis=0)
    zero = tuple(int(index) for index in np.flatnonzero(raw_scale <= 1e-12))
    scale = np.where(raw_scale <= 1e-12, 1.0, raw_scale)
    return Standardization(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        zero_variance_features=zero,
    )


def _sigmoid(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -40.0, 40.0)
    result: FloatArray = 1.0 / (1.0 + np.exp(-clipped))
    return result


def _balanced_sample_weights(labels: BoolArray) -> FloatArray:
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("logistic training requires both classes")
    weights = np.where(
        labels,
        labels.size / (2.0 * positives),
        labels.size / (2.0 * negatives),
    )
    result: FloatArray = weights.astype(np.float64, copy=False)
    return result


def _logistic_loss(
    design: FloatArray,
    labels: FloatArray,
    sample_weights: FloatArray,
    parameters: FloatArray,
    l2_penalty: float,
) -> float:
    logits = design @ parameters
    # logaddexp is the stable softplus(logit).
    data_loss = np.sum(sample_weights * (np.logaddexp(0.0, logits) - labels * logits))
    data_loss /= float(np.sum(sample_weights))
    return float(data_loss + 0.5 * l2_penalty * np.dot(parameters[1:], parameters[1:]))


@dataclass(frozen=True, slots=True)
class LogisticController:
    intercept: float
    coefficients: tuple[float, ...]
    standardization: Standardization
    l2_penalty: float
    iterations: int
    converged: bool
    optimizer: str = "damped_Newton_IRLS"

    @property
    def trainable_parameter_count(self) -> int:
        return len(self.coefficients) + 1

    @property
    def fitted_preprocessing_scalar_count(self) -> int:
        return 2 * len(self.coefficients)

    def score(self, values: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
        standardized = self.standardization.transform(values)
        logits = self.intercept + standardized @ np.asarray(self.coefficients, dtype=np.float64)
        return _sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "l2_logistic_regression",
            "feature_names": list(FEATURE_NAMES),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "standardization": self.standardization.to_dict(),
            "l2_penalty": self.l2_penalty,
            "optimizer": self.optimizer,
            "iterations": self.iterations,
            "converged": self.converged,
            "trainable_parameter_count": self.trainable_parameter_count,
            "fitted_preprocessing_scalar_count": self.fitted_preprocessing_scalar_count,
            "backbone_parameters_trainable": 0,
        }


def fit_logistic_controller(
    values: Sequence[Sequence[float]] | FloatArray,
    labels: Sequence[bool] | BoolArray,
    *,
    l2_penalty: float,
    max_iterations: int,
    tolerance: float,
    observer: Callable[[], None] | None = None,
) -> LogisticController:
    """Fit a tiny class-balanced L2 logistic model by deterministic IRLS."""

    features = _as_float_array(values)
    outcomes = _as_bool_array(labels, size=features.shape[0])
    if not math.isfinite(l2_penalty) or l2_penalty <= 0.0:
        raise ValueError("l2_penalty must be finite and positive")
    if max_iterations <= 0 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("optimizer limits must be positive")
    standardization = fit_standardization(features)
    standardized = standardization.transform(features)
    design = np.column_stack((np.ones(features.shape[0], dtype=np.float64), standardized))
    numeric_labels = outcomes.astype(np.float64)
    sample_weights = _balanced_sample_weights(outcomes)
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.zeros(design.shape[1], dtype=np.float64)
    regularizer[1:] = l2_penalty
    converged = False
    iterations = 0
    if observer is not None:
        observer()

    for iteration in range(1, max_iterations + 1):
        probabilities = _sigmoid(design @ parameters)
        total_weight = float(np.sum(sample_weights))
        residual_weights = sample_weights * (probabilities - numeric_labels)
        gradient = (design.T @ residual_weights) / total_weight + regularizer * parameters
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = (design.T @ (design * curvature[:, None])) / total_weight
        hessian += np.diag(regularizer)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        old_loss = _logistic_loss(
            design, numeric_labels, sample_weights, parameters, l2_penalty
        )
        step_scale = 1.0
        candidate = parameters - step
        while (
            _logistic_loss(design, numeric_labels, sample_weights, candidate, l2_penalty)
            > old_loss
            and step_scale > 2.0**-24
        ):
            step_scale *= 0.5
            candidate = parameters - step_scale * step
        parameters = candidate
        iterations = iteration
        if observer is not None:
            observer()
        if float(np.max(np.abs(step_scale * step))) <= tolerance:
            converged = True
            break

    return LogisticController(
        intercept=float(parameters[0]),
        coefficients=tuple(float(value) for value in parameters[1:]),
        standardization=standardization,
        l2_penalty=l2_penalty,
        iterations=iterations,
        converged=converged,
    )


def _gini(labels: BoolArray) -> float:
    if labels.size == 0:
        return 0.0
    positive = float(np.mean(labels))
    return 2.0 * positive * (1.0 - positive)


@dataclass(frozen=True, slots=True)
class TreeNode:
    probability: float
    sample_count: int
    positive_count: int
    feature_index: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None

    @property
    def node_count(self) -> int:
        return 1 + (0 if self.left is None else self.left.node_count) + (
            0 if self.right is None else self.right.node_count
        )

    @property
    def leaf_count(self) -> int:
        if self.is_leaf:
            return 1
        if self.left is None or self.right is None:
            raise RuntimeError("tree node is missing a child")
        return self.left.leaf_count + self.right.leaf_count

    @property
    def internal_count(self) -> int:
        return self.node_count - self.leaf_count

    @property
    def learned_scalar_count(self) -> int:
        # Each internal node learns a feature index and threshold; each leaf
        # learns one empirical probability.
        return 2 * self.internal_count + self.leaf_count

    def score_one(self, row: FloatArray) -> float:
        node = self
        while not node.is_leaf:
            if (
                node.feature_index is None
                or node.threshold is None
                or node.left is None
                or node.right is None
            ):
                raise RuntimeError("malformed shallow tree")
            node = node.left if row[node.feature_index] < node.threshold else node.right
        return node.probability

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "probability": self.probability,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
        }
        if not self.is_leaf:
            if self.feature_index is None or self.left is None or self.right is None:
                raise RuntimeError("malformed shallow tree")
            value.update(
                {
                    "feature_index": self.feature_index,
                    "feature_name": FEATURE_NAMES[self.feature_index],
                    "threshold": self.threshold,
                    "left": self.left.to_dict(),
                    "right": self.right.to_dict(),
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class ShallowTreeController:
    root: TreeNode
    max_depth: int
    min_leaf_samples: int
    laplace: float

    @property
    def learned_scalar_count(self) -> int:
        return self.root.learned_scalar_count

    def score(self, values: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
        features = _as_float_array(values)
        if features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("feature dimension does not match shallow tree")
        return np.asarray([self.root.score_one(row) for row in features], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "deterministic_shallow_CART_control",
            "feature_names": list(FEATURE_NAMES),
            "max_depth": self.max_depth,
            "min_leaf_samples": self.min_leaf_samples,
            "leaf_probability_smoothing": f"Laplace({self.laplace})",
            "node_count": self.root.node_count,
            "leaf_count": self.root.leaf_count,
            "internal_count": self.root.internal_count,
            "learned_scalar_count": self.learned_scalar_count,
            "optimizer": "exhaustive_gini_split_search_with_deterministic_ties",
            "backbone_parameters_trainable": 0,
            "root": self.root.to_dict(),
        }


def fit_shallow_tree(
    values: Sequence[Sequence[float]] | FloatArray,
    labels: Sequence[bool] | BoolArray,
    *,
    max_depth: int,
    min_leaf_samples: int,
    laplace: float,
    observer: Callable[[], None] | None = None,
) -> ShallowTreeController:
    """Fit a deterministic depth-limited CART control on training rows only."""

    features = _as_float_array(values)
    outcomes = _as_bool_array(labels, size=features.shape[0])
    if set(outcomes.tolist()) != {False, True}:
        raise ValueError("tree training requires both classes")
    if max_depth < 1 or min_leaf_samples < 1 or not math.isfinite(laplace) or laplace <= 0:
        raise ValueError("invalid shallow-tree hyperparameters")

    def build(indices: NDArray[np.int64], depth: int) -> TreeNode:
        selected_labels = outcomes[indices]
        positive_count = int(np.sum(selected_labels))
        probability = (positive_count + laplace) / (indices.size + 2.0 * laplace)
        leaf = TreeNode(
            probability=float(probability),
            sample_count=int(indices.size),
            positive_count=positive_count,
        )
        if depth >= max_depth or positive_count in {0, int(indices.size)}:
            return leaf

        parent_gini = _gini(selected_labels)
        best: tuple[float, int, float, NDArray[np.int64], NDArray[np.int64]] | None = None
        for feature_index in range(features.shape[1]):
            unique = np.unique(features[indices, feature_index])
            for lower, upper in pairwise(unique):
                threshold = float((lower + upper) / 2.0)
                left_mask = features[indices, feature_index] < threshold
                left = indices[left_mask]
                right = indices[~left_mask]
                if left.size < min_leaf_samples or right.size < min_leaf_samples:
                    continue
                weighted = (
                    left.size * _gini(outcomes[left]) + right.size * _gini(outcomes[right])
                ) / indices.size
                gain = parent_gini - weighted
                candidate = (float(gain), feature_index, threshold, left, right)
                if best is None:
                    best = candidate
                else:
                    best_key = (best[0], -best[1], -best[2])
                    candidate_key = (candidate[0], -candidate[1], -candidate[2])
                    if candidate_key > best_key:
                        best = candidate
        if best is None or best[0] <= 1e-15:
            return leaf
        _, feature_index, threshold, left_indices, right_indices = best
        if observer is not None:
            observer()
        return TreeNode(
            probability=float(probability),
            sample_count=int(indices.size),
            positive_count=positive_count,
            feature_index=feature_index,
            threshold=threshold,
            left=build(left_indices, depth + 1),
            right=build(right_indices, depth + 1),
        )

    if observer is not None:
        observer()
    root = build(np.arange(features.shape[0], dtype=np.int64), 0)
    return ShallowTreeController(
        root=root,
        max_depth=max_depth,
        min_leaf_samples=min_leaf_samples,
        laplace=laplace,
    )


def error_roc_auc(
    labels: Sequence[bool] | BoolArray, scores: Sequence[float] | FloatArray
) -> float:
    outcomes = np.asarray(labels, dtype=np.bool_)
    values = np.asarray(scores, dtype=np.float64)
    if outcomes.ndim != 1 or values.ndim != 1 or outcomes.shape != values.shape:
        raise ValueError("labels and scores must be aligned vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    positive_scores = values[outcomes]
    negative_scores = values[~outcomes]
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("ROC AUC requires both classes")
    credit = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            credit += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    return credit / float(positive_scores.size * negative_scores.size)


def binary_metrics(
    labels: Sequence[bool] | BoolArray,
    scores: Sequence[float] | FloatArray,
    actions: Sequence[bool] | BoolArray,
) -> dict[str, Any]:
    outcomes = np.asarray(labels, dtype=np.bool_)
    values = np.asarray(scores, dtype=np.float64)
    decisions = np.asarray(actions, dtype=np.bool_)
    if outcomes.ndim != 1 or values.shape != outcomes.shape or decisions.shape != outcomes.shape:
        raise ValueError("labels, scores, and actions must be aligned vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    true_positive = int(np.sum(outcomes & decisions))
    true_negative = int(np.sum(~outcomes & ~decisions))
    false_positive = int(np.sum(~outcomes & decisions))
    false_negative = int(np.sum(outcomes & ~decisions))
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    if positive_count == 0 or negative_count == 0:
        raise ValueError("balanced metrics require both classes")
    sensitivity = true_positive / positive_count
    specificity = true_negative / negative_count
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision is not None and precision + sensitivity > 0.0
        else None
    )
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    numeric = outcomes.astype(np.float64)
    log_loss = -float(np.mean(numeric * np.log(clipped) + (1.0 - numeric) * np.log(1.0 - clipped)))
    return {
        "n": int(outcomes.size),
        "class_support": {
            "compressed_error": positive_count,
            "compressed_success": negative_count,
        },
        "confusion": {
            "true_positive_error_recovered": true_positive,
            "true_negative_success_accepted": true_negative,
            "false_positive_success_recovered": false_positive,
            "false_negative_error_accepted": false_negative,
        },
        "accuracy": (true_positive + true_negative) / outcomes.size,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "error_recall": sensitivity,
        "success_specificity": specificity,
        "recovery_precision": precision,
        "f1_error": f1,
        "error_roc_auc": error_roc_auc(outcomes, values),
        "brier_score": float(np.mean((values - numeric) ** 2)),
        "log_loss": log_loss,
        "recovered": int(np.sum(decisions)),
        "accepted": int(outcomes.size - np.sum(decisions)),
        "coverage": float(np.mean(~decisions)),
        "selective_risk": (
            float(np.mean(outcomes[~decisions])) if np.any(~decisions) else None
        ),
    }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    threshold: float
    balanced_accuracy: float
    recovered: int
    accepted: int
    candidate_count: int
    tie_break: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "balanced_accuracy": self.balanced_accuracy,
            "recovered": self.recovered,
            "accepted": self.accepted,
            "candidate_count": self.candidate_count,
            "objective": "maximum_balanced_accuracy",
            "tie_break": self.tie_break,
            "fit_partition": "calibration_only",
        }


def calibrate_threshold(
    labels: Sequence[bool] | BoolArray,
    scores: Sequence[float] | FloatArray,
) -> CalibrationResult:
    """Pick a tie-safe threshold on calibration labels only."""

    outcomes = np.asarray(labels, dtype=np.bool_)
    values = np.asarray(scores, dtype=np.float64)
    if outcomes.ndim != 1 or values.shape != outcomes.shape or outcomes.size == 0:
        raise ValueError("calibration labels and scores must be non-empty aligned vectors")
    if set(outcomes.tolist()) != {False, True}:
        raise ValueError("threshold calibration requires both classes")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be finite")
    unique = sorted({float(value) for value in values}, reverse=True)
    thresholds = [float(np.nextafter(max(unique), math.inf)), *unique]
    best: tuple[tuple[float, int, float], CalibrationResult] | None = None
    tie_break = "fewer_recoveries_then_higher_threshold"
    for threshold in thresholds:
        actions = values >= threshold
        metrics = binary_metrics(outcomes, values, actions)
        recovered = int(np.sum(actions))
        result = CalibrationResult(
            threshold=threshold,
            balanced_accuracy=float(metrics["balanced_accuracy"]),
            recovered=recovered,
            accepted=int(outcomes.size - recovered),
            candidate_count=len(thresholds),
            tie_break=tie_break,
        )
        key = (result.balanced_accuracy, -result.recovered, result.threshold)
        if best is None or key > best[0]:
            best = (key, result)
    if best is None:  # pragma: no cover - non-empty input guarantees candidates
        raise RuntimeError("threshold calibration produced no candidate")
    return best[1]


def hash_budget_scores(cluster_ids: Sequence[str], *, salt: str) -> FloatArray:
    """Return feature/label-blind continuous scores for a hash-budget control."""

    ids = list(cluster_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("budget-control cluster IDs must be non-empty and unique")
    ranked = sorted(ids, key=lambda cluster_id: (cluster_digest(cluster_id, salt=salt), cluster_id))
    rank = {cluster_id: index for index, cluster_id in enumerate(ranked)}
    denominator = max(1, len(ids) - 1)
    # Earlier hashes have higher recovery priority.
    return np.asarray(
        [1.0 - rank[cluster_id] / denominator for cluster_id in ids], dtype=np.float64
    )


def exact_budget_actions(
    scores: Sequence[float] | FloatArray, *, recover_count: int
) -> BoolArray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("budget scores must be a finite non-empty vector")
    if not 0 <= recover_count <= values.size:
        raise ValueError("recovery budget is outside the evaluation population")
    order = np.argsort(-values, kind="stable")
    decisions = np.zeros(values.size, dtype=np.bool_)
    decisions[order[:recover_count]] = True
    return decisions


def risk_coverage_curve(
    labels: Sequence[bool] | BoolArray,
    full_cache_correct: Sequence[bool] | BoolArray,
    scores: Sequence[float] | FloatArray,
) -> tuple[dict[str, Any], ...]:
    """Enumerate held-out descriptive, tie-safe risk/coverage points."""

    outcomes = np.asarray(labels, dtype=np.bool_)
    full = np.asarray(full_cache_correct, dtype=np.bool_)
    values = np.asarray(scores, dtype=np.float64)
    if outcomes.ndim != 1 or full.shape != outcomes.shape or values.shape != outcomes.shape:
        raise ValueError("risk/coverage inputs must be aligned vectors")
    unique = sorted({float(value) for value in values}, reverse=True)
    if not unique:
        raise ValueError("risk/coverage curve requires observations")
    thresholds = [float(np.nextafter(max(unique), math.inf)), *unique]
    points: list[dict[str, Any]] = []
    compressed_correct = ~outcomes
    for threshold in thresholds:
        recover = values >= threshold
        accept = ~recover
        system_correct = np.where(recover, full, compressed_correct)
        points.append(
            {
                "threshold": threshold,
                "accepted": int(np.sum(accept)),
                "recovered": int(np.sum(recover)),
                "coverage": float(np.mean(accept)),
                "selective_risk": float(np.mean(outcomes[accept])) if np.any(accept) else None,
                "system_accuracy": float(np.mean(system_correct)),
                "rescued_errors": int(np.sum(recover & outcomes & full)),
                "harmed_successes": int(np.sum(recover & ~outcomes & ~full)),
                "descriptive_only": True,
            }
        )
    return tuple(points)
