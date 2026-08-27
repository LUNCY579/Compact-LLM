"""Section 16 controlled mathematical-feasibility experiment for PCSM.

Every object in this module is synthetic. The experiment asks whether a small
generator can learn which structural peer maps determine a hidden sparse optimum,
then make one exact-support update. It is not a language-model experiment and
must never be reported as evidence of real-model or long-context improvement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Self, cast

import numpy as np
from numpy.typing import NDArray

from .controls import (
    bitwise_frozen_audit,
    deterministic_random_mask,
    deterministic_topk_mask,
    normalize_update_on_support,
    support_digest,
    support_size,
)
from .statistics import paired_comparison, summarize_sample

SYNTHETIC_LABEL: Final = "SYNTHETIC_MATHEMATICAL_FEASIBILITY_ONLY"
SYNTHETIC_DISCLAIMER: Final = (
    "SYNTHETIC MATHEMATICAL FEASIBILITY ONLY; NOT EVIDENCE OF REAL-MODEL "
    "OR LONG-CONTEXT IMPROVEMENT."
)
REQUIRED_SECTION16_ARMS: Final[tuple[str, ...]] = (
    "no_update",
    "random_support",
    "target_only_support",
    "gradient_topk",
    "coordinate_shuffled_peers",
    "real_pcsm_peers",
    "dense_noisy_update",
    "oracle_support_update",
)
MATCHED_SPARSE_ARMS: Final[tuple[str, ...]] = (
    "random_support",
    "target_only_support",
    "gradient_topk",
    "coordinate_shuffled_peers",
    "real_pcsm_peers",
)
OPTIONAL_ABLATION_ARMS: Final[tuple[str, ...]] = (
    "magnitude_support",
    "real_pcsm_peers_no_stabilizer",
)
LEGACY_ARM_ALIASES: Final[dict[str, str]] = {
    "dense_adamw": "dense_noisy_update",
    "random_mask": "random_support",
    "magnitude_mask": "magnitude_support",
    "gradient_mask": "gradient_topk",
    "shuffled_peer_pcsm": "coordinate_shuffled_peers",
    "no_peer_pcsm": "target_only_support",
    "pcsm_no_stabilizer": "real_pcsm_peers_no_stabilizer",
    "real_peer_pcsm": "real_pcsm_peers",
}
_KNOWN_ARMS: Final = frozenset(REQUIRED_SECTION16_ARMS + OPTIONAL_ABLATION_ARMS)
_TRUE_STRUCTURAL_WEIGHTS: Final[NDArray[np.float32]] = np.asarray(
    (0.68, 0.18, 0.09, 0.05), dtype=np.float32
)


def _sequence_value(raw: Mapping[str, object], key: str) -> Sequence[object]:
    value = raw[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{key} must be a sequence")
    return value


def _integer_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return converted


def _float_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _canonical_arm(name: str) -> str:
    return LEGACY_ARM_ALIASES.get(name, name)


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Validated configuration for the controlled Section 16 experiment."""

    label: str
    seeds: tuple[int, ...]
    matrix_shape: tuple[int, int]
    latent_rank: int
    noise_standard_deviation: float
    budget_fractions: tuple[float, ...]
    steps: int
    learning_rate: float
    arms: tuple[str, ...]
    peer_count: int = 6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    stabilizer_strength: float = 0.30

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        """Parse YAML/JSON-shaped data, accepting audited legacy arm aliases."""

        seeds = tuple(
            _integer_value(value, "seed") for value in _sequence_value(raw, "seeds")
        )
        shape_values = tuple(
            _integer_value(value, "matrix_shape entry")
            for value in _sequence_value(raw, "matrix_shape")
        )
        if len(shape_values) != 2:
            raise ValueError("matrix_shape must have exactly two dimensions")
        budgets = tuple(
            _float_value(value, "budget fraction")
            for value in _sequence_value(raw, "budget_fractions")
        )
        config = cls(
            label=str(raw["label"]),
            seeds=seeds,
            matrix_shape=(shape_values[0], shape_values[1]),
            latent_rank=_integer_value(raw["latent_rank"], "latent_rank"),
            noise_standard_deviation=_float_value(
                raw["noise_standard_deviation"], "noise_standard_deviation"
            ),
            budget_fractions=budgets,
            steps=_integer_value(raw["steps"], "steps"),
            learning_rate=_float_value(raw["learning_rate"], "learning_rate"),
            arms=tuple(str(value) for value in _sequence_value(raw, "arms")),
            peer_count=_integer_value(raw.get("peer_count", 6), "peer_count"),
            weight_decay=_float_value(raw.get("weight_decay", 0.01), "weight_decay"),
            beta1=_float_value(raw.get("beta1", 0.9), "beta1"),
            beta2=_float_value(raw.get("beta2", 0.999), "beta2"),
            epsilon=_float_value(raw.get("epsilon", 1e-8), "epsilon"),
            stabilizer_strength=_float_value(
                raw.get("stabilizer_strength", 0.30), "stabilizer_strength"
            ),
        )
        config.validate()
        return config

    @property
    def resolved_arms(self) -> tuple[str, ...]:
        """Return canonical A--H arms followed by declared extra ablations."""

        canonical = tuple(_canonical_arm(name) for name in self.arms)
        extras = tuple(
            name for name in canonical if name not in REQUIRED_SECTION16_ARMS
        )
        return REQUIRED_SECTION16_ARMS + extras

    def validate(self) -> None:
        if self.label != SYNTHETIC_LABEL:
            raise ValueError(f"label must be exactly {SYNTHETIC_LABEL!r}")
        if len(self.seeds) < 5 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("at least five distinct deterministic seeds are required")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        rows, columns = self.matrix_shape
        if rows <= 0 or columns <= 0:
            raise ValueError("matrix dimensions must be positive")
        if not 1 <= self.latent_rank <= min(self.matrix_shape):
            raise ValueError("latent_rank must fit inside both matrix dimensions")
        if self.noise_standard_deviation <= 0.0:
            raise ValueError("noise_standard_deviation must be positive")
        if not self.budget_fractions or any(
            not 0.0 < fraction <= 1.0 for fraction in self.budget_fractions
        ):
            raise ValueError("budget fractions must be non-empty and in (0, 1]")
        if len(set(self.budget_fractions)) != len(self.budget_fractions):
            raise ValueError("budget fractions must be distinct")
        if self.steps < 2 or self.learning_rate <= 0.0:
            raise ValueError("steps must be at least two and learning_rate positive")
        if self.peer_count < 2:
            raise ValueError("peer_count must be at least two")
        if not 0.0 <= self.weight_decay < 1.0:
            raise ValueError("weight_decay must be in [0, 1)")
        if not 0.0 < self.beta1 < 1.0 or not 0.0 < self.beta2 < 1.0:
            raise ValueError("Adam beta values must be in (0, 1)")
        if self.epsilon <= 0.0 or self.stabilizer_strength < 0.0:
            raise ValueError(
                "epsilon must be positive and stabilizer strength non-negative"
            )
        if not self.arms:
            raise ValueError("arms must be non-empty")
        canonical = tuple(_canonical_arm(name) for name in self.arms)
        if len(set(canonical)) != len(canonical):
            raise ValueError("arms must remain distinct after legacy alias resolution")
        unknown = sorted(set(canonical) - _KNOWN_ARMS)
        if unknown:
            valid = ", ".join(sorted(_KNOWN_ARMS))
            raise ValueError(
                f"unknown control arm {unknown[0]!r}; expected one of: {valid}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "seeds": list(self.seeds),
            "matrix_shape": list(self.matrix_shape),
            "latent_rank": self.latent_rank,
            "noise_standard_deviation": self.noise_standard_deviation,
            "budget_fractions": list(self.budget_fractions),
            "steps": self.steps,
            "steps_meaning": "meta_training_steps; every arm makes at most one update",
            "learning_rate": self.learning_rate,
            "declared_arms": list(self.arms),
            "arms": list(self.resolved_arms),
            "peer_count": self.peer_count,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "stabilizer_strength": self.stabilizer_strength,
        }


@dataclass(frozen=True, slots=True)
class SyntheticProblem:
    """One held-out structured task plus independently generated meta tasks."""

    seed: int
    target: NDArray[np.float64]
    initial: NDArray[np.float64]
    peers: NDArray[np.float64]
    calibration_gradient: NDArray[np.float64]
    step_gradient_noise: NDArray[np.float64]
    hidden_support_score: NDArray[np.float64]
    hidden_update_direction: NDArray[np.float64]
    meta_feature_weights: NDArray[np.float32]
    meta_training_loss_curve: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _BudgetTask:
    k: int
    oracle_support: NDArray[np.bool_]
    oracle_delta: NDArray[np.float64]
    optimum: NDArray[np.float64]
    observed_gradient: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """A fixed support and post-normalization stabilizer generated once."""

    support: NDArray[np.bool_]
    stabilizer: NDArray[np.float64]
    peer_attention: tuple[float, ...]
    score_mean: float
    score_standard_deviation: float
    effective_stabilizer_strength: float
    comparison_class: str
    fairness_matched: bool
    declared_support_k: int


@dataclass(frozen=True, slots=True)
class SyntheticRunRecord:
    """Raw per-seed evidence for one named arm and one requested budget."""

    seed: int
    arm: str
    section16_required_arm: bool
    comparison_class: str
    fairness_matched: bool
    requested_budget_fraction: float
    effective_budget_fraction: float
    parameter_count: int
    declared_support_k: int
    support_k: int
    support_union_k: int
    support_cardinality_exact: bool
    changed_parameter_count: int
    support_sha256: str
    exact_frozen_coordinates: bool
    changed_frozen_coordinate_count: int
    update_count: int
    initial_loss: float
    final_loss: float
    improvement: float
    relative_improvement: float
    oracle_support_recall: float
    per_step_update_l2: tuple[float, ...]
    target_per_step_update_l2: float
    loss_curve: tuple[float, ...]
    peer_attention: tuple[float, ...]
    plan_score_mean: float
    plan_score_standard_deviation: float
    effective_stabilizer_strength: float
    meta_training_initial_loss: float
    meta_training_final_loss: float
    meta_training_loss_decreased: bool
    meta_training_loss_curve: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_label": SYNTHETIC_LABEL,
            "disclaimer": SYNTHETIC_DISCLAIMER,
            "real_model_improvement_claim": False,
            "seed": self.seed,
            "arm": self.arm,
            "section16_required_arm": self.section16_required_arm,
            "comparison_class": self.comparison_class,
            "fairness_matched": self.fairness_matched,
            "requested_budget_fraction": self.requested_budget_fraction,
            "effective_budget_fraction": self.effective_budget_fraction,
            "parameter_count": self.parameter_count,
            "declared_support_k": self.declared_support_k,
            "support_k": self.support_k,
            "support_union_k": self.support_union_k,
            "support_cardinality_exact": self.support_cardinality_exact,
            "changed_parameter_count": self.changed_parameter_count,
            "support_sha256": self.support_sha256,
            "exact_frozen_coordinates": self.exact_frozen_coordinates,
            "changed_frozen_coordinate_count": self.changed_frozen_coordinate_count,
            "update_count": self.update_count,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improvement": self.improvement,
            "relative_improvement": self.relative_improvement,
            "oracle_support_recall": self.oracle_support_recall,
            "per_step_update_l2": list(self.per_step_update_l2),
            "target_per_step_update_l2": self.target_per_step_update_l2,
            "loss_curve": list(self.loss_curve),
            "peer_attention": list(self.peer_attention),
            "plan_score_mean": self.plan_score_mean,
            "plan_score_standard_deviation": self.plan_score_standard_deviation,
            "effective_stabilizer_strength": self.effective_stabilizer_strength,
            "meta_training_initial_loss": self.meta_training_initial_loss,
            "meta_training_final_loss": self.meta_training_final_loss,
            "meta_training_loss_decreased": self.meta_training_loss_decreased,
            "meta_training_loss_curve": list(self.meta_training_loss_curve),
        }


def _standardize(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    values = np.asarray(matrix, dtype=np.float32)
    centered = values - np.mean(values, dtype=np.float32)
    scale = float(np.std(centered, dtype=np.float32))
    if scale <= np.finfo(np.float32).eps:
        return np.zeros_like(centered)
    return np.asarray(centered / np.float32(scale), dtype=np.float32)


def _latent_matrix(
    rng: np.random.Generator,
    shape: tuple[int, int],
    rank: int,
) -> NDArray[np.float64]:
    rows, columns = shape
    left = rng.normal(size=(rows, rank))
    right = rng.normal(size=(columns, rank))
    singular_scales = np.geomspace(1.0, 0.25, num=rank)
    matrix = (left * singular_scales) @ right.T / np.sqrt(float(rank))
    standardized = _standardize(np.asarray(matrix, dtype=np.float32))
    return np.asarray(standardized, dtype=np.float64)


def _structured_latent(
    rng: np.random.Generator,
    config: SyntheticConfig,
) -> NDArray[np.float64]:
    rows, columns = config.matrix_shape
    core = _latent_matrix(rng, config.matrix_shape, config.latent_rank)
    row_axis = np.linspace(-np.pi, np.pi, rows, dtype=np.float64)[:, None]
    column_axis = np.linspace(-np.pi, np.pi, columns, dtype=np.float64)[None, :]
    structure = 0.28 * np.sin(row_axis) + 0.22 * np.cos(column_axis)
    return np.asarray(0.72 * core + structure, dtype=np.float64)


def _noisy_peers(
    latent: NDArray[np.float64],
    rng: np.random.Generator,
    config: SyntheticConfig,
) -> NDArray[np.float64]:
    peers: list[NDArray[np.float64]] = []
    for _ in range(config.peer_count):
        low_rank_noise = _latent_matrix(rng, config.matrix_shape, config.latent_rank)
        independent_noise = rng.normal(size=config.matrix_shape)
        noise = config.noise_standard_deviation * (
            0.30 * low_rank_noise + 0.70 * independent_noise
        )
        peers.append(np.asarray(latent + noise, dtype=np.float64))
    return np.stack(peers, axis=0).astype(np.float64, copy=False)


def _property_maps(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return standardized element, row, column, and joint structural maps."""

    values = np.asarray(matrix, dtype=np.float32)
    centered = values - np.mean(values, dtype=np.float32)
    epsilon = np.float32(np.finfo(np.float32).eps)
    rms = np.sqrt(np.mean(centered * centered, dtype=np.float32) + epsilon)
    element = np.abs(centered) / rms
    row = np.sqrt(np.mean(values * values, axis=1, keepdims=True) + epsilon)
    row = np.broadcast_to(
        row / (np.mean(row, dtype=np.float32) + epsilon), values.shape
    )
    column = np.sqrt(np.mean(values * values, axis=0, keepdims=True) + epsilon)
    column = np.broadcast_to(
        column / (np.mean(column, dtype=np.float32) + epsilon), values.shape
    )
    joint = row * column
    return np.stack(
        tuple(_standardize(item) for item in (element, row, column, joint)), axis=0
    ).astype(np.float32, copy=False)


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float32]:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / np.sum(exponentials), dtype=np.float32)


def _peer_features(
    peers: NDArray[np.float64],
) -> tuple[NDArray[np.float32], tuple[float, ...]]:
    """Aggregate same-role peer maps with transparent centroid similarity."""

    flattened = np.asarray(peers, dtype=np.float32).reshape(peers.shape[0], -1)
    centered = flattened - np.mean(flattened, axis=1, keepdims=True)
    centroid = np.mean(centered, axis=0)
    centroid_norm = float(np.linalg.vector_norm(centroid))
    logits: list[float] = []
    for peer in centered:
        denominator = float(np.linalg.vector_norm(peer)) * centroid_norm
        similarity = (
            0.0
            if denominator <= np.finfo(np.float32).eps
            else float(np.dot(peer, centroid) / denominator)
        )
        logits.append(similarity / 0.20)
    attention = _softmax(np.asarray(logits, dtype=np.float32))
    maps = np.stack(tuple(_property_maps(peer) for peer in peers), axis=0)
    aggregate = np.tensordot(attention, maps, axes=(0, 0)).astype(
        np.float32, copy=False
    )
    return aggregate, tuple(float(value) for value in attention)


def _hidden_score_from_peer_features(
    features: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Construct the hidden optimum score directly from peer property maps."""

    score = np.tensordot(_TRUE_STRUCTURAL_WEIGHTS, features, axes=(0, 0))
    return np.asarray(_standardize(score), dtype=np.float64)


def _meta_training_data(
    rng: np.random.Generator,
    config: SyntheticConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    feature_rows: list[NDArray[np.float32]] = []
    targets: list[NDArray[np.float32]] = []
    for _ in range(max(6, config.peer_count)):
        latent = _structured_latent(rng, config)
        peers = _noisy_peers(latent, rng, config)
        features, _ = _peer_features(peers)
        feature_rows.append(np.moveaxis(features, 0, -1).reshape(-1, 4))
        targets.append(
            np.asarray(
                _hidden_score_from_peer_features(features), dtype=np.float32
            ).reshape(-1)
        )
    return np.concatenate(feature_rows, axis=0), np.concatenate(targets, axis=0)


def _meta_train_generator(
    rng: np.random.Generator,
    config: SyntheticConfig,
) -> tuple[NDArray[np.float32], tuple[float, ...]]:
    """Fit nonnegative feature weights on independent tasks with monotone GD."""

    features, targets = _meta_training_data(rng, config)
    logits: NDArray[np.float32] = np.zeros(4, dtype=np.float32)

    def loss_for(candidate: NDArray[np.float32]) -> float:
        weights = _softmax(candidate)
        errors = features @ weights - targets
        return float(np.mean(errors * errors, dtype=np.float32))

    loss_curve = [loss_for(logits)]
    for _ in range(config.steps):
        weights = _softmax(logits)
        errors = features @ weights - targets
        gradient_weights = np.asarray(
            2.0 * np.mean(errors[:, None] * features, axis=0), dtype=np.float32
        )
        gradient_logits = weights * (
            gradient_weights - np.dot(gradient_weights, weights)
        )
        step_size = np.float32(2.0)
        candidate = logits - step_size * gradient_logits
        candidate_loss = loss_for(candidate)
        while candidate_loss > loss_curve[-1] and step_size > np.float32(1e-5):
            step_size *= np.float32(0.5)
            candidate = logits - step_size * gradient_logits
            candidate_loss = loss_for(candidate)
        logits = np.asarray(candidate, dtype=np.float32)
        loss_curve.append(candidate_loss)
    return _softmax(logits), tuple(loss_curve)


def generate_problem(config: SyntheticConfig, seed: int) -> SyntheticProblem:
    """Generate one held-out task after independent generator meta-training."""

    if seed not in config.seeds:
        raise ValueError("seed must be declared in the experiment configuration")
    evaluation_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC051]))
    meta_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4D37A]))
    meta_weights, meta_curve = _meta_train_generator(meta_rng, config)
    latent = _structured_latent(evaluation_rng, config)
    peers = _noisy_peers(latent, evaluation_rng, config)
    unrelated = _structured_latent(evaluation_rng, config)
    initial = 0.28 * latent + 0.72 * unrelated
    evaluation_peer_features, _ = _peer_features(peers)
    hidden_score = _hidden_score_from_peer_features(evaluation_peer_features)
    hidden_direction = np.where(latent >= np.median(latent), 1.0, -1.0)
    gradient_noise = evaluation_rng.normal(size=config.matrix_shape)
    return SyntheticProblem(
        seed=seed,
        target=latent,
        initial=np.asarray(initial, dtype=np.float64),
        peers=peers,
        calibration_gradient=np.asarray(gradient_noise, dtype=np.float64),
        step_gradient_noise=np.asarray(gradient_noise[None, :, :], dtype=np.float64),
        hidden_support_score=hidden_score,
        hidden_update_direction=np.asarray(hidden_direction, dtype=np.float64),
        meta_feature_weights=meta_weights,
        meta_training_loss_curve=meta_curve,
    )


def _budget_task(
    problem: SyntheticProblem,
    config: SyntheticConfig,
    requested_fraction: float,
) -> _BudgetTask:
    n_parameters = problem.initial.size
    k = support_size(n_parameters, requested_fraction)
    oracle_support = deterministic_topk_mask(problem.hidden_support_score, k)
    active_score = problem.hidden_support_score[oracle_support]
    low = float(np.min(active_score))
    spread = max(float(np.max(active_score)) - low, np.finfo(np.float64).eps)
    scaled_score = np.zeros_like(problem.hidden_support_score)
    scaled_score[oracle_support] = (active_score - low) / spread
    oracle_delta = np.zeros_like(problem.initial)
    oracle_delta[oracle_support] = (
        config.learning_rate
        * (2.4 + 0.8 * scaled_score[oracle_support])
        * problem.hidden_update_direction[oracle_support]
    )
    optimum = problem.initial + oracle_delta
    observed_gradient = -oracle_delta + (
        1.35 * config.noise_standard_deviation * problem.calibration_gradient
    )
    return _BudgetTask(
        k=k,
        oracle_support=oracle_support,
        oracle_delta=oracle_delta,
        optimum=optimum,
        observed_gradient=np.asarray(observed_gradient, dtype=np.float64),
    )


def _shuffle_peer_coordinates(
    peers: NDArray[np.float64], seed: int
) -> NDArray[np.float64]:
    shuffled = np.empty_like(peers)
    for peer_index in range(peers.shape[0]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5A17, peer_index]))
        permutation = rng.permutation(peers[peer_index].size)
        shuffled[peer_index] = (
            peers[peer_index].reshape(-1)[permutation].reshape(peers.shape[1:])
        )
    return shuffled


def _stabilizer_from_quality(
    quality: NDArray[np.float64],
    support: NDArray[np.bool_],
    strength: float,
) -> NDArray[np.float64]:
    stabilizer = np.ones_like(quality, dtype=np.float64)
    active_quality = quality[support]
    if active_quality.size == 0 or strength == 0.0:
        return stabilizer
    centered = active_quality - float(np.mean(active_quality))
    scale = float(np.std(centered))
    if scale <= np.finfo(np.float64).eps:
        return stabilizer
    raw = np.exp(strength * np.tanh(centered / scale))
    raw = np.clip(raw, 0.40, 2.50)
    stabilizer[support] = raw / float(np.sqrt(np.mean(raw * raw)))
    return stabilizer


def make_arm_plan(
    problem: SyntheticProblem,
    config: SyntheticConfig,
    arm: str,
    requested_fraction: float,
) -> ArmPlan:
    """Generate a fixed support once for one canonical Section 16 arm."""

    canonical = _canonical_arm(arm)
    if canonical not in _KNOWN_ARMS:
        valid = ", ".join(sorted(_KNOWN_ARMS))
        raise ValueError(f"unknown control arm {arm!r}; expected one of: {valid}")
    task = _budget_task(problem, config, requested_fraction)
    n_parameters = problem.initial.size
    attention: tuple[float, ...] = ()
    fairness_matched = canonical in MATCHED_SPARSE_ARMS or canonical in {
        "magnitude_support",
        "real_pcsm_peers_no_stabilizer",
    }
    comparison_class = "matched_sparse_K_and_update_l2"
    k = task.k

    if canonical == "no_update":
        score = np.zeros_like(problem.initial)
        support = np.zeros_like(problem.initial, dtype=np.bool_)
        declared_k = 0
        comparison_class = "unmatched_zero_update_reference"
        fairness_matched = False
    elif canonical == "dense_noisy_update":
        score = np.abs(task.observed_gradient)
        support = np.ones_like(problem.initial, dtype=np.bool_)
        declared_k = n_parameters
        comparison_class = "unmatched_dense_reference"
        fairness_matched = False
    elif canonical == "oracle_support_update":
        score = problem.hidden_support_score
        support = task.oracle_support.copy()
        declared_k = k
        comparison_class = "unmatched_privileged_oracle_reference"
        fairness_matched = False
    elif canonical == "random_support":
        score = np.ones_like(problem.initial)
        rng = np.random.default_rng(np.random.SeedSequence([problem.seed, k, 0xA11CE]))
        support = deterministic_random_mask(problem.initial.shape, k, rng)
        declared_k = k
    elif canonical == "target_only_support":
        target_features = _property_maps(problem.initial)
        score = np.asarray(np.mean(target_features, axis=0), dtype=np.float64)
        support = deterministic_topk_mask(score, k)
        declared_k = k
    elif canonical == "gradient_topk":
        score = np.abs(task.observed_gradient)
        support = deterministic_topk_mask(score, k)
        declared_k = k
    elif canonical == "magnitude_support":
        score = np.abs(problem.initial)
        support = deterministic_topk_mask(score, k)
        declared_k = k
    else:
        source_peers = (
            _shuffle_peer_coordinates(problem.peers, problem.seed)
            if canonical == "coordinate_shuffled_peers"
            else problem.peers
        )
        peer_features, attention = _peer_features(source_peers)
        score = np.asarray(
            np.tensordot(problem.meta_feature_weights, peer_features, axes=(0, 0)),
            dtype=np.float64,
        )
        support = deterministic_topk_mask(score, k)
        declared_k = k

    stabilizer = np.ones_like(problem.initial, dtype=np.float64)
    effective_strength = 0.0
    if canonical in {"coordinate_shuffled_peers", "real_pcsm_peers"}:
        effective_strength = config.stabilizer_strength * (1.0 - k / n_parameters)
        stabilizer = _stabilizer_from_quality(score, support, effective_strength)
    return ArmPlan(
        support=support,
        stabilizer=stabilizer,
        peer_attention=attention,
        score_mean=float(np.mean(score)),
        score_standard_deviation=float(np.std(score)),
        effective_stabilizer_strength=effective_strength,
        comparison_class=comparison_class,
        fairness_matched=fairness_matched,
        declared_support_k=declared_k,
    )


def _quadratic_loss(weights: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    difference = weights - target
    return 0.5 * float(np.mean(difference * difference))


def _changed_coordinate_count(
    before: NDArray[np.float64], after: NDArray[np.float64]
) -> int:
    byte_dtype = np.dtype((np.void, before.dtype.itemsize))
    before_bits = np.ascontiguousarray(before).view(byte_dtype).reshape(-1)
    after_bits = np.ascontiguousarray(after).view(byte_dtype).reshape(-1)
    return int(np.count_nonzero(before_bits != after_bits))


def run_arm(
    problem: SyntheticProblem,
    config: SyntheticConfig,
    arm: str,
    requested_fraction: float,
) -> SyntheticRunRecord:
    """Make at most one update, then perform exact-support byte audits."""

    canonical = _canonical_arm(arm)
    task = _budget_task(problem, config, requested_fraction)
    plan = make_arm_plan(problem, config, canonical, requested_fraction)
    support = plan.support
    k = int(np.count_nonzero(support))
    weights = problem.initial.copy()
    initial_loss = _quadratic_loss(weights, task.optimum)
    step_update_norms: tuple[float, ...] = ()
    update_count = 0
    target_step_l2 = 0.0

    if canonical == "oracle_support_update":
        update = task.oracle_delta.copy()
        weights[support] += update[support]
        target_step_l2 = float(np.linalg.vector_norm(update[support]))
        step_update_norms = (target_step_l2,)
        update_count = 1
    elif canonical != "no_update":
        target_step_l2 = config.learning_rate * np.sqrt(float(k))
        normalized_gradient = np.zeros_like(weights)
        normalized_gradient[support] = np.sign(task.observed_gradient[support])
        raw_update = np.zeros_like(weights)
        raw_update[support] = -config.learning_rate * (
            plan.stabilizer[support] * normalized_gradient[support]
            + config.weight_decay * weights[support]
        )
        update = normalize_update_on_support(raw_update, support, target_step_l2)
        weights[support] += update[support]
        step_update_norms = (float(np.linalg.vector_norm(update[support])),)
        update_count = 1

    frozen_audit = bitwise_frozen_audit(problem.initial, weights, support)
    final_loss = _quadratic_loss(weights, task.optimum)
    improvement = initial_loss - final_loss
    oracle_recall = (
        0.0
        if k == 0
        else float(np.count_nonzero(support & task.oracle_support) / task.k)
    )
    meta_curve = problem.meta_training_loss_curve
    changed = _changed_coordinate_count(problem.initial, weights)
    return SyntheticRunRecord(
        seed=problem.seed,
        arm=canonical,
        section16_required_arm=canonical in REQUIRED_SECTION16_ARMS,
        comparison_class=plan.comparison_class,
        fairness_matched=plan.fairness_matched,
        requested_budget_fraction=requested_fraction,
        effective_budget_fraction=k / problem.initial.size,
        parameter_count=problem.initial.size,
        declared_support_k=plan.declared_support_k,
        support_k=k,
        support_union_k=k,
        support_cardinality_exact=k == plan.declared_support_k,
        changed_parameter_count=changed,
        support_sha256=support_digest(support),
        exact_frozen_coordinates=frozen_audit.exact,
        changed_frozen_coordinate_count=frozen_audit.changed_frozen_coordinate_count,
        update_count=update_count,
        initial_loss=initial_loss,
        final_loss=final_loss,
        improvement=improvement,
        relative_improvement=(improvement / initial_loss if initial_loss else 0.0),
        oracle_support_recall=oracle_recall,
        per_step_update_l2=step_update_norms,
        target_per_step_update_l2=target_step_l2,
        loss_curve=(initial_loss, final_loss),
        peer_attention=plan.peer_attention,
        plan_score_mean=plan.score_mean,
        plan_score_standard_deviation=plan.score_standard_deviation,
        effective_stabilizer_strength=plan.effective_stabilizer_strength,
        meta_training_initial_loss=meta_curve[0],
        meta_training_final_loss=meta_curve[-1],
        meta_training_loss_decreased=meta_curve[-1] < meta_curve[0],
        meta_training_loss_curve=meta_curve,
    )


def run_experiment(config: SyntheticConfig) -> list[SyntheticRunRecord]:
    """Run every canonical arm/budget on identical paired per-seed tasks."""

    config.validate()
    records: list[SyntheticRunRecord] = []
    for seed in config.seeds:
        problem = generate_problem(config, seed)
        for budget in config.budget_fractions:
            for arm in config.resolved_arms:
                records.append(run_arm(problem, config, arm, budget))
    return records


def _matched_budget_audit(
    records: Sequence[SyntheticRunRecord], config: SyntheticConfig
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    expected = tuple(arm for arm in config.resolved_arms if arm in MATCHED_SPARSE_ARMS)
    for seed in config.seeds:
        for budget in config.budget_fractions:
            group = [
                record
                for record in records
                if record.seed == seed
                and record.requested_budget_fraction == budget
                and record.arm in expected
            ]
            support_sizes = {record.support_k for record in group}
            norm_match = all(
                len(record.per_step_update_l2) == 1
                and np.isclose(
                    record.per_step_update_l2[0],
                    record.target_per_step_update_l2,
                    rtol=1e-12,
                    atol=1e-12,
                )
                for record in group
            )
            if len(group) != len(expected) or len(support_sizes) != 1 or not norm_match:
                failures.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "record_count": len(group),
                        "expected_record_count": len(expected),
                        "support_k_values": sorted(support_sizes),
                        "per_step_norms_match": norm_match,
                    }
                )
    return {
        "definition": (
            "Required sparse arms B--F use identical K and identical one-step "
            "parameter-space update L2 for each paired seed and budget."
        ),
        "matched_arms": list(expected),
        "explicitly_unmatched": {
            "dense_noisy_update": "dense full-support reference; not K-matched",
            "oracle_support_update": (
                "privileged clean support/update reference; not update-budget-matched"
            ),
        },
        "passed": not failures,
        "failures": failures,
    }


def _required_arm_audit(
    records: Sequence[SyntheticRunRecord], config: SyntheticConfig
) -> dict[str, object]:
    expected_per_arm = len(config.seeds) * len(config.budget_fractions)
    counts = {
        arm: sum(record.arm == arm for record in records)
        for arm in REQUIRED_SECTION16_ARMS
    }
    return {
        "required_arms": list(REQUIRED_SECTION16_ARMS),
        "expected_records_per_arm": expected_per_arm,
        "records_per_arm": counts,
        "passed": all(count == expected_per_arm for count in counts.values()),
    }


def _meta_training_audit(
    records: Sequence[SyntheticRunRecord], config: SyntheticConfig
) -> dict[str, object]:
    per_seed: list[dict[str, object]] = []
    for seed in config.seeds:
        record = next(record for record in records if record.seed == seed)
        per_seed.append(
            {
                "seed": seed,
                "initial_loss": record.meta_training_initial_loss,
                "final_loss": record.meta_training_final_loss,
                "decreased": record.meta_training_loss_decreased,
                "loss_curve": list(record.meta_training_loss_curve),
            }
        )
    return {
        "definition": (
            "Generator feature weights are fit on independent synthetic meta tasks; "
            "the held-out evaluation matrix is never used for meta-training."
        ),
        "per_seed": per_seed,
        "all_seeds_decreased": all(bool(item["decreased"]) for item in per_seed),
        "mean_initial_loss": float(
            np.mean([float(cast(float, item["initial_loss"])) for item in per_seed])
        ),
        "mean_final_loss": float(
            np.mean([float(cast(float, item["final_loss"])) for item in per_seed])
        ),
    }


def _decisive_checks(
    records: Sequence[SyntheticRunRecord],
    config: SyntheticConfig,
) -> dict[str, object]:
    paired_keys = [
        (seed, budget) for seed in config.seeds for budget in config.budget_fractions
    ]
    lookup = {
        (record.seed, record.requested_budget_fraction, record.arm): record
        for record in records
    }

    def mean_metric(arm: str, metric: str) -> float:
        return float(
            np.mean(
                [getattr(lookup[(*key, arm)], metric) for key in paired_keys],
                dtype=np.float64,
            )
        )

    real_loss = mean_metric("real_pcsm_peers", "final_loss")
    random_loss = mean_metric("random_support", "final_loss")
    gradient_loss = mean_metric("gradient_topk", "final_loss")
    shuffled_loss = mean_metric("coordinate_shuffled_peers", "final_loss")
    real_recall = mean_metric("real_pcsm_peers", "oracle_support_recall")
    random_recall = mean_metric("random_support", "oracle_support_recall")
    meta_audit = _meta_training_audit(records, config)
    checks: dict[str, dict[str, object]] = {
        "loss_real_peers_lt_random": {
            "real": real_loss,
            "control": random_loss,
            "passed": real_loss < random_loss,
        },
        "loss_real_peers_lt_gradient_topk": {
            "real": real_loss,
            "control": gradient_loss,
            "passed": real_loss < gradient_loss,
        },
        "loss_real_peers_lt_coordinate_shuffled_peers": {
            "real": real_loss,
            "control": shuffled_loss,
            "passed": real_loss < shuffled_loss,
        },
        "recall_real_peers_gt_random": {
            "real": real_recall,
            "control": random_recall,
            "passed": real_recall > random_recall,
        },
        "meta_training_loss_decreases": {
            "mean_initial": meta_audit["mean_initial_loss"],
            "mean_final": meta_audit["mean_final_loss"],
            "all_seeds_decreased": meta_audit["all_seeds_decreased"],
            "passed": bool(meta_audit["all_seeds_decreased"]),
        },
        "at_least_five_independent_seeds": {
            "seed_count": len(config.seeds),
            "passed": len(config.seeds) >= 5,
        },
    }
    return {
        "aggregation": (
            "paired mean over all declared seeds and budgets; full-support ties at "
            "budget 1.0 are retained rather than discarded"
        ),
        "checks": checks,
        "all_passed": all(bool(item["passed"]) for item in checks.values()),
    }


def summarize_experiment(
    records: Sequence[SyntheticRunRecord],
    config: SyntheticConfig,
    *,
    bootstrap_resamples: int = 10_000,
    statistics_seed: int = 20260826,
) -> dict[str, object]:
    """Create Section 16 summaries, paired effects, and explicit non-claims."""

    expected_records = (
        len(config.seeds) * len(config.budget_fractions) * len(config.resolved_arms)
    )
    if len(records) != expected_records:
        raise ValueError(
            f"expected {expected_records} records from the configuration; "
            f"got {len(records)}"
        )

    arm_summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for budget_index, budget in enumerate(config.budget_fractions):
        for arm in config.resolved_arms:
            group = sorted(
                (
                    record
                    for record in records
                    if record.arm == arm and record.requested_budget_fraction == budget
                ),
                key=lambda record: record.seed,
            )
            arm_summaries.append(
                {
                    "requested_budget_fraction": budget,
                    "arm": arm,
                    "comparison_class": group[0].comparison_class,
                    "fairness_matched": group[0].fairness_matched,
                    "final_loss": summarize_sample(
                        [record.final_loss for record in group]
                    ),
                    "improvement": summarize_sample(
                        [record.improvement for record in group]
                    ),
                    "oracle_support_recall": summarize_sample(
                        [record.oracle_support_recall for record in group]
                    ),
                    "all_support_cardinality_audits_pass": all(
                        record.support_cardinality_exact for record in group
                    ),
                    "all_exact_frozen_coordinate_audits_pass": all(
                        record.exact_frozen_coordinates for record in group
                    ),
                }
            )

        real_by_seed = {
            record.seed: record
            for record in records
            if record.arm == "real_pcsm_peers"
            and record.requested_budget_fraction == budget
        }
        for control_index, control in enumerate(config.resolved_arms):
            if control == "real_pcsm_peers":
                continue
            control_by_seed = {
                record.seed: record
                for record in records
                if record.arm == control and record.requested_budget_fraction == budget
            }
            paired_seeds = sorted(set(real_by_seed) & set(control_by_seed))
            comparison = paired_comparison(
                [control_by_seed[seed].final_loss for seed in paired_seeds],
                [real_by_seed[seed].final_loss for seed in paired_seeds],
                bootstrap_resamples=bootstrap_resamples,
                seed=statistics_seed + 100 * budget_index + control_index,
            )
            comparisons.append(
                {
                    "requested_budget_fraction": budget,
                    "first_arm": control,
                    "second_arm": "real_pcsm_peers",
                    "metric": "final_loss",
                    "positive_difference_favors": "real_pcsm_peers",
                    "paired_seeds": paired_seeds,
                    **comparison,
                }
            )

    required_audit = _required_arm_audit(records, config)
    matched_audit = _matched_budget_audit(records, config)
    meta_audit = _meta_training_audit(records, config)
    decisive = _decisive_checks(records, config)
    frozen_pass = all(record.exact_frozen_coordinates for record in records)
    cardinality_pass = all(record.support_cardinality_exact for record in records)
    one_update_pass = all(
        record.update_count == (0 if record.arm == "no_update" else 1)
        for record in records
    )
    return {
        "evidence_label": SYNTHETIC_LABEL,
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "claim_boundary": (
            "The implementation can learn and exploit peer-dependent structural "
            "information when it genuinely determines a synthetic hidden sparse "
            "optimum. This says nothing about transformer-layer cooperation."
        ),
        "real_model_improvement_claim": False,
        "advancement_decision": "NOT_APPLICABLE_SYNTHETIC_FEASIBILITY_ONLY",
        "section_16_protocol": "A-H one-update controlled feasibility experiment",
        "config": config.to_dict(),
        "synthetic_problem_definition": {
            "latent": "structured low-rank matrix with row/column effects",
            "peers": "independent noisy same-role observations of the latent matrix",
            "peer_maps": "element, row, column, and joint structural properties",
            "hidden_sparse_optimum": (
                "exact TopK of a held-out combination of clean latent structural maps"
            ),
            "observed_gradient": (
                "true sparse gradient plus substantial independent coordinate noise"
            ),
            "optimization": "each nonzero arm makes exactly one update",
            "meta_training": (
                "feature weights trained on independent generated tasks, never the "
                "held-out evaluation matrix"
            ),
        },
        "record_count": len(records),
        "seed_count": len(config.seeds),
        "required_section16_arm_audit": required_audit,
        "one_update_protocol_audit": {
            "definition": "A makes zero updates; B--H each make exactly one update",
            "passed": one_update_pass,
        },
        "exact_support_audit": {
            "all_declared_cardinalities_exact": cardinality_pass,
            "all_support_unions_equal_fixed_support": all(
                record.support_union_k == record.support_k for record in records
            ),
            "all_frozen_coordinates_bitwise_exact": frozen_pass,
            "passed": cardinality_pass and frozen_pass,
        },
        "matched_sparse_update_budget_audit": matched_audit,
        "dense_oracle_fairness_labels": {
            "dense_noisy_update": "UNMATCHED_DENSE_REFERENCE",
            "oracle_support_update": "UNMATCHED_PRIVILEGED_ORACLE_REFERENCE",
        },
        "meta_training_audit": meta_audit,
        "decisive_section16_checks": decisive,
        "arm_summaries": arm_summaries,
        "paired_real_peer_comparisons": comparisons,
    }
