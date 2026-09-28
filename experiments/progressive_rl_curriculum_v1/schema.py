"""Strict, serializable contracts shared by the controller and its runtimes."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = 1
SPLITS = {"train", "calibration", "retention_train", "retention_eval"}
BANDS = {"unknown", "hard", "frontier", "easy", "mastered"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive(value: int, name: str) -> None:
    require(type(value) is int and value > 0, f"{name} must be a positive integer")


def fraction(value: float, name: str, *, zero: bool = True) -> None:
    require(type(value) in (int, float) and math.isfinite(value)
            and (0 <= value <= 1 if zero else 0 < value <= 1), f"invalid {name}")


@dataclass(frozen=True)
class Task:
    id: str
    source: str
    revision: str
    group: str
    split: str
    prompt: str
    answer: str | None = None
    domain: str = "reasoning"
    verifier: str = "exact-v1"
    requirements: tuple[str, ...] = ("text",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "source", "revision", "group", "prompt", "domain", "verifier"):
            require(isinstance(getattr(self, name), str) and bool(getattr(self, name).strip()),
                    f"missing {name}")
        require(":" not in self.source and ":" not in self.id, "source/id may not contain a colon")
        require(isinstance(self.metadata, dict), "metadata must be an object")
        require(isinstance(self.requirements, (tuple, list)), "requirements must be a sequence")
        require(self.split in SPLITS, "test/challenge/sealed/unknown splits are forbidden")
        require(self.answer is None or isinstance(self.answer, str), "answer must be text")
        require(self.verifier != "exact-v1" or self.answer is not None, "exact task needs answer")
        require(bool(self.requirements) and all(isinstance(x, str) and x for x in self.requirements),
                "requirements must name capabilities")
        canonical(asdict(self))

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    def public(self) -> "Prompt":
        # No gold, grading config, split labels, or private environment metadata to policy.
        return Prompt(self.key, self.prompt)


@dataclass(frozen=True)
class Prompt:
    key: str
    text: str


@dataclass(frozen=True)
class Episode:
    key: str
    text: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    old_logps: tuple[float, ...]
    policy_version: int
    terminated: bool = True
    valid_action: bool = True
    infrastructure_error: str | None = None

    def __post_init__(self) -> None:
        require(bool(self.prompt_ids) and bool(self.response_ids), "empty token sequence")
        require(len(self.response_ids) == len(self.old_logps), "token/logprob mismatch")
        require(all(type(i) is int and i >= 0 for i in (*self.prompt_ids, *self.response_ids)),
                "invalid token IDs")
        require(all(math.isfinite(x) and x <= 1e-6 for x in self.old_logps), "invalid logprobs")
        require(type(self.policy_version) is int and self.policy_version >= 0, "invalid version")


@dataclass(frozen=True)
class Stage:
    name: str
    fraction: float
    source_weights: dict[str, float]
    band_weights: dict[str, float]
    replay_fraction: float = .1
    group_size: int = 4
    batch_tasks: int = 10
    max_new_tokens: int = 512
    max_updates: int = 40
    eval_every: int = 5

    def __post_init__(self) -> None:
        require(bool(self.name), "stage name required")
        fraction(self.fraction, "dataset fraction", zero=False)
        fraction(self.replay_fraction, "replay fraction")
        require(0 < self.replay_fraction < 1, "reserve a nonzero replay share below 1")
        for name in ("group_size", "batch_tasks", "max_new_tokens", "max_updates", "eval_every"):
            positive(getattr(self, name), name)
        require(self.group_size >= 2, "GRPO needs at least two completions")
        require(self.batch_tasks >= 2, "batch must hold acquisition and replay")
        for weights in (self.source_weights, self.band_weights):
            require(isinstance(weights, dict) and all(isinstance(k, str) and k for k in weights), "invalid mixture keys")
            require(bool(weights), "empty mixture")
            for value in weights.values():
                fraction(value, "mixture weight")
            require(math.isclose(sum(weights.values()), 1, abs_tol=1e-8), "mixture must sum to one")
        require(set(self.band_weights) <= BANDS, "unknown difficulty band")
        require(self.band_weights.get("unknown", 0) > 0, "unknown tasks need exploration")


@dataclass(frozen=True)
class Gates:
    min_eval_groups: int = 128
    min_train_groups: int = 16
    min_mixed_fraction: float = .1
    min_valid_action_rate: float = .95
    min_gain_lcb: float = 0.0
    max_retention_drop: float = .02
    max_truncation: float = .10
    consecutive_passes: int = 2
    min_observations: int = 16
    stale_after_updates: int = 5
    zero_signal_patience: int = 3
    bootstrap_samples: int = 1000

    def __post_init__(self) -> None:
        for name in ("min_eval_groups", "min_train_groups", "consecutive_passes",
                     "min_observations", "stale_after_updates", "zero_signal_patience",
                     "bootstrap_samples"):
            positive(getattr(self, name), name)
        for name in ("min_mixed_fraction", "min_valid_action_rate", "max_retention_drop", "max_truncation"):
            fraction(getattr(self, name), name)
        require(math.isfinite(self.min_gain_lcb) and -1 <= self.min_gain_lcb <= 1, "invalid gain")


@dataclass(frozen=True)
class Limits:
    generated_tokens: int = 10_000_000
    processed_tokens: int = 30_000_000
    rollout_calls: int = 100_000
    optimizer_steps: int = 1000
    wall_seconds: int = 86400
    max_prompt_tokens: int = 1024
    max_context_tokens: int = 2048
    host_free_gib: float = 7.4
    worker_free_gib: float = 6.0
    gpu_free_gib: float = 4.0
    disk_free_gib: float = 4.0

    def __post_init__(self) -> None:
        for name in ("generated_tokens", "processed_tokens", "rollout_calls", "optimizer_steps",
                     "wall_seconds", "max_prompt_tokens", "max_context_tokens"):
            positive(getattr(self, name), name)
        for name in ("host_free_gib", "worker_free_gib", "gpu_free_gib", "disk_free_gib"):
            require(math.isfinite(getattr(self, name)) and getattr(self, name) >= 0, f"invalid {name}")
        require(self.host_free_gib >= self.worker_free_gib, "worker guard exceeds preflight")


@dataclass(frozen=True)
class Plan:
    name: str
    stages: tuple[Stage, ...]
    gates: Gates = field(default_factory=Gates)
    limits: Limits = field(default_factory=Limits)
    seed: int = 20260927
    qualification_tasks: int = 32
    learning_rate: float = 1e-5
    kl_beta: float = .02
    clip_epsilon: float = .2
    trainable_patterns: tuple[str, ...] = (".*lora_[AB].*",)
    replay_ce_weight: float = .1
    bootstrap_max_steps: int = 20
    schema: int = SCHEMA

    def __post_init__(self) -> None:
        require(self.schema == SCHEMA and bool(self.name), "unsupported schema or missing name")
        require(bool(self.stages), "no stages")
        require(len({s.name for s in self.stages}) == len(self.stages), "duplicate stages")
        require(list(s.fraction for s in self.stages) == sorted(s.fraction for s in self.stages),
                "unlock fractions must be monotonic")
        for previous, current in zip(self.stages, self.stages[1:]):
            require(set(previous.source_weights) <= set(current.source_weights), "eligible source sets must be nested")
        require(type(self.seed) is int and self.seed >= 0, "invalid seed")
        positive(self.qualification_tasks, "qualification_tasks")
        positive(self.bootstrap_max_steps, "bootstrap_max_steps")
        fraction(self.clip_epsilon, "clip epsilon", zero=False)
        require(self.clip_epsilon < 1, "clip epsilon must be below one")
        for x in (self.learning_rate, self.kl_beta, self.replay_ce_weight):
            require(math.isfinite(x) and x >= 0, "invalid optimization coefficient")
        require(self.learning_rate > 0 and bool(self.trainable_patterns), "missing optimizer policy")
        for stage in self.stages:
            require(self.limits.max_prompt_tokens + stage.max_new_tokens <= self.limits.max_context_tokens,
                    f"stage {stage.name} exceeds qualified context limit")

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Plan":
        obj = dict(value)
        obj["stages"] = tuple(Stage(**x) for x in obj["stages"])
        obj["gates"] = Gates(**obj.get("gates", {}))
        obj["limits"] = Limits(**obj.get("limits", {}))
        obj["trainable_patterns"] = tuple(obj.get("trainable_patterns", (".*lora_[AB].*",)))
        return cls(**obj)  # Unknown keys are errors, never silently ignored.
