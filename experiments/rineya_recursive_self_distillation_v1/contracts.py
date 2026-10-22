"""Versioned, strict contracts. Public prompts never contain grader sidecars."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BASE_SHA256 = "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7"
PARENT_COMMIT = "5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80"
ARMS = ("sft", "frozen", "evolving", "frozen_srcl", "evolving_srcl", "srcl_only")
SPLITS = ("train", "train_selection", "calibration", "retention_train", "retention_eval")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path, max_bytes: int = 32 * 1024**2) -> Any:
    p = Path(path)
    require(p.is_file() and p.stat().st_size <= max_bytes, "missing/oversized JSON")
    def pairs(items):
        obj = {}
        for k, v in items:
            require(k not in obj, "duplicate JSON key")
            obj[k] = v
        return obj
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=invalid)


@dataclass(frozen=True)
class Problem:
    key: str
    text: str


@dataclass(frozen=True)
class Task:
    key: str
    source: str
    revision: str
    parent: str
    split: str
    prompt: str
    answer: str
    solution: str
    verifier: str
    provenance: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            require(type(value) is str and bool(value.strip()) and len(value) <= 32000,
                    "invalid task field: " + name)
        require(self.split in SPLITS, "sealed/test/challenge/unknown role forbidden")
        require(self.verifier in ("exact-v1", "final-int-v1"), "unsupported verifier")

    def public(self) -> Problem:
        return Problem(self.key, self.prompt)


@dataclass(frozen=True)
class Rollout:
    key: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    text: str
    logps: tuple[float, ...]
    policy_version: int
    terminated: bool

    def __post_init__(self):
        require(bool(self.prompt_ids) and bool(self.response_ids), "empty rollout")
        require(len(self.response_ids) == len(self.logps), "token/logprob alignment")
        require(all(type(x) is int and x >= 0 for x in self.prompt_ids + self.response_ids),
                "invalid token IDs")
        require(all(math.isfinite(x) and x <= 1e-6 for x in self.logps), "invalid logprobs")
        require(type(self.policy_version) is int and self.policy_version >= 0, "invalid policy version")
        require(type(self.terminated) is bool, "termination must be boolean")


@dataclass(frozen=True)
class Plan:
    protocol: str = "RSD-001-v1"
    seed: int = 101
    rounds: int = 32
    batch_tasks: int = 4
    replay_tasks: int = 1
    eval_every: int = 4
    max_new_tokens: int = 128
    max_context_tokens: int = 1024
    chunk_tokens: int = 8
    learning_rate: float = 1e-5
    guidance_weight: float = 1.0
    concise_weight: float = 0.5
    replay_weight: float = 0.1
    max_grad_norm: float = 1.0
    max_retention_drop: float = 0.02
    max_truncation: float = 0.25
    minimum_effect: float = 0.02
    bootstrap_samples: int = 2000
    generated_token_limit: int = 500000
    processed_token_limit: int = 100000000
    model_call_limit: int = 500000
    host_free_gib: float = 7.4
    worker_free_gib: float = 6.0
    gpu_free_gib: float = 4.0
    gpu_worker_free_gib: float = 0.5
    disk_free_gib: float = 4.0
    wall_seconds: int = 1800

    def __post_init__(self):
        require(self.protocol == "RSD-001-v1", "unknown protocol")
        integer_fields = ("seed", "rounds", "batch_tasks", "replay_tasks", "eval_every",
                          "max_new_tokens", "max_context_tokens", "chunk_tokens", "bootstrap_samples",
                          "generated_token_limit", "processed_token_limit", "model_call_limit", "wall_seconds")
        for name in integer_fields:
            value = getattr(self, name)
            require(type(value) is int and value > 0, "invalid integer: " + name)
        for name, value in asdict(self).items():
            if name not in integer_fields and name != "protocol":
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        "invalid finite coefficient: " + name)
        require(self.rounds <= 256 and self.batch_tasks <= 16 and self.replay_tasks <= 4,
                "bounded pilot only; register a new protocol for expansion")
        require(self.max_new_tokens <= 512 and self.max_context_tokens <= 1024
                and self.max_new_tokens < self.max_context_tokens, "unqualified length")
        require(self.chunk_tokens <= 32 and self.learning_rate > 0 and self.max_grad_norm > 0,
                "invalid optimizer/chunk")
        require(self.guidance_weight > 0 and self.concise_weight > 0 and self.replay_weight > 0,
                "positive registered weights required")
        require(0 <= self.max_retention_drop < 1 and 0 <= self.max_truncation < 1
                and 0 < self.minimum_effect < 1, "invalid outcome thresholds")
        require(self.host_free_gib >= 7.4 and self.worker_free_gib >= 6.0
                and self.host_free_gib >= self.worker_free_gib and self.gpu_free_gib >= 4.0
                and self.gpu_worker_free_gib >= .5 and self.disk_free_gib >= 4.0
                and self.wall_seconds <= 1800, "resource guard may not be weakened")
        require(100 <= self.bootstrap_samples <= 20000, "bootstrap budget")

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        return cls(**read_json(path))


def arm_spec(name: str) -> dict:
    require(name in ARMS, "unknown arm")
    return {"arm": name, "teacher": "frozen" if name.startswith("frozen") else
            "evolving" if name.startswith("evolving") else "none",
            "srcl": name.endswith("srcl") or name == "srcl_only", "sft": name == "sft"}


def purpose_seed(seed: int, step: int, key: str, purpose: str) -> int:
    # Independent rewrite RNG never changes the next student's rollout RNG stream.
    return int(digest([seed, step, key, purpose])[:15], 16)
