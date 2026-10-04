"""RDD-001 immutable plans and bounded, canonical evidence I/O (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


class ContractError(ValueError):
    """An experiment precondition failed; never silently repair an input."""


def require(condition, message: str) -> None:
    if not condition:
        raise ContractError(message)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path, limit: int = 32 * 1024**2):
    p = Path(path)
    require(p.is_file() and p.stat().st_size <= limit, "missing or oversized JSON")
    def reject_pairs(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def reject_constant(value):
        raise ContractError("nonfinite JSON constant: " + value)
    return json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=reject_pairs,
                      parse_constant=reject_constant)


def write_new(path: str | Path, value) -> None:
    p = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: interrupted evidence remains visibly incomplete, never overwritten.
    with p.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()


def sealed(payload: dict) -> dict:
    return {"payload": payload, "sha256": digest(payload)}


def unseal(envelope: dict) -> dict:
    require(type(envelope) is dict and set(envelope) == {"payload", "sha256"}, "envelope fields")
    require(digest(envelope["payload"]) == envelope["sha256"], "evidence checksum mismatch")
    return envelope["payload"]


@dataclass(frozen=True)
class Plan:
    schema: int = 1
    seeds: tuple[int, ...] = (17, 29, 43)
    learning_rates: tuple[float, ...] = (1e-5, 2e-5)
    steps: int = 32
    batch_tasks: int = 2
    replay_tasks: int = 1
    replay_weight: float = 0.1
    max_new_tokens: int = 128
    max_context_tokens: int = 1024
    chunk_tokens: int = 8
    checkpoint_every: int = 8
    eval_samples: int = 4
    sampling_temperature: float = 1.0
    max_grad_norm: float = 1.0
    rank: int = 8
    alpha: float = 16.0
    weight_decay: float = 0.0
    teacher_min_accuracy: float = 0.6
    teacher_min_advantage: float = 0.1
    min_acquisition_gain: float = 0.03
    max_retention_drop: float = 0.02

    def __post_init__(self):
        require(type(self.schema) is int and self.schema == 1, "plan schema")
        require(len(self.seeds) > 0 and len(set(self.seeds)) == len(self.seeds)
                and all(type(s) is int and 0 <= s < 2**31 for s in self.seeds), "independent integer seeds")
        require(len(self.learning_rates) == 2 and self.learning_rates[0] < self.learning_rates[1]
                and all(type(lr) in (int, float) and math.isfinite(lr) and 0 < lr <= .01
                        for lr in self.learning_rates), "exactly two increasing finite learning rates")
        for k in ("steps", "batch_tasks", "max_new_tokens", "max_context_tokens", "chunk_tokens",
                  "checkpoint_every", "eval_samples", "rank"):
            require(type(getattr(self, k)) is int and getattr(self, k) > 0, k + " must be positive integer")
        require(self.steps <= 10000 and self.max_context_tokens <= 32768 and self.eval_samples <= 64,
                "bounded experiment only")
        require(self.max_new_tokens < self.max_context_tokens, "response budget exceeds context")
        require(type(self.replay_tasks) is int and self.replay_tasks >= 0, "replay task count")
        for k in ("replay_weight", "weight_decay", "teacher_min_accuracy", "teacher_min_advantage",
                  "min_acquisition_gain", "max_retention_drop"):
            v = getattr(self, k)
            require(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1, k + " out of range")
        require((self.replay_tasks == 0) == (self.replay_weight == 0), "replay weight/exposure must agree")
        for k in ("sampling_temperature", "alpha", "max_grad_norm"):
            v = getattr(self, k)
            require(type(v) in (int, float) and math.isfinite(v) and v > 0, k + " invalid")

    @classmethod
    def load(cls, path):
        data = read_json(path)
        require(type(data) is dict and not (set(data) - set(cls.__dataclass_fields__)), "unknown plan fields")
        for k in ("seeds", "learning_rates"):
            if k in data:
                require(type(data[k]) is list, k + " must be list")
                data[k] = tuple(data[k])
        return cls(**data)

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Arm:
    name: str
    source: str
    objective: str
    learning_rate: float
    seed: int


def matrix(plan: Plan, stage: str = "full") -> list[Arm]:
    require(stage in {"screen", "full"}, "stage must be screen or full")
    cells = [("A", "teacher", "forward"), ("B", "student", "forward")]
    if stage == "full":
        cells += [("C", "teacher", "reverse"), ("D", "student", "reverse")]
    cells += [("SFT", "solution", "ce")]
    return [Arm(f"{code}-lr{li}-s{seed}", source, loss, lr, seed)
            for seed in plan.seeds for code, source, loss in cells
            for li, lr in enumerate(plan.learning_rates)]


def validate_arm(plan: Plan, arm: Arm):
    require(arm in matrix(plan), "arm is not in frozen factorial")


def step_seed(seed: int, key: str, step: int, purpose: str) -> int:
    return int(digest([seed, key, step, purpose])[:15], 16)
