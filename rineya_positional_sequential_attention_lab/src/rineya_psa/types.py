"""Shared frozen types. Torch-free by design so CPU analysis stays cheap."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAMILIES: tuple[str, ...] = (
    "exact_retrieval",
    "semantic_single_hop",
    "multi_hop",
    "state_constraint",
)
FORMATS: tuple[str, ...] = ("natural", "shuffled", "list", "json")
LOCAL_POSITIONS: tuple[str, ...] = ("first", "middle", "last", "na")
SPLITS: tuple[str, ...] = ("calibration", "validation", "sealed")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    path: Path
    store_dtype: str
    trust_remote_code: bool
    expected_arch: str
    ungatable: bool


@dataclass(frozen=True)
class Arch:
    """Architecture facts needed for head bookkeeping."""

    num_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int

    @property
    def kv_group_size(self) -> int:
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError(
                f"{self.num_q_heads} query heads do not divide into "
                f"{self.num_kv_heads} KV heads"
            )
        return self.num_q_heads // self.num_kv_heads

    def q_heads_for_kv(self, kv_head: int) -> tuple[int, ...]:
        """Query-head indices sharing KV head ``kv_head``.

        Mirrors ``transformers.models.qwen3.modeling_qwen3.repeat_kv``, which
        expands along a new axis inserted after the KV-head axis, i.e. query head
        ``g`` reads KV head ``g // kv_group_size``.
        """
        if not 0 <= kv_head < self.num_kv_heads:
            raise IndexError(f"kv_head {kv_head} out of range [0,{self.num_kv_heads})")
        g = self.kv_group_size
        return tuple(range(kv_head * g, (kv_head + 1) * g))

    def kv_head_for_q(self, q_head: int) -> int:
        if not 0 <= q_head < self.num_q_heads:
            raise IndexError(f"q_head {q_head} out of range [0,{self.num_q_heads})")
        return q_head // self.kv_group_size


@dataclass(frozen=True)
class PositionGrid:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("empty position grid")
        if any(not 0.0 <= v <= 1.0 for v in self.values):
            raise ValueError(f"positions must lie in [0,1]: {self.values}")
        if list(self.values) != sorted(self.values):
            raise ValueError("position grid must be sorted ascending")

    @property
    def middle(self) -> float:
        return min(self.values, key=lambda v: abs(v - 0.5))

    @property
    def edges(self) -> tuple[float, float]:
        return self.values[0], self.values[-1]


SCREENING_GRID = PositionGrid((0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98))
CALIBRATION_GRID = PositionGrid((0.05, 0.25, 0.50, 0.75, 0.95))


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    scenario_id: str
    family: str
    fmt: str
    length_bucket: str
    target_positions: tuple[float, ...]
    local_position: str
    split: str
    system: str
    user: str
    answer: str
    answer_aliases: tuple[str, ...]
    distractors: tuple[str, ...]
    evidence_sentences: tuple[str, ...]
    seed: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown format {self.fmt!r}")
        if self.local_position not in LOCAL_POSITIONS:
            raise ValueError(f"unknown local_position {self.local_position!r}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "scenario_id": self.scenario_id,
            "family": self.family,
            "fmt": self.fmt,
            "length_bucket": self.length_bucket,
            "target_positions": list(self.target_positions),
            "local_position": self.local_position,
            "split": self.split,
            "system": self.system,
            "user": self.user,
            "answer": self.answer,
            "answer_aliases": list(self.answer_aliases),
            "distractors": list(self.distractors),
            "evidence_sentences": list(self.evidence_sentences),
            "seed": self.seed,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> BenchmarkItem:
        return cls(
            item_id=d["item_id"],
            scenario_id=d["scenario_id"],
            family=d["family"],
            fmt=d["fmt"],
            length_bucket=d["length_bucket"],
            target_positions=tuple(d["target_positions"]),
            local_position=d["local_position"],
            split=d["split"],
            system=d["system"],
            user=d["user"],
            answer=d["answer"],
            answer_aliases=tuple(d.get("answer_aliases", ())),
            distractors=tuple(d.get("distractors", ())),
            evidence_sentences=tuple(d.get("evidence_sentences", ())),
            seed=int(d["seed"]),
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class TokenizedItem:
    item_id: str
    input_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    evidence_spans: tuple[tuple[int, int], ...]
    actual_positions: tuple[float, ...]

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class UShapeFit:
    beta0: float
    beta1: float
    beta2: float
    se0: float
    se1: float
    se2: float
    r_squared: float
    n: int


def split_for_scenario(scenario_id: str) -> str:
    """Deterministic three-way split keyed on the scenario, not the item.

    All positional variants of one source example land in the same split, which is
    what makes the paired statistics in ``statistics/`` legitimate.
    """
    h = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:8], "big") % 100
    if bucket < 40:
        return "calibration"
    if bucket < 70:
        return "validation"
    return "sealed"


def stable_id(*parts: object) -> str:
    """Short deterministic id from arbitrary parts. Used for item_id/config hashes."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def normalize_answer(text: str) -> str:
    """Project-standard answer normalisation. No fuzzy matching."""
    s = " ".join(str(text).strip().split()).casefold()
    while s and s[-1] in ".,;:!?\"')]}":
        s = s[:-1].rstrip()
    while s and s[0] in "\"'([{":
        s = s[1:].lstrip()
    return s


def matches_gold(prediction: str, gold: str, aliases: Sequence[str] = ()) -> bool:
    p = normalize_answer(prediction)
    if not p:
        return False
    targets = {normalize_answer(gold), *(normalize_answer(a) for a in aliases)}
    targets.discard("")
    return p in targets
