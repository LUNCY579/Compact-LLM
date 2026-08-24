"""Deterministic B0-B8 baseline policy implementations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..core.counts import CountTransformContext, CountTransformName, transform_counts
from ..core.randomness import stable_rng
from ..core.ranking import Selection, normalized_rank, stable_topk
from ..core.state import SignificanceState, StateInvariantError


class PolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyContext:
    budget: int | NDArray[Any]
    exponent: float = 1.0
    sink_tokens: int = 0
    recent_tokens: int = 0
    current_position: int | None = None
    chunk_size: int | None = None
    previous_counts: NDArray[Any] | None = None
    count_noise_std: float = 0.25
    seed: int = 0
    event_id: str = ""
    oracle_evidence: NDArray[Any] | None = None
    allow_oracle: bool = False
    source_semantics_verified: bool = False


class BaselinePolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def select(self, state: SignificanceState, context: PolicyContext) -> Selection: ...


def _budget_array(context: PolicyContext, state: SignificanceState) -> NDArray[np.int64]:
    value = np.asarray(context.budget, dtype=np.int64)
    if value.ndim == 0:
        value = np.full(state.shape[:2], int(value), dtype=np.int64)
    if value.shape != state.shape[:2] or np.any(value < 0):
        raise PolicyError("budget must be a non-negative scalar or [layers, kv_heads] array")
    return value


def protection_mask(state: SignificanceState, context: PolicyContext) -> NDArray[np.bool_]:
    if context.sink_tokens < 0 or context.recent_tokens < 0:
        raise PolicyError("sink_tokens and recent_tokens must be non-negative")
    result = np.zeros(state.shape, dtype=np.bool_)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            slots = np.flatnonzero(state.active[layer, head])
            ordered = sorted(
                slots.tolist(),
                key=lambda slot: (
                    int(state.positions[layer, head, slot]),
                    int(state.entry_ids[layer, head, slot]),
                ),
            )
            for slot in ordered[: context.sink_tokens]:
                result[layer, head, slot] = True
            if context.recent_tokens:
                for slot in ordered[-context.recent_tokens :]:
                    result[layer, head, slot] = True
    budgets = _budget_array(context, state)
    protected_count = result.sum(axis=-1)
    if np.any(protected_count > budgets):
        location = np.argwhere(protected_count > budgets)[0]
        raise PolicyError(
            "sink/recent protection exceeds budget at "
            f"layer={int(location[0])}, kv_head={int(location[1])}"
        )
    return result


@dataclass(frozen=True, slots=True)
class FullCachePolicy:
    policy_id: str = "B0_full_cache"

    def select(self, state: SignificanceState, context: PolicyContext) -> Selection:
        counts = state.active.sum(axis=-1, dtype=np.int64)
        return stable_topk(
            state,
            np.where(state.active, 0.0, -np.inf),
            counts,
            policy_id=self.policy_id,
        )


@dataclass(frozen=True, slots=True)
class SignificancePolicy:
    policy_id: str
    fixed_exponent: float | None = None
    count_transform: CountTransformName = "true"
    require_source_verification: bool = False

    def select(self, state: SignificanceState, context: PolicyContext) -> Selection:
        if self.require_source_verification and not context.source_semantics_verified:
            raise PolicyError(
                "B1 is disabled until authoritative score/exposure semantics are verified; "
                "use B2/B3 for formula-level experiments"
            )
        count_context = CountTransformContext(
            current_position=context.current_position,
            chunk_size=context.chunk_size,
            previous_counts=context.previous_counts,
            noise_std=context.count_noise_std,
            seed=context.seed,
            event_id=context.event_id,
        )
        counts = transform_counts(state, self.count_transform, count_context)
        exponent = context.exponent if self.fixed_exponent is None else self.fixed_exponent
        rank = normalized_rank(state, exponent, counts=counts)
        return stable_topk(
            state,
            rank,
            context.budget,
            protected=protection_mask(state, context),
            policy_id=self.policy_id,
        )


@dataclass(frozen=True, slots=True)
class PositionalPolicy:
    policy_id: str = "B4_sink_recent"

    def select(self, state: SignificanceState, context: PolicyContext) -> Selection:
        # Recent entries receive larger rank. Sink protection is handled separately.
        rank = np.where(state.active, state.positions.astype(np.float64), -np.inf)
        return stable_topk(
            state,
            rank,
            context.budget,
            protected=protection_mask(state, context),
            policy_id=self.policy_id,
        )


def _shuffle_values(
    state: SignificanceState,
    values: NDArray[np.float64],
    context: PolicyContext,
) -> NDArray[np.float64]:
    shuffled = np.full(state.shape, -np.inf, dtype=np.float64)
    protected = protection_mask(state, context)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            slots = np.flatnonzero(state.active[layer, head] & ~protected[layer, head])
            row = values[layer, head, slots].copy()
            stable_rng(context.seed, context.event_id, layer, head, "shuffle-rank").shuffle(row)
            shuffled[layer, head, slots] = row
            shuffled[layer, head, protected[layer, head]] = values[layer, head, protected[layer, head]]
    return shuffled


@dataclass(frozen=True, slots=True)
class ShuffledSignificancePolicy:
    policy_id: str = "B6_shuffled_significance"

    def select(self, state: SignificanceState, context: PolicyContext) -> Selection:
        rank = normalized_rank(state, context.exponent)
        rank = _shuffle_values(state, rank, context)
        return stable_topk(
            state,
            rank,
            context.budget,
            protected=protection_mask(state, context),
            policy_id=self.policy_id,
        )


@dataclass(frozen=True, slots=True)
class RandomPolicy:
    policy_id: str = "B7_random_matched"

    def select(self, state: SignificanceState, context: PolicyContext) -> Selection:
        rank = np.full(state.shape, -np.inf, dtype=np.float64)
        for layer in range(state.layers):
            for head in range(state.kv_heads):
                slots = np.flatnonzero(state.active[layer, head])
                rank[layer, head, slots] = stable_rng(
                    context.seed,
                    context.event_id,
                    layer,
                    head,
                    "random-policy",
                ).random(slots.size)
        return stable_topk(
            state,
            rank,
            context.budget,
            protected=protection_mask(state, context),
            policy_id=self.policy_id,
        )


@dataclass(frozen=True, slots=True)
class OracleEvidencePolicy:
    policy_id: str = "B8_oracle_evidence"

    def select(self, state: SignificanceState, context: PolicyContext) -> Selection:
        if not context.allow_oracle:
            raise PolicyError("B8 requires allow_oracle=true and is synthetic-only")
        if context.oracle_evidence is None:
            raise PolicyError("B8 requires an oracle_evidence annotation")
        evidence = np.asarray(context.oracle_evidence, dtype=np.bool_)
        if evidence.shape != state.shape:
            raise StateInvariantError("oracle_evidence must match state shape")
        # Evidence outranks all non-evidence; recency breaks ties deterministically.
        scale = max(1.0, float(np.max(state.positions[state.active], initial=0) + 1))
        rank = np.where(state.active, state.positions / scale, -np.inf)
        rank = np.where(evidence & state.active, rank + 2.0, rank)
        return stable_topk(
            state,
            rank,
            context.budget,
            protected=protection_mask(state, context),
            policy_id=self.policy_id,
        )


def available_baselines() -> tuple[str, ...]:
    return ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")


def policy_from_baseline(name: str) -> BaselinePolicy:
    key = name.strip().upper()
    policies: dict[str, BaselinePolicy] = {
        "B0": FullCachePolicy(),
        "B1": SignificancePolicy(
            policy_id="B1_source_compatible",
            fixed_exponent=1.0,
            require_source_verification=True,
        ),
        "B2": SignificancePolicy(policy_id="B2_raw_accumulation", fixed_exponent=0.0),
        "B3": SignificancePolicy(policy_id="B3_observation_normalized", fixed_exponent=1.0),
        "B4": PositionalPolicy(),
        "B5": SignificancePolicy(
            policy_id="B5_age_exposure_approximation",
            fixed_exponent=1.0,
            count_transform="age",
        ),
        "B6": ShuffledSignificancePolicy(),
        "B7": RandomPolicy(),
        "B8": OracleEvidencePolicy(),
    }
    if key not in policies:
        raise PolicyError(f"unknown baseline {name!r}; expected one of {available_baselines()}")
    return policies[key]


def with_event(context: PolicyContext, event_id: str) -> PolicyContext:
    """Return a context carrying the current event identity for stable randomness."""

    return replace(context, event_id=event_id)
