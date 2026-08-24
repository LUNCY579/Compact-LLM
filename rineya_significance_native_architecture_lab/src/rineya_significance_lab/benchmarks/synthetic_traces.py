"""Synthetic full-candidate score traces isolating repeated-exposure bias."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.ranking import normalized_rank
from ..core.state import SignificanceState
from ..policies import PolicyContext, policy_from_baseline
from ..traces import TraceEvent, TraceManifest, TraceWriter


@dataclass(frozen=True, slots=True)
class ExposureTraceSpec:
    case_id: str = "tier-a-exposure-bias"
    rounds: int = 6
    entries_per_round: int = 2
    layers: int = 1
    kv_heads: int = 2
    budget: int = 2
    early_utility: float = 1.0
    late_utility: float = 1.25
    seed: int = 17

    def validate(self) -> None:
        if min(self.rounds, self.entries_per_round, self.layers, self.kv_heads) <= 0:
            raise ValueError("round/entry/layer/head counts must be positive")
        if self.budget <= 0 or self.budget > self.rounds * self.entries_per_round:
            raise ValueError("budget outside valid slot range")
        if self.early_utility <= 0 or self.late_utility <= self.early_utility:
            raise ValueError("late utility must be greater than positive early utility")


def _state_before(
    spec: ExposureTraceSpec,
    event_index: int,
) -> tuple[
    SignificanceState,
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
]:
    total = spec.rounds * spec.entries_per_round
    active_count = (event_index + 1) * spec.entries_per_round
    state = SignificanceState.empty(spec.layers, spec.kv_heads, total)
    ids = np.arange(10_000, 10_000 + active_count, dtype=np.int64)
    positions = np.arange(active_count, dtype=np.int64)
    tokens = np.arange(100, 100 + active_count, dtype=np.int64)
    state.entry_ids[:, :, :active_count] = ids
    state.positions[:, :, :active_count] = positions
    state.token_ids[:, :, :active_count] = tokens
    state.active[:, :, :active_count] = True
    # Full-candidate history: every entry receives its intrinsic utility once per
    # scoring round after arrival. This is independent of the recorded selection.
    arrivals = np.floor_divide(np.arange(active_count), spec.entries_per_round)
    prior_exposures = np.maximum(event_index - arrivals, 0).astype(np.float32)
    utility = np.full(active_count, spec.early_utility, dtype=np.float32)
    late_start = (spec.rounds - 1) * spec.entries_per_round
    utility[np.arange(active_count) >= late_start] = spec.late_utility
    state.counts[:, :, :active_count] = prior_exposures
    state.scores[:, :, :active_count] = prior_exposures * utility
    score_delta = np.zeros(state.shape, dtype=np.float32)
    count_delta = np.zeros(state.shape, dtype=np.float32)
    score_delta[:, :, :active_count] = utility
    count_delta[:, :, :active_count] = 1.0
    state.validate()
    oracle = np.zeros(state.shape, dtype=np.bool_)
    oracle[:, :, max(0, late_start) :active_count] = True
    return state, score_delta, count_delta, oracle


def build_exposure_events(spec: ExposureTraceSpec) -> list[TraceEvent]:
    spec.validate()
    policy = policy_from_baseline("B3")
    events: list[TraceEvent] = []
    for event_index in range(spec.rounds):
        state, score_delta, count_delta, oracle = _state_before(spec, event_index)
        observed = state.clone()
        observed.update(score_delta, count_delta)
        event_id = f"round-{event_index:03d}"
        selection = policy.select(
            observed,
            PolicyContext(
                budget=spec.budget,
                current_position=(event_index + 1) * spec.entries_per_round,
                chunk_size=spec.entries_per_round,
                seed=spec.seed,
                event_id=event_id,
            ),
        )
        events.append(
            TraceEvent(
                case_id=spec.case_id,
                event_index=event_index,
                event_id=event_id,
                kind="synthetic_scoring_boundary",
                boundary_position=(event_index + 1) * spec.entries_per_round,
                state_before=state,
                score_delta=score_delta,
                count_delta=count_delta,
                recorded_indices=selection.indices,
                policy_id=selection.policy_id,
                oracle_evidence=oracle,
                metadata={
                    "synthetic": True,
                    "open_loop": True,
                    "entries_arrived": (event_index + 1) * spec.entries_per_round,
                },
            )
        )
    return events


def write_exposure_trace(root: str | Path, spec: ExposureTraceSpec) -> Path:
    manifest = TraceManifest(
        case_id=spec.case_id,
        support="FULL_CANDIDATE",
        execution_mode="OPEN_LOOP_TRACE_SIMULATION",
        score_semantics={
            "score": "intrinsic_utility_per_realized_scoring_event",
            "count": "one_per_realized_scoring_event",
            "dtype": "float32",
            "gqa_mapping": "synthetic_kv_head_native",
            "pooling": False,
            "tie_break": "absolute_position_then_entry_id_lab_strengthening",
        },
        information_access={
            "answer": False,
            "future_tokens": False,
            "future_hidden_state": False,
            "oracle_evidence": False,
        },
        metadata={
            "tier": "A",
            "synthetic": True,
            "claims_scope": "mechanism-only; no model inference",
            "spec": {
                "rounds": spec.rounds,
                "entries_per_round": spec.entries_per_round,
                "layers": spec.layers,
                "kv_heads": spec.kv_heads,
                "budget": spec.budget,
                "early_utility": spec.early_utility,
                "late_utility": spec.late_utility,
                "seed": spec.seed,
            },
        },
    )
    destination = Path(root)
    writer = TraceWriter(destination, manifest)
    for event in build_exposure_events(spec):
        writer.write_event(event)
    return destination


def exposure_bias_table(spec: ExposureTraceSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in build_exposure_events(spec):
        state = event.observed_state()
        if event.oracle_evidence is None:
            raise AssertionError("synthetic exposure events require oracle annotations")
        raw = normalized_rank(state, 0.0)
        normalized = normalized_rank(state, 1.0)
        active = state.active[0, 0]
        for slot in np.flatnonzero(active):
            rows.append(
                {
                    "case_id": spec.case_id,
                    "round": event.event_index + 1,
                    "entry_id": int(state.entry_ids[0, 0, slot]),
                    "position": int(state.positions[0, 0, slot]),
                    "score": float(state.scores[0, 0, slot]),
                    "count": float(state.counts[0, 0, slot]),
                    "raw_rank": float(raw[0, 0, slot]),
                    "normalized_rank": float(normalized[0, 0, slot]),
                    "is_late_high_utility": bool(event.oracle_evidence[0, 0, slot]),
                }
            )
    return rows
