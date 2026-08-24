"""Logical trace schema for replay-valid Cornerstone-1 observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..core.state import SignificanceState, StateInvariantError

TRACE_SCHEMA_VERSION = "1.0"
TraceSupport = Literal["FULL_CANDIDATE", "ON_POLICY"]
ExecutionMode = Literal["OPEN_LOOP_TRACE_SIMULATION", "CLOSED_LOOP_MODEL_EXECUTION"]


@dataclass(frozen=True, slots=True)
class TraceManifest:
    case_id: str
    support: TraceSupport
    execution_mode: ExecutionMode
    score_semantics: Mapping[str, Any]
    information_access: Mapping[str, bool]
    schema_version: str = TRACE_SCHEMA_VERSION
    model_id: str | None = None
    model_hash: str | None = None
    tokenizer_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema {self.schema_version!r}")
        if not self.case_id:
            raise ValueError("trace case_id is required")
        if self.support not in {"FULL_CANDIDATE", "ON_POLICY"}:
            raise ValueError(f"invalid trace support: {self.support}")
        if self.execution_mode not in {"OPEN_LOOP_TRACE_SIMULATION", "CLOSED_LOOP_MODEL_EXECUTION"}:
            raise ValueError(f"invalid execution mode: {self.execution_mode}")
        required = {"answer", "future_tokens", "future_hidden_state", "oracle_evidence"}
        missing = required - set(self.information_access)
        if missing:
            raise ValueError(f"information_access missing keys: {sorted(missing)}")
        if self.information_access["answer"] or self.information_access["future_tokens"]:
            raise ValueError("answer/future-token access is forbidden")
        if self.information_access["future_hidden_state"]:
            raise ValueError("future hidden-state access is forbidden")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "support": self.support,
            "execution_mode": self.execution_mode,
            "score_semantics": dict(self.score_semantics),
            "information_access": dict(self.information_access),
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceManifest:
        manifest = cls(
            schema_version=str(payload.get("schema_version", "")),
            case_id=str(payload.get("case_id", "")),
            support=str(payload.get("support", "")),  # type: ignore[arg-type]
            execution_mode=str(payload.get("execution_mode", "")),  # type: ignore[arg-type]
            score_semantics=dict(payload.get("score_semantics", {})),
            information_access={
                str(key): bool(value) for key, value in dict(payload.get("information_access", {})).items()
            },
            model_id=None if payload.get("model_id") is None else str(payload["model_id"]),
            model_hash=None if payload.get("model_hash") is None else str(payload["model_hash"]),
            tokenizer_hash=None if payload.get("tokenizer_hash") is None else str(payload["tokenizer_hash"]),
            metadata=dict(payload.get("metadata", {})),
        )
        manifest.validate()
        return manifest


@dataclass(slots=True)
class TraceEvent:
    case_id: str
    event_index: int
    event_id: str
    kind: str
    boundary_position: int
    state_before: SignificanceState
    score_delta: NDArray[np.float32]
    count_delta: NDArray[np.float32]
    recorded_indices: NDArray[np.int64]
    policy_id: str
    oracle_evidence: NDArray[np.bool_] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.case_id or not self.event_id or self.event_index < 0 or self.boundary_position < 0:
            raise ValueError("event identity/index/boundary are invalid")
        self.state_before.validate()
        self.score_delta = np.asarray(self.score_delta, dtype=np.float32)
        self.count_delta = np.asarray(self.count_delta, dtype=np.float32)
        self.recorded_indices = np.asarray(self.recorded_indices, dtype=np.int64)
        if self.score_delta.shape != self.state_before.shape or self.count_delta.shape != self.state_before.shape:
            raise StateInvariantError("trace deltas must match state shape")
        if not np.isfinite(self.score_delta).all() or not np.isfinite(self.count_delta).all():
            raise StateInvariantError("trace deltas must be finite")
        if np.any(self.count_delta < 0):
            raise StateInvariantError("trace count deltas must be non-negative")
        if np.any(self.score_delta[~self.state_before.active] != 0) or np.any(
            self.count_delta[~self.state_before.active] != 0
        ):
            raise StateInvariantError("inactive trace entries cannot receive observations")
        # selected_entry_ids performs complete index validation.
        self.state_before.selected_entry_ids(self.recorded_indices)
        if self.oracle_evidence is not None:
            self.oracle_evidence = np.asarray(self.oracle_evidence, dtype=np.bool_)
            if self.oracle_evidence.shape != self.state_before.shape:
                raise StateInvariantError("oracle_evidence must match state shape")

    def observed_state(self) -> SignificanceState:
        self.validate()
        state = self.state_before.clone()
        state.update(self.score_delta, self.count_delta)
        return state

    def recorded_entry_ids(self) -> NDArray[np.int64]:
        return self.observed_state().selected_entry_ids(self.recorded_indices)
