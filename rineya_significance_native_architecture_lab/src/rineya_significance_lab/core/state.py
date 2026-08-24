"""Aligned persistent significance state with semantic entry identity."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


class StateInvariantError(ValueError):
    """Raised when cache-entry alignment or numeric invariants are violated."""


def _array(value: NDArray[Any], dtype: DTypeLike) -> NDArray[Any]:
    return np.asarray(value, dtype=dtype)


@dataclass(slots=True)
class SignificanceState:
    """State carried with cached semantic entries.

    All fields have shape ``[layers, kv_heads, slots]``. The same semantic
    token may appear in several layers or heads, but an active ``entry_id``
    must be unique within a single layer/head row. Inactive slots use ``-1``
    metadata and zero score/count values.
    """

    entry_ids: IntArray
    positions: IntArray
    token_ids: IntArray
    scores: FloatArray
    counts: FloatArray
    active: BoolArray

    def __post_init__(self) -> None:
        self.entry_ids = _array(self.entry_ids, np.int64)
        self.positions = _array(self.positions, np.int64)
        self.token_ids = _array(self.token_ids, np.int64)
        self.scores = _array(self.scores, np.float32)
        self.counts = _array(self.counts, np.float32)
        self.active = _array(self.active, np.bool_)
        self.validate()

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.entry_ids.shape)  # type: ignore[return-value]

    @property
    def layers(self) -> int:
        return self.shape[0]

    @property
    def kv_heads(self) -> int:
        return self.shape[1]

    @property
    def slots(self) -> int:
        return self.shape[2]

    @classmethod
    def empty(cls, layers: int, kv_heads: int, slots: int) -> SignificanceState:
        if min(layers, kv_heads, slots) < 0 or layers == 0 or kv_heads == 0:
            raise ValueError("layers and kv_heads must be positive; slots must be non-negative")
        shape = (layers, kv_heads, slots)
        return cls(
            entry_ids=np.full(shape, -1, dtype=np.int64),
            positions=np.full(shape, -1, dtype=np.int64),
            token_ids=np.full(shape, -1, dtype=np.int64),
            scores=np.zeros(shape, dtype=np.float32),
            counts=np.zeros(shape, dtype=np.float32),
            active=np.zeros(shape, dtype=np.bool_),
        )

    @classmethod
    def from_entries(
        cls,
        entry_ids: NDArray[Any],
        positions: NDArray[Any],
        token_ids: NDArray[Any],
        *,
        layers: int = 1,
        kv_heads: int = 1,
        scores: NDArray[Any] | None = None,
        counts: NDArray[Any] | None = None,
    ) -> SignificanceState:
        """Broadcast one token sequence across layer/KV-head rows."""

        ids = np.asarray(entry_ids, dtype=np.int64)
        pos = np.asarray(positions, dtype=np.int64)
        toks = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or pos.shape != ids.shape or toks.shape != ids.shape:
            raise ValueError("entry_ids, positions, and token_ids must be equal-length vectors")
        shape = (layers, kv_heads, ids.size)
        def expand(value: NDArray[Any]) -> NDArray[Any]:
            return np.broadcast_to(value, shape).copy()

        score_values = (
            np.zeros(shape, dtype=np.float32)
            if scores is None
            else np.asarray(np.broadcast_to(scores, shape), dtype=np.float32).copy()
        )
        count_values = (
            np.zeros(shape, dtype=np.float32)
            if counts is None
            else np.asarray(np.broadcast_to(counts, shape), dtype=np.float32).copy()
        )
        return cls(
            entry_ids=expand(ids),
            positions=expand(pos),
            token_ids=expand(toks),
            scores=score_values,
            counts=count_values,
            active=np.ones(shape, dtype=np.bool_),
        )

    def validate(self) -> None:
        arrays: dict[str, NDArray[Any]] = {
            "positions": self.positions,
            "token_ids": self.token_ids,
            "scores": self.scores,
            "counts": self.counts,
            "active": self.active,
        }
        if self.entry_ids.ndim != 3:
            raise StateInvariantError("state arrays must have shape [layers, kv_heads, slots]")
        for name, value in arrays.items():
            if value.shape != self.entry_ids.shape:
                raise StateInvariantError(f"{name} shape {value.shape} != entry_ids shape {self.entry_ids.shape}")
        if not np.isfinite(self.scores).all():
            raise StateInvariantError("scores must be finite")
        if not np.isfinite(self.counts).all() or np.any(self.counts < 0):
            raise StateInvariantError("counts must be finite and non-negative")
        if np.any(self.active & (self.entry_ids < 0)):
            raise StateInvariantError("active entries require non-negative entry_ids")
        if np.any(self.active & (self.positions < 0)):
            raise StateInvariantError("active entries require non-negative absolute positions")
        inactive = ~self.active
        if np.any(self.entry_ids[inactive] != -1) or np.any(self.positions[inactive] != -1):
            raise StateInvariantError("inactive entry_ids and positions must be -1")
        if np.any(self.token_ids[inactive] != -1):
            raise StateInvariantError("inactive token_ids must be -1")
        if np.any(self.scores[inactive] != 0) or np.any(self.counts[inactive] != 0):
            raise StateInvariantError("inactive score/count values must be zero")
        for layer in range(self.layers):
            for head in range(self.kv_heads):
                ids = self.entry_ids[layer, head, self.active[layer, head]]
                if ids.size != np.unique(ids).size:
                    raise StateInvariantError(f"duplicate active entry_id in layer={layer}, kv_head={head}")

    def clone(self) -> SignificanceState:
        return SignificanceState(
            entry_ids=self.entry_ids.copy(),
            positions=self.positions.copy(),
            token_ids=self.token_ids.copy(),
            scores=self.scores.copy(),
            counts=self.counts.copy(),
            active=self.active.copy(),
        )

    def update(self, score_delta: NDArray[Any], count_delta: NDArray[Any]) -> None:
        score = np.asarray(score_delta, dtype=np.float32)
        count = np.asarray(count_delta, dtype=np.float32)
        if score.shape != self.shape or count.shape != self.shape:
            raise StateInvariantError("score/count deltas must match state shape")
        if not np.isfinite(score).all() or not np.isfinite(count).all() or np.any(count < 0):
            raise StateInvariantError("deltas must be finite and count deltas non-negative")
        if np.any(score[~self.active] != 0) or np.any(count[~self.active] != 0):
            raise StateInvariantError("inactive entries cannot receive observations")
        self.scores += score
        self.counts += count
        self.validate()

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (self.entry_ids, self.positions, self.token_ids, self.scores, self.counts, self.active):
            contiguous = np.ascontiguousarray(value)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def selected_entry_ids(self, indices: IntArray) -> IntArray:
        selection = _validate_indices(self, indices)
        result = np.full(selection.shape, -1, dtype=np.int64)
        for layer in range(self.layers):
            for head in range(self.kv_heads):
                valid = selection[layer, head] >= 0
                result[layer, head, valid] = self.entry_ids[layer, head, selection[layer, head, valid]]
        return result


def _validate_indices(state: SignificanceState, indices: NDArray[Any]) -> IntArray:
    selection = np.asarray(indices, dtype=np.int64)
    if selection.ndim != 3 or selection.shape[:2] != state.shape[:2]:
        raise StateInvariantError("selection indices must have shape [layers, kv_heads, selected_slots]")
    if np.any(selection < -1) or np.any(selection >= state.slots):
        raise StateInvariantError("selection index outside state slots")
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            row = selection[layer, head]
            chosen = row[row >= 0]
            if chosen.size != np.unique(chosen).size:
                raise StateInvariantError("selection contains duplicate indices")
            if chosen.size and not state.active[layer, head, chosen].all():
                raise StateInvariantError("selection references inactive entries")
            if np.any(row[:-1] < 0) and np.any(row[np.flatnonzero(row < 0)[0] + 1 :] >= 0):
                raise StateInvariantError("selection padding must be trailing")
    return selection


def atomic_gather(
    state: SignificanceState,
    indices: NDArray[Any],
    payloads: Mapping[str, NDArray[Any]] | None = None,
) -> tuple[SignificanceState, dict[str, NDArray[Any]]]:
    """Gather aligned state and optional K/V-like payloads as one checked operation.

    Payload arrays must begin with ``[layers, kv_heads, slots]`` and may have
    arbitrary trailing dimensions, such as a head dimension for K and V.
    """

    selection = _validate_indices(state, indices)
    out_shape = selection.shape
    active = selection >= 0

    def gather_field(source: NDArray[Any], fill: int | float | bool) -> NDArray[Any]:
        output = np.full(out_shape, fill, dtype=source.dtype)
        for layer in range(state.layers):
            for head in range(state.kv_heads):
                valid = active[layer, head]
                output[layer, head, valid] = source[layer, head, selection[layer, head, valid]]
        return output

    gathered_state = SignificanceState(
        entry_ids=gather_field(state.entry_ids, -1),
        positions=gather_field(state.positions, -1),
        token_ids=gather_field(state.token_ids, -1),
        scores=gather_field(state.scores, 0.0),
        counts=gather_field(state.counts, 0.0),
        active=active.copy(),
    )
    gathered_payloads: dict[str, NDArray[Any]] = {}
    for name, source_value in (payloads or {}).items():
        source = np.asarray(source_value)
        if source.ndim < 3 or source.shape[:3] != state.shape:
            raise StateInvariantError(f"payload {name!r} does not align with state shape")
        output = np.zeros((*out_shape, *source.shape[3:]), dtype=source.dtype)
        for layer in range(state.layers):
            for head in range(state.kv_heads):
                valid = active[layer, head]
                output[layer, head, valid, ...] = source[layer, head, selection[layer, head, valid], ...]
        gathered_payloads[name] = output
    gathered_state.validate()
    return gathered_state, gathered_payloads
