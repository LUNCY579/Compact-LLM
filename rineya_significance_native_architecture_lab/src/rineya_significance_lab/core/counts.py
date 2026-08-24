"""Causal ablations over realized exposure counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .randomness import stable_rng
from .state import SignificanceState, StateInvariantError

CountTransformName = Literal[
    "true",
    "constant",
    "age",
    "shuffled",
    "circular",
    "delayed",
    "noisy",
    "head_averaged",
]


@dataclass(frozen=True, slots=True)
class CountTransformContext:
    current_position: int | None = None
    chunk_size: int | None = None
    previous_counts: NDArray[Any] | None = None
    noise_std: float = 0.25
    seed: int = 0
    event_id: str = ""


def _per_row_permutation(
    state: SignificanceState,
    values: NDArray[Any],
    *,
    seed: int,
    event_id: str,
) -> NDArray[np.float64]:
    result = np.zeros(state.shape, dtype=np.float64)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            slots = np.flatnonzero(state.active[layer, head])
            shuffled = values[layer, head, slots].copy()
            stable_rng(seed, event_id, layer, head, "counts").shuffle(shuffled)
            result[layer, head, slots] = shuffled
    return result


def _head_average_by_entry(state: SignificanceState) -> NDArray[np.float64]:
    result = np.zeros(state.shape, dtype=np.float64)
    for layer in range(state.layers):
        totals: dict[int, list[float]] = {}
        for head in range(state.kv_heads):
            for slot in np.flatnonzero(state.active[layer, head]):
                entry_id = int(state.entry_ids[layer, head, slot])
                totals.setdefault(entry_id, []).append(float(state.counts[layer, head, slot]))
        means = {entry_id: float(np.mean(row)) for entry_id, row in totals.items()}
        for head in range(state.kv_heads):
            for slot in np.flatnonzero(state.active[layer, head]):
                result[layer, head, slot] = means[int(state.entry_ids[layer, head, slot])]
    return result


def transform_counts(
    state: SignificanceState,
    name: CountTransformName,
    context: CountTransformContext | None = None,
) -> NDArray[np.float64]:
    context = context or CountTransformContext()
    result: NDArray[np.float64]
    if name == "true":
        result = np.asarray(state.counts, dtype=np.float64).copy()
    elif name == "constant":
        result = np.asarray(np.where(state.active, 1.0, 0.0), dtype=np.float64)
    elif name == "age":
        if context.current_position is None or context.chunk_size is None or context.chunk_size <= 0:
            raise ValueError("age counts require current_position and positive chunk_size")
        # boundary_position is one past the newest absolute token. Convert it
        # to the newest observed position before deriving chunk exposure; using
        # the one-past value gives the first token of every chunk an extra count.
        age = np.maximum(context.current_position - 1 - state.positions, 0)
        result = np.asarray(
            np.where(state.active, np.floor_divide(age, context.chunk_size) + 1.0, 0.0),
            dtype=np.float64,
        )
    elif name == "shuffled":
        result = _per_row_permutation(
            state,
            state.counts,
            seed=context.seed,
            event_id=context.event_id,
        )
    elif name == "circular":
        result = np.zeros(state.shape, dtype=np.float64)
        for layer in range(state.layers):
            for head in range(state.kv_heads):
                slots = np.flatnonzero(state.active[layer, head])
                if slots.size:
                    order = sorted(
                        slots.tolist(),
                        key=lambda slot: (
                            int(state.positions[layer, head, slot]),
                            int(state.entry_ids[layer, head, slot]),
                        ),
                    )
                    result[layer, head, order] = np.roll(state.counts[layer, head, order], 1)
    elif name == "delayed":
        if context.previous_counts is None:
            raise ValueError("delayed counts require previous_counts")
        previous = np.asarray(context.previous_counts, dtype=np.float64)
        if previous.shape != state.shape:
            raise StateInvariantError("previous_counts must match state")
        result = np.asarray(np.where(state.active, previous, 0.0), dtype=np.float64)
    elif name == "noisy":
        if not np.isfinite(context.noise_std) or context.noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")
        noise = stable_rng(context.seed, context.event_id, "count-noise").normal(
            0.0,
            context.noise_std,
            size=state.shape,
        )
        result = np.asarray(
            np.where(state.active, np.maximum(state.counts + noise, 0.0), 0.0),
            dtype=np.float64,
        )
    elif name == "head_averaged":
        result = _head_average_by_entry(state)
    else:
        raise ValueError(f"unknown count transform: {name}")
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result[~state.active] != 0):
        raise StateInvariantError(f"count transform {name!r} violated count invariants")
    return result
