"""Independent scalar reference for GFlowRL qualification.

This module intentionally uses Python scalars and loops rather than torch operations so the
qualification test does not merely restate the implementation with the same tensor primitives.
It is not a trainer and has no model/runtime side effects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReferenceObjective:
    loss: float
    partition_log_z: tuple[float, ...]
    valid_counts: tuple[int, ...]


def _same_rectangular_shape(name: str, value: Sequence[Sequence[float]], rows: int, cols: int) -> None:
    if len(value) != rows or any(len(row) != cols for row in value):
        raise ValueError(f"{name} shape mismatch")


def reference_gflowrl_loss(
    current: Sequence[Sequence[float]],
    old: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    rewards: Sequence[Sequence[float]],
    lengths: Sequence[Sequence[float]],
    valid: Sequence[Sequence[bool]] | None = None,
    *,
    beta: float = 1.0,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
    importance_clip: float = 2.0,
) -> ReferenceObjective:
    """Evaluate the paper Eq. 4-8 qualification formula with scalar arithmetic."""

    if not current or not current[0]:
        raise ValueError("current must be a nonempty rectangular matrix")
    rows, cols = len(current), len(current[0])
    if any(len(row) != cols for row in current):
        raise ValueError("current must be rectangular")
    for name, value in (
        ("old", old),
        ("reference", reference),
        ("rewards", rewards),
        ("lengths", lengths),
    ):
        _same_rectangular_shape(name, value, rows, cols)
    if valid is None:
        valid = [[True] * cols for _ in range(rows)]
    if len(valid) != rows or any(len(row) != cols for row in valid):
        raise ValueError("valid shape mismatch")
    if not (math.isfinite(beta) and beta > 0):
        raise ValueError("beta must be finite and positive")
    if not (math.isfinite(epsilon_low) and epsilon_low >= 0):
        raise ValueError("epsilon_low must be finite and nonnegative")
    if not (math.isfinite(epsilon_high) and epsilon_high >= 0):
        raise ValueError("epsilon_high must be finite and nonnegative")
    if not (math.isfinite(importance_clip) and importance_clip > 0):
        raise ValueError("importance_clip must be finite and positive")

    for matrices in (current, old, reference, rewards, lengths):
        for row in matrices:
            for value in row:
                if not math.isfinite(float(value)):
                    raise FloatingPointError("nonfinite scalar input")
    if any(float(value) <= 0 for row in lengths for value in row):
        raise ValueError("lengths must be positive")

    group_losses: list[float] = []
    partition: list[float] = []
    counts: list[int] = []
    cap_log = math.log(importance_clip)

    for prompt_index in range(rows):
        active = [i for i in range(cols) if bool(valid[prompt_index][i])]
        counts.append(len(active))
        if not active:
            partition.append(0.0)
            continue

        z = sum(
            beta * float(rewards[prompt_index][i])
            + float(reference[prompt_index][i])
            - float(old[prompt_index][i])
            for i in active
        ) / len(active)
        partition.append(z)

        per_rollout: list[float] = []
        for i in active:
            length = float(lengths[prompt_index][i])
            gap = (
                z
                + (float(old[prompt_index][i]) - float(reference[prompt_index][i])) / length
                - beta * float(rewards[prompt_index][i])
            )
            clipped = min(epsilon_high, max(-epsilon_low, gap))
            update = (float(current[prompt_index][i]) - float(old[prompt_index][i])) / length
            residual = clipped + update
            log_ratio = float(current[prompt_index][i]) - float(old[prompt_index][i])
            weight = math.exp(min(log_ratio, cap_log))
            per_rollout.append(weight * residual * residual)
        group_losses.append(sum(per_rollout) / len(per_rollout))

    return ReferenceObjective(
        loss=(sum(group_losses) / len(group_losses)) if group_losses else 0.0,
        partition_log_z=tuple(partition),
        valid_counts=tuple(counts),
    )
