"""Narrow adapters between rich geometry plans and exact native updates."""

from __future__ import annotations

from .application import NativeUpdatePlan, TensorUpdatePlan
from .plan import PCSMPlan


def to_native_update_plan(plan: PCSMPlan) -> NativeUpdatePlan:
    entries: list[TensorUpdatePlan] = []
    for entry in plan.entries:
        indices = entry.flat_indices
        selected_stabilizer = entry.stabilizer.reshape(-1).index_select(0, indices)
        entries.append(
            TensorUpdatePlan(
                name=entry.name,
                numel=entry.numel,
                flat_indices=indices,
                stabilizer=selected_stabilizer,
                shape=entry.shape,
            )
        )
    return NativeUpdatePlan(tuple(entries))


__all__ = ["to_native_update_plan"]
