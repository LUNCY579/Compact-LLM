"""Open-loop policy replay with explicit on-policy divergence invalidation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..policies import BaselinePolicy, PolicyContext
from .io import TraceReader


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    status: str
    valid: bool
    events_replayed: int
    valid_through_event: int | None
    diverged_at_event: int | None
    selections: tuple[dict[str, Any], ...]
    limitation: str


def replay_trace(
    trace: str | Path | TraceReader,
    policy: BaselinePolicy,
    context: PolicyContext,
) -> ReplayOutcome:
    reader = trace if isinstance(trace, TraceReader) else TraceReader(trace)
    selections: list[dict[str, Any]] = []
    valid_through: int | None = None
    diverged_at: int | None = None
    for event in reader.events():
        observed = event.observed_state()
        event_context = replace(
            context,
            current_position=event.boundary_position,
            event_id=event.event_id,
            oracle_evidence=event.oracle_evidence,
            allow_oracle=context.allow_oracle and event.oracle_evidence is not None,
        )
        selection = policy.select(observed, event_context)
        selected_ids = observed.selected_entry_ids(selection.indices)
        recorded_ids = event.recorded_entry_ids()
        same = selected_ids.shape == recorded_ids.shape and np.array_equal(selected_ids, recorded_ids)
        selections.append(
            {
                "event_index": event.event_index,
                "event_id": event.event_id,
                "policy_id": selection.policy_id,
                "selected_entry_ids": selected_ids.tolist(),
                "matches_recorded": bool(same),
            }
        )
        if reader.manifest.support == "ON_POLICY" and not same:
            diverged_at = event.event_index
            return ReplayOutcome(
                status="DIVERGED_ON_POLICY_UNSUPPORTED",
                valid=False,
                events_replayed=len(selections),
                valid_through_event=valid_through,
                diverged_at_event=diverged_at,
                selections=tuple(selections),
                limitation=(
                    "The recorded trace only contains on-policy future observations. "
                    "Counterfactual replay is invalid after the first changed retention set."
                ),
            )
        valid_through = event.event_index
    limitation = (
        "Open-loop replay does not model changes to future hidden states or attention, even with full-candidate traces."
    )
    return ReplayOutcome(
        status="COMPLETE_OPEN_LOOP",
        valid=True,
        events_replayed=len(selections),
        valid_through_event=valid_through,
        diverged_at_event=diverged_at,
        selections=tuple(selections),
        limitation=limitation,
    )
