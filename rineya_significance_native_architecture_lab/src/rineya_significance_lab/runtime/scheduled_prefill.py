"""Explicit fixed-terminal schedules for compression-round causal controls.

The ordinary constant-chunk prefill leaves the last (possibly short) chunk
uncompressed.  Comparing different chunk sizes would therefore change both
the number of compression rounds *and* the final resident cache size.  This
module instead partitions a prefix at the requested chunk size, compresses
after every prefix chunk, and appends one fixed-size terminal chunk.  The
terminal chunk and retained-history budget are identical across schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .cache import CompressionReport, SignificanceCache
from .policy import SignificancePolicy
from .transformers_runtime import chunked_prefill, score_with_query


@dataclass(frozen=True, slots=True)
class FixedTerminalSchedule:
    """A token-exact schedule with a fixed uncompressed terminal suffix."""

    schedule_id: str
    input_tokens: int
    prefix_tokens: int
    prefix_chunk_tokens: int
    terminal_active_tokens: int
    chunk_sizes: tuple[int, ...]
    boundary_positions: tuple[int, ...]
    compression_rounds: int
    retained_history_budget: int
    final_resident_entries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "input_tokens": self.input_tokens,
            "prefix_tokens": self.prefix_tokens,
            "prefix_chunk_tokens": self.prefix_chunk_tokens,
            "terminal_active_tokens": self.terminal_active_tokens,
            "chunk_sizes": list(self.chunk_sizes),
            "boundary_positions": list(self.boundary_positions),
            "compression_rounds": self.compression_rounds,
            "retained_history_budget": self.retained_history_budget,
            "final_resident_entries": self.final_resident_entries,
        }


def build_fixed_terminal_schedule(
    *,
    schedule_id: str,
    input_tokens: int,
    prefix_chunk_tokens: int,
    terminal_active_tokens: int,
    retained_history_budget: int,
) -> FixedTerminalSchedule:
    """Construct the causal schedule without inspecting model state.

    Every prefix chunk ends in one query-scoring/compression boundary.  The
    terminal suffix is appended after the last compression and is never
    compressed, so all schedules have the same final resident entry count.
    """

    if not schedule_id:
        raise ValueError("schedule_id must be non-empty")
    values = {
        "input_tokens": input_tokens,
        "prefix_chunk_tokens": prefix_chunk_tokens,
        "terminal_active_tokens": terminal_active_tokens,
        "retained_history_budget": retained_history_budget,
    }
    if any(int(value) <= 0 for value in values.values()):
        raise ValueError(f"schedule sizes must be positive: {values}")
    if terminal_active_tokens >= input_tokens:
        raise ValueError("terminal_active_tokens must leave a non-empty scheduled prefix")

    prefix_tokens = input_tokens - terminal_active_tokens
    prefix_sizes = tuple(
        min(prefix_chunk_tokens, prefix_tokens - start)
        for start in range(0, prefix_tokens, prefix_chunk_tokens)
    )
    if not prefix_sizes or sum(prefix_sizes) != prefix_tokens:
        raise RuntimeError("fixed-terminal prefix partition is inconsistent")
    boundary_positions: list[int] = []
    cursor = 0
    for size in prefix_sizes:
        cursor += size
        boundary_positions.append(cursor)
    final_resident = min(prefix_tokens, retained_history_budget) + terminal_active_tokens
    return FixedTerminalSchedule(
        schedule_id=schedule_id,
        input_tokens=input_tokens,
        prefix_tokens=prefix_tokens,
        prefix_chunk_tokens=prefix_chunk_tokens,
        terminal_active_tokens=terminal_active_tokens,
        chunk_sizes=(*prefix_sizes, terminal_active_tokens),
        boundary_positions=tuple(boundary_positions),
        compression_rounds=len(prefix_sizes),
        retained_history_budget=retained_history_budget,
        final_resident_entries=final_resident,
    )


@dataclass(frozen=True)
class ScheduledPrefillResult:
    """Model output plus auditable per-boundary cache accounting."""

    output: Any
    compression_reports: tuple[CompressionReport, ...]
    boundary_accounting: tuple[dict[str, int], ...]
    schedule: FixedTerminalSchedule
    scoring_events_per_layer: tuple[int, ...]
    max_transient_resident_entries: int
    query_probe_token_evaluations: int


@torch.no_grad()
def fixed_terminal_prefill(
    model: Any,
    input_ids: torch.Tensor,
    cache: SignificanceCache,
    *,
    schedule: FixedTerminalSchedule,
    policy: SignificancePolicy,
    query_ids: torch.Tensor,
    verify_query_rollback: bool = True,
    force_repeat_kv: bool = True,
    logits_to_keep: int | None = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> ScheduledPrefillResult:
    """Execute a fixed-terminal schedule with one query probe per round."""

    if input_ids.ndim != 2 or int(input_ids.shape[-1]) != schedule.input_tokens:
        raise ValueError(
            f"input_ids must have schedule length {schedule.input_tokens}, got {tuple(input_ids.shape)}"
        )
    if query_ids.ndim != 2 or int(query_ids.shape[-1]) <= 0:
        raise ValueError("query_ids must have shape [batch, q>0]")
    if policy.budget != schedule.retained_history_budget:
        raise ValueError("policy budget must equal the schedule retained-history budget")
    if cache.stored_length() != 0 or cache.true_length != 0:
        raise ValueError("fixed-terminal prefill requires an empty cache")

    initial_events = tuple(cache.scoring_events())
    reports: list[CompressionReport] = []
    boundaries: list[dict[str, int]] = []
    cursor = 0
    output: Any | None = None
    max_transient = 0
    query_tokens = int(query_ids.shape[-1])
    prefix_chunk_count = schedule.compression_rounds

    for chunk_index, chunk_size in enumerate(schedule.chunk_sizes):
        piece = input_ids[:, cursor : cursor + chunk_size]
        cursor += chunk_size
        result = chunked_prefill(
            model,
            piece,
            cache,
            chunk_size=chunk_size,
            policy=None,
            collect_logits=False,
            force_repeat_kv=force_repeat_kv,
            logits_to_keep=logits_to_keep,
            model_kwargs=model_kwargs,
        )
        output = result.output
        resident_before_probe = cache.stored_length()
        max_transient = max(max_transient, resident_before_probe)
        if chunk_index < prefix_chunk_count:
            max_transient = max(max_transient, resident_before_probe + query_tokens)
            score_with_query(
                model,
                cache,
                query_ids,
                verify_rollback=verify_query_rollback,
                force_repeat_kv=force_repeat_kv,
                logits_to_keep=1,
                model_kwargs=model_kwargs,
            )
            report = cache.compress(policy)
            reports.append(report)
            boundaries.append(
                {
                    "boundary_index": chunk_index,
                    "absolute_token_position": schedule.boundary_positions[chunk_index],
                    "history_entries_before_probe": resident_before_probe,
                    "temporary_query_entries": query_tokens,
                    "resident_after_compression": cache.stored_length(),
                }
            )

    if output is None or cursor != schedule.input_tokens:
        raise RuntimeError("fixed-terminal prefill did not consume the exact input")
    cache.assert_aligned()
    if cache.true_length != schedule.input_tokens:
        raise RuntimeError(
            f"semantic length mismatch: runtime={cache.true_length}, schedule={schedule.input_tokens}"
        )
    if cache.stored_length() != schedule.final_resident_entries:
        raise RuntimeError(
            "final resident cache mismatch: "
            f"runtime={cache.stored_length()}, schedule={schedule.final_resident_entries}"
        )
    final_events = tuple(cache.scoring_events())
    event_deltas = tuple(
        after - before for before, after in zip(initial_events, final_events, strict=True)
    )
    if event_deltas != (schedule.compression_rounds,) * len(event_deltas):
        raise RuntimeError(
            f"query access mismatch: expected {schedule.compression_rounds} events/layer, "
            f"got {event_deltas}"
        )
    return ScheduledPrefillResult(
        output=output,
        compression_reports=tuple(reports),
        boundary_accounting=tuple(boundaries),
        schedule=schedule,
        scoring_events_per_layer=event_deltas,
        max_transient_resident_entries=max_transient,
        query_probe_token_evaluations=query_tokens * schedule.compression_rounds,
    )
