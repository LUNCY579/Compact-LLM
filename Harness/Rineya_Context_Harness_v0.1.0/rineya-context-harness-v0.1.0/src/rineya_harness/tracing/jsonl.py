from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from rineya_harness.clock import Clock, SystemClock
from rineya_harness.domain.enums import TraceEventType, WorkflowNode
from rineya_harness.domain.models import TraceEvent
from rineya_harness.ids import new_id


class JsonlTraceRecorder:
    """Connected per-run trace with one JSON object per line."""

    def __init__(
        self,
        directory: str | Path,
        *,
        clock: Clock | None = None,
        max_payload_chars: int = 20_000,
        include_tool_payloads: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        self.max_payload_chars = max_payload_chars
        self.include_tool_payloads = include_tool_payloads
        self._lock = threading.Lock()

    def emit(
        self,
        *,
        trace_id: str,
        run_id: str,
        event_type: TraceEventType,
        node: WorkflowNode | None = None,
        payload: dict[str, Any] | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            trace_id=trace_id,
            span_id=span_id or new_id("span"),
            parent_span_id=parent_span_id,
            run_id=run_id,
            event_type=event_type,
            node=node,
            payload=self._event_payload(event_type, payload or {}),
            created_at=self.clock.now(),
        )
        target = self.directory / f"{run_id}.jsonl"
        with self._lock, target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")
            handle.flush()
        return event

    @contextmanager
    def span(
        self,
        *,
        trace_id: str,
        run_id: str,
        node: WorkflowNode,
        payload: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> Iterator[str]:
        span_id = new_id("span")
        self.emit(
            trace_id=trace_id,
            run_id=run_id,
            event_type=TraceEventType.NODE_ENTERED,
            node=node,
            payload=payload,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        try:
            yield span_id
        except Exception as exc:
            self.emit(
                trace_id=trace_id,
                run_id=run_id,
                event_type=TraceEventType.ERROR,
                node=node,
                payload={"error_type": type(exc).__name__, "message": str(exc)},
                span_id=span_id,
                parent_span_id=parent_span_id,
            )
            raise
        finally:
            self.emit(
                trace_id=trace_id,
                run_id=run_id,
                event_type=TraceEventType.NODE_EXITED,
                node=node,
                span_id=span_id,
                parent_span_id=parent_span_id,
            )

    def read(self, run_id: str) -> list[TraceEvent]:
        target = self.directory / f"{run_id}.jsonl"
        if not target.exists():
            return []
        events: list[TraceEvent] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(TraceEvent.model_validate_json(line))
        return events

    def _event_payload(
        self,
        event_type: TraceEventType,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            not self.include_tool_payloads
            and event_type
            in {
                TraceEventType.TOOL_STARTED,
                TraceEventType.TOOL_FINISHED,
                TraceEventType.TOOL_FAILED,
            }
        ):
            return {
                "redacted": True,
                "original_keys": sorted(str(key) for key in payload),
            }
        return self._bounded_payload(payload)

    def _bounded_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if len(serialized) <= self.max_payload_chars:
            return payload
        return {
            "truncated": True,
            "original_chars": len(serialized),
            "preview": serialized[: self.max_payload_chars],
        }
