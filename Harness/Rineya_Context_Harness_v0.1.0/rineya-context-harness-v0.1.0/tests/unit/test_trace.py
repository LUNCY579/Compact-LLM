from __future__ import annotations

from rineya_harness.domain.enums import TraceEventType, WorkflowNode
from rineya_harness.tracing.jsonl import JsonlTraceRecorder


def test_jsonl_trace_round_trip(tmp_path) -> None:
    recorder = JsonlTraceRecorder(tmp_path)
    recorder.emit(
        trace_id="trace",
        run_id="run",
        event_type=TraceEventType.RUN_STARTED,
        node=WorkflowNode.INTAKE,
        payload={"x": 1},
    )
    events = recorder.read("run")
    assert len(events) == 1
    assert events[0].payload == {"x": 1}


def test_trace_payload_is_bounded(tmp_path) -> None:
    recorder = JsonlTraceRecorder(tmp_path, max_payload_chars=20)
    event = recorder.emit(
        trace_id="trace",
        run_id="run",
        event_type=TraceEventType.ERROR,
        payload={"large": "x" * 100},
    )
    assert event.payload["truncated"] is True


def test_trace_can_redact_tool_payloads(tmp_path) -> None:
    recorder = JsonlTraceRecorder(tmp_path, include_tool_payloads=False)
    event = recorder.emit(
        trace_id="trace",
        run_id="run",
        event_type=TraceEventType.TOOL_FINISHED,
        payload={"query": "private query", "documents": 4},
    )
    assert event.payload == {
        "redacted": True,
        "original_keys": ["documents", "query"],
    }
