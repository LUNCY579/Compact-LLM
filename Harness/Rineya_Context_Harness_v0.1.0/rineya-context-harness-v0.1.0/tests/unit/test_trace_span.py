from __future__ import annotations

import pytest

from rineya_harness.domain.enums import TraceEventType, WorkflowNode
from rineya_harness.tracing.jsonl import JsonlTraceRecorder


def test_trace_span_success(tmp_path) -> None:
    recorder = JsonlTraceRecorder(tmp_path)
    with recorder.span(trace_id="trace", run_id="run", node=WorkflowNode.SEARCH):
        pass
    events = recorder.read("run")
    assert [event.event_type for event in events] == [
        TraceEventType.NODE_ENTERED,
        TraceEventType.NODE_EXITED,
    ]


def test_trace_span_error(tmp_path) -> None:
    recorder = JsonlTraceRecorder(tmp_path)
    with pytest.raises(RuntimeError):
        with recorder.span(trace_id="trace", run_id="run", node=WorkflowNode.SEARCH):
            raise RuntimeError("boom")
    events = recorder.read("run")
    assert any(event.event_type is TraceEventType.ERROR for event in events)
