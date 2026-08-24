from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rineya_significance_lab.benchmarks import (
    ExposureTraceSpec,
    build_exposure_events,
    write_exposure_trace,
)
from rineya_significance_lab.policies import PolicyContext, policy_from_baseline
from rineya_significance_lab.traces import (
    TraceManifest,
    TraceReader,
    TraceValidationError,
    TraceWriter,
    replay_trace,
    validate_trace,
)


def test_npz_jsonl_trace_round_trip_and_validation(tmp_path: Path) -> None:
    spec = ExposureTraceSpec(rounds=4, layers=2, kv_heads=2, budget=2)
    root = write_exposure_trace(tmp_path / "trace", spec)
    report = validate_trace(root)
    assert report["status"] == "PASS"
    events = list(TraceReader(root).events())
    assert len(events) == 4
    assert events[-1].state_before.scores.dtype == np.float32
    assert events[-1].score_delta.dtype == np.float32


def test_trace_checksum_tampering_is_detected(tmp_path: Path) -> None:
    root = write_exposure_trace(tmp_path / "trace", ExposureTraceSpec(rounds=2))
    shard = root / "event_00000.npz"
    with shard.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(TraceValidationError, match="checksum"):
        list(TraceReader(root).events())


def test_full_candidate_replay_allows_selection_divergence_but_labels_open_loop(tmp_path: Path) -> None:
    spec = ExposureTraceSpec(rounds=6, budget=2)
    root = write_exposure_trace(tmp_path / "trace", spec)
    context = PolicyContext(budget=2, chunk_size=2, seed=17)
    raw = replay_trace(root, policy_from_baseline("B2"), context)
    normalized = replay_trace(root, policy_from_baseline("B3"), context)
    assert raw.valid and normalized.valid
    assert raw.status == "COMPLETE_OPEN_LOOP"
    assert "does not model" in raw.limitation
    assert any(not row["matches_recorded"] for row in raw.selections)
    assert all(row["matches_recorded"] for row in normalized.selections)


def test_on_policy_trace_stops_at_first_counterfactual_divergence(tmp_path: Path) -> None:
    spec = ExposureTraceSpec(case_id="on-policy", rounds=6, budget=2)
    manifest = TraceManifest(
        case_id=spec.case_id,
        support="ON_POLICY",
        execution_mode="OPEN_LOOP_TRACE_SIMULATION",
        score_semantics={"dtype": "float32"},
        information_access={
            "answer": False,
            "future_tokens": False,
            "future_hidden_state": False,
            "oracle_evidence": False,
        },
    )
    writer = TraceWriter(tmp_path / "trace", manifest)
    for event in build_exposure_events(spec):
        writer.write_event(event)
    outcome = replay_trace(
        tmp_path / "trace",
        policy_from_baseline("B2"),
        PolicyContext(budget=2, chunk_size=2, seed=17),
    )
    assert not outcome.valid
    assert outcome.status == "DIVERGED_ON_POLICY_UNSUPPORTED"
    assert outcome.diverged_at_event is not None


def test_trace_manifest_forbids_future_or_answer_access() -> None:
    payload = {
        "schema_version": "1.0",
        "case_id": "bad",
        "support": "FULL_CANDIDATE",
        "execution_mode": "OPEN_LOOP_TRACE_SIMULATION",
        "score_semantics": {},
        "information_access": {
            "answer": True,
            "future_tokens": False,
            "future_hidden_state": False,
            "oracle_evidence": False,
        },
    }
    with pytest.raises(ValueError, match="forbidden"):
        TraceManifest.from_dict(payload)
