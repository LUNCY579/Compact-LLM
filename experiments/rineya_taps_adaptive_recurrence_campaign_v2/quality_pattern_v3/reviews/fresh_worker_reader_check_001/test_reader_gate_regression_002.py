"""Independent held-d07 regressions. All native/source/asset plumbing is mock."""
import importlib.util
from pathlib import Path

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_analysis as a

_path = Path(__file__).with_name("test_reader_gate_witness_001.py")
_spec = importlib.util.spec_from_file_location("reviewer_old_witness", _path)
_old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_old)


def setup(tmp_path, monkeypatch):
    values = _old.fixture(tmp_path, monkeypatch)
    run, admission, terminal, receipt, samples, ledger, opened = values
    terminal.update(supervisor_wall_seconds=101., started_utc="2026-01-01T00:00:00+00:00", ended_utc="2026-01-01T00:02:00+00:00")
    samples[0]["elapsed_monotonic"] = 50.
    ledger[0]["timestamp_utc"] = terminal["started_utc"]; ledger[1]["ended_utc"] = terminal["ended_utc"]
    return values


@pytest.mark.parametrize("location,key,value", [
    ("sample", "rss_bytes", float("nan")), ("sample", "host_available_bytes", float("inf")),
    ("sample", "worker_seconds", -1.), ("sample", "elapsed_monotonic", float("nan")),
    ("terminal", "native_worker_seconds", -1.), ("terminal", "supervisor_wall_seconds", float("nan")),
    ("body", "parent_elapsed_seconds", -1.), ("body", "parent_elapsed_seconds", float("inf")),
    ("terminal", "ended_utc", "2025-01-01T00:00:00+00:00"), ("terminal", "started_utc", "2026-01-01T00:00:00"),
])
def test_parent_and_cost_witness_now_refuses(tmp_path, monkeypatch, location, key, value):
    run, admission, terminal, receipt, samples, ledger, opened = setup(tmp_path, monkeypatch)
    # Isolate parent/cost gate; actual five-file export join is exercised separately.
    monkeypatch.setattr(a, "validate_native_population", lambda admission, population: population)
    assert a.load_native(run)["counts"]["failed_forwards"] == 2
    target = samples[0] if location == "sample" else terminal if location == "terminal" else receipt
    target[key] = value
    if key == "native_worker_seconds": ledger[1][key] = value
    with pytest.raises(RuntimeError): a.load_native(run)


def test_original_missing_export_witness_now_nulls_primary_without_labels(tmp_path, monkeypatch):
    run, admission, terminal, receipt, samples, ledger, opened = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "DATA", a.V3 / "data")
    for field in ("scientific_bank", "preservation_bank", "scientific_token_binding", "preservation_token_binding", "prompt_export_manifest"):
        admission[field + "_path"] = str(a.DATA / ("ABSENT_" + field))
        admission[field + "_sha256"] = "0" * 64
    def forbidden(_): raise AssertionError("offline labels reached")
    monkeypatch.setattr(a, "read_offline_labels", forbidden)
    report = a.run(run, a.V3 / "analyses/rejected")
    assert report["status"] == "INCOMPLETE_NO_EFFICACY" and report["primary"] is None
    assert report["errors"] and not report["fresh_labels_read"]
