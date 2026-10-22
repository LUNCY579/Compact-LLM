"""Full parent-run missing-human-gate probe; no model/GPU/process creation."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import runtime_debug_supervisor as parent


def test_prepared_parent_run_refuses_before_creating_child_or_run(tmp_path, monkeypatch):
    original = parent.PACKAGE
    a = json.loads((original / "runtime_debug_admission_v1.json").read_text(encoding="utf-8"))
    package = tmp_path / "CPU_GATE_FIXTURE_NOT_AUTHORIZED"
    package.mkdir()
    out = package / "must_not_exist"
    path = package / "prepared_admission.json"
    a.update(review_receipt="fixture_review.json", human_extension_path=str(package / "HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json"))
    path.write_text(json.dumps(a), encoding="utf-8")
    (package / "fixture_review.json").write_text(json.dumps({"status": "PASS_RUNTIME_DEBUG_FP32_DISPATCH",
        "admission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "CPU_fixture_not_actual_review": True}), encoding="utf-8")
    (package / "BUDGET_LEDGER.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(parent, "PACKAGE", package)
    monkeypatch.setattr(parent.psutil, "virtual_memory", lambda: SimpleNamespace(available=9*1024**3))
    monkeypatch.setattr(parent, "gpu_inventory", lambda: {"uuid": a["gpu_uuid"], "driver_version": a["gpu_driver_version"], "free_bytes": 6*1024**3})
    calls = []
    monkeypatch.setattr(parent.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(PermissionError, match="NEEDS_EXPLICIT_ADDITIONAL_ATTEMPT"):
        parent.run(Path(a["source"]), out, path)
    assert calls == [] and not out.exists() and not (package / "native_worker.lock").exists()
    assert not (package / "HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json").exists()
