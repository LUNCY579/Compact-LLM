"""Actual failed-closure path set and actual parent no-permit probes; no model."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import (
    causal_replay_extended_authority as authority,
    cpu_causal_replay_native as replay,
    cpu_causal_replay_supervisor as parent,
)


def test_every_actual_failed90_path_is_accepted_with_current_verified_bytes():
    old = json.loads((replay.PACKAGE / "runs/native-cpu-causal-replay-001/admission.json").read_text())
    current = {name: replay.sha(replay.REPO / name) for name in old["dispatch_hashes"]}
    assert len(current) == 90 and replay.EXACT_QUALIFICATION_DEV in current
    replay.verify_source_closure(current)
    changed = dict(current)
    changed[replay.EXACT_QUALIFICATION_DEV] = "wrong"
    with pytest.raises(RuntimeError, match="source closure"):
        replay.verify_source_closure(changed)


def test_actual_parent_missing_actual_allowance_stops_before_lock_and_Popen(tmp_path, monkeypatch):
    package = tmp_path / "FAKE_PACKET_NO_HUMAN_APPROVAL"
    package.mkdir()
    source = package / "NO_MODEL"
    out = package / "NEVER_CREATED"
    path = package / "admission.json"
    values = {
        "status": "READY_FOR_CPU_CAUSAL_REPLAY_DISPATCH",
        "source": str(source.resolve()), "interpreter": sys.executable,
        "dispatch_hashes": {}, "review_receipt": "SYNTHETIC_NOT_APPROVAL.json",
        "hard_deadline_seconds": 4500, "requires_additional_human_authorization": True,
        "human_extension_path": str(package / "HUMAN_CAUSAL_REPLAY_EXTENDED_ALLOWANCE_001.json"),
        "human_extension_sha256": "ACTUAL_FILE_ABSENT",
    }
    path.write_text(json.dumps(values))
    (package / values["review_receipt"]).write_text(json.dumps({
        "status": "PASS_CPU_CAUSAL_REPLAY_DISPATCH",
        "admission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixture_not_real_review": True,
    }))
    (package / "BUDGET_LEDGER.jsonl").write_text("")
    monkeypatch.setattr(parent, "PACKAGE", package)
    monkeypatch.setattr(parent, "ROOT", package)
    monkeypatch.setattr(authority, "PACKAGE", package)
    monkeypatch.setattr(parent.psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * 1024**3))
    monkeypatch.setattr(parent.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("No allowance: Popen must not be reached")
    monkeypatch.setattr(parent.subprocess, "Popen", forbidden)
    with pytest.raises(PermissionError, match="actual human allowance absent"):
        parent.run(source, out, path)
    assert not calls and not out.exists() and not (package / "native_worker.lock").exists()
    assert (package / "BUDGET_LEDGER.jsonl").read_text() == ""
