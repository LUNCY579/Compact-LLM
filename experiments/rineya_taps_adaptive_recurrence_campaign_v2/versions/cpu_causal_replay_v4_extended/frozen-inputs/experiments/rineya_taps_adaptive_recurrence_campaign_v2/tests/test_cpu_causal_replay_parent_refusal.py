"""Author parent-only negative probes. Fabricated packets grant no authority."""
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import cpu_causal_replay_supervisor as parent


def fixture_packet(tmp_path, monkeypatch, available=9 * 1024**3):
    package = tmp_path / "AUTHOR_CPU_FIXTURE_NOT_NATIVE"
    package.mkdir()
    source = package / "NO_MODEL"
    out = package / "run"
    admission = package / "admission.json"
    payload = {
        "status": "READY_FOR_CPU_CAUSAL_REPLAY_DISPATCH",
        "source": str(source.resolve()),
        "interpreter": sys.executable,
        "dispatch_hashes": {},
        "review_receipt": "SYNTHETIC_NOT_APPROVAL.json",
        "hard_deadline_seconds": 3600,
    }
    admission.write_text(json.dumps(payload))
    (package / payload["review_receipt"]).write_text(json.dumps({
        "status": "PASS_CPU_CAUSAL_REPLAY_DISPATCH",
        "admission_sha256": hashlib.sha256(admission.read_bytes()).hexdigest(),
        "fixture_not_real_review": True,
    }))
    (package / "BUDGET_LEDGER.jsonl").write_text("")
    monkeypatch.setattr(parent, "PACKAGE", package)
    monkeypatch.setattr(parent, "ROOT", package)
    monkeypatch.setattr(parent.psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(parent.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    observed = []
    def no_launch(*args, **kwargs):
        observed.append((args, kwargs))
        raise AssertionError("Author negative probe must never launch a process")
    monkeypatch.setattr(subprocess, "Popen", no_launch)
    return package, source, out, admission, observed


def test_existing_output_preserves_historical_receipt(tmp_path, monkeypatch):
    package, source, out, admission, launched = fixture_packet(tmp_path, monkeypatch)
    out.mkdir()
    receipt = out / "terminal_receipt.json"
    historical = b'{"status":"HISTORICAL_IMMUTABLE","marker":1776}\n'
    receipt.write_bytes(historical)
    sibling = out / "historical.log"
    sibling.write_bytes(b"preserved\n")
    try:
        parent.run(source, out, admission)
    except (FileExistsError, ValueError, RuntimeError):
        pass
    assert receipt.read_bytes() == historical
    assert sibling.read_bytes() == b"preserved\n"
    assert not launched
    assert not (package / "native_worker.lock").exists()


def test_parent_ram_below_unchanged_floor_never_launches(tmp_path, monkeypatch):
    available = int(8.4 * 1024**3)
    package, source, out, admission, launched = fixture_packet(tmp_path, monkeypatch, available)
    import pytest
    with pytest.raises(RuntimeError, match="7.4GiB floor plus1GiB"):
        parent.run(source, out, admission)
    assert not out.exists()
    assert not launched
    assert not (package / "native_worker.lock").exists()
