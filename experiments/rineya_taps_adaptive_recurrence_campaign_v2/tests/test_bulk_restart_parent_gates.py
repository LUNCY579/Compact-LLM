"""Fabricated no-model packets probe parent and bootstrap refusal boundaries."""
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_restart_entry as entry
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_restart_supervisor as parent


def packet(tmp_path, monkeypatch):
    package = tmp_path / "FABRICATED_RESTART_PARENT_NOT_NATIVE"
    package.mkdir()
    source, out, admission = package / "NO_MODEL", package / "out", package / "admission.json"
    data = {"status": "READY_FOR_BULK_RESTART_FP32_DISPATCH", "source": str(source.resolve()),
            "interpreter": sys.executable, "dispatch_hashes": {}, "review_receipt": "FAKE_REVIEW.json",
            "human_extension_path": str(package / "NO_REAL_ALLOWANCE"),
            "requires_additional_human_authorization": True, "human_extension_sha256": "NOT_APPROVAL"}
    admission.write_text(json.dumps(data))
    (package / data["review_receipt"]).write_text(json.dumps({"status": "PASS_BULK_RESTART_FP32_DISPATCH",
        "admission_sha256": hashlib.sha256(admission.read_bytes()).hexdigest(), "synthetic_not_approval": True}))
    monkeypatch.setattr(parent, "PACKAGE", package)
    monkeypatch.setattr(parent, "ROOT", package)
    launches = []
    def forbidden(*args, **kwargs):
        launches.append((args, kwargs))
        raise AssertionError("negative probe must not launch")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    return package, source, out, admission, launches


def test_missing_actual_scope_refuses_before_output_lock_or_process(tmp_path, monkeypatch):
    package, source, out, admission, launches = packet(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="actual human allowance absent"):
        parent.run(source, out, admission)
    assert not launches and not out.exists() and not (package / "native_worker.lock").exists()


def test_existing_output_preserves_receipt_even_without_new_allowance(tmp_path, monkeypatch):
    package, source, out, admission, launches = packet(tmp_path, monkeypatch)
    out.mkdir()
    receipt = out / "terminal_receipt.json"
    old = b'{"status":"HISTORICAL_DO_NOT_REPLACE"}\n'
    receipt.write_bytes(old)
    with pytest.raises(FileExistsError, match="preserve historical"):
        parent.run(source, out, admission)
    assert receipt.read_bytes() == old and not launches
    assert not (package / "native_worker.lock").exists()


def test_unchanged_parent_ram_floor_refuses_without_process(tmp_path, monkeypatch):
    package, source, out, admission, launches = packet(tmp_path, monkeypatch)
    monkeypatch.setattr(parent, "validate_human_extension", lambda _: True)  # synthetic isolation only
    monkeypatch.setattr(parent.psutil, "virtual_memory", lambda: SimpleNamespace(available=7*1024**3-1))
    with pytest.raises(RuntimeError, match="host floor"):
        parent.run(source, out, admission)
    assert not launches and not out.exists() and not (package / "native_worker.lock").exists()


@pytest.mark.parametrize("contained,clock,allowed", [(False, "4", False), (True, "nan", False),
                                                   (True, "6", False), (True, "4", True)])
def test_bootstrap_containment_and_parent_clock_before_native_import(tmp_path, monkeypatch, contained, clock, allowed):
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"contained": contained}))
    admission = tmp_path / "SYNTHETIC_NOT_APPROVAL.json"
    admission.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["entry", "--release", str(release), "--source", str(tmp_path / "NO_MODEL"),
                                      "--out", str(tmp_path / "NO_RUN"), "--admission", str(admission)])
    monkeypatch.setattr(entry, "validate_human_scope", lambda _: True)  # no real grant is made
    monkeypatch.setattr(entry.time, "perf_counter", lambda: 5.)
    monkeypatch.setenv("BULK_RESTART_PARENT_STARTED_PERF_COUNTER", clock)
    imports = []
    monkeypatch.setattr(entry.runpy, "run_module", lambda *args, **kwargs: imports.append((args, kwargs)))
    if allowed:
        entry.main()
        assert imports[0][0] == ("experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_restart_native",)
    else:
        with pytest.raises(RuntimeError):
            entry.main()
        assert not imports
    assert not (tmp_path / "NO_RUN").exists()
