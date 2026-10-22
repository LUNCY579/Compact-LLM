"""RUNTIME-DEBUG-only bootstrap and parent deadline probes; no model or GPU invocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import runtime_debug_entry, runtime_debug_supervisor
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_supervision import (
    FILE_BARRIER_TREE_WORKER, NO_WINDOW, SUSPENDED, WORKER, sentinel, stop_owned,
)


@pytest.mark.parametrize("defect", ["missing", "wrong_hash", "wrong_scope", "consumed"])
def test_human_extra_attempt_gate_refuses_without_model_creation(tmp_path, monkeypatch, defect):
    package = tmp_path / "ONLY_CPU_AUTHORIZATION_FIXTURE"
    package.mkdir()
    worker = package / "runtime_debug.py"
    worker.write_text("# CPU fixture, never executed\n")
    allowance = package / "HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json"
    grant = {"status": "EXPLICIT_ADDITIONAL_ATTEMPT_AUTHORIZED", "lane": "GPU_FP32_RUNTIME_DEBUG",
        "attempts": 1, "max_native_worker_seconds": 300, "parent_run": "native-bulk-001",
        "training": False, "full_matrix_retry": False,
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(), "fixture_not_permission": True}
    if defect == "wrong_scope":
        grant["full_matrix_retry"] = True
    if defect != "missing":
        allowance.write_text(json.dumps(grant))
    (package / "BUDGET_LEDGER.jsonl").write_text(json.dumps({"event": "native_started", "lane": "GPU_FP32_RUNTIME_DEBUG"}) + "\n" if defect == "consumed" else "")
    admission = {"human_extension_path": str(allowance), "human_extension_sha256": hashlib.sha256(allowance.read_bytes()).hexdigest() if allowance.exists() else "missing"}
    if defect == "wrong_hash":
        admission["human_extension_sha256"] = "changed"
    monkeypatch.setattr(runtime_debug_supervisor, "PACKAGE", package)
    calls = []
    monkeypatch.setattr(runtime_debug_supervisor.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(PermissionError):
        runtime_debug_supervisor.validate_human_extension(admission)
    assert calls == []
    assert not (package / "native_worker.lock").exists()


@pytest.mark.parametrize("contained", [False, True])
def test_runtime_debug_bootstrap_requires_containment_and_preserves_explicit_reader(tmp_path, monkeypatch, contained):
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"contained": contained}))
    source, out, admission = tmp_path / "no_weights", tmp_path / "not_run", tmp_path / "fixture_admission"
    monkeypatch.setattr(sys, "argv", ["runtime_debug_entry", "--release", str(release), "--source", str(source),
                        "--out", str(out), "--admission", str(admission)])
    observed = []
    monkeypatch.setattr(runtime_debug_entry.runpy, "run_module", lambda *a, **k: observed.append((a, k, sys.argv[:])))
    if contained:
        runtime_debug_entry.main()
        module = "experiments.rineya_taps_adaptive_recurrence_campaign_v2.runtime_debug"
        assert observed == [((module,), {"run_name": "__main__"}, [module, "--source", str(source), "--out", str(out), "--admission", str(admission)])]
    else:
        with pytest.raises(RuntimeError, match="containment"):
            runtime_debug_entry.main()
        assert not observed
    assert not out.exists()


@pytest.mark.skipif(os.name != "nt", reason="registered Windows job lane")
def test_runtime_debug_actual_parent_hard_deadline_kills_only_cpu_tree(tmp_path, monkeypatch):
    package = tmp_path / "RUNTIME_DEBUG_CPU_FIXTURE_NOT_NATIVE"
    package.mkdir()
    source, out, path = package / "no_weights", package / "run", package / "admission.json"
    admission = {"status": "READY_FOR_RUNTIME_DEBUG_FP32_DISPATCH", "source": str(source.resolve()),
                 "interpreter": sys.executable, "dispatch_hashes": {}, "review_receipt": "fixture_review.json",
                 "gpu_uuid": "CPU_FAKE", "gpu_driver_version": "CPU_FAKE", "hard_deadline_seconds": 300, "idle_forward_deadline_seconds": 300}
    path.write_text(json.dumps(admission))
    (package / "fixture_review.json").write_text(json.dumps({"status": "PASS_RUNTIME_DEBUG_FP32_DISPATCH",
        "admission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "fixture_not_real_review": True}))
    (package / "BUDGET_LEDGER.jsonl").write_text("")
    monkeypatch.setattr(runtime_debug_supervisor, "ROOT", package)
    monkeypatch.setattr(runtime_debug_supervisor, "PACKAGE", package)
    monkeypatch.setattr(runtime_debug_supervisor, "sources", lambda: {})
    monkeypatch.setattr(runtime_debug_supervisor, "validate_human_extension", lambda admission: {"fixture_only": True})
    monkeypatch.setattr(runtime_debug_supervisor, "git", lambda *a: "CPU_FIXTURE")
    monkeypatch.setattr(runtime_debug_supervisor.psutil, "virtual_memory", lambda: SimpleNamespace(available=9 * 1024**3))
    monkeypatch.setattr(runtime_debug_supervisor.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(runtime_debug_supervisor, "gpu_inventory", lambda: {"uuid": "CPU_FAKE", "driver_version": "CPU_FAKE", "free_bytes": 6 * 1024**3})
    real_popen, owned, reached = subprocess.Popen, [], package / "members.json"
    def launch(command, **kwargs):
        assert command[command.index("-m") + 1].endswith(".runtime_debug_entry") and "--stage" not in command
        assert kwargs["creationflags"] & SUSPENDED
        assert kwargs["env"]["CUDA_LAUNCH_BLOCKING"] == "1"
        assert "RUNTIME_DEBUG_PARENT_STARTED_MONOTONIC" in kwargs["env"]
        release = command[command.index("--release") + 1]
        child = real_popen([sys.executable, "-B", "-c", FILE_BARRIER_TREE_WORKER, release, str(reached), WORKER],
                          **{**kwargs, "creationflags": kwargs["creationflags"] | NO_WINDOW})
        owned.append(child)
        return child
    unrelated = sentinel(package / "unrelated.pid")
    # Shorten only the fixture monotonic clock; never the registered production cap.
    real_clock = runtime_debug_supervisor.time.monotonic
    start_clock = real_clock()
    monkeypatch.setattr(runtime_debug_supervisor.time, "monotonic", lambda: start_clock + (real_clock() - start_clock) * 150)
    monkeypatch.setattr(runtime_debug_supervisor.subprocess, "Popen", launch)
    try:
        assert runtime_debug_supervisor.run(source, out, path) == 2
        members = json.loads(reached.read_text())
        assert members["model_modules"] == []
        assert not psutil.pid_exists(members["worker"]) and not psutil.pid_exists(members["child"])
        assert unrelated.poll() is None
        terminal = json.loads((out / "terminal_receipt.json").read_text())
        assert terminal["status"] == "SUPERVISOR_SAFETY_STOP" and terminal["stop_reason"] == "HARD_DEADLINE"
        assert terminal["native_worker_seconds"] > 2 and not (package / "native_worker.lock").exists()
    finally:
        for child in owned:
            stop_owned(child)
        stop_owned(unrelated)
