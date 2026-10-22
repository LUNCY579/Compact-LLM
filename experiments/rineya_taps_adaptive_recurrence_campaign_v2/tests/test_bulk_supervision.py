"""BULK-only bootstrap and parent deadline probes; no model or GPU invocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_entry, bulk_supervisor
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_supervision import (
    FILE_BARRIER_TREE_WORKER, NO_WINDOW, SUSPENDED, WORKER, sentinel, stop_owned,
)


@pytest.mark.parametrize("contained", [False, True])
def test_bulk_bootstrap_requires_containment_and_preserves_explicit_reader(tmp_path, monkeypatch, contained):
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"contained": contained}))
    source, out, admission = tmp_path / "no_weights", tmp_path / "not_run", tmp_path / "fixture_admission"
    monkeypatch.setattr(sys, "argv", ["bulk_entry", "--release", str(release), "--source", str(source),
                        "--out", str(out), "--admission", str(admission)])
    observed = []
    monkeypatch.setattr(bulk_entry.runpy, "run_module", lambda *a, **k: observed.append((a, k, sys.argv[:])))
    if contained:
        bulk_entry.main()
        module = "experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_native"
        assert observed == [((module,), {"run_name": "__main__"}, [module, "--source", str(source), "--out", str(out), "--admission", str(admission)])]
    else:
        with pytest.raises(RuntimeError, match="containment"):
            bulk_entry.main()
        assert not observed
    assert not out.exists()


@pytest.mark.skipif(os.name != "nt", reason="registered Windows job lane")
def test_bulk_actual_parent_hard_deadline_kills_only_cpu_tree(tmp_path, monkeypatch):
    package = tmp_path / "BULK_CPU_FIXTURE_NOT_NATIVE"
    package.mkdir()
    source, out, path = package / "no_weights", package / "run", package / "admission.json"
    admission = {"status": "READY_FOR_BULK_FP32_DISPATCH", "source": str(source.resolve()),
                 "interpreter": sys.executable, "dispatch_hashes": {}, "review_receipt": "fixture_review.json",
                 "gpu_uuid": "CPU_FAKE", "gpu_driver_version": "CPU_FAKE", "hard_deadline_seconds": 2}
    path.write_text(json.dumps(admission))
    (package / "fixture_review.json").write_text(json.dumps({"status": "PASS_BULK_FP32_DISPATCH",
        "admission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "fixture_not_real_review": True}))
    (package / "BUDGET_LEDGER.jsonl").write_text("")
    monkeypatch.setattr(bulk_supervisor, "ROOT", package)
    monkeypatch.setattr(bulk_supervisor, "PACKAGE", package)
    monkeypatch.setattr(bulk_supervisor, "sources", lambda: {})
    monkeypatch.setattr(bulk_supervisor, "git", lambda *a: "CPU_FIXTURE")
    monkeypatch.setattr(bulk_supervisor.psutil, "virtual_memory", lambda: SimpleNamespace(available=9 * 1024**3))
    monkeypatch.setattr(bulk_supervisor.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(bulk_supervisor, "gpu_inventory", lambda: {"uuid": "CPU_FAKE", "driver_version": "CPU_FAKE", "free_bytes": 6 * 1024**3})
    real_popen, owned, reached = subprocess.Popen, [], package / "members.json"
    def launch(command, **kwargs):
        assert command[command.index("-m") + 1].endswith(".bulk_entry") and "--stage" not in command
        assert kwargs["creationflags"] & SUSPENDED
        release = command[command.index("--release") + 1]
        child = real_popen([sys.executable, "-B", "-c", FILE_BARRIER_TREE_WORKER, release, str(reached), WORKER],
                          **{**kwargs, "creationflags": kwargs["creationflags"] | NO_WINDOW})
        owned.append(child)
        return child
    unrelated = sentinel(package / "unrelated.pid")
    monkeypatch.setattr(bulk_supervisor.subprocess, "Popen", launch)
    try:
        assert bulk_supervisor.run(source, out, path) == 2
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
