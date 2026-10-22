"""GPU-parent admission/containment tests using CPU fixtures only.

gpu_inventory is stubbed; GPU/model/native modules never execute. Fabricated
admission/review files are unit-test inputs, not independent authorization.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import gpu_entry, gpu_supervisor, native_supervisor
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.supervision import WindowsJob
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_supervision import (
    WORKER, FILE_BARRIER_TREE_WORKER, FILE_BARRIER_WORKER, NO_WINDOW,
    SUSPENDED, sentinel, stop_owned, is_running_identity,
)


pytestmark = pytest.mark.skipif(os.name != "nt", reason="registered Windows parent lane")
GIB = 1024**3
NATIVE_MODULE = "experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_native"


def cpu_contract(tmp_path, monkeypatch, module=gpu_supervisor):
    package = tmp_path / "CPU_GPU_FIXTURE_NOT_NATIVE"
    package.mkdir()
    source, out = package / "no_weights", package / "run"
    admission_path = package / "fixture_admission.json"
    gpu = module is gpu_supervisor
    admission = {
        "status": "READY_FOR_GPU_FP32_DISPATCH" if gpu else "READY_FOR_REFERENCE_DISPATCH",
        "source": str(source.resolve()), "interpreter": sys.executable,
        "dispatch_hashes": {}, "review_receipt": "fixture_review.json",
        "hard_deadline_seconds": 2, "gpu_uuid": "CPU_FIXTURE_GPU_UUID",
        "gpu_driver_version": "CPU_FIXTURE_DRIVER",
        "kind": "CPU_TEST_FIXTURE_NOT_NATIVE_PERMISSION",
    }
    review = {"status": "PASS_GPU_FP32_DISPATCH" if gpu else "PASS_REFERENCE_DISPATCH",
              "kind": "FABRICATED_FIXTURE_NOT_INDEPENDENT_REVIEW"}
    inventory = {"index": 0, "uuid": admission["gpu_uuid"], "name": "CPU_FIXTURE_NOT_A_GPU",
                 "total_bytes": 8 * GIB, "free_bytes": 6 * GIB,
                 "utilization_percent": 0, "driver_version": admission["gpu_driver_version"]}

    def freeze():
        admission_path.write_text(json.dumps(admission), encoding="utf-8")
        review["admission_sha256"] = hashlib.sha256(admission_path.read_bytes()).hexdigest()
        (package / admission["review_receipt"]).write_text(json.dumps(review), encoding="utf-8")

    freeze()
    (package / "BUDGET_LEDGER.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "PACKAGE", package)
    monkeypatch.setattr(module, "ROOT", package)
    monkeypatch.setattr(module, "sources", lambda: {"CPU_FIXTURE": "no_model"})
    monkeypatch.setattr(module, "git", lambda *args: "CPU_FIXTURE")
    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: SimpleNamespace(available=9 * GIB))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    if gpu:
        monkeypatch.setattr(module, "gpu_inventory", lambda: dict(inventory))
    return SimpleNamespace(package=package, source=source, out=out, admission_path=admission_path,
                           admission=admission, review=review, inventory=inventory, freeze=freeze)


@pytest.mark.parametrize("failure", [
    "wrong_review_status", "wrong_review_digest", "changed_source", "host_ram",
    "gpu_vram", "gpu_uuid", "gpu_driver", "source_identity", "interpreter_identity",
    "wrong_admission_status", "unknown_lock", "outside_scope",
])
def test_gpu_admission_refuses_before_launch_and_preserves_unowned_lock(tmp_path, monkeypatch, failure):
    contract = cpu_contract(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(gpu_supervisor.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    expected = RuntimeError
    if failure == "wrong_review_status":
        contract.review["status"] = "REVIEW_NOT_PASSED"
        contract.freeze()
    elif failure == "wrong_review_digest":
        contract.review["admission_sha256"] = "incorrect"
        (contract.package / "fixture_review.json").write_text(json.dumps(contract.review))
    elif failure == "changed_source":
        bound = contract.package / "bound_dispatch.py"
        bound.write_text("CPU fixture source before change")
        contract.admission["dispatch_hashes"]["bound_dispatch.py"] = hashlib.sha256(bound.read_bytes()).hexdigest()
        contract.freeze()
        bound.write_text("CPU fixture changed after review")
    elif failure == "host_ram":
        monkeypatch.setattr(gpu_supervisor.psutil, "virtual_memory", lambda: SimpleNamespace(available=7 * GIB - 1))
    elif failure == "gpu_vram":
        contract.inventory["free_bytes"] = 6 * GIB - 1
    elif failure in ("gpu_uuid", "gpu_driver"):
        contract.inventory["uuid" if failure == "gpu_uuid" else "driver_version"] = "different"
    elif failure == "source_identity":
        contract.admission["source"] = str(contract.package / "wrong_source")
        contract.freeze()
    elif failure == "interpreter_identity":
        contract.admission["interpreter"] = "wrong_python.exe"
        contract.freeze()
    elif failure == "wrong_admission_status":
        contract.admission["status"] = "NOT_READY"
        contract.freeze()
    elif failure == "unknown_lock":
        (contract.package / "native_worker.lock").write_text("OTHER_OWNER_UNCHANGED")
        expected = FileExistsError
    elif failure == "outside_scope":
        contract.out = tmp_path / "outside_campaign"
        expected = ValueError
    before_ledger = (contract.package / "BUDGET_LEDGER.jsonl").read_bytes()
    before_modules = set(sys.modules)
    with pytest.raises(expected):
        gpu_supervisor.run(contract.source, contract.out, contract.admission_path)
    assert calls == [] and not contract.out.exists()
    assert (contract.package / "BUDGET_LEDGER.jsonl").read_bytes() == before_ledger
    assert NATIVE_MODULE not in set(sys.modules) - before_modules
    if failure == "unknown_lock":
        assert (contract.package / "native_worker.lock").read_text() == "OTHER_OWNER_UNCHANGED"


def test_cpu_parent_requires_historical_floor_plus_import_reserve(tmp_path, monkeypatch):
    contract = cpu_contract(tmp_path, monkeypatch, native_supervisor)
    calls = []
    monkeypatch.setattr(native_supervisor.psutil, "virtual_memory", lambda: SimpleNamespace(available=int(8.4 * GIB) - 1))
    monkeypatch.setattr(native_supervisor.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(RuntimeError, match="plus1GiB import reserve"):
        native_supervisor.run(contract.source, contract.out, contract.admission_path)
    assert calls == [] and not contract.out.exists()
    assert not (contract.package / "native_worker.lock").exists()


@pytest.mark.parametrize("release_kind", ["false", "missing"])
def test_gpu_bootstrap_refuses_before_runpy(tmp_path, monkeypatch, release_kind):
    release = tmp_path / "release.json"
    if release_kind == "false":
        release.write_text(json.dumps({"contained": False}), encoding="utf-8")
        reason = "GPU containment not admitted"
    else:
        ticks = iter((0., 31.))
        monkeypatch.setattr(gpu_entry.time, "monotonic", lambda: next(ticks))
        reason = "parent did not establish GPU worker containment"
    dispatched = []
    monkeypatch.setattr(sys, "argv", [
        "gpu_entry", "--release", str(release), "--source", str(tmp_path / "no_weights"),
        "--out", str(tmp_path / "not_run"), "--admission", str(tmp_path / "fixture_admission"),
    ])
    monkeypatch.setattr(gpu_entry.runpy, "run_module", lambda *a, **k: dispatched.append((a, k)))
    with pytest.raises(RuntimeError, match=reason):
        gpu_entry.main()
    assert dispatched == [] and not (tmp_path / "not_run").exists()


@pytest.mark.parametrize("stage", ["qualification", "dev"])
def test_gpu_bootstrap_preserves_explicit_successor_stage_without_execution(tmp_path, monkeypatch, stage):
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"contained": True}), encoding="utf-8")
    source, out, admission = tmp_path / "no_weights", tmp_path / "not_run", tmp_path / "fixture_admission"
    dispatched = []
    monkeypatch.setattr(sys, "argv", [
        "gpu_entry", "--release", str(release), "--source", str(source),
        "--out", str(out), "--admission", str(admission), "--stage", stage,
    ])
    monkeypatch.setattr(gpu_entry.runpy, "run_module", lambda *a, **k: dispatched.append((a, k, sys.argv[:])))
    gpu_entry.main()
    assert dispatched == [((NATIVE_MODULE,), {"run_name": "__main__"},
                           [NATIVE_MODULE, "--stage", stage, "--source", str(source),
                            "--out", str(out), "--admission", str(admission)])]
    assert not out.exists()


def launch_cpu_substitute(monkeypatch, contract, tree):
    original_popen, owned = subprocess.Popen, []
    reached = contract.package / "actual_cpu_members.json"

    def fixture_popen(command, **kwargs):
        assert kwargs["creationflags"] & SUSPENDED
        assert command[command.index("-m") + 1].endswith(".gpu_entry")
        release = command[command.index("--release") + 1]
        fixture = FILE_BARRIER_TREE_WORKER if tree else FILE_BARRIER_WORKER
        command = [sys.executable, "-B", "-c", fixture, release, str(reached)]
        if tree:
            command.append(WORKER)
        process = original_popen(command, **{**kwargs, "creationflags": kwargs["creationflags"] | NO_WINDOW})
        owned.append(process)
        return process

    monkeypatch.setattr(gpu_supervisor.subprocess, "Popen", fixture_popen)
    return owned, reached


def test_gpu_assignment_refusal_kills_suspended_child_without_release(tmp_path, monkeypatch):
    contract = cpu_contract(tmp_path, monkeypatch)

    class RefusingJob(WindowsJob):
        def assign(self, process):
            time.sleep(.04)
            raise OSError("deliberate CPU GPU-parent assignment refusal")

    monkeypatch.setattr(gpu_supervisor, "WindowsJob", RefusingJob)
    owned, reached = launch_cpu_substitute(monkeypatch, contract, tree=False)
    try:
        assert gpu_supervisor.run(contract.source, contract.out, contract.admission_path) == 2
        assert len(owned) == 1 and owned[0].poll() is not None
        assert not reached.exists() and not (contract.out / "release.json").exists()
        assert not (contract.package / "native_worker.lock").exists()
        receipt = json.loads((contract.out / "terminal_receipt.json").read_text(encoding="utf-8"))
        assert "assignment refusal" in receipt["error"]
        assert 0 < receipt["native_worker_seconds"] < 5
        ledger = [json.loads(x) for x in (contract.package / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
        assert [row["event"] for row in ledger] == ["native_completed"]
    finally:
        for process in owned:
            stop_owned(process)


def test_gpu_parent_deadline_contains_cpu_tree_preserves_sentinel_and_charges_terminal(tmp_path, monkeypatch):
    contract = cpu_contract(tmp_path, monkeypatch)
    # Exact GPU parent admission boundaries are allowed; inventory is entirely fake.
    monkeypatch.setattr(gpu_supervisor.psutil, "virtual_memory", lambda: SimpleNamespace(available=7 * GIB))
    unrelated = sentinel(contract.package / "unrelated_gpu_parent.pid")
    owned, reached = launch_cpu_substitute(monkeypatch, contract, tree=True)
    before_modules = set(sys.modules)
    try:
        assert gpu_supervisor.run(contract.source, contract.out, contract.admission_path) == 2
        record = json.loads(reached.read_text(encoding="utf-8"))
        assert record["model_modules"] == []
        assert len(owned) == 1 and owned[0].poll() is not None
        assert not psutil.pid_exists(record["worker"]) and not psutil.pid_exists(record["child"])
        assert unrelated.poll() is None and is_running_identity(unrelated.fixture_identity)
        assert not (contract.package / "native_worker.lock").exists()
        receipt = json.loads((contract.out / "terminal_receipt.json").read_text(encoding="utf-8"))
        assert receipt["lane"] == "GPU_FP32_SUCCESSOR"
        assert receipt["status"] == "SUPERVISOR_SAFETY_STOP" and receipt["stop_reason"] == "HARD_DEADLINE"
        assert 2 <= receipt["native_worker_seconds"] < 12 and receipt["peak_rss_bytes"] > 0
        ledger = [json.loads(x) for x in (contract.package / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
        assert [row["event"] for row in ledger] == ["native_started", "native_completed"]
        assert ledger[-1]["native_worker_seconds"] == receipt["native_worker_seconds"]
        samples = [json.loads(x) for x in (contract.out / "resources.jsonl").read_text().splitlines()]
        assert max(row["rss_bytes"] for row in samples) == receipt["peak_rss_bytes"]
        assert NATIVE_MODULE not in set(sys.modules) - before_modules
    finally:
        for process in owned:
            stop_owned(process)
        stop_owned(unrelated)
