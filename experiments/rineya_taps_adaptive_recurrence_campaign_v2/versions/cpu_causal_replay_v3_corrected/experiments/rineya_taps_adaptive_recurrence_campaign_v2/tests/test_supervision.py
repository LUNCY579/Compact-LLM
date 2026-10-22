"""CPU-only Windows containment and release-barrier qualification.

Every process is an owned sleeping fixture; no v1 entrypoint or model loads.
Run with -B and -p no:cacheprovider in an exclusive campaign-local basetemp.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
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

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import native_entry
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import native_supervisor
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.supervision import (
    BasicLimits, ExtendedLimits, IOCounters, ThreadEntry, WindowsJob, resource_sample,
)


pytestmark = pytest.mark.skipif(os.name != "nt", reason="exact Windows Job lane")
ROOT = Path(__file__).resolve().parents[3]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SUSPENDED = 0x4

WORKER = r"""
import json, os, pathlib, subprocess, sys, time
if sys.stdin.readline() != "GO\n":
    raise SystemExit(61)
child_marker = pathlib.Path(sys.argv[1] + ".childpid")
child_code = "import os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(str(os.getpid())); t.replace(p); time.sleep(300)"
child = subprocess.Popen(
    [sys.executable, "-B", "-c", child_code, str(child_marker)],
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
)
started = time.monotonic()
while not child_marker.is_file():
    if time.monotonic() - started > 5:
        raise SystemExit(63)
    time.sleep(.02)
marker = pathlib.Path(sys.argv[1])
temporary = marker.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "worker": os.getpid(), "child": int(child_marker.read_text()),
    "child_launcher": child.pid,
    "model_modules": [n for n in ("torch", "transformers") if n in sys.modules],
}))
temporary.replace(marker)
time.sleep(300)
"""

PARENT = r"""
import json, pathlib, subprocess, sys, time
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.supervision import WindowsJob
job = WindowsJob()
p = subprocess.Popen(
    [sys.executable, "-B", "-c", sys.argv[1], sys.argv[2]],
    stdin=subprocess.PIPE,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x4,
)
job.assign(p)
job.resume_suspended(p)
p.stdin.write(b"GO\n")
p.stdin.flush()
pathlib.Path(sys.argv[2] + ".ownerpid").write_text(str(__import__("os").getpid()))
time.sleep(300)
"""

FILE_BARRIER_WORKER = r"""
import pathlib, sys, time
release, reached = map(pathlib.Path, sys.argv[1:3])
started = time.monotonic()
while not release.is_file():
    if time.monotonic() - started > 30:
        raise SystemExit(62)
    time.sleep(.02)
reached.write_text("CPU fixture entered after release; no model import")
time.sleep(300)
"""

FILE_BARRIER_TREE_WORKER = r"""
import io, pathlib, sys, time
release, marker, fixture_code = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
started = time.monotonic()
while not release.is_file():
    if time.monotonic() - started > 30:
        raise SystemExit(62)
    time.sleep(.02)
sys.stdin = io.StringIO("GO\n")
sys.argv = [sys.argv[0], marker]
exec(fixture_code)
"""


def wait_until(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("owned CPU fixture did not reach expected state")


def stop_owned(process):
    """Kill only this test's Popen handle, never a process enumerated by name."""
    members = []
    if process.poll() is None:
        try:
            members = [(p.pid, p.create_time()) for p in psutil.Process(process.pid).children(recursive=True)]
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=8)
    for identity in reversed(members):
        stop_owned_identity(identity)
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()


def stop_owned_identity(identity):
    try:
        proc = psutil.Process(identity[0])
        if proc.create_time() == identity[1]:
            proc.kill()
            proc.wait(timeout=8)
    except psutil.NoSuchProcess:
        pass


def is_running_identity(identity):
    try:
        proc = psutil.Process(identity[0])
        return proc.create_time() == identity[1] and proc.is_running()
    except psutil.NoSuchProcess:
        return False


def sentinel(marker):
    code = "import os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(str(os.getpid())); t.replace(p); time.sleep(300)"
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", code, str(marker)],
        creationflags=NO_WINDOW,
    )
    wait_until(marker.is_file)
    actual = psutil.Process(int(marker.read_text()))
    process.fixture_identity = (actual.pid, actual.create_time())
    return process


def assert_in_exact_job(job, pid):
    api = job.api
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
    api.IsProcessInJob.restype = wintypes.BOOL
    handle = api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    assert handle
    try:
        answer = wintypes.BOOL()
        assert api.IsProcessInJob(handle, job.handle, ctypes.byref(answer))
        assert answer.value, f"fixture PID {pid} escaped the exact campaign Job"
    finally:
        api.CloseHandle(handle)


def start_contained(job, marker):
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", WORKER, str(marker)],
        stdin=subprocess.PIPE, creationflags=NO_WINDOW | SUSPENDED,
    )
    try:
        time.sleep(.04)
        assert not marker.exists()
        assert psutil.Process(process.pid).children(recursive=True) == []
        job.assign(process)
        assert_in_exact_job(job, process.pid)
        job.resume_suspended(process)
        process.stdin.write(b"GO\n")
        process.stdin.flush()
        wait_until(marker.is_file)
        record = json.loads(marker.read_text())
        # Windows virtualenv python.exe can redirect to a child interpreter.
        tree = {process.pid, *(p.pid for p in psutil.Process(process.pid).children(recursive=True))}
        assert record["worker"] in tree and record["child"] in tree
        assert record["model_modules"] == []
        assert_in_exact_job(job, record["worker"])
        assert_in_exact_job(job, record["child"])
        members = [psutil.Process(record[key]) for key in ("worker", "child")]
        return process, [(p.pid, p.create_time()) for p in members]
    except BaseException:
        stop_owned(process)
        raise


def test_win64_structure_layout_matches_windows_abi():
    assert ctypes.sizeof(ctypes.c_void_p) == 8, "registered host is 64-bit"
    assert ctypes.sizeof(BasicLimits) == 64
    assert ctypes.sizeof(IOCounters) == 48
    assert ctypes.sizeof(ExtendedLimits) == 144
    assert ctypes.sizeof(ThreadEntry) == 28
    assert BasicLimits.LimitFlags.offset == 16
    assert BasicLimits.MinimumWorkingSetSize.offset == 24
    assert ExtendedLimits.IoInfo.offset == 64
    assert ExtendedLimits.ProcessMemoryLimit.offset == 112


def test_job_handle_is_non_inheritable_and_close_is_idempotent():
    job = WindowsJob()
    try:
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.GetHandleInformation.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        api.GetHandleInformation.restype = wintypes.BOOL
        flags = wintypes.DWORD()
        assert api.GetHandleInformation(job.handle, ctypes.byref(flags))
        assert flags.value & 1 == 0  # HANDLE_FLAG_INHERIT
    finally:
        job.close()
        job.close()
    assert job.handle is None


@pytest.mark.parametrize("method", ["close", "terminate"])
def test_job_stops_assigned_worker_and_descendant_only(tmp_path, method):
    unrelated = sentinel(tmp_path / "unrelated_close_terminate.pid")
    job, process, members = WindowsJob(), None, []
    try:
        process, members = start_contained(job, tmp_path / "members.json")
        assert process.poll() is None and all(is_running_identity(p) for p in members)
        getattr(job, method)()
        process.wait(timeout=8)
        wait_until(lambda: all(not is_running_identity(p) for p in members))
        assert unrelated.poll() is None and is_running_identity(unrelated.fixture_identity)
    finally:
        job.close()
        if process is not None:
            stop_owned(process)
        for identity in members:
            stop_owned_identity(identity)
        stop_owned(unrelated)


def test_forced_parent_death_closes_job_and_kills_descendants(tmp_path):
    marker = tmp_path / "parent_members.json"
    unrelated = sentinel(tmp_path / "unrelated_parent_death.pid")
    parent, owner, worker, child = None, None, None, None
    try:
        parent = subprocess.Popen(
            [sys.executable, "-B", "-c", PARENT, WORKER, str(marker)],
            cwd=ROOT, creationflags=NO_WINDOW,
        )
        wait_until(marker.is_file)
        owner_marker = Path(str(marker) + ".ownerpid")
        wait_until(owner_marker.is_file)
        record = json.loads(marker.read_text())
        assert record["model_modules"] == []
        actual_owner = psutil.Process(int(owner_marker.read_text()))
        owner = (actual_owner.pid, actual_owner.create_time())
        worker_proc, child_proc = psutil.Process(record["worker"]), psutil.Process(record["child"])
        worker = (worker_proc.pid, worker_proc.create_time())
        child = (child_proc.pid, child_proc.create_time())
        actual_owner.kill()  # kill the real interpreter that owns the Job handle
        actual_owner.wait(timeout=8)
        parent.wait(timeout=8)
        wait_until(lambda: not is_running_identity(worker) and not is_running_identity(child))
        assert unrelated.poll() is None and is_running_identity(unrelated.fixture_identity)
    finally:
        if parent is not None:
            stop_owned(parent)
        for identity in (owner, worker, child):
            if identity is not None:
                stop_owned_identity(identity)
        stop_owned(unrelated)


def test_resource_sample_includes_owned_descendants_and_excludes_sentinel(tmp_path):
    unrelated = sentinel(tmp_path / "unrelated_resources.pid")
    job, process, members = WindowsJob(), None, []
    try:
        process, members = start_contained(job, tmp_path / "resources_members.json")
        sample = resource_sample(process)
        pids = {row["pid"] for row in sample["members"]}
        assert process.pid in pids and all(p[0] in pids for p in members)
        assert unrelated.pid not in pids
        assert unrelated.fixture_identity[0] not in pids
        assert sample["rss_bytes"] == sum(row["rss_bytes"] for row in sample["members"])
        assert sample["rss_bytes"] > 0 and sample["host_available_bytes"] > 0
    finally:
        job.close()
        if process is not None:
            stop_owned(process)
        for identity in members:
            stop_owned_identity(identity)
        stop_owned(unrelated)


def test_native_entry_rejects_false_containment_before_module_dispatch(tmp_path, monkeypatch):
    release = tmp_path / "false_release.json"
    release.write_text(json.dumps({"contained": False}))
    dispatched = []
    monkeypatch.setattr(sys, "argv", [
        "native_entry", "--release", str(release), "--source", str(tmp_path / "no_weights"),
        "--out", str(tmp_path / "not_run"),
    ])
    monkeypatch.setattr(native_entry.runpy, "run_module", lambda *a, **k: dispatched.append((a, k)))
    with pytest.raises(RuntimeError, match="containment not admitted"):
        native_entry.main()
    assert dispatched == []


def test_native_entry_missing_release_times_out_before_module_dispatch(tmp_path, monkeypatch):
    dispatched = []
    ticks = iter((0., 31.))
    monkeypatch.setattr(sys, "argv", [
        "native_entry", "--release", str(tmp_path / "absent.json"),
        "--source", str(tmp_path / "no_weights"), "--out", str(tmp_path / "not_run"),
    ])
    monkeypatch.setattr(native_entry.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(native_entry.runpy, "run_module", lambda *a, **k: dispatched.append((a, k)))
    with pytest.raises(RuntimeError, match="parent did not establish containment"):
        native_entry.main()
    assert dispatched == []


def test_native_entry_positive_release_preserves_v1_command_without_executing_it(tmp_path, monkeypatch):
    release = tmp_path / "cpu_fixture_release.json"
    source, out = tmp_path / "no_weights", tmp_path / "not_run"
    release.write_text(json.dumps({"contained": True, "token": "CPU_FIXTURE_ONLY"}))
    dispatched = []
    monkeypatch.setattr(sys, "argv", [
        "native_entry", "--release", str(release), "--source", str(source), "--out", str(out),
    ])
    monkeypatch.setattr(native_entry.runpy, "run_module", lambda *a, **k: dispatched.append((a, k, sys.argv[:])))
    native_entry.main()
    assert dispatched == [(("experiments.rineya_taps_adaptive_recurrence_v1",),
                           {"run_name": "__main__"},
                           ["experiments.rineya_taps_adaptive_recurrence_v1", "native-dev",
                            "--source", str(source), "--out", str(out)])]
    assert not out.exists()


def test_real_supervisor_assignment_failure_kills_unassigned_worker_before_release(tmp_path, monkeypatch):
    """Actual dispatcher cleanup, with fake admission and a no-model child."""
    package = tmp_path / "CPU_FIXTURE_CAMPAIGN"
    package.mkdir()
    source, out = package / "no_weights", package / "fixture_run"
    admission_path = package / "fixture_admission.json"
    admission = {
        "status": "READY_FOR_REFERENCE_DISPATCH", "source": str(source.resolve()),
        "interpreter": sys.executable, "dispatch_hashes": {},
        "review_receipt": "fixture_review.json", "hard_deadline_seconds": 10,
        "kind": "CPU_TEST_FIXTURE_NOT_A_NATIVE_PERMISSION",
    }
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    review = {
        "status": "PASS_REFERENCE_DISPATCH",
        "admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "kind": "FABRICATED_FIXTURE_NOT_INDEPENDENT_REVIEW",
    }
    (package / "fixture_review.json").write_text(json.dumps(review))
    original_popen = subprocess.Popen
    owned = []
    reached = package / "must_not_reach_import.txt"

    def fixture_popen(command, **kwargs):
        assert kwargs["creationflags"] & SUSPENDED
        release = command[command.index("--release") + 1]
        process = original_popen(
            [sys.executable, "-B", "-c", FILE_BARRIER_WORKER, release, str(reached)],
            **{**kwargs, "creationflags": kwargs["creationflags"] | NO_WINDOW},
        )
        owned.append(process)
        return process

    class RefusingJob(WindowsJob):
        def assign(self, process):
            time.sleep(.04)  # measurable failure cost even on a coarse Windows clock
            raise OSError("deliberate CPU-fixture assignment failure")

    monkeypatch.setattr(native_supervisor, "PACKAGE", package)
    monkeypatch.setattr(native_supervisor, "sources", lambda: {"CPU_FIXTURE": "no_model"})
    monkeypatch.setattr(native_supervisor, "git", lambda *args: "CPU_FIXTURE")
    monkeypatch.setattr(native_supervisor, "WindowsJob", RefusingJob)
    monkeypatch.setattr(native_supervisor.psutil, "virtual_memory", lambda: SimpleNamespace(available=9 * 1024**3))
    monkeypatch.setattr(native_supervisor.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(native_supervisor.subprocess, "Popen", fixture_popen)
    try:
        started = time.monotonic()
        assert native_supervisor.run(source, out, admission_path) == 2
        assert time.monotonic() - started < 5
        assert len(owned) == 1 and owned[0].poll() is not None
        assert not (out / "release.json").exists()
        assert not reached.exists()
        assert not (package / "native_worker.lock").exists()
        receipt = json.loads((out / "terminal_receipt.json").read_text())
        assert "assignment failure" in receipt["error"]
        assert 0 < receipt["native_worker_seconds"] < 5
        rows = [json.loads(line) for line in (package / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["native_completed"]
    finally:
        for process in owned:
            stop_owned(process)


def test_real_supervisor_hard_deadline_stops_cpu_tree_and_records_terminal_cost(tmp_path, monkeypatch):
    """Exercise actual dispatcher release/monitor/termination with no model."""
    package = tmp_path / "CPU_DEADLINE_FIXTURE_CAMPAIGN"
    package.mkdir()
    source, out = package / "no_weights", package / "fixture_run"
    admission_path = package / "fixture_admission.json"
    admission = {
        "status": "READY_FOR_REFERENCE_DISPATCH", "source": str(source.resolve()),
        "interpreter": sys.executable, "dispatch_hashes": {},
        "review_receipt": "fixture_review.json", "hard_deadline_seconds": 2,
        "kind": "CPU_TEST_FIXTURE_NOT_A_NATIVE_PERMISSION",
    }
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    (package / "fixture_review.json").write_text(json.dumps({
        "status": "PASS_REFERENCE_DISPATCH",
        "admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "kind": "FABRICATED_FIXTURE_NOT_INDEPENDENT_REVIEW",
    }))
    original_popen = subprocess.Popen
    owned = []
    reached = package / "actual_cpu_members.json"
    unrelated = sentinel(package / "unrelated_deadline.pid")

    def fixture_popen(command, **kwargs):
        assert kwargs["creationflags"] & SUSPENDED
        release = command[command.index("--release") + 1]
        process = original_popen(
            [sys.executable, "-B", "-c", FILE_BARRIER_TREE_WORKER,
             release, str(reached), WORKER],
            **{**kwargs, "creationflags": kwargs["creationflags"] | NO_WINDOW},
        )
        owned.append(process)
        return process

    monkeypatch.setattr(native_supervisor, "PACKAGE", package)
    monkeypatch.setattr(native_supervisor, "sources", lambda: {"CPU_FIXTURE": "no_model"})
    monkeypatch.setattr(native_supervisor, "git", lambda *args: "CPU_FIXTURE")
    monkeypatch.setattr(native_supervisor.psutil, "virtual_memory", lambda: SimpleNamespace(available=9 * 1024**3))
    monkeypatch.setattr(native_supervisor.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(native_supervisor.subprocess, "Popen", fixture_popen)
    try:
        assert native_supervisor.run(source, out, admission_path) == 2
        assert len(owned) == 1 and owned[0].poll() is not None
        record = json.loads(reached.read_text())
        assert record["model_modules"] == []
        assert not psutil.pid_exists(record["worker"])
        assert not psutil.pid_exists(record["child"])
        assert unrelated.poll() is None and is_running_identity(unrelated.fixture_identity)
        assert json.loads((out / "release.json").read_text())["contained"] is True
        assert not (package / "native_worker.lock").exists()
        receipt = json.loads((out / "terminal_receipt.json").read_text())
        assert receipt["status"] == "SUPERVISOR_SAFETY_STOP"
        assert receipt["stop_reason"] == "HARD_DEADLINE"
        assert 2 <= receipt["native_worker_seconds"] < 12
        assert receipt["peak_rss_bytes"] > 0
        rows = [json.loads(line) for line in (package / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["native_started", "native_completed"]
        assert rows[-1]["native_worker_seconds"] == receipt["native_worker_seconds"]
        samples = [json.loads(line) for line in (out / "resources.jsonl").read_text().splitlines()]
        assert samples and max(row["rss_bytes"] for row in samples) == receipt["peak_rss_bytes"]
    finally:
        for process in owned:
            stop_owned(process)
        stop_owned(unrelated)
