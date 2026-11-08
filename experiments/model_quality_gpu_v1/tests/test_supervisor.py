"""Real subprocess mechanics on CPU; no tensor runtime or pretrained assets."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from experiments.model_quality_gpu_v1 import integrity as I
from experiments.model_quality_gpu_v1 import supervisor as S


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv(S.CONTEXT_ENV, raising=False)


def python(code, *args, cwd=I.ROOT):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=20)


def test_public_import_is_stdlib_only():
    result = python("""
import sys
class RefuseHeavy:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','numpy','psutil','transformers','safetensors'}:
            raise AssertionError('heavy public import: ' + fullname)
sys.meta_path.insert(0, RefuseHeavy())
from experiments.model_quality_gpu_v1.run import main
assert main(['--help']) is None
""")
    assert result.returncode == 0, result.stderr
    assert "qualify" in result.stdout


def test_private_worker_refuses_before_import_or_output(tmp_path):
    out = tmp_path / "must-not-exist"
    result = python("""
import sys
class RefuseHeavy:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','numpy','psutil','transformers','safetensors'}:
            raise AssertionError('heavy bypass import: ' + fullname)
sys.meta_path.insert(0, RefuseHeavy())
from experiments.model_quality_gpu_v1.run import main
raise SystemExit(main(['qualify','--out',sys.argv[1],'--_worker']))
""", out)
    assert result.returncode == 2
    assert "supervisor context absent" in result.stderr
    assert not out.exists()


def test_context_claim_cwd_one_use_and_no_native_import(tmp_path, monkeypatch):
    out = tmp_path / "attempt"
    code = """
import os, sys
from pathlib import Path
from experiments.model_quality_gpu_v1 import supervisor as S
from experiments.model_quality_gpu_v1 import integrity as I
token = os.environ[S.CONTEXT_ENV]
S.claim_worker(Path(sys.argv[1]))
assert Path.cwd().resolve() == I.ROOT
assert 'torch' not in sys.modules
os.environ[S.CONTEXT_ENV] = token
try:
    S.claim_worker(Path(sys.argv[1]))
except FileExistsError:
    pass
else:
    raise AssertionError('worker context reused')
print('context accepted once')
"""
    monkeypatch.chdir(tmp_path)
    assert S.bounded_child([sys.executable, "-c", code, str(out)], out, 15) == 0
    assert I.read_json(out / "launch.json")["cwd"] == str(I.ROOT)
    assert I.read_json(out / "worker_claim.json")["supervisor_pid"] == os.getpid()
    assert I.read_json(out / "supervisor.json")["status"] == "EXITED"
    assert "context accepted once" in (out / "worker.log").read_text()


@pytest.mark.parametrize("corruption", ["parent", "token", "argv", "cwd", "out", "finished"])
def test_context_rejects_mismatch_before_claim(tmp_path, monkeypatch, corruption):
    out = I.new_dir(tmp_path / corruption)
    token = "test-only-context"
    launch = {"schema": 1, "argv": sys.orig_argv, "timeout_seconds": 10,
              "cwd": str(I.ROOT), "out": str(out), "supervisor_pid": os.getppid(),
              "context_sha256": I.digest(token)}
    if corruption == "parent": launch["supervisor_pid"] = -1
    if corruption == "token": launch["context_sha256"] = "bad"
    if corruption == "argv": launch["argv"] = ["another command"]
    if corruption == "cwd": launch["cwd"] = str(tmp_path)
    if corruption == "out": launch["out"] = str(tmp_path)
    if corruption == "finished": I.write_new(out / "supervisor.json", {"status": "EXITED"})
    I.write_new(out / "launch.json", launch)
    I.write_new(out / "spawn.json", {"child_pid": os.getpid(),
        "supervisor_pid": launch["supervisor_pid"], "launch_sha256": I.digest(launch)})
    monkeypatch.setenv(S.CONTEXT_ENV, token)
    monkeypatch.chdir(I.ROOT)
    with pytest.raises(ValueError): S.claim_worker(out)
    assert not (out / "worker_claim.json").exists()


def test_start_failure_and_create_only_preserve_evidence(tmp_path):
    out = tmp_path / "failed"
    assert S.bounded_child([str(tmp_path / "no-python")], out, 10) == 2
    receipt = I.read_json(out / "supervisor.json")
    assert receipt["status"] == "START_FAILED" and receipt["child_pid"] is None
    assert I.read_json(out / "supervisor_failure.json")["status"] == "START_FAILED"
    assert (out / "worker.log").is_file()
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        S.bounded_child([sys.executable, "-c", "pass"], out, 10)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_timeout_reaps_own_child_and_retains_checkpoint(tmp_path):
    out = tmp_path / "timeout"
    code = """
import pathlib, sys, time
p = pathlib.Path(sys.argv[1]) / 'checkpoint-00001'
p.mkdir()
(p / 'state.safetensors').write_bytes(b'fixture-state')
time.sleep(20)
"""
    assert S.bounded_child([sys.executable, "-c", code, str(out)], out, 1) == 2
    receipt = I.read_json(out / "supervisor.json")
    assert receipt["status"] == "TIMEOUT_OR_INTERRUPTED"
    assert receipt["cleanup"]["child_exited"]
    assert (out / "checkpoint-00001" / "state.safetensors").read_bytes() == b"fixture-state"


def test_nonzero_child_is_not_success_even_with_result(tmp_path):
    out = tmp_path / "failed-result"
    code = "import pathlib,sys;pathlib.Path(sys.argv[1],'result.json').write_text('{}');sys.exit(7)"
    assert S.bounded_child([sys.executable, "-c", code, str(out)], out, 10) == 2
    assert I.read_json(out / "supervisor.json")["returncode"] == 7
    assert (out / "result.json").is_file()


@pytest.mark.parametrize("name", ["cell_inproc.sh", "run_cell.sh"])
def test_retired_wrapper_refuses_and_keeps_checkpoints(tmp_path, name):
    shell = shutil.which("bash")
    if os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file(): shell = str(candidate)
    if shell is None:
        pytest.skip("bash unavailable; wrapper syntax/contents alone do not execute it")
    out = I.new_dir(tmp_path / name)
    checkpoint = I.new_dir(out / "checkpoint-00001")
    (checkpoint / "fixture-state").write_bytes(b"unchanged")
    result = subprocess.run([shell, (I.PACKAGE / name).as_posix(), "response_1e-5", "unused", "unused", str(out)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "REFUSED" in result.stderr
    assert (checkpoint / "fixture-state").read_bytes() == b"unchanged"
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint-00001"]


def test_native_checkout_rejects_wrong_branch(monkeypatch):
    def fake_output(command, **kwargs):
        if command[-2:] == ["rev-parse", "--show-toplevel"]: return str(I.ROOT)
        if command[-2:] == ["branch", "--show-current"]: return "research/model-quality-targeted-gpu-v1"
        raise AssertionError("wrong branch should fail before remaining checks")
    monkeypatch.setattr(I.subprocess, "check_output", fake_output)
    with pytest.raises(ValueError, match="dedicated branch"): I.clean_checkout()


def test_cli_resolves_paths_before_fixed_child_cwd(tmp_path, monkeypatch):
    from experiments.model_quality_gpu_v1 import run
    from experiments.model_quality_gpu_v1.config import Design
    I.write_new(tmp_path / "plan.json", {})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "validate_plan", lambda value: {"design": Design().as_dict()})
    captured = {}
    def capture(command, directory, timeout):
        captured.update(command=command, directory=directory, timeout=timeout)
        return 17
    monkeypatch.setattr(run, "bounded_child", capture)
    assert run.main(["train", "--source", "source", "--out", "attempt", "--plan", "plan.json",
                     "--review", "review.json", "--qualification", "qualification.json",
                     "--resume", "old/checkpoint-00001", "--resource-lock", "gpu.lock",
                     "--arm", "response_1e-5", "--confirm-training"]) == 17
    command = captured["command"]
    for flag, relative in [("source", "source"), ("out", "attempt"), ("plan", "plan.json"),
                           ("review", "review.json"), ("qualification", "qualification.json"),
                           ("resume", "old/checkpoint-00001"), ("resource-lock", "gpu.lock")]:
        assert command[command.index("--" + flag) + 1] == str((tmp_path / relative).absolute())
    assert command[-1] == "--_worker" and captured["timeout"] == Design().timeout_seconds


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_supervisor_failure_after_spawn_reaps_child(tmp_path, monkeypatch, error):
    out = tmp_path / "spawn-failure"
    write = S.write_new
    def failing_write(path, value):
        if path.name == "spawn.json": raise error("injected post-spawn failure")
        return write(path, value)
    monkeypatch.setattr(S, "write_new", failing_write)
    assert S.bounded_child([sys.executable, "-c", "import time;time.sleep(20)"], out, 10) == 2
    receipt = I.read_json(out / "supervisor.json")
    assert receipt["cleanup"]["child_exited"]
    assert receipt["status"] == ("TIMEOUT_OR_INTERRUPTED" if error is KeyboardInterrupt
                                  else "SUPERVISOR_FAILED")
    assert I.read_json(out / "supervisor_failure.json")["type"] == error.__name__


def test_timeout_terminates_grandchild_but_not_unrelated_control(tmp_path):
    import psutil
    out = tmp_path / "tree-timeout"
    grandchild_pid_file = tmp_path / "grandchild.pid"
    control_pid_file = tmp_path / "control.pid"
    sleeper = "import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)"
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    control = subprocess.Popen([sys.executable, "-c", sleeper, str(control_pid_file)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
    worker = """
import subprocess, sys, time
subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])
time.sleep(30)
"""
    grandchild_pid = None
    try:
        assert S.bounded_child([sys.executable, "-c", worker, str(grandchild_pid_file), sleeper],
                               out, 2) == 2
        assert control.poll() is None
        assert control_pid_file.is_file() and grandchild_pid_file.is_file()
        grandchild_pid = int(grandchild_pid_file.read_text())
        deadline = time.monotonic() + 3
        while psutil.pid_exists(grandchild_pid) and time.monotonic() < deadline:
            process = psutil.Process(grandchild_pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.02)
        if psutil.pid_exists(grandchild_pid):
            assert psutil.Process(grandchild_pid).status() == psutil.STATUS_ZOMBIE
        assert psutil.pid_exists(int(control_pid_file.read_text()))
        assert I.read_json(out / "supervisor.json")["cleanup"]["child_exited"]
    finally:
        S._stop_child(control)
        if grandchild_pid is not None and psutil.pid_exists(grandchild_pid):
            try:
                psutil.Process(grandchild_pid).kill()
            except psutil.NoSuchProcess:
                pass
