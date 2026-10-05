"""One-use foreground worker custody. No native runtime imported by supervisor."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from .integrity import ROOT, digest, new_dir, read_json, write_new

CONTEXT_ENV = "RQSFT001_SUPERVISOR_CONTEXT"


def claim_worker(directory):
    token = os.environ.pop(CONTEXT_ENV, None)
    if token is None:
        raise PermissionError("Native worker requires public RQSFT foreground supervisor")
    directory = Path(directory).absolute()
    launch = read_json(directory / "launch.json")
    if (launch.get("experiment") != "RQSFT-001" or launch.get("context_sha256") != digest(token)
            or launch.get("cwd") != str(ROOT) or launch.get("out") != str(directory)
            or Path.cwd().resolve() != ROOT or (directory / "supervisor.json").exists()):
        raise PermissionError("Worker context/phase/checkout differs from supervisor")
    deadline = time.monotonic() + 2
    while not (directory / "spawn.json").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    spawn = read_json(directory / "spawn.json")
    original, wanted = getattr(sys, "orig_argv", []), launch.get("argv", [])
    interpreters = {Path(sys.executable).resolve()}
    if os.name == "nt":
        interpreters.add(Path(sys._base_executable).resolve())
    if (not original or not wanted or original[1:] != wanted[1:]
            or Path(wanted[0]).resolve() != Path(sys.executable).resolve()
            or Path(original[0]).resolve() not in interpreters):
        raise PermissionError("Worker interpreter/arguments differ from supervisor")
    direct = os.getpid() == spawn.get("child_pid") and os.getppid() == launch.get("supervisor_pid")
    redirected = os.name == "nt" and os.getppid() == spawn.get("child_pid")
    if spawn.get("launch_sha256") != digest(launch) or not (direct or redirected):
        raise PermissionError("Worker is not recorded supervisor child")
    write_new(directory / "worker_claim.json", {"pid": os.getpid(), "parent_pid": os.getppid(),
        "launch_sha256": digest(launch), "claimed_unix": time.time()})
    return launch


def _stop(child):
    notes = []
    if child.poll() is None:
        try:
            if os.name == "nt":
                response = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                if response.returncode:
                    notes.append("Owned tree termination returncode " + str(response.returncode))
            else:
                os.killpg(child.pid, signal.SIGKILL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(str(exc))
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)
        notes.append("Owned direct-child termination fallback")
    return {"child_exited": child.poll() is not None, "notes": notes}


def bounded_child(command, output, timeout):
    if type(timeout) is not int or not 1 <= timeout <= 7200 or not isinstance(command, list) or not command:
        raise ValueError("Bounded explicit child argument vector required")
    directory = new_dir(output)
    token = uuid.uuid4().hex
    launch = {"experiment": "RQSFT-001", "argv": command, "timeout_seconds": timeout,
        "cwd": str(ROOT), "out": str(directory), "supervisor_pid": os.getpid(),
        "context_sha256": digest(token),
        "legacy_mq_lock": str(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock")}
    write_new(directory / "launch.json", launch)
    cache, temp = directory / "caches", directory / "temp"
    cache.mkdir(); temp.mkdir()
    env = dict(os.environ, HF_HOME=str(cache / "hf"), HF_HUB_CACHE=str(cache / "hf" / "hub"),
        HUGGINGFACE_HUB_CACHE=str(cache / "hf" / "hub"), TRANSFORMERS_CACHE=str(cache / "transformers"),
        TORCH_HOME=str(cache / "torch"), TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        CUDA_CACHE_PATH=str(cache / "cuda"), XDG_CACHE_HOME=str(cache),
        TRITON_CACHE_DIR=str(cache / "triton"), TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TMP=str(temp), TEMP=str(temp), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env[CONTEXT_ENV] = token
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    child, code, cleanup, reason = None, None, None, "START_FAILED"
    started = time.monotonic()
    try:
        with (directory / "worker.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, **options)
            write_new(directory / "spawn.json", {"child_pid": child.pid,
                "supervisor_pid": os.getpid(), "launch_sha256": digest(launch)})
            reason = "EXITED"
            try:
                code = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                reason = "TIMEOUT_OR_INTERRUPTED"
                cleanup = _stop(child)
                code = child.poll()
    except BaseException as exc:
        reason = "START_FAILED" if child is None else "SUPERVISOR_FAILED"
        if child is not None:
            cleanup = _stop(child); code = child.poll()
        write_new(directory / "supervisor_failure.json", {"status": reason,
            "type": type(exc).__name__, "reason": str(exc), "promotion_allowed": False})
    finally:
        write_new(directory / "supervisor.json", {"experiment": "RQSFT-001", "status": reason,
            "returncode": code, "child_pid": child.pid if child else None,
            "wall_seconds": time.monotonic() - started, "cleanup": cleanup, "promotion_allowed": False})
    return 0 if reason == "EXITED" and code == 0 else 2
