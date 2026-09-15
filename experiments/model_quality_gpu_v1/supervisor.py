"""Stdlib-only foreground supervision and one-use private worker context.

The context prevents accidentally bypassing supervision. It is not a security
boundary against a local caller who can modify this code or forge its environment.
No checkpoint cleanup, retry, native import or detached job is performed here.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .integrity import ROOT, digest, new_dir, read_json, write_new

CONTEXT_ENV = "MQ001_SUPERVISOR_CONTEXT"
STOP_TIMEOUT = 5


def claim_worker(directory: Path) -> None:
    """Validate a live direct-parent launch before any worker output or native import."""
    token = os.environ.pop(CONTEXT_ENV, None)
    if token is None:
        raise ValueError("supervisor context absent; use the public qualify/train command")
    directory = Path(directory).absolute()
    launch = read_json(directory / "launch.json")
    if (launch.get("schema") != 1 or launch.get("context_sha256") != digest(token)
            or launch.get("out") != str(directory)
            or launch.get("cwd") != str(ROOT)
            or Path.cwd().resolve() != ROOT
            or type(launch.get("timeout_seconds")) is not int
            or launch["timeout_seconds"] <= 0):
        raise ValueError("worker does not match its supervisor launch")
    original = getattr(sys, "orig_argv", [])
    wanted = launch.get("argv", [])
    interpreters = {Path(sys.executable).resolve()}
    if os.name == "nt":
        # Windows venv redirectors start the base interpreter as their child.
        interpreters.add(Path(sys._base_executable).resolve())
    if (not wanted or not original or original[1:] != wanted[1:]
            or Path(wanted[0]).resolve() != Path(sys.executable).resolve()
            or Path(original[0]).resolve() not in interpreters):
        raise ValueError("worker arguments/interpreter differ from supervisor launch")
    deadline = time.monotonic() + 2
    while not (directory / "spawn.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    spawn = read_json(directory / "spawn.json")
    direct = os.getpid() == spawn.get("child_pid") and os.getppid() == launch.get("supervisor_pid")
    redirected = os.name == "nt" and os.getppid() == spawn.get("child_pid")
    if (spawn.get("launch_sha256") != digest(launch)
            or spawn.get("supervisor_pid") != launch.get("supervisor_pid")
            or not (direct or redirected)):
        raise ValueError("worker is not the recorded supervisor child")
    if (directory / "supervisor.json").exists():
        raise ValueError("supervisor launch has already finished")
    write_new(directory / "worker_claim.json", {
        "schema": 1, "pid": os.getpid(), "parent_pid": os.getppid(),
        "supervisor_pid": launch["supervisor_pid"],
        "launcher_redirected": bool(redirected),
        "launch_sha256": digest(launch), "claimed_unix": time.time(),
    })


def _stop_child(child) -> dict:
    """Terminate only this child tree, with a bounded fallback if tree cleanup fails."""
    notes = []
    if child.poll() is None:
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    check=False, capture_output=True, timeout=STOP_TIMEOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode:
                    notes.append(f"taskkill returncode {result.returncode}")
            else:
                os.killpg(child.pid, signal.SIGKILL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(f"tree termination: {type(exc).__name__}: {exc}")
    try:
        child.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        notes.append("tree termination did not finish; direct-child kill fallback")
        try:
            child.kill()
            child.wait(timeout=STOP_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(f"direct-child termination: {type(exc).__name__}: {exc}")
    return {"child_exited": child.poll() is not None, "notes": notes}


def bounded_child(command: list[str], directory: Path, timeout: int) -> int:
    """Create one attempt and wait in foreground, retaining every terminal receipt."""
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("positive watchdog timeout required")
    if (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or not arg for arg in command)):
        raise ValueError("nonempty subprocess argument list required")
    directory = new_dir(directory)
    token = uuid.uuid4().hex
    launch = {"schema": 1, "argv": command, "timeout_seconds": timeout,
              "cwd": str(ROOT), "out": str(directory), "supervisor_pid": os.getpid(),
              "context_sha256": digest(token)}
    write_new(directory / "launch.json", launch)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env[CONTEXT_ENV] = token
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    reason, code, child, cleanup = "START_FAILED", None, None, None
    started = time.monotonic()
    try:
        with (directory / "worker.log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, cwd=ROOT, **options)
            write_new(directory / "spawn.json", {
                "child_pid": child.pid, "supervisor_pid": os.getpid(),
                "launch_sha256": digest(launch),
            })
            reason = "EXITED"
            try:
                code = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                reason = "TIMEOUT_OR_INTERRUPTED"
                cleanup = _stop_child(child)
                code = child.poll()
    except (Exception, KeyboardInterrupt) as exc:
        # Worker failures use their own failure.json; never overwrite them here.
        reason = ("TIMEOUT_OR_INTERRUPTED" if isinstance(exc, KeyboardInterrupt)
                  else "START_FAILED" if child is None else "SUPERVISOR_FAILED")
        if child is not None:
            cleanup = _stop_child(child)
            code = child.poll()
        write_new(directory / "supervisor_failure.json", {
            "status": reason, "type": type(exc).__name__, "message": str(exc),
            "quality_or_release_permission": False,
        })
    finally:
        write_new(directory / "supervisor.json", {
            "status": reason, "returncode": code,
            "child_pid": child.pid if child is not None else None,
            "wall_seconds": time.monotonic() - started, "cleanup": cleanup,
            "quality_or_release_permission": False,
        })
    return 0 if reason == "EXITED" and code == 0 else 2
