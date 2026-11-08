"""Section H supervisor: GPU lease holder, external hard wall and own-lease release.

Replaces the original dependency on another lane's containment verifier (section H
item 2) with the programme's existing launch convention: the shared `gpu_lease` lock,
a watchdog with a hard wall, a tree kill of OUR worker only, and token-checked release
of OUR lease. Start it detached through launch_wmi.py so a session teardown cannot kill
it. It never waits for, reclaims from or terminates a foreign process or lock, and it
refuses to start without an explicit operator grant for this frozen screen.

Each launch gets its own attempt directory under native_attempts/. The screen output
(native_screen_sectionh.json) is written by the worker only once the model has loaded;
an attempt that stops earlier leaves the screen unspent (registered retry rule).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
REPO = HERE.parent.parent
GPU_LEASE_RELATIVE = "experiments/fresh_dual_skill_instrument_v1/gpu_lease.py"
LEASE_LABEL = "triskill-geo001-sectionh"       # equals worker.LEASE_LABEL (CPU test)
HARD_WALL_SECONDS = 600
KILL_GRACE_SECONDS = 60
MIN_GRANT_CHARS = 12
FREEZE_NAME = "FREEZE_SECTIONH.json"
ATTEMPTS_DIR = "native_attempts"
START_NAME = "SUPERVISOR_START_SECTIONH.json"   # equals worker.START_NAME (CPU test)
RECEIPT_NAME = "SUPERVISOR_RECEIPT_SECTIONH.json"
LOG_NAME = "native_worker_sectionh.log"
WORKER_PID_NAME = "WORKER_PID_SECTIONH.json"   # written by the worker itself
OUT_NAME = "native_screen_sectionh.json"
PINNED_LOCAL = ("worker.py", "supervise.py", "dependency_binding.py", "benchmark.py", "geometry.py")
LOCK_RETRIES, LOCK_RETRY_SECONDS = 40, 0.25


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify_freeze(freeze_path: Path, freeze_sha: str, here: Path = HERE, repo: Path = REPO) -> dict:
    """The supervisor pins the bootstrap: freeze digest, then the files it will execute."""
    data = freeze_path.read_bytes()
    if not freeze_sha or _sha256_bytes(data) != freeze_sha:
        raise ValueError("STOP_FREEZE_DIGEST_MISMATCH")
    freeze = json.loads(data)
    files = freeze.get("files", {})
    for name in PINNED_LOCAL:
        if files.get(name) != _sha256_bytes((here / name).read_bytes()):
            raise ValueError(f"STOP_FREEZE_FILE_MISMATCH:{name}")
    lease = freeze.get("supervisor_dependencies", {}).get("gpu_lease", {})
    if lease.get("path") != GPU_LEASE_RELATIVE or len(str(lease.get("sha256", ""))) != 64:
        raise ValueError("STOP_FREEZE_MISSING_GPU_LEASE_PIN")
    return freeze


def load_gpu_lease(freeze: dict, repo: Path = REPO):
    """Execute the programme's gpu_lease.py only from bytes matching the freeze pin."""
    entry = freeze["supervisor_dependencies"]["gpu_lease"]
    path = repo / entry["path"]
    source = path.read_bytes()
    if _sha256_bytes(source) != entry["sha256"]:
        raise ValueError("STOP_GPU_LEASE_DIGEST_MISMATCH")
    spec = importlib.util.spec_from_file_location(f"_sectionh_gpu_lease_{entry['sha256'][:16]}", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _retry(action, retries=LOCK_RETRIES, delay=LOCK_RETRY_SECONDS, sleep=time.sleep):
    """Windows refuses replace/unlink while another process has the lock open: retry."""
    for attempt in range(retries):
        try:
            return action()
        except PermissionError:
            if attempt == retries - 1:
                raise
            sleep(delay)


def set_worker_pids(lock: Path, token: str, pids, sleep=time.sleep) -> None:
    """Token-checked rewrite: record our worker so peers never reclaim the lease under it."""
    def rewrite():
        record = json.loads(lock.read_text(encoding="utf-8"))
        if record.get("token") != token:
            raise RuntimeError("STOP_LEASE_TOKEN_CHANGED")
        record["worker_pids"] = list(pids)
        temporary = lock.with_suffix(lock.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        try:
            os.replace(temporary, lock)
        except PermissionError:
            temporary.unlink(missing_ok=True)
            raise
    _retry(rewrite, sleep=sleep)


def release_own_lease(lease, token: str, sleep=time.sleep) -> bool:
    """gpu_lease.release is token-checked but single-shot; retry until our lock is gone."""
    for _ in range(LOCK_RETRIES):
        lease.release(token)
        try:
            if json.loads(lease.LOCK.read_text(encoding="utf-8")).get("token") != token:
                return True
        except (OSError, ValueError):
            return True
        sleep(LOCK_RETRY_SECONDS)
    return False


def kill_tree(pid: int, run=subprocess.run) -> dict:
    """Kill only our own worker and its descendants."""
    completed = run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=60)
    return {"pid": pid, "returncode": completed.returncode, "stdout": completed.stdout[-500:],
            "stderr": completed.stderr[-500:]}


def pid_alive(pid) -> bool:
    import psutil
    return bool(pid) and psutil.pid_exists(int(pid))


def _write_create_only(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def new_attempt_dir(here: Path = HERE) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = here / ATTEMPTS_DIR / f"{stamp}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def supervise(freeze_path: Path, freeze_sha: str, grant: str, *, here: Path = HERE, repo: Path = REPO,
              python: str = sys.executable, popen=subprocess.Popen, clock=time.monotonic,
              lease_module=None, kill=kill_tree, alive=pid_alive, hard_wall: int = HARD_WALL_SECONDS,
              sleep=time.sleep, attempt_dir: Path | None = None) -> dict:
    if not isinstance(grant, str) or len(grant.strip()) < MIN_GRANT_CHARS:
        raise ValueError("STOP_NO_OPERATOR_GRANT")
    started = clock()
    freeze = verify_freeze(freeze_path, freeze_sha, here, repo)
    out = here / OUT_NAME
    if out.exists():
        raise FileExistsError(f"STOP_SCREEN_ALREADY_SPENT:{out.name}")
    lease = lease_module if lease_module is not None else load_gpu_lease(freeze, repo)
    # Never reclaim: gpu_lease.acquire would unlink another lane's stale lock.
    if Path(lease.LOCK).exists():
        raise RuntimeError("STOP_LEASE_PRESENT")
    holder = lease.foreign_gpu_holder()
    if holder is not None:
        raise RuntimeError(f"STOP_FOREIGN_GPU_HOLDER:{holder[0]}")
    attempt = attempt_dir if attempt_dir is not None else new_attempt_dir(here)
    receipt = {"experiment": "TRISKILL-GEO-001-SECTIONH", "operator_grant": grant.strip(),
               "utc_start": _utc(), "freeze_sha256": freeze_sha, "hard_wall_seconds": hard_wall,
               "supervisor_pid": os.getpid(), "lease_label": LEASE_LABEL, "attempt_dir": str(attempt),
               "status": "STARTED"}
    token = lease.acquire(LEASE_LABEL, "section H bounded zero-training screen (<=600 s)", timeout_seconds=0)
    child, worker_pid = None, None
    try:
        _write_create_only(attempt / START_NAME, {**receipt, "lease_token": token, "lock": str(lease.LOCK),
                                                  "manual_release_if_supervisor_died":
                                                      "gpu_lease.release(lease_token) after confirming no worker is alive"})
        command = [python, "-u", str(here / "worker.py"), "run", "--freeze", str(freeze_path),
                   "--freeze-sha256", freeze_sha, "--lease-token", token, "--out", str(out),
                   "--attempt-dir", str(attempt)]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        with (attempt / LOG_NAME).open("xb") as log:
            child = popen(command, cwd=str(here), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                          env=env, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            receipt["worker_launcher_pid"] = child.pid
            set_worker_pids(lease.LOCK, token, [child.pid], sleep=sleep)
            remaining = max(0.0, hard_wall - (clock() - started))
            try:
                receipt["worker_returncode"] = child.wait(timeout=remaining)
                receipt["status"] = "WORKER_EXITED"
            except subprocess.TimeoutExpired:
                receipt["kill"] = kill(child.pid)
                try:
                    receipt["worker_returncode"] = child.wait(timeout=KILL_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    receipt["worker_returncode"] = None
                receipt["status"] = "STOP_SUPERVISOR_HARD_WALL"
    except BaseException as exc:
        receipt["status"] = f"SUPERVISOR_ERROR:{exc!r}"
        raise
    finally:
        pid_file = attempt / WORKER_PID_NAME
        if pid_file.exists():
            try:
                worker_pid = json.loads(pid_file.read_text(encoding="utf-8")).get("pid")
            except (OSError, ValueError):
                worker_pid = None
        receipt["worker_pid"] = worker_pid
        if worker_pid is not None and alive(worker_pid):
            receipt["worker_pid_kill"] = kill(worker_pid)   # our own CUDA worker, after its launcher
        worker_exited = (child is None or child.poll() is not None) and not (worker_pid is not None and alive(worker_pid))
        if worker_exited:
            receipt["lease_released"] = release_own_lease(lease, token, sleep=sleep)
        else:
            # Never free the card under a possibly live worker; worker_pids blocks reclaim.
            receipt["lease_released"] = False
            receipt["lease_retained"] = "worker may still be alive; release manually after confirming it exited"
        receipt.update({"worker_exited": worker_exited, "utc_end": _utc(), "elapsed_seconds": clock() - started,
                        "screen_output": out.name, "screen_output_exists": out.exists()})
        _write_create_only(attempt / RECEIPT_NAME, receipt)
    return receipt


def _record_failure(here: Path, exc: BaseException) -> Path:
    folder = here / ATTEMPTS_DIR
    folder.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"SUPERVISOR_FAILURE_{stamp}-{os.getpid()}.json"
    _write_create_only(path, {"utc": _utc(), "error": repr(exc), "traceback": traceback.format_exc()})
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=HERE / FREEZE_NAME)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--grant", required=True,
                        help="the operator's explicit grant for this frozen screen, recorded verbatim")
    args = parser.parse_args()
    try:
        print(json.dumps(supervise(args.freeze, args.freeze_sha256, args.grant), indent=2))
    except BaseException as exc:  # under WMI there is no console: every refusal leaves a receipt
        _record_failure(HERE, exc)
        raise
