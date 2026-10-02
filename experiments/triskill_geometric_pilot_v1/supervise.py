"""Section H supervisor: GPU lease holder, external hard wall and own-lease release.

Replaces the original dependency on another lane's containment verifier (section H
item 2) with the programme's existing launch convention: the shared `gpu_lease` lock,
a watchdog with a hard wall, a tree kill of OUR worker only, and token-checked release
of OUR lease. Start it detached through launch_wmi.py so a session teardown cannot kill
it. It never waits for, reclaims from or terminates a foreign process, and it refuses to
start without an explicit operator grant for this frozen screen.
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
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
REPO = HERE.parent.parent
GPU_LEASE_RELATIVE = "experiments/fresh_dual_skill_instrument_v1/gpu_lease.py"
LEASE_LABEL = "triskill-geo001-sectionh"       # equals worker.LEASE_LABEL (CPU test)
HARD_WALL_SECONDS = 600
KILL_GRACE_SECONDS = 60
MIN_GRANT_CHARS = 12
FREEZE_NAME = "FREEZE_SECTIONH.json"
START_NAME = "SUPERVISOR_START_SECTIONH.json"
RECEIPT_NAME = "SUPERVISOR_RECEIPT_SECTIONH.json"
LOG_NAME = "native_worker_sectionh.log"
OUT_NAME = "native_screen_sectionh.json"
PINNED_LOCAL = ("worker.py", "supervise.py", "dependency_binding.py", "benchmark.py", "geometry.py")


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


def set_worker_pids(lock: Path, token: str, pids) -> None:
    """Token-checked rewrite: record our worker so peers never reclaim the lease under it."""
    record = json.loads(lock.read_text(encoding="utf-8"))
    if record.get("token") != token:
        raise RuntimeError("STOP_LEASE_TOKEN_CHANGED")
    record["worker_pids"] = list(pids)
    temporary = lock.with_suffix(lock.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temporary, lock)


def kill_tree(pid: int, run=subprocess.run) -> dict:
    """Kill only our own worker and its descendants."""
    completed = run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=60)
    return {"returncode": completed.returncode, "stdout": completed.stdout[-500:], "stderr": completed.stderr[-500:]}


def _write_create_only(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def supervise(freeze_path: Path, freeze_sha: str, grant: str, *, here: Path = HERE, repo: Path = REPO,
              python: str = sys.executable, popen=subprocess.Popen, clock=time.monotonic,
              lease_module=None, kill=kill_tree, hard_wall: int = HARD_WALL_SECONDS) -> dict:
    if not isinstance(grant, str) or len(grant.strip()) < MIN_GRANT_CHARS:
        raise ValueError("STOP_NO_OPERATOR_GRANT")
    started = clock()
    freeze = verify_freeze(freeze_path, freeze_sha, here, repo)
    out, receipt_path, start_path, log_path = (here / OUT_NAME, here / RECEIPT_NAME,
                                               here / START_NAME, here / LOG_NAME)
    for path in (out, receipt_path, start_path, log_path):
        if path.exists():
            raise FileExistsError(f"STOP_CREATE_ONLY:{path.name}")
    lease = lease_module if lease_module is not None else load_gpu_lease(freeze, repo)
    receipt = {"experiment": "TRISKILL-GEO-001-SECTIONH", "operator_grant": grant.strip(),
               "utc_start": _utc(), "freeze_sha256": freeze_sha, "hard_wall_seconds": hard_wall,
               "supervisor_pid": os.getpid(), "lease_label": LEASE_LABEL, "status": "STARTED"}
    holder = lease.foreign_gpu_holder()
    if holder is not None:
        raise RuntimeError(f"STOP_FOREIGN_GPU_HOLDER:{holder[0]}")
    token = lease.acquire(LEASE_LABEL, "section H bounded zero-training screen (<=600 s)", timeout_seconds=0)
    child = None
    try:
        _write_create_only(start_path, {**receipt, "lease_token": token, "lock": str(lease.LOCK),
                                        "manual_release_if_supervisor_died":
                                            "gpu_lease.release(lease_token) after confirming no worker is alive"})
        command = [python, "-u", str(here / "worker.py"), "run", "--freeze", str(freeze_path),
                   "--freeze-sha256", freeze_sha, "--lease-token", token, "--out", str(out)]
        with log_path.open("xb") as log:
            child = popen(command, cwd=str(here), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                          creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            receipt["worker_launcher_pid"] = child.pid
            set_worker_pids(lease.LOCK, token, [child.pid])
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
        worker_exited = child is None or child.poll() is not None
        if worker_exited:
            lease.release(token)  # our own lease, token-checked by gpu_lease
        else:
            # Never free the card under a possibly live worker; worker_pids blocks reclaim.
            receipt["lease_retained"] = "worker may still be alive; release manually after confirming it exited"
        try:
            still = json.loads(lease.LOCK.read_text(encoding="utf-8")).get("token") == token
        except (OSError, ValueError):
            still = False
        receipt.update({"lease_released": not still, "worker_exited": worker_exited, "utc_end": _utc(),
                        "elapsed_seconds": clock() - started, "worker_output": out.name,
                        "worker_output_exists": out.exists()})
        _write_create_only(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=HERE / FREEZE_NAME)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--grant", required=True,
                        help="the operator's explicit grant for this frozen screen, recorded verbatim")
    args = parser.parse_args()
    print(json.dumps(supervise(args.freeze, args.freeze_sha256, args.grant), indent=2))
