"""Revised, unfrozen PARENTQ supervisor. Native execution requires resolved
decisions, independent review, an exact freeze and attempt-bound operator grant.
All results are external. Own leases are released only after bounded cleanup
confirms the identity-bound worker has exited. Frozen GEO is unchanged.
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

import parentq_safety as safety

HERE = Path(os.path.abspath(__file__)).parent
REPO = HERE.parent.parent
GPU_LEASE_RELATIVE = "experiments/fresh_dual_skill_instrument_v1/gpu_lease.py"
LEASE_LABEL = "triskill-geo001-parentq"        # equals parentq_worker.LEASE_LABEL (CPU test)
HARD_WALL_SECONDS = 600
KILL_GRACE_SECONDS = 60
MIN_GRANT_CHARS = 12
FREEZE_NAME = "FREEZE_PARENTQ.json"
ATTEMPTS_DIR = "native_attempts"
START_NAME = "SUPERVISOR_START_PARENTQ.json"    # equals parentq_worker.START_NAME (CPU test)
RECEIPT_NAME = "SUPERVISOR_RECEIPT_PARENTQ.json"
LOG_NAME = "native_worker_parentq.log"
WORKER_PID_NAME = "WORKER_PID_PARENTQ.json"    # written by the worker itself
OUT_NAME = "native_parentq.json"
INCOMPLETE_NAME = "incomplete_result_parentq.json"
MAX_LOADED_ATTEMPTS = 3
LOAD_BEGUN_NAME = "LOAD_BEGUN_PARENTQ.json"
PINNED_LOCAL = ("parentq_worker.py", "parentq_supervise.py", "parentq_rows.py", "parentq_decide.py",
                "parentq_bank.json", "parentq_canary.json", "parentq_safety.py",
                "parentq_launch_wmi.py", "READINESS_PARENTQ.json")
GEO_FREEZE_SHA256 = "f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687"
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
    pinned_root = freeze.get("runtime_root")
    if (not isinstance(pinned_root, str) or not Path(pinned_root).is_absolute()
            or Path(pinned_root).resolve() != safety.RUNTIME_ROOT.resolve()):
        raise ValueError("STOP_NONCANONICAL_NATIVE_RUNTIME_ROOT")
    files = freeze.get("files", {})
    for name in PINNED_LOCAL:
        if files.get(name) != _sha256_bytes((here / name).read_bytes()):
            raise ValueError(f"STOP_FREEZE_FILE_MISMATCH:{name}")
    lease = freeze.get("supervisor_dependencies", {}).get("gpu_lease", {})
    if lease.get("path") != GPU_LEASE_RELATIVE or len(str(lease.get("sha256", ""))) != 64:
        raise ValueError("STOP_FREEZE_MISSING_GPU_LEASE_PIN")
    geo_freeze = here.parent / "triskill_geometric_pilot_v1" / "FREEZE_SECTIONH.json"
    geo_source = geo_freeze.read_bytes()
    if (freeze.get("geo_freeze", {}).get("sha256") != GEO_FREEZE_SHA256
            or _sha256_bytes(geo_source) != GEO_FREEZE_SHA256):
        raise ValueError("STOP_GEO_FREEZE_MISMATCH")
    geo = json.loads(geo_source)
    if lease != geo.get("supervisor_dependencies", {}).get("gpu_lease"):
        raise ValueError("STOP_GPU_LEASE_PIN_DIFFERS_FROM_GEO")
    safety.validate_readiness(here)
    return freeze


def load_gpu_lease(freeze: dict, repo: Path = REPO):
    """Execute the programme's gpu_lease.py only from bytes matching the freeze pin."""
    entry = freeze["supervisor_dependencies"]["gpu_lease"]
    path = repo / entry["path"]
    source = path.read_bytes()
    if _sha256_bytes(source) != entry["sha256"]:
        raise ValueError("STOP_GPU_LEASE_DIGEST_MISMATCH")
    spec = importlib.util.spec_from_file_location(f"_parentq_gpu_lease_{entry['sha256'][:16]}", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    module._try_reclaim = lambda: False  # closes exists/open('x') race for foreign stale locks
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
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            # An unreadable present lock is uncertainty, never release evidence.
            pass
        sleep(LOCK_RETRY_SECONDS)
    return False


def kill_tree(pid: int, created: float, run=subprocess.run, identity=safety.process_create_time) -> dict:
    """Never taskkill a recycled or unobservable PID; retain every error."""
    try:
        observed = identity(pid)
        if observed is None:
            return {"pid": pid, "status": "ALREADY_EXITED"}
        if observed != created:
            return {"pid": pid, "status": "PID_IDENTITY_CHANGED", "kill_issued": False}
        completed = run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=60)
        return {"pid": pid, "create_time": created, "returncode": completed.returncode,
                "stdout": completed.stdout[-500:], "stderr": completed.stderr[-500:]}
    except Exception as exc:
        return {"pid": pid, "create_time": created, "status": "TASKKILL_ERROR", "error": repr(exc)}


def pid_alive(pid) -> bool:
    import psutil
    return bool(pid) and psutil.pid_exists(int(pid))


def _write_create_only(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def loaded_attempts(runtime_root: Path, historical_here: Path = HERE) -> int:
    """Markers count killed/crashed loads too. Historical results remain charged."""
    folders = {Path(runtime_root) / ATTEMPTS_DIR, Path(historical_here) / ATTEMPTS_DIR}
    charged = set()
    for folder in folders:
        if folder.exists():
            for attempt in folder.iterdir():
                if attempt.is_dir() and ((attempt / LOAD_BEGUN_NAME).exists()
                                         or (attempt / INCOMPLETE_NAME).exists()):
                    charged.add(attempt.resolve())
    return len(charged)


def new_attempt_dir(runtime_root: Path, attempt_number: int) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = runtime_root / ATTEMPTS_DIR / f"attempt-{attempt_number:04d}-{stamp}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_grant(grant: str, attempt_number: int, freeze_sha: str, runtime_root: Path,
                 historical_here: Path = HERE) -> dict:
    try:
        value = json.loads(grant)
    except (ValueError, TypeError) as exc:
        raise ValueError("STOP_NO_ATTEMPT_BOUND_OPERATOR_GRANT") from exc
    if (not isinstance(value, dict) or value.get("experiment") != "TRISKILL-GEO-001-PARENTQ"
            or type(value.get("attempt_number")) is not int
            or value.get("attempt_number") != attempt_number
            or value.get("freeze_sha256") != freeze_sha
            or not isinstance(value.get("operator_grant"), str)
            or len(value["operator_grant"].strip()) < MIN_GRANT_CHARS
            or not isinstance(value.get("hardware_slot"), str)
            or not value["hardware_slot"].strip()):
        raise ValueError("STOP_NO_ATTEMPT_BOUND_OPERATOR_GRANT")
    used = hashlib.sha256(grant.encode("utf-8")).hexdigest()
    for root in {runtime_root, historical_here}:
        for path in (root / ATTEMPTS_DIR).glob("*/" + START_NAME):
            previous = json.loads(path.read_bytes())
            previous_text = previous.get("operator_grant")
            try:
                prior_operator_text = json.loads(previous_text).get("operator_grant")
            except (ValueError, TypeError, AttributeError):
                prior_operator_text = previous_text
            if (previous.get("grant_sha256") == used or previous_text == grant
                    or prior_operator_text == value["operator_grant"]):
                raise RuntimeError("STOP_OPERATOR_GRANT_ALREADY_USED")
    return value


def next_attempt_number(runtime_root: Path, historical_here: Path = HERE) -> int:
    """Launch sequence, including blocked launches, is independent of load count."""
    starts = set()
    for root in {runtime_root, historical_here}:
        starts.update(p.resolve() for p in (root / ATTEMPTS_DIR).glob("*/" + START_NAME))
    return len(starts) + 1


def supervise(freeze_path: Path, freeze_sha: str, grant: str, *, here: Path = HERE, repo: Path = REPO,
              python: str = sys.executable, popen=subprocess.Popen, clock=time.monotonic,
              lease_module=None, kill=kill_tree, alive=pid_alive, hard_wall: int = HARD_WALL_SECONDS,
              sleep=time.sleep, attempt_dir: Path | None = None, runtime_root: Path = safety.RUNTIME_ROOT,
              attempt_number: int | None = None, central_path: Path | None = None,
              hardware_factory=safety.HardwareHandle, crs_db: Path | None = None,
              crs_check=safety.check_crs_idle, identity=safety.process_create_time) -> dict:
    started = clock()
    runtime_root = safety.validate_runtime_root(runtime_root, repo)
    freeze = verify_freeze(freeze_path, freeze_sha, here, repo)
    if runtime_root != Path(freeze["runtime_root"]).resolve():
        raise ValueError("STOP_NATIVE_RUNTIME_ROOT_PIN_MISMATCH")
    out = runtime_root / OUT_NAME
    if out.exists() or (here / OUT_NAME).exists():
        raise FileExistsError(f"STOP_PARENTQ_ALREADY_DECIDED:{out.name}")
    if loaded_attempts(runtime_root, here) >= MAX_LOADED_ATTEMPTS:
        raise RuntimeError("STOP_PARENTQ_ATTEMPTS_EXHAUSTED")
    number = next_attempt_number(runtime_root, here)
    if type(attempt_number) is not int or attempt_number != number:
        raise ValueError("STOP_ATTEMPT_NUMBER_MISMATCH")
    grant_value = verify_grant(grant, number, freeze_sha, runtime_root, here)
    lease = lease_module if lease_module is not None else load_gpu_lease(freeze, repo)
    lease._try_reclaim = lambda: False
    if Path(lease.LOCK).exists():
        raise RuntimeError("STOP_LEASE_PRESENT")
    central = central_path if central_path is not None else safety.central_lock_path(repo)
    if central.exists():
        raise RuntimeError("STOP_CENTRAL_LEASE_PRESENT")
    crs_db = crs_db if crs_db is not None else safety.crs_path()
    crs_check(crs_db)
    holder = lease.foreign_gpu_holder()
    if holder is not None:
        raise RuntimeError(f"STOP_FOREIGN_GPU_HOLDER:{holder[0]}")
    attempt = attempt_dir if attempt_dir is not None else new_attempt_dir(runtime_root, number)
    if not Path(attempt).resolve().is_relative_to(runtime_root / ATTEMPTS_DIR):
        raise ValueError("STOP_ATTEMPT_OUTPUT_OUTSIDE_RUNTIME")
    created = identity(os.getpid())
    if created is None:
        raise RuntimeError("STOP_SUPERVISOR_IDENTITY_UNAVAILABLE")
    receipt = {"experiment": "TRISKILL-GEO-001-PARENTQ", "operator_grant": grant.strip(),
               "utc_start": _utc(), "freeze_sha256": freeze_sha, "hard_wall_seconds": hard_wall,
               "supervisor_pid": os.getpid(), "lease_label": LEASE_LABEL, "attempt_dir": str(attempt),
               "supervisor_create_time": created, "runtime_root": str(runtime_root),
               "attempt_number": number, "hardware_slot": grant_value["hardware_slot"],
               "grant_sha256": hashlib.sha256(grant.encode("utf-8")).hexdigest(), "status": "STARTED"}
    child, worker_pid, child_created, token, hardware = None, None, None, None, None
    central_acquired = False
    start_record = None

    def guarded_kill(pid, created_time):
        try:
            if created_time is None or not safety.same_process(pid, created_time, identity):
                return {"pid": pid, "status": "IDENTITY_NOT_CURRENT", "kill_issued": False}
            return kill(pid, created_time)
        except Exception as exc:
            return {"pid": pid, "status": "TASKKILL_ERROR", "error": repr(exc)}

    try:
        hardware = hardware_factory()
        token = lease.acquire(LEASE_LABEL, "GEO parent qualification, BASE/T/H only (<=600 s)", timeout_seconds=0)
        safety.create_central(central, token, created)
        central_acquired = True
        hardware.record({"pid": os.getpid(), "create_time": created, "token": token,
                         "assigned_worktree": str(repo), "stage": "PARENTQ_NATIVE", "utc": _utc()})
        # Serialise launch sequence and grant use with the complete ownership set.
        if next_attempt_number(runtime_root, here) != number:
            raise RuntimeError("STOP_ATTEMPT_SEQUENCE_CHANGED")
        verify_grant(grant, number, freeze_sha, runtime_root, here)
        start_record = {**receipt, "lease_token": token, "lock": str(lease.LOCK),
                        "central_lock": str(central), "hardware_lock": str(safety.HARDWARE_LOCK),
                        "hardware_lock_token": token, "crs_db": str(crs_db),
                        "manual_release_if_supervisor_died": "Confirm all owned identities exited before token-owned release"}
        _write_create_only(attempt / START_NAME, start_record)
        safety.check_runtime_ownership(start_record, token, identity=identity, crs_check=crs_check)
        command = [python, "-B", "-u", str(here / "parentq_worker.py"), "run", "--freeze", str(freeze_path),
                   "--freeze-sha256", freeze_sha, "--lease-token", token, "--out", str(out),
                   "--attempt-dir", str(attempt)]
        for directory in (runtime_root / "tmp", runtime_root / "cache/huggingface", runtime_root / "cache/torch"):
            directory.mkdir(parents=True, exist_ok=True)
        # gpu_lease's shared LOCK is a literal path in its pinned bytes. Runtime
        # cache/temp redirection must never change that coordination path.
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               "TMP": str(runtime_root / "tmp"), "TEMP": str(runtime_root / "tmp"),
               "HF_HOME": str(runtime_root / "cache/huggingface"),
               "HF_HUB_CACHE": str(runtime_root / "cache/huggingface/hub"),
               "TORCH_HOME": str(runtime_root / "cache/torch"),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        with (attempt / LOG_NAME).open("xb") as log:
            child = popen(command, cwd=str(here), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                          env=env, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            receipt["worker_launcher_pid"] = child.pid
            child_created = identity(child.pid)
            if child_created is None:
                raise RuntimeError("STOP_CHILD_IDENTITY_UNAVAILABLE")
            receipt["worker_launcher_create_time"] = child_created
            set_worker_pids(lease.LOCK, token, [child.pid], sleep=sleep)
            set_worker_pids(central, token, [child.pid], sleep=sleep)
            while True:
                safety.check_runtime_ownership(start_record, token, identity=identity, crs_check=crs_check)
                remaining = hard_wall - (clock() - started)
                if remaining <= 0:
                    receipt["kill"] = guarded_kill(child.pid, child_created)
                    receipt["status"] = "STOP_SUPERVISOR_HARD_WALL"
                    break
                try:
                    receipt["worker_returncode"] = child.wait(timeout=min(1.0, remaining))
                    receipt["status"] = "WORKER_EXITED"
                    break
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as exc:
        receipt["status"] = f"SUPERVISOR_ERROR:{exc!r}"
        raise
    finally:
        # Includes failures after popen but before the normal waiting loop.
        if child is not None and child.poll() is None:
            receipt["cleanup_kill"] = guarded_kill(child.pid, child_created)
            if child_created is None:
                # Popen's retained Windows handle identifies the exact created process.
                try:
                    child.kill()
                    receipt["cleanup_handle_kill"] = True
                except Exception as exc:
                    receipt["cleanup_handle_kill_error"] = repr(exc)
            try:
                receipt["worker_returncode"] = child.wait(timeout=KILL_GRACE_SECONDS)
            except Exception as exc:
                receipt["worker_returncode"] = None
                receipt["cleanup_wait_error"] = repr(exc)
        pid_file = attempt / WORKER_PID_NAME
        worker_created = None
        if pid_file.exists():
            try:
                record = json.loads(pid_file.read_bytes())
                if not isinstance(record, dict):
                    raise ValueError("non-object PID record")
                if record.get("lease_token") == token and record.get("freeze_sha256") == freeze_sha:
                    worker_pid, worker_created = record.get("pid"), record.get("create_time")
                else:
                    receipt["worker_pid_record_error"] = "UNBOUND_PID_RECORD"
            except (OSError, ValueError):
                receipt["worker_pid_record_error"] = "UNREADABLE_PID_RECORD"
        elif child is not None:
            # Launcher exit cannot prove an unrecorded native descendant exited.
            receipt["worker_pid_record_error"] = "MISSING_WORKER_IDENTITY_RECORD"
        receipt["worker_pid"] = worker_pid
        if worker_pid is not None and alive(worker_pid) and worker_pid != getattr(child, "pid", None):
            receipt["worker_pid_kill"] = guarded_kill(worker_pid, worker_created)
            for _ in range(KILL_GRACE_SECONDS):
                if not alive(worker_pid):
                    break
                sleep(1)
        worker_exited = ((child is None or child.poll() is not None)
                         and not (worker_pid is not None and alive(worker_pid))
                         and "worker_pid_record_error" not in receipt)
        if worker_exited and token:
            try:
                receipt["lease_released"] = release_own_lease(lease, token, sleep=sleep)
                receipt["central_lease_released"] = not central_acquired or safety.release_central(central, token)
            except Exception as exc:
                receipt["lease_released"] = False
                receipt["release_error"] = repr(exc)
        elif not worker_exited:
            receipt["lease_released"] = False
            receipt["lease_retained"] = "worker may still be alive; release manually after confirming it exited"
        if hardware is not None and worker_exited:
            try:
                hardware.close()
                receipt["hardware_handle_closed"] = True
            except Exception as exc:
                receipt["hardware_handle_closed"] = False
                receipt["hardware_close_error"] = repr(exc)
        elif hardware is not None:
            # Central/temp locks retain unresolved-worker accounting after a
            # supervisor exit. An OS handle survives only while this process lives.
            receipt["hardware_handle_retained_until_supervisor_exit"] = True
            _RETAINED_HARDWARE_HANDLES.append(hardware)
        receipt.update({"worker_exited": worker_exited, "utc_end": _utc(), "elapsed_seconds": clock() - started,
                        "result_output": str(out), "result_output_exists": out.exists(),
                        "loaded_attempts_ever": loaded_attempts(runtime_root, here)})
        _write_create_only(attempt / RECEIPT_NAME, receipt)
    return receipt


_RETAINED_HARDWARE_HANDLES = []


def _record_failure(runtime_root: Path, exc: BaseException) -> Path:
    folder = runtime_root / ATTEMPTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"SUPERVISOR_FAILURE_{stamp}-{os.getpid()}.json"
    _write_create_only(path, {"utc": _utc(), "supervisor_pid": os.getpid(), "error": repr(exc),
                              "traceback": traceback.format_exc()})
    return path


def main(argv=None) -> dict:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--freeze", type=Path, default=HERE / FREEZE_NAME)
        parser.add_argument("--freeze-sha256", required=True)
        parser.add_argument("--grant", required=True, help="attempt/freeze-bound operator JSON grant")
        parser.add_argument("--attempt-number", type=int, required=True)
        parser.add_argument("--runtime-root", type=Path, default=safety.RUNTIME_ROOT)
        args = parser.parse_args(argv)
        return supervise(args.freeze, args.freeze_sha256, args.grant,
                         runtime_root=args.runtime_root, attempt_number=args.attempt_number)
    except BaseException as exc:  # under WMI there is no console: every refusal leaves a receipt
        # Parse failures have no trusted requested output path yet.
        try:
            root = safety.validate_runtime_root(getattr(locals().get("args"), "runtime_root", safety.RUNTIME_ROOT), REPO)
        except ValueError:
            root = safety.RUNTIME_ROOT
        _record_failure(root, exc)
        raise


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
