"""Model-free PARENTQ ownership. No foreign lock is reclaimed.

Hardware coordination uses Windows FileShare.None. Its surviving file is not
a busy indication. Central/temp JSON locks are create-only and token-owned.
CRS is queried read-only; unavailable, stale or incompatible state refuses.
"""
from __future__ import annotations

import ctypes
import json
import os
import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path

BUSINESS = Path("C:/Users/clclc/BusinessProject")
HARDWARE_LOCK = BUSINESS / "ChatBubble-agent-prompts-20261002/hardware-worker.lock"
RUNTIME_ROOT = BUSINESS / "triskill-geo001-parentq-runtime"
LEASE_LABEL = "triskill-geo001-parentq"
REQUIRED_DECISIONS = frozenset({"estimand_and_calibration", "gate_reliability", "representativeness_and_collision",
    "canary_threshold_and_route", "padding_failure_route", "exposed_row_policy", "q2_diagnostics_and_ceiling",
    "retry_and_nondeterminism", "geo_supervision_amendment", "source_backup_and_hardware_slot"})


def validate_runtime_root(root: Path, repo: Path | None = None) -> Path:
    root = Path(root).resolve()
    repo = Path(repo or Path(__file__).resolve().parents[2]).resolve()
    if not root.is_relative_to(BUSINESS.resolve()) or ".codex" in [p.lower() for p in root.parts]:
        raise ValueError("STOP_OUTPUT_OUTSIDE_BUSINESSPROJECT")
    if root == repo or root.is_relative_to(repo):
        raise ValueError("STOP_OUTPUT_INSIDE_SOURCE")
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            raise ValueError("STOP_OUTPUT_INSIDE_GIT_CHECKOUT")
    return root


def validate_readiness(here: Path) -> dict:
    value = json.loads((Path(here) / "READINESS_PARENTQ.json").read_bytes())
    if (value.get("status") != "READY_FOR_REVIEWED_PROPOSAL"
            or value.get("native_execution_authorized") is not False
            or set(value.get("decisions", {})) != REQUIRED_DECISIONS
            or any(d.get("status") != "RESOLVED" for d in value["decisions"].values())
            or value.get("independent_review", {}).get("status") != "PASS"):
        raise RuntimeError("STOP_PARENTQ_DECISIONS_OR_REVIEW_PENDING")
    return value


def process_create_time(pid: int) -> float | None:
    import psutil
    try:
        return psutil.Process(int(pid)).create_time()
    except psutil.NoSuchProcess:
        return None
    except (psutil.AccessDenied, ValueError) as exc:
        raise RuntimeError("STOP_PROCESS_IDENTITY_UNAVAILABLE") from exc


def same_process(pid: int, created: float, identity=process_create_time) -> bool:
    observed = identity(pid)
    return observed is not None and observed == created


def central_lock_path(repo: Path) -> Path:
    common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                     cwd=repo, text=True, timeout=10).strip()
    return Path(common) / "rineya-central-gpu0.lock"


def crs_path() -> Path:
    if not os.environ.get("LOCALAPPDATA"):
        raise RuntimeError("STOP_CRS_PATH_UNAVAILABLE")
    return Path(os.environ["LOCALAPPDATA"]) / "Rineya/CRS-001/host_state.sqlite3"


def check_crs_idle(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError("STOP_CRS_STATE_UNAVAILABLE")
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
            db.execute("PRAGMA query_only=ON")
            if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise RuntimeError("STOP_CRS_STATE_CORRUPT")
            leases = db.execute("SELECT kind,status FROM leases").fetchall()
            active = db.execute("SELECT key,state FROM jobs WHERE state IN ('RESERVED','RUNNING','OUTCOME_UNKNOWN')").fetchall()
            if any(status != "RECONCILED" for _, status in leases) or active:
                raise RuntimeError("STOP_CRS_OWNER_ACTIVE_OR_UNRECONCILED")
            return {"path": str(path), "leases": leases, "active_jobs": len(active), "mode": "ro"}
    except (sqlite3.Error, OSError) as exc:
        raise RuntimeError("STOP_CRS_STATE_UNAVAILABLE_OR_INCOMPATIBLE") from exc


def check_token(path: Path, token: str) -> dict:
    try:
        record = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise RuntimeError("STOP_LEASE_UNREADABLE:" + str(path)) from exc
    if not isinstance(record, dict) or record.get("token") != token or record.get("label") != LEASE_LABEL:
        raise RuntimeError("STOP_LEASE_TOKEN_CHANGED:" + str(path))
    return record


def check_runtime_ownership(start: dict, token: str, *, identity=process_create_time,
                            crs_check=check_crs_idle) -> None:
    check_token(Path(start["lock"]), token)
    check_token(Path(start["central_lock"]), token)
    if not same_process(start["supervisor_pid"], start["supervisor_create_time"], identity):
        raise RuntimeError("STOP_SUPERVISOR_IDENTITY_LOST")
    if start.get("hardware_lock_token") != token or Path(start["hardware_lock"]) != HARDWARE_LOCK:
        raise RuntimeError("STOP_HARDWARE_OWNERSHIP_UNBOUND")
    crs_check(Path(start["crs_db"]))


def create_central(path: Path, token: str, created: float) -> None:
    record = {"pid": os.getpid(), "created": time.time(), "create_time": created,
              "token": token, "label": LEASE_LABEL, "worker_pids": []}
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())


def release_central(path: Path, token: str) -> bool:
    try:
        check_token(path, token)
        Path(path).unlink()
        return True
    except (OSError, RuntimeError):
        return not Path(path).exists()


class HardwareHandle:
    """Windows FileShare.None, retained until close; never deletes the file."""
    def __init__(self, path: Path = HARDWARE_LOCK):
        if os.name != "nt":
            raise RuntimeError("STOP_HARDWARE_LOCK_REQUIRES_WINDOWS")
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        self.kernel.CreateFileW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.kernel.WriteFile.argtypes = (wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
        self.kernel.SetFilePointer.argtypes = (wintypes.HANDLE, wintypes.LONG, wintypes.LPVOID, wintypes.DWORD)
        self.kernel.SetEndOfFile.argtypes = (wintypes.HANDLE,)
        self.kernel.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
        self.path = Path(path)
        self.handle = self.kernel.CreateFileW(str(path), 0xC0000000, 0, None, 4, 0x80, None)
        if self.handle == ctypes.c_void_p(-1).value:
            raise RuntimeError(f"STOP_HARDWARE_LOCK_BUSY_OR_UNAVAILABLE:{ctypes.get_last_error()}")

    def record(self, payload: dict) -> None:
        from ctypes import wintypes
        data, written = json.dumps(payload).encode("utf-8"), wintypes.DWORD()
        self.kernel.SetFilePointer(self.handle, 0, None, 0)
        if not self.kernel.SetEndOfFile(self.handle) or not self.kernel.WriteFile(
                self.handle, data, len(data), ctypes.byref(written), None) or written.value != len(data):
            raise RuntimeError("STOP_HARDWARE_LOCK_RECORD_FAILED")
        if not self.kernel.FlushFileBuffers(self.handle):
            raise RuntimeError("STOP_HARDWARE_LOCK_FLUSH_FAILED")

    def close(self) -> None:
        if self.handle is not None:
            if not self.kernel.CloseHandle(self.handle):
                raise RuntimeError("STOP_HARDWARE_LOCK_CLOSE_FAILED")
            self.handle = None
