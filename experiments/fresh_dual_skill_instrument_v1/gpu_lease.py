"""Exclusive lease for the shared GPU, matching the convention already in use on this box.

Three sessions contend for card 0 tonight. Two of them (DOSELORA-001/ARMPERP-001 and the
Rineya Delta multi-reasoning session) take an exclusive lock at the path below before touching
the card, AND scan the process table for a foreign python holding the card. That second check is
one-way: they can see an unleased run, an unleased run cannot see them. This module makes this
experiment a participant in both directions.

Lock path and record shape follow the existing artifacts
(`rineya-gpu-0.lock.stale-*.bak`: {created, pid, token, label}), extended with the fields the
peer session specified.

Release is token-checked: unlink only if the token read back is still ours, so a stale reclaim
by another session cannot delete a lock we are actively holding.
"""
from __future__ import annotations

import json, os, subprocess, time, uuid
from pathlib import Path

LOCK = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")
ORPHAN_AFTER_SECONDS = 900
MIN_FOREIGN_RSS_BYTES = 512 * 1024 * 1024


class LeaseError(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return True                      # cannot tell -> assume alive, never reclaim on ignorance
    return str(pid) in out


def _gpu_compute_pids() -> set[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    pids = set()
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def _heavy_python_pids() -> dict[int, int]:
    """python.exe processes with their RSS in bytes."""
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[1])
            rss = int(parts[4].replace(",", "").replace(" K", "").replace("K", "")) * 1024
        except ValueError:
            continue
        found[pid] = rss
    return found


def foreign_gpu_holder() -> tuple[int, int] | None:
    """A python process on the card that is not us. Returns (pid, rss) or None.

    Mirrors the peer sessions' check so the protection is symmetric rather than one-way.
    """
    mine = {os.getpid(), os.getppid()}
    heavy = _heavy_python_pids()
    on_card = _gpu_compute_pids()
    for pid, rss in heavy.items():
        if pid in mine or rss < MIN_FOREIGN_RSS_BYTES:
            continue
        # nvidia-smi on this box often reports [N/A] per-process memory, so treat presence on
        # the compute-apps list as authoritative when available and fall back to RSS alone.
        if not on_card or pid in on_card:
            return pid, rss
    return None


def _try_reclaim() -> bool:
    """Reclaim ONLY under every condition at once. Returns True if the lock was removed."""
    try:
        rec = json.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    pid = rec.get("pid")
    if rec.get("worker_pids"):
        return False
    if not isinstance(pid, int) or _pid_alive(pid):
        return False
    deadline = rec.get("orphan_after_unix", rec.get("created", 0) + ORPHAN_AFTER_SECONDS)
    if time.time() < deadline:
        return False
    backup = LOCK.with_suffix(LOCK.suffix + f".stale-{pid}.bak")
    try:
        backup.write_text(json.dumps(rec), encoding="utf-8")
        LOCK.unlink()
    except OSError:
        return False
    print(f"  [gpu-lease] reclaimed a stale lock from dead pid {pid}; backed up to {backup.name}", flush=True)
    return True


def acquire(label: str, job: str, timeout_seconds: int = 3600, poll_seconds: int = 20) -> str:
    """Block until the card is ours. Returns the token needed to release."""
    token = uuid.uuid4().hex
    waited = 0
    announced = False
    while True:
        holder = foreign_gpu_holder()
        if holder is None:
            now = time.time()
            rec = {"created": now, "pid": os.getpid(), "token": token, "label": label,
                   "job": job, "worker_pids": [],
                   "orphan_after_unix": now + ORPHAN_AFTER_SECONDS,
                   "staleness_rule": ("reclaimable ONLY if holder pid dead AND worker_pids empty "
                                      "AND orphan_after_unix passed AND no live process carries "
                                      "this label")}
            try:
                with open(LOCK, "x", encoding="utf-8") as f:
                    json.dump(rec, f)
                print(f"  [gpu-lease] acquired {LOCK.name} label={label} token={token[:8]}", flush=True)
                return token
            except FileExistsError:
                if _try_reclaim():
                    continue
        if not announced:
            why = f"foreign python pid {holder[0]} ({holder[1] / 2**30:.2f} GiB) on the card" if holder \
                  else "lock held by another session"
            print(f"  [gpu-lease] waiting: {why}", flush=True)
            announced = True
        if waited >= timeout_seconds:
            raise LeaseError(f"could not acquire {LOCK} within {timeout_seconds}s")
        time.sleep(poll_seconds)
        waited += poll_seconds


def release(token: str) -> None:
    """Unlink only if the lock still carries OUR token."""
    try:
        rec = json.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("  [gpu-lease] release: lock already gone or unreadable; nothing to do", flush=True)
        return
    if rec.get("token") != token:
        print(f"  [gpu-lease] release: lock now held by pid {rec.get('pid')} "
              f"token {str(rec.get('token'))[:8]} - NOT ours, leaving it alone", flush=True)
        return
    try:
        LOCK.unlink()
        print(f"  [gpu-lease] released {LOCK.name}", flush=True)
    except OSError as exc:
        print(f"  [gpu-lease] release failed: {exc}", flush=True)


class lease:
    """Context manager: `with lease('fds1:gate', 'score 192 rows'):`"""

    def __init__(self, label: str, job: str, timeout_seconds: int = 3600):
        self.label, self.job, self.timeout = label, job, timeout_seconds
        self.token: str | None = None

    def __enter__(self):
        self.token = acquire(self.label, self.job, self.timeout)
        return self

    def __exit__(self, *exc):
        if self.token:
            release(self.token)
        return False
