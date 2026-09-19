"""A host-RAM lease for model-loading jobs, with the liveness check the protocol was missing.

Two sessions share this box. The binding constraint is **host RAM (15.2 GiB), not VRAM
(8.19 GiB)** — a "the GPU is free" check passes while both jobs thrash, because the resource
actually contended is RAM. Measured during one collision: free RAM 1.94 GiB, VRAM 4.27 GiB free,
and a peer probe's throughput fell from 28 s to 240 s per 32 rows. **That run then died on a
timeout sized for a quiet box**, so contention converted "slow" into "no result".

## The failure this module exists to prevent

The existing lease file was treated as present-or-absent. When a job was killed, the file stayed
behind naming a dead PID, and `require_free_lease` refuses on any present lease — so **a stale
lease is indistinguishable from a held one, and both sessions deadlock against a corpse.**

The `pid` field was already in the file. The fix is only that it gets *checked*.

## Rules, and why each is the way it is

  * **Never clear a lease whose owner is alive.** That is the whole point of the lease.
  * **Never clear on timeout alone.** PID reuse is rare, but a live unrelated process inheriting
    the number would make a time-based rule wrong, and wrong in the direction of two model jobs.
  * **Back up before clearing.** A cleared lease should be reconstructable when someone later
    asks what happened.
  * **Release only your own token.** Otherwise a slow job that overran can release a lease taken
    by the next one.
  * **Check free RAM as well as the lease.** A free lease is not a quiet machine — a third
    session was observed on this box that never took the lease at all. RAM is the measurement
    that actually predicted the collision.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

DEFAULT_LEASE = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
#: Below this, a model job will thrash regardless of who holds the lease.
MIN_FREE_GIB = 3.0


def _alive(pid: int) -> bool:
    try:
        import psutil  # noqa: PLC0415
        return psutil.pid_exists(int(pid))
    except ImportError:
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, ValueError):
            return False
        except PermissionError:
            return True          # exists but not ours; treat as alive


def free_gib() -> float | None:
    try:
        import psutil  # noqa: PLC0415
        return psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        return None


def inspect(path: Path = DEFAULT_LEASE) -> dict:
    """FREE / HELD / STALE / UNREADABLE, without changing anything."""
    if not path.exists():
        return {"state": "FREE", "path": str(path)}
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                   # noqa: BLE001
        return {"state": "UNREADABLE", "path": str(path), "error": repr(e)}
    pid = j.get("pid")
    if pid is None:
        return {"state": "UNREADABLE", "path": str(path), "why": "no pid field"}
    live = _alive(pid)
    return {"state": "HELD" if live else "STALE", "path": str(path),
            "pid": pid, "owner_alive": live, "token": j.get("token"),
            "age_seconds": round(time.time() - float(j.get("created", time.time())), 1)}


def acquire(path: Path = DEFAULT_LEASE, *, label: str = "",
            min_free_gib: float = MIN_FREE_GIB) -> dict:
    """Take the lease, or refuse. Clears a STALE lease with a backup; never a HELD one."""
    st = inspect(path)
    cleared = None
    if st["state"] == "HELD":
        raise SystemExit(f"lease HELD by live PID {st['pid']} (age {st['age_seconds']}s) -- "
                         f"refusing to start a second model job")
    if st["state"] in ("STALE", "UNREADABLE"):
        bak = path.with_suffix(path.suffix + f".stale-{st.get('pid', 'unreadable')}.bak")
        shutil.copy2(path, bak)
        path.unlink()
        cleared = {"was": st, "backup": str(bak)}

    fg = free_gib()
    if fg is not None and fg < min_free_gib:
        raise SystemExit(f"only {fg:.2f} GiB host RAM free (< {min_free_gib}) -- refusing. "
                         f"A free lease is not a quiet machine; a third session on this box "
                         f"never took the lease at all.")

    token = uuid.uuid4().hex
    path.write_text(json.dumps({"created": time.time(), "pid": os.getpid(),
                                "token": token, "label": label}), encoding="utf-8")
    return {"token": token, "path": str(path), "cleared_stale": cleared,
            "free_gib_at_acquire": round(fg, 2) if fg is not None else None}


def release(token: str, path: Path = DEFAULT_LEASE) -> bool:
    """Release only if we still own it. A job that overran must not free its successor's lease."""
    if not path.exists():
        return False
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return False
    if j.get("token") != token:
        return False
    path.unlink()
    return True


if __name__ == "__main__":                                   # a one-line status probe
    s = inspect()
    fg = free_gib()
    print(json.dumps({**s, "free_gib": round(fg, 2) if fg is not None else None}, indent=2))
