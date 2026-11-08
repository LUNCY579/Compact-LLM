"""GPU lease that measures the CARD, not the bookkeeping.

Three protocols were tried in one night and the first two both had holes. Recording all three,
because the failure mode is the same each time and it is not obvious from inside any of them.

ATTEMPT 1 -- "lock pid dead => stale". Unsound as soon as workers are spawned
DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB, which is itself the
correct fix for parent-death killing a long job. The flag severs the coupling the rule depends on
BY DESIGN: supervisor dies, detached worker keeps the card, lock reads stale, next lane steals a
LIVE lease. Measured on a live lock:

    19020 (1 MB) -> 18952 (6 MB, 0.2 s CPU, THE PID IN THE LOCK) -> 56820 (4 MB)
                 -> 57352 (1613 MB, the ONLY one in nvidia-smi --query-compute-apps)

Also kills a tempting diagnostic: a lock holder at 0% CPU with no I/O is NOT evidence of a hang.
A 6 MB idle shim is what a healthy supervisor looks like for the whole life of the job.

ATTEMPT 2 -- "recorded WORKER pid absent from both nvidia-smi and the process table => stale".
Better, and still wrong. It has a BETWEEN-ARMS race: arm 1's worker exits, arm 2's worker takes
the card, and unless the lock is refreshed in that instant every listed worker is legitimately
absent -- so the lease reads stale WHILE THE CARD IS IN USE. The paired session tripped this
within an hour of us agreeing to adopt it, on exactly this shape of battery. A four-arm sequential
run has THREE such boundaries.

ATTEMPT 3, implemented here -- MEASURE THE CARD.

    foreign job := a python process in `nvidia-smi --query-compute-apps`,
                   not in my own process tree, with RSS >= MIN_FOREIGN_GIB

    acquire  := lock is free AND no foreign job
                -> write the lock
                -> RE-CHECK after RECHECK_S; if a foreign job appeared, release and go back to
                   waiting (someone else won the race)
    cannot see the card -> treat as BUSY. Never barge in on ignorance.

> **Bookkeeping has a race. The card does not.**

The RSS floor exists because desktop applications appear in `--query-compute-apps` with `[N/A]`
memory; a real training or scoring worker does not sit under half a GiB.

Bookkeeping is still WRITTEN (pid, worker_pids, label) because it makes a held lease legible to a
human and to peers -- but nothing here DECIDES on it except as a courtesy check. And absence of
the lock file is never sufficient to launch: other controllers in this programme require absence
before starting and can take it between your check and your write. TAKE the lock (O_EXCL).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

LOCK = Path(os.environ.get("TEMP", "/tmp")) / "rineya-gpu-0.lock"
MIN_FOREIGN_GIB = 0.5
RECHECK_S = 3.0


def _my_tree() -> set[int]:
    """My pid, my ancestors and all my descendants -- everything that is 'not foreign'."""
    mine = {os.getpid()}
    try:
        import psutil
        me = psutil.Process()
        mine |= {p.pid for p in me.parents()}
        mine |= {p.pid for p in me.children(recursive=True)}
    except Exception:
        pass
    return mine


def foreign_gpu_job() -> tuple[bool, str]:
    """(is_busy, why). Unreadable card counts as BUSY -- never barge in on ignorance."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return True, "nvidia-smi returned non-zero; treating the card as busy"
        pids = {int(x.strip()) for x in out.stdout.split() if x.strip().isdigit()}
    except Exception as e:
        return True, f"cannot query the card ({e}); treating it as busy"

    mine = _my_tree()
    try:
        import psutil
    except Exception:
        return (bool(pids - mine), "psutil unavailable; any non-tree GPU pid counts as busy")

    for pid in sorted(pids - mine):
        try:
            p = psutil.Process(pid)
            rss = p.memory_info().rss / 2 ** 30
            if rss >= MIN_FOREIGN_GIB:
                return True, f"foreign GPU job pid {pid} ({p.name()}, {rss:.2f} GiB RSS)"
        except Exception:
            continue          # vanished or unreadable: not evidence of a live foreign job
    return False, "card clear"


def acquire(timeout_s: float = 7200.0, poll_s: float = 20.0, label: str = "") -> str:
    token = uuid.uuid4().hex
    t0 = time.time()
    last = None
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"GPU lease not free after {timeout_s:.0f}s")

        busy, why = foreign_gpu_job()
        if busy or LOCK.exists():
            reason = why if busy else f"lock held ({_holder()})"
            if reason != last:
                print(f"[lease] waiting: {reason}", flush=True)
                last = reason
            time.sleep(poll_s)
            continue

        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                json.dump({"pid": os.getpid(), "worker_pids": [], "token": token,
                           "created": time.time(), "label": label,
                           "protocol": "card-measured; see gpu_lease.py"}, fh)
        except FileExistsError:
            time.sleep(poll_s)
            continue

        # Someone may have started between our check and our write. Re-check and yield if so.
        time.sleep(RECHECK_S)
        busy, why = foreign_gpu_job()
        if busy:
            print(f"[lease] lost the race ({why}); releasing and waiting", flush=True)
            release(token)
            time.sleep(poll_s)
            continue

        print(f"[lease] acquired by pid {os.getpid()} ({label})", flush=True)
        return token


def _holder() -> str:
    try:
        r = json.loads(LOCK.read_text())
        return f"pid {r.get('pid')} workers {r.get('worker_pids')} {r.get('label','')}".strip()
    except Exception:
        return "unreadable"


def set_workers(token: str, pids) -> None:
    """Refresh the recorded workers. Call at EVERY arm boundary, not once at the start --
    a stale worker list is what made attempt 2 unsound. Courtesy metadata only."""
    try:
        rec = json.loads(LOCK.read_text())
    except Exception:
        return
    if rec.get("token") != token:
        return                                  # never edit someone else's lease
    rec["worker_pids"] = [int(p) for p in pids]
    rec["refreshed"] = time.time()
    LOCK.write_text(json.dumps(rec))


def release(token: str) -> None:
    try:
        rec = json.loads(LOCK.read_text())
    except FileNotFoundError:
        return
    except Exception:
        return
    if rec.get("token") != token:
        print("[lease] NOT releasing: the lock is not ours", file=sys.stderr, flush=True)
        return
    try:
        LOCK.unlink()
        print("[lease] released", flush=True)
    except FileNotFoundError:
        pass
