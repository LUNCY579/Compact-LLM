from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Iterator

import psutil


@contextlib.contextmanager
def exclusive_run_lock(path: Path) -> Iterator[None]:
    """Atomic single-writer lock, with automatic cleanup of dead-PID locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                owner_pid = int(payload["pid"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                owner_pid = -1
            if owner_pid > 0 and psutil.pid_exists(owner_pid):
                raise RuntimeError(f"Run is already owned by live PID {owner_pid}: {path}")
            path.unlink(missing_ok=True)
    if descriptor is None:
        raise RuntimeError(f"Could not acquire run lock: {path}")
    try:
        payload = json.dumps({"pid": os.getpid(), "lock": str(path)}, sort_keys=True).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)
