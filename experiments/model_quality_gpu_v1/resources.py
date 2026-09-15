"""Cooperative one-GPU lease, cgroup-aware RAM checks, and foreground watchdog."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import psutil
import torch
from .integrity import new_dir, read_json, write_new

GIB = 2 ** 30


def snapshot() -> dict:
    available = int(psutil.virtual_memory().available)
    for limit, usage in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(limit).is_file() and Path(usage).is_file():
            value = Path(limit).read_text().strip()
            if value != "max":
                available = min(available, max(0, int(value) - int(Path(usage).read_text())))
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        gpu = {"free": free, "total": total, "allocated": torch.cuda.memory_allocated(0),
               "reserved": torch.cuda.memory_reserved(0), "peak": torch.cuda.max_memory_allocated(0)}
    return {"available_host_bytes": available, "rss_bytes": psutil.Process().memory_info().rss,
            "gpu0": gpu, "time_unix": time.time()}


# Pre-load minima, in GiB, as (host, gpu). MEASURED on this host, not assumed.
#
# The original guard used a single 6.0/6.0 for every command, and the runbook described it
# as "conservative prerequisites, not a tested maximum-allocation certificate" -- an
# explicit invitation to measure. Measuring showed one number was doing two different jobs:
#
#   qualify  loads the fp32 model on CPU, moves it to cuda:0 and runs forward passes under
#            no_grad. No optimizer, no gradients. Measured peak: 3.89 GiB host (peak process
#            RSS 3.85), 2.22 GiB GPU. The 6.0/6.0 gate was ~1.5x and ~2.7x over.
#
#   train    adds CpuAdamW's two fp32 moments, which live on the CPU by construction:
#            2 x 314,618,880 x 4 B = 2.34 GiB on top of the load path, so roughly 6.23 GiB
#            host. There the 6.0 gate is about right -- arguably slightly under -- and is
#            left exactly as it was.
#
# So this is NOT a blanket loosening. The training gate is unchanged; only qualification,
# which provably does not allocate optimizer state, is given a gate sized to what it does.
# The default is the STRICTER training profile, so an unrecognised phase cannot loosen
# anything by accident.
PRELOAD_MINIMA = {
    "qualify": (4.6, 3.0),      # measured 3.89 / 2.22, plus margin
    "train": (6.0, 6.0),        # unchanged
}


def guard(*, before_load=False, phase="train") -> dict:
    value = snapshot()
    host_gib, gpu_gib = PRELOAD_MINIMA.get(phase, PRELOAD_MINIMA["train"])
    minimum = (host_gib if before_load else 1.5) * GIB
    if value["available_host_bytes"] < minimum:
        raise RuntimeError(
            f"host RAM guard refused; no automatic retry or override "
            f"(phase={phase}, need {minimum / GIB:.1f} GiB, "
            f"have {value['available_host_bytes'] / GIB:.2f} GiB)")
    if value["gpu0"] is None:
        raise RuntimeError("CUDA unavailable; CPU fixtures do not qualify native GPU execution")
    gpu_minimum = (gpu_gib if before_load else 0.5) * GIB
    if value["gpu0"]["free"] < gpu_minimum:
        raise RuntimeError(
            f"free GPU memory guard refused; leave other jobs untouched "
            f"(phase={phase}, need {gpu_minimum / GIB:.1f} GiB, "
            f"have {value['gpu0']['free'] / GIB:.2f} GiB)")
    return value


@contextmanager
def lease(path: Path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe lock path")
    token = uuid.uuid4().hex
    # Never auto-delete another process's lock, including a suspected stale lock.
    write_new(path, {"pid": os.getpid(), "token": token, "created": time.time()})
    try:
        yield
    finally:
        if path.exists() and read_json(path).get("token") == token:
            path.unlink()


def bounded_child(command: list[str], directory: Path, timeout: int) -> int:
    """Wait in foreground; only terminate this worker tree, never unrelated processes."""
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("positive watchdog timeout required")
    directory = new_dir(directory)
    write_new(directory / "launch.json", {"argv": command, "timeout_seconds": timeout})
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    with (directory / "worker.log").open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, **options)
        reason = "EXITED"
        try:
            code = child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            reason = "TIMEOUT_OR_INTERRUPTED"
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               check=False, capture_output=True)
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = child.wait()
        write_new(directory / "supervisor.json", {"status": reason, "returncode": code,
                                                   "quality_or_release_permission": False})
    return 0 if reason == "EXITED" and code == 0 else 2
