"""Model-free preflight and cooperative dual lease. Never steals either lock."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

GIB = 2 ** 30
# Native inference inherits the stricter historical run_matrix 6000 MiB GPU
# admission gate and MQ's 4.6 GiB host loader floor. These are prerequisites,
# not a proof that a new representative sequence shape fits.
LIMITS = {
    "host_admission_bytes": int(4.6 * GIB),
    "gpu_admission_bytes": 6000 * 2 ** 20,
    "host_ongoing_bytes": int(1.5 * GIB),
    "gpu_ongoing_bytes": int(0.5 * GIB),
    "disk_floor_bytes": 4 * GIB,
    "worker_rss_max_bytes": 6 * GIB,
    "worker_wall_seconds": 1800,
    "total_native_wall_seconds": 108000,
    "max_model_workers": 1,
}


def lock_paths(root: Path) -> tuple[Path, Path]:
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, text=True).strip()
    # RVS stores a raw nonce; CLARE's temp lease stores JSON. Do not parse one
    # as the other or infer staleness from a timestamp.
    return (Path(common) / "rineya-central-gpu0.lock",
            Path(os.environ["LOCALAPPDATA"]) / "Temp" / "rineya-gpu-0.lock")


def snapshot(root: Path) -> dict:
    import psutil
    gpu_error, gpu, active = None, None, []
    try:
        value = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits"], text=True, timeout=15)
        fields = [v.strip() for v in value.splitlines()[0].split(",")]
        gpu = {"name": fields[0], "total_bytes": int(fields[1]) * 2 ** 20,
               "free_bytes": int(fields[2]) * 2 ** 20,
               "used_bytes": int(fields[3]) * 2 ** 20}
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits"], text=True, timeout=15).splitlines()
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        gpu_error = str(exc)
    locks = []
    for path in lock_paths(root):
        locks.append({"path": str(path), "present": path.exists(),
                      "holder_text": path.read_text(encoding="utf-8") if path.exists() else None})
    processes = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            if (info["pid"] != os.getpid() and "python" in (info["name"] or "").lower()
                    and info["memory_info"] and info["memory_info"].rss >= GIB // 2):
                processes.append({"pid": info["pid"], "name": info["name"],
                                  "rss_bytes": info["memory_info"].rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {"time_unix": time.time(), "host_available_bytes": psutil.virtual_memory().available,
            "host_total_bytes": psutil.virtual_memory().total,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "disk_free_bytes": shutil.disk_usage(root).free,
            "gpu": gpu, "gpu_error": gpu_error, "locks": locks,
            "gpu_process_inventory": active, "own_pid": os.getpid(), "possible_model_workers": processes,
            "runtime_metadata": {p: importlib.metadata.version(p) for p in
                                 ("torch", "transformers", "safetensors", "tokenizers", "numpy", "psutil")},
            "native_runtime_imported": False, "native_qualification": "NOT_RUN"}


def blockers(value: dict, *, ongoing=False, own_token: str | None = None) -> list[str]:
    reasons = []
    if value.get("gpu_error"):
        reasons.append("GPU_INVENTORY_UNAVAILABLE: " + value["gpu_error"])
    host = LIMITS["host_ongoing_bytes" if ongoing else "host_admission_bytes"]
    gpu = LIMITS["gpu_ongoing_bytes" if ongoing else "gpu_admission_bytes"]
    if value["host_available_bytes"] < host:
        reasons.append(f"HOST_RAM: {value['host_available_bytes']} < {host} bytes")
    if value["gpu"] is None or value["gpu"]["free_bytes"] < gpu:
        reasons.append(f"GPU_MEMORY: requires {gpu} free bytes")
    if value["disk_free_bytes"] < LIMITS["disk_floor_bytes"]:
        reasons.append("DISK_HEADROOM")
    if value["process_rss_bytes"] > LIMITS["worker_rss_max_bytes"]:
        reasons.append("WORKER_RSS_LIMIT")
    for lock in value["locks"]:
        if lock["present"] and lock["holder_text"] != own_token:
            reasons.append("FOREIGN_LOCK: " + lock["path"])
    if value["possible_model_workers"]:
        reasons.append("POSSIBLE_FOREIGN_MODEL_WORKER")
    # Only this current worker is exempt. A sibling/descendant model process is
    # still foreign: this design runs one worker and creates no model subprocess.
    foreign_gpu = []
    for line in value["gpu_process_inventory"]:
        pid = line.split(",", 1)[0].strip()
        if pid.isdigit() and int(pid) == value.get("own_pid"):
            continue
        foreign_gpu.append(line)
    names = "\n".join(foreign_gpu).lower()
    if any(term in names for term in ("python", "torch", "llama", "ollama", "vllm", "comfy", "train")):
        reasons.append("FOREIGN_GPU_MODEL_PROCESS")
    return reasons


def require_resources(root: Path, *, ongoing=False, own_token=None) -> dict:
    value = snapshot(root)
    reasons = blockers(value, ongoing=ongoing, own_token=own_token)
    if reasons:
        raise RuntimeError("resource guard refused: " + "; ".join(reasons))
    return value


@contextmanager
def dual_lease(paths: tuple[Path, Path]):
    """Exclusive-create both paths; rollback/release only our exact contents.

    Both readers accept arbitrary present lock contents as busy, so a common JSON
    object with pid/job/token is compatible with both lock conventions.
    """
    token = json.dumps({"pid": os.getpid(), "job": "EFE-FUNC-001",
                        "token": uuid.uuid4().hex}, sort_keys=True)
    acquired = []
    try:
        for path in paths:
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("symlink in lock path")
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(token)
            acquired.append(path)
        yield token
    finally:
        for path in reversed(acquired):
            if path.exists() and path.read_text(encoding="utf-8") == token:
                path.unlink()


def elapsed_guard(start: float, aggregate_used: float = 0) -> None:
    elapsed = time.monotonic() - start
    if elapsed > LIMITS["worker_wall_seconds"]:
        raise RuntimeError("worker wall budget reached")
    if elapsed + aggregate_used > LIMITS["total_native_wall_seconds"]:
        raise RuntimeError("total native wall budget reached")
