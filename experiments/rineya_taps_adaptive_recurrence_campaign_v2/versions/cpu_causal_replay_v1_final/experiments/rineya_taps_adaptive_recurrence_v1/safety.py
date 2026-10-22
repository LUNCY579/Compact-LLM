"""Fail-closed resources and append-only evidence; no automatic downloads."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time
import psutil

GIB = 1024**3
BASE = "03db218ceebc61a2bc04f2482cf7e094045e01c1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


class Journal:
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8")
        self.count = 0

    def append(self, value):
        self.file.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        self.file.flush()
        self.count += 1

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def memory_guard(available_bytes, required_bytes, *, reserve_bytes=256*1024**2):
    if any(type(v) is not int or v < 0 for v in (available_bytes, required_bytes, reserve_bytes)):
        raise ValueError("memory quantities must be nonnegative integer bytes")
    if available_bytes < required_bytes + reserve_bytes:
        raise MemoryError(f"need {required_bytes+reserve_bytes} free bytes; available {available_bytes}")


def resources():
    import torch
    available = psutil.virtual_memory().available
    # Respect container limits rather than reporting the host's unallocatable RAM.
    try:
        cap = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if cap != "max":
            available = min(available, max(0, int(cap)-current))
    except (OSError, ValueError):
        pass
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers_installed": importlib.util.find_spec("transformers") is not None,
            "cuda_available": torch.cuda.is_available(), "free_ram_bytes": available,
            "cuda_free_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
            "rss_bytes": psutil.Process().memory_info().rss}


def runtime_guard(started, max_seconds=900., max_rss_bytes=6*GIB):
    if time.perf_counter()-started > max_seconds:
        raise TimeoutError("registered per-case wall budget exceeded between forwards")
    if psutil.Process().memory_info().rss > max_rss_bytes:
        raise MemoryError("registered process RSS ceiling exceeded")


def native_preflight(source=None):
    r = resources()
    reasons = []
    if not r["transformers_installed"]:
        reasons.append("Transformers is not installed")
    if source is None or not (Path(source)/"model.safetensors").is_file():
        reasons.append("exact local canonical Supa checkpoint is absent/unconfigured")
    try:
        memory_guard(r["free_ram_bytes"], math.ceil(7.4*GIB), reserve_bytes=0)
    except MemoryError as exc:
        reasons.append(str(exc))
    if r["torch"] != "2.12.1+cu126" or r["python"] != "3.11.15":
        reasons.append("historical ARMR Python 3.11.15 / torch 2.12.1+cu126 runtime not matched")
    return {"status": "BLOCKED" if reasons else "METADATA_READY_NOT_QUALIFIED",
            "resources": r, "reasons": reasons, "native_runs": 0,
            "training_attempted": False, "cuda_lane_qualified": False,
            "note": "CPU FP32 lane; CUDA absence is reported, not the only blocker."}
