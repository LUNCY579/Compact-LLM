"""GPU contention helpers for sequential 8GB evaluation."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from .utils import utc_timestamp


def _our_pids(*, ours: str | None = None) -> set[str]:
    pids = {str(os.getpid()), str(os.getppid())}
    if ours is not None:
        pids.add(str(ours))
    return {pid for pid in pids if pid and pid != "0"}


def competing_python_gpu_rows(*, ours: str | None = None) -> list[str]:
    """Return nvidia-smi rows for other Python/CUDA jobs. Ignore desktop overlays."""
    skip = _our_pids(ours=ours)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    hits: list[str] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if "python" not in lower and "pytorch" not in lower:
            continue
        pid = text.split(",", 1)[0].strip()
        if pid in skip:
            continue
        hits.append(text)
    return hits


def wait_for_python_gpu(poll_seconds: int = 60, *, clear_polls: int = 2, settle_seconds: int = 15) -> None:
    """Block until other Python GPU jobs are gone for consecutive clear polls.

    A single empty nvidia-smi snapshot is not enough: another eval can start in
    the gap between our previous candidate teardown and the next load.
    """
    consecutive = 0
    needed = max(1, int(clear_polls))
    while True:
        hits = competing_python_gpu_rows()
        if hits:
            consecutive = 0
            print(
                f"[{utc_timestamp()}] waiting {poll_seconds}s for GPU; other Python jobs:\n"
                + "\n".join(hits),
                flush=True,
            )
            time.sleep(max(1, int(poll_seconds)))
            continue
        consecutive += 1
        if consecutive >= needed:
            print(f"[{utc_timestamp()}] GPU clear of other Python jobs", flush=True)
            return
        print(
            f"[{utc_timestamp()}] GPU looks clear ({consecutive}/{needed}); "
            f"settling {settle_seconds}s",
            flush=True,
        )
        time.sleep(max(1, int(settle_seconds)))


def model_from_pretrained_kwargs(
    protocol: Mapping[str, Any],
    *,
    offload_root: Path | None = None,
) -> dict[str, Any]:
    """device_map auto plus optional VRAM cap so accelerate can offload to CPU."""
    kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": str(protocol.get("device_map", "auto")),
        "low_cpu_mem_usage": True,
    }
    gpu_gib = protocol.get("max_memory_gpu_gib")
    if gpu_gib is not None:
        cpu_gib = protocol.get("max_memory_cpu_gib", 30)
        kwargs["max_memory"] = {
            0: f"{float(gpu_gib)}GiB",
            "cpu": f"{float(cpu_gib)}GiB",
        }
    offload = protocol.get("offload_folder")
    if offload:
        kwargs["offload_folder"] = str(offload)
    elif offload_root is not None and gpu_gib is not None:
        folder = Path(offload_root) / "_hf_offload"
        folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(folder)
    return kwargs
