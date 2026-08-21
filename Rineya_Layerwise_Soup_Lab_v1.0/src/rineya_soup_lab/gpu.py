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


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


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


def gpu_memory_gib() -> tuple[float | None, float | None]:
    """Return (used_gib, total_gib) from nvidia-smi, or (None, None)."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None
    line = completed.stdout.splitlines()[0]
    parts = [item.strip() for item in line.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        used_mib = float(parts[0])
        total_mib = float(parts[1])
    except ValueError:
        return None, None
    return used_mib / 1024.0, total_mib / 1024.0


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


def should_wait_for_gpu(protocol: Mapping[str, Any]) -> bool:
    flag = _env_flag("RINEYA_WAIT_FOR_GPU")
    if flag is not None:
        return flag
    return bool(protocol.get("wait_for_gpu", True))


def resolve_gpu_memory_gib(protocol: Mapping[str, Any], *, headroom_gib: float = 0.75) -> float | None:
    """Weight-placement cap. Env RINEYA_MAX_MEMORY_GPU_GIB overrides the YAML value.

    Also clamps to currently free VRAM minus headroom so a busy 8GB card cannot be
    asked for more than it has.
    """
    env_cap = _env_float("RINEYA_MAX_MEMORY_GPU_GIB")
    cfg_cap = protocol.get("max_memory_gpu_gib")
    if env_cap is not None:
        gpu_gib = float(env_cap)
    elif cfg_cap is not None:
        gpu_gib = float(cfg_cap)
    else:
        return None
    used, total = gpu_memory_gib()
    if used is not None and total is not None:
        free = max(0.0, total - used)
        clamped = min(gpu_gib, max(0.5, free - float(headroom_gib)))
        if clamped + 1e-9 < gpu_gib:
            print(
                f"[{utc_timestamp()}] clamping GPU weight budget {gpu_gib:.2f}GiB -> "
                f"{clamped:.2f}GiB (used={used:.2f} total={total:.2f} headroom={headroom_gib:.2f})",
                flush=True,
            )
            gpu_gib = clamped
    return gpu_gib


def apply_cuda_process_limit(protocol: Mapping[str, Any]) -> None:
    """Hard per-process CUDA allocator cap. Does not change the run config digest."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    env_frac = _env_float("RINEYA_CUDA_MEMORY_FRACTION")
    cfg_frac = protocol.get("cuda_memory_fraction")
    fraction = env_frac if env_frac is not None else (float(cfg_frac) if cfg_frac is not None else None)
    if fraction is None:
        gpu_gib = resolve_gpu_memory_gib(protocol)
        _, total = gpu_memory_gib()
        if gpu_gib is not None and total and total > 0:
            fraction = min(0.95, max(0.10, gpu_gib / total))
    if fraction is None:
        return
    fraction = min(0.95, max(0.10, float(fraction)))
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(
        f"[{utc_timestamp()}] CUDA per-process memory fraction={fraction:.3f}",
        flush=True,
    )


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
    gpu_gib = resolve_gpu_memory_gib(protocol)
    if gpu_gib is not None:
        cpu_env = _env_float("RINEYA_MAX_MEMORY_CPU_GIB")
        cpu_gib = cpu_env if cpu_env is not None else float(protocol.get("max_memory_cpu_gib", 30))
        kwargs["max_memory"] = {
            0: f"{float(gpu_gib):.1f}GiB",
            "cpu": f"{float(cpu_gib):.1f}GiB",
        }
    offload = protocol.get("offload_folder")
    if offload:
        kwargs["offload_folder"] = str(offload)
    elif offload_root is not None and gpu_gib is not None:
        folder = Path(offload_root) / "_hf_offload"
        folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(folder)
    return kwargs
