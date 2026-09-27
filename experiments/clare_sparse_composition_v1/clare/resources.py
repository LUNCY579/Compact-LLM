"""Hardware discipline: host RAM (vendored hostbudget), the shared GPU lease, VRAM and disk.

Floors follow the programme's established callers, not new looser ones:
* host RAM  : hostbudget.require(3.6 GiB) before a GPU training job with GPU-resident optimiser
              state (callers use 3.6 for GPU training, 3.0 for probes); guard(1.5) between updates.
* GPU lease : C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock via
              experiments/fresh_dual_skill_instrument_v1/gpu_lease.py (imported, never vendored).
* VRAM      : >= 3.5 GiB free before scoring (run_gpu_batteries.py VRAM_FLOOR_GIB); training needs
              the registered estimate below plus that floor's margin.
* disk      : a run must leave >= 4 GiB free after its registered artifact estimate
              (run_gpu_batteries.py DISK_FLOOR_GIB).

A failed preflight is a BLOCKED status, not a research result.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO_EXPERIMENTS = EXP.parent
GIB = 1024 ** 3

HOST_FLOOR_TRAIN_GIB = 3.6
HOST_FLOOR_PROBE_GIB = 3.0
HOST_GUARD_GIB = 1.5
VRAM_FLOOR_GIB = 3.5
DISK_FLOOR_GIB = 4.0


def hostbudget():
    sys.path.insert(0, str(EXP / "vendor"))
    try:
        import hostbudget as hb  # vendored, byte-identical to 42eb390b
    finally:
        sys.path.pop(0)
    return hb


def gpu_lease():
    sys.path.insert(0, str(REPO_EXPERIMENTS / "fresh_dual_skill_instrument_v1"))
    try:
        import gpu_lease as gl
    finally:
        sys.path.pop(0)
    return gl


def disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / GIB


def vram_free_gib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
        return float(out[0]) / 1024.0
    except Exception:
        return None


def display_adapters() -> list[dict]:
    """Win32_VideoController inventory (name, driver, status): distinguishes 'no NVIDIA device
    present' (e.g. the dGPU disabled by the laptop's GPU mode) from 'nvidia-smi failed'."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,Status | ConvertTo-Json"],
                             capture_output=True, text=True, timeout=60, check=True).stdout
        import json
        data = json.loads(out) if out.strip() else []
        return data if isinstance(data, list) else [data]
    except Exception as exc:
        return [{"error": str(exc)}]


def preflight(kind: str, out_dir: Path, artifact_gib: float, vram_need_gib: float | None = None) -> dict:
    """Measure the resources that actually fail. Returns a record with status READY or BLOCKED."""
    hb = hostbudget()
    snap = hb.snapshot()
    host_floor = HOST_FLOOR_TRAIN_GIB if kind == "train" else HOST_FLOOR_PROBE_GIB
    disk = disk_free_gib(out_dir if out_dir.exists() else out_dir.parent)
    vram = vram_free_gib()
    need_vram = VRAM_FLOOR_GIB if vram_need_gib is None else max(VRAM_FLOOR_GIB, vram_need_gib)
    adapters = display_adapters()
    nvidia_present = any("NVIDIA" in str(a.get("Name", "")) for a in adapters)
    blockers = []
    if snap["free_phys_gib"] < host_floor:
        blockers.append(f"host free {snap['free_phys_gib']:.2f} GiB < {host_floor} GiB")
    if disk - artifact_gib < DISK_FLOOR_GIB:
        blockers.append(f"disk free {disk:.2f} GiB - artifacts {artifact_gib:.2f} GiB < floor {DISK_FLOOR_GIB} GiB")
    if not nvidia_present:
        blockers.append("no NVIDIA display adapter present (Win32_VideoController lists: "
                        + ", ".join(str(a.get("Name", a)) for a in adapters) + ")")
    if vram is None:
        blockers.append("VRAM unmeasurable (nvidia-smi failed)")
    elif vram < need_vram:
        blockers.append(f"VRAM free {vram:.2f} GiB < {need_vram:.2f} GiB")
    gl = gpu_lease()
    holder = None
    try:
        holder = gl.foreign_gpu_holder()
    except Exception as exc:  # pragma: no cover - diagnostic only
        holder = f"unknown ({exc})"
    lock_present = Path(gl.LOCK).exists()
    if lock_present:
        blockers.append(f"GPU lease held ({gl.LOCK})")
    return {"kind": kind, "status": "BLOCKED" if blockers else "READY", "blockers": blockers,
            "host": snap, "host_floor_gib": host_floor, "disk_free_gib": disk, "artifact_gib": artifact_gib,
            "disk_floor_gib": DISK_FLOOR_GIB, "vram_free_gib": vram, "vram_need_gib": need_vram,
            "display_adapters": adapters, "nvidia_present": nvidia_present,
            "foreign_gpu_holder": holder, "gpu_lock_present": lock_present}
