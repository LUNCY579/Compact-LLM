"""Lightweight resource instrumentation for v3/v4 diagnostic jobs.

Optimises for minimum instantaneous load: samples RSS on a short interval,
records model-load counts, and writes a small JSON sidecar. Never holds model
weights itself.
"""

from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .hashing import atomic_write_json
from .v2 import gpu_snapshot


@dataclass
class ResourceTrace:
    label: str
    started_monotonic: float = field(default_factory=time.monotonic)
    ended_monotonic: float | None = None
    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    start_rss_bytes: int = 0
    end_rss_bytes: int = 0
    model_load_count: int = 0
    disk_bytes_written: int = 0
    gpu: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def sample_rss(self) -> int:
        try:
            import psutil

            rss = int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            rss = 0
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        return rss

    def record_model_load(self) -> None:
        self.model_load_count += 1
        self.sample_rss()

    def record_disk(self, path: Path) -> None:
        if path.is_file():
            self.disk_bytes_written += path.stat().st_size
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    self.disk_bytes_written += child.stat().st_size

    def finish(self) -> dict[str, Any]:
        self.end_rss_bytes = self.sample_rss()
        self.ended_monotonic = time.monotonic()
        self.wall_seconds = self.ended_monotonic - self.started_monotonic
        self.gpu = gpu_snapshot()
        return asdict(self)

    def write(self, path: Path) -> dict[str, Any]:
        payload = self.finish()
        atomic_write_json(path, payload)
        return payload


@contextmanager
def resource_trace(label: str, output: Path | None = None) -> Iterator[ResourceTrace]:
    trace = ResourceTrace(label=label)
    trace.start_rss_bytes = trace.sample_rss()
    try:
        yield trace
    finally:
        gc.collect()
        if output is not None:
            trace.write(output)
        else:
            trace.finish()


def refuse_gpu_if_busy(
    *,
    util_threshold: float = 5.0,
    vram_mib_threshold: float = 512.0,
) -> dict[str, Any]:
    """Stricter than v2: refuse GPU when util ≥5% or VRAM ≥512 MiB."""
    snap = gpu_snapshot()
    if not snap.get("available"):
        return {**snap, "refuse": True, "refuse_reason": "cuda_unavailable_or_smi_failed"}
    summary = snap.get("summary") or ""
    for line in summary.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            util = float(parts[1])
            mem = float(parts[2])
            if util >= util_threshold or mem >= vram_mib_threshold:
                return {
                    **snap,
                    "refuse": True,
                    "refuse_reason": f"gpu_busy util={util} mem_mib={mem}",
                }
    return {**snap, "refuse": False, "refuse_reason": "ok"}
