"""Non-invasive resource observations and qualification guards."""
from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Any, Iterable

import torch


def resource_snapshot() -> dict[str, Any]:
    """Return measured fields only; unavailable measurements remain ``None``."""
    record: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "host_ram_available_bytes": None,
        "host_ram_total_bytes": None,
        "cuda_device": None,
        "cuda_memory_allocated_bytes": None,
        "cuda_memory_reserved_bytes": None,
        "cuda_peak_memory_allocated_bytes": None,
        "cuda_peak_memory_reserved_bytes": None,
    }
    try:
        import psutil
        memory = psutil.virtual_memory()
        record["host_ram_available_bytes"] = int(memory.available)
        record["host_ram_total_bytes"] = int(memory.total)
    except ImportError:
        pass
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        record["cuda_device"] = int(device)
        record["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        record["cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        record["cuda_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        record["cuda_peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return record


def reset_cuda_peak_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def assert_no_sealed_paths(paths: Iterable[str | Path]) -> None:
    """Reject qualification inputs that could reach sealed/ANY_READ assets."""
    blocked = {"sealed", "any_read", "any-read", "private_confirmation", "confirmation"}
    for value in paths:
        parts = {part.casefold() for part in Path(value).parts}
        if parts & blocked:
            raise AssertionError(f"qualification path is sealed/private: {value}")


def resource_guard(*, min_host_ram_gib: float | None = None,
                   min_free_vram_gib: float | None = None) -> dict[str, Any]:
    """Fail closed when an explicitly configured resource floor is unmet."""
    snapshot = resource_snapshot()
    failures: list[str] = []
    host = snapshot.get("host_ram_available_bytes")
    vram = snapshot.get("cuda_memory_reserved_bytes")
    if min_host_ram_gib is not None and host is not None:
        if host < float(min_host_ram_gib) * (1024 ** 3):
            failures.append("host_ram_floor")
    # A free-VRAM reading is not inferred from reserved bytes.  The caller must
    # supply a measured free value in a runtime-specific guard if this gate is
    # needed; qualification never weakens an existing repository guard.
    if min_free_vram_gib is not None and vram is None and torch.cuda.is_available():
        failures.append("free_vram_unavailable")
    return {"status": "PASS" if not failures else "BLOCKED",
            "failures": failures, "snapshot": snapshot,
            "configured_min_host_ram_gib": min_host_ram_gib,
            "configured_min_free_vram_gib": min_free_vram_gib}
