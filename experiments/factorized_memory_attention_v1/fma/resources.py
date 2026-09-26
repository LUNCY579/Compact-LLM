"""Resource guards for native FMA work."""
from __future__ import annotations

import os

# Existing exact-host architecture work uses a 7.4 GiB free-host-RAM qualification
# boundary. This experiment refuses to lower it.
MIN_FREE_HOST_GIB = 7.4


def free_host_bytes() -> int:
    if hasattr(os, "sysconf"):
        names = getattr(os, "sysconf_names", {})
        if "SC_AVPHYS_PAGES" in names and "SC_PAGE_SIZE" in names:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    raise RuntimeError("cannot determine free host RAM on this platform")


def require_host_ram(min_free_gib: float = MIN_FREE_HOST_GIB) -> dict:
    if min_free_gib < MIN_FREE_HOST_GIB:
        raise ValueError(
            f"host-RAM guard cannot be weakened below {MIN_FREE_HOST_GIB} GiB"
        )
    free = free_host_bytes()
    required = int(min_free_gib * 1024**3)
    if free < required:
        raise MemoryError(
            f"BLOCKED: free host RAM {free / 1024**3:.3f} GiB "
            f"< required {min_free_gib:.3f} GiB"
        )
    return {
        "free_host_bytes": free,
        "min_required_bytes": required,
        "min_required_gib": min_free_gib,
    }
