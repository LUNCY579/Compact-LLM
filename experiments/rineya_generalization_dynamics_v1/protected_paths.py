"""Reject named protected-bank paths before metadata or payload content access.

This is a conservative local path boundary, not an authentication or bank seal.
Check both caller spelling and resolved targets so links cannot hide a protected
component. The helper performs no content reads, hashes, or model operations.
"""
from __future__ import annotations

from pathlib import Path

from .common import require


_PREFIXES = ("sealed", "spent", "any_read", "confirmation", "barred")
_NAMES = ("check", "final", "bank", "banks")


def _protected(component: str, include_instruments: bool) -> bool:
    name = component.casefold().replace("-", "_")
    return name.startswith(_PREFIXES) or (include_instruments and any(
        name == marker or name.startswith((marker + "_", marker + "."))
        for marker in _NAMES
    ))


def reject_protected_path(value: str | Path, label: str = "Public metadata", *,
                          include_instruments: bool = False) -> Path:
    """Return the resolved path only after lexical and target bank checks.

    Original training exports may be named ``final``. Aggregate-only callers
    use ``include_instruments=True`` to also reject CHECK/FINAL/bank components.
    ``checkpoint-*`` remains valid in either mode.
    Callers retain their own absolute-path, file-type and size requirements.
    """
    path = Path(value)
    for spelling in (path, path.expanduser()):
        require(not any(_protected(part, include_instruments) for part in spelling.parts),
                f"{label}: Protected banks are outside this review")
    resolved = path.expanduser().resolve()
    require(not any(_protected(part, include_instruments) for part in resolved.parts),
            f"{label}: Protected banks are outside this review")
    return resolved
