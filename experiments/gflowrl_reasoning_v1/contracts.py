"""Data, split, termination, and runtime contracts for GFRL-QUAL-001."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SEALED_PATH_MARKERS = frozenset({"sealed", "confirmation", "holdout", "test", "tests"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetBinding:
    name: str
    split: str
    path: str
    sha256: str
    sealed: bool = False

    def assert_optimization_allowed(self) -> None:
        if self.sealed or self.split.lower() not in {"train", "dev"}:
            raise PermissionError(
                f"optimization is denied for split={self.split!r}, sealed={self.sealed}"
            )


def assert_sealed_asset_denied(path: str | Path) -> None:
    """Deny confirmation/sealed assets by path without opening or hashing them."""

    parts = {part.casefold() for part in Path(path).parts}
    if parts & SEALED_PATH_MARKERS:
        raise PermissionError(f"sealed or confirmation asset denied: {path}")


@dataclass(frozen=True)
class Termination:
    eos_seen: bool
    truncated: bool
    valid: bool


def validate_termination(termination: Termination) -> None:
    if termination.eos_seen and termination.truncated:
        raise ValueError("a rollout cannot be both EOS-terminated and truncated")
    if termination.truncated and termination.valid:
        raise ValueError("truncated rollouts must be masked from the qualification loss")


def runtime_fingerprint() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "packages": packages,
    }
