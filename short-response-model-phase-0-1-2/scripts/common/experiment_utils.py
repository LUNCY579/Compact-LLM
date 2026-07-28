from __future__ import annotations

"""Small dependency-free helpers shared by every phase.

The phase runners intentionally keep orchestration in PowerShell and durable
artifact handling in Python.  These helpers centralize JSON/JSONL I/O,
reproducible hashing, environment capture, and checked subprocess execution so
individual phase scripts do not silently drift.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON and return the decoded value."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    """Write stable, human-readable UTF-8 JSON and create parent folders."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while reporting the exact malformed line."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: every JSONL row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON objects as one UTF-8 line each."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a file or directory tree in a path-independent deterministic way.

    Directory fingerprints include each relative file path, size, and file
    digest.  They are used to detect accidental model/router provenance drift
    between Phases 4B, 6, 7, and 8.
    """

    original_value = str(path)
    if not path.exists():
        # Remote Hugging Face IDs cannot be hashed locally. Hash the literal ID
        # rather than its cwd-dependent resolved path.
        return "external:" + hashlib.sha256(original_value.encode("utf-8")).hexdigest()
    path = path.resolve()
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise ValueError(f"Cannot fingerprint unsupported path type: {path}")

    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(child.stat().st_size).encode("ascii"))
        digest.update(sha256(child).encode("ascii"))
    return digest.hexdigest()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Print and execute a command, failing immediately on non-zero status."""

    print("$", subprocess.list2cmdline(cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, cwd=cwd, env=merged_env)


def environment() -> dict[str, Any]:
    """Capture the minimum environment fields required in experiment reports."""

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
