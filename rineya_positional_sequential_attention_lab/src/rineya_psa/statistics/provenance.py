"""SPEC section 9: every result file carries a sibling ``*.provenance.json``.

``stamp()`` answers "what produced this number" -- commit, working-tree cleanliness,
package versions, device, OS, UTC time, and the caller's config with its sha256.
``write_result()`` is the only writer other modules should use: it calls
``paths.assert_readonly_target`` before it opens anything.

DEVIATION: the GPU name is read from ``nvidia-smi`` rather than
``torch.cuda.get_device_name``. The latter forces a lazy CUDA init and allocates a
~300 MB context on the card, and another process owns the GPU while this lab runs
analysis. If a CUDA context already exists in this process the torch API is used instead,
since there is nothing left to pay for. ``gpu_source`` records which path ran.
"""

from __future__ import annotations

import getpass
import hashlib
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rineya_psa.paths import LAB_ROOT, REPO_ROOT, assert_readonly_target

__all__ = [
    "config_hash",
    "canonical_json",
    "git_state",
    "json_default",
    "package_versions",
    "stamp",
    "write_result",
]

_VERSIONED_PACKAGES = ("numpy", "scipy", "torch", "transformers", "pandas", "sklearn")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, compact separators, unicode preserved."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=json_default
    )


def config_hash(config: Mapping[str, Any] | None) -> str:
    """sha256 of the canonical form of ``config``; stable across dict insertion order."""
    return hashlib.sha256(canonical_json(dict(config or {})).encode("utf-8")).hexdigest()


def json_default(value: Any) -> Any:
    """Fallback encoder for numpy scalars, Paths, sets, and frozen dataclasses."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if hasattr(value, "item") and hasattr(value, "dtype"):  # numpy scalar
        return value.item()
    if hasattr(value, "tolist"):  # numpy array
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _git(*args: str, cwd: Path, strip: bool = True) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment specific
        return None
    if out.returncode != 0:
        return None
    # ``--porcelain`` encodes the status in the first two columns, so the leading space of
    # an unstaged change is load-bearing and must survive.
    return out.stdout.strip() if strip else out.stdout


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Commit, branch, and dirty flag for the repository that owns this lab."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    commit = _git("rev-parse", "HEAD", cwd=root)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    status = _git("status", "--porcelain", cwd=root, strip=False)
    paths = (
        []
        if not status
        else sorted({line[3:].strip() for line in status.splitlines() if len(line) > 3})
    )
    return {
        "repo_root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status.strip()),
        "dirty_paths": paths[:50],
        "dirty_path_count": len(paths),
    }


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages whose behaviour can move a result."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _VERSIONED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a missing optional package is a fact to record
            versions[name] = None
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _gpu_info() -> dict[str, Any]:
    """GPU identity without forcing a CUDA context. See the module DEVIATION note."""
    info: dict[str, Any] = {
        "cuda_available": None,
        "device_count": None,
        "name": None,
        "total_memory_bytes": None,
        "gpu_source": "none",
    }
    try:
        torch = importlib.import_module("torch")
    except Exception:  # noqa: BLE001 - torch-free interpreters are supported
        return info

    info["cuda_available"] = bool(torch.cuda.is_available())
    try:
        info["device_count"] = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - driver query can fail on a headless box
        info["device_count"] = 0

    if info["cuda_available"] and torch.cuda.is_initialized():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        info["name"] = str(props.name)
        info["total_memory_bytes"] = int(props.total_memory)
        info["gpu_source"] = "torch"
        return info

    if info["cuda_available"] and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return info
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        if line:
            parts = [p.strip() for p in line.split(",")]
            info["name"] = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                info["total_memory_bytes"] = int(parts[1]) * 1024 * 1024
            info["gpu_source"] = "nvidia-smi"
    return info


def stamp(
    config: Mapping[str, Any] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """The provenance payload attached to every artifact this lab writes."""
    cfg = dict(config or {})
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in some sandboxes
        user = None
    return {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lab_root": str(LAB_ROOT),
        "git": git_state(repo_root),
        "versions": package_versions(),
        "gpu": _gpu_info(),
        "platform": {
            "os": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "hostname": socket.gethostname(),
            "user": user,
            "executable": sys.executable,
        },
        "config": cfg,
        "config_sha256": config_hash(cfg),
        "extra": dict(extra or {}),
    }


def _write_payload(obj: Any, target: Path) -> None:
    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        if not isinstance(obj, (list, tuple)):
            raise TypeError(f"{target.name}: .jsonl needs a sequence of records")
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in obj:
                handle.write(canonical_json(row) + "\n")
        return
    if suffix in {".parquet", ".csv"}:
        frame = obj
        if not hasattr(frame, "to_parquet"):
            pd = importlib.import_module("pandas")
            frame = pd.DataFrame(list(obj))
        if suffix == ".parquet":
            frame.to_parquet(target, index=False)
        else:
            frame.to_csv(target, index=False)
        return
    if suffix not in {".json", ""}:
        raise ValueError(
            f"{target.name}: write_result handles .json/.jsonl/.csv/.parquet, not {suffix!r}"
        )
    payload = obj
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_result(
    obj: Any,
    path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``obj`` to ``path`` and a sibling ``<name>.provenance.json``.

    Both writes are guarded by ``paths.assert_readonly_target``, so a stray path inside
    ``Qwen3-0.6B/`` or ``Rineya_Published_Cores/`` raises before any file is opened.
    Format follows the suffix: ``.json`` (default), ``.jsonl``, ``.csv``, ``.parquet``.
    """
    target = Path(path)
    sidecar = target.with_name(target.name + ".provenance.json")
    assert_readonly_target(target)
    assert_readonly_target(sidecar)

    target.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(obj, target)

    payload = stamp(config, extra=extra)
    payload["artifact"] = {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    return target
