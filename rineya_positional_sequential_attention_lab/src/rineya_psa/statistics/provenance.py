"""SPEC section 9: every result file carries a sibling ``*.provenance.json``.

``stamp()`` answers "what produced this number" -- commit, working-tree cleanliness,
package versions, device, OS, UTC time, the SPEC 10 determinism record, and the caller's
config with its sha256. ``write_result()`` is the only writer other modules should use:
it calls ``paths.assert_readonly_target`` before it opens anything. Writers that must
produce the payload themselves (parquet through pandas, ``np.savez``, a hashing pass that
streams to disk) call ``write_sidecar()`` on the finished file instead, so "every result
file has a sibling ``*.provenance.json``" holds for those too.

A sidecar with ``"config": {}`` used to be ambiguous: it could mean "this run genuinely
had no config" or "the caller forgot to pass one". ``config`` is therefore a *required*
argument in spirit -- omitting it still writes, because losing an expensive artifact over
a bookkeeping slip is worse, but it raises :class:`ProvenanceWarning` and stamps
``"config_source": "missing"`` so the gap is visible in the artifact itself rather than
only in the terminal that has since scrolled away.

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
import warnings
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rineya_psa import determinism
from rineya_psa.paths import LAB_ROOT, REPO_ROOT, assert_readonly_target

__all__ = [
    "MISSING_CONFIG",
    "ProvenanceWarning",
    "config_hash",
    "canonical_json",
    "git_state",
    "json_default",
    "package_versions",
    "sidecar_path",
    "stamp",
    "write_result",
    "write_sidecar",
]

_VERSIONED_PACKAGES = ("numpy", "scipy", "torch", "transformers", "pandas", "sklearn")

#: How many working-tree entries a sidecar records before it truncates.
_MAX_DIRTY_ENTRIES = 200


class ProvenanceWarning(UserWarning):
    """Raised as a warning when an artifact is written without its caller config."""


class _MissingConfig:
    """Sentinel distinguishing "no config argument" from ``config=None``/``config={}``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "MISSING_CONFIG"

    def __bool__(self) -> bool:
        return False


#: Default for the ``config`` argument. Passing ``None`` or ``{}`` explicitly is an
#: assertion that the run really had no config and is recorded as such.
MISSING_CONFIG = _MissingConfig()


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


def _parse_porcelain(status: str) -> list[dict[str, str]]:
    """``git status --porcelain`` lines as ``{"status": "XY", "path": ...}`` records.

    The two status columns are the whole point: ``"??"`` (an untracked file that cannot
    have changed the code that ran) and ``" M"`` (a tracked source file that did) both
    reduce to a bare path, which is why the old ``dirty_paths`` list could not tell a
    stale artifact from a current one. Rename entries record the destination.
    """
    entries: list[dict[str, str]] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:].strip()
        if " -> " in path:  # rename/copy: "R  old -> new"
            path = path.split(" -> ", 1)[1].strip()
        path = path.strip('"')
        if path:
            entries.append({"status": code, "path": path})
    return sorted(entries, key=lambda e: (e["path"], e["status"]))


def _default_scope() -> str | None:
    """This lab's path relative to the repository root, as a posix prefix."""
    try:
        return LAB_ROOT.relative_to(REPO_ROOT).as_posix()
    except ValueError:  # pragma: no cover - lab outside the repo
        return None


def git_state(repo_root: Path | None = None, *, scope: str | None = None) -> dict[str, Any]:
    """Commit, branch, and a working-tree description precise enough to date an artifact.

    Beyond the commit, this answers the question a reader of an old artifact actually
    has: *was the code that produced this the code at that commit?* Three fields do that.

    ``tracked_dirty_paths`` lists only files git already tracks, so untracked scratch in
    an unrelated sibling lab no longer masquerades as a code change -- which it did for
    every sidecar in ``results/``, all of which recorded ``dirty: true``.

    ``scope_dirty`` narrows the same question to this lab's own subtree, since a dirty
    file in another lab cannot have changed this lab's numbers.

    ``diff_sha256`` hashes ``git diff HEAD``, i.e. the exact content of the uncommitted
    tracked edits. Two artifacts stamped with the same commit *and* the same
    ``diff_sha256`` were produced by byte-identical source; differing hashes prove they
    were not, which no combination of commit and path list can establish.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    prefix = _default_scope() if scope is None else scope
    commit = _git("rev-parse", "HEAD", cwd=root)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    status = _git("status", "--porcelain", cwd=root, strip=False)

    entries = [] if not status else _parse_porcelain(status)
    paths = sorted({e["path"] for e in entries})
    tracked = sorted({e["path"] for e in entries if e["status"] != "??"})
    untracked = sorted({e["path"] for e in entries if e["status"] == "??"})
    in_scope = (
        []
        if prefix is None
        else sorted(p for p in tracked if p == prefix or p.startswith(prefix.rstrip("/") + "/"))
    )

    diff = _git("diff", "HEAD", "--no-ext-diff", "--no-textconv", cwd=root, strip=False)
    diff_sha = None if diff is None else hashlib.sha256(diff.encode("utf-8")).hexdigest()

    if status is None:
        code_state = "unknown"
    elif tracked:
        code_state = "dirty-tracked"
    elif untracked:
        code_state = "clean-tracked-untracked-present"
    else:
        code_state = "clean"

    return {
        "repo_root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status.strip()),
        "dirty_paths": paths[:50],
        "dirty_path_count": len(paths),
        "dirty_entries": entries[:_MAX_DIRTY_ENTRIES],
        "tracked_dirty_paths": tracked[:_MAX_DIRTY_ENTRIES],
        "tracked_dirty_count": len(tracked),
        "untracked_paths": untracked[:_MAX_DIRTY_ENTRIES],
        "untracked_count": len(untracked),
        "scope": prefix,
        "scope_dirty": None if status is None else bool(in_scope),
        "scope_dirty_paths": in_scope[:_MAX_DIRTY_ENTRIES],
        "diff_sha256": diff_sha,
        "diff_bytes": None if diff is None else len(diff.encode("utf-8")),
        "code_state": code_state,
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


def _resolve_config(
    config: Mapping[str, Any] | None | _MissingConfig,
    *,
    what: str,
    stacklevel: int,
) -> tuple[dict[str, Any], str]:
    """``(config_dict, config_source)``; warns loudly when the caller supplied nothing."""
    if isinstance(config, _MissingConfig):
        warnings.warn(
            f"{what}: no config= was passed, so the sidecar cannot say what produced "
            "this artifact. Pass the run config (or config={} to assert there was "
            "none). Recording config_source='missing'.",
            ProvenanceWarning,
            stacklevel=stacklevel,
        )
        return {}, "missing"
    return dict(config or {}), "caller"


def stamp(
    config: Mapping[str, Any] | None | _MissingConfig = MISSING_CONFIG,
    *,
    extra: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """The provenance payload attached to every artifact this lab writes."""
    cfg, config_source = _resolve_config(config, what="stamp()", stacklevel=3)
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in some sandboxes
        user = None
    return {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lab_root": str(LAB_ROOT),
        "git": git_state(repo_root),
        "versions": package_versions(),
        "determinism": determinism.describe(),
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
        "config_source": config_source,
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


def sidecar_path(path: str | Path) -> Path:
    """The one true sidecar name for ``path``: ``<name>.provenance.json``."""
    target = Path(path)
    return target.with_name(target.name + ".provenance.json")


def write_sidecar(
    path: str | Path,
    *,
    config: Mapping[str, Any] | None | _MissingConfig = MISSING_CONFIG,
    extra: Mapping[str, Any] | None = None,
    artifact_extra: Mapping[str, Any] | None = None,
    _stacklevel: int = 3,
) -> Path:
    """Stamp an artifact that some other writer already put on disk.

    ``write_result`` cannot serialise everything: ``evaluation.runner.write_parquet``
    hands the frame to pandas, ``significance.matrices`` calls ``np.savez_compressed``,
    and the audit streams CSV as it hashes. Those writers previously had to reimplement
    the sidecar (and ``write_parquet`` simply did not, which is why the primary Phase-3
    parquet has none). They can now write their file however they like and call this on
    the result. Returns the sidecar path.

    ``artifact_extra`` is merged into the ``artifact`` block for format-specific facts,
    e.g. ``{"arrays": [...]}`` for an npz or ``{"rows": n}`` for a frame.
    """
    target = Path(path)
    sidecar = sidecar_path(target)
    assert_readonly_target(target)
    assert_readonly_target(sidecar)
    if not target.is_file():
        raise FileNotFoundError(f"{target}: cannot stamp an artifact that was not written")

    # Resolved here rather than inside ``stamp`` so the warning points at the caller's
    # line instead of at this module.
    cfg, config_source = _resolve_config(
        config, what=f"write_sidecar({target.name})", stacklevel=_stacklevel
    )
    payload = stamp(cfg, extra=extra)
    payload["config_source"] = config_source
    payload["artifact"] = {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        **dict(artifact_extra or {}),
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    return sidecar


def write_result(
    obj: Any,
    path: str | Path,
    *,
    config: Mapping[str, Any] | None | _MissingConfig = MISSING_CONFIG,
    extra: Mapping[str, Any] | None = None,
    artifact_extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``obj`` to ``path`` and a sibling ``<name>.provenance.json``.

    Both writes are guarded by ``paths.assert_readonly_target``, so a stray path inside
    ``Qwen3-0.6B/`` or ``Rineya_Published_Cores/`` raises before any file is opened.
    Format follows the suffix: ``.json`` (default), ``.jsonl``, ``.csv``, ``.parquet``.

    ``config`` is not optional in any meaningful sense -- omitting it emits a
    :class:`ProvenanceWarning` and stamps ``config_source="missing"``.
    """
    target = Path(path)
    assert_readonly_target(target)
    assert_readonly_target(sidecar_path(target))

    target.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(obj, target)

    # stacklevel 4: _resolve_config -> write_sidecar -> write_result -> caller.
    write_sidecar(
        target,
        config=config,
        extra=extra,
        artifact_extra=artifact_extra,
        _stacklevel=4,
    )
    return target
