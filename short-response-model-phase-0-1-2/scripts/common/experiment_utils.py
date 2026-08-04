from __future__ import annotations

"""Small dependency-free helpers shared by every phase.

The phase runners intentionally keep orchestration in PowerShell and durable
artifact handling in Python. These helpers centralize atomic JSON/text writes,
JSONL recovery, reproducible hashing, environment capture, and checked
subprocess execution so individual phase scripts do not silently drift.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON and return the decoded value."""

    if not path.exists():
        recover_atomic_write(path, validator="json")
    return json.loads(path.read_text(encoding="utf-8"))


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def _atomic_sibling_candidates(path: Path) -> list[Path]:
    if not path.parent.exists():
        return []
    return sorted(
        path.parent.glob(f".{path.name}.tmp-*"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )


def _temporary_writer_is_alive(path: Path) -> bool:
    match = re.search(r"\.tmp-(\d+)-", path.name)
    if match is None:
        return False
    pid = int(match.group(1))
    if pid == os.getpid():
        return True
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a harmless existence probe on Windows;
        # non-control signals are implemented with TerminateProcess. Query a
        # limited process handle instead.
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            # Access denied still proves that a process with this PID exists.
            return ctypes.get_last_error() == 5
        except (AttributeError, OSError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _validate_recovery_bytes(data: bytes, validator: str | None) -> None:
    text = data.decode("utf-8")
    if validator == "json":
        json.loads(text)
    elif validator == "jsonl":
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Recovered JSONL row {line_number} is not an object")
    elif validator not in {None, "text"}:
        raise ValueError(f"Unsupported atomic recovery validator: {validator}")


def recover_atomic_write(path: Path, *, validator: str | None = None) -> bool:
    """Recover a complete orphaned sibling temp file after a hard interruption.

    Normal atomic writes remove their temporary sibling in ``finally``. A hard
    process termination can leave the fully flushed temp file behind before
    ``os.replace`` executes. When the destination is absent, this helper scans
    those siblings newest-first, ignores files owned by a still-running writer,
    validates their complete contents, and atomically promotes the first valid
    candidate. It never replaces an existing destination.
    """

    if path.exists():
        return False
    candidates = _atomic_sibling_candidates(path)
    for candidate in candidates:
        if _temporary_writer_is_alive(candidate):
            continue
        try:
            data = candidate.read_bytes()
            _validate_recovery_bytes(data, validator)
            os.replace(candidate, path)
            for stale in candidates:
                if stale == candidate or _temporary_writer_is_alive(stale):
                    continue
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass
            return True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
            continue
    return False


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace *path* with complete bytes on Windows and POSIX.

    The temporary file is a sibling so ``os.replace`` stays on the same volume.
    Data is flushed and fsynced before replacement; failed writes leave the old
    destination intact and best-effort cleanup removes the temporary sibling.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace a UTF-8 text or Markdown file."""

    atomic_write_bytes(path, text.encode(encoding))


def write_json(path: Path, obj: Any) -> None:
    """Atomically write stable, human-readable UTF-8 JSON."""

    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while reporting the exact malformed line."""

    rows, warnings = read_jsonl_recover(path, repair_truncated_tail=False)
    if warnings:
        raise ValueError(warnings[0])
    return rows


def read_jsonl_recover(
    path: Path,
    *,
    id_field: str | None = None,
    repair_truncated_tail: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read JSONL, safely recovering a malformed final line when requested.

    A malformed non-final line remains a structural error. A malformed final
    line is treated as a likely interrupted append: every earlier valid row is
    preserved and the file is atomically rewritten without the damaged tail.
    Duplicate stable IDs are deduplicated by keeping the first valid completed
    row, which matches append-only resume semantics and avoids silent evidence
    replacement.
    """

    if not path.exists() and not recover_atomic_write(path, validator="jsonl"):
        return [], []
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    final_nonempty = nonempty_indexes[-1] if nonempty_indexes else -1
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    needs_rewrite = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        line_number = index + 1
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"{path}:{line_number}: {exc}"
            if repair_truncated_tail and index == final_nonempty:
                warnings.append(f"Recovered malformed final JSONL line: {message}")
                needs_rewrite = True
                break
            raise ValueError(message) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: every JSONL row must be an object")
        if id_field is not None:
            raw_id = value.get(id_field)
            if raw_id is None:
                raise ValueError(f"{path}:{line_number}: missing stable ID field {id_field!r}")
            stable_id = str(raw_id)
            if stable_id in seen:
                warnings.append(f"Ignored duplicate {id_field}={stable_id!r} at {path}:{line_number}")
                needs_rewrite = True
                continue
            seen.add(stable_id)
        rows.append(value)
    if needs_rewrite:
        write_jsonl(path, rows)
    return rows, warnings


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    """Append one completed JSON object and force it to stable storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write JSON objects as one UTF-8 line each."""

    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def sha256(path: Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a file or directory tree in a path-independent deterministic way."""

    original_value = str(path)
    if not path.exists():
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
