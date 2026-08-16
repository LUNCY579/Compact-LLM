from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


IGNORED_DIR_NAMES = {".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__", ".venv"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
MANIFEST_NAME = "RELEASE_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shipped_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == MANIFEST_NAME:
            continue
        if any(part in IGNORED_DIR_NAMES for part in relative.parts):
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        files[relative.as_posix()] = path
    return files


def _safe_manifest_path(root: Path, value: Any) -> tuple[str, Path] | None:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
        return None
    resolved = (root / relative).resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved or root_resolved not in resolved.parents:
        return None
    return relative.as_posix(), resolved


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"Missing release manifest: {manifest_path}")
        return 1
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid release manifest: {exc}")
        return 1
    if str(payload.get("schema_version")) != "1.0":
        print(f"Unsupported release manifest schema: {payload.get('schema_version')!r}")
        return 1

    errors: list[str] = []
    entries = payload.get("files")
    if not isinstance(entries, list):
        print("Release manifest 'files' must be a list")
        return 1

    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("non-object manifest entry")
            continue
        safe = _safe_manifest_path(root, entry.get("path"))
        if safe is None:
            errors.append(f"unsafe path: {entry.get('path')!r}")
            continue
        relative, path = safe
        if relative in expected_paths:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        expected_paths.add(relative)
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        try:
            expected_bytes = int(entry["bytes"])
            expected_hash = str(entry["sha256"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid metadata: {relative}")
            continue
        if path.stat().st_size != expected_bytes:
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256_file(path) != expected_hash:
            errors.append(f"hash mismatch: {relative}")

    actual_paths = set(_shipped_files(root))
    for relative in sorted(actual_paths - expected_paths):
        errors.append(f"unmanifested file: {relative}")
    for relative in sorted(expected_paths - actual_paths):
        # Missing files are normally reported above; this catches ignored/path-policy drift.
        errors.append(f"manifested file excluded by release policy: {relative}")

    if errors:
        print("Release verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Release manifest verified: {len(expected_paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
