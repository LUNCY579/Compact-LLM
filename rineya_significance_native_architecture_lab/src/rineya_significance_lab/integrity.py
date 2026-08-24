"""Deterministic content hashing for source trees and run artifacts.

The helpers in this module deliberately hash file bytes and normalized relative
paths only.  Timestamps, permissions, absolute checkout locations, and native
path separators are excluded so equivalent runs can be compared across hosts.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path, PureWindowsPath

CONTENT_MANIFEST_NAME = "content_manifest.json"
_SHA256_HEX_LENGTH = 64

PathLike = str | os.PathLike[str]


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative_path(path: PathLike) -> str:
    """Return a canonical POSIX-style relative path.

    Both slash styles are accepted, redundant ``.`` components are removed,
    and ``..`` components are resolved without allowing the path to escape its
    conceptual root.  Absolute POSIX, Windows drive, and UNC paths are rejected.
    Unicode is normalized to NFC so path spelling is stable across filesystems.
    """

    raw = os.fspath(path)
    if not isinstance(raw, str):  # pragma: no cover - PathLike[str] promises str
        raise TypeError("path must resolve to text")
    canonical = raw.replace("\\", "/")
    windows_path = PureWindowsPath(raw)
    if canonical.startswith("/") or windows_path.drive:
        raise ValueError(f"expected a relative path, got {raw!r}")

    parts: list[str] = []
    for part in canonical.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"relative path escapes its root: {raw!r}")
            parts.pop()
            continue
        parts.append(unicodedata.normalize("NFC", part))

    if not parts:
        raise ValueError("relative file path must not be empty")
    return "/".join(parts)


def _resolve_selection(root: Path, selected: PathLike) -> Path:
    raw = os.fspath(selected)
    candidate = Path(raw)
    if not candidate.is_absolute():
        # Accept manifests/configs authored with the other platform's separator.
        native_relative = raw.replace("\\", os.sep).replace("/", os.sep)
        candidate = root / native_relative
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"selected path is outside the hash root: {selected!r}") from error
    return resolved


def _relative_key(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:  # also protects against symlinks leaving the tree
        raise ValueError(f"file is outside the hash root: {path}") from error
    return normalize_relative_path(relative)


def _selected_files(root: Path, selected_paths: Iterable[PathLike]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for selected in selected_paths:
        resolved = _resolve_selection(root, selected)
        candidates = [resolved] if resolved.is_file() else resolved.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved_file = candidate.resolve(strict=True)
            key = _relative_key(root, candidate)
            try:
                resolved_file.relative_to(root)
            except ValueError as error:
                raise ValueError(f"selected file resolves outside the hash root: {candidate}") from error
            previous = files.get(key)
            if previous is not None and previous != resolved_file:
                raise ValueError(f"multiple files normalize to the same path: {key!r}")
            files[key] = resolved_file
    return files


def hash_tree(root: PathLike, selected_paths: Iterable[PathLike]) -> dict[str, str]:
    """Hash selected files/directories beneath ``root``.

    Directories are traversed recursively.  Returned keys are normalized,
    repository-relative paths and are inserted in lexical order.  Repeated or
    overlapping selections are deduplicated.
    """

    resolved_root = Path(root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    if isinstance(selected_paths, (str, os.PathLike)):
        paths: Iterable[PathLike] = (selected_paths,)
    else:
        paths = selected_paths
    files = _selected_files(resolved_root, paths)
    return {key: _sha256_file(files[key]) for key in sorted(files)}


def artifact_hash_map(
    run_dir: PathLike,
    *,
    content_manifest: PathLike = CONTENT_MANIFEST_NAME,
) -> dict[str, str]:
    """Recursively hash run artifacts, excluding the content manifest itself.

    ``content_manifest`` is interpreted relative to ``run_dir``.  Excluding it
    avoids the self-referential impossibility of a manifest containing its own
    final digest.  No other file class (including hidden files) is implicitly
    excluded.
    """

    root = Path(run_dir).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    excluded = normalize_relative_path(content_manifest)
    hashes = hash_tree(root, (root,))
    return {path: digest for path, digest in hashes.items() if path != excluded}


def content_tree_sha256(file_hashes: Mapping[PathLike, str]) -> str:
    """Return the SHA-256 of a canonical file-hash map.

    The canonical payload is compact UTF-8 JSON with normalized keys, sorted by
    key.  This definition is intentionally simple enough to reproduce outside
    Python::

        sha256(json.dumps(map, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))
    """

    normalized: dict[str, str] = {}
    for raw_path, raw_digest in file_hashes.items():
        path = normalize_relative_path(raw_path)
        digest = raw_digest.lower()
        if len(digest) != _SHA256_HEX_LENGTH:
            raise ValueError(f"invalid SHA-256 digest for {path!r}: {raw_digest!r}")
        try:
            bytes.fromhex(digest)
        except ValueError as error:
            raise ValueError(f"invalid SHA-256 digest for {path!r}: {raw_digest!r}") from error
        if path in normalized:
            raise ValueError(f"multiple hash entries normalize to the same path: {path!r}")
        normalized[path] = digest

    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
