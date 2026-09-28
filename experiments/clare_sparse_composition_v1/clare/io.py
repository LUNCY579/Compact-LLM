"""Byte-exact, overwrite-refusing file helpers.

Everything is written in binary mode with LF line endings: Windows text mode rewrites LF to CRLF
and silently changes every recorded sha256 (see the programme's line-ending incident). Outputs
are never overwritten unless the caller passes ``overwrite=True`` explicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_bytes(path: str | os.PathLike, data: bytes, overwrite: bool = False) -> str:
    """Write ``data`` atomically-ish (temp file + rename); return its sha256."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"stale partial file {tmp}; inspect before removing")
    with open(tmp, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_bytes(data)


def write_json(path: str | os.PathLike, obj: Any, overwrite: bool = False) -> str:
    return write_bytes(path, pretty_json(obj).encode("ascii"), overwrite=overwrite)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
