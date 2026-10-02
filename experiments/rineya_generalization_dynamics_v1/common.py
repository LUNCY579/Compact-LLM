"""Fail-closed IO, content identities, and single-writer experiment state."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator


class Blocked(ValueError):
    """The requested operation violates the registered experiment contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Blocked(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> Any:
    def reject(value: str) -> None:
        raise Blocked(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_new(path: Path, value: Any) -> None:
    """Exclusive create: incomplete or previously used artifacts are never replaced."""
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def code_hash() -> str:
    root = Path(__file__).parent
    return digest({p.relative_to(root).as_posix(): file_hash(p)
                   for p in sorted(root.glob("*.py"))})


def signed(payload: dict) -> dict:
    require("sha256" not in payload, "Digest field already present")
    return {**payload, "sha256": digest(payload)}


def verify(payload: dict) -> dict:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    require(payload.get("sha256") == digest(body), "Artifact digest mismatch")
    return payload


@contextlib.contextmanager
def exclusive(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise Blocked(f"Busy or interrupted owner: {path}; inspect manually") from exc
    try:
        with stream:
            stream.write(str(os.getpid()))
            stream.flush()
        yield
    finally:
        path.unlink()


def normalized_prompt_hash(prompt: str) -> str:
    return digest(" ".join(prompt.split()))
