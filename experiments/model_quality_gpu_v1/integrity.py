"""Canonical JSON, bounded hashing, exclusive outputs, and executable provenance."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
BASE = "a36de96b30eb03265db3a18e464a2e2a23ee67bf"
BRANCH = "research/model-quality-targeted-gpu-v1"


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(text: str) -> Any:
    def bad(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=bad)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def regular(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink/traversal paths are not accepted")
    if not path.is_file():
        raise ValueError(f"missing regular file: {path.name}")
    return path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, limit: int = 32 * 1024 * 1024) -> Any:
    path = regular(path)
    if path.stat().st_size > limit:
        raise ValueError("JSON exceeds the bounded reader limit")
    return decode(path.read_text(encoding="utf-8"))


def new_dir(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def write_new(path: Path, value: Any) -> None:
    data = canonical(value) + b"\n"
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.absolute().parents):
        raise ValueError("unsafe output path")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def seal(body: dict) -> dict:
    if "sha256" in body:
        raise ValueError("already sealed")
    return {**body, "sha256": digest(body)}


def unseal(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("sealed object required")
    body = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != digest(body):
        raise ValueError("object digest mismatch")
    return body


def code_identity() -> dict:
    files = {}
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(PACKAGE)
        if set(rel.parts) & {"__pycache__", ".pytest_cache", "outputs", "evidence"}:
            continue
        if path.is_file():
            # Git's Windows CRLF conversion must not alter the semantic code identity.
            files[rel.as_posix()] = hashlib.sha256(
                regular(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"sha256": digest(files), "files": files}


def clean_checkout() -> str:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                       text=True, stderr=subprocess.STDOUT).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise ValueError("full repository checkout required")
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("native execution requires the dedicated branch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("native execution requires a clean checkout")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASE, "HEAD"],
                   check=True, capture_output=True)
    return git("rev-parse", "HEAD")
