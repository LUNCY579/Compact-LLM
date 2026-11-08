"""Small, dependency-free, create-only research artifact primitives."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

class ContractError(ValueError):
    """An input is incomplete, stale, incompatible, or outside its declared role."""

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)

def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result

def loads(text: str) -> Any:
    def bad_constant(value):
        raise ContractError(f"nonfinite JSON: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=bad_constant)

def read_json(path: Path) -> Any:
    return loads(Path(path).read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"blank JSONL line {number}")
            row = loads(line)
            require(isinstance(row, dict), f"non-object JSONL line {number}")
            rows.append(row)
    return rows

def new_directory(path: Path) -> Path:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing existing output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path

def write_new(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()

def write_jsonl_new(path: Path, rows: list[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return file_hash(path)

def positive_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0

def code_digest(directory: Path | None = None) -> str:
    """Only executable package code and the protocol; never mutable evidence."""
    directory = Path(directory or Path(__file__).parent)
    paths = sorted(directory.glob("*.py")) + [directory / "protocol.json", directory / "SOURCE_PINS.json"]
    return digest({p.name: file_hash(p) for p in paths})
