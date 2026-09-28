"""Bounded JSONL IO and fresh, no-clobber output bundles with a last-written receipt."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from .contracts import canonical, loads, require


def read_jsonl(path: Path, *, max_bytes: int = 64 * 1024 * 1024, max_rows: int = 100000):
    require(path.is_file() and path.stat().st_size <= max_bytes, "missing/oversize JSONL")
    rows, size = [], 0
    with path.open("rb") as handle:
        for line in handle:
            size += len(line)
            require(size <= max_bytes and len(rows) < max_rows, "JSONL budget exceeded")
            require(line.strip() != b"", "blank JSONL row")
            value = loads(line.decode("utf-8"))
            require(type(value) is dict, "JSONL object required")
            rows.append(value)
    require(bool(rows), "empty input JSONL")
    return rows


def write_bundle(directory: Path, files: dict[str, object], *, evidence: str) -> dict:
    require(bool(files) and "MANIFEST.json" not in files, "invalid bundle members")
    for name in files:
        require(Path(name).name == name and name not in {".", ".."}, "flat safe filenames required")
    # mkdir is the exclusive writer admission. Partial directories are preserved
    # and lack MANIFEST.json; never auto-delete/reuse a failed run directory.
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, value in files.items():
        data = (("\n".join(canonical(row) for row in value) + ("\n" if value else ""))
                if name.endswith(".jsonl") else canonical(value) + "\n").encode("utf-8")
        with (directory / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        hashes[name] = hashlib.sha256(data).hexdigest()
    receipt = {"schema": 1, "state": "COMPLETE", "evidence": evidence, "sha256": hashes}
    with (directory / "MANIFEST.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical(receipt) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def verify_bundle(directory: Path) -> dict:
    manifest = loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == 1 and manifest["state"] == "COMPLETE", "incomplete bundle")
    require(set(p.name for p in directory.iterdir()) == set(manifest["sha256"]) | {"MANIFEST.json"},
            "unexpected/missing bundle members")
    for name, expected in manifest["sha256"].items():
        require(Path(name).name == name and name not in {".", ".."}, "unsafe member")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "invalid bundle member")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "bundle corruption")
    return manifest
