#!/usr/bin/env python3
"""Combine disjoint prediction JSONL splits with duplicate-id protection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def combine(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError(f"{path}:{line_no}: expected an object with id")
            row_id = str(row["id"])
            if row_id in seen:
                raise ValueError(
                    f"Duplicate prediction id {row_id!r} in {seen[row_id]} and {path}"
                )
            seen[row_id] = path
            rows.append(row)
    if not rows:
        raise ValueError("No prediction rows found")
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [path.resolve(strict=True) for path in args.input]
    output = args.output.resolve()
    rows = combine(inputs)
    atomic_write(
        output,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    manifest = {
        "schema_version": "1.0",
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs
        ],
        "output": str(output),
        "rows": len(rows),
        "duplicate_ids": 0,
    }
    atomic_write(
        output.with_suffix(output.suffix + ".manifest.json"),
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"output": str(output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
