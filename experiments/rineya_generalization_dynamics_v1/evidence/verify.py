"""Verify this published archive using only the standard library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    index = json.loads((root / "PUBLICATION_INDEX.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in index.items() if key != "sha256"}
    if index.get("sha256") != digest(body):
        raise ValueError("Publication index digest mismatch")
    expected = {"PUBLICATION_INDEX.json"}
    for entry in index["published_files"]:
        relative = Path(entry["published_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in expected:
            raise ValueError("Unsafe or duplicate publication path")
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or unsafe published payload: {relative}")
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"Published payload changed: {relative}")
        expected.add(relative.as_posix())
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts}
    if actual != expected:
        raise ValueError("Publication inventory changed")
    print(json.dumps({"status": "PASS_CONTENT_INTEGRITY", "files": len(expected) - 1,
                      "native_execution_verified": False}, sort_keys=True))


if __name__ == "__main__":
    main()
