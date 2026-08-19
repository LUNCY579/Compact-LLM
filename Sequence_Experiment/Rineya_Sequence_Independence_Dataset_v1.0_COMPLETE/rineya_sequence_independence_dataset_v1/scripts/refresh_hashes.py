from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "manifests" / "sha256sums.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


paths = []
for path in ROOT.rglob("*"):
    if not path.is_file() or path == OUT:
        continue
    if "__pycache__" in path.parts or "derived_views" in path.parts:
        continue
    if path.suffix in {".zip", ".gz"}:
        continue
    paths.append(path)

lines = [f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in sorted(paths)]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} hashes to {OUT}")
