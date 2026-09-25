"""Write a deterministic hash manifest for the experiment package."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    package = root / "experiments" / "soft_token_reasoning_v1"
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json" and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
            rel = path.relative_to(root).as_posix()
            files[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "schema_version": "1.0",
        "package": "experiments/soft_token_reasoning_v1",
        "base_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "origin/main"], text=True).strip(),
        "source_head_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip(),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "excluded_from_self_hash": "MANIFEST.json",
        "files": files,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"source_head_sha": payload["source_head_sha"], "file_count": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
