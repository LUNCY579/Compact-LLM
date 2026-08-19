from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    required = [
        "README.md",
        "pyproject.toml",
        "src/rineya_state_lab/cli.py",
        "src/rineya_state_lab/training/run.py",
        "src/rineya_state_lab/evaluation/runner.py",
        "configs/train/full_context_lora.yaml",
        "configs/train/split_stateful_lora.yaml",
        "assets/datasets/Rineya_Sequence_Independence_Dataset_v1.0_COMPLETE.zip",
    ]
    errors = []
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {relative}")
    archive = root / required[-1]
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()
                if bad:
                    errors.append(f"dataset CRC failure: {bad}")
                if len(zf.namelist()) < 20:
                    errors.append("dataset archive contains too few entries")
        except Exception as exc:
            errors.append(f"dataset archive error: {type(exc).__name__}: {exc}")
    file_count = sum(path.is_file() for path in root.rglob("*"))
    source_bytes = sum(path.stat().st_size for path in root.rglob("*.py"))
    if file_count < 50:
        errors.append(f"repository unexpectedly small: {file_count} files")
    if source_bytes < 50_000:
        errors.append(f"Python implementation unexpectedly small: {source_bytes} bytes")
    report = {
        "root": str(root),
        "passed": not errors,
        "errors": errors,
        "file_count": file_count,
        "python_source_bytes": source_bytes,
        "dataset_sha256": sha256(archive) if archive.is_file() else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
