from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from generate_manifest import files_for, project_version


_TRANSIENT_FILES = {".coverage", "coverage.xml"}
_TRANSIENT_DIRECTORIES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def clean(root: Path) -> None:
    for directory in _TRANSIENT_DIRECTORIES:
        for path in root.rglob(directory):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    for path in root.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    for name in _TRANSIENT_FILES:
        (root / name).unlink(missing_ok=True)


def _run_validation(root: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "scripts/export_schemas.py"], cwd=root, env=env, check=True)
    subprocess.run(
        [sys.executable, "scripts/validate_repository.py", "--with-tests"],
        cwd=root,
        env=env,
        check=True,
    )
    clean(root)
    subprocess.run(
        [sys.executable, "scripts/generate_manifest.py"],
        cwd=root,
        env=env,
        check=True,
    )


def package(root: Path, output: Path, *, validate: bool = True) -> tuple[str, int]:
    root = root.resolve()
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    clean(root)
    if validate:
        _run_validation(root)
        clean(root)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    version = project_version(root)
    prefix = f"rineya-context-harness-v{version}/"
    excluded = {output, sidecar}

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for path, relative in files_for(root, excluded_paths=excluded):
            info = zipfile.ZipInfo(prefix + relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.suffix in {".sh", ".ps1"} else 0o644
            info.external_attr = mode << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)

    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP member: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("release ZIP contains duplicate member names")
        if any(name.endswith(".coverage") or "__pycache__" in name for name in names):
            raise RuntimeError("release ZIP contains transient test/cache artifacts")
        if not any(name.endswith("artifacts/release/validation_report.json") for name in names):
            raise RuntimeError("release ZIP is missing validation_report.json")
        if not any(name.endswith("artifacts/release/repository_manifest.json") for name in names):
            raise RuntimeError("release ZIP is missing repository_manifest.json")
        count = len(names)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return digest, count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Rineya_Context_Harness_v0.1.0.zip"),
    )
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else (root / args.output).resolve()
    digest, count = package(root, output, validate=not args.skip_validation)
    print(f"Created: {output}")
    print(f"Members: {count}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
