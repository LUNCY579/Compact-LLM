from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ..utils import atomic_write_json, sha256_file, timed, utc_now_iso
from .layout import DatasetLayout, discover_dataset_root


class UnsafeArchiveError(ValueError):
    pass


def _validate_member(member: zipfile.ZipInfo) -> PurePosixPath:
    raw = member.filename.replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or raw.startswith("/") or path.is_absolute():
        raise UnsafeArchiveError(f"Archive contains an absolute path: {member.filename!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveError(f"Archive contains an unsafe path: {member.filename!r}")
    if len(path.parts[0]) >= 2 and path.parts[0][1:2] == ":":
        raise UnsafeArchiveError(f"Archive contains a Windows drive path: {member.filename!r}")
    mode = (member.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise UnsafeArchiveError(f"Archive contains a symbolic link: {member.filename!r}")
    return path


def safe_extract_zip(archive: str | Path, destination: str | Path) -> list[Path]:
    source = Path(archive).expanduser().resolve()
    dest = Path(destination).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(source) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"CRC failure in {bad}")
        for member in zf.infolist():
            relative = _validate_member(member)
            target = (dest / Path(*relative.parts)).resolve()
            if os.path.commonpath([str(dest), str(target)]) != str(dest):
                raise UnsafeArchiveError(f"Archive member escapes destination: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            extracted.append(target)
    return extracted


def prepare_dataset(
    archive: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(archive).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_path = Path(output).expanduser().resolve()
    archive_hash = sha256_file(source)
    marker = output_path / "PREPARED_DATASET_MANIFEST.json"
    if marker.is_file() and not overwrite:
        import json

        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("archive_sha256") == archive_hash:
            root = discover_dataset_root(output_path)
            return {**existing, "reused": True, "dataset_root": str(root)}
        raise FileExistsError(
            f"{output_path} already contains a different prepared dataset. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rineya-prepare-", dir=str(output_path.parent)) as tmp_name:
        temp_root = Path(tmp_name) / "payload"
        with timed() as timing:
            extracted = safe_extract_zip(source, temp_root)
            dataset_root = discover_dataset_root(temp_root)
            DatasetLayout(dataset_root).assert_exists()
            if output_path.exists():
                if not overwrite:
                    raise FileExistsError(output_path)
                shutil.rmtree(output_path)
            shutil.move(str(temp_root), str(output_path))
        final_root = discover_dataset_root(output_path)

    manifest = {
        "schema_version": "1.0",
        "prepared_at": utc_now_iso(),
        "archive": str(source),
        "archive_sha256": archive_hash,
        "archive_size_bytes": source.stat().st_size,
        "output": str(output_path),
        "dataset_root": str(final_root),
        "files_extracted": len(extracted),
        "elapsed_seconds": timing["seconds"],
        "reused": False,
    }
    atomic_write_json(marker, manifest)
    return manifest
