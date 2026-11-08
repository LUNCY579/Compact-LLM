"""Canonical hashing helpers used by every persisted PCSM artifact."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def sha256_bytes(value: bytes) -> str:
    """Return the lower-case SHA-256 digest for *value*."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for stable experiment identifiers."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def mask_bytes(mask: torch.Tensor | np.ndarray) -> bytes:
    """Return canonical row-major binary bytes, rejecting non-binary masks."""

    if isinstance(mask, torch.Tensor):
        value = mask.detach().to(device="cpu")
        if value.dtype is not torch.bool:
            if not torch.all((value == 0) | (value == 1)):
                raise ValueError("Mask contains values outside {0, 1}")
            value = value.to(dtype=torch.bool)
        array = value.contiguous().numpy()
    else:
        array = np.asarray(mask)
        if array.dtype != np.bool_:
            if not np.logical_or(array == 0, array == 1).all():
                raise ValueError("Mask contains values outside {0, 1}")
            array = array.astype(np.bool_, copy=False)
        array = np.ascontiguousarray(array)
    shape = cast(bytes, np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    payload = array.astype(np.uint8, copy=False).tobytes(order="C")
    return shape + payload


def mask_hash(mask: torch.Tensor | np.ndarray) -> str:
    return sha256_bytes(mask_bytes(mask))


def tree_hash(root: Path, relative_paths: Iterable[Path] | None = None) -> str:
    """Hash path names, sizes, and contents for an auditable file tree."""

    paths = (
        sorted(path for path in root.rglob("*") if path.is_file())
        if relative_paths is None
        else sorted(root / path for path in relative_paths)
    )
    records: list[Mapping[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return canonical_json_hash(records)


def fingerprint_tensors(
    tensors: Mapping[str, torch.Tensor], *, exclude: Iterable[str] = ()
) -> str:
    """Hash tensor names, dtype/shape, and exact bytes without retaining copies."""

    excluded = set(exclude)
    digest = hashlib.sha256()
    digest.update(b"rineya-pcsm-native-tensors-v1\0")
    for name in sorted(tensors):
        if name in excluded:
            continue
        tensor = tensors[name].detach().to(device="cpu").contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<q", int(dimension)))
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()
