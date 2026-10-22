from __future__ import annotations

import contextlib
import gc
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .hashing import atomic_write_json, canonical_sha256, sha256_file


SINGLE_WEIGHTS = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"
SHARD_PATTERN = re.compile(r"model-\d{5}-of-\d{5}\.safetensors$")
SIDECAR_NAMES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


class CheckpointError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_shard_path(root: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise CheckpointError(f"unsafe shard path in index: {filename!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise CheckpointError(f"shard path escapes checkpoint: {filename!r}")
    if not resolved.is_file():
        raise CheckpointError(f"missing checkpoint shard: {resolved}")
    return resolved


def discover_weight_map(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    index_path = path / INDEX_NAME
    if index_path.is_file():
        raw = _read_json(index_path).get("weight_map")
        if not isinstance(raw, dict) or not raw:
            raise CheckpointError(f"empty or invalid weight_map: {index_path}")
        result: dict[str, str] = {}
        for raw_key, raw_filename in raw.items():
            key = str(raw_key)
            filename = str(raw_filename)
            if not key or not filename:
                raise CheckpointError(f"empty key or shard in {index_path}")
            _safe_shard_path(path, filename)
            result[key] = Path(filename).as_posix()
        return result

    single = path / SINGLE_WEIGHTS
    if not single.is_file():
        raise CheckpointError(f"no safetensors checkpoint found under {path}")
    with safe_open(str(single), framework="pt", device="cpu") as handle:
        return {key: SINGLE_WEIGHTS for key in handle.keys()}


@dataclass(frozen=True)
class TensorDescriptor:
    key: str
    shape: tuple[int, ...]
    dtype: str
    source_file: str
    numel: int
    bytes: int


class CheckpointReader:
    """Read one tensor at a time from a validated safetensors checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.path}")
        self.weight_map = discover_weight_map(self.path)
        self._stack = contextlib.ExitStack()
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "CheckpointReader":
        for filename in sorted(set(self.weight_map.values())):
            self._handles[filename] = self._stack.enter_context(
                safe_open(str(self.path / filename), framework="pt", device="cpu")
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stack.close()
        self._handles = {}

    @property
    def keys(self) -> list[str]:
        return sorted(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        handle = self._handles.get(filename)
        if handle is not None:
            return handle.get_tensor(key)
        with safe_open(
            str(self.path / filename), framework="pt", device="cpu"
        ) as one_handle:
            return one_handle.get_tensor(key)

    def descriptor(self, key: str) -> TensorDescriptor:
        tensor = self.get_tensor(key)
        return TensorDescriptor(
            key=key,
            shape=tuple(int(value) for value in tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            source_file=self.weight_map[key],
            numel=int(tensor.numel()),
            bytes=int(tensor.numel() * tensor.element_size()),
        )

    def config(self) -> dict[str, Any]:
        config_path = self.path / "config.json"
        if not config_path.is_file():
            raise CheckpointError(f"missing config.json: {self.path}")
        raw = _read_json(config_path)
        if not isinstance(raw, dict):
            raise CheckpointError(f"config is not an object: {config_path}")
        return raw

    def is_quantized(self) -> bool:
        config = self.config()
        if config.get("quantization_config"):
            return True
        return any(
            marker in key.lower()
            for key in self.keys
            for marker in ("qweight", "qzeros", "scales", "g_idx")
        )


class StreamingSafetensorsWriter:
    def __init__(self, destination: Path, max_shard_bytes: int) -> None:
        self.destination = destination
        self.max_shard_bytes = max(1, int(max_shard_bytes))
        self.destination.mkdir(parents=True, exist_ok=True)
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._parts: list[Path] = []
        self._part_keys: list[list[str]] = []
        self._total_size = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().cpu().contiguous()
        size = int(value.numel() * value.element_size())
        if self._buffer and self._buffer_bytes + size > self.max_shard_bytes:
            self._flush()
        if key in self._buffer:
            raise CheckpointError(f"duplicate output tensor: {key}")
        self._buffer[key] = value
        self._buffer_bytes += size
        self._total_size += size
        if self._buffer_bytes >= self.max_shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        part = self.destination / f".part-{len(self._parts) + 1:05d}.safetensors"
        save_file(self._buffer, str(part), metadata={"format": "pt"})
        self._parts.append(part)
        self._part_keys.append(list(self._buffer))
        self._buffer = {}
        self._buffer_bytes = 0
        gc.collect()

    def finalize(self) -> dict[str, Any]:
        self._flush()
        if not self._parts:
            raise CheckpointError("no output tensors were written")
        weight_map: dict[str, str] = {}
        if len(self._parts) == 1:
            final = self.destination / SINGLE_WEIGHTS
            os.replace(self._parts[0], final)
            for key in self._part_keys[0]:
                weight_map[key] = final.name
            return {
                "weight_map": weight_map,
                "total_size": self._total_size,
                "shards": [final.name],
            }
        total = len(self._parts)
        shards: list[str] = []
        for index, (part, keys) in enumerate(
            zip(self._parts, self._part_keys, strict=True), 1
        ):
            final = self.destination / f"model-{index:05d}-of-{total:05d}.safetensors"
            os.replace(part, final)
            shards.append(final.name)
            for key in keys:
                weight_map[key] = final.name
        atomic_write_json(
            self.destination / INDEX_NAME,
            {"metadata": {"total_size": self._total_size}, "weight_map": weight_map},
        )
        return {
            "weight_map": weight_map,
            "total_size": self._total_size,
            "shards": shards,
        }


def copy_sidecars(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for item in sorted(source.iterdir(), key=lambda path: path.name):
        if item.is_file() and (
            item.name in SIDECAR_NAMES
            or (item.name.startswith("modeling_") and item.suffix == ".py")
        ):
            shutil.copy2(item, destination / item.name)
            copied.append(item.name)
    return copied


def checkpoint_fingerprint(path: Path, *, hash_weights: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    mapping = discover_weight_map(path)
    filenames = sorted(set(mapping.values()))
    weights = []
    for filename in filenames:
        weight_path = path / filename
        record: dict[str, Any] = {
            "name": filename,
            "bytes": weight_path.stat().st_size,
        }
        if hash_weights:
            record["sha256"] = sha256_file(weight_path)
        weights.append(record)
    index_path = path / INDEX_NAME
    config_path = path / "config.json"
    identity = {
        "tensor_count": len(mapping),
        "weight_files": weights,
        "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
    }
    return {**identity, "digest": canonical_sha256(identity), "path": str(path)}


def iter_weight_files(path: Path) -> Iterator[Path]:
    mapping = discover_weight_map(path)
    for filename in sorted(set(mapping.values())):
        yield path / filename


def validate_safe_destination(destination: Path, sources: list[Path]) -> Path:
    resolved = destination.expanduser().resolve()
    anchor = resolved.anchor
    if str(resolved) in {anchor, resolved.drive + "\\"} or len(resolved.parts) < 3:
        raise CheckpointError(f"refusing broad destination: {resolved}")
    for source in sources:
        source_resolved = source.expanduser().resolve()
        if (
            resolved == source_resolved
            or resolved in source_resolved.parents
            or source_resolved in resolved.parents
        ):
            raise CheckpointError(
                f"destination overlaps source checkpoint: {resolved} vs {source_resolved}"
            )
    return resolved
