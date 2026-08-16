from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..errors import CompatibilityError
from ..utils import read_json, replace_file, sha256_file, stable_id, write_json


WEIGHT_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
)


def _validated_index_weight_map(path: Path, raw: Any, *, index_path: Path) -> dict[str, str]:
    """Validate an index weight map without trusting path strings from JSON.

    Hugging Face index files normally contain simple relative shard names.  We allow
    nested relative paths, but every referenced file must resolve inside the checkpoint
    directory and exist as a regular file.
    """

    if not isinstance(raw, dict) or not raw:
        raise CompatibilityError(f"Empty or invalid weight map: {index_path}")
    root = path.resolve()
    mapping: dict[str, str] = {}
    for raw_key, raw_filename in raw.items():
        key = str(raw_key)
        filename = str(raw_filename)
        if not key or not filename:
            raise CompatibilityError(f"Invalid empty tensor key or shard name in {index_path}")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise CompatibilityError(
                f"Unsafe shard path {filename!r} in checkpoint index {index_path}"
            )
        resolved = (root / relative).resolve()
        if resolved == root or root not in resolved.parents:
            raise CompatibilityError(
                f"Shard path escapes checkpoint directory: {filename!r} in {index_path}"
            )
        if not resolved.is_file():
            raise CompatibilityError(
                f"Checkpoint index references a missing shard: {resolved}"
            )
        mapping[key] = resolved.relative_to(root).as_posix()
    return mapping


@dataclass(frozen=True)
class TensorDescriptor:
    key: str
    shape: tuple[int, ...]
    dtype: str
    source_file: str
    numel: int
    bytes: int


class CheckpointReader:
    """Read Hugging Face safetensors or PyTorch-bin checkpoints by tensor key."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.exists() or not self.path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {self.path}")
        self.format, self.weight_map = discover_weight_map(self.path)
        self._stack = contextlib.ExitStack()
        self._safe_handles: dict[str, Any] = {}
        self._bin_cache_file: str | None = None
        self._bin_cache: dict[str, torch.Tensor] | None = None

    def __enter__(self) -> "CheckpointReader":
        if self.format == "safetensors":
            for filename in sorted(set(self.weight_map.values())):
                handle = self._stack.enter_context(
                    safe_open(str(self.path / filename), framework="pt", device="cpu")
                )
                self._safe_handles[filename] = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._bin_cache = None
        self._stack.close()

    @property
    def keys(self) -> list[str]:
        return sorted(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        if self.format == "safetensors":
            handle = self._safe_handles.get(filename)
            if handle is None:
                with safe_open(str(self.path / filename), framework="pt", device="cpu") as one:
                    return one.get_tensor(key)
            return handle.get_tensor(key)
        if self._bin_cache_file != filename:
            self._bin_cache = torch.load(
                self.path / filename,
                map_location="cpu",
                weights_only=True,
            )
            self._bin_cache_file = filename
        assert self._bin_cache is not None
        value = self._bin_cache[key]
        if not isinstance(value, torch.Tensor):
            raise CompatibilityError(f"Non-tensor value in checkpoint for {key}")
        return value

    def descriptor(self, key: str) -> TensorDescriptor:
        tensor = self.get_tensor(key)
        return TensorDescriptor(
            key=key,
            shape=tuple(int(value) for value in tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            source_file=self.weight_map[key],
            numel=int(tensor.numel()),
            bytes=int(tensor.numel() * tensor.element_size()),
        )

    def descriptors(self) -> dict[str, TensorDescriptor]:
        return {key: self.descriptor(key) for key in self.keys}

    def config(self) -> dict[str, Any]:
        path = self.path / "config.json"
        return read_json(path) if path.exists() else {}

    def has_quantization_markers(self) -> bool:
        config = self.config()
        if config.get("quantization_config"):
            return True
        names = {item.name.lower() for item in self.path.iterdir()}
        return any(marker in name for name in names for marker in ("gptq", "awq", "gguf", "quantize"))


class StreamingSafetensorsWriter:
    """Write tensors into bounded safetensors shards without holding the model twice."""

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
        tensor = tensor.detach().cpu().contiguous()
        size = int(tensor.numel() * tensor.element_size())
        if self._buffer and self._buffer_bytes + size > self.max_shard_bytes:
            self._flush()
        self._buffer[key] = tensor
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

    def finalize(self) -> dict[str, Any]:
        self._flush()
        if not self._parts:
            raise RuntimeError("No tensors were written.")
        weight_map: dict[str, str] = {}
        if len(self._parts) == 1:
            final = self.destination / "model.safetensors"
            replace_file(self._parts[0], final)
            for key in self._part_keys[0]:
                weight_map[key] = final.name
            return {"weight_map": weight_map, "total_size": self._total_size, "shards": [final.name]}
        total = len(self._parts)
        shards: list[str] = []
        for index, (part, keys) in enumerate(zip(self._parts, self._part_keys), 1):
            final = self.destination / f"model-{index:05d}-of-{total:05d}.safetensors"
            replace_file(part, final)
            shards.append(final.name)
            for key in keys:
                weight_map[key] = final.name
        index_payload = {"metadata": {"total_size": self._total_size}, "weight_map": weight_map}
        write_json(self.destination / "model.safetensors.index.json", index_payload)
        return {"weight_map": weight_map, "total_size": self._total_size, "shards": shards}


def discover_weight_map(path: Path) -> tuple[str, dict[str, str]]:
    safe_index = path / "model.safetensors.index.json"
    if safe_index.exists():
        payload = read_json(safe_index)
        weight_map = _validated_index_weight_map(
            path, payload.get("weight_map"), index_path=safe_index
        )
        return "safetensors", weight_map
    single_safe = path / "model.safetensors"
    if single_safe.exists():
        with safe_open(str(single_safe), framework="pt", device="cpu") as handle:
            return "safetensors", {key: single_safe.name for key in handle.keys()}
    shard_files = sorted(path.glob("model-*.safetensors"))
    if shard_files:
        mapping: dict[str, str] = {}
        for shard in shard_files:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in mapping:
                        raise CompatibilityError(f"Duplicate tensor key {key} across safetensors shards.")
                    mapping[key] = shard.name
        return "safetensors", mapping

    bin_index = path / "pytorch_model.bin.index.json"
    if bin_index.exists():
        payload = read_json(bin_index)
        weight_map = _validated_index_weight_map(
            path, payload.get("weight_map"), index_path=bin_index
        )
        return "pytorch_bin", weight_map
    single_bin = path / "pytorch_model.bin"
    if single_bin.exists():
        state = torch.load(single_bin, map_location="cpu", weights_only=True)
        return "pytorch_bin", {str(key): single_bin.name for key in state}
    bin_shards = sorted(path.glob("pytorch_model-*.bin"))
    if bin_shards:
        mapping = {}
        for shard in bin_shards:
            state = torch.load(shard, map_location="cpu", weights_only=True)
            for key in state:
                if key in mapping:
                    raise CompatibilityError(f"Duplicate tensor key {key} across bin shards.")
                mapping[str(key)] = shard.name
        return "pytorch_bin", mapping
    raise CompatibilityError(
        f"No Hugging Face safetensors or pytorch_model.bin checkpoint found under {path}"
    )


def checkpoint_weight_files(path: Path) -> list[Path]:
    _, mapping = discover_weight_map(path)
    return sorted({path / filename for filename in mapping.values()})


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    fmt, mapping = discover_weight_map(path)
    files = sorted({filename for filename in mapping.values()})
    index_names = [
        name
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if (path / name).exists()
    ]
    config_path = path / "config.json"
    return {
        "path": str(path),
        "format": fmt,
        "tensor_count": len(mapping),
        "weight_files": [
            {
                "name": name,
                "bytes": (path / name).stat().st_size,
                "mtime_ns": (path / name).stat().st_mtime_ns,
                "sha256": sha256_file(path / name),
            }
            for name in files
        ],
        "index_files": [
            {
                "name": name,
                "bytes": (path / name).stat().st_size,
                "mtime_ns": (path / name).stat().st_mtime_ns,
                "sha256": sha256_file(path / name),
            }
            for name in index_names
        ],
        "config_sha256": sha256_file(config_path) if config_path.exists() else None,
    }


def checkpoint_fingerprint_digest(fingerprint: dict[str, Any]) -> str:
    """Return a path-independent digest of checkpoint weights and configuration."""

    identity = {
        "format": fingerprint.get("format"),
        "tensor_count": fingerprint.get("tensor_count"),
        "weight_files": [
            {
                "name": item.get("name"),
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            for item in (fingerprint.get("weight_files") or [])
        ],
        "index_files": [
            {
                "name": item.get("name"),
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            for item in (fingerprint.get("index_files") or [])
        ],
        "config_sha256": fingerprint.get("config_sha256"),
    }
    return stable_id(identity, length=64)


def parse_size_gb(value: float) -> int:
    return int(float(value) * 1024**3)


def is_weight_filename(name: str) -> bool:
    lowered = name.lower()
    return bool(
        lowered in {"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"}
        or re.fullmatch(r"model-\d+-of-\d+\.safetensors", lowered)
        or re.fullmatch(r"pytorch_model-\d+-of-\d+\.bin", lowered)
        or lowered.startswith(".part-")
    )
