"""Create-only JSON/safetensors artifacts. Never deserialize a pickle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from .memory import FrozenBilinear


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: str | Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def new_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_memory(path: str | Path, model: FrozenBilinear, receipt: dict) -> dict:
    path = new_directory(path)
    weights = path / "memory.safetensors"
    save_file({name: t.detach().cpu().contiguous() for name, t in model.named_buffers()}, str(weights))
    manifest = {"schema": "RHF-001-memory-v1", "weights_sha256": sha256(weights),
                "weights_bytes": weights.stat().st_size, "receipt": receipt,
                "shapes": {name: list(t.shape) for name, t in model.named_buffers()},
                "native_qualified": False}
    json_write(path / "manifest.json", manifest)
    return manifest


def load_memory(path: str | Path, max_bytes: int = 256 * 1024**2) -> tuple[FrozenBilinear, dict]:
    path = Path(path)
    if path.is_symlink() or {p.name for p in path.iterdir()} != {"manifest.json", "memory.safetensors"}:
        raise ValueError("unexpected memory artifact entries")
    manifest_path, weights = path / "manifest.json", path / "memory.safetensors"
    if manifest_path.is_symlink() or weights.is_symlink():
        raise ValueError("symlinked artifacts are rejected")
    if manifest_path.stat().st_size > 1024**2 or weights.stat().st_size > max_bytes:
        raise MemoryError("artifact exceeds load budget")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "RHF-001-memory-v1":
        raise ValueError("unknown artifact schema")
    if weights.stat().st_size != manifest["weights_bytes"] or sha256(weights) != manifest["weights_sha256"]:
        raise ValueError("artifact hash/size mismatch")
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        if set(f.keys()) != {"a", "g", "b"}:
            raise ValueError("unexpected tensors")
        shapes = {name: f.get_slice(name).get_shape() for name in f.keys()}
        if shapes != manifest["shapes"]:
            raise ValueError("shape manifest mismatch")
    tensors = load_file(str(weights), device="cpu")
    model = FrozenBilinear(tensors["a"], tensors["g"], tensors["b"])
    return model, manifest
