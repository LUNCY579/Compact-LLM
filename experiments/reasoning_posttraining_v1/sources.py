"""Read-only release verification before importing supplied solver code."""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import zipfile

from experiments.model_quality_gpu_v1.integrity import digest, file_hash, read_json, regular

ARCHIVES = {
    "forge_v1": ("rineya_reasoning_forge_v1", "3d490c386bf5d0d832a860403253aeff29266ade84f93c384a0a3645cff0eaac"),
    "forge_v2": ("rineya_reasoning_forge_rl_v2", "d3bcd3d84c2d123c9876d334e867b0387f78cb7e93c7b625eafddbb0c685b663"),
}


def stable_hash(path: Path) -> str:
    path = regular(path)
    before = path.stat()
    value = file_hash(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed while hashing")
    return value


def verify_release(archive: Path, root: Path, release: str) -> dict:
    prefix, expected = ARCHIVES[release]
    archive, root = regular(archive), Path(root).absolute()
    if stable_hash(archive) != expected:
        raise ValueError("supplied archive does not match the authorized release")
    files = {}
    with zipfile.ZipFile(archive) as source:
        if sum(x.file_size for x in source.infolist()) > 512 * 2**20:
            raise ValueError("release exceeds bounded expansion inventory")
        for member in source.infolist():
            p = PurePosixPath(member.filename)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != prefix:
                raise ValueError("unsafe archive path")
            if member.is_dir():
                continue
            rel = PurePosixPath(*p.parts[1:]).as_posix()
            if rel in files:
                raise ValueError("duplicate archive member")
            h = hashlib.sha256()
            with source.open(member) as stream:
                for chunk in iter(lambda: stream.read(2**20), b""):
                    h.update(chunk)
            local = root.joinpath(*p.parts[1:])
            if stable_hash(local) != h.hexdigest() or local.stat().st_size != member.file_size:
                raise ValueError(f"release tree mismatch: {rel}")
            files[rel] = {"sha256": h.hexdigest(), "bytes": member.file_size}
    unexpected = {p.relative_to(root).as_posix() for p in root.rglob("*.py")} - set(files)
    if unexpected:
        raise ValueError(f"unlisted executable source: {sorted(unexpected)}")
    return {"release": release, "archive": str(archive), "archive_sha256": expected,
            "root": str(root), "files": files, "tree_sha256": digest(files),
            "evidence_class": "FRESH_BYTE_VERIFICATION_NO_MODEL_EXECUTION"}


def import_forge_v2(binding: dict):
    if binding["release"] != "forge_v2":
        raise ValueError("wrong solver release")
    root = Path(binding["root"]).resolve()
    for rel, entry in binding["files"].items():
        if rel.endswith(".py") and stable_hash(root / rel) != entry["sha256"]:
            raise ValueError("solver source changed after release verification")
    for name, module in list(sys.modules.items()):
        if name == "forge_rl" or name.startswith("forge_rl."):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise ValueError("a different solver package is already imported")
    sys.path.insert(0, str(root))
    try:
        modules = {n: importlib.import_module("forge_rl." + n)
                   for n in ("common", "text", "oracle", "rewards", "validate")}
    finally:
        sys.path.remove(str(root))
    return modules


def import_forge_v1_validator(binding: dict):
    """Only its read-only validate_row/solve_prompt functions may be called here.

    The upstream validate_release writes its original report and is not invoked.
    """
    if binding["release"] != "forge_v1":
        raise ValueError("wrong primitive solver release")
    path = Path(binding["root"]) / "scripts/verify.py"
    if stable_hash(path) != binding["files"]["scripts/verify.py"]["sha256"]:
        raise ValueError("primitive solver source changed")
    spec = importlib.util.spec_from_file_location("rrp_verified_forge_v1_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safetensor_inventory(path: Path, expected_hash: str) -> dict:
    """Count serialized unique tensors from a bounded header, without model loading."""
    if stable_hash(path) != expected_hash:
        raise ValueError("source checkpoint hash mismatch")
    with regular(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 2**20:
            raise ValueError("unexpectedly large safetensors header")
        from experiments.model_quality_gpu_v1.integrity import decode
        header = decode(stream.read(size).decode("utf-8"))
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    total = sum(math.prod(v["shape"]) for v in entries.values())
    tail = {k: v for k, v in entries.items() if any(k.startswith(f"model.layers.{i}.") for i in range(8, 28))}
    trainable = sum(math.prod(v["shape"]) for v in tail.values())
    if (total, trainable, len(tail)) != (596049920, 314618880, 220):
        raise ValueError("source serialized parameter inventory differs")
    return {"weight_sha256": expected_hash, "unique_serialized_parameters": total,
            "tail20_parameters": trainable, "tail20_tensors": len(tail),
            "stored_dtypes": sorted({v["dtype"] for v in entries.values()}),
            "native_instantiated_inventory": "NOT_RUN", "model_weights_loaded": False}
