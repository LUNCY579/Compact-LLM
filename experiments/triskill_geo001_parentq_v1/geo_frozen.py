"""Load frozen TRISKILL-GEO-001 section H modules from their pinned bytes (CPU contexts).

The native worker binds them through GEO's own dependency binding and validates the whole
GEO freeze (parentq_worker.py). CPU builders and tests use this narrower loader: it checks
FREEZE_SECTIONH.json against its registered digest, then each module against that freeze,
before executing the verified bytes. Nothing here imports torch or reads a checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
GEO_DIR = HERE.parent / "triskill_geometric_pilot_v1"
GEO_FREEZE_NAME = "FREEZE_SECTIONH.json"
GEO_FREEZE_SHA256 = "f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687"
_CACHE: dict[str, object] = {}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def geo_freeze(path: Path | None = None) -> dict:
    data = (path or GEO_DIR / GEO_FREEZE_NAME).read_bytes()
    if sha256_bytes(data) != GEO_FREEZE_SHA256:
        raise ValueError("STOP_GEO_FREEZE_DIGEST_MISMATCH")
    return json.loads(data)


def load(name: str):
    """A fresh module executed from the frozen bytes of GEO's `<name>.py`."""
    if name in _CACHE:
        return _CACHE[name]
    expected = geo_freeze()["files"].get(f"{name}.py")
    path = GEO_DIR / f"{name}.py"
    source = path.read_bytes()
    if expected is None or sha256_bytes(source) != expected:
        raise ValueError(f"STOP_GEO_MODULE_NOT_FROZEN:{name}")
    spec = importlib.util.spec_from_file_location(f"_parentq_geo_{name}_{expected[:12]}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    _CACHE[name] = module
    return module
