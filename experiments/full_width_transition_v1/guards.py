"""Separate integrity/authorization boundary; historical receipts never authorize FWT.

Hashes detect stale or accidentally altered evidence, not malicious forgery.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol() -> dict:
    return json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))


def check_dependencies() -> dict:
    lock = json.loads((ROOT / "dependency_lock.json").read_text(encoding="utf-8"))
    for name, wanted in lock["git_blob_sha1_lf"].items():
        data = (REPO / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != wanted:
            raise ValueError(f"pinned dependency changed: {name}")
    return lock


def implementation() -> str:
    lock = check_dependencies()
    files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if path.is_file() and not any(x in {"outputs", "verification", "__pycache__", ".pytest_cache"} for x in relative.parts):
            if path.suffix in {".py", ".json", ".md", ".txt"}:
                files[relative.as_posix()] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return hashlib.sha256(canonical({"files": files, "dependencies": lock})).hexdigest()


def runtime() -> dict:
    import torch
    try:
        hf = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        hf = None
    libraries = {}
    for name in ("safetensors", "numpy", "tokenizers"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {"libraries": libraries, "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__, "transformers": hf,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "device": "cpu", "dtype": "float32", "mkldnn": torch.backends.mkldnn.enabled,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def require_runtime():
    actual, wanted = runtime(), protocol()["runtime"]
    for key, value in wanted.items():
        if actual.get(key) != value:
            raise ValueError(f"unqualified runtime {key}: expected {value!r}, got {actual.get(key)!r}")
    return actual


def source_lock() -> dict:
    check_dependencies()
    return json.loads((REPO / "experiments/architecture_benchmark_001/source_lock.json").read_text(encoding="utf-8"))


def audit_source(directory: Path) -> str:
    directory = Path(directory)
    lock = source_lock()
    for name, wanted in lock["assets"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha_file(path) != wanted:
            raise ValueError(f"missing or mismatched official source asset: {name}")
    if (directory / "model.safetensors").stat().st_size != lock["weight_bytes"]:
        raise ValueError("source weight byte count mismatch")
    return hashlib.sha256(canonical(lock)).hexdigest()


def binding(source_digest: str) -> dict:
    return {"implementation": implementation(), "source": source_digest, "runtime": runtime(),
            "fixtures": sha_file(ROOT / "fixtures.json")}


def receipt(stage: str, bound: dict, checks: dict, measurements: dict, parent: str | None = None) -> dict:
    p = protocol()
    if stage not in p["required_checks"]:
        raise ValueError("stage is closed or unknown")
    required = p["required_checks"][stage]
    passed = set(checks) == set(required) and all(checks.get(k) is True for k in required)
    body = {"schema": 1, "experiment": "FWT-001", "stage": stage,
            "status": "PASS" if passed else "FAIL", "binding": bound,
            "checks": checks, "measurements": measurements, "parent_sha256": parent,
            "permissions": p["permissions"],
            "licenses_25_step_engineering_only": stage == "P1" and passed}
    body["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def validate_parent(path: Path, expected: dict) -> str:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = body.pop("sha256", None)
    if digest != hashlib.sha256(canonical(body)).hexdigest():
        raise ValueError("parent receipt integrity failure")
    p = protocol()
    if (body.get("schema") != 1 or body.get("experiment") != "FWT-001" or body.get("stage") != "P1"
            or body.get("status") != "PASS" or body.get("binding") != expected
            or body.get("permissions") != p["permissions"]
            or body.get("licenses_25_step_engineering_only") is not True
            or body.get("parent_sha256") is not None
            or set(body.get("checks", {})) != set(p["required_checks"]["P1"])
            or not all(body["checks"][k] is True for k in p["required_checks"]["P1"])):
        raise ValueError("missing, failed, foreign, stale or over-permissive P1 parent")
    return digest


def new_output(path: Path, source: Path) -> Path:
    path, source = Path(path).resolve(), Path(source).resolve()
    legacy = (REPO / "experiments/architecture_benchmark_001").resolve()
    allowed = (ROOT / "outputs").resolve()
    if path == allowed or not path.is_relative_to(allowed) or path.is_relative_to(source) or path.is_relative_to(legacy):
        raise ValueError("output must be a new child of this package's outputs/, outside source and legacy")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value):
    # Exclusive creation preserves failed attempts and forbids overwrite/resume guesses.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
