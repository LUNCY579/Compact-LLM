"""Fail-closed experiment boundaries, immutable outputs and source-bound receipts."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]


def reject_duplicates(pairs):
    value = {}
    for k, v in pairs:
        if k in value:
            raise ValueError(f"duplicate JSON field: {k}")
        value[k] = v
    return value


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()


def protocol():
    return read_json(PACKAGE / "protocol.json")


def regular_file(path: Path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"regular, non-symlink asset required: {path.name}")


def verify_dependencies(root: Path = ROOT) -> dict:
    lock = read_json(PACKAGE / "dependency_lock.json")
    for name, expected in lock["git_blob_sha1"].items():
        p = root / name
        regular_file(p)
        if git_blob(p.read_bytes().replace(b"\r\n", b"\n")) != expected:
            raise ValueError(f"pinned dependency mismatch: {name}")
    return lock


def runtime(require=False) -> dict:
    expected = protocol()["runtime"]
    versions = {}
    for name in expected:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(python=sys.version, platform=platform.platform(), libraries=versions,
                  device="cpu", dtype="float32", backend="eager", threads=1)
    if require and versions != expected:
        raise ValueError(f"runtime pins not satisfied: {versions}; expected {expected}")
    return result


def checkout() -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise ValueError("run from the full repository checkout, not a copied subdirectory")
    if git("branch", "--show-current") != protocol()["branch"]:
        raise ValueError("official execution requires the isolated experiment branch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("official execution requires a clean checkout")
    if any(line.startswith("S ") for line in git("ls-files", "-t").splitlines()):
        raise ValueError("sparse checkout cannot qualify official execution")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    protocol()["base_commit"], "HEAD"], check=True)
    deps = verify_dependencies()
    names = git("ls-files", "experiments/cross_scale_ced_v1").splitlines()
    actual = {str(p.relative_to(ROOT)) for p in PACKAGE.rglob("*") if p.is_file()
              and not set(p.relative_to(PACKAGE).parts) & {"outputs", "__pycache__", ".pytest_cache"}}
    if not names or actual != set(names):
        raise ValueError("package includes missing or untracked implementation files")
    hashes = {name: sha256_file(ROOT/name) for name in names}
    return dict(commit=git("rev-parse", "HEAD"), package_sha256=digest(hashes),
                dependencies=deps, files=hashes)


def available_ram_bytes() -> int:
    import psutil
    available = int(psutil.virtual_memory().available)
    # Respect container memory ceilings as well as host physical RAM.
    for limit, used in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(limit).exists() and Path(used).exists():
            text = Path(limit).read_text().strip()
            if text != "max":
                available = min(available, max(0, int(text)-int(Path(used).read_text())))
    return available


def require_resources(phase: str) -> dict:
    need = protocol()["minimum_available_ram_gib"][phase]*2**30
    have = available_ram_bytes()
    if have < need:
        raise ValueError(f"host-memory guard: {have/2**30:.2f} GiB available, {need/2**30} required; no override")
    return dict(available_bytes=have, required_bytes=need, phase=phase)


def new_directory(path: Path, *, scoped=False) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("output symlinks are forbidden")
    if scoped:
        try:
            relative = path.relative_to((PACKAGE/"outputs").absolute())
        except ValueError as exc:
            raise ValueError("official outputs must be inside this experiment's outputs directory") from exc
        if not relative.parts or ".." in relative.parts:
            raise ValueError("use a unique run subdirectory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def write_json_new(path: Path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def permissions():
    return dict(protocol()["permissions"])


def seal_receipt(body: dict) -> dict:
    if "receipt_sha256" in body:
        raise ValueError("receipt is already sealed")
    return dict(body, receipt_sha256=digest(body))


def validate_p1(path: Path, binding: dict) -> dict:
    regular_file(path)
    receipt = read_json(path)
    checksum = receipt.pop("receipt_sha256", None)
    if checksum != digest(receipt):
        raise ValueError("receipt checksum mismatch")
    if (receipt.get("schema") != 1 or receipt.get("experiment") != "XSC-001"
            or receipt.get("phase") != "P1" or receipt.get("status") != "PASS"
            or receipt.get("binding") != binding or receipt.get("permissions") != permissions()
            or receipt.get("licenses") != protocol()["P1_licenses"]):
        raise ValueError("receipt scope, permissions or source/runtime/checkout binding mismatch")
    checks = receipt.get("checks", {})
    if set(checks) != set(protocol()["P1_checks"]) or any(v is not True for v in checks.values()):
        raise ValueError("incomplete P1 checks")
    artifacts = receipt.get("artifacts", {})
    expected_artifacts = {"qualification.json", "binding.json"}
    for mode in ("H", "P"):
        expected_artifacts.update({f"{mode}_active_interface/manifest.json", f"{mode}_active_interface/interface.safetensors"})
    if set(artifacts) != expected_artifacts:
        raise ValueError("missing qualification evidence")
    for name, sha in artifacts.items():
        p = path.parent/name
        regular_file(p)
        if sha256_file(p) != sha:
            raise ValueError("receipt artifact mismatch")
    evidence = read_json(path.parent/"qualification.json")
    if (evidence.get("schema") != 1 or evidence.get("experiment") != "XSC-001"
            or evidence.get("phase") != "P1" or evidence.get("binding") != binding
            or evidence.get("matched_inventory") is not True):
        raise ValueError("qualification artifact has the wrong scope")
    if set(evidence.get("results", {})) != {"H", "P"}:
        raise ValueError("qualification artifact is missing an arm")
    for mode in ("H", "P"):
        result = evidence["results"][mode]
        for name in ("zero_identity_max_abs", "future_leak_max_abs", "repeated_position_causality",
                     "segment_isolation", "padding_isolation", "disabled_identity_max_abs", "checkpoint_max_abs"):
            if result.get(name) != 0.0:
                raise ValueError(f"unverified exact invariant: {mode}/{name}")
        for name in ("passed", "all_masked_finite", "cache_immutability", "frozen_backbones_unchanged"):
            if result.get(name) is not True:
                raise ValueError(f"unverified mechanical check: {mode}/{name}")
        if result.get("gradient_tensors") != 148 or len(set(result.get("gradient_tensor_names", []))) != 148:
            raise ValueError("not a complete E28/D36 interface gradient check")
        for name in ("cache_schedule_max_abs", "tail_logits_max_abs", "encoder_intervention_max_abs"):
            value = result.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid qualification metric")
        if result["encoder_intervention_max_abs"] <= 1e-7:
            raise ValueError("inactive memory path cannot authorize engineering training")
    for role in ("encoder", "decoder"):
        native = evidence.get(f"native_{role}", {})
        if (native.get("passed") is not True or native.get("cached_native_checked") is not True
                or native.get("positions") != [0, 257, 32760]
                or not isinstance(native.get("max_abs"), (float, int)) or not math.isfinite(native["max_abs"])):
            raise ValueError("native source comparison is incomplete")
    return receipt
