"""Independent, fail-closed D28 receipts. Hashes are integrity checks, not signatures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
from experiments.full_width_transition_v1.guards import canonical, sha_file, write_json, runtime

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
BASE = "94fbe17945a9d7d88ac2422dab124a95d35ee653"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol():
    return json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))


def dependencies():
    lock = json.loads((ROOT / "dependency_lock.json").read_text(encoding="utf-8"))
    for name, expected in lock["git_blob_sha1_lf"].items():
        path = REPO / name
        require(not path.is_symlink(), f"symlink dependency: {name}")
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(actual == expected, f"pinned dependency changed: {name}")
    return lock


def implementation_files():
    return [p for p in sorted(ROOT.rglob("*")) if p.is_file()
            and not set(p.relative_to(ROOT).parts).intersection({"outputs", "verification", "__pycache__", ".pytest_cache"})
            and p.suffix in {".py", ".json", ".md", ".txt"}]


def implementation():
    lock = dependencies()
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
             for p in implementation_files()}
    return hashlib.sha256(canonical({"files": files, "dependencies": lock})).hexdigest()


def checkout():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()
    head = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    paths = [str(p.relative_to(REPO)) for p in implementation_files()] + list(dependencies()["git_blob_sha1_lf"])
    for path in paths:
        git("ls-files", "--error-unmatch", path)
    require(not git("diff", "HEAD", "--", *paths), "uncommitted implementation/dependency changes")
    return head


def binding(source: Path):
    import torch
    from experiments.full_width_transition_v1.guards import audit_source, require_runtime
    dependencies()
    source = Path(source)
    require(not source.is_symlink() and source.resolve() == (REPO / "Supa").resolve(),
            "official source must be this checkout's Supa/ installation")
    source_hash = audit_source(source)
    actual = require_runtime()
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled("cpu"),
            "default dtype/autocast differs from FP32 lane")
    for key, value in protocol()["libraries"].items():
        require(actual["libraries"].get(key) == value, f"unqualified library: {key}")
    return {"profile": "official_supa", "implementation": implementation(), "source": source_hash,
            "runtime": actual, "checkout": checkout(), "fixtures": sha_file(ROOT / "fixtures.json")}


def receipt(stage, bound, checks, measurements, parent=None):
    p = protocol()
    require(stage in p["required_checks"], "closed or unknown stage")
    require(bound.get("profile") == "official_supa", "tiny evidence cannot produce official receipts")
    required = p["required_checks"][stage]
    passed = set(checks) == set(required) and all(checks.get(k) is True for k in required)
    require((stage == "P1" and parent is None) or (stage == "P2" and isinstance(parent, str) and len(parent) == 64),
            "invalid stage parent")
    body = {"schema": 1, "experiment": "D28-001", "stage": stage, "status": "PASS" if passed else "FAIL",
            "binding": bound, "checks": checks, "measurements": measurements, "parent_sha256": parent,
            "permissions": p["permissions"], "licenses_engineering_smoke": stage == "P1" and passed}
    body["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def validate_parent(path: Path, expected):
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = body.pop("sha256", None)
    require(digest == hashlib.sha256(canonical(body)).hexdigest(), "parent integrity failure")
    p = protocol()
    require(set(body) == {"schema", "experiment", "stage", "status", "binding", "checks", "measurements",
                          "parent_sha256", "permissions", "licenses_engineering_smoke"}, "parent schema mismatch")
    require(expected.get("profile") == "official_supa" and body["binding"] == expected, "foreign/stale parent binding")
    require(body["schema"] == 1 and body["experiment"] == "D28-001" and body["stage"] == "P1"
            and body["status"] == "PASS" and body["parent_sha256"] is None
            and body["licenses_engineering_smoke"] is True and body["permissions"] == p["permissions"],
            "failed, foreign or over-permissive parent")
    required = p["required_checks"]["P1"]
    require(set(body["checks"]) == set(required) and all(body["checks"][k] is True for k in required),
            "incomplete P1 checks")
    require(isinstance(body["measurements"], dict), "missing P1 checkpoint measurements")
    for arm in ("H56", "P56", "C56"):
        measured = body["measurements"].get(arm)
        require(isinstance(measured, dict), f"missing {arm} roundtrip artifacts")
        for filename, key in (("model.safetensors", "checkpoint_weights_sha256"),
                              ("config.json", "checkpoint_config_sha256")):
            artifact = Path(path).parent / (arm + "_initial") / filename
            require(not artifact.is_symlink() and not artifact.parent.is_symlink() and artifact.is_file(),
                    f"missing/symlink P1 artifact: {arm}/{filename}")
            require(sha_file(artifact) == measured.get(key), f"changed P1 artifact: {arm}/{filename}")
    return digest


def new_output(path: Path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink output path")
    allowed, resolved = (ROOT / "outputs").resolve(), path.resolve()
    require(resolved != allowed and resolved.is_relative_to(allowed), "output must be a new child of D28 outputs/")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved
