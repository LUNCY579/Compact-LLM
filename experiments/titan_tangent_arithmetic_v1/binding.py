"""Stdlib-only local preflight. Missing local files never disprove a remote backup."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess

SUPA_SHA256 = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
TITAN_PARTIAL_SHA256 = "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d"


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_digest() -> str:
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def checked_file(item: dict) -> Path:
    p = Path(item["path"]).expanduser().resolve()
    expected = item["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("missing exact SHA256")
    if not p.is_file():
        raise FileNotFoundError(f"LOCAL_ASSET_NOT_OBSERVED: {p}; remote custody not assessed")
    if digest(p) != expected:
        raise ValueError(f"asset hash mismatch: {p}")
    return p


def qualification_identity(m: dict) -> str:
    fields = {k: m[k] for k in ("base", "scope_prefixes", "runtime", "qualification_probe", "package_sha256")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preflight(m: dict, *, check_git: bool = True, check_runtime: bool = True) -> dict:
    if m.get("schema") != "TTA-001-HOST-1" or m.get("action") not in {"qualify", "diagnostic", "study"}:
        raise ValueError("unknown manifest schema or action")
    if m.get("device") != "cpu":
        raise ValueError("first native adapter is CPU-only; no GPU lease is acquired")
    if m.get("package_sha256") != package_digest():
        raise ValueError("source package changed after registration")
    if not re.fullmatch(r"[0-9a-f]{40}", m.get("source_commit", "")):
        raise ValueError("exact Git commit required")
    if check_git:
        root = Path(__file__).resolve().parents[2]
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        if head != m["source_commit"] or dirty:
            raise ValueError("clean exact-source checkout required")
    base = m["base"]
    root = Path(base["directory"]).expanduser().resolve()
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    if not required <= set(base["files"]):
        raise ValueError("incomplete model/tokenizer file binding")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(base["files"]):
        raise ValueError("base directory has unbound or missing files")
    for name, sha in base["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError("base path traversal")
        checked_file({"path": str(p), "sha256": sha})
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("only dense, non-custom-code Qwen3 is eligible")
    if base["family"] == "Supa-1.7.0" and base["files"]["model.safetensors"] != SUPA_SHA256:
        raise ValueError("Supa source differs from historical source")
    if base["family"] not in {"Supa-1.7.0", "Titan-FP32", "Titan-Sparse-FP32", "Nimbus-1.6.0", "Halo-1.2.0"}:
        raise ValueError("unregistered family; routed/Supa-A1 custom decoding is excluded")
    prefixes = m["scope_prefixes"]
    if not prefixes or len(prefixes) != len(set(prefixes)) or any(not isinstance(p, str) or not p for p in prefixes):
        raise ValueError("explicit unique scope prefixes required")
    data = m["data"]
    probe = m["qualification_probe"]
    if (probe["status"] != "developmental" or probe["human_reviewed"] is not True
            or probe["tokenizer_sha256"] != base["files"]["tokenizer.json"]):
        raise ValueError("qualification probe must be reviewed and tokenizer-bound")
    checked_file(probe)
    if data["status"] != "developmental" or data["human_reviewed"] is not True:
        raise ValueError("no spent/sealed/unreviewed bank is eligible")
    if data["tokenizer_sha256"] != base["files"]["tokenizer.json"]:
        raise ValueError("tokenized bank belongs to a different tokenizer")
    checked_file(data)
    checked_file(m["forbidden_parents"])
    if m["action"] == "diagnostic" and not m.get("historical_endpoints"):
        raise ValueError("diagnostic requires a bound FP32 trained endpoint")
    for endpoint in m.get("historical_endpoints", []):
        if endpoint["representation"] != "trained_partial_fp32":
            raise ValueError("endpoint is trained weights, not a ready-made delta")
        if endpoint["base_sha256"] != base["files"]["model.safetensors"]:
            raise ValueError("cross-anchor delta transfer is not authorized by this experiment")
        checked_file(endpoint)
        if endpoint.get("name") == "Titan-seed17" and endpoint["sha256"] != TITAN_PARTIAL_SHA256:
            raise ValueError("wrong original FP32 Titan endpoint")
    if len(m.get("historical_endpoints", [])) > 2:
        raise ValueError("at most two independent historical endpoints")
    limits = m["limits"]
    for k in ("max_wall_seconds", "max_logical_calls", "max_length", "max_rows", "max_optimizer_steps"):
        if type(limits[k]) is not int or limits[k] < (0 if k == "max_optimizer_steps" else 1):
            raise ValueError(f"invalid limit: {k}")
    if limits["max_length"] > 4096 or limits["max_rows"] > 4096:
        raise ValueError("first-session bound exceeded")
    if not math.isfinite(limits["min_available_ram_gib"]) or limits["min_available_ram_gib"] < 12:
        raise ValueError("12 GiB analytical free-RAM floor cannot be lowered")
    if m["action"] != "study" and limits["max_optimizer_steps"] != 0:
        raise ValueError("qualification/diagnostic cannot reserve optimizer steps")
    if check_runtime:
        for name in ("torch", "transformers", "safetensors", "psutil"):
            if importlib.metadata.version(name) != m["runtime"][name]:
                raise ValueError(f"runtime version mismatch: {name}")
    if m["action"] != "qualify":
        qp = checked_file(m["qualification_receipt"])
        q = json.loads(qp.read_text())
        if (q.get("status") != "QUALIFICATION_PASS" or
                q.get("qualification_identity") != qualification_identity(m) or
                q.get("quality_claim") is not False):
            raise ValueError("missing matching qualification receipt")
    return {"status": "PREFLIGHT_PASS_NOT_EXECUTED", "qualification_identity": qualification_identity(m),
            "vocab_size": config["vocab_size"], "model_loads": 0, "optimizer_steps": 0}
