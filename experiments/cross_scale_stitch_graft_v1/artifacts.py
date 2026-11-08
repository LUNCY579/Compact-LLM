"""New-directory, hash-bound safetensors artifacts. Never pickle or overwrite."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from .alignment import AffineFit
from .contracts import Capture, Identity, RowKey, Site, require, sha256

ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path, *, max_bytes: int = 16777216):
    require(not path.is_symlink() and path.is_file(), "expected regular JSON file")
    require(path.stat().st_size <= max_bytes, "JSON exceeds budget")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def new_directory(path: Path) -> Path:
    path = path.absolute()
    resolved = path.resolve()
    require(resolved != ROOT and ROOT not in resolved.parents, "outputs must be outside experiment source")
    require(not path.exists() and not path.is_symlink(), "output already exists; never overwrite")
    path.mkdir(parents=False, exist_ok=False)
    return path


def save_capture(bank: Capture, path: Path) -> str:
    bank.validate()
    output = new_directory(path)
    payload = output / "values.safetensors"
    save_file({"values": bank.values.detach().contiguous()}, str(payload))
    write_json(output / "manifest.json", {"schema": "xsg-capture-v1", "contract": bank.contract,
               "payload_sha256": file_sha256(payload), "dtype": str(bank.values.dtype),
               "shape": list(bank.values.shape)})
    return file_sha256(output / "manifest.json")


def _manifest(path: Path, expected: str, schema: str) -> dict:
    sha256(expected)
    require(not path.is_symlink() and path.is_dir(), "artifact must be a directory, not symlink")
    metadata = path / "manifest.json"
    require(not metadata.is_symlink() and metadata.is_file(), "artifact incomplete or symlinked")
    require(metadata.stat().st_size <= 16777216, "manifest exceeds budget")
    require(file_sha256(metadata) == expected, "manifest digest mismatch")
    result = strict_json(metadata)
    require(result.get("schema") == schema, "artifact schema mismatch")
    return result


def _payload(path: Path, expected: str, max_bytes: int) -> dict[str, torch.Tensor]:
    sha256(expected)
    require(type(max_bytes) is int and max_bytes > 0, "positive payload budget required")
    require(path.is_file() and not path.is_symlink(), "missing/symlink payload")
    require(path.stat().st_size <= max_bytes, "payload exceeds budget")
    require(file_sha256(path) == expected, "payload digest mismatch")
    # safetensors validates offsets/shapes; no arbitrary-code deserialization.
    return load_file(str(path), device="cpu")


def load_capture(path: Path, expected_manifest_sha256: str, *, max_bytes: int = 536870912) -> Capture:
    m = _manifest(path, expected_manifest_sha256, "xsg-capture-v1")
    require(set(m) == {"schema", "contract", "payload_sha256", "dtype", "shape"}, "unexpected capture manifest fields")
    values = _payload(path / "values.safetensors", m["payload_sha256"], max_bytes)
    require(set(values) == {"values"}, "unexpected capture tensors")
    tensor = values["values"]
    require(str(tensor.dtype) == m["dtype"] and list(tensor.shape) == m["shape"], "payload shape/dtype drift")
    c = m["contract"]
    require(set(c) == {"identity", "site", "role", "registration_sha256", "rows"}, "unexpected capture contract fields")
    bank = Capture(Identity(**c["identity"]), Site(**c["site"]), c["role"], c["registration_sha256"],
                   tuple(RowKey(**row) for row in c["rows"]), tensor)
    bank.validate()
    return bank


def save_bridge(fit: AffineFit, report: dict, source: Capture, target: Capture, path: Path) -> str:
    require(fit.weight.shape == (target.site.width, source.site.width) and
            fit.bias.shape == (target.site.width,), "bridge geometry mismatch")
    require(torch.isfinite(fit.weight).all().item() and torch.isfinite(fit.bias).all().item(), "nonfinite bridge")
    output = new_directory(path)
    payload = output / "bridge.safetensors"
    save_file({"weight": fit.weight.contiguous(), "bias": fit.bias.contiguous()}, str(payload))
    write_json(output / "manifest.json", {"schema": "xsg-bridge-v1", "payload_sha256": file_sha256(payload),
               "source_identity": asdict(source.identity), "target_identity": asdict(target.identity),
               "source_site": asdict(source.site), "target_site": asdict(target.site),
               "registration_sha256": source.registration_sha256, "count": fit.count,
               "ridge_absolute": fit.ridge_absolute, "objective": fit.objective, "report": report,
               "native_qualification": False})
    return file_sha256(output / "manifest.json")


def load_bridge(path: Path, expected_manifest_sha256: str, *, source: Identity, target: Identity,
                source_site: Site, target_site: Site, registration_sha256: str,
                max_bytes: int = 134217728) -> AffineFit:
    m = _manifest(path, expected_manifest_sha256, "xsg-bridge-v1")
    for name, expected in (("source_identity", asdict(source)), ("target_identity", asdict(target)),
                           ("source_site", asdict(source_site)), ("target_site", asdict(target_site)),
                           ("registration_sha256", registration_sha256)):
        require(m.get(name) == expected, f"bridge {name} mismatch")
    values = _payload(path / "bridge.safetensors", m["payload_sha256"], max_bytes)
    require(set(values) == {"weight", "bias"}, "unexpected bridge tensors")
    require(values["weight"].shape == (target_site.width, source_site.width) and
            values["bias"].shape == (target_site.width,), "bridge shape mismatch")
    require(all(v.dtype == torch.float64 and torch.isfinite(v).all().item() for v in values.values()),
            "fit artifacts must contain finite FP64 tensors")
    require(type(m["count"]) is int and m["count"] >= 2 and m["ridge_absolute"] > 0 and
            m["objective"] == "centered_affine_ridge_sum_v1", "invalid fitting metadata")
    return AffineFit(values["weight"], values["bias"], m["count"], m["ridge_absolute"], m["objective"])


def preflight_fit_request(request_path: Path, spec: dict) -> dict:
    """Advisory instantaneous RAM guard before loading four local capture payloads.

    Cap total payload at 256 MiB in this bounded CLI. Caller APIs can be used for
    larger registered experiments only after their own allocation/host guards.
    """
    import psutil
    total = 0
    manifests = {}
    for name in ("source", "target", "check_source", "check_target"):
        item = spec[name]
        require(set(item) == {"directory", "manifest_sha256"}, "invalid capture reference")
        directory = request_path.parent / item["directory"]
        manifests[name] = _manifest(directory, item["manifest_sha256"], "xsg-capture-v1")
        payload = directory / "values.safetensors"
        require(payload.is_file() and not payload.is_symlink(), "invalid capture payload")
        total += payload.stat().st_size
    require(total <= 268435456, "combined capture payload exceeds 256 MiB CLI budget")
    d = manifests["source"]["contract"]["site"]["width"]
    o = manifests["target"]["contract"]["site"]["width"]
    require(type(d) is int and type(o) is int and 0 < d <= 4096 and 0 < o <= 4096, "unsupported CLI width")
    workspace = 8 * (6*d*d + 4*d*o + 4*d + 4*o)
    require(workspace <= 536870912, "ridge workspace exceeds 512 MiB CLI budget")
    needed = 3 * total + workspace + 268435456
    available = int(psutil.virtual_memory().available)
    limit_path, current_path = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if limit_path.exists() or current_path.exists():
        require(limit_path.is_file() and current_path.is_file(), "incomplete cgroup-v2 memory accounting")
        limit = limit_path.read_text().strip()
        if limit != "max":
            available = min(available, max(0, int(limit) - int(current_path.read_text())))
    require(available >= needed, "insufficient available host/cgroup RAM; do not bypass the guard")
    return {"capture_payload_bytes": total, "estimated_incremental_peak_and_margin_bytes": needed,
            "available_host_cgroup_bytes_at_preflight": available, "allocation_guaranteed": False}
