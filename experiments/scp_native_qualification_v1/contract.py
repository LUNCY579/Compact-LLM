"""Strict, model-free prospective contract. Valid JSON is not an execution permit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

PARENT = "3ad2769a9f8343a9697e1720fcdc9efa18c3845a"
BRANCH = "research/scp-native-qualification-boundary-v1"
EXPERIMENT = "SCP-001-NQ-001"
PACKAGE = Path(__file__).resolve().parent
NATIVE_BLOCK = "BLOCKED_NATIVE_INTEGRATION_AND_SEPARATE_HOST_APPROVAL"
WEIGHTS = {
    "source": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
    "partial": "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d",
    "subject": "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284",
}
SIDECARS = {
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SOURCE_CONFIG = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
RUNTIME = {"torch": "2.10.0", "transformers": "4.57.1", "safetensors": "0.7.0"}
CLOSED = ("training", "backward", "optimizer", "teacher", "scoring", "confirmation",
          "release", "cloud", "protected_reads", "model_selection")
LIMITS = {
    "total_owned_work_wall_seconds": 900,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 620,
    "processed_training_tokens_all_attempts": 0,
    # This successor defines this inherited journal unit as ACTUAL/WORST-CASE FORWARDS.
    "evaluation_and_qualification_model_calls": 934,
    "generated_tokens_all_candidates": 928,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 64 * 2**20,
    "adaptive_model_selection_rounds": 0,
    "model_worker_launches": 1,
    "retries": 0,
    "cloud_or_teacher_spend": 0,
}
PROBE = {"id": "scp-nq-wire-001", "world_id": "scp-nq-wire-world-001",
         "system": "This is an unscored interface fixture. Follow the requested output format.",
         "user": "The label written on the box is LUMEN. Return that label."}


class Refusal(ValueError):
    """A failed closed-boundary condition; never retry automatically."""


def require(condition, message):
    if not condition:
        raise Refusal(message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise Refusal("value is not canonical finite JSON") from exc


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise Refusal("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def hash_string(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def safe_path(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "symlink or parent traversal refused")
    return path


def file_hash(path):
    path = safe_path(path)
    require(path.is_file(), "missing regular file: " + str(path))
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "asset changed during hashing")
    return h.hexdigest()


def write_new(path, value):
    path = safe_path(path)
    payload = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def validate_probe(probe):
    require(type(probe) is dict and probe == PROBE, "only the fixed label-free, unscored wiring probe is admitted")


def draft():
    return {"schema": "SCP_NQ_REQUEST_V1", "experiment": EXPERIMENT, "decision": "DRAFT_NOT_APPROVED",
            "parent_commit": PARENT, "branch": BRANCH, "successor_commit": None,
            "protocol_sha256": None, "asset_report_sha256": None, "host_sha256": None,
            "probe_sha256": digest(PROBE), "weight_sha256": dict(WEIGHTS),
            "runtime": dict(RUNTIME), "permissions": {k: False for k in CLOSED},
            "limits": dict(LIMITS), "generation_requests": 12, "qualification_forwards": 6,
            "window_tokens": 2048, "watchdog_seconds": 600, "max_process_rss_bytes": None,
            "evidence_root": None, "approved_by": None, "authorization_reference": None,
            "native_execution_permitted": False}


def validate_request(value):
    """Validate a *prospective* complete request. Never consume or issue authority."""
    require(type(value) is dict and set(value) == set(draft()), "request keys differ")
    for key in ("schema", "experiment", "parent_commit", "branch", "probe_sha256", "weight_sha256",
                "runtime", "limits", "generation_requests", "qualification_forwards", "window_tokens",
                "watchdog_seconds"):
        require(canonical(value[key]) == canonical(draft()[key]), "request contract differs: " + key)
    require(value["decision"] in ("DRAFT_NOT_APPROVED", "APPROVED_INFERENCE_ONLY"), "old or unknown grant")
    require(type(value["permissions"]) is dict and set(value["permissions"]) == set(CLOSED)
            and all(v is False for v in value["permissions"].values()), "closed permission widened")
    require(value["native_execution_permitted"] is False, "this release cannot enable native dispatch")
    require(hash_string(value["successor_commit"], 40) and value["successor_commit"] != PARENT,
            "exact successor commit missing")
    for key in ("protocol_sha256", "asset_report_sha256", "host_sha256"):
        require(hash_string(value[key]), "exact binding missing: " + key)
    require(type(value["max_process_rss_bytes"]) is int and value["max_process_rss_bytes"] > 0,
            "finite positive process RSS ceiling missing")
    require(type(value["evidence_root"]) is str and Path(value["evidence_root"]).is_absolute(),
            "absolute external evidence path missing")
    safe_path(value["evidence_root"])
    if value["decision"] == "APPROVED_INFERENCE_ONLY":
        for key in ("approved_by", "authorization_reference"):
            require(type(value[key]) is str and bool(value[key].strip()), "explicit approval reference missing")
    else:
        require(value["approved_by"] is None and value["authorization_reference"] is None,
                "draft must not impersonate approval")
    return {"classification": "PROSPECTIVE_CONTRACT_ONLY", "request_sha256": digest(value),
            "native_execution_permitted": False, "permit_consumed": False}


def refuse_native(*args, **kwargs):
    # Unconditional and before filesystem/model imports. No JSON/env flag overrides it.
    raise Refusal(NATIVE_BLOCK)
