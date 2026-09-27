"""Frozen study plans, provenance, split isolation, and fail-closed admission."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata
from . import EXPERIMENT, PARENT_SHA
from .core import Arm, require, design

HASH = re.compile(r"[0-9a-f]{64}\Z")
DEFAULTS = dict(rank=16, protected_rank=16, shared_rank=16, agreement=0.75,
                private_gain=0.5, beta=0.0, iterations=6)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_arg(value, field="hash"):
    require(isinstance(value, str) and HASH.fullmatch(value) is not None, f"unbound {field}")
    return value


def read_json(path):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, f"duplicate JSON key: {k}")
            out[k] = v
        return out
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(result, dict), "expected JSON object")
    return result


def write_new(path, value):
    """Exclusive creation, never overwrite evidence. Partial writes are not valid JSON."""
    encoded = canonical(value) + b"\n"
    with Path(path).open("xb") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())


def source_digest():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def default_plan():
    return {"schema": 1, "experiment": EXPERIMENT, "parent_sha": PARENT_SHA,
            "donors": ["titan_sparse", "opallo"], "coefficients": [0.5, 0.5],
            "settings": dict(DEFAULTS),
            "tensors": ["model.layers.16.self_attn.q_proj.weight"],
            "unselected_policy": "frozen_base",
            "arms": [a.__dict__ for a in design()],
            "leave_one_out": {"full": "P1S1C1", "minus_protection": "P0S1C1",
                              "minus_shared_private": "P1S0C1", "minus_ctm": "P1S1C0"},
            "native_authorized": False, "training_authorized": False,
            "retention_margin": 0.0, "min_capability_gain": 0.0,
            "max_tensor_elements": 4_194_304}



def control_recipes(plan):
    """Fixed controls derived from the code-bound donor list and frozen doses."""
    donors, coefficients = plan["donors"], plan["coefficients"]
    recipes = {"base": {"kind": "endpoint", "source": "supa"}}
    for i, donor in enumerate(donors):
        unit = [0.0]*len(donors); unit[i] = 1.0
        dose = [0.0]*len(donors); dose[i] = coefficients[i]
        recipes["endpoint_"+donor] = {"kind": "endpoint", "source": donor}
        recipes["scoped_"+donor] = {"kind": "linear", "coefficients": unit}
        recipes["dose_"+donor] = {"kind": "linear", "coefficients": dose}
    recipes["delta_sum"] = {"kind": "linear", "coefficients": [1.0]*len(donors)}
    return recipes


def registered_names(plan):
    return {a.name for a in validate_plan(plan)} | set(control_recipes(plan))


def validate_plan(plan):
    require(set(plan) == set(default_plan()), "missing/unknown plan fields")
    require(plan["schema"] == 1 and plan["experiment"] == EXPERIMENT and plan["parent_sha"] == PARENT_SHA, "wrong plan identity")
    donors = plan["donors"]
    require(isinstance(donors, list) and 2 <= len(donors) <= 4 and len(set(donors)) == len(donors), "2-4 unique donors required")
    require(all(d in ("titan", "titan_sparse", "opallo") for d in donors), "v1 binds published common-base endpoints only")
    cs = plan["coefficients"]
    require(len(cs) == len(donors) and all(type(a) in (int, float) and 0 <= a <= 1 for a in cs), "invalid coefficient vector")
    require(abs(sum(cs)-1) < 1e-12, "primary dose coefficients must sum to one")
    require(set(plan["settings"]) == set(DEFAULTS), "settings mismatch")
    s = plan["settings"]
    for key in ("rank", "shared_rank", "protected_rank"):
        require(type(s[key]) is int and 0 <= s[key] <= 128, f"invalid {key}")
    for key in ("private_gain", "beta"):
        require(type(s[key]) in (int, float) and 0 <= s[key] <= 1, f"invalid {key}")
    require(type(s["agreement"]) in (int, float) and 0 < s["agreement"] <= 1, "invalid agreement")
    require(type(s["iterations"]) is int and 0 <= s["iterations"] <= 32, "invalid iterations")
    tensors = plan["tensors"]
    require(isinstance(tensors, list) and 0 < len(tensors) <= 196 and len(set(tensors)) == len(tensors), "unique explicit tensor list required")
    pattern = r"model\.layers\.(?:[0-9]|1[0-9]|2[0-7])\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:up|down|gate)_proj)\.weight"
    require(all(isinstance(t, str) and re.fullmatch(pattern, t) for t in tensors), "only explicit attention/MLP matrices supported")
    require(plan["unselected_policy"] == "frozen_base", "unselected policy must be frozen_base")
    require(type(plan["max_tensor_elements"]) is int and 0 < plan["max_tensor_elements"] <= 4_194_304, "workspace ceiling")
    for field in ("native_authorized", "training_authorized"):
        require(type(plan[field]) is bool, f"{field} must be boolean")
    for field in ("retention_margin", "min_capability_gain"):
        require(type(plan[field]) in (int, float) and 0 <= plan[field] <= 1, f"invalid {field}")
    arms = [Arm(**a).validate() for a in plan["arms"]]
    require(len(arms) == len({a.name for a in arms}) and arms, "duplicate/empty arms")
    require(len({(a.protect, a.share, a.compression, a.norm_match) for a in arms}) == len(arms), "duplicate mechanical cells")
    require(all(v in {a.name for a in arms} for v in plan["leave_one_out"].values()), "unknown ablation alias")
    canonical(plan)  # also rejects nonfinite values anywhere in the plan
    return arms


def audit_splits(splits):
    """Checks supplied ledgers, not proof that the supplied ancestry is complete."""
    require(set(splits) >= {"TRAIN", "CAL", "CHECK"}, "TRAIN/CAL/CHECK required")
    seen = {kind: {} for kind in ("id", "parent_id", "prompt")}
    counts = {}
    for split, rows in splits.items():
        require(isinstance(rows, list) and rows, f"empty {split} ledger")
        local_ids = set()
        for row in rows:
            for key in ("id", "parent_id", "prompt"):
                value = row.get(key)
                require(isinstance(value, str) and value.strip(), f"missing {key}")
                normalized = " ".join(unicodedata.normalize("NFKC", value).split()) if key == "prompt" else value
                identity = digest(normalized) if key == "prompt" else normalized
                previous = seen[key].get(identity)
                require(previous in (None, split), f"{key} leakage: {previous}/{split}")
                seen[key][identity] = split
            require(row["id"] not in local_ids, f"duplicate id within {split}")
            local_ids.add(row["id"])
        counts[split] = len(rows)
    return {"status": "SUPPLIED_LEDGERS_DISJOINT", "counts": counts,
            "hashes": {k: digest(v) for k, v in splits.items()}, "ancestry_completeness_verified": False}


def freeze(plan, artifact_hashes, *, cal_sha256, check_sha256, audit_sha256):
    arms = validate_plan(plan)
    require(set(artifact_hashes) <= registered_names(plan) and "base" in artifact_hashes and len(artifact_hashes) >= 2,
            "freeze requires the base and at least one registered candidate; unknown artifacts refused")
    for key, value in artifact_hashes.items():
        hash_arg(value, key)
    for key, value in (("cal", cal_sha256), ("check", check_sha256), ("audit", audit_sha256)):
        hash_arg(value, key)
    require(cal_sha256 != check_sha256, "CAL and CHECK identical")
    return {"schema": 1, "experiment": EXPERIMENT, "plan_sha256": digest(plan),
            "code_sha256": source_digest(), "artifacts": dict(artifact_hashes),
            "registered_but_not_screened": sorted(registered_names(plan)-set(artifact_hashes)),
            "cal_sha256": cal_sha256, "check_sha256": check_sha256,
            "audit_sha256": audit_sha256, "selection_split": "CAL", "check_tuning_permitted": False}


def admit(plan, grant, action, *, arm=None):
    """Must run before model/checkpoint I/O. Receipt is operator-attested, not signed."""
    arms = validate_plan(plan)
    require(action in ("preflight", "build", "collect", "train"), "unknown operation")
    require(grant.get("experiment") == EXPERIMENT and grant.get("action") == action, "wrong operation grant")
    require(grant.get("plan_sha256") == digest(plan), "grant not bound to plan")
    require(grant.get("code_sha256") == source_digest(), "grant not bound to code")
    require(grant.get("authorized") is True and grant.get("operator_reviewed") is True, "operator grant required")
    if action in ("collect", "train"):
        require(plan["native_authorized"] is True, "native execution remains disabled")
    if action == "train":
        require(plan["training_authorized"] is True, "training remains disabled")
    if action != "preflight":
        require(arm in registered_names(plan) and grant.get("arm") == arm, "arm-bound grant required")
    expiry = datetime.fromisoformat(grant.get("expires_utc", "").replace("Z", "+00:00"))
    require(expiry.tzinfo is not None and expiry > datetime.now(timezone.utc), "expired or naive grant")
    require(isinstance(grant.get("grant_id"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", grant["grant_id"]), "invalid grant id")
    require(grant.get("max_attempts") == 1, "single-attempt grant required")
    for key in ("lease_path", "attempt_directory", "output_parent"):
        require(isinstance(grant.get(key), str) and Path(grant[key]).is_absolute(), f"absolute {key} required")
    return grant


@contextmanager
def execution_lease(grant, *, memory_reader=None):
    """Shared host lease, two consecutive admission checks, consumed attempt.

    Never deletes someone else's lock, retries, lowers thresholds, or kills jobs.
    Failed attempts stay spent; lock release is limited to this process's inode.
    """
    if memory_reader is None:
        import psutil
        memory_reader = lambda: psutil.virtual_memory().available
    lock = Path(grant["lease_path"])
    attempts = Path(grant["attempt_directory"])
    require(lock.parent.is_dir() and attempts.is_dir(), "operator must provision shared lease/attempt paths")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    inode = os.fstat(fd).st_ino
    try:
        os.write(fd, canonical({"experiment": EXPERIMENT, "grant_id": grant["grant_id"], "pid": os.getpid()}))
        os.fsync(fd)
        write_new(attempts / (grant["grant_id"] + ".spent.json"), {"status": "ATTEMPT_CONSUMED", "action": grant["action"]})
        for i in range(2):
            require(memory_reader() >= 7.4 * 1024**3, "host admission requires two consecutive >=7.4 GiB readings")
            if i == 0:
                time.sleep(0.05)
        def worker_guard():
            require(memory_reader() >= 6.0 * 1024**3, "worker host RAM below 6.0 GiB")
        yield worker_guard
    finally:
        os.close(fd)
        if lock.exists() and lock.stat().st_ino == inode:
            lock.unlink()
