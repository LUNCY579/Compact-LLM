"""Explicit exposed-source exclusion. Never discover/read a protected bank."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .schema import canonical, digest
from .custody import REPO, BUSINESS, read_pinned

EXPOSED_T = {
    "experiments/composition_confirmation_v1/T_BANK_V3X.json": "173aa372466d5e3cc8efbebe4cc7e7ba09a8eb34de4598ce8f9d96a2b8e6557c",
    "experiments/composition_confirmation_v2/T_BANK_V4.json": "82f0a6423dd6b58b81319708e7fbda7c5880b9734742323ea11574ff354c93c0",
}
T_TRAIN = BUSINESS / "rineya-dataset-retention-records/rrp015-20260917/control-prepared/microfit-train.jsonl"
T_TRAIN_SHA = "0881c027d166f7cdd0286518aff20e5966f92120f4f3877cbca6855fbee2d9e1"
GEO_FREEZE_SHA = "f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687"


def arithmetic_key(family, payload):
    """Canonical touched role inputs AND outputs, compatible with the disclosed GEO key."""
    b, e = payload["before"], payload["event"]
    kind = e["kind"]
    if family in {"ledger", "ledger_replay"}:
        account = e["account"]
        a, v = b[account], e["amount"]
        if kind == "scale":
            out = a * v if e["status"] == "posted" else a
            key = ("ledger_replay", kind, e["status"], a, v, out)
        elif kind == "add":
            out = a + v if e["status"] == "posted" else a
            key = ("ledger_replay", kind, e["status"], a, v, out)
        elif kind == "transfer":
            other = b[e["other"]]
            key = ("ledger_replay", kind, e["status"], a, other, v,
                   a - v if e["status"] == "posted" else a,
                   other + v if e["status"] == "posted" else other)
        else:
            raise ValueError("STOP_UNSUPPORTED_LEDGER_EVENT")
    elif family in {"register", "register_machine"}:
        a = b[e["register"]]
        if kind == "add":
            key = ("register_machine", kind, a, e["value"], (a + e["value"]) % 97)
        elif kind == "conditional":
            other, threshold, v = b[e["other"]], e["threshold"], e["value"]
            branch = other > threshold
            key = ("register_machine", kind, a, other, threshold, v, branch,
                   (a + (v if branch else -v)) % 97)
        elif kind == "swap":
            other = b[e["other"]]
            key = ("register_machine", kind, a, other, other, a)
        elif kind == "mix":
            other = b[e["other"]]
            key = ("register_machine", kind, a, other, e["value"], (a + 2 * other + e["value"]) % 97)
        else:
            raise ValueError("STOP_UNSUPPORTED_REGISTER_EVENT")
    else:
        raise ValueError("STOP_UNSUPPORTED_COLLISION_FAMILY")
    return canonical(key)


def exposed_keys():
    keys, pins, counts = set(), {}, {}
    def add(source, family, payload):
        if family not in {"ledger_replay", "register_machine"}:
            return
        keys.add(arithmetic_key(family, payload))
        counts[source] = counts.get(source, 0) + 1
    for relative, pin in EXPOSED_T.items():
        data = read_pinned(REPO / relative, pin)
        pins[relative] = pin
        for row in json.loads(data)["rows"]:
            prompt = row["prompt"]
            add(relative, row["family"], json.loads(prompt[-1]["content"] if isinstance(prompt, list) else prompt))
    data = read_pinned(T_TRAIN, T_TRAIN_SHA)
    pins[str(T_TRAIN)] = T_TRAIN_SHA
    for line in data.decode("utf-8").splitlines():
        row = json.loads(line)
        user = next(m["content"] for m in row["prompt"] if m["role"] == "user")
        add("delta_T_training", row["family"], json.loads(user))
    # Read only the parent metadata in already-exposed frozen membership, never model outcomes.
    frozen = json.loads(read_pinned(REPO / "experiments/triskill_geometric_pilot_v1/FREEZE_SECTIONH.json", GEO_FREEZE_SHA))
    for filename in ("screen_sectionh.json", "development_sectionh.json", "screen.json", "development.json"):
        data = read_pinned(REPO / "experiments/triskill_geometric_pilot_v1" / filename, frozen["files"][filename])
        pins[filename] = frozen["files"][filename]
        for row in json.loads(data):
            if row["task_kind"] == "transition_parent":
                add(filename, row["metadata"]["transition_family"], json.loads(row["user"]))
    parent_path = REPO / "experiments/triskill_geo001_parentq_v1/parentq_bank.json"
    parent_data = parent_path.read_bytes()
    pins[str(parent_path)] = hashlib.sha256(parent_data).hexdigest()
    for row in json.loads(parent_data):
        if row["task_kind"] == "transition_parent":
            add("parentq_bank", row["metadata"]["transition_family"], json.loads(row["user"]))
    return keys, {"allowed_source_sha256": pins, "counts": counts, "unique_operative_keys": len(keys),
                  "scope": "LOCAL_AND_EXPOSED_ONLY", "global_pretraining_exclusion": False,
                  "old_sealed_banks_accessed": False}


def audit_partition(rows_by_split, forbidden=()):
    seen_ids, seen_worlds, seen_pairs, seen_keys = {}, {}, {}, {}
    errors = []
    forbidden = set(forbidden)
    for split, rows in rows_by_split.items():
        for row in rows:
            rid, world, key = row["row_id"], row["world_id"], row["operative_key"]
            if rid in seen_ids:
                errors.append(f"duplicate row:{rid}")
            seen_ids[rid] = split
            for value, store, name in ((world, seen_worlds, "world"), (row.get("pair_id"), seen_pairs, "pair"),
                                       (key, seen_keys, "operative")):
                if value is not None and value in store and store[value] != split:
                    errors.append(f"cross-split {name}:{value}")
                if value is not None:
                    store[value] = split
            if split in {"dev", "final"} and key in forbidden:
                errors.append(f"exposed arithmetic:{rid}")
    return {"passed": not errors, "errors": errors, "counts": {k: len(v) for k, v in rows_by_split.items()},
            "independent_worlds": {k: len({r['world_id'] for r in v}) for k, v in rows_by_split.items()},
            "scope": "LOCAL_EXPOSED_MEMBERSHIP_ONLY_NO_PROTECTED_HELDOUT_CLAIM"}
