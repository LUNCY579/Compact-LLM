"""Pure JSON projection and exact byte-bound rejoin; never retokenizes data."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import re

ARRAY_FIELDS = ("policy_rows", "oracle_rows_CPU_only")
PROJECTION_FIELDS = {"projection_kind", "projection_schema_version", "full_result_sha256",
                     "policy_file", "oracle_file"}
MAX_JSON_BYTES = 32 * 1024 ** 2


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _hash(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "exact SHA256")


def _ref(value):
    require(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
            and type(value["path"]) is str and Path(value["path"]).is_absolute()
            and type(value["bytes"]) is int and 0 < value["bytes"] <= MAX_JSON_BYTES, "exact bounded array file reference")
    _hash(value["sha256"])


def parse(raw):
    require(type(raw) is bytes and len(raw) <= MAX_JSON_BYTES, "bounded immutable JSON file bytes")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field"); result[key] = value
        return result
    def finite_float(text):
        value = float(text); require(math.isfinite(value), "nonfinite JSON float"); return value
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=finite_float,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def project(result, policy_file_ref, oracle_file_ref):
    require(type(result) is dict and not (set(result) & PROJECTION_FIELDS)
            and all(type(result.get(field)) is list for field in ARRAY_FIELDS)
            and result.get("sha256") == digest({k: v for k, v in result.items() if k != "sha256"}),
            "sealed full tokenizer result with both arrays required")
    _ref(policy_file_ref); _ref(oracle_file_ref)
    require(policy_file_ref["path"] != oracle_file_ref["path"], "distinct policy/oracle files")
    value = {k: deepcopy(v) for k, v in result.items() if k not in (*ARRAY_FIELDS, "sha256")}
    value.update(projection_kind="SYNTH_ARITHMETIC_TOKEN_PROJECTION001", projection_schema_version=1,
        full_result_sha256=result["sha256"], policy_file=deepcopy(policy_file_ref), oracle_file=deepcopy(oracle_file_ref))
    value["sha256"] = digest(value)
    return value


def _projection_header(projected):
    require(type(projected) is dict and PROJECTION_FIELDS <= set(projected)
            and not (set(projected) & set(ARRAY_FIELDS))
            and projected["projection_kind"] == "SYNTH_ARITHMETIC_TOKEN_PROJECTION001"
            and type(projected["projection_schema_version"]) is int and projected["projection_schema_version"] == 1
            and projected.get("sha256") == digest({k: v for k, v in projected.items() if k != "sha256"}),
            "exact self-sealed projected metadata")
    _hash(projected["full_result_sha256"])
    _ref(projected["policy_file"]); _ref(projected["oracle_file"])
    require(projected["policy_file"]["path"] != projected["oracle_file"]["path"], "distinct policy/oracle files")


def join(projected, policy_payload_bytes, oracle_payload_bytes):
    _projection_header(projected)
    payloads = []
    for ref_name, raw in (("policy_file", policy_payload_bytes), ("oracle_file", oracle_payload_bytes)):
        ref = projected[ref_name]; _ref(ref)
        require(type(raw) is bytes and len(raw) == ref["bytes"]
                and hashlib.sha256(raw).hexdigest() == ref["sha256"], "array payload bytes changed")
        payloads.append(parse(raw))
    policy, oracle = payloads
    require(type(policy) is dict and set(policy) == {"rows", "native_scoring_authorized"}
            and policy["native_scoring_authorized"] is False and type(policy["rows"]) is list,
            "exact prompt-only array envelope")
    require(type(oracle) is dict and set(oracle) == {"rows", "oracle_exposed_CPU_only"}
            and oracle["oracle_exposed_CPU_only"] is True and type(oracle["rows"]) is list,
            "exact CPU oracle array envelope")
    full = {k: deepcopy(v) for k, v in projected.items() if k not in PROJECTION_FIELDS | {"sha256"}}
    full.update(policy_rows=policy["rows"], oracle_rows_CPU_only=oracle["rows"])
    require(digest(full) == projected["full_result_sha256"], "original full tokenizer result checksum differs")
    full["sha256"] = projected["full_result_sha256"]
    return full


def read_join(projected):
    _projection_header(projected)
    def read(ref):
        path = Path(ref["path"])
        require(path.is_file() and not path.is_symlink(), "regular selected JSON array file")
        with path.open("rb") as stream:
            require(os.fstat(stream.fileno()).st_size == ref["bytes"], "physical array size changed before read")
            raw = stream.read(ref["bytes"] + 1)
            require(os.fstat(stream.fileno()).st_size == ref["bytes"], "physical array size changed after read")
        require(len(raw) == ref["bytes"], "physical array length changed")
        return raw
    return join(projected, read(projected["policy_file"]), read(projected["oracle_file"]))
