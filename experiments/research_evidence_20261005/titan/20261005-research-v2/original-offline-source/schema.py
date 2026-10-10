"""Shared offline quality packet interface; no model/framework imports."""
from __future__ import annotations
import hashlib
import json

EXPERIMENT = "TITAN-QUALITY-REPAIR-V1"
SCHEMA_VERSION = 1
REPAIR_FAMILIES = ("ledger", "register", "answerability", "grounded_qa")
RETENTION_FAMILIES = ("arithmetic", "instruction", "concise_grounded")
FAMILIES = REPAIR_FAMILIES + RETENTION_FAMILIES
STYLES = ("canonical", "native_a", "native_b", "natural")
SEEDS = (17, 29)
ABSTENTION = "I don't know based on the available information."
DEVELOPMENT_WORLDS_PER_FAMILY = 64
FINAL_WORLDS_PER_FAMILY = 512
TRAIN_EXAMPLES = 2000
REPLAY_EXAMPLES = 400
MAX_SEQUENCE_LENGTH = 2048
MAX_OUTPUT_TOKENS = 64


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def typed_equal(left, right):
    """JSON gold equality without Python's bool/int or float/int coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_row(row):
    required = {"schema_version", "row_id", "world_id", "pair_id", "split", "family", "variant_id",
                "system", "user", "answerable", "expected", "contract", "payload", "operative_key",
                "support_expansion", "strata"}
    if not isinstance(row, dict) or not required <= set(row):
        raise ValueError("STOP_INCOMPLETE_ROW_SCHEMA")
    if row["schema_version"] != SCHEMA_VERSION or row["family"] not in FAMILIES:
        raise ValueError("STOP_ROW_SCHEMA_OR_FAMILY")
    if row["split"] not in {"train", "dev", "final", "replay", "fixture"} or row["variant_id"] not in STYLES:
        raise ValueError("STOP_ROW_SPLIT_OR_VARIANT")
    if type(row["answerable"]) is not bool or type(row["support_expansion"]) is not bool:
        raise ValueError("STOP_ROW_BOOLEAN_TYPES")
    for field in ("row_id", "world_id", "system", "user", "operative_key"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"STOP_EMPTY_ROW_FIELD:{field}")
    if not isinstance(row["payload"], dict) or not isinstance(row["contract"], dict) or not isinstance(row["strata"], dict):
        raise ValueError("STOP_ROW_STRUCTURED_FIELDS")
    if row["contract"].get("kind") not in {"json_object", "integer", "text"}:
        raise ValueError("STOP_UNKNOWN_OUTPUT_CONTRACT")
    if row["pair_id"] is not None and not isinstance(row["pair_id"], str):
        raise ValueError("STOP_PAIR_ID_TYPE")
    if not row["answerable"] and row["expected"] is not None:
        raise ValueError("STOP_UNANSWERABLE_GOLD_MUST_BE_NULL")
    return row
