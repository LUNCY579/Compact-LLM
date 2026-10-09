"""Versioned, content-addressed inputs. No remote reads or inferred alignment."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

ROLES = {"prompt", "content", "format", "mixed"}
SPLITS = {"train", "development", "confirmation"}
EVIDENCE = {"synthetic_fixture", "tiny_random_model", "native_model"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load(path: str | Path) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f, parse_constant=invalid)


def write_new(path: str | Path, value: Any) -> None:
    """Publish a complete file atomically without replacing existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".rgi-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)  # atomic and refuses an existing destination
    finally:
        os.unlink(temporary)


def seal(value: dict) -> dict:
    require("content_sha256" not in value, "Already sealed")
    return {**value, "content_sha256": digest(value)}


def verify(value: dict) -> None:
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    require(value.get("content_sha256") == digest(body), "Content hash mismatch")


def validate_bank(bank: dict) -> None:
    verify(bank)
    require(bank.get("schema_version") == 1 and bank.get("kind") == "encoded_bank", "Wrong bank schema")
    require(checked_hash(bank.get("tokenizer_sha256")), "Missing tokenizer identity")
    require(bank.get("evidence_class") in EVIDENCE, "Missing evidence class")
    require(isinstance(bank.get("items"), list) and bool(bank["items"]), "Empty bank")
    ids, payload_splits, lineage_splits, parents = set(), {}, {}, {}
    for item in bank["items"]:
        for field in ("item_id", "parent_id", "lineage_id", "family", "operation", "rule_case", "suite", "format_id"):
            require(isinstance(item.get(field), str) and bool(item[field]), f"Missing {field}")
        require(item["item_id"] not in ids, "Duplicate item_id")
        ids.add(item["item_id"])
        require(item.get("split") in SPLITS, "Unknown split")
        require(type(item.get("horizon")) is int and item["horizon"] >= 1, "Invalid horizon")
        require(type(item.get("target_rule")) is bool, "target_rule must be explicit")
        require(checked_hash(item.get("payload_sha256")), "Missing semantic payload hash")
        token_ids, roles = item.get("input_ids"), item.get("roles")
        require(isinstance(token_ids, list) and len(token_ids) >= 2, "Need context and targets")
        require(all(type(x) is int and x >= 0 for x in token_ids), "Invalid token IDs")
        require(isinstance(roles, list) and len(roles) == len(token_ids), "Mask length mismatch")
        require(set(roles) <= ROLES and roles[0] == "prompt", "Invalid roles or missing context")
        first = next((i for i, r in enumerate(roles) if r != "prompt"), len(roles))
        require(first < len(roles) and "prompt" not in roles[first:], "Empty/discontinuous target mask")
        for mapping, key in ((payload_splits, item["payload_sha256"]), (lineage_splits, item["lineage_id"])):
            require(mapping.setdefault(key, item["split"]) == item["split"], "Cross-split semantic/lineage leakage")
        signature = (item["family"], item["split"], item["lineage_id"])
        require(parents.setdefault(item["parent_id"], signature) == signature, "Parent identity mismatch")


def assert_disjoint(*banks: dict) -> None:
    seen_payloads, seen_lineages = set(), set()
    for bank in banks:
        validate_bank(bank)
        payloads = {x["payload_sha256"] for x in bank["items"]}
        lineages = {x["lineage_id"] for x in bank["items"]}
        require(not (payloads & seen_payloads or lineages & seen_lineages), "Banks overlap semantically or by lineage")
        seen_payloads |= payloads
        seen_lineages |= lineages


def score_rows(bank: dict, losses: list[list[float]]) -> list[dict]:
    """Convert answer-only losses to fully identified teacher-forced token rows."""
    validate_bank(bank)
    require(len(losses) == len(bank["items"]), "Missing item predictions")
    rows = []
    for item, values in zip(bank["items"], losses):
        positions = [i for i, r in enumerate(item["roles"]) if r != "prompt"]
        require(len(positions) == len(values), "Missing target-token predictions")
        for pos, loss in zip(positions, values):
            require(math.isfinite(float(loss)) and float(loss) >= 0, "Invalid NLL")
            rows.append({**{k: item[k] for k in ("item_id", "parent_id", "lineage_id", "family", "operation",
                         "rule_case", "suite", "split", "horizon", "payload_sha256", "format_id")},
                         "position": pos, "target_id": item["input_ids"][pos], "role": item["roles"][pos],
                         "context_sha256": digest(item["input_ids"][:pos]), "nll": float(loss)})
    return rows


def artifact(bank: dict, rows: list[dict], model_id: str, host_sha: str,
             components: dict[str, float] | None = None, precision: str = "float32",
             execution_contract: dict | None = None) -> dict:
    validate_bank(bank)
    value = seal({"schema_version": 1, "kind": "loss_artifact", "completed": True,
                  "bank_sha256": bank["content_sha256"], "tokenizer_sha256": bank["tokenizer_sha256"],
                  "evidence_class": bank["evidence_class"], "model_id": model_id,
                  "host_sha256": host_sha, "components": components or {}, "precision": precision,
                  "scoring_mode": "teacher_forced", "context_policy": "shared_reference",
                  "execution_contract": execution_contract or {"collector": "external_or_fixture", "precision": "float32"},
                  "rows": rows, "expected_tokens": sum(sum(r != "prompt" for r in x["roles"]) for x in bank["items"]),
                  "quality_improvement_established": False})
    validate_artifact(value)
    return value


def validate_artifact(value: dict) -> None:
    verify(value)
    require(value.get("schema_version") == 1 and value.get("kind") == "loss_artifact", "Wrong artifact schema")
    require(value.get("completed") is True and value.get("rows"), "Incomplete or empty artifact")
    require(value.get("scoring_mode") == "teacher_forced" and value.get("context_policy") == "shared_reference",
            "Self-rollout losses are not aligned RGI observations")
    require(value.get("evidence_class") in EVIDENCE, "Unknown evidence class")
    require(isinstance(value.get("execution_contract"), dict) and value["execution_contract"], "Missing execution contract")
    require(value.get("precision") == "float32", "Only registered FP32 comparisons are supported")
    for key in ("bank_sha256", "tokenizer_sha256", "host_sha256"):
        require(checked_hash(value.get(key)), f"Invalid {key}")
    require(isinstance(value.get("components"), dict), "Missing composition provenance")
    for key, coef in value["components"].items():
        require(checked_hash(key) and isinstance(coef, (float, int)) and not isinstance(coef, bool)
                and math.isfinite(coef) and coef != 0, "Invalid adapter component")
    require(value.get("expected_tokens") == len(value["rows"]), "Incomplete token coverage")
    ids = set()
    parent_signatures = {}
    for row in value["rows"]:
        for field in ("item_id", "parent_id", "lineage_id", "family", "operation", "rule_case", "suite", "format_id"):
            require(isinstance(row.get(field), str) and bool(row[field]), f"Missing row {field}")
        signature = (row["family"], row["lineage_id"], row["split"])
        require(parent_signatures.setdefault(row["parent_id"], signature) == signature, "Conflicting parent metadata")
        key = (row["item_id"], row["position"])
        require(key not in ids, "Duplicate token prediction")
        ids.add(key)
        require(type(row["position"]) is int and row["position"] >= 1, "Invalid prediction position")
        require(type(row["target_id"]) is int and row["target_id"] >= 0, "Invalid target ID")
        require(row["role"] in ROLES - {"prompt"} and row["split"] in SPLITS, "Invalid row metadata")
        require(checked_hash(row["context_sha256"]), "Invalid context hash")
        require(isinstance(row["nll"], (float, int)) and not isinstance(row["nll"], bool)
                and math.isfinite(row["nll"]) and row["nll"] >= 0, "Invalid NLL")


def align(*artifacts: dict) -> tuple[list[dict], list[list[float]]]:
    """Exact key join; never silently intersect, truncate or trust row order."""
    require(len(artifacts) >= 2, "Need at least two models")
    for a in artifacts:
        validate_artifact(a)
    for key in ("bank_sha256", "tokenizer_sha256", "precision", "evidence_class", "scoring_mode", "context_policy", "execution_contract"):
        require(all(a[key] == artifacts[0][key] for a in artifacts), f"Alignment mismatch: {key}")
    maps = [{(r["item_id"], r["position"]): r for r in a["rows"]} for a in artifacts]
    keys = sorted(maps[0])
    require(all(set(m) == set(keys) for m in maps), "Missing or extra evaluation tokens")
    rows = [maps[0][k] for k in keys]
    for m in maps[1:]:
        for k in keys:
            require({f: v for f, v in m[k].items() if f != "nll"} ==
                    {f: v for f, v in maps[0][k].items() if f != "nll"}, "Token/context/stratum mismatch")
    return rows, [[m[k]["nll"] for k in keys] for m in maps]
