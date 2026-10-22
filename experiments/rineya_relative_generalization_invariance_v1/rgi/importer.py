"""Import explicitly normalized existing per-token evidence; never infer NLL from accuracy."""
from __future__ import annotations
import json
from .io import (align, artifact, file_hash, require, score_rows, validate_bank, verify, seal)


def import_losses(bank: dict, rows_path: str, receipt: dict) -> dict:
    validate_bank(bank)
    verify(receipt)
    require(receipt.get("kind") == "external_token_loss_receipt", "Missing producer receipt")
    require(receipt["records_sha256"] == file_hash(rows_path), "Source record bytes changed")
    require(receipt["bank_sha256"] == bank["content_sha256"], "Producer used a different bank")
    require(receipt["tokenizer_sha256"] == bank["tokenizer_sha256"], "Producer tokenizer mismatch")
    require(receipt["evidence_class"] == bank["evidence_class"], "Producer evidence class mismatch")
    require(receipt["scoring_mode"] == "teacher_forced" and receipt["context_policy"] == "shared_reference",
            "Only shared-reference teacher-forced losses can be imported")
    expected = score_rows(bank, [[0.0] * sum(r != "prompt" for r in x["roles"]) for x in bank["items"]])
    lookup = {(r["item_id"], r["position"]): r for r in expected}
    rows, seen = [], set()
    with open(rows_path, encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank record in source stream")
            source = json.loads(line)
            key = (source["item_id"], source["position"])
            require(key in lookup and key not in seen, "Unknown or duplicate source token")
            seen.add(key)
            row = lookup[key]
            for field in ("target_id", "context_sha256", "role"):
                require(source[field] == row[field], f"Source {field} does not match the frozen bank")
            rows.append({**row, "nll": source["nll"]})
    require(seen == lookup.keys(), "Missing source tokens; aggregate scores cannot be expanded to NLL")
    result = artifact(bank, rows, receipt["model_id"], receipt["host_sha256"], receipt["components"],
                      execution_contract=receipt.get("execution_contract"))
    canary = artifact(bank, expected, "coverage-canary", receipt["host_sha256"], receipt["components"],
                      execution_contract=receipt.get("execution_contract"))
    align(canary, result)
    return seal({**{k: v for k, v in result.items() if k != "content_sha256"},
                 "source_provenance": {"records_sha256": receipt["records_sha256"],
                                       "producer_receipt_sha256": receipt["content_sha256"]}})
