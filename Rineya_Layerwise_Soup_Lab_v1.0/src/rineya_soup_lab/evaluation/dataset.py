from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..errors import ConfigurationError
from ..utils import canonical_json, read_jsonl, sha256_file, write_json


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    messages = [dict(message) for message in record.get("messages", [])]
    reference = record.get("reference")
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        messages = messages[:-1]
    metadata = record.get("metadata") or {}
    normalized["messages"] = messages
    normalized["reference"] = reference
    normalized["category"] = str(
        record.get("category") or metadata.get("reply_class") or "uncategorized"
    )
    if not isinstance(record.get("expected"), dict):
        normalized["expected"] = {
            "min_tokens": int(metadata.get("target_token_min", 1)),
            "max_tokens": int(metadata.get("target_token_max", 32)),
        }
    if not normalized.get("id"):
        payload = {"messages": messages, "reference": reference, "category": normalized["category"]}
        normalized["id"] = "auto_" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]
    return normalized


def load_dataset(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [normalize_record(row) for row in read_jsonl(path)]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ConfigurationError(f"Dataset contains no usable rows: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
        raise ConfigurationError(f"Dataset contains duplicate IDs: {duplicates[:10]}")
    for row in rows:
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise ConfigurationError(f"Dataset row {row['id']} has no messages.")
    return rows


def prompt_digest(record: Mapping[str, Any]) -> str:
    messages = [message for message in record.get("messages", []) if message.get("role") != "assistant"]
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()


def audit_splits(
    dev_path: Path,
    test_path: Path | None,
    *,
    allow_id_overlap: bool,
    allow_prompt_overlap: bool,
) -> dict[str, Any]:
    dev = load_dataset(dev_path)
    test = load_dataset(test_path) if test_path else []
    dev_ids = {str(row["id"]) for row in dev}
    test_ids = {str(row["id"]) for row in test}
    id_overlap = sorted(dev_ids & test_ids)
    dev_prompts = {prompt_digest(row): str(row["id"]) for row in dev}
    test_prompts = {prompt_digest(row): str(row["id"]) for row in test}
    prompt_overlap = sorted(set(dev_prompts) & set(test_prompts))
    errors: list[str] = []
    if id_overlap and not allow_id_overlap:
        errors.append(f"Development/test ID overlap: {id_overlap[:10]}")
    if prompt_overlap and not allow_prompt_overlap:
        examples = [(dev_prompts[value], test_prompts[value]) for value in prompt_overlap[:10]]
        errors.append(f"Development/test prompt overlap: {examples}")
    return {
        "valid": not errors,
        "errors": errors,
        "dev": {"path": str(dev_path), "sha256": sha256_file(dev_path), "case_count": len(dev)},
        "test": (
            {"path": str(test_path), "sha256": sha256_file(test_path), "case_count": len(test)}
            if test_path
            else None
        ),
        "id_overlap_count": len(id_overlap),
        "prompt_overlap_count": len(prompt_overlap),
        "id_overlap": id_overlap[:100],
        "prompt_overlap": prompt_overlap[:100],
    }


def require_split_audit(report: Mapping[str, Any]) -> None:
    if not report.get("valid"):
        raise ConfigurationError("Dataset split audit failed:\n- " + "\n- ".join(report.get("errors") or []))


def write_split_audit(path: Path, report: Mapping[str, Any]) -> None:
    write_json(path, dict(report))
