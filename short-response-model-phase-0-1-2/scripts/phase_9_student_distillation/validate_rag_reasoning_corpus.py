from __future__ import annotations

"""Validate the repository-native Phase 9 RAG Reasoning Corpus v2.1.

This validator is intentionally model-free. It verifies the invariants that must
hold before Phase 9, 9B, 10, or 11 may consume the corpus: hashes, row counts,
review status, authored split isolation, answerability ontology, canonical safe
abstentions, and the two auditable procedure-resolution repairs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ABSTENTION_TEXT = "I don't know based on the available information."
REPAIRS = {
    "train_procedure_semver_family_04_conflicting_inputs": "Version 2.10.0.",
    "train_procedure_iso_date_family_04_conflicting_inputs": "2032-11-02.",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            out.append(value)
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_ids(row: dict[str, Any]) -> set[str]:
    ids = {
        str(value)
        for value in [
            *(row.get("support_document_ids") or []),
            *(row.get("retrieved_document_ids") or []),
            *(row.get("retrieved_procedure_ids") or []),
        ]
        if value
    }
    for item in row.get("evidence") or []:
        value = item.get("document_id") or item.get("record_id")
        if value:
            ids.add(str(value))
    for item in row.get("procedures") or []:
        value = item.get("procedure_id") or item.get("record_id")
        if value:
            ids.add(str(value))
    return ids


def validate(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifests/dataset_manifest.json")
    hashes = read_json(root / "manifests/file_hashes.json")
    problems: list[str] = []

    if str(manifest.get("schema_version")) != "2.1":
        problems.append("dataset schema_version must be 2.1")
    expected_counts = {"train": 392, "dev": 67, "test": 71}
    if manifest.get("split_counts") != expected_counts:
        problems.append(f"unexpected split counts: {manifest.get('split_counts')}")

    for relative, expected in hashes.items():
        path = root / relative
        if not path.exists():
            problems.append(f"missing hashed file: {relative}")
        elif sha256(path) != expected:
            problems.append(f"hash mismatch: {relative}")

    reviewed = read_jsonl(root / "teacher_candidates.reviewed.jsonl")
    if len(reviewed) != 530:
        problems.append(f"expected 530 reviewed rows, found {len(reviewed)}")
    if any(str(row.get("review_status") or "").casefold() not in {"accepted", "approved"} for row in reviewed):
        problems.append("all canonical reviewed rows must be accepted/approved")
    if any(str(row.get("decision") or "").casefold() == "procedure" for row in reviewed):
        problems.append("decision=procedure is forbidden; procedure is a mode, not answerability")
    for row in reviewed:
        if str(row.get("decision") or "").casefold() == "abstain":
            if str(row.get("approved_answer") or row.get("answer") or "") != ABSTENTION_TEXT:
                problems.append(f"{row.get('id')}: noncanonical abstention")

    by_id = {str(row.get("id")): row for row in reviewed}
    for row_id, expected in REPAIRS.items():
        row = by_id.get(row_id)
        if not row:
            problems.append(f"missing repaired row: {row_id}")
            continue
        answer = str(row.get("approved_answer") or row.get("answer") or "")
        if row.get("decision") != "answer" or answer != expected:
            problems.append(f"{row_id}: procedure-resolution repair is not canonical")

    topic_owner: dict[str, str] = {}
    record_owner: dict[str, str] = {}
    split_ids: dict[str, set[str]] = {}
    for split, count in expected_counts.items():
        rows = read_jsonl(root / f"reviewed_splits/{split}.reviewed.jsonl")
        split_ids[split] = {str(row.get("id")) for row in rows}
        if len(rows) != count:
            problems.append(f"{split}: expected {count} rows, found {len(rows)}")
        for row in rows:
            if str(row.get("intended_split")) != split:
                problems.append(f"{row.get('id')}: intended_split mismatch")
            topic = str(row.get("topic_id") or "")
            prior = topic_owner.setdefault(topic, split)
            if prior != split:
                problems.append(f"topic leakage: {topic} -> {prior}/{split}")
            for record_id in referenced_ids(row):
                prior = record_owner.setdefault(record_id, split)
                if prior != split:
                    problems.append(f"document/procedure leakage: {record_id} -> {prior}/{split}")

    if (split_ids["train"] & split_ids["dev"]) or (split_ids["train"] & split_ids["test"]) or (split_ids["dev"] & split_ids["test"]):
        problems.append("row ID leakage across authored splits")
    if set.union(*split_ids.values()) != set(by_id):
        problems.append("authored splits do not exactly partition canonical reviewed rows")

    knowledge = read_jsonl(root / "knowledge.jsonl")
    if len(knowledge) != 464:
        problems.append(f"expected 464 knowledge documents, found {len(knowledge)}")
    for row in knowledge:
        trust = str(row.get("trust_level") or "standard").casefold()
        if trust in {"superseded", "untrusted", "tentative"} and not isinstance(row.get("source_state"), dict):
            problems.append(f"{row.get('document_id')}: special-trust memory missing source_state")
        if trust == "superseded" and not isinstance(row.get("validity_conditions"), dict):
            problems.append(f"{row.get('document_id')}: superseded memory missing validity_conditions")

    return {
        "status": "PASS" if not problems else "FAIL",
        "root": str(root),
        "reviewed_rows": len(reviewed),
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "knowledge_documents": len(knowledge),
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    packet = validate(args.corpus_root.resolve())
    text = json.dumps(packet, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if packet["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
