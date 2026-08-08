from __future__ import annotations

"""Strict model-free validation for repository-native Phase 9 RAG corpora.

Supports the v1.7.3 530-row corpus and the v1.7.5 large corrective corpus.
The validator checks hashes, authored split isolation, answerability ontology,
review provenance, safe abstentions, document/procedure ownership, and duplicate
question leakage without assuming one fixed corpus size.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ABSTENTION_TEXT = "I don't know based on the available information."
V173_REPAIRS = {
    "train_procedure_semver_family_04_conflicting_inputs": "Version 2.10.0.",
    "train_procedure_iso_date_family_04_conflicting_inputs": "2032-11-02.",
}
ACCEPTED_STATUSES = {"accepted", "approved", "auto_verified"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def referenced_ids(row: dict[str, Any]) -> set[str]:
    ids = {str(v) for v in [*(row.get("support_document_ids") or []), *(row.get("retrieved_document_ids") or []), *(row.get("retrieved_procedure_ids") or [])] if v}
    for item in row.get("evidence") or []:
        value = item.get("document_id") or item.get("record_id")
        if value: ids.add(str(value))
    for item in row.get("procedures") or []:
        value = item.get("procedure_id") or item.get("record_id")
        if value: ids.add(str(value))
    return ids


def norm_question(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").casefold()).split())


def validate(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifests/dataset_manifest.json")
    hashes = read_json(root / "manifests/file_hashes.json")
    problems: list[str] = []
    schema = str(manifest.get("schema_version") or "")
    if schema not in {"2.1", "3.0"}:
        problems.append(f"unsupported dataset schema_version: {schema}")

    for relative, expected_raw in hashes.items():
        path = root / relative
        expected = expected_raw.get("sha256") if isinstance(expected_raw, dict) else expected_raw
        if not path.exists():
            problems.append(f"missing hashed file: {relative}")
        elif sha256(path) != expected:
            problems.append(f"hash mismatch: {relative}")

    candidates = [root / "teacher_candidates.reviewed.jsonl", root / "teacher_candidates.auto_verified.jsonl", root / "prompts.all.jsonl"]
    canonical_path = next((p for p in candidates if p.exists()), None)
    if canonical_path is None:
        return {"status":"FAIL","root":str(root),"problems":["canonical Phase 9 JSONL is missing"]}
    reviewed = read_jsonl(canonical_path)
    expected_total = int(manifest.get("canonical_rows") or manifest.get("reviewed_rows") or len(reviewed))
    if len(reviewed) != expected_total:
        problems.append(f"expected {expected_total} canonical rows, found {len(reviewed)}")
    if any(str(r.get("review_status") or "").casefold() not in ACCEPTED_STATUSES for r in reviewed):
        problems.append("all canonical rows must be accepted/approved/auto_verified")
    if any(str(r.get("decision") or "").casefold() == "procedure" for r in reviewed):
        problems.append("decision=procedure is forbidden; procedure is a mode, not answerability")
    for row in reviewed:
        if str(row.get("decision") or "").casefold() == "abstain":
            if str(row.get("approved_answer") or row.get("answer") or "") != ABSTENTION_TEXT:
                problems.append(f"{row.get('id')}: noncanonical abstention")
        if str(row.get("mode") or "").casefold() == "procedure" and not (row.get("procedures") or row.get("retrieved_procedure_ids")):
            problems.append(f"{row.get('id')}: procedure row lacks a trusted procedure")

    if schema == "2.1":
        by_id = {str(r.get("id")): r for r in reviewed}
        for rid, expected in V173_REPAIRS.items():
            row = by_id.get(rid)
            if not row or row.get("decision") != "answer" or str(row.get("approved_answer") or row.get("answer") or "") != expected:
                problems.append(f"{rid}: v1.7.3 procedure-resolution repair is not canonical")

    expected_counts = manifest.get("split_counts") or {}
    topic_owner: dict[str, str] = {}
    record_owner: dict[str, str] = {}
    split_ids: dict[str, set[str]] = {}
    question_owner: dict[str, str] = {}
    by_id = {str(r.get("id")): r for r in reviewed}
    for split in ("train", "dev", "test"):
        path = root / "reviewed_splits" / f"{split}.reviewed.jsonl"
        rows = read_jsonl(path)
        split_ids[split] = {str(r.get("id")) for r in rows}
        expected = int(expected_counts.get(split, len(rows)))
        if len(rows) != expected:
            problems.append(f"{split}: expected {expected} rows, found {len(rows)}")
        for row in rows:
            if str(row.get("intended_split")) != split:
                problems.append(f"{row.get('id')}: intended_split mismatch")
            topic = str(row.get("topic_id") or "")
            prior = topic_owner.setdefault(topic, split)
            if prior != split: problems.append(f"topic leakage: {topic} -> {prior}/{split}")
            q = norm_question(str(row.get("question") or ""))
            if q:
                qprior = question_owner.setdefault(q, split)
                if qprior != split: problems.append(f"exact normalized question leakage: {q[:80]} -> {qprior}/{split}")
            for record_id in referenced_ids(row):
                prior = record_owner.setdefault(record_id, split)
                if prior != split: problems.append(f"document/procedure leakage: {record_id} -> {prior}/{split}")
    if set.union(*split_ids.values()) != set(by_id):
        problems.append("authored splits do not exactly partition canonical rows")
    if any(split_ids[a] & split_ids[b] for a,b in (("train","dev"),("train","test"),("dev","test"))):
        problems.append("row ID leakage across authored splits")

    knowledge = read_jsonl(root / "knowledge.jsonl")
    procedures = read_jsonl(root / "procedures.jsonl")
    expected_knowledge = int(manifest.get("knowledge_documents") or len(knowledge))
    expected_procedures = int(manifest.get("procedures") or len(procedures))
    if len(knowledge) != expected_knowledge: problems.append(f"expected {expected_knowledge} knowledge documents, found {len(knowledge)}")
    if len(procedures) != expected_procedures: problems.append(f"expected {expected_procedures} procedures, found {len(procedures)}")
    known_ids = {str(x.get("document_id")) for x in knowledge} | {str(x.get("procedure_id")) for x in procedures}
    referenced = {x for r in reviewed for x in referenced_ids(r)}
    missing = sorted(referenced - known_ids)
    if missing: problems.append(f"canonical rows reference unknown records: {missing[:8]}")
    for row in knowledge:
        trust = str(row.get("trust_level") or "standard").casefold()
        if trust in {"superseded", "untrusted", "tentative"} and not isinstance(row.get("source_state"), dict):
            problems.append(f"{row.get('document_id')}: special-trust memory missing source_state")
        if trust == "superseded" and not isinstance(row.get("validity_conditions"), dict):
            problems.append(f"{row.get('document_id')}: superseded memory missing validity_conditions")

    return {
        "status":"PASS" if not problems else "FAIL", "root":str(root), "schema_version":schema,
        "canonical_rows":len(reviewed), "reviewed_rows":len(reviewed), "split_counts":{k:len(v) for k,v in split_ids.items()},
        "knowledge_documents":len(knowledge), "procedures":len(procedures), "problems":problems,
    }


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--corpus-root",type=Path,required=True); p.add_argument("--out",type=Path,default=None)
    args=p.parse_args(); packet=validate(args.corpus_root.resolve()); text=json.dumps(packet,indent=2); print(text)
    if args.out: args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(text+"\n",encoding="utf-8")
    if packet["status"]!="PASS": raise SystemExit(2)

if __name__=="__main__": main()
