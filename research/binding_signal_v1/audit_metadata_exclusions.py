"""Audit candidate against approved exclusion metadata; never open sealed rows.

This records literal term contact and metadata-only ID checks. It cannot clear
the complete exclusion gate: the LC stopword algorithm and reserved row hashes
are unavailable. It never filters or rewrites the candidate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
CANDIDATE = RECORDS / "binding-signal-prep-20260920/candidate-v1"
MANIFEST = RECORDS / "lcr-exclusion-manifest/LCR_EXCLUSION_MANIFEST.json"
MANIFEST_SHA = "54b02b35c37b1cc842dbf413aac56980d3d3ade203f3ea5092bb332416914858"
QUARANTINE = BASE / "ChatBubble-retention-pilot-20260917/research/reasoning_posttraining_v1/QUARANTINE.json"
SEALED_METADATA = (
    RECORDS / "rci001-20260918/RCI_SEAL.json",
    RECORDS / "rci002-20260918/RCI_SEAL.json",
    RECORDS / "hskill-views-001/QUARANTINE.json",
    RECORDS / "hskill-views-002/QUARANTINE.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    # For literal term matching only, NOT the manifest's unspecified 8-gram rule.
    return " ".join(re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).split())


def contains_term(text, term):
    normalized = normalize(term)
    return bool(normalized) and " " + normalized + " " in " " + normalize(text) + " "


def term_audit(rows, manifest):
    fields = ("forbidden_proper_nouns", "forbidden_titles", "forbidden_domains", "forbidden_task_type_names")
    contacts = {field: [] for field in fields}
    for field in fields:
        for term in manifest[field]:
            ids = [row["id"] for row in rows if any(
                contains_term(message["content"], term)
                for message in row["messages"] if message["role"] == "user")]
            if ids:
                contacts[field].append({"term": term, "row_count": len(ids), "row_ids": ids})
    people = {normalize(person) for row in rows for person in row["world"]["people"]}
    forbidden = {normalize(term) for term in manifest["forbidden_proper_nouns"]}
    return {
        "status": "BLOCKED_LITERAL_CONTACT" if any(contacts.values()) else "NO_LITERAL_CONTACT_ONLY",
        "literal_contacts": contacts,
        "exact_full_person_name_overlap": sorted(people & forbidden),
        "interpretation": "All listed terms are reported, including common single words; no automatic semantic adjudication. Literal domain absence is not proof of semantic domain exclusion.",
        "eight_gram_check": "UNVERIFIED_STOPWORD_LIST_AND_IMPLEMENTATION_UNAVAILABLE",
        "eight_gram_collision_count": None,
    }


def identifier_audit(rows, catalogue):
    candidate_ids = {str(row[key]) for row in rows for key in ("id", "parent_id", "base_world_id") if row.get(key)}
    reports = []
    for entry in catalogue["entries"]:
        ids = entry.get("ids")
        reports.append({
            "key": entry["key"], "bars": entry["bars"],
            "metadata_id_count": len(ids) if isinstance(ids, list) else None,
            "metadata_id_set_sha256": entry.get("id_set_sha256"),
            "overlap": sorted(candidate_ids & set(map(str, ids))) if isinstance(ids, list) else None,
            "status": ("BLOCKED_ID_CONTACT" if candidate_ids & set(map(str, ids)) else "NO_ID_CONTACT_ONLY") if isinstance(ids, list) else "UNVERIFIED_NO_APPROVED_ROW_INDEX",
            "raw_dataset_opened": False,
        })
    return reports


def main():
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Approved manifest identity changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if any(manifest[key] is not False for key in ("contains_answer_key", "contains_answers", "contains_questions")):
        raise ValueError("Not an approved content-free exclusion aid")
    build = json.loads((CANDIDATE / "BUILD.json").read_text(encoding="utf-8"))
    by_split = {}
    for split in ("TRAIN", "DEV"):
        path = CANDIDATE / (split + ".jsonl")
        if sha(path) != build["files_sha256"][path.name]:
            raise ValueError("Frozen candidate identity changed: " + path.name)
        by_split[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    catalogue = json.loads(QUARANTINE.read_text(encoding="utf-8"))
    # These files are metadata only. Do not follow paths inside them to raw banks.
    metadata_receipts = [{"path": str(path), "sha256": sha(path), "status": "UNVERIFIED_NO_APPROVED_ROW_INDEX"} for path in SEALED_METADATA]
    record = {
        "schema": 1, "record": "BINDING_SIGNAL_APPROVED_METADATA_EXCLUSION_AUDIT",
        "status": "BLOCKED_EXTERNAL_EXCLUSIONS", "training_permission": False,
        "candidate_files_sha256": build["files_sha256"],
        "build_sha256": sha(CANDIDATE / "BUILD.json"), "audit_code_sha256": sha(Path(__file__)),
        "manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "source_sha256_from_metadata": manifest["source_sha256"], "raw_source_opened": False},
        "literal_normalization": "NFKC, casefold, non-word/underscore to space, whitespace collapse; whole phrase token boundaries; no stopword removal",
        "long_context_by_split": {split: term_audit(rows, manifest) for split, rows in by_split.items()},
        "quarantine_metadata": {"path": str(QUARANTINE), "sha256": sha(QUARANTINE), "entries": identifier_audit(sum(by_split.values(), []), catalogue)},
        "additional_reserved_metadata": metadata_receipts,
        "raw_sealed_files_opened": 0, "model_calls": 0, "candidate_rows_modified": 0,
        "remaining_requirements": [
            "Preserve this candidate and report contacts; any replacement is an explicit new version.",
            "Obtain the approved LC stopword list and exact hashing implementation before claiming 8-gram exclusion.",
            "Obtain custodian-approved row/parent/entity/prompt exclusion indices for reserved and spent banks; whole-file hashes cannot prove row disjointness.",
            "Join this scope with the separate open-bank audit; neither grants training permission.",
        ],
    }
    out = CANDIDATE / "METADATA_EXCLUSION_AUDIT.json"
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(out), "status": record["status"], "literal_terms_by_split": {split: {key: len(value) for key, value in audit["literal_contacts"].items()} for split, audit in record["long_context_by_split"].items()}}))


if __name__ == "__main__":
    main()
