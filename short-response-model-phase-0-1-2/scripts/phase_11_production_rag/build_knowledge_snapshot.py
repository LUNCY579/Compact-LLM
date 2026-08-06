from __future__ import annotations

"""Create an immutable, hashed production knowledge/procedure snapshot."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json, write_jsonl
from scripts.phase_9_student_distillation.phase9_common import (
    SQLiteFTSRetriever,
    stable_json_hash,
    validate_knowledge_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a versioned Phase 11 production knowledge snapshot.")
    parser.add_argument("--knowledge", type=Path, action="append", default=[])
    parser.add_argument("--procedures", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-name", default="production_snapshot")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load(paths: list[Path], *, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for raw in read_jsonl(path):
            record = validate_knowledge_record(raw, kind=kind)
            record_id = record["record_id"]
            fingerprint = stable_json_hash(record)
            prior = seen.get(record_id)
            if prior is not None and prior != fingerprint:
                raise ValueError(
                    f"Conflicting versions of {kind} record {record_id}; production snapshots require unique stable IDs"
                )
            if prior is None:
                seen[record_id] = fingerprint
                rows.append(record)
    return rows


def main() -> None:
    args = parse_args()
    if not args.knowledge and not args.procedures:
        raise ValueError("Provide at least one --knowledge or --procedures file")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Snapshot directory exists; use --overwrite intentionally: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    knowledge = load(args.knowledge, kind="knowledge")
    procedures = load(args.procedures, kind="procedure")
    knowledge_path = args.output_dir / "knowledge.jsonl"
    procedure_path = args.output_dir / "procedures.jsonl"
    index_path = args.output_dir / "knowledge.sqlite3"
    write_jsonl(knowledge_path, knowledge)
    write_jsonl(procedure_path, procedures)
    with SQLiteFTSRetriever(index_path) as retriever:
        retriever.initialize(rebuild=True)
        indexed = retriever.add_records([*knowledge, *procedures])

    manifest = {
        "schema_version": "1.0",
        "phase": "11_production_rag",
        "snapshot_name": args.snapshot_name,
        "source_knowledge_files": [str(path) for path in args.knowledge],
        "source_procedure_files": [str(path) for path in args.procedures],
        "counts": {
            "knowledge": len(knowledge),
            "procedures": len(procedures),
            "indexed": indexed,
        },
        "artifacts": {
            "knowledge": {"path": str(knowledge_path), "sha256": sha256(knowledge_path)},
            "procedures": {"path": str(procedure_path), "sha256": sha256(procedure_path)},
            "index": {"path": str(index_path), "bytes": index_path.stat().st_size},
        },
        "trust_policy": {
            "procedures": "must be curated/trusted; retrieved factual documents cannot redefine system behavior",
            "duplicate_ids": "identical duplicates are deduplicated; conflicting duplicates fail the snapshot",
        },
    }
    write_json(args.output_dir / "snapshot_manifest.json", manifest)
    print(args.output_dir / "snapshot_manifest.json")


if __name__ == "__main__":
    main()
