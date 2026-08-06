from __future__ import annotations

"""Build the low-memory SQLite FTS5 knowledge/procedure index for Phase 9."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl
from scripts.phase_9_student_distillation.phase9_common import (
    SQLiteFTSRetriever,
    validate_knowledge_record,
    write_index_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SQLite FTS5 index for Phase 9 RAG.")
    parser.add_argument("--knowledge", type=Path, action="append", default=[])
    parser.add_argument("--procedures", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "phase_9_student_distillation" / "knowledge.sqlite3",
    )
    parser.add_argument("--append", action="store_true", help="Append instead of rebuilding the database.")
    return parser.parse_args()


def load_records(paths: list[Path], *, kind: str) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            normalized = validate_knowledge_record(row, kind=kind)
            record_id = normalized["record_id"]
            if record_id in seen:
                raise ValueError(f"Duplicate {kind} ID across input files: {record_id}")
            seen.add(record_id)
            records.append(normalized)
    return records


def main() -> None:
    args = parse_args()
    if not args.knowledge and not args.procedures:
        raise ValueError("Provide at least one --knowledge or --procedures JSONL file.")
    knowledge = load_records(args.knowledge, kind="knowledge")
    procedures = load_records(args.procedures, kind="procedure")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with SQLiteFTSRetriever(args.output) as retriever:
        retriever.initialize(rebuild=not args.append)
        count = retriever.add_records([*knowledge, *procedures])
    manifest = write_index_manifest(
        output=args.output,
        knowledge_paths=args.knowledge,
        procedure_paths=args.procedures,
        record_count=count,
    )
    print(f"Indexed {count} records into {args.output}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
