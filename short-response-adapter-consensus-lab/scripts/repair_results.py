from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.evaluation.summary import write_summary_files
from consensus_lab.io import read_jsonl, write_json_atomic
from consensus_lab.paths import LabPaths


KEY_FIELDS = ("dataset", "committee", "method", "example_id")
DETERMINISTIC_FIELDS = (
    "config_sha256",
    "evaluation_protocol_version",
    "prompt_hash",
    "output_token_ids",
    "output",
    "correct",
    "gold",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and remove duplicate durable result rows.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--archive-dir", type=Path, default=LAB_ROOT / "results/archive/concurrent_workers_raw"
    )
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)
    source = paths.assert_output_path(args.input)
    archive_dir = paths.assert_output_path(args.archive_dir)
    rows = read_jsonl(source)
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in KEY_FIELDS)
        grouped.setdefault(key, []).append(row)
    conflicts: list[dict[str, object]] = []
    for key, copies in grouped.items():
        first = copies[0]
        for other in copies[1:]:
            differing = [field for field in DETERMINISTIC_FIELDS if first.get(field) != other.get(field)]
            if differing:
                conflicts.append({"key": list(key), "differing_fields": differing})
    if conflicts:
        raise RuntimeError(f"Duplicate rows disagree on deterministic fields: {conflicts[:5]}")
    deduplicated = [copies[0] for copies in grouped.values()]
    deduplicated.sort(key=lambda row: tuple(str(row.get(field) or "") for field in KEY_FIELDS))
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / source.name
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite prior raw archive: {archive}")
    shutil.copy2(source, archive)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=source.name + ".", suffix=".tmp", dir=source.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in deduplicated:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, source)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    summary_stem = source.stem
    write_summary_files(
        deduplicated,
        LAB_ROOT / f"results/summaries/{summary_stem}.json",
        LAB_ROOT / f"results/summaries/{summary_stem}.csv",
    )
    audit = {
        "schema_version": "1.0",
        "source": str(source),
        "archive": str(archive),
        "rows_before": len(rows),
        "rows_after": len(deduplicated),
        "duplicates_removed": len(rows) - len(deduplicated),
        "duplicate_multiplicities": {
            str(key): value for key, value in Counter(len(group) for group in grouped.values()).items()
        },
        "deterministic_conflicts": conflicts,
    }
    audit_path = archive_dir / f"{source.stem}.repair.json"
    write_json_atomic(audit_path, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
