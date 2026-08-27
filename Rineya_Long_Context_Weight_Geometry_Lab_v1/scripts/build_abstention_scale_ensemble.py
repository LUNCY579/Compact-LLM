#!/usr/bin/env python3
"""Combine two scale predictions with an answer-blind UNKNOWN gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "id" not in row or "prediction" not in row:
            raise ValueError(f"{path}:{line_no}: expected id and prediction fields")
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate id {row_id}")
        if row.get("status", "ok") != "ok":
            raise ValueError(f"{path}:{line_no}: non-ok prediction for {row_id}")
        seen.add(row_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    return rows


def normalize(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", value)
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def build_ensemble(
    low_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    low_ids = [str(row["id"]) for row in low_rows]
    high_by_id = {str(row["id"]): row for row in high_rows}
    if set(low_ids) != set(high_by_id) or len(low_rows) != len(high_rows):
        raise ValueError("Low/high prediction files do not cover exactly the same ids")

    result: list[dict[str, Any]] = []
    abstained = 0
    for low in low_rows:
        row_id = str(low["id"])
        high = high_by_id[row_id]
        high_abstains = normalize(str(high["prediction"])) == "unknown"
        selected = high if high_abstains else low
        abstained += int(high_abstains)
        result.append(
            {
                "id": row_id,
                "status": "ok",
                "prediction": str(selected["prediction"]),
                "ensemble_method": "high_scale_unknown_else_low_scale",
                "selected_arm": "high" if high_abstains else "low",
                "low_prediction": str(low["prediction"]),
                "high_prediction": str(high["prediction"]),
                "low_adapter_scale": low.get("adapter_scale"),
                "high_adapter_scale": high.get("adapter_scale"),
            }
        )
    return result, abstained


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    low = args.low.resolve(strict=True)
    high = args.high.resolve(strict=True)
    output = args.output.resolve()
    rows, abstained = build_ensemble(load_jsonl(low), load_jsonl(high))
    atomic_write(
        output,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    manifest = {
        "schema_version": "1.0",
        "method": "high_scale_unknown_else_low_scale",
        "answer_key_accessed": False,
        "low": {"path": str(low), "sha256": sha256(low)},
        "high": {"path": str(high), "sha256": sha256(high)},
        "output": str(output),
        "rows": len(rows),
        "high_scale_abstentions": abstained,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "abstained": abstained}))


if __name__ == "__main__":
    main()
