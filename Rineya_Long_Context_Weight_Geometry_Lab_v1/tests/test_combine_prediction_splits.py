from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "combine_prediction_splits.py"
SPEC = importlib.util.spec_from_file_location("combine_prediction_splits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_disjoint_splits_are_combined_in_input_order(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write(first, [{"id": "a", "prediction": "one"}])
    write(second, [{"id": "b", "prediction": "two"}])

    assert [row["id"] for row in MODULE.combine([first, second])] == ["a", "b"]


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write(first, [{"id": "same", "prediction": "one"}])
    write(second, [{"id": "same", "prediction": "two"}])

    with pytest.raises(ValueError, match="Duplicate prediction id"):
        MODULE.combine([first, second])
