from __future__ import annotations

"""Durable append-only result storage used by long-running evaluations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from scripts.common.experiment_utils import append_jsonl_durable, read_jsonl_recover, write_jsonl


@dataclass
class DurableJsonlStore:
    """Persist completed examples immediately and resume by stable ID."""

    path: Path
    id_field: str = "id"
    rows_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    recovery_warnings: list[str] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path, *, id_field: str = "id", resume: bool = True) -> "DurableJsonlStore":
        store = cls(path=path, id_field=id_field)
        if resume:
            rows, warnings = read_jsonl_recover(path, id_field=id_field, repair_truncated_tail=True)
            store.rows_by_id = {str(row[id_field]): row for row in rows}
            store.recovery_warnings = warnings
        else:
            write_jsonl(path, [])
        return store

    def contains(self, stable_id: str) -> bool:
        return str(stable_id) in self.rows_by_id

    def append(self, row: dict[str, Any]) -> bool:
        stable_id = str(row[self.id_field])
        if stable_id in self.rows_by_id:
            return False
        append_jsonl_durable(self.path, row)
        self.rows_by_id[stable_id] = row
        return True

    def rows_in_order(self, stable_ids: Iterable[str]) -> list[dict[str, Any]]:
        return [self.rows_by_id[str(value)] for value in stable_ids if str(value) in self.rows_by_id]

    def finalize_order(self, stable_ids: Iterable[str]) -> list[dict[str, Any]]:
        rows = self.rows_in_order(stable_ids)
        write_jsonl(self.path, rows)
        self.rows_by_id = {str(row[self.id_field]): row for row in rows}
        return rows
