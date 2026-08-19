from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..utils import stable_int_hash


SUPPORTED_VIEWS = {"full_context", "split_stateful", "split_next_step"}


@dataclass(frozen=True)
class IndexedRow:
    offset: int
    problem_id: str
    transition_id: str | None
    step_index: int | None
    total_steps: int | None
    sort_key: tuple[Any, ...]


def select_messages(record: dict[str, Any], view: str) -> list[dict[str, str]]:
    if view == "full_context":
        messages = record.get("messages")
    elif view == "split_stateful":
        messages = record.get("stateful_rollout_messages")
    elif view == "split_next_step":
        messages = record.get("messages")
    else:
        raise ValueError(f"Unsupported training view {view!r}; expected one of {sorted(SUPPORTED_VIEWS)}")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"Record does not contain a valid {view} conversation")
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


def scan_jsonl_index(path: str | Path) -> list[IndexedRow]:
    file_path = Path(path)
    rows: list[IndexedRow] = []
    with file_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            pid = str(record["problem_id"])
            tid = str(record["transition_id"]) if record.get("transition_id") is not None else None
            step = int(record["step_index"]) if record.get("step_index") is not None else None
            total = int(record["total_steps"]) if record.get("total_steps") is not None else None
            rows.append(
                IndexedRow(
                    offset=offset,
                    problem_id=pid,
                    transition_id=tid,
                    step_index=step,
                    total_steps=total,
                    sort_key=(pid, step or 0, tid or ""),
                )
            )
    return rows


def choose_index_rows(
    rows: list[IndexedRow],
    *,
    seed: int,
    max_examples: int | None = None,
    one_per_problem: bool = False,
    preserve_trajectory_order: bool = False,
) -> list[IndexedRow]:
    selected = rows
    if one_per_problem:
        groups: dict[str, list[IndexedRow]] = {}
        for row in rows:
            groups.setdefault(row.problem_id, []).append(row)
        selected = []
        for pid, options in groups.items():
            ordered = sorted(options, key=lambda item: item.sort_key)
            index = stable_int_hash(f"{seed}:{pid}") % len(ordered)
            selected.append(ordered[index])
    if preserve_trajectory_order:
        selected = sorted(selected, key=lambda row: row.sort_key)
    else:
        selected = sorted(selected, key=lambda row: stable_int_hash(f"{seed}:{row.transition_id or row.problem_id}"))
    if max_examples is not None:
        selected = selected[: int(max_examples)]
    return selected


def read_at_offset(path: str | Path, offset: int) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    return json.loads(line)


def iter_selected_records(path: str | Path, rows: list[IndexedRow]) -> Iterator[dict[str, Any]]:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        for row in rows:
            handle.seek(row.offset)
            yield json.loads(handle.readline())
