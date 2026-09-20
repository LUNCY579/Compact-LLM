"""Pre-tokenized causal-LM contract. No implicit chat rendering or EOS insertion."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch


@dataclass(frozen=True)
class Row:
    id: str
    parent_id: str
    task: str
    split: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def supervised_tokens(self) -> int:
        return sum(x != -100 for x in self.labels[1:])

    def batch(self, device="cpu") -> tuple[dict, torch.Tensor]:
        ids = torch.tensor([self.input_ids], dtype=torch.long, device=device)
        labels = torch.tensor([self.labels], dtype=torch.long, device=device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels


def load_rows(path: str | Path, *, vocab_size: int, max_length: int,
              forbidden_parents: set[str] | None = None) -> list[Row]:
    rows, seen, parents, content_splits = [], set(), {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if set(raw) != {"id", "parent_id", "task", "split", "input_ids", "labels"}:
            raise ValueError("row schema mismatch")
        row = Row(**{**raw, "input_ids": tuple(raw["input_ids"]), "labels": tuple(raw["labels"])})
        if any(not isinstance(x, str) or not x for x in (row.id, row.parent_id, row.task)):
            raise ValueError("nonempty string identities required")
        if row.id in seen or row.split not in {"train", "dev", "test", "diagnostic"}:
            raise ValueError("duplicate ID or ineligible split")
        if len(row.input_ids) != len(row.labels) or not 2 <= len(row.input_ids) <= max_length:
            raise ValueError("length mismatch or context overflow; truncation is forbidden")
        if any(type(x) is not int or not 0 <= x < vocab_size for x in row.input_ids):
            raise ValueError("token outside vocabulary")
        if any(type(x) is not int or (x != -100 and x != y)
               for x, y in zip(row.labels, row.input_ids)):
            raise ValueError("labels must be masked input IDs")
        if row.labels[0] != -100 or row.supervised_tokens == 0:
            raise ValueError("missing shifted response supervision")
        if forbidden_parents and row.parent_id in forbidden_parents:
            raise ValueError("forbidden historical parent")
        if row.parent_id in parents and parents[row.parent_id] != row.split:
            raise ValueError("parent leakage across splits")
        content = hashlib.sha256(json.dumps(row.input_ids).encode()).hexdigest()
        if content in content_splits and content_splits[content] != row.split:
            raise ValueError("identical sequence leakage across splits")
        parents[row.parent_id], content_splits[content] = row.split, row.split
        seen.add(row.id); rows.append(row)
    if not rows:
        raise ValueError("empty bank")
    return rows


def require_design(rows: list[Row], tasks: tuple[str, str]) -> None:
    if len(tasks) != 2 or len(set(tasks)) != 2:
        raise ValueError("two distinct independent tasks required")
    for task in tasks:
        for split in ("train", "dev", "test"):
            if not any(r.task == task and r.split == split for r in rows):
                raise ValueError(f"missing {task}/{split}")
    # Explicitly disjoint row supports. Semantic skill overlap remains a limitation.
    if any(r.task not in tasks for r in rows if r.split == "train"):
        raise ValueError("unexpected training task; no hidden rehearsal")
