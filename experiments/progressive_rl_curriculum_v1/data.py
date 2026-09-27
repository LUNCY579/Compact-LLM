"""Offline source adapters, contamination checks and answer-blind policy inputs."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .schema import Task, digest, require


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


class Catalog:
    def __init__(self, tasks: Iterable[Task]):
        self.tasks = tuple(tasks)
        require(bool(self.tasks), "empty catalog")
        self.by_key = {t.key: t for t in self.tasks}
        require(len(self.by_key) == len(self.tasks), "duplicate source/task ID")
        groups: dict[str, str] = {}
        prompts: dict[str, str] = {}
        revisions: dict[str, str] = {}
        for task in self.tasks:
            require(revisions.setdefault(task.source, task.revision) == task.revision,
                    "multiple revisions for one source")
            # group is deliberately global: adapters must namespace independent parents.
            require(groups.setdefault(task.group, task.split) == task.split,
                    "parent group crosses splits")
            fingerprint = digest(normalized(task.prompt))
            require(prompts.setdefault(fingerprint, task.split) == task.split,
                    "normalized prompt crosses splits (including replay/evaluation)")
        self.fingerprint = digest(sorted((asdict(t) for t in self.tasks), key=lambda x: (x["source"], x["id"])))

    def subset(self, split: str, capabilities: frozenset[str]) -> list[Task]:
        return [t for t in self.tasks if t.split == split and set(t.requirements) <= capabilities]

    def verify(self, key: str, text: str, terminated: bool = True) -> float:
        task = self.by_key[key]
        require(task.verifier == "exact-v1", "external verifier requires a separately qualified runtime")
        # Intentionally strict; adding semantic/regex reward matching changes the experiment.
        return float(terminated and text.strip() == task.answer.strip())

    def census(self, capabilities: frozenset[str]) -> dict:
        result: dict = {"compatible": {}, "blocked": {}}
        for task in self.tasks:
            bucket = "compatible" if set(task.requirements) <= capabilities else "blocked"
            key = f"{task.source}/{task.split}/{task.verifier}"
            result[bucket][key] = result[bucket].get(key, 0) + 1
        return result


def load_catalog(path: Path) -> Catalog:
    require(path.stat().st_size <= 256 * 2**20, "catalog exceeds 256 MiB; prepare a bounded subset/shard")
    tasks = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                value["requirements"] = tuple(value.get("requirements", ("text",)))
                tasks.append(Task(**value))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return Catalog(tasks)


def write_catalog(catalog: Catalog, path: Path) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for task in catalog.tasks:
            stream.write(json.dumps(asdict(task), ensure_ascii=False, allow_nan=False) + "\n")


def import_mimo(rows: Iterable[dict], *, revision: str, split: str, domain: str,
                group_field: str) -> Catalog:
    """Import local JSONL exports only. Never pull images, run Docker, or use remote code.

    MiMo rows are environment specifications, NOT demonstrations. A source-supplied
    parent grouping field must be declared; guessing independent task IDs leaks forks.
    All rows stay blocked for the bundled text-only backend.
    """
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "pin an exact HF commit revision")
    tasks = []
    for row in rows:
        require(group_field in row and row[group_field] is not None, "missing declared parent group")
        messages = row.get("prompt")
        require(isinstance(messages, (str, list)), "MiMo prompt must be text or messages")
        if isinstance(messages, list):
            require(all(isinstance(m, dict) and m.get("role") in {"system", "user"}
                        and isinstance(m.get("content"), str) for m in messages),
                    "non-text or pre-completed MiMo prompt needs an explicit adapter")
            text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        else:
            text = messages
        key = str(row.get("id") or digest(row))
        tasks.append(Task(id=key, source="XiaomiMiMo/MiMo-V2.6-RL-oss", revision=revision,
                          group=f"mimo:{domain}:{row[group_field]}", split=split,
                          prompt=text, domain=domain, verifier=f"mimo-{domain}-external-v1",
                          requirements=("text", f"mimo:{domain}"),
                          metadata={"environment_spec": row, "spec_sha256": digest(row),
                                    "execution": "BLOCKED_UNQUALIFIED_ENVIRONMENT"}))
    return Catalog(tasks)


def verified_demonstrations(catalog: Catalog, rows: Iterable[dict]) -> list[tuple[Task, str]]:
    accepted = []
    seen = set()
    for row in rows:
        require(set(row) == {"task_key", "response", "teacher", "provenance"}, "invalid teacher row")
        task = catalog.by_key[row["task_key"]]
        require(task.split == "train", "bootstrap may only use training tasks")
        require(isinstance(row["teacher"], str) and bool(row["teacher"]), "teacher binding required")
        require(isinstance(row["provenance"], str) and bool(row["provenance"]), "teacher provenance required")
        require(isinstance(row["response"], str), "teacher response must be text")
        key = (task.key, row["response"])
        if key not in seen and catalog.verify(task.key, row["response"]):
            accepted.append((task, row["response"]))
            seen.add(key)
    require(bool(accepted), "no verifier-approved training demonstrations")
    return accepted
