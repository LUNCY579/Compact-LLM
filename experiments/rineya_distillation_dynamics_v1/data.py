"""Public-only model inputs, parent isolation, and deterministic answer verification."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from .contracts import digest, read_json, require

DEV_ROLES = {"train", "retention_train", "teacher_dev", "dev_acquisition", "dev_retention", "dev_hard"}
CONFIRM_ROLES = {"confirm_acquisition", "confirm_retention", "confirm_hard"}


@dataclass(frozen=True)
class Problem:
    key: str
    prompt: str


@dataclass(frozen=True)
class Task:
    key: str
    parent: str
    source: str
    revision: str
    role: str
    prompt: str
    answer: str
    solution: str
    verifier: str

    def public(self) -> Problem:
        return Problem(self.key, self.prompt)


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def verify(task: Task, text: str) -> bool:
    require(type(text) is str, "non-text answer")
    if task.verifier == "exact-v1":
        return text.strip() == task.answer
    require(task.verifier == "final-int-v1", "unknown verifier")
    matches = re.findall(r"<final>(-?(?:0|[1-9][0-9]*))</final>", text)
    return (len(matches) == 1 and text.count("<final>") == 1 and text.count("</final>") == 1
            and text.rstrip().endswith(f"<final>{matches[0]}</final>") and matches[0] == task.answer)


def audit(tasks: list[Task], exclusions: dict | None = None, *, confirmation=False) -> dict:
    allowed = CONFIRM_ROLES if confirmation else DEV_ROLES
    require(0 < len(tasks) <= 10000, "catalog size")
    e = exclusions if exclusions is not None else {"parents": [], "prompt_hashes": []}
    require(set(e) == {"parents", "prompt_hashes"} and all(type(v) is list for v in e.values()),
            "exclusions schema")
    require(all(type(x) is str and x for v in e.values() for x in v), "invalid exclusion identity")
    require(all(re.fullmatch(r"[0-9a-f]{64}", x) for x in e["prompt_hashes"]), "bad exclusion hash")
    keys, parents, prompts, counts = set(), {}, {}, {}
    for t in tasks:
        require(type(t) is Task, "typed task required")
        require(all(type(v) is str for v in asdict(t).values()), "task fields must be strings")
        require(all(getattr(t, f).strip() for f in ("key", "parent", "source", "revision", "prompt", "answer")),
                "empty task identity or content")
        require(t.role in allowed, "role prohibited in this catalog")
        require(len(t.prompt) <= 200000 and len(t.solution) <= 100000, "task text bounds")
        require(t.key not in keys, "duplicate task key")
        keys.add(t.key)
        require(t.parent not in e["parents"], "excluded parent")
        ph = digest(normalized(t.prompt))
        require(ph not in e["prompt_hashes"], "excluded prompt")
        require(t.parent not in parents or parents[t.parent] == t.role, "parent crosses roles")
        require(ph not in prompts, "normalized prompt duplicate (even within role)")
        parents[t.parent], prompts[ph] = t.role, t.role
        counts[t.role] = counts.get(t.role, 0) + 1
        require(t.verifier in {"exact-v1", "final-int-v1"}, "unknown grader")
        if t.verifier == "final-int-v1":
            require(bool(re.fullmatch(r"-?(?:0|[1-9][0-9]*)", t.answer)) and t.answer != "-0", "canonical integer")
        if t.role in {"train", "retention_train"}:
            require(bool(t.solution) and verify(t, t.solution), "invalid training reference")
        else:
            require(t.solution == "", "held-out solutions must not enter catalog")
    return {"roles": counts, "parents": len(parents), "normalized_unique_prompts": len(prompts),
            "semantic_decontamination": False, "catalog_sha256": digest([asdict(t) for t in tasks])}


def load_catalog(path, exclusions=None, *, confirmation=False):
    raw = read_json(path)
    require(type(raw) is dict and set(raw) == {"schema", "provenance", "tasks"} and raw["schema"] == 1,
            "catalog schema")
    require(raw["provenance"] in {"reviewed_external", "mechanical_fixture"}, "explicit provenance label")
    require(type(raw["tasks"]) is list, "task list")
    tasks = []
    for row in raw["tasks"]:
        require(type(row) is dict and set(row) == set(Task.__dataclass_fields__), "exact task fields required")
        tasks.append(Task(**row))
    return tasks, raw["provenance"], audit(tasks, exclusions, confirmation=confirmation)


def cross_audit(dev: list[Task], confirm: list[Task]):
    require(not ({t.key for t in dev} & {t.key for t in confirm}), "confirmation key collision")
    require(not ({t.parent for t in dev} & {t.parent for t in confirm}), "confirmation parent collision")
    require(not ({normalized(t.prompt) for t in dev} & {normalized(t.prompt) for t in confirm}),
            "confirmation prompt collision")


def require_roles(tasks, replay: bool):
    needed = DEV_ROLES - (set() if replay else {"retention_train"})
    require(needed <= {t.role for t in tasks}, "required development/training role missing")
    if replay:
        train_sources = {t.source for t in tasks if t.role == "retention_train"}
        eval_sources = {t.source for t in tasks if t.role == "dev_retention"}
        require(train_sources == eval_sources, "retention sources must be represented on both disjoint roles")
