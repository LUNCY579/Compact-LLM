"""Bounded catalogs, conservative verifiers and answer-unaware PRLC bridging."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .contracts import SPLITS, Task, canonical, digest, read_json, require

FINAL = re.compile(r"<final>(-?(?:0|[1-9][0-9]*))</final>\s*\Z")


def verify(task: Task, text: str, terminated: bool = True) -> tuple[bool, str]:
    if not terminated:
        return False, "not_terminated"
    if not isinstance(text, str) or not text.strip():
        return False, "empty"
    if task.verifier == "exact-v1":
        return (True, "endpoint_only") if text.strip() == task.answer.strip() else (False, "wrong_answer")
    matches = list(FINAL.finditer(text))
    if len(matches) != 1 or text.count("<final>") != 1 or text.count("</final>") != 1:
        return False, "invalid_structure"
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", task.answer):
        return False, "invalid_gold"
    return (True, "endpoint_only") if matches[0][1] == task.answer else (False, "wrong_answer")


def normalized(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def validate_catalog(tasks: list[Task], exclusions: dict | None = None) -> dict:
    require(bool(tasks), "empty catalog")
    keys, parents, prompts = set(), {}, {}
    banned = exclusions or {"parents": [], "normalized_prompt_sha256": []}
    require(set(banned) == {"parents", "normalized_prompt_sha256"}, "invalid exclusion registry")
    require(all(isinstance(v, list) and all(type(x) is str for x in v) for v in banned.values()),
            "invalid exclusion entries")
    for task in tasks:
        require(task.key not in keys, "duplicate task key")
        keys.add(task.key)
        text = normalized(task.prompt)
        for registry, key in ((parents, task.parent), (prompts, text)):
            require(key not in registry or registry[key] == task.split, "cross-role collision")
            registry[key] = task.split
        require(task.parent not in banned["parents"] and digest(text) not in banned["normalized_prompt_sha256"],
                "historical/confirmation exclusion collision")
        require(verify(task, task.solution)[0], "reference fails endpoint verifier: " + task.key)
    return {"rows": len(tasks), "parents": len(parents), "roles": dict(Counter(t.split for t in tasks)),
            "catalog_digest": digest([asdict(t) for t in tasks]),
            "exclusions_digest": digest(banned), "semantic_decontamination": False,
            "reference_validation": "endpoint-only; intermediate reasoning not certified"}


def load_catalog(path: str | Path, exclusions_path: str | Path | None = None) -> tuple[list[Task], dict]:
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) <= 10000, "catalog must be a bounded JSON array")
    tasks = [Task(**r) for r in rows]
    return tasks, validate_catalog(tasks, read_json(exclusions_path) if exclusions_path else None)


def require_training_roles(tasks: list[Task]) -> None:
    require(set(SPLITS) <= {t.split for t in tasks}, "all five disjoint roles must be enrolled")
    train_sources = {t.source for t in tasks if t.split == "retention_train"}
    eval_sources = {t.source for t in tasks if t.split == "retention_eval"}
    require(train_sources == eval_sources, "retention training/evaluation sources must match")


def accept_rewrite(task: Task, original, rewrite) -> tuple[bool, str]:
    require(task.split == "train", "rewrites may only supervise TRAIN")
    require(original.key == rewrite.key == task.key, "rewrite task identity mismatch")
    require(original.policy_version == rewrite.policy_version, "rewrite snapshot mismatch")
    if not original.terminated:
        return False, "original_truncated"
    if len(rewrite.response_ids) >= len(original.response_ids):
        return False, "not_shorter"
    return verify(task, rewrite.text, rewrite.terminated)


def from_prlc(task, solution: str, provenance: str) -> Task:
    """An explicit adapter; no automatic MiMo capability or split relabeling."""
    require(tuple(task.requirements) == ("text",) and task.verifier == "exact-v1",
            "only audited deterministic text PRLC tasks supported")
    require(task.answer is not None, "missing expected answer")
    return Task(task.key, task.source, task.revision, task.group, task.split, task.prompt,
                task.answer, solution, "exact-v1", provenance)


def to_prlc_prompt(problem):
    from experiments.progressive_rl_curriculum_v1.schema import Prompt
    return Prompt(problem.key, problem.text)


def demo_catalog(per_role: int = 4) -> list[Task]:
    require(type(per_role) is int and 1 <= per_role <= 100, "invalid fixture size")
    tasks = []
    for role_index, role in enumerate(SPLITS):
        for i in range(per_role):
            a, b = 10 * role_index + i + 1, 3
            value = (a + b) * 2
            tasks.append(Task(f"fixture:{role}:{i}", "owned-arithmetic", "rsd-fixture-v1",
                              f"owned-arithmetic:world-{role_index}-{i}", role,
                              f"Start with {a}. Add {b}, then multiply by 2. Show the calculation; "
                              "end with <final>integer</final>.", str(value),
                              f"({a}+{b})*2={value}. <final>{value}</final>",
                              "final-int-v1", "owned deterministic fixture; NOT a native study enrollment"))
    validate_catalog(tasks)
    return tasks


def write_catalog(path: str | Path, tasks: list[Task]) -> None:
    validate_catalog(tasks)
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(json.dumps([asdict(t) for t in tasks], indent=2, allow_nan=False) + "\n")
