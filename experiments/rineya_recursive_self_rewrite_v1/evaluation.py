"""H0-only panels, complete paired coverage, parent-cluster uncertainty."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .core import Backend, Ledger, Limits, Task, digest, require, validate_catalog
from .runtime import audit, execute


def evaluate_panel(tasks: list[Task], backends: dict[str, Backend], limits: Limits,
                   *, attempts=3, seed=37) -> dict:
    validate_catalog(tasks)
    require(all(t.split in {"calibration", "retention_eval"} for t in tasks), "evaluation roles")
    require(type(attempts) is int and attempts >= 1 and bool(backends), "evaluation coverage")
    records, ledgers = {}, {}
    for name, backend in backends.items():
        records[name], ledgers[name] = [], []
        for task in tasks:
            ledger = Ledger(limits)
            for attempt in range(attempts):
                records[name].append(execute(task, backend, ledger, arm="evaluation",
                                             attempt=attempt, seed=seed, run_id="evaluation"))
            ledgers[name].append(dict(task=task.key, report=ledger.report(), calls=ledger.events))
    return dict(panel_hash=digest(sorted(t.fingerprint for t in tasks)), attempts=attempts,
                seed=seed, task_keys=sorted(t.key for t in tasks), records=records, ledgers=ledgers)


def pass_at_k(n: int, correct: int, k: int) -> float:
    require(all(type(v) is int for v in (n, correct, k)) and 0 <= correct <= n and 1 <= k <= n,
            "pass@k arguments")
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k) if n - correct >= k else 1.0


def paired_effect(base: list[dict], candidate: list[dict], *, samples=2000, seed=37) -> dict:
    require(type(samples) is int and samples >= 100, "bootstrap samples")
    def indexed(rows):
        out = {}
        for row in rows:
            key = (row["task"], row["attempt"], row["seed"])
            require(key not in out, "duplicate paired outcome")
            out[key] = row
        return out
    b, c = indexed(base), indexed(candidate)
    require(bool(b) and set(b) == set(c), "incomplete paired outcomes")
    groups = defaultdict(list)
    gains = damages = 0
    for key, left in b.items():
        right = c[key]
        require(all(left[f] == right[f] for f in ("task_hash", "parent", "source", "skill", "split", "code_hash")), "paired provenance")
        require(all(r["harness"] == "H0" and r["runbook"] is None and r["reviewer"] is None
                    and r["arm"] == "evaluation" for r in (left, right)), "privileged evaluation")
        difference = int(right["success"]) - int(left["success"])
        gains += difference == 1
        damages += difference == -1
        groups[left["parent"]].append(difference)
    values = [sum(group) / len(group) for _, group in sorted(groups.items())]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return dict(mean=sum(values) / len(values), lower=boot[int(.025 * (samples - 1))],
                upper=boot[int(.975 * (samples - 1))], parents=len(values), paired_attempts=len(b),
                repairs=gains, damages=damages, estimand="parent-macro paired success difference",
                interval="paired parent bootstrap percentile; exploratory, not a safety guarantee")


def compare_panel(panel: dict, tasks: list[Task], *, baseline: str, candidate: str) -> dict:
    catalog = validate_catalog(tasks)
    require(panel["panel_hash"] == digest(sorted(t.fingerprint for t in tasks)), "panel drift")
    expected = {(key, attempt, panel["seed"]) for key in catalog for attempt in range(panel["attempts"])}
    for name in (baseline, candidate):
        records = panel["records"][name]
        require(len(records) == len(expected), "missing/extra evaluation attempts")
        require({(r["task"], r["attempt"], r["seed"]) for r in records} == expected, "panel coverage")
        for row in records:
            audit(row, catalog[row["task"]])
    left, right = panel["records"][baseline], panel["records"][candidate]
    groups = {}
    for role in ("calibration", "retention_eval"):
        a, b = [r for r in left if r["split"] == role], [r for r in right if r["split"] == role]
        if a:
            groups[role] = paired_effect(a, b)
    per_source = {}
    for source, skill in sorted({(r["source"], r["skill"]) for r in left}):
        a = [r for r in left if (r["source"], r["skill"]) == (source, skill)]
        b = [r for r in right if (r["source"], r["skill"]) == (source, skill)]
        per_source[source + "/" + skill] = paired_effect(a, b)
    pass_metrics = {}
    for name in (baseline, candidate):
        by_task = defaultdict(list)
        for row in panel["records"][name]:
            by_task[row["task"]].append(row)
        pass_metrics[name] = {
            "pass_at_1": sum(r["success"] for rows in by_task.values() for r in rows) / len(expected),
            "supported_success_rate": sum(r["success"] and r["grounded"] for rows in by_task.values() for r in rows) / len(expected),
            "pass_at_k": {str(k): sum(pass_at_k(len(rows), sum(r["success"] for r in rows), k)
                                           for rows in by_task.values()) / len(by_task)
                          for k in range(1, panel["attempts"] + 1)},
        }
    return dict(groups=groups, per_source_skill=per_source, pass_metrics=pass_metrics,
                promotion="NOT_AUTHORIZED: freeze controls, independent confirmation and retention gates first")
