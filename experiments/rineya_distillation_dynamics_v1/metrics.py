"""Paired acquisition/retention, parent-cluster intervals, coverage and selection."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .contracts import require, step_seed
from .data import Task, verify


def evaluate(backend, tasks: list[Task], role: str, seed: int) -> dict:
    require(role in {"student", "teacher"}, "evaluator model role")
    rows = {}
    for task in tasks:
        greedy = backend.sample(task.public(), role, step_seed(seed, task.key, 0, "eval-greedy"), greedy=True)
        draws = [backend.sample(task.public(), role, step_seed(seed, task.key, i, "eval-draw"))
                 for i in range(backend.plan.eval_samples)]
        # Deployment selector is majority of complete visible strings, never the gold answer.
        texts = [r.text.strip() for r in draws]
        counts = Counter(texts)
        selected = max(range(len(texts)), key=lambda i: (counts[texts[i]], -i))
        rows[task.key] = {"parent": task.parent, "source": task.source, "role": task.role,
                          "correct": verify(task, greedy.text), "text": greedy.text,
                          "terminated": greedy.terminated,
                          "samples": [verify(task, r.text) for r in draws], "sample_texts": texts,
                          "selected_index": selected, "selected_correct": verify(task, texts[selected]),
                          "generated_tokens": len(greedy.response_ids) + sum(len(r.response_ids) for r in draws)}
    return rows


def pass_at_k(correct: int, n: int, k: int) -> float:
    require(type(correct) is int and type(n) is int and type(k) is int and 0 <= correct <= n and 0 < k <= n,
            "pass@k requires 0 <= c <= n and 1 <= k <= n")
    return 1.0 if n-correct < k else 1 - math.comb(n-correct, k)/math.comb(n, k)


def paired(base: dict, result: dict, *, bootstrap=1000, seed=7301) -> dict:
    require(bool(base) and set(base) == set(result), "complete paired key coverage required")
    require(type(bootstrap) is int and bootstrap >= 0, "bootstrap count")
    groups, differences = defaultdict(list), defaultdict(list)
    for key, b in base.items():
        r = result[key]
        require(all(b[k] == r[k] for k in ("parent", "source", "role")), "paired metadata mismatch")
        require(type(b["correct"]) is bool and type(r["correct"]) is bool
                and type(r["selected_correct"]) is bool, "boolean outcomes required")
        require(bool(r["samples"]) and all(type(x) is bool for x in r["samples"]), "sampling outcomes")
        group = r["role"] + "/" + r["source"]
        groups[group].append((b, r))
        differences[r["parent"]].append(int(r["correct"]) - int(b["correct"]))
    summaries = {}
    for key, rows in groups.items():
        b = sum(x[0]["correct"] for x in rows)
        c = sum(x[1]["correct"] for x in rows)
        kept = sum(x[0]["correct"] and x[1]["correct"] for x in rows)
        n = len(rows)
        sample_counts = {len(x[1]["samples"]) for x in rows}
        require(len(sample_counts) == 1, "evaluation sample budget drift")
        draws = sample_counts.pop()
        summaries[key] = {"n": n, "base_correct": b, "correct": c, "accuracy": c/n,
                          "gain": (c-b)/n, "retained": kept, "damaged": b-kept, "rescued": c-kept,
                          "reference_retention": kept/b if b else None,
                          "majority_selected_accuracy": sum(x[1]["selected_correct"] for x in rows)/n,
                          "generated_tokens": sum(x[1]["generated_tokens"] for x in rows),
                          "pass_at_k": {str(k): sum(pass_at_k(sum(x[1]["samples"]), draws, k)
                                                     for x in rows)/n
                                        for k in sorted({1, min(2, draws), draws})}}
    interval = None
    if bootstrap:
        rng = random.Random(seed)
        parents = sorted(differences)
        samples = []
        for _ in range(bootstrap):
            vals = [v for p in rng.choices(parents, k=len(parents)) for v in differences[p]]
            samples.append(sum(vals)/len(vals))
        samples.sort()
        interval = [samples[int(.025*(bootstrap-1))], samples[int(.975*(bootstrap-1))]]
    return {"groups": summaries, "all_tasks_paired_gain_ci95": interval,
            "interval_scope": "parent bootstrap within one seed; NOT across-training-seed uncertainty",
            "parent_clusters": len(differences)}


def decision_features(summary: dict, *, prefix="dev") -> dict:
    groups = summary["groups"]
    acq = [v for k, v in groups.items() if k.startswith(prefix + "_acquisition/")]
    ret = [v for k, v in groups.items() if k.startswith(prefix + "_retention/")]
    hard = [v for k, v in groups.items() if k.startswith(prefix + "_hard/")]
    require(acq and ret and hard, "acquisition, retention and hard generalization all required")
    return {"acquisition_gain": sum(v["gain"]*v["n"] for v in acq)/sum(v["n"] for v in acq),
            "worst_retention_drop": max(-v["gain"] for v in ret),
            "damaged": sum(v["damaged"] for v in ret),
            "hard_accuracy": sum(v["correct"] for v in hard)/sum(v["n"] for v in hard)}
