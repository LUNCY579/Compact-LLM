"""Reviewer-free evaluation and paired parent-cluster descriptive intervals."""
from __future__ import annotations

import random
from collections import defaultdict
from .collection import Ledger, Student, collect
from .contracts import HARNESS_HASH, Action, Limits, Task, digest, require, validate_catalog
from .datasets import replay


def evaluate(tasks: list[Task], student: Student, limits: Limits, *, panel_hash: str, seed: int = 0):
    validate_catalog(tasks)
    require(all(t.split in {"calibration", "retention_eval"} for t in tasks), "evaluation roles only")
    manifest = digest(sorted((t.key, t.fingerprint) for t in tasks))
    require(panel_hash == manifest, "frozen panel mismatch")
    ledger = Ledger(limits)
    results = []
    for task in sorted(tasks, key=lambda t: t.key):
        record = collect(task, student, mode="student_only", limits=limits, ledger=ledger,
                         evaluation=True, seed=seed)
        replay(record, task)
        actions = [Action.parse(s["accepted"]) for s in record["steps"]]
        kinds = [a.kind for a in actions]
        results.append({"key": task.key, "parent": task.parent, "source": task.source,
                        "skill": task.skill, "split": task.split, "panel_hash": manifest,
                        "harness_hash": HARNESS_HASH, "limits_hash": digest(record["limits"]),
                        "seed": seed, "success": record["success"], "status": record["status"],
                        "checked": "check" in kinds, "retrieved": "retrieve" in kinds,
                        "abstained": "abstain" in kinds, "steps": len(actions),
                        "tool_calls": kinds.count("check") + kinds.count("retrieve"),
                        "reviewer_calls": record["cost"]["reviewer_calls"]})
    return results


def paired_report(base: list[dict], candidate: list[dict], *, seed: int = 0, samples: int = 1000):
    require(type(samples) is int and samples >= 100, "at least 100 bootstrap replicates")
    def index(rows):
        require(bool(rows), "empty evaluation")
        require(len({r["key"] for r in rows}) == len(rows), "duplicate eval task")
        require(len({(r["panel_hash"], r["harness_hash"], r["limits_hash"], r["seed"])
                     for r in rows}) == 1, "mixed evaluation identities")
        for r in rows:
            require(r["reviewer_calls"] == 0, "reviewed trajectory is not deployment evidence")
            require(type(r["success"]) is bool, "invalid success type")
        return {r["key"]: r for r in rows}
    b, c = index(base), index(candidate)
    require(set(b) == set(c), "incomplete paired coverage")
    clusters = defaultdict(list)
    banks = defaultdict(lambda: {"base": [], "candidate": []})
    for key in sorted(b):
        for field in ("parent", "source", "split", "skill", "panel_hash", "harness_hash", "limits_hash", "seed"):
            require(b[key][field] == c[key][field], f"paired identity drift: {field}")
        clusters[b[key]["parent"]].append(int(c[key]["success"]) - int(b[key]["success"]))
        bank = banks[b[key]["source"]]
        bank["base"].append(int(b[key]["success"]))
        bank["candidate"].append(int(c[key]["success"]))
    # Equal parent weighting, not an IID-row interval over correlated variants.
    deltas = [sum(v) / len(v) for _, v in sorted(clusters.items())]
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
    sources = {source: {"tasks": len(v["base"]), "base": sum(v["base"])/len(v["base"]),
                        "candidate": sum(v["candidate"])/len(v["candidate"]),
                        "delta": (sum(v["candidate"])-sum(v["base"]))/len(v["base"])}
               for source, v in banks.items()}
    return {"tasks": len(b), "parent_groups": len(deltas),
            "parent_mean_delta": sum(deltas)/len(deltas),
            "descriptive_95_interval": [bootstrap[int(.025*samples)], bootstrap[min(samples-1, int(.975*samples))]],
            "sources": sources, "confirmatory": False,
            "warning": "Adaptive calibration and fixture panels do not establish native efficacy; audit each retention source."}
