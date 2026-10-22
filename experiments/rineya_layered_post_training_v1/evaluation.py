"""Development-only selection and cluster-paired confirmation statistics."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .config import digest, require
from .data import fingerprint, score


def evaluate(runtime, rows, split):
    require(split in {"select", "confirm"} and rows and all(r.split == split for r in rows), "evaluation split mismatch")
    records = []
    with __import__("torch").no_grad():
        for row in sorted(rows, key=lambda r: r.id):
            seed = int(digest([runtime.config.seed, "evaluation", row.id])[:12], 16)
            ep = runtime.collect(row.prompt, 1, seed)[0]
            records.append({"id": row.id, "group": row.group, "suite": row.suite, "score": score(row, ep.text, ep.terminated), "terminated": ep.terminated, "response_tokens": len(ep.response)})
    return {"split": split, "data_hash": fingerprint(rows), "records": records, "scores": macro_scores(records)}


def macro_scores(records):
    """Equal weight per prompt family, then per suite; repeated variants cannot dominate."""
    buckets = defaultdict(lambda: defaultdict(list))
    require(bool(records), "empty evaluation records")
    ids = set()
    for r in records:
        require(r["id"] not in ids and math.isfinite(r["score"]) and 0 <= r["score"] <= 1, "invalid/duplicate score")
        ids.add(r["id"])
        buckets[r["suite"]][r["group"]].append(r["score"])
    return {suite: sum(sum(v) / len(v) for v in groups.values()) / len(groups) for suite, groups in sorted(buckets.items())}


def eligible(report, baseline, config):
    require(report["split"] == baseline["split"] == "select", "only development scores may select checkpoints")
    require(report["data_hash"] == baseline["data_hash"], "selection datasets differ")
    require(set(report["scores"]) >= set(config.required_suites) and set(baseline["scores"]) >= set(config.required_suites), "missing mandatory suite")
    for suite in config.required_suites:
        groups = {x["group"] for x in report["records"] if x["suite"] == suite}
        require(len(groups) >= config.min_eval_groups, f"insufficient independent groups in {suite}")
    return all(report["scores"][s] >= baseline["scores"][s] - config.max_retention_drop - 1e-12 for s in config.required_suites if s != "acquisition")


def choose(candidates, baseline, config):
    """Return a feasible checkpoint; no scalar reward may hide a protected-suite loss."""
    good = [(key, report) for key, report in candidates.items() if eligible(report, baseline, config)]
    require(bool(good), "no feasible checkpoint (include the untouched baseline)")
    return max(good, key=lambda x: (x[1]["scores"]["acquisition"], sum(v for k, v in x[1]["scores"].items() if k != "acquisition"), x[0]))[0]


def paired_confirmation(candidate, baseline, config):
    require(candidate["split"] == baseline["split"] == "confirm" and candidate["data_hash"] == baseline["data_hash"], "confirmation reports do not match")
    a = {r["id"]: r for r in candidate["records"]}
    b = {r["id"]: r for r in baseline["records"]}
    require(len(a) == len(candidate["records"]) and len(b) == len(baseline["records"]) and set(a) == set(b), "confirmation must be exactly paired")
    grouped = defaultdict(lambda: defaultdict(list))
    for key in a:
        require((a[key]["group"], a[key]["suite"]) == (b[key]["group"], b[key]["suite"]), "pairing metadata mismatch")
        grouped[a[key]["suite"]][a[key]["group"]].append(a[key]["score"] - b[key]["score"])
    require(set(grouped) >= set(config.required_suites), "missing confirmation suite")
    output = {}
    for suite, groups in sorted(grouped.items()):
        differences = [sum(v) / len(v) for _, v in sorted(groups.items())]
        require(len(differences) >= config.min_eval_groups, f"too few confirmation groups in {suite}")
        rng = random.Random(int(digest([config.seed, suite])[:12], 16))
        draws = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(config.bootstrap_samples))
        output[suite] = {"groups": len(differences), "delta": sum(differences) / len(differences), "ci95": [draws[int(.025 * len(draws))], draws[min(len(draws) - 1, int(.975 * len(draws)))]]}
    point_gate = output["acquisition"]["delta"] >= config.min_acquisition_gain
    retention_gate = all(output[s]["ci95"][0] >= -config.max_retention_drop for s in config.required_suites if s != "acquisition")
    return {"paired_cluster_bootstrap": output, "candidate_pass": point_gate and output["acquisition"]["ci95"][0] > 0 and retention_gate, "claim": "single-seed confirmation only; require independent seeds and matched-compute controls"}


def value_metrics(values, rewards):
    require(len(values) == len(rewards) and bool(values), "empty/mismatched critic audit")
    require(all(math.isfinite(v) for v in values) and all(r in (0., 1.) for r in rewards), "invalid critic audit")
    positive = [v for v, r in zip(values, rewards) if r == 1.]
    negative = [v for v, r in zip(values, rewards) if r == 0.]
    auc = None if not positive or not negative else sum(float(p > n) + .5 * float(p == n) for p in positive for n in negative) / (len(positive) * len(negative))
    mean = sum(rewards) / len(rewards)
    return {"count": len(values), "auroc": auc, "mse": sum((v - r)**2 for v, r in zip(values, rewards)) / len(values), "constant_mse": sum((r - mean)**2 for r in rewards) / len(rewards), "raw_value_not_probability": True}
