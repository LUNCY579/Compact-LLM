"""Paired functional evidence, parent-aware intervals, CAL selection, CHECK lock."""
from __future__ import annotations
import itertools
import math
import numpy as np
from .core import require
from .protocol import hash_arg, digest
from . import EXPERIMENT

BOUND = ("plan_sha256", "dataset_sha256", "scorer_sha256", "runtime_sha256",
         "decoding_sha256", "tokenizer_sha256", "freeze_sha256")


def validate_run(run):
    require(run.get("schema") == 1, "run schema mismatch")
    meta = run["metadata"]
    require(meta.get("experiment") == EXPERIMENT and meta.get("split") in ("CAL", "CHECK"), "run identity mismatch")
    for key in BOUND + ("artifact_sha256",):
        hash_arg(meta.get(key), key)
    require(meta.get("evaluation_dtype") == "float32", "matched FP32 evaluation required")
    require(isinstance(meta.get("arm"), str) and meta["arm"], "arm missing")
    require(run.get("rows"), "empty run")
    ids = set()
    parents = {}
    for r in run["rows"]:
        require(set(r) == {"id", "parent_id", "domain", "role", "correct", "loss"}, "scored row fields mismatch")
        for key in ("id", "parent_id", "domain"):
            require(isinstance(r[key], str) and r[key], "row identity missing")
        require(r["id"] not in ids, "duplicate scored row")
        ids.add(r["id"])
        require(r["role"] in ("capability", "retention"), "unknown role")
        identity = (r["domain"], r["role"])
        require(parents.get(r["parent_id"], identity) == identity, "parent spans domain/role")
        parents[r["parent_id"]] = identity
        require(type(r["correct"]) is bool, "correct must be boolean")
        require(r["loss"] is None or (type(r["loss"]) in (int, float) and math.isfinite(r["loss"])), "loss must be finite or null")
    return run


def align(runs):
    require(len(runs) >= 2, "two or more runs required")
    for run in runs:
        validate_run(run)
    first = runs[0]
    ids = sorted(r["id"] for r in first["rows"])
    maps = []
    for run in runs:
        for key in BOUND + ("split", "evaluation_dtype"):
            require(first["metadata"][key] == run["metadata"][key], f"paired metadata differs: {key}")
        mapping = {r["id"]: r for r in run["rows"]}
        require(sorted(mapping) == ids, "missing/extra paired rows")
        maps.append(mapping)
    for row_id in ids:
        for key in ("parent_id", "domain", "role"):
            require(all(m[row_id][key] == maps[0][row_id][key] for m in maps), f"paired row identity differs: {key}")
    return ids, maps


def parent_interval(values, parents, *, seed=1729, draws=2000):
    """Equal-parent estimand. Do not mislabel this as an equal-row interval."""
    require(len(values) == len(parents) and len(values) > 0, "empty/mismatched interval")
    require(type(draws) is int and 100 <= draws <= 10000, "bootstrap draw budget")
    groups = sorted(set(parents))
    means = np.array([np.mean([v for v, p in zip(values, parents) if p == group]) for group in groups])
    require(np.isfinite(means).all(), "nonfinite interval values")
    if len(groups) < 2:
        return {"estimate": float(means.mean()), "ci95": None, "parents": len(groups), "estimand": "equal_parent_mean"}
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(means, len(means), replace=True).mean() for _ in range(draws)])
    return {"estimate": float(means.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "parents": len(groups), "estimand": "equal_parent_mean", "seed": seed, "draws": draws,
            "interval_is_descriptive_not_multiplicity_adjusted": True}


def compare(reference, candidate):
    ids, (a, b) = align([reference, candidate])
    domains = {}
    for domain in sorted({a[i]["domain"] for i in ids}):
        rows = [i for i in ids if a[i]["domain"] == domain]
        old = np.array([a[i]["correct"] for i in rows], dtype=int)
        new = np.array([b[i]["correct"] for i in rows], dtype=int)
        domains[domain] = {"n": len(rows), "reference_correct": int(old.sum()), "candidate_correct": int(new.sum()),
                           "gains": int(((new == 1) & (old == 0)).sum()),
                           "damages": int(((new == 0) & (old == 1)).sum()),
                           "retained_correct": int(((new == 1) & (old == 1)).sum()),
                           "accuracy_delta": float((new-old).mean()),
                           "parent_interval": parent_interval((new-old).tolist(), [a[i]["parent_id"] for i in rows])}
    return {"reference": reference["metadata"]["arm"], "candidate": candidate["metadata"]["arm"],
            "split": reference["metadata"]["split"], "domains": domains,
            "release_permitted": False, "no_forgetting_claim": False}


def interaction(base, a, b, combined):
    ids, ms = align([base, a, b, combined])
    require(all(m[i]["loss"] is not None for m in ms for i in ids), "interaction requires per-row losses")
    values = [ms[3][i]["loss"]-ms[1][i]["loss"]-ms[2][i]["loss"]+ms[0][i]["loss"] for i in ids]
    return {"formula": "L(A+B)-L(A)-L(B)+L(base)",
            "mean": float(np.mean(values)),
            "interval": parent_interval(values, [ms[0][i]["parent_id"] for i in ids]),
            "interpretation": "loss interaction at caller-declared matched doses; not automatically synergy"}


def factorial_effects(runs):
    names = [f"P{p}S{s}C{c}" for p, s, c in itertools.product((0, 1), repeat=3)]
    require(set(runs) == set(names), "exact eight factorial cells required")
    ids, maps = align([runs[n] for n in names])
    means = np.array([[int(m[i]["correct"]) for i in ids] for m in maps])
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    effects = {}
    for count in (1, 2, 3):
        for axes in itertools.combinations(range(3), count):
            contrast = np.prod(signs[:, axes], axis=1)
            per_row = contrast @ means / 4.0
            label = "x".join("PSC"[j] for j in axes)
            effects[label] = parent_interval(per_row.tolist(), [maps[0][i]["parent_id"] for i in ids])
    return {"effects": effects, "normalization": "2-level factorial effects: signed sum / 4", "causal_claim": False}


def validate_frozen_run(run, frozen):
    validate_run(run)
    m = run["metadata"]
    require(m["plan_sha256"] == frozen["plan_sha256"] and m["freeze_sha256"] == digest(frozen), "run/freeze mismatch")
    require(m["artifact_sha256"] == frozen["artifacts"].get(m["arm"]), "unfrozen artifact")
    expected = frozen["cal_sha256"] if m["split"] == "CAL" else frozen["check_sha256"]
    require(m["dataset_sha256"] == expected, "unfrozen dataset")


def select_on_cal(reference, candidates, frozen, *, retention_margin=0.0, min_gain=0.0):
    """Frozen worst-domain retention screen; not a significance/promotion test."""
    require(reference["metadata"]["split"] == "CAL", "selection must not open CHECK")
    require(reference["metadata"]["arm"] == "base", "selection retention reference must be the frozen base")
    require(candidates, "no candidates")
    for run in [reference] + candidates:
        validate_frozen_run(run, frozen)
    seen = {reference["metadata"]["arm"]}
    scored = []
    for candidate in candidates:
        name = candidate["metadata"]["arm"]
        require(name not in seen, "duplicate selection arm")
        seen.add(name)
        ids, (a, b) = align([reference, candidate])
        role_scores = {}
        for role in ("capability", "retention"):
            groups = sorted({a[i]["domain"] for i in ids if a[i]["role"] == role})
            require(groups, f"missing {role} domains")
            role_scores[role] = [float(np.mean([int(b[i]["correct"])-int(a[i]["correct"])
                                               for i in ids if a[i]["domain"] == d and a[i]["role"] == role])) for d in groups]
        gain = float(np.mean(role_scores["capability"]))
        worst = min(role_scores["retention"])
        scored.append({"arm": name, "capability_gain": gain, "worst_retention_delta": worst,
                       "eligible": worst >= -retention_margin and gain > min_gain})
    eligible = sorted((r for r in scored if r["eligible"]), key=lambda r: (-r["capability_gain"], -r["worst_retention_delta"], r["arm"]))
    return {"schema": 1, "freeze_sha256": digest(frozen), "selection_split": "CAL",
            "selected": eligible[0]["arm"] if eligible else None, "candidates": scored,
            "retention_margin": retention_margin, "min_gain": min_gain,
            "cal_run_hashes": [digest(r) for r in [reference]+candidates],
            "selection_is_not_promotion": True}


def confirm_check(run, frozen, selection):
    validate_frozen_run(run, frozen)
    require(run["metadata"]["split"] == "CHECK", "confirmation must be CHECK")
    require(selection["freeze_sha256"] == digest(frozen) and selection["selection_split"] == "CAL", "selection not bound")
    require(selection["selected"] is not None and run["metadata"]["arm"] == selection["selected"], "CHECK arm differs from locked selection")
    return {"status": "FROZEN_CANDIDATE_CHECK_ACCEPTED_FOR_ANALYSIS", "run_sha256": digest(run),
            "selection_sha256": digest(selection), "release_permitted": False}
