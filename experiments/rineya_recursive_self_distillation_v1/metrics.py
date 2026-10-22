"""Paired parent-level estimates; adaptive panels never certify final efficacy."""
from __future__ import annotations

import random
from collections import defaultdict

from .contracts import require


def _index(rows):
    require(bool(rows), "empty evaluation")
    index = {}
    for r in rows:
        require(type(r["correct"]) is bool and type(r["terminated"]) is bool
                and type(r["tokens"]) is int and r["tokens"] > 0, "invalid evaluation outcome")
        require(r["key"] not in index, "duplicate evaluation key")
        index[r["key"]] = r
    return index


def paired_effect(before, after, *, samples=2000, seed=101):
    a, b = _index(before), _index(after)
    require(set(a) == set(b), "incomplete paired coverage")
    groups = defaultdict(list)
    gains = losses = 0
    for key, x in a.items():
        y = b[key]
        require(all(x[n] == y[n] for n in ("parent", "source", "split", "budget")),
                "pair metadata/decoding budget drift")
        delta = int(y["correct"]) - int(x["correct"])
        groups[x["parent"]].append(delta)
        gains += delta == 1
        losses += delta == -1
    values = [sum(v) / len(v) for _, v in sorted(groups.items())]
    point = sum(values) / len(values)
    require(type(samples) is int and 100 <= samples <= 20000, "invalid bootstrap size")
    rng = random.Random(seed)
    distribution = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return {"parent_macro_delta": point, "ci95": [distribution[int(samples * .025)],
            distribution[min(samples - 1, int(samples * .975))]], "parents": len(values),
            "rows": len(a), "source_wrong_to_correct": gains, "source_correct_to_wrong": losses,
            "scope": "exploratory paired parent bootstrap; not sequentially valid or confirmatory"}


def summary(rows):
    index = _index(rows)
    groups = defaultdict(list)
    for r in index.values():
        groups[r["parent"]].append(int(r["correct"]))
    return {"rows": len(rows), "parents": len(groups),
            "parent_macro_accuracy": sum(sum(v) / len(v) for v in groups.values()) / len(groups),
            "row_accuracy": sum(r["correct"] for r in rows) / len(rows),
            "mean_tokens": sum(r["tokens"] for r in rows) / len(rows),
            "truncation_rate": sum(not r["terminated"] for r in rows) / len(rows)}


def retention_stop(baseline, current, max_drop: float, max_truncation: float):
    require(set(_index(baseline)) == set(_index(current)), "retention paired coverage")
    sources = sorted({r["source"] for r in baseline if r["split"] == "retention_eval"})
    require(bool(sources), "retention panel missing")
    details, stop = {}, False
    for source in sources:
        a = [r for r in baseline if r["split"] == "retention_eval" and r["source"] == source]
        b = [r for r in current if r["split"] == "retention_eval" and r["source"] == source]
        effect = paired_effect(a, b, samples=100)
        drop = -effect["parent_macro_delta"]
        details[source] = {"observed_drop": drop, "repairs": effect["source_wrong_to_correct"],
                           "damages": effect["source_correct_to_wrong"], "summary": summary(b)}
        stop |= drop > max_drop or details[source]["summary"]["truncation_rate"] > max_truncation
    return bool(stop), details
