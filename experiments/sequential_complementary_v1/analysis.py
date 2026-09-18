"""Strict paired analysis. Never convert nonsignificance into preservation."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random
from .integrity import require


def final_answer(text: str, contract="FINAL_LINE_V1"):
    require(type(text) is str, "prediction must be text")
    if contract == "EXACT_SHORT_V1":
        return text.strip() if text.strip() and "\n" not in text.strip() else None
    require(contract == "FINAL_LINE_V1", "unknown answer contract")
    lines = text.strip().splitlines()
    if not lines or text.count("FINAL:") != 1 or not lines[-1].startswith("FINAL: "):
        return None
    answer = lines[-1][7:].strip()
    return answer or None


def score_records(public_ids, labels, predictions):
    """One record per registered case; missing/duplicate rows cause an explicit failure."""
    ids = list(public_ids)
    require(ids and len(ids) == len(set(ids)), "empty/duplicate population")
    require(len(labels) == len(ids) and {r["id"] for r in labels} == set(ids), "label population mismatch")
    require(len(predictions) == len(ids) and {r["id"] for r in predictions} == set(ids), "prediction population mismatch")
    expected = {r["id"]: r for r in labels}
    observed = {r["id"]: r for r in predictions}
    rows = []
    for id_ in ids:
        label, prediction = expected[id_], observed[id_]
        require(type(prediction["terminated"]) is bool, "termination must be explicit")
        parsed = final_answer(prediction["text"], label["output_contract"])
        correct = prediction["terminated"] and parsed == label["reference"]
        unknown = prediction["terminated"] and parsed == "UNKNOWN"
        rows.append({"id": id_, "world_id": label["world_id"], "family": label["family"],
                     "correct": bool(correct), "unanswerable": label["reference"] == "UNKNOWN",
                     "abstained": bool(unknown), "malformed": parsed is None,
                     "terminated": prediction["terminated"]})
    return rows


def _index(rows):
    require(rows and len({r["id"] for r in rows}) == len(rows), "empty/duplicate paired rows")
    require(all(type(r["correct"]) is bool for r in rows), "correctness must be Boolean")
    return {r["id"]: r for r in rows}


def paired(a, b):
    ia, ib = _index(a), _index(b)
    require(ia.keys() == ib.keys(), "paired populations differ; intersections are not allowed")
    counts = {"both_correct": 0, "repairs": 0, "damage": 0, "both_wrong": 0}
    for id_, row in ia.items():
        other = ib[id_]
        require(row["world_id"] == other["world_id"] and row["family"] == other["family"], "paired lineage mismatch")
        key = "both_correct" if row["correct"] and other["correct"] else "damage" if row["correct"] else "repairs" if other["correct"] else "both_wrong"
        counts[key] += 1
    n = len(ia)
    discordant = counts["repairs"] + counts["damage"]
    p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(counts["repairs"], counts["damage"]) + 1)) / 2**discordant) if discordant else 1.0
    clustered = len({r["world_id"] for r in a}) != n
    return {"n": n, **counts, "net_gain": counts["repairs"] - counts["damage"],
            "source_correct": counts["both_correct"] + counts["damage"],
            "candidate_correct": counts["both_correct"] + counts["repairs"],
            "no_observed_damage": counts["damage"] == 0,
            "exact_mcnemar_two_sided_p": None if clustered else p,
            "mcnemar_note": "not reported for repeated-world rows" if clustered else "unadjusted diagnostic; not an equivalence test",
            "preservation_established": False}


def cluster_interval(a, b, *, seed=17, replicates=2000):
    table = paired(a, b)
    require(type(replicates) is int and 100 <= replicates <= 100000, "bounded bootstrap replicates required")
    ia, ib = _index(a), _index(b)
    groups = defaultdict(list)
    for id_, row in ia.items():
        groups[row["world_id"]].append(int(ib[id_]["correct"]) - int(row["correct"]))
    if len(groups) < 2:
        return {"status": "INCONCLUSIVE", "reason": "fewer than two independent worlds"}
    values = list(groups.values())
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(sum(g) for g in sample) / sum(map(len, sample)))
    draws.sort()
    return {"status": "DESCRIPTIVE_OPEN_DEV", "worlds": len(groups), "replicates": replicates,
            "seed": seed, "point": table["net_gain"] / table["n"],
            "percentile_95": [draws[int(.025 * (replicates - 1))], draws[int(.975 * (replicates - 1))]],
            "confirmatory": False}


def factorial(a, b, c, d):
    """Old ordinary A / old MR B / new ordinary C / new MR D."""
    paired(a, b); paired(a, c); paired(a, d)
    cells = [_index(x) for x in (a, b, c, d)]
    patterns = Counter()
    records = []
    for id_ in cells[0]:
        values = [int(cell[id_]["correct"]) for cell in cells]
        av, bv, cv, dv = values
        patterns["".join(map(str, values))] += 1
        records.append({"id": id_, "world_id": cells[0][id_]["world_id"],
                        "family": cells[0][id_]["family"], "A": av, "B": bv, "C": cv, "D": dv,
                        "training_direct": cv - av, "multi_before": bv - av,
                        "multi_after": dv - cv, "interaction": (dv - cv) - (bv - av)})
    keys = ("training_direct", "multi_before", "multi_after", "interaction")
    return {"n": len(records), "patterns": dict(sorted(patterns.items())), "records": records,
            "effects": {k: sum(r[k] for r in records) / len(records) for k in keys},
            "by_family": {f: {k: sum(r[k] for r in records if r["family"] == f) /
                                           sum(r["family"] == f for r in records) for k in keys}
                          for f in sorted({r["family"] for r in records})},
            "classification": "DESCRIPTIVE_OPEN_DEV_INTERACTION", "synergy_established": False,
            "interpretation": "A positive interaction estimate alone does not establish synergy."}


def complementarity(candidate_correct, selected_correct):
    """Analysis-only oracle; not importable into a generation policy by data flow."""
    require(candidate_correct and len(candidate_correct) == len(selected_correct), "oracle population mismatch")
    require(all(len(r) == 4 and all(type(v) is bool for v in r) for r in candidate_correct), "four Boolean candidates required")
    require(all(type(v) is bool for v in selected_correct), "Boolean selections required")
    require(not any(s and not any(r) for r, s in zip(candidate_correct, selected_correct)),
            "selector-only new answer violates candidate-selection contract")
    oracle = sum(any(r) for r in candidate_correct)
    selected = sum(selected_correct)
    pairs = {}
    for i in range(4):
        for j in range(i + 1, 4):
            pairs[f"R{i}:R{j}"] = dict(Counter(f"{int(r[i])}{int(r[j])}" for r in candidate_correct))
    return {"candidate_oracle_correct": oracle, "selected_correct": selected,
            "selector_utilization": selected / oracle if oracle else None,
            "correct_candidate_but_wrong_selection": oracle - selected,
            "pairwise_correctness": pairs, "oracle_is_deployable": False}
