"""Exact generated-answer metrics, paired diagnostics, and descriptive uncertainty."""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from .common import require
from .diagnostics import FAMILIES

PROBES = tuple(f for f in FAMILIES if f != "retention")


def integer_answer(text: str) -> str | None:
    text = text.strip()
    return str(int(text)) if re.fullmatch(r"[+-]?[0-9]{1,20}", text) else None


def grade(rows: list[dict], predictions: list[dict]) -> dict[str, dict]:
    require(len(predictions) == len(rows), "Incomplete prediction bank")
    require(len({p["id"] for p in predictions}) == len(predictions), "Duplicate prediction")
    indexed = {p["id"]: p for p in predictions}
    require(set(indexed) == {r["id"] for r in rows}, "Prediction IDs do not match suite")
    scored = {}
    for row in rows:
        p = indexed[row["id"]]
        require(isinstance(p["text"], str), "Invalid generated text")
        for name in ("gold_logp", "distractor_logp"):
            require(type(p[name]) in (float, int) and math.isfinite(p[name]) and p[name] <= 0,
                    "Invalid answer log probability")
        for name in ("gold_tokens", "distractor_tokens"):
            require(type(p[name]) is int and p[name] > 0, "Missing answer-token count")
        require(p["stop_reason"] in {"eos", "length"}, "Unknown generation stop")
        parsed = integer_answer(p["text"]) if p["stop_reason"] == "eos" else None
        margin = p["gold_logp"] - p["distractor_logp"]
        # Stable logistic; this is only a two-candidate conditional probability.
        prob = 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))
        scored[row["id"]] = {"correct": parsed == row["answer"],
                              "shortcut": parsed == row["distractor"],
                              "format_error": parsed is None, "margin": margin,
                              "candidate_probability": prob,
                              "mean_token_margin": p["gold_logp"] / p["gold_tokens"] - p["distractor_logp"] / p["distractor_tokens"]}
    return scored


def mean(values) -> float:
    values = list(values)
    require(bool(values), "Undefined mean: empty population")
    return sum(values) / len(values)


def summarize(rows: list[dict], predictions: list[dict], reference: list[dict] | None = None) -> dict:
    scored = grade(rows, predictions)
    ref = grade(rows, reference) if reference is not None else scored
    families = {}
    for family in FAMILIES:
        members = [r for r in rows if r["family"] == family]
        clean = {r["parent_id"]: r for r in members if r["condition"] == "clean"}
        misleading = {r["parent_id"]: r for r in members if r["condition"] == "misleading"}
        own_solved = [p for p, r in clean.items() if scored[r["id"]]["correct"]]
        ref_solved = [p for p, r in clean.items() if ref[r["id"]]["correct"]]
        families[family] = {
            "pairs": len(clean),
            "clean": mean(scored[r["id"]]["correct"] for r in clean.values()),
            "misleading": mean(scored[r["id"]]["correct"] for r in misleading.values()),
            "shortcut_rate": mean(scored[r["id"]]["shortcut"] for r in misleading.values()),
            "own_clean_denominator": len(own_solved),
            "conditional_robustness": mean(scored[misleading[p]["id"]]["correct"] for p in own_solved) if own_solved else None,
            "reference_clean_denominator": len(ref_solved),
            "fixed_reference_robustness": mean(scored[misleading[p]["id"]]["correct"] for p in ref_solved) if ref_solved else None,
            "mean_logp_margin": mean(scored[r["id"]]["margin"] for r in misleading.values()),
            "mean_token_margin": mean(scored[r["id"]]["mean_token_margin"] for r in misleading.values()),
            "candidate_preference": mean(scored[r["id"]]["margin"] > 0 for r in misleading.values()),
            "format_error_rate": mean(scored[r["id"]]["format_error"] for r in members),
        }
    retention = [r for r in rows if r["family"] == "retention"]
    solved = [r for r in retention if ref[r["id"]]["correct"]]
    components = defaultdict(dict)
    for r in rows:
        if r.get("skill_case_id"):
            components[r["skill_case_id"]][(r["family"], r["condition"])] = r
    supported = [case for case in components.values() if all(scored[case[(f, "clean")]["id"]]["correct"] for f in ("retrieve_R", "transform_S"))]
    composition = {"component_solved_denominator": len(supported),
                   "clean_RS_given_R_and_S": mean(scored[c[("compose_RS", "clean")]["id"]]["correct"] for c in supported) if supported else None,
                   "misleading_RS_given_R_and_S": mean(scored[c[("compose_RS", "misleading")]["id"]]["correct"] for c in supported) if supported else None}
    return {"families": families, "composition_given_components": composition,
            "ordinary_macro": mean(families[f]["clean"] for f in PROBES),
            "robust_macro": mean(families[f]["misleading"] for f in PROBES),
            "retention": {"reference_solved": len(solved),
                          "preserved_fraction": mean(scored[r["id"]]["correct"] for r in solved) if solved else None,
                          "damages": sum(not scored[r["id"]]["correct"] for r in solved),
                          "repairs": sum(scored[r["id"]]["correct"] and not ref[r["id"]]["correct"] for r in retention)}}


def eligible(summary: dict, baseline: dict, policy: dict) -> tuple[bool, list[str]]:
    reasons = []
    for family in PROBES:
        if summary["families"][family]["clean"] < policy["min_clean_accuracy"]:
            reasons.append(f"clean_competence:{family}")
    r = summary["retention"]
    if r["reference_solved"] < policy["min_reference_solved"]:
        reasons.append("insufficient_reference_retention_support")
    if r["preserved_fraction"] is None or r["preserved_fraction"] < 1 - policy["max_retention_damage_fraction"]:
        reasons.append("retention_damage")
    if summary["ordinary_macro"] < baseline["ordinary_macro"] + policy["min_acquisition_delta"]:
        reasons.append("acquisition_floor")
    return not reasons, reasons


def paired_comparison(rows: list[dict], a: list[dict], b: list[dict], resamples: int = 2000) -> dict:
    """A minus B; resample parent units within family, never prompt variants independently."""
    sa, sb = grade(rows, a), grade(rows, b)
    clusters = defaultdict(list)
    strata = {}
    for r in rows:
        if r["family"] in PROBES and r["condition"] == "misleading":
            cluster = r["cluster_id"]
            clusters[cluster].append(int(sa[r["id"]]["correct"]) - int(sb[r["id"]]["correct"]))
            strata[cluster] = "linked_R_S_RS" if r.get("skill_case_id") else r["family"]
    require(resamples >= 100, "Too few bootstrap resamples")
    groups = defaultdict(list)
    for cluster, values in clusters.items():
        groups[strata[cluster]].append(mean(values))
    require(set(groups) == {"successive", "repetitive", "linked_R_S_RS"}, "Missing paired cluster stratum")
    weights = {"successive": 1, "repetitive": 1, "linked_R_S_RS": 3}
    diffs = [d for values in clusters.values() for d in values]
    wins, losses = diffs.count(1), diffs.count(-1)
    cluster_wins = sum(sum(values) > 0 for values in clusters.values())
    cluster_losses = sum(sum(values) < 0 for values in clusters.values())
    n = cluster_wins + cluster_losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(cluster_wins, cluster_losses) + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(1001)
    samples = sorted(sum(weights[name] * mean(rng.choices(group, k=len(group))) for name, group in groups.items()) / 5 for _ in range(resamples))
    return {"macro_difference": sum(weights[name] * mean(group) for name, group in groups.items()) / 5,
            "paired_wins": wins, "paired_losses": losses,
            "independent_clusters": len(clusters), "cluster_sign_two_sided_p": p,
            "stratified_cluster_bootstrap_95_interval": [samples[int(.025 * resamples)], samples[int(.975 * resamples) - 1]],
            "resamples": resamples, "inference": "descriptive; linked R/S/RS resampled together; no seed-level replication or multiplicity correction"}



def trajectory(rows: list[dict], results: list[list[dict]]) -> dict:
    scored = [grade(rows, result) for result in results]
    changes = []
    for before, after in zip(scored, scored[1:]):
        changes.append({"repairs": sum(not before[k]["correct"] and after[k]["correct"] for k in before),
                        "damages": sum(before[k]["correct"] and not after[k]["correct"] for k in before)})
    reversals = sum(any(scored[i][r["id"]]["correct"] == scored[i+2][r["id"]]["correct"] != scored[i+1][r["id"]]["correct"]
                        for i in range(len(scored) - 2)) for r in rows)
    return {"adjacent_transitions": changes, "rows_with_three_checkpoint_reversal": reversals,
            "interpretation": "behavioral transitions only; not proof of circuit-level mode-hopping"}
