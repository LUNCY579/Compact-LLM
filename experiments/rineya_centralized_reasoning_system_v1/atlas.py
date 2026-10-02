"""Reproducible exploratory estimates. Unmeasured values stay undefined."""
from __future__ import annotations

import math
import random
import re


REQUIRED_ROW_FIELDS = frozenset({
    "study", "recipe", "checkpoint_sha256", "source_checkpoint_sha256", "evidence_class",
    "family", "composition_depth", "train_draw_seed", "order_seed", "eval_family_seed",
    "unique_parents", "presentations", "processed_tokens", "supervised_tokens", "updates",
    "worker_seconds", "peak_ram_bytes", "peak_vram_bytes", "optimizer", "update_scope",
    "data_sha256", "scorer_sha256", "qualification_sha256", "code_commit", "exposure",
    "acquisition", "transfer", "retention", "calibration", "failures", "stop_reason"})


def validate_row(row):
    if not isinstance(row, dict) or set(row) != REQUIRED_ROW_FIELDS:
        raise ValueError("atlas row must contain every provenance/dose/measurement field")
    if row["evidence_class"] not in {"NATIVE_EXPLORATORY", "NATIVE_CONFIRMATION", "UNMEASURED"}:
        raise ValueError("CPU mechanics cannot become a neural learning-atlas result")
    for key in ("unique_parents", "presentations", "processed_tokens", "supervised_tokens", "updates", "failures"):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f"nonnegative measured integer required: {key}")
    for key in ("worker_seconds", "peak_ram_bytes", "peak_vram_bytes"):
        value = row[key]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("nonfinite resource measurement")
    if row["supervised_tokens"] > row["processed_tokens"]:
        raise ValueError("supervised tokens exceed processed inputs")
    for key in ("acquisition", "transfer", "retention", "calibration"):
        metric = row[key]
        if metric is not None:
            if not isinstance(metric, dict) or set(metric) != {"estimate", "interval95", "scope"}:
                raise ValueError("metric must bind estimate, uncertainty and scope")
            values = [metric["estimate"]] + (metric["interval95"] if isinstance(metric["interval95"], list) and len(metric["interval95"]) == 2 else [])
            if len(values) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or values[1] > values[2] or not metric["scope"]:
                raise ValueError("invalid/undefined metric uncertainty")
    if row["evidence_class"] != "UNMEASURED":
        for key in ("checkpoint_sha256", "source_checkpoint_sha256", "data_sha256", "scorer_sha256", "qualification_sha256"):
            if not isinstance(row[key], str) or not re.fullmatch(r"[a-f0-9]{64}", row[key]):
                raise ValueError("native atlas row requires exact hash bindings")
        if not isinstance(row["code_commit"], str) or not re.fullmatch(r"[a-f0-9]{40}", row["code_commit"]) or row["acquisition"] is None or row["worker_seconds"] is None:
            raise ValueError("native atlas row lacks measured acquisition/cost/code")
        if not isinstance(row["optimizer"], dict) or not row["optimizer"] or not row["update_scope"] or not row["exposure"]:
            raise ValueError("native atlas row lacks recipe/exposure")
    return row


def efficiency(gain, row):
    validate_row(row)
    if gain is not None and (type(gain) not in (int, float) or not math.isfinite(gain)):
        raise ValueError("finite measured gain required")
    if gain is not None and row["evidence_class"] == "UNMEASURED":
        raise ValueError("unmeasured row cannot establish numeric efficiency")
    return {name: gain / row[name] if gain is not None and row[name] not in (None, 0) else None
            for name in ("unique_parents", "processed_tokens", "supervised_tokens", "updates", "worker_seconds")}


def paired_parent_interval(records, *, resamples=2000, seed=20261001):
    """Bootstrap whole parent means; descriptive conditional-on-family-mix CI.

    records are {parent_id,family,B,C}; duplicate parent variants stay together.
    A supplied-fixture interval is not a native efficacy estimate.
    """
    if type(resamples) is not int or not 100 <= resamples <= 10000 or not records:
        raise ValueError("bounded nonempty parent interval request required")
    parents = {}
    for row in records:
        if set(row) != {"parent_id", "family", "B", "C"}:
            raise ValueError("paired record schema")
        for arm in ("B", "C"):
            if type(row[arm]) not in (int, float) or not math.isfinite(row[arm]) or not 0 <= row[arm] <= 1:
                raise ValueError("finite score in [0,1]")
        group = parents.setdefault(row["parent_id"], {"family": row["family"], "deltas": []})
        if group["family"] != row["family"]:
            raise ValueError("parent changes family")
        group["deltas"].append(row["C"] - row["B"])
    strata = {}
    for parent in parents.values():
        strata.setdefault(parent["family"], []).append(sum(parent["deltas"])/len(parent["deltas"]))
    n = len(parents)
    observed = sum(sum(v) for v in strata.values()) / n
    rng = random.Random(seed)
    samples = sorted(sum(sum(rng.choice(v) for _ in v) for v in strata.values()) / n
                     for _ in range(resamples))
    return {"delta": observed, "parent_count": n, "variant_count": len(records),
            "family_count": len(strata), "interval95": [samples[int(.025*resamples)], samples[int(.975*resamples)-1]],
            "seed": seed, "resamples": resamples,
            "scope": "exploratory_conditional_on_observed_family_mix_not_new_family_generalization",
            "sequential_confirmation": False}


def skill_matrices(checkpoints, source, incumbent=None):
    """Each checkpoint maps skill -> item-id -> semantic success bool.

    Fixed item membership prevents hidden loss-denominator changes. Undefined
    source-correct denominators stay None. These are diagnostics, not NI proof.
    """
    incumbent = source if incumbent is None else incumbent
    for label, panel in [("source", source), ("incumbent", incumbent), *checkpoints.items()]:
        if set(panel) != set(source):
            raise ValueError(f"skill membership changed: {label}")
        for skill, items in panel.items():
            if set(items) != set(source[skill]) or any(type(v) is not bool for v in items.values()):
                raise ValueError("fixed boolean item panels required")
    performance, forgetting = {}, {}
    for checkpoint, panel in checkpoints.items():
        performance[checkpoint], forgetting[checkpoint] = {}, {}
        for skill, items in panel.items():
            n = len(items)
            src = source[skill]
            inc = incumbent[skill]
            retained = sum(src[k] and items[k] for k in items)
            original_correct = sum(src.values())
            performance[checkpoint][skill] = sum(items.values())/n if n else None
            forgetting[checkpoint][skill] = {
                "source_losses": sum(src[k] and not items[k] for k in items),
                "source_repairs": sum(not src[k] and items[k] for k in items),
                "incumbent_losses": sum(inc[k] and not items[k] for k in items),
                "source_correct_denominator": original_correct,
                "source_correct_retention": retained/original_correct if original_correct else None,
                "backward_transfer_vs_source": (sum(items.values())-sum(src.values()))/n if n else None,
                "noninferiority_established": False}
    return {"performance": performance, "forgetting": forgetting,
            "worst_backward_transfer": min((r["backward_transfer_vs_source"] for x in forgetting.values() for r in x.values()
                                              if r["backward_transfer_vs_source"] is not None), default=None)}


def schedule(candidates, *, seed, explore_probability=.25):
    """Bounded interpretable development allocation, never a promotion rule."""
    if not candidates or not 0 < explore_probability <= 1:
        raise ValueError("positive randomized exploration and eligible candidates required")
    values = []
    for candidate in candidates:
        if set(candidate) != {"id", "gain", "uncertainty", "forgetting_risk", "cost"}:
            raise ValueError("scheduler candidate schema")
        if any(type(candidate[k]) not in (int, float) or not math.isfinite(candidate[k]) for k in ("gain", "uncertainty", "forgetting_risk", "cost")) or candidate["cost"] <= 0 or candidate["uncertainty"] < 0 or not 0 <= candidate["forgetting_risk"] <= 1:
            raise ValueError("invalid cost/uncertainty/risk")
        values.append((candidate["gain"] + candidate["uncertainty"] - candidate["forgetting_risk"])/candidate["cost"])
    best = max(range(len(values)), key=values.__getitem__)
    probabilities = [explore_probability/len(candidates) + (1-explore_probability if i == best else 0) for i in range(len(candidates))]
    rng = random.Random(seed)
    chosen = rng.choices(range(len(candidates)), weights=probabilities)[0]
    return {"selected": candidates[chosen]["id"], "seed": seed,
            "eligible": candidates, "selection_probabilities": probabilities,
            "values": values, "rationale": "cost-normalized gain+uncertainty-forgetting with randomized exploration",
            "evidence": "proposed_development_policy_unmeasured_vs_random_or_fixed_control"}
