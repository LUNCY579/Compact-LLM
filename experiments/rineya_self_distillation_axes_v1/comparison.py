"""Prospective row-paired IID/transfer/replay accounting; no model execution."""
from __future__ import annotations

from statistics import mean
import math

from .analysis import behavioral_metrics, last_three_endpoint


def _cohort(records, ids):
    selected = [r for r in records if r["id"] in set(ids)]
    if len(selected) != len(ids) or {r["id"] for r in selected} != set(ids):
        raise ValueError("Missing/duplicate cohort rows; do not silently exclude failures")
    return selected


def summarize_cell(result, cohorts):
    """Keep harder transfer outside primary acquisition and replay source-correct."""
    iid = cohorts["iid_acquisition"]
    hard = cohorts["harder_transfer"]
    retention = cohorts["retention_arithmetic"] + cohorts["retention_spatial"]
    expected = set(iid + hard + retention)
    baseline, checkpoints = result["baseline"], result["checkpoints"]
    if len(checkpoints) != 3 or sorted(c["step"] for c in checkpoints) != [10, 11, 12]:
        raise ValueError("Exactly the prospective final-three evaluations required")
    for rows in [baseline] + [c["rows"] for c in checkpoints]:
        if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
            raise ValueError("Complete fixed DEV coverage required")
        for row in rows:
            expected_task = ("retention_arithmetic" if row["id"] in cohorts["retention_arithmetic"] else
                             "retention_spatial" if row["id"] in cohorts["retention_spatial"] else "modular_state")
            expected_purpose = "retention" if row["id"] in retention else "acquisition"
            if (row["task"], row["purpose"]) != (expected_task, expected_purpose):
                raise ValueError("Frozen cohort task/purpose changed; retention tasks cannot be pooled")
    def endpoint(ids):
        return last_three_endpoint(_cohort(baseline, ids + retention),
            [{"step": c["step"], "rows": _cohort(c["rows"], ids + retention)} for c in checkpoints])
    primary, transfer = endpoint(iid), endpoint(hard)
    final = next(c["rows"] for c in checkpoints if c["step"] == 12)
    replay = behavioral_metrics(_cohort(baseline, iid + retention), _cohort(final, iid + retention))
    replay_tasks = [t for t in replay["tasks"] if t["purpose"] == "retention"]
    adequate_source = all(t["initial_correct"] >= 4 for t in replay_tasks)
    retention_margin_met = adequate_source and all(t["damages"] <= 1 for t in replay_tasks)
    iid_detail = []
    for checkpoint in sorted(checkpoints, key=lambda c: c["step"]):
        metrics = behavioral_metrics(_cohort(baseline, iid + retention), _cohort(checkpoint["rows"], iid + retention))
        iid_detail.extend(t for t in metrics["tasks"] if t["purpose"] == "acquisition")
    new_iid = mean(t["repairs"] for t in iid_detail)
    acquisition_met = primary["acquisition_pp"] > 0 and new_iid >= 2
    steps = result["steps"]
    if len(steps) != 12 or [s["step"] for s in steps] != list(range(1, 13)):
        raise ValueError("All twelve attempted updates must remain accounted")
    if any(not isinstance(s.get(k), (int, float)) or isinstance(s[k], bool)
           or not math.isfinite(s[k]) or s[k] < 0 for s in steps for k in ("update_l2", "preclip_gradient_norm")):
        raise ValueError("Finite actual update/gradient accounting required")
    costs = result["all_forward_counts_including_eval"]
    return {"arm_id": result["arm_id"], "iid_acquisition_pp": primary["acquisition_pp"],
            "mean_new_iid_successes": new_iid, "harder_transfer_pp": transfer["acquisition_pp"],
            "harder_final_detail": transfer["final_detail"], "retention_pp": primary["retention_pp"],
            "retention_tasks": replay_tasks, "adequate_source_success_denominators": adequate_source,
            "retention_margin_met": retention_margin_met, "acquisition_condition_met": acquisition_met,
            "measured_tradeoff_condition_met": acquisition_met and retention_margin_met,
            "cumulative_update_l2": sum(s["update_l2"] for s in steps),
            "preclip_gradient_norms": [s["preclip_gradient_norm"] for s in steps],
            "cost": costs, "iid_steps": [10, 11, 12], "promotion_allowed": False,
            "interpretation": "A retention margin alone is insufficient; harder transfer and update/cost accounting remain separate. DEV only."}


def paired_screen(results, packet):
    """Require six exact-source paired cells; no independent-draw inference."""
    from .proposal import TINY_ARMS, validate_packet
    validate_packet(packet)
    expected = {(seed, arm.name) for seed in packet["design"]["seeds"] for arm in TINY_ARMS}
    if set(results) != expected:
        raise ValueError("Account every prospective seed/arm; missing cells are not dropped")
    first = next(iter(results.values()))
    source_hash = first["initial_parameters_sha256"]
    reference_baseline = sorted(first["baseline"], key=lambda r: r["id"])
    code_runtime = (first["identity"]["code_sha256"], first["identity"]["runtime"])
    cells = {}
    for (seed, arm), result in results.items():
        identity = result["identity"]
        expected_arm = next(a for a in TINY_ARMS if a.name == arm)
        if (result["arm_id"] != arm or result["initial_parameters_sha256"] != source_hash
                or result.get("arm") != vars(expected_arm)
                or identity["config_sha256"] != packet["design"]["config_sha256_by_seed"][str(seed)]
                or identity["checkpoint_sha256"] != packet["source"]["checkpoint_sha256"]
                or identity["dataset_sha256"] != packet["data"]["bindings"]["screen.jsonl"]["sha256"]
                or identity["code_sha256"] != packet["code_sha256"]
                or (identity["code_sha256"], identity["runtime"]) != code_runtime):
            raise ValueError("Native cells did not share exact reviewed source/data/code/runtime")
        for records in [result["baseline"]] + [c["rows"] for c in result["checkpoints"]]:
            for row in records:
                frozen = packet["data"]["row_identity"].get(row["id"])
                if frozen is None or any(row[k] != frozen[k] for k in ("parent_id", "task", "purpose")):
                    raise ValueError("Frozen per-row evaluation identity changed")
        if sorted(result["baseline"], key=lambda r: r["id"]) != reference_baseline:
            raise ValueError("Source baseline changed; no candidate-dependent retention denominator")
        if [s["row_id"] for s in result["steps"]] != packet["design"]["train_row_streams"][str(seed)]:
            raise ValueError("Paired ordered acquisition exposure changed")
        cells[(seed, arm)] = summarize_cell(result, packet["data"]["cohorts"])
    contrasts = []
    for seed in packet["design"]["seeds"]:
        left, right = (cells[(seed, a.name)] for a in TINY_ARMS[:2])
        acquisition_matched = right["iid_acquisition_pp"] >= left["iid_acquisition_pp"]
        weaker_update = right["cumulative_update_l2"] < left["cumulative_update_l2"]
        left_damages = sum(t["damages"] for t in left["retention_tasks"])
        right_damages = sum(t["damages"] for t in right["retention_tasks"])
        contrasts.append({"seed": seed, "right_minus_left": {
            key: right[key] - left[key] for key in ("iid_acquisition_pp", "harder_transfer_pp", "retention_pp", "cumulative_update_l2")},
            "right_acquisition_at_least_left": acquisition_matched, "right_update_is_smaller": weaker_update,
            "right_less_source_damage": right_damages < left_damages,
            "right_measured_tradeoff_improvement": (right["measured_tradeoff_condition_met"] and acquisition_matched
                                                    and right_damages < left_damages),
            "attenuation_confounded_or_inconclusive": weaker_update and not acquisition_matched,
            "interpretation": "Attenuation-confounded/inconclusive if less damage accompanies smaller update without acquisition match; no causal retention claim from update weakness."})
    return {"status": "EXPLORATORY_DEV_COMPARISON", "cells": [dict(value, seed=seed) for (seed, _), value in cells.items()],
            "paired_rollout_contrasts": contrasts, "independent_training_draws": 1,
            "paired_order_seeds": len(packet["design"]["seeds"]), "promotion_allowed": False,
            "limitations": "No seed/row pooling as independent evidence, no broad retention or SFT superiority claim; one SFT LR."}
