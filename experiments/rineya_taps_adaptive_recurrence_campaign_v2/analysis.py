"""Deterministic raw-journal analysis of the frozen original native DEV screen.

No model imports, protected input paths, outcome-driven scorer changes or speed
claims. Partial matrices retain observed diagnostics but suppress comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
GRID = (.8, .9, 1., 1.1, 1.2)
SEEDS = (17, 29)
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_journal(path):
    if not path.is_file():
        return [], [f"missing journal: {path.name}"]
    rows, errors = [], []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("journal row is not an object")
            rows.append(row)
        except (ValueError, TypeError) as error:
            errors.append(f"{path.name} line {index}: {type(error).__name__}: {error}")
    return rows, errors


def policy(row):
    if row.get("arm") == "A2":
        return f"A2:{float(row['fixed_alpha']):g}" if number(row.get("fixed_alpha")) else "MALFORMED_A2"
    return str(row.get("arm", "MISSING_ARM"))


def token_count(row):
    return len(row["token_ids"]) if isinstance(row.get("token_ids"), list) else 0


def cell(row):
    return (row.get("split"), row.get("seed"), row.get("case_id"), policy(row), row.get("extra_loops"))


def expected_cells(metadata):
    result = set()
    ordinary = [("A0", 0), ("A1", 1)] + [(f"A2:{alpha:g}", 1) for alpha in GRID] + [("A3", 1), ("A5", 1)]
    for item in metadata:
        if item["split"] == "DEV":
            for seed in SEEDS:
                result.update(("DEV", seed, item["case_id"], arm, extra) for arm, extra in ordinary)
        elif item["split"] == "SMOKE":
            result.update(("SMOKE", 17, item["case_id"], arm, extra) for arm, extra in ordinary)
            result.update(("SMOKE", 17, item["case_id"], arm, extra)
                          for arm in ("qualification_legacy", "qualification_new") for extra in (0, 1))
        else:
            raise ValueError("metadata permits only DEV and SMOKE")
    return result


def known_sum(rows, field):
    values = [row.get(field) for row in rows]
    known = [value for value in values if number(value)]
    return {"complete_value": sum(known) if known and len(known) == len(rows) else None,
            "known_subtotal": sum(known) if known else None,
            "missing_rows": len(rows) - len(known), "measured_rows": len(known), "rows": len(rows)}


def distribution(values):
    values = sorted(value for value in values if number(value))
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "median": None}
    return {"n": len(values), "mean": mean(values), "min": values[0], "max": values[-1],
            "median": statistics.median(values)}


def grid_headroom(adaptive, best_fixed, oracle, complete):
    if not complete:
        return {"headroom": None, "capture": None, "status": "INCOMPLETE_MATRIX"}
    headroom = oracle - best_fixed
    return {"headroom": headroom,
            "capture": (adaptive - best_fixed) / headroom if headroom > TOL else None,
            "status": "FINITE_CONSTANT_GRID_ONLY" if headroom > TOL else "NO_POSITIVE_CONSTANT_GRID_HEADROOM"}


def validate_lineage(admission, admission_sha, review, source_manifest, binding_meta,
                     binding_sha, protocol_sha, registration, preflight, selection,
                     trajectories, expected_config, expected_taps):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(admission.get("status") == "READY_FOR_REFERENCE_DISPATCH", "run admission missing/not ready")
    require(review.get("status") == "PASS_REFERENCE_DISPATCH" and review.get("admission_sha256") == admission_sha,
            "independent run review does not bind admission")
    require(admission.get("binding_sha256") == binding_sha, "run admission differs from exact metadata binding")
    require(admission.get("checkpoint_files") == binding_meta.get("checkpoint_files")
            and admission.get("source") == binding_meta.get("source_path"), "run checkpoint/source identity differs")
    require(admission.get("source_runtime") == binding_meta.get("runtime"), "run exact runtime binding differs")
    require(admission.get("maximum_shapes") == binding_meta.get("shapes"), "run shape binding differs")
    require(admission.get("training") is False and admission.get("native_test") is False
            and admission.get("gpu_lane") is False, "original reference lane authority differs")
    dispatch = admission.get("dispatch_hashes", {})
    protocol_key = PACKAGE.relative_to(REPO).as_posix() + "/PREREGISTRATION.md"
    require(dispatch.get(protocol_key) == protocol_sha, "run primary protocol binding differs")
    require(bool(source_manifest.get("head")) and bool(source_manifest.get("sha256")), "run source manifest absent")
    for relative, digest in dispatch.items():
        if relative.endswith(".py"):
            require(source_manifest.get("sha256", {}).get(relative) == digest,
                    "admission/source-manifest code mismatch: " + relative)
    for split, name in (("DEV", "mergedev_fast.jsonl"), ("SMOKE", "supa_layer_ablation_smoke.jsonl")):
        key = "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/" + name
        require(dispatch.get(key) == binding_meta.get("datasets", {}).get(split, {}).get("sha256"),
                "admitted data hash differs: " + split)
    require(registration.get("base") == "03db218ceebc61a2bc04f2482cf7e094045e01c1"
            and registration.get("grid") == list(GRID) and registration.get("seeds") == list(SEEDS)
            and registration.get("native_layers") == [12, 13] and registration.get("extra_loops") == [0, 1],
            "native registered source/operator/grid/seed contract differs")
    require(registration.get("config") == expected_config and registration.get("taps") == expected_taps,
            "native generation/controller configuration differs from frozen reference")
    dev_sha = binding_meta.get("datasets", {}).get("DEV", {}).get("sha256")
    require(registration.get("dataset_sha256") == dev_sha, "native registration DEV data mismatch")
    require(registration.get("training") is False and registration.get("gold_nll_status") == "NOT_COLLECTED_CAUSAL_GUARD",
            "native gold/training closure differs")
    require(preflight.get("status") == "METADATA_READY_NOT_QUALIFIED" and preflight.get("reasons") == [],
            "native preflight missing/blocked")
    require(preflight.get("historical_receipt_sha256") == binding_meta.get("historical_receipt_sha256"),
            "native controlling historical receipt differs")
    runtime = binding_meta.get("runtime", {})
    require(all(preflight.get("resources", {}).get(key) == runtime.get(key) for key in ("python", "torch")),
            "native preflight exact runtime mismatch")
    require(selection.get("config") == expected_taps and selection.get("dataset_sha256") == dev_sha
            and selection.get("native_TEST_authorized") is False, "frozen fixed/pool configuration or data binding differs")
    pool = [min(1.2, max(.8, row["alpha"])) for row in trajectories
            if row.get("arm") == "A3" and row.get("split") == "DEV" and row.get("iteration", -1) > 0
            and number(row.get("alpha"))] or [1.]
    require(selection.get("pool") == pool, "frozen random pool differs from generation-only DEV A3 journal")
    return errors


def validate_traces(physical, trajectories):
    """Join chronological per-evaluation chunks; v1 traces omit fixed_alpha.

    Every new-wrapper evaluation starts at forward_index1/iteration0. Preserve
    order to identify A2 first-pass rows, whose unit alpha alone is ambiguous.
    """
    errors, chunks = [], []
    for event in trajectories:
        if event.get("forward_index") == 1 and event.get("iteration") == 0:
            chunks.append([])
        if not chunks:
            errors.append("trajectory starts without an evaluation boundary")
            continue
        chunks[-1].append(event)
    new_rows = [row for row in physical if row.get("arm") != "qualification_legacy"]
    if len(chunks) != len(new_rows):
        errors.append("trajectory evaluation coverage differs from physical new-wrapper rows")
    for row, chunk in zip(new_rows, chunks):
        calls = token_count(row)
        expected = [(index, iteration) for index in range(1, calls + 1)
                    for iteration in range(1 + (row["extra_loops"] if type(row.get("extra_loops")) is int else 0))]
        actual = [(event.get("forward_index"), event.get("iteration")) for event in chunk]
        if actual != expected:
            errors.append("trajectory forward/extra-update coverage differs: " + str(row.get("case_id")))
        for event in chunk:
            if any(event.get(key) != row.get(key) for key in ("case_id", "split", "seed", "arm", "extra_loops")):
                errors.append("trajectory/outcome provenance differs")
            alpha = event.get("alpha")
            if not number(alpha) or not .8 - 1e-6 <= alpha <= 1.2 + 1e-6:
                errors.append("trajectory alpha missing/nonfinite/out of bounds")
            elif event.get("iteration") == 0 and abs(alpha - 1) > 1e-6:
                errors.append("ordinary first-pass trajectory was scaled")
            elif row.get("arm") == "A2" and event.get("iteration") == 1 and abs(alpha - row.get("fixed_alpha", -9)) > 1e-6:
                errors.append("fixed-alpha trajectory differs from its outcome arm")
    return errors


def summarize(rows, trajectories, metadata, selection, native_receipt, worker_receipt,
              resources=(), budget_events=(), read_errors=(), practical_margin=.05):
    if practical_margin != .05:
        raise ValueError("the registered .05 margin cannot be changed by analysis")
    ids = {item["case_id"]: item for item in metadata}
    if len(ids) != len(metadata):
        raise ValueError("duplicate metadata case IDs")
    physical, reused, errors = [], [], list(read_errors)
    for row in rows:
        if row.get("physical_run") is True:
            physical.append(row)
        elif row.get("physical_run") is False and row.get("arm") == "A4_example":
            reused.append(row)
        else:
            errors.append("unknown physical/reuse classification")
        if row.get("case_id") not in ids:
            errors.append("outcome case absent from admitted metadata")
    keyed = defaultdict(list)
    for row in physical:
        try:
            keyed[cell(row)].append(row)
        except (KeyError, TypeError, ValueError):
            errors.append("malformed physical cell key")
    expected = expected_cells(metadata)
    missing = sorted(expected - set(keyed), key=repr)
    unexpected = sorted(set(keyed) - expected, key=repr)
    duplicates = [key for key, group in keyed.items() if len(group) != 1]
    if unexpected or duplicates:
        errors.append("unexpected or duplicate physical cells")
    unsafe = [row for row in physical if row.get("safety_stopped") is True]
    invalid_scores = [row for row in physical if row.get("score") not in (0, 1)]
    if invalid_scores:
        errors.append("nonbinary or missing registered task score")
    if any(row.get("invalid") not in (True, False) or not isinstance(row.get("token_ids"), list)
           or not isinstance(row.get("response"), str) or not isinstance(row.get("termination"), str)
           for row in physical):
        errors.append("required native generation/validity fields missing")
    if any(ids[row["case_id"]].get("format_scorer") is not None and not number(row.get("format_score"))
           for row in physical if row.get("case_id") in ids):
        errors.append("required format score missing")
    if any(row.get("gold_nll") is not None or row.get("gold_answer") is not None for row in physical):
        errors.append("disabled full-gold diagnostic appears in native outcome journal")
    errors.extend(validate_traces(physical, trajectories))
    terminal_ok = (native_receipt.get("status") == "DEV_COMPLETE_NOT_TESTED"
                   and native_receipt.get("source_unchanged") is True
                   and native_receipt.get("checkpoint_files_unchanged") is True
                   and native_receipt.get("trainable_parameters") == 0
                   and native_receipt.get("training") is False
                   and worker_receipt.get("status") == "WORKER_TERMINAL"
                   and worker_receipt.get("exit_code") == 0)
    matrix_complete = not missing and not unexpected and not duplicates and not unsafe and not errors
    dev_ids = [item["case_id"] for item in metadata if item["split"] == "DEV"]
    grouped = defaultdict(list)
    for row in physical:
        if row.get("split") == "DEV" and row.get("score") in (0, 1) and row.get("case_id") in ids:
            grouped[(policy(row), row["case_id"])].append(row)
    items = {}
    repeat_divergence = []
    for (arm, case_id), group in grouped.items():
        scores = [row["score"] for row in group]
        if len({tuple(row["token_ids"]) if isinstance(row.get("token_ids"), list) else () for row in group}) > 1:
            repeat_divergence.append({"policy": arm, "case_id": case_id})
        formats = [row["format_score"] for row in group if number(row.get("format_score"))]
        items[(arm, case_id)] = {"case_id": case_id, "family": ids[case_id]["family"],
                                "cluster": ids[case_id]["independent_cluster"], "repeats": len(group),
                                "score": mean(scores), "format_score": mean(formats),
                                "invalid_rate": mean(int(row["invalid"]) for row in group if "invalid" in row),
                                "literal_UNKNOWN_rate": mean(int(row.get("response", "").strip() == "UNKNOWN") for row in group),
                                "literal_UNKNOWN_definition": "response.strip() == 'UNKNOWN'",
                                "semantic_abstention": "NOT_ASSESSED",
                                "terminations": dict(Counter(row.get("termination", "MISSING") for row in group)),
                                "rows": group}
    if repeat_divergence:
        errors.append("deterministic nominal-seed repeats differ in generated token IDs")
    observed = {arm: mean(item["score"] for (name, _), item in items.items() if name == arm)
                for arm in ("A0", "A1", "A3", "A5", *(f"A2:{alpha:g}" for alpha in GRID))}
    fixed = {alpha: observed[f"A2:{alpha:g}"] for alpha in GRID}
    envelope_alpha = max(GRID, key=lambda alpha: (fixed[alpha], -alpha)) if all(number(x) for x in fixed.values()) else None
    frozen_alpha = selection.get("alpha")
    freeze_valid = frozen_alpha in GRID and envelope_alpha == frozen_alpha
    if not freeze_valid:
        errors.append("missing/mismatched frozen DEV-selected fixed alpha")
    oracle_expected = {(seed, case_id) for seed in SEEDS for case_id in dev_ids}
    oracle_keys = [(row.get("seed"), row.get("case_id")) for row in reused]
    oracle_valid = set(oracle_keys) == oracle_expected and len(oracle_keys) == len(oracle_expected)
    oracle_scores = defaultdict(list)
    if matrix_complete:
        for seed, case_id in oracle_expected:
            fixed_rows = [keyed[("DEV", seed, case_id, f"A2:{alpha:g}", 1)][0] for alpha in GRID]
            best = max(fixed_rows, key=lambda row: (row["score"], -float(row["fixed_alpha"])))
            oracle_scores[case_id].append(best["score"])
            declared = [row for row in reused if row.get("seed") == seed and row.get("case_id") == case_id]
            if len(declared) != 1 or any(declared[0].get(k) != best.get(k) for k in ("score", "fixed_alpha", "token_ids")):
                oracle_valid = False
    if not oracle_valid:
        errors.append("missing/incorrect nonphysical constant-oracle rows")
    complete = matrix_complete and terminal_ok and freeze_valid and oracle_valid and not errors
    oracle_accuracy = mean(mean(scores) for scores in oracle_scores.values()) if complete else None
    best_accuracy = fixed[envelope_alpha] if complete else None
    headroom = grid_headroom(observed["A3"], best_accuracy, oracle_accuracy, complete)
    tables = []
    for arm in ("A0", "A1", "A3", "A5", *(f"A2:{alpha:g}" for alpha in GRID)):
        arm_items = [item for (name, _), item in items.items() if name == arm]
        tables.append({"policy": arm, "case_count": len(arm_items),
                       "accuracy": observed[arm] if complete else None,
                       "observed_partial_accuracy": observed[arm],
                       "status": "COMPLETE_EXPOSED_DEV" if complete else "PARTIAL_OBSERVED_ONLY",
                       "format_score": mean(item["format_score"] for item in arm_items if number(item["format_score"])),
                       "literal_UNKNOWN_rate": mean(item["literal_UNKNOWN_rate"] for item in arm_items),
                       "semantic_abstention": "NOT_ASSESSED",
                       "invalid_rate": mean(item["invalid_rate"] for item in arm_items if number(item["invalid_rate"])),
                       "termination_counts": dict(sum((Counter(item["terminations"]) for item in arm_items), Counter()))})
    pairs = {}
    if complete:
        contrasts = [("frozen_fixed", "A3", f"A2:{frozen_alpha:g}"), ("intact", "A3", "A0"),
                     ("unit", "A3", "A1"), ("legacy_A5", "A3", "A5"), ("fixed_1.2", "A3", "A2:1.2"),
                     ("H1_fixed_vs_unit", f"A2:{frozen_alpha:g}", "A1")]
        for name, treatment, comparator in contrasts:
            per_case, family_rows, cluster_rows = [], defaultdict(list), defaultdict(list)
            for case_id in dev_ids:
                a, b = items[(treatment, case_id)], items[(comparator, case_id)]
                delta = a["score"] - b["score"]
                row = {"case_id": case_id, "family": a["family"], "cluster": a["cluster"],
                       "treatment_score": a["score"], "A3_score": a["score"] if treatment == "A3" else None,
                       "comparator_score": b["score"], "delta": delta,
                       "format_delta": a["format_score"] - b["format_score"] if number(a["format_score"]) and number(b["format_score"]) else None,
                       "invalid_delta": a["invalid_rate"] - b["invalid_rate"] if number(a["invalid_rate"]) and number(b["invalid_rate"]) else None,
                       "treatment_literal_UNKNOWN_rate": a["literal_UNKNOWN_rate"],
                       "A3_literal_UNKNOWN_rate": a["literal_UNKNOWN_rate"] if treatment == "A3" else None,
                       "comparator_literal_UNKNOWN_rate": b["literal_UNKNOWN_rate"],
                       "semantic_abstention": "NOT_ASSESSED", "unsupported_answers": "NOT_ASSESSED"}
                per_case.append(row)
                family_rows[a["family"]].append(row)
                cluster_rows[a["cluster"]].append(delta)
            families = {family: {"items": len(group), "mean_delta": mean(row["delta"] for row in group),
                                "wins": [row["case_id"] for row in group if row["delta"] > TOL],
                                "losses": [row["case_id"] for row in group if row["delta"] < -TOL]}
                        for family, group in sorted(family_rows.items())}
            score_delta = mean(row["delta"] for row in per_case)
            observed_preservation = (all(row["delta"] >= -TOL for row in per_case)
                                     and all(row["format_delta"] is None or row["format_delta"] >= -TOL for row in per_case)
                                     and all(row["invalid_delta"] is None or row["invalid_delta"] <= TOL for row in per_case))
            pairs[name] = {"treatment": treatment, "comparator": comparator, "score_delta": score_delta,
                           "meets_registered_exploratory_margin": score_delta >= practical_margin - TOL,
                           "wins": [row["case_id"] for row in per_case if row["delta"] > TOL],
                           "losses": [row["case_id"] for row in per_case if row["delta"] < -TOL],
                           "source_correct_cases": sum(row["comparator_score"] == 1 for row in per_case),
                           "source_correct_retained": sum(row["comparator_score"] == 1 and row["treatment_score"] == 1 for row in per_case),
                           "observed_score_format_invalid_preservation": observed_preservation,
                           "complete_semantic_preservation": "NOT_ASSESSED",
                           "per_case": per_case, "families": families,
                           "provisional_cluster_deltas": {cluster: mean(values) for cluster, values in sorted(cluster_rows.items())},
                           "cluster_macro_delta_secondary": mean(mean(values) for values in cluster_rows.values()),
                           "confirmatory_p_value": None, "confidence_interval": None,
                           "uncertainty_status": "small exposed provisional clusters; independence unverified; DEV selection prevents confirmation"}
    features = [row for row in trajectories if row.get("arm") == "A3" and row.get("split") == "DEV"
                and row.get("iteration", -1) > 0 and row.get("case_id") in ids]
    fields = ("alpha", "balance", "progress", "fluctuation", "delta_norm", "delta_cosine", "previous_ema_cosine", "ema_cosine", "movement", "state_norm")
    feature_distribution = {field: distribution(row.get(field) for row in features) for field in fields}
    movements = [row["movement"] / max(row["state_norm"], 1e-12) for row in features
                 if number(row.get("movement")) and number(row.get("state_norm"))]
    feature_distribution["normalized_movement"] = distribution(movements)
    feature_distribution["normalized_raw_update"] = distribution(
        row["delta_norm"] / max(row["state_norm"], 1e-12) for row in features
        if number(row.get("delta_norm")) and number(row.get("state_norm")))
    alphas = [row["alpha"] for row in features if number(row.get("alpha"))]
    feature_cases = defaultdict(list)
    for row in features:
        feature_cases[row["case_id"]].append(row)
    per_item_features = {case_id: {"family": ids[case_id]["family"], "prefix_observations_including_repeats": len(group),
                                 "mean_alpha": mean(row["alpha"] for row in group if number(row.get("alpha"))),
                                 "mean_balance": mean(row["balance"] for row in group if number(row.get("balance")))}
                         for case_id, group in sorted(feature_cases.items())}
    physical_cost = {field: known_sum(physical, field) for field in
                     ("wall_seconds", "controller_seconds", "controller_logging_seconds",
                      "full_model_calls_including_gold_nll", "processed_token_positions",
                      "total_token_positions_including_gold_nll", "generation_dense_matmul_flops_estimate")}
    cost_arms = {arm: {"physical_evaluations": len(group), "wall_seconds": known_sum(group, "wall_seconds")}
                 for arm, group in ((arm, [row for row in physical if policy(row) == arm])
                                    for arm in sorted({policy(row) for row in physical}))}
    completed_budget = [event for event in budget_events if event.get("event") == "native_completed"]
    budget_runs = [event.get("run") for event in completed_budget]
    duplicate_budget = len(set(budget_runs)) != len(budget_runs)
    cumulative = (sum(event["native_worker_seconds"] for event in completed_budget)
                  if completed_budget and not duplicate_budget and all(number(event.get("native_worker_seconds")) for event in completed_budget) else None)
    rss = [row["rss_bytes"] for row in resources if number(row.get("rss_bytes"))]
    if number(worker_receipt.get("peak_rss_bytes")):
        rss.append(worker_receipt["peak_rss_bytes"])
    clusters = Counter(ids[case_id]["independent_cluster"] for case_id in dev_ids)
    floors = {}
    for split in ("DEV", "SMOKE"):
        bank = [item for item in metadata if item["split"] == split]
        if bank:
            constants = {answer: mean(item["audit"]["constant_answer_scores"][answer] for item in bank)
                         for answer in bank[0]["audit"]["constant_answer_scores"]}
            floors[split] = {"items": len(bank), "prompt_copy_accuracy": mean(item["audit"]["prompt_copy_score"] for item in bank),
                             "constant_accuracy": constants, "best_tested_constant_accuracy": max(constants.values()),
                             "prediction_input": "prompt only or fixed literal; scorer targets used only to evaluate",
                             "query_blind": "NOT_IMPLEMENTED", "positive_solver": "NOT_VERIFIED"}
    return {"status": "COMPLETE_EXPLORATORY_REFERENCE_ONLY" if complete else "INCOMPLETE_NO_PRIMARY_COMPARISON",
            "primary_endpoint": "inherited binary task score; equal item weights after within-item repeat averaging",
            "practical_margin": practical_margin, "complete": complete, "matrix_cells_complete": matrix_complete,
            "terminal_status_valid": terminal_ok, "errors": errors, "missing_cells": missing,
            "unexpected_cells": unexpected, "duplicate_cells": duplicates,
            "unsafe_rows": len(unsafe), "deterministic_repeat_divergence": repeat_divergence,
            "physical_rows": len(physical), "reused_oracle_rows": len(reused), "expected_physical_rows": len(expected),
            "registered_seeds": list(SEEDS), "seeds_are_independent_examples": False,
            "dev_items": len(dev_ids), "provisional_cluster_count": len(clusters), "cluster_sizes": dict(clusters),
            "generator_independence": "NOT_ESTABLISHED", "tables": tables,
            "fixed_alpha_curve": [{"alpha": alpha, "accuracy": fixed[alpha] if complete else None,
                                   "observed_partial_accuracy": fixed[alpha]} for alpha in GRID],
            "frozen_fixed_comparator": {"alpha": frozen_alpha, "accuracy": fixed.get(frozen_alpha) if complete else None,
                                        "selection_surface": "same exposed DEV; frozen before A5, not independent validation",
                                        "raw_selection_verified": freeze_valid},
            "retrospective_global_fixed_envelope": {"alpha": envelope_alpha, "accuracy": best_accuracy,
                                                    "diagnostic_only": True},
            "constant_oracle": {"accuracy": oracle_accuracy, "physical_run": False, "scope": "per-example constant finite grid only",
                                "reported_rows_verified": oracle_valid, **headroom},
            "per_step_oracle": "NOT_ENUMERATED_NATIVE_PREFIX_SCHEDULES", "pairs": pairs,
            "features": {"prefix_weighted": feature_distribution, "per_item": per_item_features,
                         "normalized_movement_definition": "controlled movement / max(state_norm,1e-12)",
                         "normalized_raw_update_definition": "delta_norm / max(state_norm,1e-12)",
                         "lower_saturation_fraction": mean(int(alpha <= .8 + 1e-6) for alpha in alphas),
                         "upper_saturation_fraction": mean(int(alpha >= 1.2 - 1e-6) for alpha in alphas),
                         "pool_weighting": "generated prefix observations with nominal deterministic repeats; long generations contribute more",
                         "predictive_validation": "NOT_RUN"},
            "costs": {"physical_row_sums": physical_cost, "per_arm": cost_arms,
                      "reused_oracle_additional_generation_cost": 0,
                      "reused_oracle_cost_note": "copied row timings/counters excluded; generation already charged to A2",
                      "generation_calls_inferred_from_completed_token_lists": sum(token_count(row) for row in physical) if all(isinstance(row.get("token_ids"), list) for row in physical) else None,
                      "initial_block_applications_inferred": sum(token_count(row) for row in physical) if all(isinstance(row.get("token_ids"), list) for row in physical) else None,
                      "extra_block_applications_inferred": sum(token_count(row) * row["extra_loops"] for row in physical) if all(type(row.get("extra_loops")) is int for row in physical) else None,
                      "block_width": 2, "inferred_count_note": "completed generation rows only; failed/unwritten forwards remain unmeasured",
                      "worker_wall_seconds": worker_receipt.get("native_worker_seconds"),
                      "supervisor_wall_seconds": worker_receipt.get("supervisor_wall_seconds"),
                      "peak_sampled_worker_tree_rss_bytes": max(rss) if rss else None,
                      "resource_samples": len(resources), "cumulative_native_worker_seconds": cumulative,
                      "remaining_native_seconds": max(0, 21600 - cumulative) if cumulative is not None else None,
                      "duplicate_budget_run_events": duplicate_budget,
                      "latency_scope": "instrumented diagnostic path; no deployable or matched-quality speed claim"},
            "own_bank_floors": floors, "legacy_A5": "per-example first draw repeated at every prefix; nominal config.seed stays17; H4 control incomplete",
            "semantic_correctness": "NOT_ASSESSED", "unsupported_answers": "NOT_ASSESSED",
            "independent_replication": "NOT_RUN", "native_TEST": "CLOSED", "efficacy_claim_supported": False}


def run(run_root, out, binding, issue_receipt):
    run_root, out, binding, issue_receipt = [Path(path).resolve(strict=path != out) for path in (run_root, out, binding, issue_receipt)]
    if any(PACKAGE not in path.parents for path in (run_root, out, binding, issue_receipt)):
        raise PermissionError("analysis paths must stay inside this campaign")
    if run_root == out or run_root in out.parents or out in run_root.parents:
        raise PermissionError("analysis output must not overlap the source run")
    if binding != (PACKAGE / "runs/binding-001/PROMPT_AUDIT.json").resolve():
        raise PermissionError("original reference analyzer requires the exact binding-001 prompt packet")
    out.mkdir(parents=True, exist_ok=False)
    terminal = {"status": "FAIL", "native_model_loads": 0, "started_utc": datetime.now(timezone.utc).isoformat()}
    inputs = [binding, issue_receipt, PACKAGE / "PREREGISTRATION.md", PACKAGE / "BUDGET_LEDGER.jsonl"]
    try:
        protocol = (PACKAGE / "PREREGISTRATION.md").read_text(encoding="utf-8")
        if "Practical exploratory improvement margin: .05" not in protocol:
            raise RuntimeError("registered primary margin text changed; review analyzer contract")
        metadata = json.loads(binding.read_text(encoding="utf-8"))
        issue = json.loads(issue_receipt.read_text(encoding="utf-8"))
        result = run_root / "result"
        rows, errors = read_journal(result / "outcomes.jsonl")
        traces, trace_errors = read_journal(result / "trajectories.jsonl")
        resources, resource_errors = read_journal(run_root / "resources.jsonl")
        budget, budget_errors = read_journal(PACKAGE / "BUDGET_LEDGER.jsonl")
        def optional(name):
            path = Path(name)
            if path.is_file():
                inputs.append(path)
                return json.loads(path.read_text(encoding="utf-8"))
            return {}
        selection = optional(result / "frozen_dev_selection.json")
        native = optional(result / "receipt.json")
        worker = optional(run_root / "terminal_receipt.json")
        for path in (result / "outcomes.jsonl", result / "trajectories.jsonl", run_root / "resources.jsonl"):
            if path.is_file():
                inputs.append(path)
        binding_meta = optional(binding.parent / "MODEL_DATA_BINDING.json")
        binding_receipt = optional(binding.parent / "receipt.json")
        if binding_receipt.get("prompt_audit_sha256") != sha(binding):
            raise RuntimeError("prompt packet differs from its tokenizer-binding receipt")
        if binding_receipt.get("binding_sha256") != sha(binding.parent / "MODEL_DATA_BINDING.json"):
            raise RuntimeError("model/data binding differs from its receipt")
        if binding_meta:
            if selection.get("dataset_sha256") != binding_meta["datasets"]["DEV"]["sha256"]:
                errors.append("frozen selection DEV dataset binding missing/mismatched")
        admission = optional(run_root / "admission.json")
        review = optional(run_root / "review.json")
        source_manifest = optional(run_root / "source_manifest.json")
        registration = optional(result / "registration.json")
        preflight = optional(result / "preflight.json")
        history_path = REPO / "experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json"
        reference_config_path = REPO / "experiments/rineya_taps_adaptive_recurrence_v1/configs/inference_v1.json"
        inputs.extend((history_path, reference_config_path))
        expected_config = dict(json.loads(history_path.read_text(encoding="utf-8"))["config"])
        expected_config["max_rss_gib"] = 6
        expected_taps = json.loads(reference_config_path.read_text(encoding="utf-8"))["controller"]
        if sha(history_path) != binding_meta.get("historical_receipt_sha256"):
            errors.append("current controlling historical receipt differs from model binding")
        lineage_errors = validate_lineage(
            admission, sha(run_root / "admission.json") if admission else None, review,
            source_manifest, binding_meta, sha(binding.parent / "MODEL_DATA_BINDING.json"),
            sha(PACKAGE / "PREREGISTRATION.md"), registration, preflight, selection, traces,
            expected_config, expected_taps)
        errors.extend(lineage_errors)
        scorer_path = REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/rineya_cross_scale/evaluation/scoring.py"
        inputs.append(scorer_path)
        if sha(scorer_path) != binding_meta.get("scorer_sha256"):
            errors.append("pinned pure scorer source hash differs")
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("campaign_analysis_pinned_scorer", scorer_path)
            scorer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scorer)
            admitted_items = {item["case_id"]: item for item in metadata}
            for row in rows:
                item = admitted_items.get(row.get("case_id"))
                if item is None or not isinstance(row.get("response"), str):
                    continue
                if scorer.score_response(row["response"], item["scorer"])["score"] != row.get("score"):
                    errors.append("stored task score differs from pinned scorer: " + row["case_id"])
                if item.get("format_scorer") is not None and scorer.score_response(row["response"], item["format_scorer"])["score"] != row.get("format_score"):
                    errors.append("stored format score differs from pinned scorer: " + row["case_id"])
        report = summarize(rows, traces, metadata, selection, native, worker, resources, budget,
                           errors + trace_errors + resource_errors + budget_errors)
        report["independent_issue_receipt"] = {"path": str(issue_receipt), "sha256": sha(issue_receipt),
                                               "issue_ids": [item["id"] for item in issue.get("issues", [])],
                                               "status": issue.get("status")}
        report["run_lineage"] = {"valid": not lineage_errors, "errors": lineage_errors,
                                 "admission_id": admission.get("id"), "source_head": source_manifest.get("head"),
                                 "frozen_pool_status": "generation-only DEV journal verified" if not lineage_errors else "NOT_VERIFIED"}
        manifest = {"analysis_source_sha256": sha(__file__),
                    "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                    "input_sha256": {str(path): sha(path) for path in inputs if path.is_file()},
                    "run_root": str(run_root), "native_model_loads": 0}
        for name, value in (("REPORT.json", report), ("MANIFEST.json", manifest)):
            with (out / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        lines = ["# Frozen native reference analysis", "", report["status"], "",
                 "Primary endpoint: inherited binary task score, with repeat averaging within case. Practical margin: .05.",
                 "Exposed development evidence; semantic correctness and independent replication remain unassessed.", "",
                 "| Policy | Accuracy | Observed items |", "|---|---:|---:|"]
        for row in report["tables"]:
            lines.append(f"| {row['policy']} | {row['accuracy']} | {row['case_count']} |")
        lines += ["", f"Finite constant-grid headroom: {report['constant_oracle']['headroom']}; capture: {report['constant_oracle']['capture']}.",
                  f"Physical evaluations: {report['physical_rows']}/{report['expected_physical_rows']}; reused oracle rows: {report['reused_oracle_rows']}.",
                  f"Provisional clusters: {report['provisional_cluster_count']}; generator independence is unverified.",
                  "Legacy A5 repeats one per-example draw across prefixes and duplicates nominal seed configurations; intended H4 remains incomplete.",
                  "Costs are from the instrumented diagnostic path; no deployment speed claim."]
        (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        terminal.update(status="ANALYZED", matrix_complete=report["complete"], report_sha256=sha(out / "REPORT.json"))
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        terminal["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with (out / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(terminal, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return 0 if terminal["status"] == "ANALYZED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binding", type=Path, default=PACKAGE / "runs/binding-001/PROMPT_AUDIT.json")
    parser.add_argument("--issue-receipt", type=Path, default=PACKAGE / "reviews/initial_issue_receipt.json")
    args = parser.parse_args()
    raise SystemExit(run(args.run_root, args.out, args.binding, args.issue_receipt))
