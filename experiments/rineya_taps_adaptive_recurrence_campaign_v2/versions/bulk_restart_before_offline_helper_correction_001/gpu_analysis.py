"""Explicit GPU FP32 successor analysis; no original-CPU guard substitution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import struct
import subprocess

from .analysis import (GRID, SEEDS, TOL, distribution, expected_cells, grid_headroom,
                       known_sum, mean, number, policy, read_journal, sha, token_count,
                       validate_traces)

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
GIB = 1024**3


def gpu_expected_cells(metadata):
    cells = expected_cells(metadata)
    cells.update(("SMOKE", 17, item["case_id"], arm, 1) for item in metadata if item["split"] == "SMOKE"
                 for arm in ("qualification_reference_A3", "qualification_successor_A3"))
    return cells


def key(row):
    return (row.get("split"), row.get("seed"), row.get("case_id"), policy(row), row.get("extra_loops"))


def validate_gpu_traces(physical, traces, pool):
    errors = validate_traces(physical, traces)
    for event in traces:
        if event.get("configured_seed") != event.get("seed"):
            errors.append("GPU trajectory configured seed differs from nominal replicate")
    random_groups = defaultdict(list)
    for event in traces:
        if event.get("arm") == "A5" and event.get("iteration") == 1:
            random_groups[(event.get("split"), event.get("seed"), event.get("case_id"))].append(event)
    for (_, seed, case_id), events in random_groups.items():
        if not pool:
            errors.append("corrected GPU A5 has no frozen pool")
            continue
        binding = json.dumps([seed, "A5", case_id], ensure_ascii=False, separators=(",", ":"))
        rng = random.Random(int.from_bytes(hashlib.sha256(binding.encode()).digest(), "big"))
        for index, event in enumerate(events, 1):
            alpha = struct.unpack("f", struct.pack("f", rng.choice(pool)))[0]
            if event.get("random_draw_index") != index or event.get("forward_index") != index or event.get("alpha") != alpha:
                errors.append("GPU A5 prefix RNG stream/pool identity differs")
    return errors


def forward_audit(physical, journal, metadata=()):
    errors, chunks = [], []
    prompt_tokens = {item["case_id"]: item.get("input_tokens") for item in metadata}
    ordinary = [event for event in journal if event.get("event", "").startswith("FORWARD_")]
    for event in ordinary:
        if event.get("event") == "FORWARD_STARTED" and event.get("full_model_call") == 1:
            chunks.append([])
        if not chunks:
            errors.append("GPU ordinary forward journal missing evaluation start")
        else:
            chunks[-1].append(event)
    if len(chunks) != len(physical):
        errors.append("GPU ordinary forward evaluation coverage differs")
    for row, chunk in zip(physical, chunks):
        calls = token_count(row)
        prompt = prompt_tokens.get(row.get("case_id"))
        if metadata and (type(prompt) is not int or prompt < 1):
            errors.append("GPU bound input-token count missing")
        expected = [(kind, index) for index in range(1, calls + 1)
                    for kind in ("FORWARD_STARTED", "FORWARD_COMPLETED")]
        if [(event.get("event"), event.get("full_model_call")) for event in chunk] != expected:
            errors.append("GPU ordinary forward start/completion sequence differs")
        if row.get("full_model_calls_including_gold_nll") != calls:
            errors.append("GPU outcome completed-forward count differs")
        for event in chunk:
            if any(event.get(field) != row.get(field) for field in ("case_id", "arm", "seed", "split", "extra_loops")):
                errors.append("GPU forward/outcome provenance differs")
            if type(prompt) is int and event.get("prefix_tokens") != prompt + event.get("full_model_call", 0) - 1:
                errors.append("GPU journal prefix length differs from bound prompt and call index")
        if type(prompt) is int:
            positions = calls * prompt + calls * (calls - 1) // 2
            if row.get("processed_token_positions") != positions or row.get("total_token_positions_including_gold_nll") != positions:
                errors.append("GPU outcome processed prefix-position sum differs")
            if row.get("initial_block_applications") != calls or row.get("extra_layer_applications") != 2 * calls * row.get("extra_loops", 0):
                errors.append("GPU actual initial/extra block application counters differ")
    maximum_starts = [event for event in journal if event.get("event") == "MAXIMUM_FORWARD_STARTED"]
    maximum_done = [event for event in journal if event.get("event") == "MAXIMUM_FORWARD_COMPLETED"]
    maximum_failed = [event for event in journal if event.get("event") == "MAXIMUM_FORWARD_FAILED"]
    cache_starts = [event for event in journal if event.get("event") == "CACHE_REJECTION_STARTED"]
    cache_rejected = [event for event in journal if event.get("event") == "CACHE_REJECTED"]
    # Six identity forwards, one forced partial, restored forward, eight arm
    # maxima, and one frozen-pool A5 maximum =17 attempted computations.
    if len(maximum_starts) != 17 or len(maximum_done) != 16 or len(maximum_failed) != 1:
        errors.append("GPU registered maximum-shape attempt matrix incomplete")
    if maximum_failed and ("EXPECTED_FORCED_RESTORATION_EXCEPTION" not in maximum_failed[0].get("error", "")
                           or maximum_failed[0].get("completed") is not False
                           or maximum_failed[0].get("partial_compute") != "UNKNOWN"):
        errors.append("GPU expected forced partial failure not honestly recorded")
    starts = {event.get("attempt") for event in maximum_starts}
    terminal = [event.get("attempt") for event in maximum_done + maximum_failed]
    if starts != set(terminal) or len(terminal) != len(set(terminal)):
        errors.append("GPU maximum attempt/terminal pairing differs")
    if starts != set(range(1, 18)) or any(event.get("prefix_tokens") != 2248 for event in maximum_starts + maximum_done + maximum_failed):
        errors.append("GPU maximum attempt IDs/prefix shapes differ")
    if len(cache_starts) != 2 or len(cache_rejected) != 2:
        errors.append("GPU both cache-rejection attempts not recorded")
    if {(event.get("use_cache"), event.get("cache_object_present")) for event in cache_starts} != {(True, False), (False, True)}:
        errors.append("GPU cache-rejection variants differ")
    if any(event.get("model_compute_started") is not False or event.get("wrapper_forward_count") != 0
           for event in cache_rejected):
        errors.append("GPU cache rejection did not precede model computation")
    completed = [event for event in journal if event.get("event") in ("FORWARD_COMPLETED", "MAXIMUM_FORWARD_COMPLETED")]
    measured_attempts = [event for event in journal if event.get("event") in
                         ("FORWARD_COMPLETED", "FORWARD_FAILED", "MAXIMUM_FORWARD_COMPLETED", "MAXIMUM_FORWARD_FAILED", "CACHE_REJECTED")]
    return errors, {"ordinary_attempts": sum(event.get("event") == "FORWARD_STARTED" for event in ordinary),
                    "ordinary_completed": sum(event.get("event") == "FORWARD_COMPLETED" for event in ordinary),
                    "ordinary_failed": sum(event.get("event") == "FORWARD_FAILED" for event in ordinary),
                    "maximum_attempts": len(maximum_starts), "maximum_completed": len(maximum_done),
                    "maximum_expected_partial_failures": len(maximum_failed), "cache_rejected_before_compute": len(cache_rejected),
                    "partial_failure_compute_positions": None,
                    "completed_prefix_positions_including_maximum_checks": known_sum(completed, "prefix_tokens"),
                    "completed_full_model_calls_including_maximum_checks": len(completed),
                    "completed_forward_seconds": known_sum(completed, "forward_seconds"),
                    "all_measured_attempt_seconds": known_sum(measured_attempts, "forward_seconds"),
                    "partial_compute_note": "partial forward cost unknown; elapsed and worker costs include it"}


def aggregate(rows, traces, metadata, selection, native, worker, qualification, a5_qualification,
              forwards, resources=(), budget=(), issues=()):
    errors = list(issues)
    admitted = {item["case_id"]: item for item in metadata}
    if len(admitted) != len(metadata):
        raise ValueError("duplicate metadata IDs")
    physical = [row for row in rows if row.get("physical_run") is True]
    reused = [row for row in rows if row.get("physical_run") is False and row.get("arm") == "A4_example"]
    if len(physical) + len(reused) != len(rows):
        errors.append("unclassified GPU physical/reuse outcome")
    indexed = defaultdict(list)
    for row in physical:
        indexed[key(row)].append(row)
        if row.get("case_id") not in admitted or row.get("configured_seed") != row.get("seed"):
            errors.append("GPU case/configured-seed provenance differs")
        if row.get("score") not in (0, 1) or row.get("gold_nll") is not None or row.get("gold_answer") is not None:
            errors.append("GPU binary endpoint or gold diagnostic closure differs")
        if row.get("safety_stopped") is not False:
            errors.append("GPU native safety stop/missing status")
    expected = gpu_expected_cells(metadata)
    missing = sorted(expected - set(indexed), key=repr)
    extra = sorted(set(indexed) - expected, key=repr)
    duplicates = [cell for cell, group in indexed.items() if len(group) != 1]
    if extra or duplicates:
        errors.append("GPU physical matrix extra/duplicate cells")
    pool = selection.get("pool", [])
    errors.extend(validate_gpu_traces(physical, traces, pool))
    forward_errors, forward_costs = forward_audit(physical, forwards, metadata)
    errors.extend(forward_errors)
    terminal_ok = (native.get("status") == "GPU_DEV_COMPLETE_NOT_TESTED" and native.get("source_unchanged") is True
                   and native.get("checkpoint_files_unchanged") is True and native.get("trainable_parameters") == 0
                   and native.get("training") is False and worker.get("status") == "WORKER_TERMINAL" and worker.get("exit_code") == 0)
    qualification_ok = (qualification.get("status") == "QUALIFIED_GPU_FP32_REFERENCE_OPERATOR"
                        and qualification.get("original_new_A3_identity") == "PASS"
                        and qualification.get("L0_L2_one_shot") == "PASS_EVERY_COMPLETED_FORWARD"
                        and qualification.get("scaled_arms_smoke") == "PASS"
                        and qualification.get("checkpoint_files_unchanged") is True
                        and qualification.get("before_state_sha256") == qualification.get("after_state_sha256")
                        and bool(qualification.get("after_state_sha256")))
    maximum = qualification.get("maximum_shape", [])
    qualification_ok &= (len(maximum) == 1 and maximum[0].get("maximum_prefix_tokens") == 2248
                         and maximum[0].get("identity") == "PASS" and maximum[0].get("forced_exception_restoration") == "PASS"
                         and maximum[0].get("cache_rejection") == "PASS")
    a5_ok = (a5_qualification.get("status") == "PASS_FROZEN_POOL_SMOKE_AND_MAXIMUM_SHAPE"
             and a5_qualification.get("actual_seed_binding") is True)
    if not terminal_ok or not qualification_ok or not a5_ok:
        errors.append("GPU terminal/native qualification/frozen A5 gate missing")
    if not qualification.get("mask_positions", {}).get("positions") or not qualification.get("mask_positions", {}).get("replay_conditioning"):
        errors.append("GPU native positions/conditioning qualification receipt missing")
    grouped = defaultdict(list)
    for row in physical:
        if row.get("split") == "DEV" and row.get("case_id") in admitted:
            grouped[(policy(row), row["case_id"])].append(row)
    repeated_divergence = []
    items = {}
    for (arm, case_id), group in grouped.items():
        if arm != "A5" and len({tuple(row.get("token_ids", [])) for row in group}) > 1:
            repeated_divergence.append({"policy": arm, "case_id": case_id})
        items[(arm, case_id)] = {"score": mean(row["score"] for row in group),
                                "format": mean(row["format_score"] for row in group if number(row.get("format_score"))),
                                "invalid": mean(int(row["invalid"]) for row in group if "invalid" in row),
                                "literal_UNKNOWN": mean(int(row.get("response", "").strip() == "UNKNOWN") for row in group),
                                "rows": group}
    if repeated_divergence:
        errors.append("GPU deterministic A0/A1/A2/A3 token repeats diverge")
    dev_ids = [item["case_id"] for item in metadata if item["split"] == "DEV"]
    policies = ["A0", "A1", "A3", "A5", *(f"A2:{alpha:g}" for alpha in GRID)]
    observed = {arm: mean(item["score"] for (name, _), item in items.items() if name == arm) for arm in policies}
    curve = {alpha: observed[f"A2:{alpha:g}"] for alpha in GRID}
    envelope = max(GRID, key=lambda alpha: (curve[alpha], -alpha)) if all(number(value) for value in curve.values()) else None
    frozen = selection.get("alpha")
    if frozen not in GRID or frozen != envelope:
        errors.append("GPU frozen fixed comparator differs from recomputed DEV selection")
    oracle_keys = [(row.get("seed"), row.get("case_id")) for row in reused]
    expected_oracle = {(seed, case_id) for seed in SEEDS for case_id in dev_ids}
    oracle_valid = len(oracle_keys) == len(expected_oracle) and set(oracle_keys) == expected_oracle
    oracle = defaultdict(list)
    if not missing and not extra and not duplicates:
        for seed, case_id in expected_oracle:
            rr = [indexed[("DEV", seed, case_id, f"A2:{alpha:g}", 1)][0] for alpha in GRID]
            best = max(rr, key=lambda row: (row["score"], -row["fixed_alpha"]))
            oracle[case_id].append(best["score"])
            declared = [row for row in reused if row.get("seed") == seed and row.get("case_id") == case_id]
            if len(declared) != 1 or any(declared[0].get(field) != best.get(field) for field in ("score", "fixed_alpha", "token_ids")):
                oracle_valid = False
    if not oracle_valid:
        errors.append("GPU nonphysical constant oracle is missing/not exact reuse")
    complete = not missing and not errors
    oracle_score = mean(mean(scores) for scores in oracle.values()) if complete else None
    best_score = curve[envelope] if complete else None
    h = grid_headroom(observed["A3"], best_score, oracle_score, complete)
    tables = [{"policy": arm, "accuracy": observed[arm] if complete else None,
               "observed_partial_accuracy": observed[arm], "case_count": sum(name == arm for name, _ in items),
               "status": "COMPLETE_EXPOSED_GPU_DEV" if complete else "PARTIAL_OBSERVED_ONLY",
               "literal_UNKNOWN_rate": mean(item["literal_UNKNOWN"] for (name, _), item in items.items() if name == arm),
               "literal_UNKNOWN_definition": "response.strip() == 'UNKNOWN'", "semantic_abstention": "NOT_ASSESSED",
               "format_score": mean(item["format"] for (name, _), item in items.items() if name == arm and number(item["format"])),
               "termination_counts": dict(Counter(row.get("termination", "MISSING") for row in physical if row.get("split") == "DEV" and policy(row) == arm))}
              for arm in policies]
    pairs = {}
    if complete:
        for label, treatment, comparator in (("frozen_fixed", "A3", f"A2:{frozen:g}"), ("unit", "A3", "A1"),
                                              ("intact", "A3", "A0"), ("corrected_A5", "A3", "A5"),
                                              ("fixed_1.2", "A3", "A2:1.2"), ("H1_fixed_vs_unit", f"A2:{frozen:g}", "A1")):
            case_rows, families, clusters = [], defaultdict(list), defaultdict(list)
            for case_id in dev_ids:
                a, b = items[(treatment, case_id)], items[(comparator, case_id)]
                row = {"case_id": case_id, "family": admitted[case_id]["family"],
                       "cluster": admitted[case_id]["independent_cluster"], "treatment_score": a["score"],
                       "comparator_score": b["score"], "delta": a["score"] - b["score"],
                       "format_delta": a["format"] - b["format"] if number(a["format"]) and number(b["format"]) else None,
                       "invalid_delta": a["invalid"] - b["invalid"] if number(a["invalid"]) and number(b["invalid"]) else None,
                       "treatment_literal_UNKNOWN_rate": a["literal_UNKNOWN"], "comparator_literal_UNKNOWN_rate": b["literal_UNKNOWN"],
                       "semantic_abstention": "NOT_ASSESSED", "unsupported_answers": "NOT_ASSESSED"}
                case_rows.append(row)
                families[row["family"]].append(row)
                clusters[row["cluster"]].append(row["delta"])
            delta = mean(row["delta"] for row in case_rows)
            pairs[label] = {"treatment": treatment, "comparator": comparator, "score_delta": delta,
                            "meets_registered_exploratory_margin": delta >= .05 - TOL,
                            "wins": [row["case_id"] for row in case_rows if row["delta"] > TOL],
                            "losses": [row["case_id"] for row in case_rows if row["delta"] < -TOL],
                            "source_correct_cases": sum(row["comparator_score"] == 1 for row in case_rows),
                            "source_correct_retained": sum(row["comparator_score"] == 1 and row["treatment_score"] == 1 for row in case_rows),
                            "observed_score_format_invalid_preservation": all(row["delta"] >= -TOL and
                                (row["format_delta"] is None or row["format_delta"] >= -TOL) and
                                (row["invalid_delta"] is None or row["invalid_delta"] <= TOL) for row in case_rows),
                            "complete_semantic_preservation": "NOT_ASSESSED", "per_case": case_rows,
                            "families": {family: {"items": len(group), "delta": mean(row["delta"] for row in group),
                                                   "wins": sum(row["delta"] > TOL for row in group), "losses": sum(row["delta"] < -TOL for row in group)}
                                         for family, group in sorted(families.items())},
                            "provisional_cluster_deltas": {cluster: mean(values) for cluster, values in sorted(clusters.items())},
                            "confirmatory_p_value": None, "confidence_interval": None,
                            "uncertainty": "small exposed provisional clusters; repeated seeds averaged within item; independence not established"}
    features = [event for event in traces if event.get("arm") == "A3" and event.get("split") == "DEV" and event.get("iteration") == 1]
    alphas = [event["alpha"] for event in features if number(event.get("alpha"))]
    feature_items = defaultdict(list)
    for event in features:
        feature_items[event["case_id"]].append(event)
    fields = ("alpha", "balance", "progress", "fluctuation", "delta_norm", "delta_cosine", "previous_ema_cosine", "movement", "state_norm")
    feature_distributions = {field: distribution(event.get(field) for event in features) for field in fields}
    feature_distributions["normalized_raw_update"] = distribution(event["delta_norm"] / max(event["state_norm"], 1e-12)
        for event in features if number(event.get("delta_norm")) and number(event.get("state_norm")))
    feature_distributions["normalized_controlled_movement"] = distribution(event["movement"] / max(event["state_norm"], 1e-12)
        for event in features if number(event.get("movement")) and number(event.get("state_norm")))
    samples = [event["resources"] for event in forwards if isinstance(event.get("resources"), dict)]
    samples += [sample for sample in (native.get("after_load_resources"), native.get("final_resources"), native.get("final_resources_on_stop")) if isinstance(sample, dict)]
    samples += [item["resources"] for item in qualification.get("maximum_shape", []) if isinstance(item.get("resources"), dict)]
    cumulative_events = [event for event in budget if event.get("event") == "native_completed"]
    duplicate_budget = len({event.get("run") for event in cumulative_events}) != len(cumulative_events)
    cumulative = sum(event["native_worker_seconds"] for event in cumulative_events) if cumulative_events and not duplicate_budget and all(number(event.get("native_worker_seconds")) for event in cumulative_events) else None
    peak = lambda field: max((sample[field] for sample in samples if number(sample.get(field))), default=None)
    peak_gpu = peak("gpu_peak_allocated_bytes")
    if peak_gpu is not None and peak_gpu > 5 * GIB:
        errors.append("GPU recorded peak exceeded5GiB ceiling")
        complete = False
    if any(number(sample.get("gpu_free_bytes")) and sample["gpu_free_bytes"] < GIB for sample in samples):
        errors.append("GPU recorded global free reserve violated")
        complete = False
    floors = {}
    for split in ("DEV", "SMOKE"):
        bank = [item for item in metadata if item["split"] == split]
        if bank:
            floors[split] = {"items": len(bank), "prompt_copy_accuracy": mean(item["audit"]["prompt_copy_score"] for item in bank),
                             "constant_accuracy": {answer: mean(item["audit"]["constant_answer_scores"][answer] for item in bank)
                                                   for answer in bank[0]["audit"]["constant_answer_scores"]},
                             "predictor_inputs": "prompt only or fixed literal; target only for scoring", "positive_solver": "NOT_VERIFIED"}
    # Resource errors also suppress comparisons rather than leaving precomputed tables.
    if not complete:
        pairs = {}
        h = grid_headroom(None, None, None, False)
        oracle_score = best_score = None
        for row in tables:
            row.update(accuracy=None, status="PARTIAL_OBSERVED_ONLY")
    return {"lane": "GPU_FP32_SUCCESSOR", "status": "COMPLETE_EXPLORATORY_GPU_DEV" if complete else "INCOMPLETE_NO_PRIMARY_COMPARISON",
            "complete": complete, "errors": errors, "missing_cells": missing, "extra_cells": extra, "duplicate_cells": duplicates,
            "physical_rows": len(physical), "expected_physical_rows": len(expected), "reused_oracle_rows": len(reused),
            "native_qualification_valid": qualification_ok, "corrected_A5_qualification_valid": a5_ok,
            "deterministic_repeat_divergence": repeated_divergence, "A5_repeats": "stochastic draws; averaged within case",
            "dev_items": len(dev_ids), "provisional_clusters": dict(Counter(admitted[case_id]["independent_cluster"] for case_id in dev_ids)),
            "primary_endpoint": "inherited binary task score, equal item weights after repeat averaging", "practical_margin": .05,
            "tables": tables, "fixed_alpha_curve": [{"alpha": alpha, "accuracy": curve[alpha] if complete else None,
                                                        "observed_partial_accuracy": curve[alpha]} for alpha in GRID],
            "frozen_fixed_comparator": {"alpha": frozen, "accuracy": curve.get(frozen) if complete else None,
                                        "selection_surface": "same exposed DEV, not independent validation"},
            "diagnostic_global_fixed_envelope": {"alpha": envelope, "accuracy": best_score, "diagnostic_only": True},
            "constant_oracle": {"accuracy": oracle_score, "physical_run": False, "scope": "finite per-example constant grid", **h},
            "prefix_schedule_oracle": "NOT_ENUMERATED", "pairs": pairs,
            "features": {"prefix_weighted": feature_distributions,
                         "per_item_mean_alpha": {case_id: mean(event["alpha"] for event in group) for case_id, group in sorted(feature_items.items())},
                         "upper_saturation_fraction": mean(int(alpha >= 1.2 - 1e-6) for alpha in alphas),
                         "lower_saturation_fraction": mean(int(alpha <= .8 + 1e-6) for alpha in alphas),
                         "pool_weighting": "generated DEV prefixes across two deterministic A3 repeats", "predictive_validation": "NOT_RUN"},
            "costs": {"physical_row_measurements": {field: known_sum(physical, field) for field in
                         ("wall_seconds", "synchronized_forward_seconds", "controller_seconds", "controller_logging_seconds",
                          "forward_logging_seconds", "full_model_calls_including_gold_nll", "processed_token_positions", "extra_layer_applications")},
                      "forward_attempt_journal": forward_costs, "oracle_additional_generation_cost": 0,
                      "oracle_cost_note": "copied timing/counters excluded; A2 already charged",
                      "worker_wall_seconds": worker.get("native_worker_seconds"), "supervisor_wall_seconds": worker.get("supervisor_wall_seconds"),
                      "peak_gpu_allocated_bytes": peak_gpu, "peak_gpu_reserved_bytes": peak("gpu_peak_reserved_bytes"),
                      "minimum_recorded_global_gpu_free_bytes": min((sample["gpu_free_bytes"] for sample in samples if number(sample.get("gpu_free_bytes"))), default=None),
                      "peak_worker_rss_bytes": max([sample["rss_bytes"] for sample in resources if number(sample.get("rss_bytes"))] + [worker["peak_rss_bytes"]] if number(worker.get("peak_rss_bytes")) else [sample["rss_bytes"] for sample in resources if number(sample.get("rss_bytes"))], default=None),
                      "cumulative_native_worker_seconds": cumulative, "remaining_native_seconds": max(0, 21600 - cumulative) if cumulative is not None else None,
                      "timing_scope": "GPU synchronized instrumented core/controller/fullforward; includes sync/logging overhead; no production or matched-quality speed claim"},
            "own_bank_floors": floors, "semantic_correctness": "NOT_ASSESSED", "semantic_abstention": "NOT_ASSESSED",
            "unsupported_answers": "NOT_ASSESSED", "independent_replication": "NOT_RUN", "native_TEST": "CLOSED", "efficacy_claim_supported": False}


def run(run_root, out):
    run_root, out = Path(run_root).resolve(strict=True), Path(out).resolve()
    if PACKAGE not in run_root.parents or PACKAGE not in out.parents or run_root in out.parents or out in run_root.parents or run_root == out:
        raise PermissionError("GPU analysis requires separate exclusive campaign paths")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAIL", "native_model_loads": 0, "started_utc": datetime.now(timezone.utc).isoformat()}
    inputs, errors = [], []
    try:
        def read(path, required=True):
            path = Path(path)
            if path.is_file():
                inputs.append(path)
                return json.loads(path.read_text(encoding="utf-8"))
            if required:
                errors.append("missing GPU artifact: " + path.name)
            return {}
        result = run_root / "result"
        binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
        prompts_path = binding_path.parent / "PROMPT_AUDIT.json"
        binding, metadata = read(binding_path), read(prompts_path)
        binding_receipt = read(binding_path.parent / "receipt.json")
        admission, review, manifest = read(run_root / "admission.json"), read(run_root / "review.json"), read(run_root / "source_manifest.json")
        native, worker, registration = read(result / "receipt.json"), read(run_root / "terminal_receipt.json"), read(result / "registration.json")
        qualification, a5, selection = read(result / "qualification.json"), read(result / "A5_qualification.json"), read(result / "frozen_dev_selection.json")
        rows, e = read_journal(result / "outcomes.jsonl"); errors.extend(e)
        traces, e = read_journal(result / "trajectories.jsonl"); errors.extend(e)
        forwards, e = read_journal(result / "forwards.jsonl"); errors.extend(e)
        resources, e = read_journal(run_root / "resources.jsonl"); errors.extend(e)
        budget, e = read_journal(PACKAGE / "BUDGET_LEDGER.jsonl"); errors.extend(e)
        for path in (result / "outcomes.jsonl", result / "trajectories.jsonl", result / "forwards.jsonl", run_root / "resources.jsonl", PACKAGE / "BUDGET_LEDGER.jsonl"):
            if path.is_file(): inputs.append(path)
        def require(condition, message):
            if not condition: errors.append(message)
        require(binding_receipt.get("binding_sha256") == sha(binding_path) and binding_receipt.get("prompt_audit_sha256") == sha(prompts_path), "GPU exact tokenizer/data packet differs")
        admission_sha = sha(run_root / "admission.json") if admission else None
        require(admission.get("status") == "READY_FOR_GPU_FP32_DISPATCH" and review.get("status") == "PASS_GPU_FP32_DISPATCH"
                and review.get("admission_sha256") == admission_sha, "GPU independent admission/review binding differs")
        require(admission.get("binding_sha256") == sha(binding_path) and admission.get("source") == binding.get("source_path")
                and admission.get("checkpoint_files") == binding.get("checkpoint_files")
                and admission.get("maximum_shapes") == binding.get("shapes"), "GPU source/assets/shapes binding differs")
        require(admission.get("device") == "cuda:0" and admission.get("dtype") == "float32" and admission.get("attn_implementation") == "eager"
                and admission.get("TF32") is False and admission.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "GPU runtime math contract differs")
        require(admission.get("training") is False and admission.get("native_test") is False
                and admission.get("gpu_allocated_ceiling_bytes") == 5 * GIB
                and admission.get("gpu_global_free_reserve_bytes") == GIB
                and admission.get("gpu_initial_free_floor_bytes") == 6 * GIB
                and admission.get("worker_host_available_floor_bytes") == 6 * GIB
                and admission.get("worker_rss_ceiling_bytes") == 6 * GIB, "GPU authority/resource contract differs")
        require(bool(manifest.get("sha256")) and bool(manifest.get("head")), "GPU source manifest missing")
        for relative, digest in admission.get("dispatch_hashes", {}).items():
            if relative.endswith(".py"):
                require(manifest.get("sha256", {}).get(relative) == digest, "GPU source manifest differs: " + relative)
        require(registration.get("lane") == "GPU_FP32_SUCCESSOR" and registration.get("admission_sha256") == admission_sha
                and registration.get("binding_sha256") == sha(binding_path) and registration.get("native_layers") == [12, 13]
                and registration.get("extra_loops") == [0, 1] and registration.get("grid") == list(GRID)
                and registration.get("seeds") == list(SEEDS), "GPU native registration differs")
        taps_path = REPO / "experiments/rineya_taps_adaptive_recurrence_v1/configs/inference_v1.json"
        taps = read(taps_path)["controller"]
        require(registration.get("taps") == taps and selection.get("config") == taps, "GPU A3/frozen configuration differs")
        history_path = REPO / "experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json"
        history = read(history_path)
        require(sha(history_path) == binding.get("historical_receipt_sha256"), "GPU controlling historical receipt differs")
        expected_config = {**history["config"], "device": "cuda:0", "dtype": "float32", "max_rss_gib": 6}
        require(registration.get("config") == expected_config and registration.get("training") is False
                and registration.get("native_test") is False and registration.get("gold_nll_status") == "NOT_COLLECTED_CAUSAL_GUARD",
                "GPU generation/gold/training native configuration differs")
        for split, filename in (("DEV", "mergedev_fast.jsonl"), ("SMOKE", "supa_layer_ablation_smoke.jsonl")):
            relative = "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/" + filename
            require(admission.get("dispatch_hashes", {}).get(relative) == binding["datasets"][split]["sha256"], "GPU admitted data digest differs: " + split)
        protocol_path = PACKAGE / "PREREGISTRATION.md"
        inputs.append(protocol_path)
        require("Practical exploratory improvement margin: .05" in protocol_path.read_text(encoding="utf-8"), "GPU primary endpoint/margin protocol differs")
        protocol_key = protocol_path.relative_to(REPO).as_posix()
        require(admission.get("dispatch_hashes", {}).get(protocol_key) == sha(protocol_path), "GPU prospective primary protocol digest differs")
        for field in ("python", "torch", "transformers"):
            require(registration.get("runtime", {}).get(field) == binding["runtime"].get(field), "GPU registered runtime differs: " + field)
        require(registration.get("runtime", {}).get("tf32") is False and registration.get("runtime", {}).get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "GPU registered TF32/determinism differs")
        require(selection.get("dataset_sha256") == binding["datasets"]["DEV"]["sha256"] and selection.get("binding_sha256") == sha(binding_path)
                and selection.get("native_TEST_authorized") is False, "GPU frozen selection/pool data binding differs")
        pool = [min(1.2, max(.8, event["alpha"])) for event in traces if event.get("arm") == "A3" and event.get("split") == "DEV"
                and event.get("iteration") == 1 and number(event.get("alpha"))] or [1.]
        require(selection.get("pool") == pool, "GPU frozen pool differs from generation-only A3 DEV traces")
        if (result / "trajectories.jsonl").is_file():
            prefix = bytearray()
            for line in (result / "trajectories.jsonl").read_bytes().splitlines(keepends=True):
                if json.loads(line).get("arm") == "A5": break
                prefix.extend(line)
            require(selection.get("generation_trajectory_sha256_at_freeze") == hashlib.sha256(prefix).hexdigest(), "GPU frozen trajectory-prefix digest differs")
        require(native.get("loaded_state_sha256") == binding.get("historical_expected_loaded_fp32_state_sha256")
                and qualification.get("after_state_sha256") == native.get("loaded_state_sha256"), "GPU actual loaded/qualified state identity differs")
        code_path = Path(native.get("loaded_custom_code_path", "__MISSING__")).resolve()
        require((result / "hf_cache").resolve() in code_path.parents and code_path.is_file(), "GPU actual custom-code path missing/unscoped")
        if code_path.is_file() and (result / "hf_cache").resolve() in code_path.parents:
            inputs.append(code_path)
            require(sha(code_path) == native.get("loaded_custom_code_sha256") == binding["checkpoint_files"]["modeling_supa_a11.py"], "GPU actual loaded custom-code hash differs")
        scorer_path = REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/rineya_cross_scale/evaluation/scoring.py"
        inputs.append(scorer_path)
        require(sha(scorer_path) == binding.get("scorer_sha256"), "GPU pinned pure scorer differs")
        if sha(scorer_path) != binding.get("scorer_sha256"):
            raise RuntimeError("refuse to execute a scorer whose source differs from pinned binding")
        spec = importlib.util.spec_from_file_location("gpu_analysis_pinned_scorer", scorer_path)
        scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
        item_map = {item["case_id"]: item for item in metadata}
        for row in rows:
            item = item_map.get(row.get("case_id"))
            if item is None or not isinstance(row.get("response"), str): continue
            require(scorer.score_response(row["response"], item["scorer"])["score"] == row.get("score"), "GPU stored task score differs: " + row["case_id"])
            if item.get("format_scorer"):
                require(scorer.score_response(row["response"], item["format_scorer"])["score"] == row.get("format_score"), "GPU stored format score differs: " + row["case_id"])
        report = aggregate(rows, traces, metadata, selection, native, worker, qualification, a5, forwards, resources, budget, errors)
        source_manifest = {"analysis_source_sha256": sha(__file__), "helper_source_sha256": sha(PACKAGE / "analysis.py"),
                           "input_sha256": {str(path): sha(path) for path in inputs}, "native_model_loads": 0,
                           "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()}
        for name, value in (("REPORT.json", report), ("MANIFEST.json", source_manifest)):
            with (out / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        lines = ["# GPU FP32 successor analysis", "", report["status"], "",
                 "Exposed DEV binary scorer endpoint; repeat averaging within item, provisional scenario clusters; practical margin .05.",
                 "| Policy | Accuracy |", "|---|---:|"]
        lines += [f"| {row['policy']} | {row['accuracy']} |" for row in report["tables"]]
        lines += ["", f"Constant-grid headroom: {report['constant_oracle']['headroom']}; capture: {report['constant_oracle']['capture']}.",
                  f"Physical rows: {report['physical_rows']}/{report['expected_physical_rows']}; copied oracle rows: {report['reused_oracle_rows']}.",
                  "Corrected A5 varies by actual seed/prefix; repeated seeds are averaged within each case.",
                  "Synchronized instrumented timing includes sync/logging overhead; no production or matched-quality speed claim.",
                  "Semantic correctness, unsupported answers and independent replication remain unassessed."]
        (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        receipt.update(status="ANALYZED", matrix_complete=report["complete"], report_sha256=sha(out / "REPORT.json"))
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with (out / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    return 0 if receipt["status"] == "ANALYZED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.run_root, args.out))
