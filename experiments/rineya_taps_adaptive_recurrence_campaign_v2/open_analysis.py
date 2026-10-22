"""Source-only analysis for the separate OPEN data/scorer/cap intervention."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from . import open_bank
from .analysis import SEEDS, TOL, distribution, grid_headroom, known_sum, mean, number, policy, read_journal, sha, token_count
from .gpu_analysis import validate_gpu_traces

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
GRID = tuple(sorted((.8, .9, 1., 1.1, 1.2, 1.1473425965417514)))
GIB = 1024**3


def payload_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def expected_open_cells(bank, smoke):
    controls = [("A0", 0), ("A1", 1)] + [(f"A2:{alpha:g}", 1) for alpha in GRID] + [("A3", 1), ("A5", 1)]
    cells = {(row["split"], seed, row["id"], arm, extra) for row in bank for seed in SEEDS for arm, extra in controls}
    for row in smoke:
        cells.update(("SMOKE", 17, row["case_id"], arm, extra) for arm, extra in controls)
        cells.update(("SMOKE", 17, row["case_id"], arm, extra) for arm in ("qualification_legacy", "qualification_new") for extra in (0, 1))
        cells.update(("SMOKE", 17, row["case_id"], arm, 1) for arm in ("qualification_reference_A3", "qualification_successor_A3"))
    return cells


def category_mean(case_values, bank):
    categories = defaultdict(list)
    for row in bank:
        if row["id"] in case_values:
            categories[row["category"]].append(case_values[row["id"]])
    return mean(mean(values) for values in categories.values())


def raw_diagnostics(response, raw):
    normalized = open_bank.normalize_output(response)
    unknown = normalized == "UNKNOWN"
    unanswerable = raw["answerability"] == "unanswerable"
    return {"normalized_response": normalized, "answerability": raw["answerability"], "source_status": raw["source_status"],
            "canonical_UNKNOWN": unknown, "literal_UNKNOWN": response.strip() == "UNKNOWN",
            "false_abstention_on_answerable": not unanswerable and unknown,
            "correct_canonical_abstention": unanswerable and unknown,
            "unsupported_canonical_answer_on_unanswerable": (not unknown if normalized is not None else None) if unanswerable else False}


def aggregate(rows, traces, bank, smoke, token_rows, freeze, native, worker, qualification, a5, forwards, resources=(), budget=(), issues=()):
    errors = list(issues)
    ids = {row["id"]: row for row in bank}
    if len(ids) != len(bank): raise ValueError("duplicate OPEN IDs")
    if (len(bank) != 24 or len({row["parent_id"] for row in bank}) != 24
            or Counter(row["split"] for row in bank) != {"DEVELOPMENT": 16, "OPEN_VALIDATION": 8}
            or len({row["generator_family"] for row in bank}) != 8 or len({row["category"] for row in bank}) != 4):
        errors.append("OPEN registered parent/split/generator/category cardinality differs")
    dev_families = {row["generator_family"] for row in bank if row["split"] == "DEVELOPMENT"}
    val_families = {row["generator_family"] for row in bank if row["split"] == "OPEN_VALIDATION"}
    if dev_families & val_families or len(dev_families) != 4 or len(val_families) != 4:
        errors.append("OPEN generator partition differs")
    physical = [row for row in rows if row.get("physical_run") is True]
    oracle_rows = [row for row in rows if row.get("physical_run") is False and row.get("arm") == "A4_example"]
    aliases = [row for row in rows if row.get("physical_run") is False and row.get("arm") == "A2_frozen"]
    if len(physical) + len(oracle_rows) + len(aliases) != len(rows): errors.append("unclassified OPEN physical/nonphysical row")
    indexed = defaultdict(list)
    for row in physical:
        indexed[(row.get("split"), row.get("seed"), row.get("case_id"), policy(row), row.get("extra_loops"))].append(row)
        if row.get("configured_seed") != row.get("seed") or row.get("safety_stopped") is not False:
            errors.append("OPEN seed/safety contract differs")
        if row.get("gold_nll") is not None or row.get("gold_answer") is not None: errors.append("OPEN gold diagnostic guard violated")
        if type(row.get("score")) not in (int, float) or row["score"] not in (0, 1):
            errors.append("OPEN/SMOKE registered score is not binary")
        if row.get("format_score") is not None and (type(row["format_score"]) not in (int, float) or row["format_score"] not in (0, 1)):
            errors.append("OPEN/SMOKE registered format score is not binary")
        if row.get("arm") == "A2" and row.get("fixed_alpha") not in GRID:
            errors.append("OPEN fixed alpha is not an exact registered six-point constant")
        cap = 24 if row.get("split") == "SMOKE" else 8
        if not isinstance(row.get("token_ids"), list) or not 1 <= len(row["token_ids"]) <= cap:
            errors.append("OPEN/SMOKE generated-token count violates registered cap")
        raw = ids.get(row.get("case_id"))
        if raw is not None:
            expected_score = open_bank.score_response(row["response"], raw)
            if row.get("score") != int(expected_score["correct"]) or row.get("format_score") != int(expected_score["format_valid"]):
                errors.append("OPEN strict task/format regrade differs")
            if row.get("parent_id") != raw["parent_id"] or row.get("generator_family") != raw["generator_family"]:
                errors.append("OPEN parent/generator provenance differs")
            if any(row.get(field) != value for field, value in raw_diagnostics(row["response"], raw).items()):
                errors.append("OPEN answerability/UNKNOWN regrade differs")
    expected = expected_open_cells(bank, smoke)
    missing, extra = sorted(expected - set(indexed), key=repr), sorted(set(indexed) - expected, key=repr)
    duplicate = [key for key, group in indexed.items() if len(group) != 1]
    if extra or duplicate: errors.append("OPEN extra/duplicate physical cells")
    errors.extend(validate_gpu_traces(physical, traces, freeze.get("pool", [])))
    if any(event.get("arm") in ("A1", "qualification_new") and
           (not number(event.get("alpha")) or abs(event["alpha"]-1.) > 1e-6) for event in traces):
        errors.append("OPEN unit/qualification-unit trajectory used a non-unit scale")
    grouped = defaultdict(list)
    for row in physical:
        if row.get("case_id") in ids: grouped[(row["split"], policy(row), row["case_id"])].append(row)
    items, divergence = {}, []
    for (split, arm, case_id), group in grouped.items():
        if arm != "A5" and len({tuple(row["token_ids"]) for row in group}) > 1:
            divergence.append({"split": split, "policy": arm, "case_id": case_id})
        items[(split, arm, case_id)] = {"score": mean(row["score"] for row in group), "format": mean(row["format_score"] for row in group),
            "false_abstention": mean(int(row["false_abstention_on_answerable"]) for row in group),
            "unsupported": mean(int(row["unsupported_canonical_answer_on_unanswerable"]) for row in group if row["unsupported_canonical_answer_on_unanswerable"] is not None),
            "literal_UNKNOWN": mean(int(row["literal_UNKNOWN"]) for row in group), "canonical_UNKNOWN": mean(int(row["canonical_UNKNOWN"]) for row in group)}
    if divergence: errors.append("OPEN deterministic non-A5 nominal-seed outputs differ")
    dev_ids = {row["id"] for row in bank if row["split"] == "DEVELOPMENT"}
    freeze_payload = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    if (freeze.get("status") != "FROZEN_OPEN_DEVELOPMENT" or freeze.get("freeze_sha256") != payload_digest(freeze_payload)
            or set(freeze.get("selection_case_ids", [])) != dev_ids
            or freeze.get("OPEN_VALIDATION_native_exposure_at_freeze") is not False):
        errors.append("OPEN DEV freeze identity/split firewall differs")
    first_validation = next((i for i, row in enumerate(physical) if row.get("split") == "OPEN_VALIDATION"), None)
    if first_validation is not None and any(row.get("split") == "DEVELOPMENT" for row in physical[first_validation:]):
        errors.append("OPEN validation started before development evaluation completed")
    expected_pool = [min(1.2, max(.8, event["alpha"])) for event in traces if event.get("split") == "DEVELOPMENT" and event.get("arm") == "A3" and event.get("iteration") == 1]
    if freeze.get("pool") != expected_pool or not expected_pool: errors.append("OPEN frozen pool differs from DEV-only generation traces")
    # Check every forward event against bound input length and completed row count.
    token_map = {row["case_id"]: row["input_tokens"] for row in token_rows}
    token_map.update({row["case_id"]: row["input_tokens"] for row in smoke})
    chunks = []
    for event in forwards:
        if not event.get("event", "").startswith("FORWARD_"): continue
        if event["event"] == "FORWARD_STARTED" and event.get("full_model_call") == 1: chunks.append([])
        if chunks: chunks[-1].append(event)
        else: errors.append("OPEN forward journal lacks evaluation start")
    if len(chunks) != len(physical): errors.append("OPEN forward/outcome evaluation coverage differs")
    for row, chunk in zip(physical, chunks):
        calls, prompt = token_count(row), token_map.get(row.get("case_id"))
        if [(event.get("event"), event.get("full_model_call")) for event in chunk] != [(name, i) for i in range(1, calls+1) for name in ("FORWARD_STARTED", "FORWARD_COMPLETED")]:
            errors.append("OPEN forward start/completion sequence differs")
        if type(prompt) is not int: errors.append("OPEN bound prompt length missing"); continue
        if any(event.get("prefix_tokens") != prompt + event.get("full_model_call", 0) - 1 or any(event.get(field) != row.get(field) for field in ("arm", "split", "seed", "extra_loops", "case_id")) for event in chunk):
            errors.append("OPEN forward prefix/provenance differs")
        positions = calls*prompt + calls*(calls-1)//2
        if row.get("full_model_calls_including_gold_nll") != calls or row.get("initial_block_applications") != calls or row.get("processed_token_positions") != positions or row.get("extra_layer_applications") != 2*calls*row.get("extra_loops", 0):
            errors.append("OPEN forward/position/block counters differ")
    max_starts = [event for event in forwards if event.get("event") == "MAXIMUM_FORWARD_STARTED"]
    max_done = [event for event in forwards if event.get("event") == "MAXIMUM_FORWARD_COMPLETED"]
    max_failed = [event for event in forwards if event.get("event") == "MAXIMUM_FORWARD_FAILED"]
    cache = [event for event in forwards if event.get("event") == "CACHE_REJECTED"]
    if len(max_starts) != 18 or len(max_done) != 17 or len(max_failed) != 1 or len(cache) != 2:
        errors.append("OPEN complete max357/exception/cache matrix missing")
    maximum_ids = [event.get("attempt") for event in max_starts]
    terminal_ids = [event.get("attempt") for event in max_done + max_failed]
    if set(maximum_ids) != set(range(1, 19)) or set(terminal_ids) != set(maximum_ids) or len(terminal_ids) != len(set(terminal_ids)):
        errors.append("OPEN maximum attempt IDs/terminal pairing differs")
    if any(event.get("prefix_tokens") != 357 for event in max_starts + max_done + max_failed): errors.append("OPEN maximum shape differs")
    if max_failed and "EXPECTED_OPEN_FORCED_EXCEPTION" not in max_failed[0].get("error", ""): errors.append("OPEN maximum failure is not expected restoration probe")
    cache_starts = [event for event in forwards if event.get("event") == "CACHE_REJECTION_STARTED"]
    if {(event.get("use_cache"), event.get("cache_object_present")) for event in cache_starts} != {(True, False), (False, True)} or any(event.get("model_compute_started") is not False or event.get("wrapper_forward_count") != 0 for event in cache):
        errors.append("OPEN cache rejection did not cover both pre-compute contracts")
    terminal = (native.get("status") == "OPEN_MATRIX_COMPLETE_NOT_CONFIRMATORY" and native.get("source_unchanged") is True
                and native.get("checkpoint_files_unchanged") is True and native.get("trainable_parameters") == 0
                and native.get("training") is False and worker.get("status") == "WORKER_TERMINAL" and worker.get("exit_code") == 0)
    qualified = (qualification.get("status") == "PASS_OPEN_NEW_CAP_SAME_OPERATOR" and qualification.get("maximum_prefix_tokens") == 357
                 and qualification.get("noextra_unit_originalA3_identity") is True and qualification.get("forced_exception_restoration") is True
                 and qualification.get("cache_rejections") == 2 and qualification.get("source_unchanged") is True
                 and qualification.get("before_state_sha256") == qualification.get("after_state_sha256")
                 and a5.get("status") == "PASS_FROZEN_OPEN_DEV_POOL" and a5.get("actual_seed_binding") is True)
    if not terminal or not qualified: errors.append("OPEN terminal/native/new-cap/A5 qualification missing")
    frozen_alpha = freeze.get("alpha")
    expected_oracles = {(row["split"], row["id"], seed) for row in bank for seed in SEEDS}
    if len(oracle_rows) != len(expected_oracles) or {(row.get("split"), row.get("case_id"), row.get("seed")) for row in oracle_rows} != expected_oracles:
        errors.append("OPEN retrospective constant oracle coverage differs")
    validation_ids = {row["id"] for row in bank if row["split"] == "OPEN_VALIDATION"}
    if len(aliases) != 2*len(validation_ids) or {(row.get("case_id"), row.get("seed")) for row in aliases} != {(case_id, seed) for case_id in validation_ids for seed in SEEDS}:
        errors.append("OPEN frozen validation comparator aliases missing/duplicate")
    if not missing and not extra and not duplicate:
        for row in oracle_rows:
            candidates = [indexed[(row["split"], row["seed"], row["case_id"], f"A2:{alpha:g}", 1)][0] for alpha in GRID]
            best = max(candidates, key=lambda item: (item["score"], -item["fixed_alpha"]))
            if any(row.get(field) != best.get(field) for field in ("score", "fixed_alpha", "token_ids")): errors.append("OPEN oracle is not exact nonphysical fixed reuse")
        for row in aliases:
            original = indexed[("OPEN_VALIDATION", row["seed"], row["case_id"], f"A2:{frozen_alpha:g}", 1)][0]
            if any(row.get(field) != original.get(field) for field in ("score", "fixed_alpha", "token_ids")): errors.append("OPEN frozen alias differs from preselected constant rollout")
    samples = [event["resources"] for event in forwards if isinstance(event.get("resources"), dict)]
    samples += [sample for sample in (native.get("after_load_resources"), native.get("final_resources"), native.get("final_resources_on_stop"), qualification.get("resources")) if isinstance(sample, dict)]
    samples += [sample for sample in resources if isinstance(sample, dict)]
    if number(worker.get("peak_rss_bytes")): samples.append({"rss_bytes": worker["peak_rss_bytes"]})
    peak_gpu = max((sample["gpu_peak_allocated_bytes"] for sample in samples if number(sample.get("gpu_peak_allocated_bytes"))), default=None)
    if peak_gpu is not None and peak_gpu > 5*GIB: errors.append("OPEN GPU5GiB allocation ceiling exceeded")
    if any(number(sample.get("gpu_free_bytes")) and sample["gpu_free_bytes"] < GIB for sample in samples): errors.append("OPEN GPU global reserve breached")
    if any(number(sample.get("rss_bytes")) and sample["rss_bytes"] > 6*GIB for sample in samples): errors.append("OPEN recorded RSS6GiB ceiling breached")
    if any(number(sample.get("host_available_bytes")) and sample["host_available_bytes"] < GIB for sample in samples): errors.append("OPEN recorded host1GiB reserve breached")
    complete = not missing and not errors
    split_reports = {}
    for split in ("DEVELOPMENT", "OPEN_VALIDATION"):
        split_bank = [row for row in bank if row["split"] == split]
        def values(arm, field="score"):
            return {row["id"]: items[(split, arm, row["id"])][field] for row in split_bank if (split, arm, row["id"]) in items}
        curve = {alpha: category_mean(values(f"A2:{alpha:g}"), split_bank) for alpha in GRID}
        envelope = max(GRID, key=lambda alpha: (curve[alpha], -alpha)) if all(number(score) for score in curve.values()) else None
        if split == "DEVELOPMENT" and frozen_alpha != envelope:
            errors.append("OPEN frozen alpha differs from full DEV six-point selection"); complete = False
        oracle_values = defaultdict(list)
        for row in oracle_rows:
            if row["split"] == split: oracle_values[row["case_id"]].append(row["score"])
        oracle_score = category_mean({case_id: mean(scores) for case_id, scores in oracle_values.items()}, split_bank) if complete else None
        best_score = curve[envelope] if complete else None
        adaptive = category_mean(values("A3"), split_bank)
        comparisons = {}
        if complete:
            for name, comparator in (("DEV_frozen_fixed", f"A2:{frozen_alpha:g}"), ("intact", "A0"), ("unit", "A1"), ("corrected_A5", "A5")):
                per_parent = []
                for raw in split_bank:
                    a, b = items[(split, "A3", raw["id"])], items[(split, comparator, raw["id"])]
                    per_parent.append({"case_id": raw["id"], "parent_id": raw["parent_id"], "generator_family": raw["generator_family"],
                        "category": raw["category"], "treatment_score": a["score"], "comparator_score": b["score"], "delta": a["score"]-b["score"],
                        "format_delta": a["format"]-b["format"], "false_abstention_delta": a["false_abstention"]-b["false_abstention"],
                        "unsupported_delta": a["unsupported"]-b["unsupported"] if number(a["unsupported"]) and number(b["unsupported"]) else None})
                delta = category_mean({row["case_id"]: row["delta"] for row in per_parent}, split_bank)
                families = defaultdict(list)
                for row in per_parent: families[row["generator_family"]].append(row)
                comparisons[name] = {"comparator": comparator, "score_delta": delta, "meets_exploratory_margin": delta >= .05-TOL,
                    "wins": [row["case_id"] for row in per_parent if row["delta"] > TOL], "losses": [row["case_id"] for row in per_parent if row["delta"] < -TOL],
                    "source_correct": sum(row["comparator_score"] == 1 for row in per_parent),
                    "source_correct_retained": sum(row["comparator_score"] == 1 and row["treatment_score"] == 1 for row in per_parent),
                    "observed_preservation_screen": all(row["delta"] >= -TOL and row["format_delta"] >= -TOL and row["false_abstention_delta"] <= TOL
                        and (row["unsupported_delta"] is None or row["unsupported_delta"] <= TOL) for row in per_parent),
                    "noninferiority": "NOT_ESTABLISHED", "per_parent": per_parent,
                    "generator_families": {family: {"parents": len(group), "delta": mean(row["delta"] for row in group)} for family, group in families.items()},
                    "confirmatory_p_value": None, "confidence_interval": None}
        predictors = {name: 0 for name in open_bank.blind_predictions(split_bank[0]["prompt"])} if split_bank else {}
        family_floors = defaultdict(Counter)
        for raw in split_bank:
            for name, response in open_bank.blind_predictions(raw["prompt"]).items():
                score = int(open_bank.score_response(response, raw)["correct"])
                predictors[name] += score
                family_floors[raw["generator_family"]][name] += score
        policies = ["A0", "A1", *(f"A2:{alpha:g}" for alpha in GRID), "A3", "A5"]
        split_reports[split] = {"parents": len(split_bank), "generator_clusters": dict(Counter(row["generator_family"] for row in split_bank)),
            "category_weights": {category: .25 for category in {row["category"] for row in split_bank}},
            "tables": [{"policy": arm, "accuracy": category_mean(values(arm), split_bank) if complete else None,
                        "observed_partial_accuracy": category_mean(values(arm), split_bank), "format_accuracy": category_mean(values(arm, "format"), split_bank),
                        "literal_UNKNOWN_rate": category_mean(values(arm, "literal_UNKNOWN"), split_bank),
                        "canonical_UNKNOWN_rate": category_mean(values(arm, "canonical_UNKNOWN"), split_bank)} for arm in policies],
            "fixed_curve": [{"alpha": alpha, "accuracy": score if complete else None} for alpha, score in curve.items()],
            "DEV_frozen_comparator": {"alpha": frozen_alpha, "accuracy": curve.get(frozen_alpha) if complete else None},
            "retrospective_fixed_envelope": {"alpha": envelope, "accuracy": best_score, "diagnostic_only": True},
            "constant_oracle": {"accuracy": oracle_score, **grid_headroom(adaptive, best_score, oracle_score, complete), "diagnostic_only": True},
            "pairs": comparisons, "own_bank_blind_floor_counts": predictors,
            "own_bank_blind_floor_by_generator": dict(family_floors),
            "semantic_reasoning_beyond_scalar_contract": "NOT_ASSESSED", "semantic_natural_language_abstention": "NOT_ASSESSED"}
    if not complete:
        for report in split_reports.values():
            report["pairs"] = {}; report["constant_oracle"].update(accuracy=None, **grid_headroom(None, None, None, False))
            report["DEV_frozen_comparator"]["accuracy"] = report["retrospective_fixed_envelope"]["accuracy"] = None
            for row in report["tables"] + report["fixed_curve"]: row["accuracy"] = None
    completed = [event for event in forwards if event.get("event") in ("FORWARD_COMPLETED", "MAXIMUM_FORWARD_COMPLETED")]
    budget_events = [event for event in budget if event.get("event") == "native_completed"]
    cumulative = sum(event["native_worker_seconds"] for event in budget_events) if budget_events and all(number(event.get("native_worker_seconds")) for event in budget_events) else None
    features = [event for event in traces if event.get("arm") == "A3" and event.get("iteration") == 1 and event.get("split") in ("DEVELOPMENT", "OPEN_VALIDATION")]
    alphas = [event["alpha"] for event in features if number(event.get("alpha"))]
    return {"lane": "OPEN_GPU_FP32_INTERVENTION", "status": "COMPLETE_EXPLORATORY_OPEN" if complete else "INCOMPLETE_NO_PRIMARY_COMPARISON",
            "complete": complete, "errors": errors, "missing_cells": missing, "extra_cells": extra, "duplicate_cells": duplicate,
            "physical_rows": len(physical), "expected_physical_rows": len(expected), "oracle_reused_rows": len(oracle_rows), "frozen_alias_rows": len(aliases),
            "deterministic_divergence": divergence, "primary_endpoint": "strict canonical correctness, equal category weights, stochastic repeats averaged within parent",
            "splits": split_reports, "features": {field: distribution(event.get(field) for event in features) for field in ("alpha", "balance", "progress", "fluctuation", "delta_cosine")},
            "alpha_upper_saturation_fraction": mean(int(alpha >= 1.2-1e-6) for alpha in alphas),
            "alpha_lower_saturation_fraction": mean(int(alpha <= .8+1e-6) for alpha in alphas),
            "costs": {"physical_row_measurements": {field: known_sum(physical, field) for field in ("wall_seconds", "controller_seconds", "controller_logging_seconds",
                        "forward_logging_seconds", "full_model_calls_including_gold_nll", "processed_token_positions", "extra_layer_applications")},
                "all_completed_forward_count": len(completed), "all_completed_prefix_positions": known_sum(completed, "prefix_tokens"),
                "partial_forward_compute_positions": None, "max_attempts": len(max_starts), "max_completed": len(max_done), "cache_rejections": len(cache),
                "nonphysical_alias_oracle_extra_generation": 0, "worker_wall_seconds": worker.get("native_worker_seconds"),
                "peak_gpu_allocated_bytes": peak_gpu, "peak_gpu_reserved_bytes": max((sample["gpu_peak_reserved_bytes"] for sample in samples if number(sample.get("gpu_peak_reserved_bytes"))), default=None),
                "peak_sampled_worker_tree_rss_bytes": worker.get("peak_rss_bytes"), "cumulative_native_worker_seconds": cumulative,
                "timing_scope": "GPU synchronized instrumented path; includes synchronization/logging overhead; no production speed claim"},
            "controller_revision": 0, "independent_confirmation": False, "broad_intelligence_claim": False, "protected_TEST": "CLOSED"}


def run(run_root, out):
    run_root, out = Path(run_root).resolve(strict=True), Path(out).resolve()
    if PACKAGE not in run_root.parents or PACKAGE not in out.parents or run_root in out.parents or out in run_root.parents or run_root == out:
        raise PermissionError("OPEN analysis needs exclusive separate campaign paths")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAIL", "native_model_loads": 0, "started_utc": datetime.now(timezone.utc).isoformat()}
    inputs, errors = [], []
    try:
        def read(path):
            path = Path(path)
            if not path.is_file(): errors.append("missing OPEN artifact: " + path.name); return {}
            inputs.append(path); return json.loads(path.read_text(encoding="utf-8"))
        result = run_root / "result"
        admission, review, manifest = read(run_root / "admission.json"), read(run_root / "review.json"), read(run_root / "source_manifest.json")
        def require(condition, message):
            if not condition: errors.append(message)
        admission_sha = sha(run_root / "admission.json") if admission else None
        require(admission.get("status") == "READY_FOR_OPEN_FP32_DISPATCH" and review.get("status") == "PASS_OPEN_FP32_DISPATCH" and review.get("admission_sha256") == admission_sha,
                "OPEN source-bound independent admission/review differs")
        packet_path = Path(admission.get("open_bank_path", "__MISSING__")).resolve()
        token_path = Path(admission.get("open_token_binding_path", "__MISSING__")).resolve()
        if PACKAGE not in packet_path.parents or PACKAGE not in token_path.parents: raise PermissionError("OPEN source packet paths unscoped")
        bank, token = read(packet_path), read(token_path)
        for key, path in (("open_bank_sha256", packet_path), ("open_token_binding_sha256", token_path),
                          ("open_manifest_sha256", packet_path.parent / "manifest.json"), ("open_audit_sha256", packet_path.parent / "audit.json"),
                          ("open_positive_controls_sha256", packet_path.parent / "positive_controls.json")):
            inputs.append(path); require(path.is_file() and sha(path) == admission.get(key), "OPEN packet/token hash differs: " + key)
        require(sha(PACKAGE / "open_bank.py") == admission.get("open_source_sha256") == token.get("generator_scorer_sha256"), "OPEN pinned strict scorer source differs")
        if sha(PACKAGE / "open_bank.py") != admission.get("open_source_sha256"):
            raise RuntimeError("refuse to run an OPEN scorer whose source differs from reviewed binding")
        for relative, digest in admission.get("dispatch_hashes", {}).items():
            if relative.endswith(".py"): require(manifest.get("sha256", {}).get(relative) == digest, "OPEN dispatched source manifest differs: " + relative)
        model_binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
        smoke_binding_path = PACKAGE / "runs/binding-001/PROMPT_AUDIT.json"
        for path in (model_binding_path, smoke_binding_path):
            require(admission.get("dispatch_hashes", {}).get(path.relative_to(REPO).as_posix()) == sha(path),
                    "OPEN original runtime/SMOKE metadata binding differs: " + path.name)
        binding = read(model_binding_path)
        smoke = [row for row in read(smoke_binding_path) if row["split"] == "SMOKE"]
        native, worker = read(result / "receipt.json"), read(run_root / "terminal_receipt.json")
        qualifier, a5, freeze, registration = read(result / "qualification.json"), read(result / "A5_qualification.json"), read(result / "frozen_development.json"), read(result / "registration.json")
        require(registration.get("lane") == "OPEN_GPU_FP32_INTERVENTION" and registration.get("admission_sha256") == admission_sha
                and registration.get("bank_sha256") == admission.get("open_bank_sha256") and registration.get("token_binding_sha256") == sha(token_path)
                and registration.get("scorer_sha256") == admission.get("open_source_sha256") and registration.get("grid") == list(GRID), "OPEN native registration/source/scorer/grid differs")
        taps_path = REPO / "experiments/rineya_taps_adaptive_recurrence_v1/configs/inference_v1.json"
        taps = read(taps_path)["controller"]
        history_path = REPO / "experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json"
        history = read(history_path)
        expected_old_config = {**history["config"], "device": "cuda:0", "dtype": "float32", "max_rss_gib": 6}
        require(registration.get("config") == {**expected_old_config, "max_new_tokens": 8}
                and registration.get("old_smoke_config") == expected_old_config and registration.get("taps") == taps
                and freeze.get("config") == taps and registration.get("seeds") == list(SEEDS)
                and registration.get("native_layers") == [12, 13] and registration.get("extra_loops") == [0, 1]
                and registration.get("gold_nll_status") == "NOT_COLLECTED_CAUSAL_GUARD", "OPEN native model/operator/scorer/cap configuration differs")
        for field in ("python", "torch", "transformers"):
            require(registration.get("runtime", {}).get(field) == binding["runtime"].get(field), "OPEN actual runtime differs: " + field)
        require(registration.get("runtime", {}).get("tf32") is False and registration.get("runtime", {}).get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "OPEN actual FP32 math contract differs")
        protocol_path = PACKAGE / "OPEN_NATIVE_REGISTRATION.md"
        inputs.append(protocol_path)
        protocol_key = protocol_path.relative_to(REPO).as_posix()
        require(admission.get("dispatch_hashes", {}).get(protocol_key) == sha(protocol_path), "OPEN primary protocol binding differs")
        require(freeze.get("bank_sha256") == admission.get("open_bank_sha256") and freeze.get("token_binding_sha256") == sha(token_path)
                and freeze.get("runner_sha256") == admission.get("dispatch_hashes", {}).get((PACKAGE / "open_native.py").relative_to(REPO).as_posix()), "OPEN fixed/pool freeze source/data binding differs")
        require(native.get("loaded_state_sha256") == binding.get("historical_expected_loaded_fp32_state_sha256") and qualifier.get("after_state_sha256") == native.get("loaded_state_sha256"), "OPEN actual loaded/qualified state differs")
        code = Path(native.get("loaded_custom_code_path", "__MISSING__")).resolve()
        require((result / "hf_cache").resolve() in code.parents and code.is_file(), "OPEN actual custom-code path missing/unscoped")
        if (result / "hf_cache").resolve() in code.parents and code.is_file():
            inputs.append(code)
            require(sha(code) == native.get("loaded_custom_code_sha256") == binding["checkpoint_files"]["modeling_supa_a11.py"], "OPEN actual custom-code identity differs")
        require(admission.get("checkpoint_files") == binding["checkpoint_files"] and admission.get("source") == binding["source_path"], "OPEN canonical model binding differs")
        require(admission.get("training") is False and admission.get("native_test") is False
                and admission.get("max_new_tokens") == 8 and admission.get("max_open_input_tokens") == 350
                and admission.get("gpu_allocated_ceiling_bytes") == 5*GIB and admission.get("worker_rss_ceiling_bytes") == 6*GIB,
                "OPEN authority/shape/resource contract differs")
        rows, e = read_journal(result / "outcomes.jsonl"); errors.extend(e)
        traces, e = read_journal(result / "trajectories.jsonl"); errors.extend(e)
        forwards, e = read_journal(result / "forwards.jsonl"); errors.extend(e)
        resources, e = read_journal(run_root / "resources.jsonl"); errors.extend(e)
        budget, e = read_journal(PACKAGE / "BUDGET_LEDGER.jsonl"); errors.extend(e)
        for path in (result / "outcomes.jsonl", result / "trajectories.jsonl", result / "forwards.jsonl", run_root / "resources.jsonl", PACKAGE / "BUDGET_LEDGER.jsonl"):
            if path.is_file(): inputs.append(path)
        prefix = bytearray()
        if (result / "trajectories.jsonl").is_file():
            for line in (result / "trajectories.jsonl").read_bytes().splitlines(keepends=True):
                try: event = json.loads(line)
                except ValueError: errors.append("OPEN partial trajectory line at freeze audit"); break
                if event.get("arm") == "A5": break
                prefix.extend(line)
        require(hashlib.sha256(prefix).hexdigest() == freeze.get("generation_trajectory_sha256_at_freeze"), "OPEN pre-validation freeze trajectory digest differs")
        report = aggregate(rows, traces, bank, smoke, token["rows"], freeze, native, worker, qualifier, a5, forwards, resources, budget, errors)
        evidence = {"analyzer_sha256": sha(__file__),
                    "imported_helper_sha256": {name: sha(PACKAGE / name) for name in ("analysis.py", "gpu_analysis.py", "open_bank.py")},
                    "input_sha256": {str(path): sha(path) for path in inputs if path.is_file()},
                    "native_model_loads": 0, "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()}
        for name, value in (("REPORT.json", report), ("MANIFEST.json", evidence)):
            with (out / name).open("x", encoding="utf-8") as stream: json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        (out / "REPORT.md").write_text("# OPEN intervention analysis\n\n" + report["status"] + "\n\nStrict scalar correctness; equal category weights; DEV-frozen validation comparator.\nRetrospective validation envelopes/oracles are diagnostic. No broad intelligence, confirmation or production-speed claim.\n", encoding="utf-8")
        receipt.update(status="ANALYZED", matrix_complete=report["complete"], report_sha256=sha(out / "REPORT.json"))
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with (out / "receipt.json").open("x", encoding="utf-8") as stream: json.dump(receipt, stream, indent=2, sort_keys=True); stream.write("\n")
    return 0 if receipt["status"] == "ANALYZED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); raise SystemExit(run(args.run_root, args.out))
