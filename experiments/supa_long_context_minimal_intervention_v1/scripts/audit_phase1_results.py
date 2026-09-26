#!/usr/bin/env python3
"""Independently rescore Phase 1 per-arm per_example.jsonl on CPU only.

Refuse an actionable gate unless A/B/C are complete, paired, metadata-consistent,
and baseline-valid. Use exact rational arithmetic at decision thresholds.
Existing summaries/receipts are comparison targets and completion evidence only;
every accuracy or gate input is recomputed from raw per-example predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from audit_result_numbers import LAB, REPO, bootstrap, digest, load_normalize, read_jsonl, summarize

ARMS = ("A", "B", "C")
CLASSES = ("counting_sets", "conditional_rules")
SEED = 20260905
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
CORE_PATHS = {
    "a11": REPO.parent / "Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze",
    "a1": REPO.parent / "Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1-1.0.0-devfreeze",
}
PUBLISHED = {"a11": {"showcase": 45, "stability_test": 125, "all": 170},
             "a1": {"showcase": 43, "stability_test": 125, "all": 168}}
CHECKPOINT_NAMES = {"model.safetensors", "config.json", "generation_config.json",
                    "modeling_supa_a11.py", "modeling_supa_a1.py", "tokenizer_config.json",
                    "tokenizer.json", "vocab.json", "merges.txt", "CORE.json", "FILES.json"}
SUM_RESOURCES = ("wall_seconds", "model_loads", "optimizer_steps_attempted",
                 "optimizer_steps_applied", "steps_skipped_by_scaler",
                 "steps_skipped_by_safety_check", "examples_consumed")
PEAK_RESOURCES = ("peak_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
LOAD_RESOURCES = ("model_load_attempts", "model_load_seconds")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    """Stream large frozen weights without importing/loading the model."""
    path = Path(path)
    stat = path.stat()
    return _file_digest(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=None)
def _file_digest(path, mtime_ns, size):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_source_hashes():
    paths = sorted([*LAB.glob("src/**/*.py"), *LAB.glob("scripts/*.py"),
                    *LAB.glob("configs/*.json")])
    return {p.relative_to(REPO).as_posix(): file_digest(p) for p in paths}


def current_checkpoint_hashes(core):
    return {name: file_digest(CORE_PATHS[core] / name) for name in sorted(CHECKPOINT_NAMES)
            if (CORE_PATHS[core] / name).is_file()}


def current_data_hashes():
    paths = {
        "curriculum": REPO.parent / "Chat-Bubble/Rineya_LCR_SFT_v1/Rineya_LCR_SFT_v1/data/lcr_sft_v1.jsonl",
        "target_suite": REPO / "Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl",
        **{f"preservation_{cls}": REPO / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/by_class" / f"{cls}.jsonl"
           for cls in CLASSES},
    }
    return {name: file_digest(path) for name, path in paths.items()}


def short_normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower().rstrip(".")).strip()


def preservation_score(prediction, reference):
    p, r = short_normalize(prediction), short_normalize(reference)
    pn, rn = NUMBER.findall(prediction), NUMBER.findall(reference)
    matched = bool(pn) and pn[-1] == rn[-1] if rn else bool(r) and r in p
    return {"matched": bool(matched), "exact": p == r,
            "has_reference_numbers": bool(rn),
            "response_words": len(re.findall(r"\b[\w'-]+\b", prediction))}


def expected_preservation(smoke):
    rng = random.Random(SEED)
    selected = {}
    for cls in CLASSES:
        path = REPO / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/by_class" / f"{cls}.jsonl"
        rows = sorted(read_jsonl(path), key=lambda r: r["id"])
        for row in rng.sample(rows, min(8 if smoke else 100, len(rows))):
            if row["messages"][-1]["role"] == "assistant":
                reference = row["messages"][-1]["content"]
                selected[row["id"]] = {"example_id": row["id"], "reference": reference,
                                        "task_family": cls, "messages": row["messages"][:-1],
                                        "reference_numbers": NUMBER.findall(reference)}
    return selected


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def physical_jsonl(path, *, interrupted):
    """A hard kill may tear its final append; completed/logical files stay strict."""
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if interrupted and index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                return rows, True
            raise
    return rows, False


def mismatch(issues, scope, field, stored, recomputed):
    if stored != recomputed:
        issues.append({"scope": scope, "field": field, "stored": stored, "recomputed": recomputed})


def trajectory_checks(training, resource, config):
    """Reconcile one logical dose; never sum resumed prefixes across processes."""
    training = training or {}
    trajectory = training.get("trajectory", [])
    attempted = len(trajectory)
    skipped = sum(r.get("step_skipped_by_scaler") is True for r in trajectory)
    safety = sum(r.get("step_skipped_by_safety_check") is True for r in trajectory)
    applied = sum(r.get("optimizer_update_applied") is True for r in trajectory)
    finite = sum(r.get("finite_nonzero_gradient") is True and r.get("gradients_finite") is True
                 and type(r.get("grad_norm")) in (int, float) and math.isfinite(r["grad_norm"])
                 for r in trajectory)
    accum = config.get("grad_accum", 0)
    recomputed = {"attempted_from_trajectory": attempted, "scaler_skipped_from_trajectory": skipped,
                  "safety_skipped_from_trajectory": safety, "applied_from_trajectory": applied,
                  "finite_nonzero_gradient_attempts": finite}
    checks = {
        "training_present": bool(training),
        "attempt_count": attempted == config.get("max_steps") == training.get("max_steps") == training.get("optimizer_attempts"),
        "attempt_sequence": [r.get("step") for r in trajectory] == list(range(1, attempted + 1))
            and [r.get("attempt") for r in trajectory] == list(range(1, attempted + 1)),
        "applied_count": applied == training.get("steps_applied"),
        "skip_count": skipped == training.get("steps_skipped_by_scaler"),
        "safety_skip_count": safety == training.get("steps_skipped_by_safety_check"),
        "attempt_partition": attempted == applied + skipped + safety and all(
            all(type(r.get(field)) is bool for field in ("optimizer_update_applied", "step_skipped_by_scaler", "step_skipped_by_safety_check"))
            and sum(r[field] for field in ("optimizer_update_applied", "step_skipped_by_scaler", "step_skipped_by_safety_check")) == 1 for r in trajectory),
        "only_finite_nonzero_updates_applied": all(not r.get("optimizer_update_applied") or (
            r.get("finite_nonzero_gradient") is True and r.get("gradients_finite") is True
            and type(r.get("grad_norm")) in (int, float) and math.isfinite(r["grad_norm"])) for r in trajectory),
        "gradient_reaches_adapter": finite > 0 and applied > 0,
        "gradient_checkpointing": training.get("gradient_checkpointing") is True,
        "fixed_training_options": all(training.get(k) == config.get(k) for k in ("grad_accum", "lr", "seed")),
        "examples_consumed": training.get("examples_consumed") == attempted * accum,
        "examples_applied": training.get("examples_in_applied_updates") == applied * accum,
        "resource_attempt_count": resource.get("optimizer_steps_attempted") == attempted,
        "resource_applied_count": resource.get("optimizer_steps_applied") == applied,
        "resource_skip_count": resource.get("steps_skipped_by_scaler") == skipped,
        "resource_safety_skip_count": resource.get("steps_skipped_by_safety_check") == safety,
        "resource_examples_consumed": resource.get("examples_consumed") == attempted * accum,
        "gate_disabled_for_training": training.get("gate_disabled_for_training") is True,
        "gate_on_for_validation": training.get("gate_on_for_validation_evaluations") is True,
        "gate_restored": training.get("gate_restored_after_training") is True,
    }
    cumulative_applied = cumulative_skipped = cumulative_safety = applied_tokens = 0
    cumulative_valid = True
    for attempt, row in enumerate(trajectory, 1):
        cumulative_applied += row.get("optimizer_update_applied") is True
        cumulative_skipped += row.get("step_skipped_by_scaler") is True
        cumulative_safety += row.get("step_skipped_by_safety_check") is True
        tokens = row.get("answer_tokens")
        if type(tokens) is not int or tokens < 1:
            cumulative_valid = False
        elif row.get("optimizer_update_applied") is True:
            applied_tokens += tokens
        expected = {"examples_consumed": attempt * accum, "optimizer_updates_applied": cumulative_applied,
                    "steps_skipped_by_scaler_total": cumulative_skipped,
                    "steps_skipped_by_safety_check_total": cumulative_safety,
                    "examples_in_applied_updates": cumulative_applied * accum,
                    "answer_tokens_in_applied_updates": applied_tokens}
        cumulative_valid &= all(row.get(k) == v for k, v in expected.items())
    checks["trajectory_cumulative_accounting"] = cumulative_valid
    checks["answer_tokens_applied"] = training.get("answer_tokens_in_applied_updates") == applied_tokens
    checks["finite_nonzero_count"] = training.get("finite_nonzero_gradient_attempts") == finite
    period = max(1, config.get("max_steps", 0) // 4)
    checks["complete_validation_schedule"] = [r.get("step") for r in training.get("validation", [])] == [
        s for s in range(1, config.get("max_steps", 0) + 1) if s % period == 0 or s == config.get("max_steps")]
    return checks, recomputed


def aggregate_resources(resources):
    """Sum only observed costs; missing measurements stay explicitly incomplete."""
    resources = [r for r in resources if isinstance(r, dict)]
    result = {}
    incomplete = set()
    load_keys = [key for key in LOAD_RESOURCES if any(key in r for r in resources)]
    for key in (*SUM_RESOURCES, *PEAK_RESOURCES, *load_keys):
        values = [r.get(key) for r in resources]
        valid = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
        if len(valid) != len(values):
            incomplete.add(key)
        result[key] = max(valid, default=0) if key in PEAK_RESOURCES else sum(valid)
    upper = [r.get("model_loads_upper_bound", r.get("model_loads")) for r in resources]
    result["model_loads_upper_bound"] = sum(upper) if all(type(v) is int and v >= 0 for v in upper) else None
    for r in resources:
        incomplete.update(r.get("incomplete_metric_fields", []))
    result["resource_accounting_complete"] = bool(resources) and not incomplete and all(
        r.get("resource_accounting_complete", True) is True for r in resources)
    result["incomplete_metric_fields"] = sorted(incomplete)
    result["unknown_segment_ids"] = [segment for r in resources for segment in r.get("unknown_segment_ids", [])]
    result["cost_interpretation"] = "observed totals; incomplete metrics are lower bounds, not exact costs" if not result["resource_accounting_complete"] else "complete stored process measurements"
    return result


def segment_checks(out, arm, manifest, resource, info):
    """Independently inspect physical evidence behind a logical arm receipt."""
    if manifest.get("schema_version") != 3:
        return {"legacy_one_model_load": resource.get("model_loads") == 1,
                "legacy_has_no_unchecked_segments": not (out / "segments").exists()}, None
    checks = {}
    identity = json_file(out / "identity.json") or {}
    directories = sorted(p for p in (out / "segments").glob("*") if p.is_dir())
    ids = [p.name for p in directories]
    checks["physical_segment_inventory"] = bool(ids) and manifest.get("segment_ids") == ids
    checks["physical_segment_ordinals"] = all(name.startswith(f"{i:06d}-") for i, name in enumerate(ids, 1))
    checks["physical_resume_count"] = manifest.get("resume_count") == len(ids) - 1
    checks["immutable_identity"] = identity.get("schema_version") == 3 and bool(identity.get("run_id")) and all(
        identity.get(k) == manifest.get(k) for k in ("run_id", "config", "source_sha256", "canonical_core", "checkpoint_fingerprints"))
    checks["identity_checkpoint"] = identity.get("checkpoint_sha256") == manifest.get("checkpoint_sha256_before")
    checks["identity_dataset"] = identity.get("dataset_fingerprint") == info.get("dataset_fingerprint")
    checks["identity_frozen_parameters"] = bool(identity.get("frozen_parameter_sha256")) and identity.get("frozen_parameter_sha256") == info.get("frozen_base", {}).get("before")
    checks["info_run_identity"] = info.get("run_id") == identity.get("run_id") and bool(ids) and info.get("segment_id") == ids[-1]
    data, initializer = identity.get("dataset_fingerprint", {}), identity.get("adapter_initialization", {})
    encoded = json_file(out / "encoded_data.json") or {}
    checks["immutable_encoded_data"] = bool(encoded.get("train")) and bool(encoded.get("dev")) and all(
        data.get(key) == fingerprint(encoded.get(key))
        for key in ("train", "dev"))
    checks["encoded_data_counts"] = all(data.get(f"{key}_count") == len(encoded.get(key, [])) for key in ("train", "dev"))
    if arm != "A":
        train_rows = encoded.get("train", [])
        accum = manifest.get("config", {}).get("grad_accum", 0)
        trajectory = (info.get("training") or {}).get("trajectory", [])
        checks["ordered_training_answer_tokens"] = bool(train_rows) and type(accum) is int and accum > 0 and all(
            record.get("answer_tokens") == sum(train_rows[i % len(train_rows)]["n_answer_tokens"] for i in range((attempt - 1) * accum, attempt * accum))
            for attempt, record in enumerate(trajectory, 1))
    expected_fingerprints = {
        "recipe_sha256": fingerprint({"scientific_recipe": identity.get("scientific_recipe"), "dataset_fingerprint": data}),
        "train_data_sha256": data.get("train"), "eval_data_sha256": data.get("dev"),
        "base_checkpoint_sha256": fingerprint(identity.get("checkpoint_sha256")),
        "adapter_initialization_sha256": initializer.get("sha256"),
    }
    checks["checkpoint_fingerprints"] = identity.get("checkpoint_fingerprints") == expected_fingerprints and info.get("checkpoint_fingerprints") == expected_fingerprints
    checks["initializer_receipt"] = initializer.get("algorithm") == "sha256-native-trainable-parameters-v1" and re.fullmatch(r"[0-9a-f]{64}", initializer.get("sha256", "")) is not None
    recipe = identity.get("scientific_recipe", {})
    checks["scientific_recipe_identity"] = all(recipe.get(k) == v for k, v in identity.get("config", {}).items()) and recipe.get("source_sha256") == identity.get("source_sha256")
    checks["scientific_recipe_training_defaults"] = all(recipe.get(k) == v for k, v in {
        "optimizer": "AdamW", "weight_decay": 0.0, "max_grad_norm": 1.0,
        "gradient_checkpointing": True, "adapter_dtype": "float32",
        "initialization": {"A": "none", "B": "lora", "C": "nora"}[arm],
    }.items())
    segments = []
    previous_end = 0
    previous_was_terminal = True
    budget = manifest.get("config", {}).get("max_steps", 0)
    for directory in directories:
        sid = directory.name
        saved = json_file(directory / "segment_identity.json") or {}
        start = json_file(directory / "started_manifest.json")
        terminal = json_file(directory / "run_manifest.json")
        evidence = terminal or start or saved
        prefix = f"segment_{sid}"
        checks[f"{prefix}_identity"] = bool(saved) and all(saved.get(k) == identity.get(k) for k in ("run_id", "config", "source_sha256")) and saved.get("segment_id") == sid and saved.get("checkpoint_sha256_before") == identity.get("checkpoint_sha256")
        checks[f"{prefix}_source_and_config"] = all(
            all(record.get(k) == identity.get(k) for k in ("run_id", "config", "source_sha256", "canonical_core"))
            and record.get("segment_id") == sid and record.get("checkpoint_sha256_before") == identity.get("checkpoint_sha256")
            for record in (saved, start, terminal) if record)
        resumed = evidence.get("resumed_from_attempt")
        end = evidence.get("end_committed_attempt")
        checks[f"{prefix}_attempt_boundary"] = type(resumed) is int and previous_end <= resumed <= budget and (not previous_was_terminal or resumed == previous_end) and (end is None or type(end) is int and resumed <= end <= budget)
        checks[f"{prefix}_resume_checkpoint_reference"] = resumed == 0 or re.fullmatch(r"[0-9a-f]{64}", evidence.get("resumed_from_checkpoint_sha256") or "") is not None
        if type(resumed) is int:
            previous_end = end if type(end) is int else resumed
        previous_was_terminal = terminal is not None and type(end) is int
        if terminal:
            checks[f"{prefix}_terminal_integrity"] = terminal.get("status") in ("FAILED", "COMPLETE") and terminal.get("canonical_checkpoint_unchanged") is True and terminal.get("checkpoint_sha256_after") == identity.get("checkpoint_sha256") and bool(terminal.get("end_utc"))
            checks[f"{prefix}_resource_sidecar"] = json_file(directory / "resource.json") == terminal.get("resource")
        entry = json_file(directory / "training_entry.json")
        if entry:
            checks[f"{prefix}_training_identity"] = arm != "A" and entry.get("run_id") == identity.get("run_id") and entry.get("checkpoint_fingerprints") == expected_fingerprints and entry.get("total_max_steps") == budget and entry.get("resume_from_checkpoint") is bool(evidence.get("resumed_from_checkpoint_sha256"))
        if entry or terminal and terminal.get("status") == "COMPLETE":
            checks[f"{prefix}_initialization_verified"] = json_file(directory / "adapter_initialization.json") == initializer
            frozen = json_file(directory / "frozen_parameters_before.json") or {}
            checks[f"{prefix}_frozen_parameters_verified"] = frozen.get("algorithm") == "sha256-native-frozen-parameters-v1" and frozen.get("sha256") == identity.get("frozen_parameter_sha256")
        if arm != "A" and terminal and terminal.get("status") == "COMPLETE":
            checks[f"{prefix}_training_entry_present"] = bool(entry)
            checks[f"{prefix}_completed_budget"] = end == budget
        physical_progress = directory / "training_progress.jsonl"
        if physical_progress.is_file() and type(resumed) is int:
            physical_events, torn = physical_jsonl(physical_progress, interrupted=terminal is None)
            logical_trajectory = (info.get("training") or {}).get("trajectory", [])
            actual_attempts = [{k: v for k, v in r.items() if k != "event"} for r in physical_events if r.get("event") == "optimizer_attempt"]
            checks[f"{prefix}_physical_progress_prefix"] = actual_attempts == logical_trajectory[resumed:resumed + len(actual_attempts)]
        ledger = json_file(directory / "model_load.json")
        lower = upper = attempts = 0
        load_seconds = 0
        if ledger:
            loaded = ledger.get("status") == "LOADED"
            lower, upper, attempts = int(loaded), 1, 1
            load_seconds = ledger.get("model_load_seconds") if loaded else None
            checks[f"{prefix}_model_load_ledger"] = ledger.get("status") in ("LOADING", "LOADED") and ledger.get("model_loads") == lower and ledger.get("model_loads_upper_bound") == upper and ledger.get("model_load_attempts") == attempts
        if terminal and terminal.get("status") == "COMPLETE":
            checks[f"{prefix}_completed_load"] = lower == upper == 1
        recorded = (terminal or {}).get("resource", {})
        if arm != "A" and terminal and type(resumed) is int and type(end) is int:
            selected = (info.get("training") or {}).get("trajectory", [])[resumed:end]
            expected_dose = {"optimizer_steps_attempted": end - resumed,
                             "optimizer_steps_applied": sum(r.get("optimizer_update_applied") is True for r in selected),
                             "steps_skipped_by_scaler": sum(r.get("step_skipped_by_scaler") is True for r in selected),
                             "steps_skipped_by_safety_check": sum(r.get("step_skipped_by_safety_check") is True for r in selected),
                             "examples_consumed": (end - resumed) * manifest["config"].get("grad_accum", 0)}
            checks[f"{prefix}_physical_committed_dose_delta"] = all(recorded.get(k) == v for k, v in expected_dose.items())
        complete = bool(terminal) and recorded.get("resource_accounting_complete", True) is True and lower == upper
        segment_resource = {**recorded, "model_loads": lower, "model_loads_upper_bound": upper,
                            "model_load_attempts": attempts, "model_load_seconds": load_seconds,
                            "resource_accounting_complete": complete,
                            "unknown_segment_ids": [] if complete else [sid]}
        if not terminal:
            segment_resource.update({key: None for key in ("wall_seconds", *PEAK_RESOURCES)})
        segments.append({"segment_id": sid, "status": (terminal or {}).get("status", "INTERRUPTED"),
                         "resource": segment_resource, "resumed_from_attempt": resumed,
                         "end_committed_attempt": end})
    checks["final_physical_segment_complete"] = bool(segments) and segments[-1]["status"] == "COMPLETE"
    if directories:
        checks["final_segment_evaluation_artifacts"] = all((directories[-1] / name).is_file() and (out / name).is_file()
            and file_digest(directories[-1] / name) == file_digest(out / name)
            for name in (f"arm_{arm}_info.json", f"arm_{arm}_target.jsonl", f"arm_{arm}_preservation.jsonl", "per_example.jsonl", "dataset.json"))
    aggregate = aggregate_resources([s["resource"] for s in segments])
    for key in ("wall_seconds", "model_loads", "model_loads_upper_bound", *LOAD_RESOURCES, *PEAK_RESOURCES):
        expected, actual = aggregate[key], resource.get(key)
        checks[f"physical_resource_{key}"] = actual == expected or key in ("wall_seconds", "model_load_seconds") and type(actual) in (int, float) and math.isclose(actual, expected, abs_tol=0.001)
    checks["physical_resource_completeness"] = resource.get("resource_accounting_complete") is aggregate["resource_accounting_complete"]
    checks["physical_resource_unknown_segments"] = resource.get("unknown_segment_ids") == aggregate["unknown_segment_ids"]
    if not aggregate["resource_accounting_complete"]:
        checks["physical_resource_lower_bounds_explicit"] = resource.get("numeric_costs_are_lower_bounds") is True
        unknown_costs = {key for s in segments for key in ("wall_seconds", "model_load_seconds", *PEAK_RESOURCES)
                         if s["resource"].get(key) is None}
        checks["physical_resource_missing_metrics_explicit"] = unknown_costs <= set(resource.get("incomplete_metric_fields", []))
    checks["committed_dose_matches_logical_resource"] = all(manifest.get("committed_training", {}).get(k) == resource.get(k) for k in SUM_RESOURCES[2:])
    if arm != "A":
        training = info.get("training") or {}
        trajectory, validation = training.get("trajectory", []), training.get("validation", [])
        receipt = manifest.get("checkpoint_receipt") or {}
        checkpoint_path = out / "checkpoints/latest.pt"
        checks["latest_checkpoint_file"] = checkpoint_path.is_file() and receipt.get("sha256") == file_digest(checkpoint_path)
        checks["latest_checkpoint_complete"] = receipt.get("completed_attempts") == budget and "pending_validation" in receipt and receipt["pending_validation"] is None
        checks["latest_checkpoint_trajectory"] = receipt.get("trajectory_sha256") == fingerprint(trajectory) and receipt.get("evals_sha256") == fingerprint(validation)
        checkpoint = json_file(out / "committed_checkpoint.json") or {}
        checks["committed_checkpoint_receipt"] = checkpoint.get("receipt") == receipt
        progress = checkpoint.get("progress", {})
        checks["committed_checkpoint_records"] = progress.get("trajectory") == trajectory and progress.get("evals") == validation and progress.get("completed_attempts") == budget and progress.get("cursor") == budget * manifest["config"].get("grad_accum", 0) and "pending_validation" in progress and progress["pending_validation"] is None
        val_by_step = {r["step"]: r for r in validation}
        events = []
        for record in trajectory:
            events.append({"event": "optimizer_attempt", **record})
            if record.get("step") in val_by_step:
                events.append({"event": "validation", **val_by_step[record["step"]]})
        progress_path = out / "training_progress.jsonl"
        checks["logical_progress_no_duplicate_prefix"] = progress_path.is_file() and read_jsonl(progress_path) == events
    return checks, {"segments": segments, "physical_cost_aggregation": aggregate}


def incomplete_resources(out):
    """Inspect current physical files even when a kill prevented root publication."""
    directories = sorted(p for p in (out / "segments").glob("*") if p.is_dir())
    if not directories:
        return json_file(out / "resource.json")
    resources = []
    for directory in directories:
        terminal = json_file(directory / "run_manifest.json")
        ledger = json_file(directory / "model_load.json") or {}
        recorded = dict((terminal or {}).get("resource", {}))
        loaded, loading = ledger.get("status") == "LOADED", ledger.get("status") == "LOADING"
        exact = bool(terminal) and recorded.get("resource_accounting_complete", True) is True and not loading
        recorded.update(model_loads=int(loaded), model_loads_upper_bound=int(loaded or loading),
                        model_load_attempts=int(loaded or loading), model_load_seconds=ledger.get("model_load_seconds") if ledger else 0,
                        resource_accounting_complete=exact,
                        unknown_segment_ids=[] if exact else [directory.name])
        for key in ("wall_seconds", *PEAK_RESOURCES):
            if not terminal:
                recorded[key] = None
        resources.append(recorded)
    result = aggregate_resources(resources)
    # Committed dose is one logical quantity; physical prefixes are never added.
    saved = json_file(out / "committed_checkpoint.json") or {}
    checkpoint_path = out / "checkpoints/latest.pt"
    current = checkpoint_path.is_file() and saved.get("receipt", {}).get("sha256") == file_digest(checkpoint_path)
    fields = {"optimizer_steps_attempted": "completed_attempts", "optimizer_steps_applied": "applied_steps",
              "steps_skipped_by_scaler": "skipped_steps", "steps_skipped_by_safety_check": "safety_skipped_steps",
              "examples_consumed": "consumed"}
    for key, field in fields.items():
        result[key] = saved.get("progress", {}).get(field) if current else None
    result["logical_dose_receipt_current"] = current
    result["resource_accounting_complete"] = False
    result["cost_interpretation"] = "physical observed cost lower bounds; logical dose unknown unless its sidecar matches latest.pt"
    return result


def audit_arm(stage, arm, core, smoke, targets, preservation, normalize):
    out = LAB / "results" / stage / f"arm_{arm}"
    required = ["per_example.jsonl", "run_manifest.json", "resource.json", f"arm_{arm}_info.json"]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        return {"status": "INCOMPLETE", "missing_files": missing,
                "recorded_resource": incomplete_resources(out)}, None
    rows = read_jsonl(out / "per_example.jsonl")
    manifest, resource, info = [json_file(out / p) for p in required[1:]]
    issues = []
    for kind in ("target", "preservation"):
        sidecar = out / f"arm_{arm}_{kind}.jsonl"
        typed_rows = sorted(({k: v for k, v in r.items() if k not in ("evaluation", "arm")}
                             for r in rows if r.get("evaluation") == kind), key=lambda r: r["example_id"])
        sidecar_rows = sorted(read_jsonl(sidecar), key=lambda r: r["example_id"]) if sidecar.is_file() else None
        mismatch(issues, arm, f"{kind}_sidecar_matches_per_example", sidecar_rows, typed_rows)
    target_rows, preservation_rows = {}, {}
    for row in rows:
        kind, row_id = row.get("evaluation"), row["example_id"]
        mismatch(issues, row_id, "arm", row.get("arm"), arm)
        by = target_rows if kind == "target" else preservation_rows if kind == "preservation" else None
        if by is None:
            issues.append({"scope": row_id, "field": "evaluation", "stored": kind, "recomputed": "target or preservation"})
            continue
        if row_id in by:
            issues.append({"scope": row_id, "field": "duplicate", "stored": kind, "recomputed": "unique"})
        by[row_id] = dict(row)
    mismatch(issues, arm, "target_ids", sorted(target_rows), sorted(targets))
    mismatch(issues, arm, "preservation_ids", sorted(preservation_rows), sorted(preservation))
    for row_id, row in target_rows.items():
        if row_id not in targets:
            continue
        gold = targets[row_id]
        normalized = normalize(row["raw_prediction"])
        correct = normalized in {normalize(a) for a in gold["accepted_answers"]}
        malformed = (not row["raw_prediction"].strip() or "<think>" in row["raw_prediction"]
                     or "</think>" in row["raw_prediction"]
                     or len(re.findall(r"\b[\w'-]+\b", row["raw_prediction"].strip())) > 12)
        for field, computed in [("correct", correct), ("normalized_prediction", normalized),
                                 ("malformed", malformed), ("condition", "A_full")]:
            mismatch(issues, row_id, field, row.get(field), computed)
            row[field] = computed
        for field in ("split", "variant", "task_type", "answerable", "question"):
            mismatch(issues, row_id, field, row.get(field), gold[field])
    for row_id, row in preservation_rows.items():
        if row_id not in preservation:
            continue
        gold = preservation[row_id]
        for field in ("reference", "task_family"):
            mismatch(issues, row_id, field, row.get(field), gold[field])
        scores = preservation_score(row["raw_prediction"], gold["reference"])
        for field, computed in scores.items():
            mismatch(issues, row_id, field, row.get(field), computed)
            row[field] = computed
    fresh = {"target": summarize([r["correct"] for r in target_rows.values()]) if target_rows else None,
             "preservation": {key: summarize([r[key] for r in preservation_rows.values()]) if preservation_rows else None
                              for key in ("matched", "exact")},
             "preservation_by_class": {
                 cls: summarize([r["matched"] for r in preservation_rows.values() if r["task_family"] == cls])
                 for cls in CLASSES if any(r["task_family"] == cls for r in preservation_rows.values())}}
    if fresh["target"]:
        mismatch(issues, arm, "info.target", info.get("target"), fresh["target"])
    for field in ("matched", "exact"):
        mismatch(issues, arm, f"info.preservation.{field}", info.get("preservation", {}).get(field), fresh["preservation"][field])
    receipt_checks = {
        "manifest_complete": manifest.get("status") == "COMPLETE",
        "checkpoint_preservation_receipt": manifest.get("canonical_checkpoint_unchanged") is True
            and bool(manifest.get("checkpoint_sha256_before"))
            and manifest.get("checkpoint_sha256_before") == manifest.get("checkpoint_sha256_after"),
        "core_path": Path(info.get("loaded_core_path", "")).resolve() == CORE_PATHS[core].resolve(),
        "dtype": info.get("loaded_dtype") == ("float16" if core == "a11" else "bfloat16"),
        "evaluation_gate_on": info.get("evaluation_gate_on") is True,
        "frozen_base_receipt": info.get("frozen_base", {}).get("unchanged") is True
            and bool(info.get("frozen_base", {}).get("before"))
            and info.get("frozen_base", {}).get("before") == info.get("frozen_base", {}).get("after"),
        "manifest_core": manifest.get("config", {}).get("core") == core,
        "requested_step_budget": manifest.get("config", {}).get("max_steps") == (6 if smoke else 300),
    }
    config = manifest.get("config", {})
    source_hashes = current_source_hashes()
    checkpoint_hashes = current_checkpoint_hashes(core)
    expected_target_items = [{"example_id": row_id, "condition": "A_full",
                              **{key: gold[key] for key in ("split", "variant", "task_type", "answerable", "context", "question", "accepted_answers")}}
                             for row_id, gold in targets.items()]
    dataset_fingerprint = info.get("dataset_fingerprint", {})
    dataset_receipt = json_file(out / "dataset.json") or {}
    receipt_checks.update({
        "source_hashes_current": bool(source_hashes) and manifest.get("source_sha256") == source_hashes,
        "canonical_core_path": Path(manifest.get("canonical_core", "")).resolve() == CORE_PATHS[core].resolve(),
        "canonical_checkpoint_hashes_current": bool(checkpoint_hashes)
            and manifest.get("checkpoint_sha256_before") == checkpoint_hashes
            and manifest.get("checkpoint_sha256_after") == checkpoint_hashes,
        "data_hashes_current": config.get("data_sha256") == current_data_hashes(),
        "external_scorer_current": config.get("external_scorer_sha256") == file_digest(REPO / "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py"),
        "fixed_recipe": all(config.get(k) == v for k, v in {
            "arm": arm, "smoke": smoke, "rank": 4, "alpha": 8, "dropout": 0.0,
            "trainable_layers": [17, 18, 19], "protected_layers": [0, 1, 2],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "seed": 17, "max_length": 3840, "grad_accum": 8, "lr": 1e-4,
            "train_packs": 8 if smoke else 600, "dev_packs": 4 if smoke else 40,
            "step_budget_unit": "optimizer attempts", "target_context": "FULL",
            "preservation_classes": list(CLASSES), "preservation_per_class": 8 if smoke else 100,
            "eval_max_new_tokens": 24, "dtype": "float16" if core == "a11" else "bfloat16",
        }.items()),
        "evaluation_fingerprint": dataset_fingerprint.get("target") == fingerprint(expected_target_items)
            and dataset_fingerprint.get("preservation") == fingerprint([preservation[i] for i in sorted(preservation)])
            and dataset_fingerprint.get("target_count") == len(targets)
            and dataset_fingerprint.get("preservation_count") == len(preservation),
        "encoded_data_fingerprints_present": all(isinstance(dataset_fingerprint.get(k), str)
            and re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint[k]) is not None for k in ("train", "dev")),
        "fixed_dataset_counts": dataset_fingerprint.get("train_count") == (32 if smoke else 2400)
            and dataset_fingerprint.get("dev_count") == (16 if smoke else 160),
        "dataset_receipt_matches": dataset_receipt.get("fingerprint") == dataset_fingerprint,
        "no_encoded_examples_dropped": all(dataset_receipt.get(key, {}).get("n_dropped_too_long") == 0
            and dataset_receipt.get(key, {}).get("n_input") == dataset_fingerprint.get(f"{key}_count")
            and dataset_receipt.get(key, {}).get("n_kept") == dataset_fingerprint.get(f"{key}_count")
            for key in ("train", "dev")),
        "resource_matches_manifest": manifest.get("resource") == resource,
    })
    training = info.get("training")
    training_recomputed = None
    if arm == "A":
        receipt_checks["no_training"] = training is None and info.get("adapter", {}).get("trainable_params") == 0
        receipt_checks["no_training_dose"] = all(resource.get(k) == 0 for k in SUM_RESOURCES[2:])
    else:
        adapter = info.get("adapter", {})
        adapter_checks = adapter.get("checks", {})
        kind = "lora" if arm == "B" else "nora"
        receipt_checks["adapter_fixed_topology"] = all(adapter.get(k) == v for k, v in {
            "kind": kind, "arm": kind, "layers": [17, 18, 19],
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "rank": 4, "alpha": 8, "dropout": 0.0, "init_seed": 17,
            "trainable_tensor_count": 24, "trainable_params": 122880,
        }.items())
        names = adapter.get("trainable_param_names", [])
        receipt_checks["adapter_only_expected_tensors"] = len(set(names)) == len(names) == 24 and all(
            re.search(r"\.layers\.(17|18|19)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\.", name)
            for name in names)
        receipt_checks["adapter_initialization_invariants"] = adapter_checks.get("trained_layers") == [17, 18, 19] and all(
            adapter_checks.get(key) is True for key in ("all_passed", "trained_layers_as_expected",
                "protected_stem_untouched", "only_attention_projections_trainable",
                "no_mlp_embed_or_head_trainable", "zero_adapter_reproduces_frozen", "lora_B_all_zero_at_init",
                "lora_columns_left_unnormalised" if arm == "B" else "nora_columns_normalised"))
        training_checks, training_recomputed = trajectory_checks(training, resource, config)
        receipt_checks.update(training_checks)
    recovery_checks, recovery_recomputed = segment_checks(out, arm, manifest, resource, info)
    receipt_checks.update(recovery_checks)
    return {
        "status": "VALID" if not issues and all(receipt_checks.values()) else "INVALID",
        "source": str((out / "per_example.jsonl").relative_to(REPO)), "source_sha256": digest(out / "per_example.jsonl"),
        "n_observations": len(rows), "recomputed": fresh, "discrepancies": issues,
        "receipt_checks": receipt_checks, "training_trajectory_recomputation": training_recomputed,
        "recovery_recomputation": recovery_recomputed,
        "recorded_resource": resource, "dataset_fingerprint": info.get("dataset_fingerprint"),
        "config": manifest.get("config", {}),
    }, {"target": target_rows, "preservation": preservation_rows}


def audit_stage(stage, normalize):
    core = "a1" if stage.endswith("_a1") else "a11"
    smoke = "smoke" in stage
    target_file = REPO / "Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl"
    target_list = read_jsonl(target_file)
    targets = {r["id"]: r for r in (target_list[:32] if smoke else target_list)}
    preservation = expected_preservation(smoke)
    arms, records = {}, {}
    for arm in ARMS:
        try:
            arms[arm], records[arm] = audit_arm(stage, arm, core, smoke, targets, preservation, normalize)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            arms[arm], records[arm] = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}"}, None
    result = {"core": core, "smoke": smoke, "arms": arms,
              "decision": {"status": "INCOMPLETE", "winner": None, "phase_2_licensed": False},
              "resource_aggregation_from_arm_receipts": aggregate_resources([arms[arm].get("recorded_resource") for arm in ARMS])}
    if any(a["status"] == "INCOMPLETE" for a in arms.values()):
        result["decision"]["missing_or_incomplete_arms"] = [a for a in ARMS if arms[a]["status"] == "INCOMPLETE"]
        return result
    if any(records[arm] is None for arm in ARMS):
        result["decision"] = {"status": "INVALID", "winner": None, "phase_2_licensed": False,
                              "failed_arms": [arm for arm in ARMS if records[arm] is None]}
        return result
    a = records["A"]["target"]
    split_counts = Counter(row["split"] for row in a.values() if row["correct"])
    baseline = {"showcase": split_counts["showcase"], "stability_test": split_counts["stability_test"], "all": sum(r["correct"] for r in a.values())}
    pair_checks = {
        "all_arm_records_valid": all(r["status"] == "VALID" for r in arms.values()),
        "target_id_pairing": all(set(records[arm]["target"]) == set(a) for arm in ARMS),
        "preservation_id_pairing": all(set(records[arm]["preservation"]) == set(records["A"]["preservation"]) for arm in ARMS),
        "dataset_fingerprint_equal": bool(arms["A"].get("dataset_fingerprint")) and all(arms[arm].get("dataset_fingerprint") == arms["A"].get("dataset_fingerprint") for arm in ARMS),
        "baseline_validity": smoke or baseline == PUBLISHED[core],
    }
    for key in ("max_steps", "grad_accum", "lr", "max_length", "seed", "rank", "alpha", "dropout", "trainable_layers", "dtype", "data_sha256", "external_scorer_sha256"):
        pair_checks[f"matched_config_{key}"] = key in arms["A"]["config"] and all(arms[arm]["config"].get(key) == arms["A"]["config"].get(key) for arm in ARMS)
    result["pair_checks"] = pair_checks
    result["baseline_validity"] = {"recomputed": baseline, "published": None if smoke else PUBLISHED[core], "checked": not smoke}
    if not all(pair_checks.values()):
        result["decision"] = {"status": "INVALID", "winner": None, "phase_2_licensed": False, "failed_checks": [k for k,v in pair_checks.items() if not v]}
        return result
    nt, np = len(a), len(records["A"]["preservation"])
    kt = sum(r["correct"] for r in a.values())
    kp = sum(r["matched"] for r in records["A"]["preservation"].values())
    decision_arms = {}
    target_ids = sorted(a)
    preservation_ids = sorted(records["A"]["preservation"])
    for arm in ("B", "C"):
        arm_t = sum(r["correct"] for r in records[arm]["target"].values())
        arm_p = sum(r["matched"] for r in records[arm]["preservation"].values())
        gain = Fraction(100 * (arm_t - kt), nt)
        rer = Fraction(arm_t - kt, nt - kt) if nt != kt else Fraction(0)
        regression = Fraction(100 * (kp - arm_p), np)
        gain_pass = gain >= 2 or rer >= Fraction(5, 100)
        preservation_pass = regression <= 1
        decision_arms[arm] = {
            "target_gain_points": float(gain), "relative_error_reduction": float(rer),
            "preservation_regression_points": float(regression),
            "target_gain_fraction": str(gain), "relative_error_reduction_fraction": str(rer),
            "preservation_regression_fraction": str(regression),
            "meets_gain_gate": gain_pass, "meets_preservation_gate": preservation_pass,
            "meaningfully_positive": gain_pass and preservation_pass,
            "target_row_bootstrap_vs_A": bootstrap(tuple(a[i]["correct"] for i in target_ids), tuple(records[arm]["target"][i]["correct"] for i in target_ids)),
            "preservation_row_bootstrap_vs_A": bootstrap(tuple(records["A"]["preservation"][i]["matched"] for i in preservation_ids), tuple(records[arm]["preservation"][i]["matched"] for i in preservation_ids)),
        }
    positives = [a for a in ("B", "C") if decision_arms[a]["meaningfully_positive"]]
    best = max(positives, key=lambda arm: decision_arms[arm]["target_gain_points"]) if positives else None
    result["decision"] = {"status": "SMOKE_ONLY" if smoke else "VALID", "arms": decision_arms,
                          "winner": {"B": "LoRA", "C": "NoRA-INIT"}.get(best, "FROZEN"),
                          "phase_2_licensed": bool(positives) and not smoke,
                          "composition_check_licensed": False}
    stored = json_file(LAB / "results" / stage / "summary.json")
    differences = []
    if stored:
        decision = stored.get("decision", {})
        for key in ("winner", "phase_2_licensed"):
            mismatch(differences, stage, f"decision.{key}", decision.get(key), result["decision"][key])
        for arm in ("B", "C"):
            for key in ("meets_gain_gate", "meets_preservation_gate", "meaningfully_positive"):
                mismatch(differences, arm, key, decision.get("arms", {}).get(arm, {}).get(key), decision_arms[arm][key])
    result["stored_summary_decision_discrepancies"] = differences
    combined_resource = json_file(LAB / "results" / stage / "resource.json")
    result["combined_resource_discrepancies"] = []
    if combined_resource:
        for key in (*SUM_RESOURCES, *PEAK_RESOURCES):
            value = result["resource_aggregation_from_arm_receipts"][key]
            if key == "wall_seconds" and isinstance(combined_resource.get(key), (int, float)) and math.isclose(combined_resource[key], value, abs_tol=0.001):
                continue
            mismatch(result["combined_resource_discrepancies"], stage, key, combined_resource.get(key), value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=["phase1_smoke", "phase1", "phase1_a1"])
    parser.add_argument("--output", type=Path, default=LAB / "results/audit_20260905/phase1_numbers.json")
    args = parser.parse_args()
    allowed = {"phase1_smoke", "phase1_smoke_a1", "phase1", "phase1_a1"}
    if not set(args.stages) <= allowed:
        raise SystemExit(f"--stages must be drawn from {sorted(allowed)}")
    normalize, scorer = load_normalize()
    output = {"method": "raw-prediction rescoring, paired IDs, rational threshold decisions; no model/GPU",
              "scorer_sha256": digest(scorer),
              "stages": {stage: audit_stage(stage, normalize) for stage in args.stages},
              "limitations": ["Training/resource/checkpoint checks validate stored receipts and trajectory arithmetic; they do not rerun training or reconstruct historical memory measurements.",
                              "The auditor hashes latest.pt and checks its committed JSON receipt without importing model libraries; optimizer/RNG restoration correctness is covered separately by CPU recovery tests.",
                              "Hard-killed process costs and uncommitted work can be unknown or lower bounds; recovered logical training dose is counted once.",
                              "An audit-passing smoke run never licenses Phase 2.",
                              "Bootstrap intervals are descriptive; scenario clustering and multiple comparisons prevent significance claims."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decisions": {s: r["decision"] for s,r in output["stages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
