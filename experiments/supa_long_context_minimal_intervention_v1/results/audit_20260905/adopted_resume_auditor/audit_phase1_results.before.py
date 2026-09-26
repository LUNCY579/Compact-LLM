#!/usr/bin/env python3
"""Independently rescore Phase 1 per-arm per_example.jsonl on CPU only.

Refuse an actionable gate unless A/B/C are complete, paired, metadata-consistent,
and baseline-valid. Use exact rational arithmetic at decision thresholds.
Existing summaries/receipts are comparison targets and completion evidence only;
every accuracy or gate input is recomputed from raw per-example predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from fractions import Fraction
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
                selected[row["id"]] = {"reference": row["messages"][-1]["content"], "task_family": cls}
    return selected


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def mismatch(issues, scope, field, stored, recomputed):
    if stored != recomputed:
        issues.append({"scope": scope, "field": field, "stored": stored, "recomputed": recomputed})


def audit_arm(stage, arm, core, smoke, targets, preservation, normalize):
    out = LAB / "results" / stage / f"arm_{arm}"
    required = ["per_example.jsonl", "run_manifest.json", "resource.json", f"arm_{arm}_info.json"]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        return {"status": "INCOMPLETE", "missing_files": missing}, None
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
        "one_model_load": resource.get("model_loads") == 1,
        "manifest_core": manifest.get("config", {}).get("core") == core,
        "requested_step_budget": manifest.get("config", {}).get("max_steps") == (6 if smoke else 300),
    }
    training = info.get("training")
    training_recomputed = None
    if arm == "A":
        receipt_checks["no_training"] = training is None and info.get("adapter", {}).get("trainable_params") == 0
    else:
        trajectory = (training or {}).get("trajectory", [])
        attempted = len(trajectory)
        skipped = sum(r.get("step_skipped_by_scaler") is True for r in trajectory)
        safety_skipped = sum(r.get("step_skipped_by_safety_check") is True for r in trajectory)
        applied = sum(r.get("optimizer_update_applied") is True for r in trajectory)
        finite_nonzero = sum(r.get("finite_nonzero_gradient") is True and r.get("gradients_finite") is True
                             and isinstance(r.get("grad_norm"), (int, float)) and math.isfinite(r["grad_norm"])
                             for r in trajectory)
        expected_attempts = manifest.get("config", {}).get("max_steps")
        training_recomputed = {"attempted_from_trajectory": attempted, "scaler_skipped_from_trajectory": skipped,
                               "safety_skipped_from_trajectory": safety_skipped,
                               "applied_from_trajectory": applied, "finite_nonzero_gradient_attempts": finite_nonzero}
        receipt_checks.update({
            "training_present": bool(training),
            "attempt_count": attempted == expected_attempts == (training or {}).get("max_steps"),
            "attempt_sequence": [r.get("step") for r in trajectory] == list(range(1, attempted + 1)),
            "applied_count": applied == (training or {}).get("steps_applied"),
            "skip_count": skipped == (training or {}).get("steps_skipped_by_scaler"),
            "safety_skip_count": safety_skipped == (training or {}).get("steps_skipped_by_safety_check"),
            "attempt_partition": attempted == applied + skipped + safety_skipped and all(
                sum(r.get(field) is True for field in ("optimizer_update_applied", "step_skipped_by_scaler", "step_skipped_by_safety_check")) == 1 for r in trajectory),
            "only_finite_nonzero_updates_applied": all(not r.get("optimizer_update_applied")
                or (r.get("finite_nonzero_gradient") is True and r.get("gradients_finite") is True) for r in trajectory),
            "gradient_reaches_adapter": finite_nonzero > 0 and applied > 0,
            "gradient_checkpointing": (training or {}).get("gradient_checkpointing") is True,
            "examples_consumed": (training or {}).get("examples_consumed") == attempted * (training or {}).get("grad_accum", 0),
            "resource_attempt_count": resource.get("optimizer_steps_attempted") == attempted,
            "resource_applied_count": resource.get("optimizer_steps_applied") == applied,
            "resource_skip_count": resource.get("steps_skipped_by_scaler") == skipped,
            "resource_safety_skip_count": resource.get("steps_skipped_by_safety_check") == safety_skipped,
            "gate_disabled_for_training": (training or {}).get("gate_disabled_for_training") is True,
            "gate_on_for_validation": (training or {}).get("gate_on_for_validation_evaluations") is True,
            "gate_restored": (training or {}).get("gate_restored_after_training") is True,
        })
    return {
        "status": "VALID" if not issues and all(receipt_checks.values()) else "INVALID",
        "source": str((out / "per_example.jsonl").relative_to(REPO)), "source_sha256": digest(out / "per_example.jsonl"),
        "n_observations": len(rows), "recomputed": fresh, "discrepancies": issues,
        "receipt_checks": receipt_checks, "training_trajectory_recomputation": training_recomputed,
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
        arms[arm], records[arm] = audit_arm(stage, arm, core, smoke, targets, preservation, normalize)
    result = {"core": core, "smoke": smoke, "arms": arms,
              "decision": {"status": "INCOMPLETE", "phase_2_licensed": False}}
    if any(a["status"] == "INCOMPLETE" for a in arms.values()):
        result["decision"]["missing_or_incomplete_arms"] = [a for a in ARMS if arms[a]["status"] == "INCOMPLETE"]
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
    for key in ("max_steps", "grad_accum", "lr", "max_length", "seed", "rank", "alpha", "dropout", "trainable_layers", "dtype"):
        pair_checks[f"matched_config_{key}"] = key in arms["A"]["config"] and all(arms[arm]["config"].get(key) == arms["A"]["config"].get(key) for arm in ARMS)
    result["pair_checks"] = pair_checks
    result["baseline_validity"] = {"recomputed": baseline, "published": None if smoke else PUBLISHED[core], "checked": not smoke}
    if not all(pair_checks.values()):
        result["decision"] = {"status": "INVALID", "phase_2_licensed": False, "failed_checks": [k for k,v in pair_checks.items() if not v]}
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
    sum_keys = ("wall_seconds", "model_loads", "optimizer_steps_attempted", "optimizer_steps_applied", "steps_skipped_by_scaler", "steps_skipped_by_safety_check", "examples_consumed")
    max_keys = ("peak_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
    result["resource_aggregation_from_arm_receipts"] = {
        **{key: sum(arms[arm]["recorded_resource"].get(key, 0) for arm in ARMS) for key in sum_keys},
        **{key: max(arms[arm]["recorded_resource"].get(key, 0) for arm in ARMS) for key in max_keys}}
    combined_resource = json_file(LAB / "results" / stage / "resource.json")
    result["combined_resource_discrepancies"] = []
    if combined_resource:
        for key, value in result["resource_aggregation_from_arm_receipts"].items():
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
                              "An audit-passing smoke run never licenses Phase 2.",
                              "Bootstrap intervals are descriptive; scenario clustering and multiple comparisons prevent significance claims."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decisions": {s: r["decision"] for s,r in output["stages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
