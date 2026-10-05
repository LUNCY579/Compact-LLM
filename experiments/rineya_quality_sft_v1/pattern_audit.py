"""Offline conditional-pattern audit of two historical aggregate reports only.

Projection is an explicit whitelist. Individual correctness maps, predictions,
prompts, banks and model state are never inspected or emitted. Learning-rate
traces provide schedule proxies, never measured parameter-update magnitudes.
This audit cannot authorize a bank, model run, recipe change or efficacy claim.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

from . import EXPERIMENT, SOURCE_WEIGHT_SHA256
from .contract import external_output
from . import integrity as I

SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
ARCH_FIELDS = ("arm", "completed", "parameters", "trainable_parameters", "lr", "beta2", "schedule_spec",
               "supervised_target_tokens", "processed_full_sequence_tokens_derived", "train_rows_consumed",
               "optimizer_steps", "last_measured_steps", "last_measured_tokens", "source_weights_sha256",
               "train_row_ids_sha256", "optimizer", "exploration_harness_sha256", "optimizer_harness_sha256",
               "loss_objective_code_audit", "artifact_path", "artifact_commit", "artifact_text_sha256", "dev_ce_nats")
PANEL_FIELDS = ("correct_total", "retained_total", "gained_total", "lost_total", "reference_n", "terminated_total")
ASSET_FIELDS = ("model.safetensors", "config.json", "generation_config.json", "merges.txt", "tokenizer.json",
                "tokenizer_config.json", "vocab.json")
DATA_FIELDS = ("train", "dev", "monitor_probe", "probe_label_audit")
TRACE_FIELDS = ("step", "tokens", "lr", "grad_norm")
MQ_CELL_FIELDS = ("arm", "checkout", "plan_sha256", "code_sha256", "result_hashes_valid", "baseline_correct",
                  "final_correct", "scored", "retained", "gained", "lost", "baseline_ce", "final_ce",
                  "generated_mean", "termination", "literal_unknown", "finite_gradients_and_losses",
                  "row_order_sha256", "supervisor_present", "checkpoint_present")
MQ_PLAN_FIELDS = ("sha256", "seal_valid", "selection_seed", "order_seed", "train_rows", "processed_tokens",
                  "whole_targets", "response_targets", "groups", "train_id_set_sha256", "ordered_id_sha256",
                  "dev_rows", "dev_scored", "answerability_false", "mask_bounds_valid")
MATCH_FIELDS = ("source_weights_sha256", "source_assets_sha256", "data_sha256", "arm", "parameters",
                "trainable_parameters", "lr", "optimizer", "optimizer_harness_sha256",
                "loss_objective_code_audit", "artifact_commit")


def _project(value, keys):
    if not isinstance(value, dict):
        return {}
    # Expected aggregates are scalars. Nested unknown payloads cannot hitchhike
    # through an allowed field name into a derived report.
    return {key: item if type(item := value.get(key)) in {str, int, float, bool, type(None)}
            and not (type(item) is float and not math.isfinite(item)) else None for key in keys}


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _sha(value):
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def _optimizer(description, beta2):
    number = r"[0-9.eE+-]+"
    pattern = (rf"(?P<algorithm>AdamW\(offloaded\)) lr=(?P<lr>{number}) wd=(?P<wd>{number}) "
               rf"clip=(?P<clip>{number}) micro_batch=(?P<micro_batch>[0-9]+) "
               rf"betas=\((?P<beta1>{number}),(?P<beta2>{number})\) eps=(?P<eps>{number})")
    match = re.fullmatch(pattern, description) if isinstance(description, str) else None
    if not match:
        return None
    result = {"algorithm": match["algorithm"], "micro_batch": int(match["micro_batch"])}
    try:
        result.update({k: float(match[k]) for k in ("lr", "wd", "clip", "beta1", "beta2", "eps")})
    except ValueError:
        return None
    if (any(not _finite(result[k]) for k in ("lr", "wd", "clip", "beta1", "beta2", "eps"))
            or result["beta2"] != beta2 or result["lr"] <= 0 or result["wd"] < 0
            or result["clip"] <= 0 or result["eps"] <= 0 or result["micro_batch"] < 1):
        return None
    return result


def _input(path):
    target = I.check_input_path(path)
    if target.suffix.casefold() != ".json" or any(
            any(word in part.casefold() for word in ("sealed", "confirmation", "protected")) for part in target.parts):
        raise ValueError("Only explicitly supplied open aggregate JSON reports are allowed")
    return I.regular(target)


def _architecture(document):
    if not isinstance(document, dict) or document.get("schema") != 1 or not isinstance(document.get("arms"), list):
        raise ValueError("Historical architecture aggregate schema1/arms required")
    records = []
    for index, raw in enumerate(document["arms"]):
        if not isinstance(raw, dict):
            records.append({"index": index, "status": "EXCLUDED", "reasons": ["invalid_aggregate_object"]})
            continue
        row = _project(raw, ARCH_FIELDS)
        row["source_assets_sha256"] = _project(raw.get("source_assets_sha256"), ASSET_FIELDS)
        row["data_sha256"] = _project(raw.get("data_sha256"), DATA_FIELDS)
        row["panel"] = _project(raw.get("panel"), PANEL_FIELDS)
        reasons = []
        if row["completed"] is not True:
            reasons.append("partial_or_failed_attempt_not_pooled")
        if (not _integer(row["optimizer_steps"], 1) or not _integer(row["supervised_target_tokens"], 1)
                or row["last_measured_steps"] != row["optimizer_steps"]
                or row["last_measured_tokens"] != row["supervised_target_tokens"]):
            reasons.append("endpoint_not_the_complete_recorded_dose")
        if (not _integer(row["train_rows_consumed"], 1) or not _integer(row["supervised_target_tokens"], 1)
                or row["processed_full_sequence_tokens_derived"] != row["supervised_target_tokens"] + row["train_rows_consumed"]):
            reasons.append("whole_sequence_token_accounting_invalid")
        panel = row["panel"]
        if (any(not _integer(panel.get(k)) for k in PANEL_FIELDS) or panel.get("reference_n") != 47
                or panel.get("retained_total", 0) + panel.get("lost_total", 0) != 47
                or panel.get("correct_total") != panel.get("retained_total", 0) + panel.get("gained_total", 0)
                or panel.get("correct_total", 151) > 150 or panel.get("terminated_total", 151) > 150):
            reasons.append("panel_aggregate_invariants_invalid")
        if (row["source_weights_sha256"] != SOURCE_WEIGHT_SHA256
                or not all(_sha(row["source_assets_sha256"].get(k)) for k in ASSET_FIELDS)
                or row["source_assets_sha256"].get("model.safetensors") != SOURCE_WEIGHT_SHA256
                or not all(_sha(row["data_sha256"].get(k)) for k in DATA_FIELDS)
                or not all(_sha(row.get(k)) for k in ("train_row_ids_sha256", "optimizer_harness_sha256", "exploration_harness_sha256", "artifact_text_sha256"))
                or not isinstance(row["artifact_commit"], str) or not COMMIT.fullmatch(row["artifact_commit"])):
            reasons.append("exact_source_data_code_provenance_incomplete")
        if not _finite(row["lr"]) or row["lr"] <= 0 or not _finite(row["beta2"]) or not 0 <= row["beta2"] < 1:
            reasons.append("recipe_scalars_invalid")
        optimizer = _optimizer(row["optimizer"], row["beta2"])
        if (optimizer is None or optimizer["lr"] != row["lr"] or not _integer(row["parameters"], 1)
                or not _integer(row["trainable_parameters"], 1)):
            reasons.append("optimizer_descriptor_or_parameter_inventory_invalid")
        row["optimizer_without_beta2"] = {k: v for k, v in optimizer.items() if k != "beta2"} if optimizer else None
        trace = [_project(entry, TRACE_FIELDS) for entry in raw.get("step_trace", [])] if isinstance(raw.get("step_trace"), list) else []
        valid_trace = (len(trace) == row["optimizer_steps"] and bool(trace)
                       and all(entry.get("step") == i and type(entry.get("step")) is int
                               and _integer(entry.get("tokens"), 1) and _finite(entry.get("lr")) and entry["lr"] > 0
                               and _finite(entry.get("grad_norm")) and entry["grad_norm"] >= 0
                               and _finite(row["lr"]) and entry["lr"] <= row["lr"] * (1 + 1e-8)
                               for i, entry in enumerate(trace, 1)))
        if valid_trace:
            valid_trace = (all(a["tokens"] < b["tokens"] for a, b in zip(trace, trace[1:]))
                           and trace[-1]["tokens"] == row["supervised_target_tokens"])
        if not valid_trace:
            reasons.append("unique_complete_scalar_lr_trace_invalid")
        row["trace"] = ({"steps": len(trace), "sha256": I.digest(trace), "first_lr": trace[0]["lr"],
                         "last_lr": trace[-1]["lr"], "lr_sum_proxy": sum(t["lr"] for t in trace),
                         "first_lr_fraction_of_peak": trace[0]["lr"] / row["lr"],
                         "first_preclip_gradient_norm": trace[0]["grad_norm"],
                         "clipped_steps_at_norm1": sum(t["grad_norm"] > 1 for t in trace),
                         "measured_parameter_update_dose": None} if valid_trace else {"steps": len(trace)})
        name = Path(str(row["artifact_path"])).name
        match = re.search(r"_b([0-9]+)_seed([0-9]+)", name)
        row.update(index=index, budget_requested=int(match[1]) if match else None, draw=int(match[2]) if match else None,
                   status="ELIGIBLE_AGGREGATE" if not reasons else "EXCLUDED", reasons=sorted(set(reasons)))
        records.append(row)
    return records


def _mismatches(rows, fields):
    return [field for field in fields if any(I.digest(row.get(field)) != I.digest(rows[0].get(field)) for row in rows[1:])]


def _dose(records):
    selected, reasons = [], []
    for budget in (1300, 5200, 15600):
        candidates = [r for r in records if r.get("status") == "ELIGIBLE_AGGREGATE" and r.get("arm") == "B28last20"
                      and r.get("draw") == 17 and r.get("budget_requested") == budget
                      and r.get("schedule_spec") == "wsd:0.2:0.2" and r.get("beta2") == .95
                      and str(r.get("artifact_path", "")).endswith("_wsd0.2-0.2_b20.95.json")]
        if len(candidates) != 1:
            reasons.append(f"budget{budget}_requires_one_complete_exact_aggregate")
        else:
            selected.append(candidates[0])
    if len(selected) == 3:
        reasons += _mismatches(selected, MATCH_FIELDS + ("schedule_spec", "beta2"))
    if reasons:
        return {"status": "NOT_COMPARABLE", "reasons": reasons, "pooled": False}
    points = [{"artifact_path": r["artifact_path"], "artifact_sha256": r["artifact_text_sha256"],
               "requested_tokens": r["budget_requested"], "actual_supervised_tokens": r["supervised_target_tokens"],
               "processed_tokens": r["processed_full_sequence_tokens_derived"], "worlds": r["train_rows_consumed"],
               "updates": r["optimizer_steps"], "correct": r["panel"]["correct_total"],
               "retained": r["panel"]["retained_total"], "gained": r["panel"]["gained_total"],
               "lost": r["panel"]["lost_total"], "dev_ce_nats": r["dev_ce_nats"], **r["trace"]} for r in selected]
    changes = [{"left_requested": a["requested_tokens"], "right_requested": b["requested_tokens"],
                "correct_delta": b["correct"] - a["correct"], "retained_delta": b["retained"] - a["retained"]}
               for a, b in zip(points, points[1:])]
    return {"status": "DESCRIPTIVE_MATCHED_SOURCE_CORPUS_RECIPE", "points": points, "adjacent_changes": changes,
            "less_exposure_always_preserves_better_refuted_here": any(c["retained_delta"] > 0 for c in changes),
            "monotonic_quality_with_more_exposure_refuted_here": any(c["correct_delta"] < 0 for c in changes),
            "exploration_harness_changed": len({r["exploration_harness_sha256"] for r in selected}) > 1,
            "caveats": ["One training draw; unique worlds, updates and schedule pacing change together.",
                        "LR sums are schedule proxies, not measured parameter dose or comparable compute.",
                        "Exploration harness differs; this is not a clean isolated causal dose experiment."], "pooled": False}


def _schedule(records):
    pairs, excluded = [], []
    for draw in (17, 23, 29, 41):
        choices = []
        for schedule, beta in (("cosine:0.03", .999), ("wsd:0.2:0.2", .95)):
            matches = [r for r in records if r.get("status") == "ELIGIBLE_AGGREGATE" and r.get("arm") == "B28last20"
                       and r.get("draw") == draw and r.get("budget_requested") == 5200
                       and r.get("schedule_spec") == schedule and r.get("beta2") == beta]
            if len(matches) != 1:
                excluded.append({"draw": draw, "reason": "requires_one_completed_measured_recipe_per_side"})
                break
            choices.append(matches[0])
        if len(choices) != 2:
            continue
        left, right = choices
        compare = tuple(field for field in MATCH_FIELDS if field != "optimizer") + (
            "optimizer_without_beta2", "train_row_ids_sha256", "supervised_target_tokens", "train_rows_consumed", "optimizer_steps")
        mismatches = _mismatches(choices, compare)
        if mismatches:
            excluded.append({"draw": draw, "reason": "comparability_mismatch", "fields": mismatches})
            continue
        pairs.append({"draw": draw, "cosine_correct": left["panel"]["correct_total"], "wsd_correct": right["panel"]["correct_total"],
                      "correct_delta": right["panel"]["correct_total"] - left["panel"]["correct_total"],
                      "retained_delta": right["panel"]["retained_total"] - left["panel"]["retained_total"],
                      "gained_delta": right["panel"]["gained_total"] - left["panel"]["gained_total"],
                      "ordered_row_sha256": left["train_row_ids_sha256"],
                      "artifacts": [r["artifact_path"] for r in choices], "artifact_sha256s": [r["artifact_text_sha256"] for r in choices],
                      "optimizer_except_declared_beta2": left["optimizer_without_beta2"],
                      "declared_beta2": [left["beta2"], right["beta2"]],
                      "exploration_harness_changed": left["exploration_harness_sha256"] != right["exploration_harness_sha256"]})
    return {"status": "DESCRIPTIVE_BUNDLED_RECIPE_PATTERN" if len(pairs) == 4 else "INCOMPLETE_COMPARABILITY",
            "pairs": pairs, "excluded_pairs": excluded, "all_four_acquisition_directions_positive": len(pairs) == 4 and all(p["correct_delta"] > 0 for p in pairs),
            "retention_direction_uniformly_positive": len(pairs) == 4 and all(p["retained_delta"] > 0 for p in pairs),
            "schedule_only_causal_claim": False, "pooled_row_p_value": None,
            "caveats": ["Beta2, warmup, schedule and exploration harness differ together.",
                        "The four draws reuse one exposed arithmetic panel; repeated rows are not independent evaluation trials."]}


def _mq(document):
    if not isinstance(document, dict) or not isinstance(document.get("cells"), dict) or not isinstance(document.get("plans"), dict):
        raise ValueError("Historical MQ aggregate cells/plans required")
    plans = {name: _project(raw, MQ_PLAN_FIELDS) for name, raw in document["plans"].items()}
    cells, exclusions = {}, []
    for name, raw in document["cells"].items():
        row = _project(raw, MQ_CELL_FIELDS)
        row["progress"] = _project(raw.get("progress") if isinstance(raw, dict) else None, ("next_group", "processed_tokens", "supervised_tokens", "updates"))
        cells[name] = row
        if not isinstance(raw, dict) or raw.get("result_hashes_valid") is not True or raw.get("finite_gradients_and_losses") is not True:
            exclusions.append({"cell": name, "reason": "failed_or_unverified_cell_not_pooled"})
        elif name in {"cell-response_1e-5-001", "cell-whole_1e-5-001"}:
            exclusions.append({"cell": name, "reason": "superseded_scoring_version_separate_historical_evidence"})
    pairs = []
    for order in (17, 23, 29):
        names = [f"cell-s{order}-whole-w2" if order == 17 else f"cell-s{order}-whole", f"cell-s{order}-response"]
        sides = [cells.get(name, {}) for name in names]
        reasons = []
        for name, side in zip(names, sides):
            if side.get("result_hashes_valid") is not True or side.get("finite_gradients_and_losses") is not True:
                reasons.append("unverified_complete_cell:" + name)
            if (not all(_sha(side.get(k)) for k in ("plan_sha256", "code_sha256", "row_order_sha256"))
                    or not isinstance(side.get("checkout"), str) or not COMMIT.fullmatch(side["checkout"])
                    or not all(_finite(side.get(k)) and side[k] >= 0 for k in ("baseline_ce", "final_ce", "generated_mean"))):
                reasons.append("cell_code_checkout_or_finite_measurements_invalid:" + name)
            if (any(not _integer(side.get(k)) for k in ("baseline_correct", "final_correct", "scored", "retained", "gained", "lost"))
                    or side.get("retained", 0) + side.get("lost", 0) != side.get("baseline_correct")
                    or side.get("final_correct") != side.get("retained", 0) + side.get("gained", 0)):
                reasons.append("cell_aggregate_invariants:" + name)
        reasons += _mismatches(sides, ("checkout", "code_sha256", "plan_sha256", "row_order_sha256", "baseline_correct", "scored"))
        declared = [p for p in plans.values() if p.get("sha256") == sides[0].get("plan_sha256")]
        if len(declared) != 1:
            reasons.append("one_exact_matching_plan_required")
            plan = {}
        else:
            plan = declared[0]
            if (plan.get("seal_valid") is not True or plan.get("mask_bounds_valid") is not True
                    or plan.get("selection_seed") != 17 or plan.get("order_seed") != order
                    or plan.get("ordered_id_sha256") != sides[0].get("row_order_sha256")
                    or not _sha(plan.get("train_id_set_sha256"))
                    or any(not _integer(plan.get(k), 1) for k in ("train_rows", "processed_tokens", "whole_targets", "response_targets", "groups", "dev_rows", "dev_scored"))
                    or not _integer(plan.get("answerability_false"))):
                reasons.append("rowset_order_or_plan_identity_invalid")
        for name, side in zip(names, sides):
            objective = "whole" if "whole" in name else "response"
            progress = side.get("progress", {})
            if (side.get("arm") != objective + "_1e-5" or progress.get("updates") != plan.get("groups")
                    or progress.get("next_group") != plan.get("groups") or progress.get("processed_tokens") != plan.get("processed_tokens")
                    or progress.get("supervised_tokens") != plan.get(objective + "_targets")
                    or side.get("scored") != plan.get("dev_scored")):
                reasons.append("complete_plan_token_update_identity_invalid:" + name)
        if reasons:
            exclusions.append({"order": order, "reason": "pair_not_comparable", "fields": sorted(set(reasons))})
            continue
        left, right = sides
        pairs.append({"order_seed": order, "selection_seed": plan["selection_seed"], "train_id_set_sha256": plan["train_id_set_sha256"],
                      "plan_sha256": plan["sha256"], "code_sha256": left["code_sha256"], "checkout": left["checkout"],
                      "whole_correct": left["final_correct"], "response_correct": right["final_correct"],
                      "response_minus_whole": right["final_correct"] - left["final_correct"], "scored": left["scored"],
                      "source_correct": left["baseline_correct"], "whole_lost": left["lost"], "response_lost": right["lost"],
                      "processed_tokens": plan["processed_tokens"], "updates": plan["groups"], "worlds": plan["train_rows"],
                      "whole_supervised_tokens": plan["whole_targets"], "response_supervised_tokens": plan["response_targets"],
                      "global_supervision_ratio": plan["whole_targets"] / plan["response_targets"],
                      "answerability_false": plan["answerability_false"], "supervisor_present_both": all(s.get("supervisor_present") is True for s in sides)})
    distinct = len({p["train_id_set_sha256"] for p in pairs})
    return {"status": "DESCRIPTIVE_ORDER_REPEATED_PATTERN" if len(pairs) == 3 and distinct == 1 else "INCOMPLETE_COMPARABILITY",
            "pairs": pairs, "excluded_cells_or_pairs": exclusions, "distinct_training_sets": distinct,
            "independent_training_draw_replication": False,
            "all_three_response_directions_positive": len(pairs) == 3 and all(p["response_minus_whole"] > 0 for p in pairs),
            "balanced_grounding_evidence": False, "measured_parameter_update_dose": None,
            "caveats": ["One 238-world training set, three orders; this is not three independent training draws.",
                        "1944 versus24672 supervised targets is a global ratio, not a per-group gradient or FLOP proof.",
                        "Declared answerability-false count is zero; source-success denominator is only three on corrected scoring.",
                        "Corrected cells lack supervisor receipts; no checkpoints survive. Preserve their bounded historical class."]}


def analyze_historical(architecture_audit_path, mq_audit_path):
    paths = [_input(architecture_audit_path), _input(mq_audit_path)]
    before = [I.file_hash(path) for path in paths]
    records = _architecture(I.read_json(paths[0]))
    mq = _mq(I.read_json(paths[1]))
    if [I.file_hash(path) for path in paths] != before:
        raise ValueError("Historical aggregate custody changed during offline analysis")
    exclusions = [{k: r.get(k) for k in ("index", "artifact_path", "artifact_commit", "artifact_text_sha256", "completed", "status", "reasons")}
                  for r in records if r.get("status") == "EXCLUDED"]
    return {"schema": "RQSFT_HISTORICAL_PATTERN_AUDIT_V1", "experiment": EXPERIMENT,
            "inputs": [{"role": role, "path": str(path), "sha256": sha} for role, path, sha in zip(("architecture_aggregate", "mq_aggregate"), paths, before)],
            "projection": {"architecture_fields": list(ARCH_FIELDS), "panel_fields": list(PANEL_FIELDS),
                           "trace_fields": list(TRACE_FIELDS), "mq_cell_fields": list(MQ_CELL_FIELDS), "mq_plan_fields": list(MQ_PLAN_FIELDS),
                           "individual_outcomes_or_text_inspected": False},
            "architecture_record_counts": {"total": len(records), "eligible_exact_aggregates": sum(r.get("status") == "ELIGIBLE_AGGREGATE" for r in records),
                                            "excluded": len(exclusions)},
            "excluded_architecture_attempts": exclusions,
            "derived_findings": {"dose_nonmonotonicity": _dose(records), "bundled_recipe_across_draws": _schedule(records), "whole_response_across_orders": mq},
            "pending_mechanisms": ["Early-update pacing versus total exposure; data worlds and schedule position change together.",
                                   "Beta2/warmup/decay versus other exploration-harness changes.",
                                   "Response emphasis or answer bias versus verifier-backed solving and harder transfer.",
                                   "No actual parameter-update magnitudes are available in these aggregate inputs."],
            "fixed_next_falsifiers": [
                {"hypothesis": "Less update always preserves better", "opposite": "More exposure improved both score and retention from quarter to middle dose.",
                 "test": "After a separately reviewed design, hold every admitted world/group and early warmup fixed; vary one LR-dose knob and measure actual updates plus acquisition/retention. Do not silently subset the current200-row plan."},
                {"hypothesis": "WSD alone causes the repeated improvement", "opposite": "Beta2, warmup and exploration code may explain it.",
                 "test": "One prospective matched pair at fixed beta2 and warmup can test decay shape; do not infer schedule-only causality from the historical bundle."},
                {"hypothesis": "Response masks give general quality improvements", "opposite": "Answer-weight emphasis or answer bias may explain the narrow gain.",
                 "test": "Use the already proposed verified whole/response pair with measured update norms, balanced evidence pairs and harder-transfer gains; then replicate on independently selected training worlds before calling it a reproducible general effect."}],
            "next_action": "Retain the reviewed whole/response first screen; qualify its real bank first. Do not choose a weaker update as a retention winner or launch a schedule/dose grid from these aggregates.",
            "efficacy_verified": False, "native_runs_this_audit": 0, "model_calls": 0, "training_authorized": False,
            "recipe_or_gate_changed": False, "new_bank_created": False, "promotion_allowed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-audit", required=True)
    parser.add_argument("--mq-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = analyze_historical(args.architecture_audit, args.mq_audit)
    output = external_output(I.check_input_path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    I.write_new(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
