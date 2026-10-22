"""Authored aggregate fixtures only; no bank rows, model calls or native outcomes."""
import copy
import json

import pytest

from experiments.rineya_quality_sft_v1 import SOURCE_WEIGHT_SHA256
from experiments.rineya_quality_sft_v1 import pattern_audit as P


def architecture_row(budget, draw, schedule, beta2, correct, retained, steps=3):
    peak = 1e-5
    total = {1300: 1397, 5200: 5233, 15600: 15734}[budget]
    row = {"arm": "B28last20", "completed": True, "parameters": 596049920, "trainable_parameters": 314618880,
           "lr": peak, "beta2": beta2, "schedule_spec": schedule, "supervised_target_tokens": total,
           "train_rows_consumed": steps, "processed_full_sequence_tokens_derived": total + steps,
           "optimizer_steps": steps, "last_measured_steps": steps, "last_measured_tokens": total,
           "source_weights_sha256": SOURCE_WEIGHT_SHA256, "train_row_ids_sha256": str(draw)[0] * 64,
           "source_assets_sha256": {k: "a" * 64 for k in P.ASSET_FIELDS},
           "data_sha256": {k: "b" * 64 for k in P.DATA_FIELDS},
           "optimizer": f"AdamW(offloaded) lr=1e-05 wd=0 clip=1.0 micro_batch=1 betas=(0.9,{beta2}) eps=1e-8",
           "exploration_harness_sha256": ("c" if schedule.startswith("cosine") else "d") * 64,
           "optimizer_harness_sha256": "e" * 64, "loss_objective_code_audit": "whole next-token CE fixture",
           "artifact_path": f"fixture/ARM_DOSE_B28last20_lr1e-05_b{budget}_seed{draw}" + (
               "_wsd0.2-0.2_b20.95.json" if beta2 == .95 else ".json"),
           "artifact_commit": "f" * 40, "artifact_text_sha256": "1" * 64, "dev_ce_nats": 2.,
           "panel": {"correct_total": correct, "retained_total": retained, "gained_total": correct - retained,
                     "lost_total": 47 - retained, "reference_n": 47, "terminated_total": 150},
           "step_trace": [{"step": i, "tokens": total * i // steps, "lr": peak / (i + 1), "grad_norm": 56.5}
                          for i in range(1, steps + 1)]}
    row["source_assets_sha256"]["model.safetensors"] = SOURCE_WEIGHT_SHA256
    return row


def fixtures():
    arms = [architecture_row(1300, 17, "wsd:0.2:0.2", .95, 47, 39),
            architecture_row(15600, 17, "wsd:0.2:0.2", .95, 64, 38)]
    for draw, cosine, wsd, rc, rw in ((17, 67, 74, 40, 41), (23, 58, 62, 39, 37),
                                       (29, 60, 73, 39, 42), (41, 62, 68, 39, 39)):
        arms.extend([architecture_row(5200, draw, "cosine:0.03", .999, cosine, rc),
                     architecture_row(5200, draw, "wsd:0.2:0.2", .95, wsd, rw)])
    cells, plans = {}, {}
    for order in (17, 23, 29):
        key = str(order)[-1] * 64
        plan = {"sha256": key, "seal_valid": True, "selection_seed": 17, "order_seed": order,
                "train_rows": 238, "processed_tokens": 24910, "whole_targets": 24672, "response_targets": 1944,
                "groups": 13, "train_id_set_sha256": "a" * 64, "ordered_id_sha256": key,
                "dev_rows": 64, "dev_scored": 51, "answerability_false": 0, "mask_bounds_valid": True}
        plans[str(order)] = plan
        for objective in ("whole", "response"):
            final = 13 if objective == "response" else 8 if order == 29 else 7
            name = f"cell-s{order}-{objective}" + ("-w2" if order == 17 and objective == "whole" else "")
            cells[name] = {"arm": objective + "_1e-5", "checkout": "b" * 40, "plan_sha256": key, "code_sha256": "c" * 64,
                           "result_hashes_valid": True, "finite_gradients_and_losses": True, "baseline_correct": 3,
                           "final_correct": final, "scored": 51, "retained": 3, "gained": final - 3, "lost": 0,
                           "baseline_ce": 2.8, "final_ce": 2., "generated_mean": 5., "termination": 64,
                           "literal_unknown": 0, "row_order_sha256": key, "supervisor_present": False,
                           "checkpoint_present": False, "progress": {"next_group": 13, "updates": 13,
                               "processed_tokens": 24910, "supervised_tokens": plan[objective + "_targets"]}}
    return {"schema": 1, "arms": arms}, {"scope": "AUTHORED_FIXTURE_ONLY", "plans": plans, "cells": cells}


def files(tmp_path, architecture=None, mq=None):
    a, m = fixtures()
    first, second = tmp_path / "architecture-aggregate.json", tmp_path / "mq-aggregate.json"
    first.write_text(json.dumps(a if architecture is None else architecture), encoding="utf-8")
    second.write_text(json.dumps(m if mq is None else mq), encoding="utf-8")
    return first, second


def test_conditional_patterns_and_explicit_limits(tmp_path):
    report = P.analyze_historical(*files(tmp_path))
    findings = report["derived_findings"]
    dose = findings["dose_nonmonotonicity"]
    assert [p["actual_supervised_tokens"] for p in dose["points"]] == [1397, 5233, 15734]
    assert [p["correct"] for p in dose["points"]] == [47, 74, 64]
    assert [p["retained"] for p in dose["points"]] == [39, 41, 38]
    assert dose["less_exposure_always_preserves_better_refuted_here"]
    assert dose["monotonic_quality_with_more_exposure_refuted_here"]
    bundle = findings["bundled_recipe_across_draws"]
    assert [p["correct_delta"] for p in bundle["pairs"]] == [7, 4, 13, 6]
    assert [p["retained_delta"] for p in bundle["pairs"]] == [1, -2, 3, 0]
    assert all(p["exploration_harness_changed"] for p in bundle["pairs"])
    assert not bundle["schedule_only_causal_claim"] and bundle["pooled_row_p_value"] is None
    mq = findings["whole_response_across_orders"]
    assert [p["response_minus_whole"] for p in mq["pairs"]] == [6, 6, 5]
    assert mq["distinct_training_sets"] == 1 and not mq["independent_training_draw_replication"]
    assert not report["efficacy_verified"] and report["native_runs_this_audit"] == 0


def test_raw_outcome_text_and_unlisted_trace_values_never_enter_report(tmp_path):
    a, m = fixtures()
    for row in a["arms"]:
        row["per_row_correct"] = {"PRIVATE_INDIVIDUAL_PAYLOAD": True}
        row["prediction"] = "PRIVATE_PREDICTION_PAYLOAD"
        row["prompt"] = "PRIVATE_PROMPT_PAYLOAD"
        row["panel"]["per_class"] = {"private": {"lost_ids": ["PRIVATE_IDS_PAYLOAD"]}}
        for step in row["step_trace"]: step["prediction"] = "PRIVATE_TRACE_PAYLOAD"
    for cell in m["cells"].values(): cell["predictions"] = ["PRIVATE_CELL_PAYLOAD"]
    report = P.analyze_historical(*files(tmp_path, a, m))
    assert "PRIVATE" not in json.dumps(report)
    assert report["projection"]["individual_outcomes_or_text_inspected"] is False


@pytest.mark.parametrize("mutation", ["partial", "stale_endpoint", "invalid_panel", "duplicate_trace", "missing_trace"])
def test_failed_or_invalid_dose_provenance_retained_but_not_pooled(mutation, tmp_path):
    a, m = fixtures()
    row = a["arms"][0]
    if mutation == "partial": row["completed"] = False
    if mutation == "stale_endpoint": row["last_measured_tokens"] = 0
    if mutation == "invalid_panel": row["panel"]["lost_total"] += 1
    if mutation == "duplicate_trace": row["step_trace"][1]["step"] = 1
    if mutation == "missing_trace": row["step_trace"].pop()
    report = P.analyze_historical(*files(tmp_path, a, m))
    assert report["architecture_record_counts"]["excluded"] == 1
    assert report["excluded_architecture_attempts"][0]["artifact_path"] == row["artifact_path"]
    assert report["excluded_architecture_attempts"][0]["reasons"]
    assert report["derived_findings"]["dose_nonmonotonicity"]["status"] == "NOT_COMPARABLE"


@pytest.mark.parametrize("mutation", ["source", "corpus", "probe", "tail", "artifact_commit", "optimizer_code"])
def test_wrong_dose_comparability_not_hidden(mutation, tmp_path):
    a, m = fixtures()
    row = a["arms"][0]
    if mutation == "source": row["source_weights_sha256"] = "2" * 64
    if mutation == "corpus": row["data_sha256"]["train"] = "3" * 64
    if mutation == "probe": row["data_sha256"]["monitor_probe"] = "3" * 64
    if mutation == "tail": row["trainable_parameters"] -= 1
    if mutation == "artifact_commit": row["artifact_commit"] = "4" * 40
    if mutation == "optimizer_code": row["optimizer_harness_sha256"] = "5" * 64
    report = P.analyze_historical(*files(tmp_path, a, m))
    assert report["derived_findings"]["dose_nonmonotonicity"]["status"] == "NOT_COMPARABLE"


@pytest.mark.parametrize("change", ["clip=1.0", "eps=1e-8", "micro_batch=1"])
def test_declared_beta2_difference_allowed_but_other_optimizer_changes_refused(change, tmp_path):
    a, m = fixtures()
    ordinary = P.analyze_historical(*files(tmp_path, a, m))
    assert len(ordinary["derived_findings"]["bundled_recipe_across_draws"]["pairs"]) == 4
    a["arms"][3]["optimizer"] = a["arms"][3]["optimizer"].replace(change, change.split("=")[0] + "=2")
    altered = P.analyze_historical(*files(tmp_path, a, m))
    finding = altered["derived_findings"]["bundled_recipe_across_draws"]
    assert len(finding["pairs"]) == 3 and finding["excluded_pairs"][0]["draw"] == 17
    assert "optimizer_without_beta2" in finding["excluded_pairs"][0]["fields"]


@pytest.mark.parametrize("mutation", ["code", "checkout", "order", "partial", "bad_counts", "mask"])
def test_mq_missing_exact_pair_identity_blocks_repeated_pattern(mutation, tmp_path):
    a, m = fixtures()
    cell = m["cells"]["cell-s17-response"]
    if mutation == "code": cell["code_sha256"] = "d" * 64
    if mutation == "checkout": cell["checkout"] = "e" * 40
    if mutation == "order": cell["row_order_sha256"] = "f" * 64
    if mutation == "partial": cell["progress"]["updates"] -= 1
    if mutation == "bad_counts": cell["retained"] = 4
    if mutation == "mask": m["plans"]["17"]["mask_bounds_valid"] = False
    finding = P.analyze_historical(*files(tmp_path, a, m))["derived_findings"]["whole_response_across_orders"]
    assert finding["status"] == "INCOMPLETE_COMPARABILITY" and len(finding["pairs"]) == 2
    assert any(item.get("order") == 17 for item in finding["excluded_cells_or_pairs"])


def test_lr_proxy_never_becomes_parameter_dose_or_efficiency_claim(tmp_path):
    report = P.analyze_historical(*files(tmp_path))
    points = report["derived_findings"]["dose_nonmonotonicity"]["points"]
    assert all(p["lr_sum_proxy"] > 0 and p["measured_parameter_update_dose"] is None for p in points)
    mq = report["derived_findings"]["whole_response_across_orders"]
    assert mq["measured_parameter_update_dose"] is None
    assert mq["pairs"][0]["global_supervision_ratio"] == pytest.approx(24672 / 1944)
    assert not mq["balanced_grounding_evidence"]
    assert not report["recipe_or_gate_changed"] and not report["training_authorized"]


def test_input_refusal_and_create_only_output(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="open aggregate"):
        P.analyze_historical(tmp_path / "fresh_sealed.json", tmp_path / "mq.json")
    paths = files(tmp_path)
    monkeypatch.setattr(P, "external_output", lambda path: path)
    out = tmp_path / "new-offline-report.json"
    args = ["--architecture-audit", str(paths[0]), "--mq-audit", str(paths[1]), "--output", str(out)]
    assert P.main(args) == 0
    with pytest.raises(FileExistsError): P.main(args)
