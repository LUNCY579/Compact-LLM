"""Analytic scalar fixtures only; no Torch, tokenizer, model or file mutations."""
from copy import deepcopy
import ast
from functools import lru_cache
import json
from pathlib import Path

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import cpu_causal_replay_analysis as analysis


def triplet(base=3., derivative=.4, balance=.6, eos=False, curvature=2., cubic=0.):
    result = {"kind": "EOS" if eos else "answer", "full_model_calls": 3, "state_adjoints": 1}
    for key, alpha, grad in zip(("nominal", "plus", "minus"), analysis.ALPHAS, (1, 0, 0)):
        step = alpha-1
        loss = base+derivative*step+curvature*step**2+cubic*step**3
        result[key] = {"alpha": alpha, "loss": loss, "sensitivity": derivative if grad else None,
                       "state_gradient_evaluations": grad, "anchor_counts": [1, 1, 1],
                       "balance": balance, "canonical_EOT_CE": loss+.7 if eos else None,
                       "RMS": [{"normal_rms": 1., "controlled_rms": 1., "rms_ratio_to_normal": 1.}]}
    fd = (result["plus"]["loss"]-result["minus"]["loss"])/.02
    result.update(finite_difference=fd, absolute_error=abs(derivative-fd),
                  registered_tolerance=.001+.03*abs(derivative))
    return result


def save_case(bundle, case):
    targets = [t for t in bundle["targets"] if t["case_id"] == case and t["kind"] == "answer"]
    eos = next(t["result"] for t in bundle["targets"] if t["case_id"] == case and t["kind"] == "EOS")
    stored = next(o for o in bundle["outcomes"] if o["case_id"] == case)
    n = len(targets)
    means = {key: sum(t["result"][key]["loss"] for t in targets)/n for key in ("nominal", "plus", "minus")}
    derivative = sum(t["result"]["nominal"]["sensitivity"] for t in targets)/n
    fd = (means["plus"]-means["minus"])/.02
    stored.update(answer_target_count=n, mean_answer_CE=means, mean_answer_sensitivity=derivative,
                  sum_answer_CE={key: value*n for key, value in means.items()},
                  sum_answer_sensitivity=derivative*n,
                  mean_level_FD={"finite_difference": fd, "absolute_error": abs(derivative-fd),
                                 "registered_tolerance": .001+.03*abs(derivative)},
                  EOS_secondary=eos)


@lru_cache(maxsize=1)
def base_bundle():
    package = Path(analysis.__file__).resolve().parent
    population = json.loads((package/"CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json").read_text(encoding="utf-8"))["rows"]
    bank = {x["case_id"]: x for x in json.loads((package/"runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json").read_text(encoding="utf-8"))}
    resources = {"rss_bytes": 1024**3, "peak_rss_bytes": 2*1024**3, "host_available_bytes": 10*1024**3, "device": "cpu"}
    bundle = {"population": population, "targets": [], "outcomes": [], "features": [], "events": [],
              "io_checks": {key: True for key in ("admission_review_registration", "population_metadata", "frozen_source_closure", "source_snapshot", "qualification_vectors")}}
    bundle.update(bank=list(bank.values()), preflight={"resources": resources}, resources=[resources])
    events = bundle["events"]
    def event_pair(kind, base):
        events.extend([{"event": kind+"_STARTED", **base}, {"event": kind+"_COMPLETED", **base, "resources": resources}])
    for label, length in (("representative", 84), ("maximum256", 256)):
        for alpha, extra in ((1., 0), (.99, 1), (1., 1), (1.01, 1)):
            for diagnostic in (False, True):
                event_pair("VALUE_FORWARD", {"label": label, "prefix_tokens": length, "alpha": alpha,
                                              "extra_loops": extra, "diagnostic": diagnostic, "state_gradient_evaluations": 0})
    events.extend([{"event": "VALUE_FORWARD_STARTED", "prefix_tokens": 256},
                   {"event": "VALUE_FORWARD_FAILED", "error": "RuntimeError: EXPECTED_CAUSAL_SUFFIX_EXCEPTION", "partial_compute": "UNKNOWN"}])
    event_pair("VALUE_FORWARD", {"label": "restored_after_exception", "prefix_tokens": 256, "state_gradient_evaluations": 0})
    for use, present in ((True, False), (False, True)):
        events.extend([{"event": "CACHE_REJECTION_STARTED", "use_cache": use, "cache_object_present": present},
                       {"event": "CACHE_REJECTED", "model_compute_started": False, "wrapper_forward_count": 0}])
    scalar = [triplet(), triplet()]
    eos_qual = triplet(eos=True)
    for label, length, result in (("qualification_representative_answer", 84, scalar[0]),
                                  ("qualification_maximum256_answer", 256, scalar[1]),
                                  ("qualification_maximum256_EOS", 256, eos_qual)):
        for key, alpha, grad in zip(("nominal", "plus", "minus"), analysis.ALPHAS, (1, 0, 0)):
            event_pair("REPLAY_FORWARD", {"label": label, "prefix_tokens": length, "alpha": alpha,
                                           "state_gradient_evaluations": grad, "elapsed_seconds": 1. if grad else .5,
                                           "scalar_result": result[key]})
    for index, row in enumerate(population):
        case = row["case_id"]
        feature = {"case_id": case, "progress": 1., "fluctuation": 1/6, "balance": .6,
                   "input_ids_sha256": row["original_prompt_ids_sha256"], "source_prefix_sha256": row["original_prompt_ids_sha256"],
                   "teacher_conditioned": False, "target_independent": True, "preceding_gold_tokens": 0,
                   "preceding_gold_count": 0, "phase": "INPUT_ONLY", "prefix_kind": "INPUT_ONLY",
                   "frozen_before_loss": True, "loss_or_gradient_predictors": False}
        feature["freeze_sha256"] = analysis.digest(feature)
        bundle["features"].append(feature)
        stored = {key: bank[case][key] for key in ("case_id", "parent_id", "category", "difficulty_stratum", "answerability", "source_status", "variant_id", "pair_kind")}
        stored["input_only_predictor"] = feature
        bundle["outcomes"].append(stored)
        n = len(row["canonical_answer_token_ids"])
        plans = [("answer", j, t) for j, t in enumerate(row["answer_targets"])]+[("EOS", n, row["EOS"])]
        for kind, j, plan in plans:
            result = triplet(derivative=.4 if index%2 == 0 else -.4, balance=.6 if j == 0 else -.9, eos=kind == "EOS")
            target = {"case_id": case, "parent_id": row["parent_id"], "kind": kind, "answer_position": j,
                      "prefix_sha256": plan["prefix_sha256"], "prefix_tokens": plan["prefix_tokens"],
                      "preceding_gold_count": j, "teacher_conditioned": j > 0,
                      "current_future_target_positions_in_prefix": False, "result": result}
            target["target_token_id" if kind == "answer" else "accepted_EOS_ids"] = plan["target_token_id"] if kind == "answer" else list(analysis.EOS)
            bundle["targets"].append(target)
            for key, alpha, grad in zip(("nominal", "plus", "minus"), analysis.ALPHAS, (1, 0, 0)):
                event_pair("REPLAY_FORWARD", {"case_id": case, "kind": kind, "target_index": j,
                                               "prefix_ids_sha256": plan["prefix_sha256"], "prefix_tokens": plan["prefix_tokens"],
                                               "preceding_gold_count": j, "teacher_conditioned": j > 0,
                                               "current_future_target_positions_in_prefix": False, "alpha": alpha,
                                               "state_gradient_evaluations": grad, "scalar_result": result[key]})
        save_case(bundle, case)
    ledger = dict(analysis.EXPECTED_COUNTS)
    bundle["ledger"] = ledger
    bundle["receipt"] = {"status": "CPU_CAUSAL_REPLAY_COMPLETE_SURROGATE_ONLY", "lane": analysis.LANE,
                          "loaded_state_sha256": analysis.STATE, "before_state_sha256": analysis.STATE, "after_state_sha256": analysis.STATE,
                          "loaded_custom_code_sha256": analysis.CUSTOM, "parameter_gradients": "NONE", "trainable_parameters": 0,
                          "only_state_leaf_grad": True, "training": False, "optimizer_steps": 0,
                          "generation_pool_access": False, "native_test": False,
                          "source_unchanged": True, "checkpoint_files_unchanged": True, "compute_ledger": ledger,
                          "after_load_resources": resources, "final_resources": resources, "final_resources_on_stop": resources}
    bundle["terminal"] = {"status": "WORKER_TERMINAL", "exit_code": 0, "native_worker_seconds": 100.}
    prefix = "experiments/rineya_taps_adaptive_recurrence_campaign_v2/"
    bundle["admission"] = {"status": "READY_FOR_CPU_CAUSAL_REPLAY_DISPATCH", "lane": analysis.LANE, "device": "cpu", "dtype": "float32",
                           "hard_deadline_seconds": 3600, "finalization_reserve_seconds": 180, "native_finalization_reserve_seconds": 1800,
                           "dispatch_hashes": {prefix+k: v for k, v in analysis.SOURCE_PINS.items()} | {prefix+"cpu_causal_replay_native.py": "b"*64}}
    bundle["review"] = {"status": "PASS_CPU_CAUSAL_REPLAY_DISPATCH"}
    bundle["registration"] = {"lane": analysis.LANE, "device": "cpu", "dtype": "float32", "threads": 4,
                              "worker_sha256": "b"*64, "design_sha256": analysis.DESIGN, "metadata_packet_sha256": analysis.PACKET,
                              "CPU_helper_sha256": analysis.SOURCE_PINS["cpu_h3_native.py"], "math_sha256": analysis.SOURCE_PINS["sensitivity.py"],
                              "runtime": {"python": "3.11.15", "torch": "2.12.1+cu126", "transformers": "5.14.1", "qwen_sha256": analysis.QWEN}}
    bundle["qualification"] = {"status": "PASS_CPU_CAUSAL_REPLAY_MECHANICS", "device": "cpu",
                               "parameter_gradients": "NONE", "source_unchanged": True, "checkpoint_files_unchanged": True,
                               "before_state_sha256": analysis.STATE, "after_state_sha256": analysis.STATE,
                               "maximum_prefix_tokens": 256, "cache_rejections": 2, "forced_restoration": True,
                               "qualified_state_adjoints": 3, "qualified_vector_pairs": 3,
                               "resources": resources,
                               "scalar_CE_FD": scalar, "EOS_mass_FD": eos_qual,
                               "parity": [{"label": label, "alpha": alpha, "extra": extra, "exact_equal": True,
                                           "reference_sha256": "a"*64, "diagnostic_sha256": "a"*64}
                                          for label in ("representative", "maximum256") for alpha, extra in ((1., 0), (.99, 1), (1., 1), (1.01, 1))]}
    bundle["forecast"] = {"admitted": True, "device": "cpu", "factor": 1.5, "remaining_task_targets": 632,
                          "remaining_task_forwards": 1896, "internal_reserve_seconds": 180, "parent_deadline_seconds": 3600,
                          "elapsed_seconds": 50., "maximum_whole_forward_adjoint_seconds": 1., "maximum_value_forward_seconds": .5,
                          "remaining_forecast_seconds": 632*2*1.5}
    return bundle


def fixture():
    return deepcopy(base_bundle())


def test_complete_analytic_curve_preserves_case_parent_counts_and_initial_features():
    report = analysis.analyze_bundle(fixture())
    assert report["errors"] == []
    assert report["counts"]["API_attempts"] == 1925 and report["counts"]["completed_prefix_positions"] == 252696
    assert report["primary"]["direction"]["cases"] == 192
    assert report["primary"]["population"]["parent_count"] == 96
    assert report["primary"]["direction"]["coherence_agreements"] == 96
    assert all(row["balance"] == .6 for row in report["primary"]["case_rows"])


def test_preserved_global_counts_do_not_hide_missing_target_replaced_with_duplicate():
    bundle = fixture()
    bundle["targets"].pop()
    bundle["targets"].append(deepcopy(bundle["targets"][0]))
    report = analysis.analyze_bundle(bundle)
    assert report["counts"]["API_attempts"] == 1925
    assert report["primary"] is None and report["EOS_secondary"] is None
    assert any("target cell" in error for error in report["errors"])


def test_mean_cancellation_must_pass_its_own_gate_even_when_each_target_passes():
    bundle = fixture()
    case = next(row["case_id"] for row in bundle["population"] if len(row["answer_targets"]) >= 3)
    targets = [row for row in bundle["targets"] if row["case_id"] == case and row["kind"] == "answer"]
    for index, target in enumerate(targets):
        new = triplet(base=1000., derivative=100. if index == 0 else -100. if index == 1 else 0.,
                      cubic=2000. if index < 2 else 0.)
        target["result"].clear(); target["result"].update(new)
        for event in bundle["events"]:
            if event.get("case_id") == case and event.get("kind") == "answer" and event.get("target_index") == index and event["event"] == "REPLAY_FORWARD_COMPLETED":
                event["scalar_result"] = target["result"][{1.: "nominal", 1.01: "plus", .99: "minus"}[event["alpha"]]]
    save_case(bundle, case)
    report = analysis.analyze_bundle(bundle)
    assert report["primary"] is None
    assert any("mean FD cancellation" in error for error in report["errors"])
    assert not any("target FD gate" in error for error in report["errors"])


def test_teacher_feature_provenance_and_label_dependent_feature_are_refused():
    for change in ({"phase": "CAUSAL_TEACHER", "preceding_gold_count": 1}, {"loss": 0.}):
        bundle = fixture(); feature = bundle["features"][0]
        feature.update(change); feature["freeze_sha256"] = analysis.digest({k: v for k, v in feature.items() if k != "freeze_sha256"})
        report = analysis.analyze_bundle(bundle)
        assert report["primary"] is None and any("predictor firewall" in error for error in report["errors"])


def test_nonneutral_adjoint_with_zero_FD_is_retained_but_not_reliable_direction():
    bundle = fixture(); case = bundle["population"][0]["case_id"]
    for target in bundle["targets"]:
        if target["case_id"] == case and target["kind"] == "answer":
            result = target["result"]
            result["nominal"]["sensitivity"] = .00101
            result["plus"]["loss"] = result["minus"]["loss"] = result["nominal"]["loss"]
            result.update(finite_difference=0., absolute_error=.00101, registered_tolerance=.001+.03*.00101)
    save_case(bundle, case)
    report = analysis.analyze_bundle(bundle)
    assert report["errors"] == []
    summary = report["primary"]["direction"]
    assert summary["cases"] == 192 and summary["nonneutral_cases"] == 192
    assert summary["numerically_uncertain_nonneutral_cases"] == 1 and summary["reliable_sign_denominator"] == 191


def test_EOS_loss_harm_is_not_hidden_by_primary_answer_loss_improvement():
    bundle = fixture()
    case = bundle["population"][1]["case_id"]
    target = next(t for t in bundle["targets"] if t["case_id"] == case and t["kind"] == "EOS")
    result = target["result"]
    for key in ("nominal", "plus", "minus"):
        alpha = result[key]["alpha"]
        result[key]["loss"] = 3.+.3*(alpha-1)
        result[key]["canonical_EOT_CE"] = result[key]["loss"]+.7
    result["nominal"]["sensitivity"] = .3
    fd = (result["plus"]["loss"]-result["minus"]["loss"])/.02
    result.update(finite_difference=fd, absolute_error=abs(.3-fd), registered_tolerance=.001+.03*.3)
    save_case(bundle, case)
    report = analysis.analyze_bundle(bundle)
    assert report["errors"] == []
    row = next(row for row in report["primary"]["case_rows"] if row["case_id"] == case)
    assert row["answer_loss_delta_plus"] < 0
    assert row["EOS_secondary"]["loss"]["plus"] > row["EOS_secondary"]["loss"]["nominal"]
    assert "canonical_prefix_sum_descriptive" in report and report["semantic_efficacy"] is None
    assert report["EOS_secondary"]["interpretation"].endswith("No primary composite.")


@pytest.mark.parametrize("proof", ["state", "runtime", "forecast", "source"])
def test_missing_execution_proofs_cannot_become_complete_by_missing_equals_missing(proof):
    bundle = fixture()
    if proof == "state":
        for key in ("loaded_state_sha256", "before_state_sha256", "after_state_sha256"):
            bundle["receipt"][key] = None
        bundle["qualification"]["before_state_sha256"] = bundle["qualification"]["after_state_sha256"] = None
    elif proof == "runtime":
        bundle["registration"]["runtime"] = None
    elif proof == "forecast":
        bundle["forecast"] = {"admitted": True, "device": "cpu"}
    else:
        bundle["io_checks"]["frozen_source_closure"] = False
    assert analysis.analyze_bundle(bundle)["primary"] is None


def test_forbidden_future_position_hash_is_refused_even_if_same_token_value_repeats():
    bundle = fixture()
    bundle["targets"][1]["prefix_sha256"] = bundle["targets"][2]["prefix_sha256"]
    report = analysis.analyze_bundle(bundle)
    assert report["primary"] is None and any("prefix proof" in error for error in report["errors"])


@pytest.mark.parametrize("defect", ["RSS", "RMS", "stratum", "source_pin", "READY"])
def test_finite_scalars_do_not_override_resource_stratum_or_source_contract(defect):
    bundle = fixture()
    if defect == "RSS":
        bundle["resources"] = [{"rss_bytes": 6*analysis.GIB+1, "host_available_bytes": 10*analysis.GIB}]
    elif defect == "RMS":
        bundle["targets"][0]["result"]["nominal"]["RMS"][0].update(controlled_rms=5., rms_ratio_to_normal=5.)
    elif defect == "stratum":
        bundle["outcomes"][0]["difficulty_stratum"] = "posthoc_easy"
    elif defect == "source_pin":
        bundle["admission"]["dispatch_hashes"].pop("experiments/rineya_taps_adaptive_recurrence_campaign_v2/sensitivity.py")
    else:
        bundle["admission"]["status"] = "METADATA_READY_NOT_QUALIFIED"
    report = analysis.analyze_bundle(bundle)
    assert report["primary"] is None and report["errors"]


def test_tied_rank_correlation_has_known_value_and_no_variance_is_null():
    assert analysis.rank([2., 1., 2., 4.]) == [2.5, 1., 2.5, 4.]
    assert analysis.correlation(analysis.rank([1., 2., 3.]), analysis.rank([3., 2., 1.])) == -1.
    assert analysis.correlation([1., 1.], [2., 3.]) is None


def test_identical_counts_and_scalars_do_not_hide_completion_before_start():
    bundle = fixture()
    indices = [i for i, event in enumerate(bundle["events"]) if event.get("case_id") is not None][:2]
    a, b = indices
    bundle["events"][a], bundle["events"][b] = bundle["events"][b], bundle["events"][a]
    report = analysis.analyze_bundle(bundle)
    assert report["primary"] is None
    assert any("chronological" in error for error in report["errors"])


def test_completed_qualification_after_first_task_is_incomplete_not_retroactive_pass():
    bundle = fixture()
    index = next(i for i, event in enumerate(bundle["events"]) if event.get("label") == "qualification_maximum256_EOS")
    bundle["events"].append(bundle["events"].pop(index))
    report = analysis.analyze_bundle(bundle)
    assert report["primary"] is None
    assert any("before first task" in error for error in report["errors"])


def extended_fixture():
    """Synthetic IO proof only, never a real grant or permission artifact."""
    bundle = fixture()
    prefix = "experiments/rineya_taps_adaptive_recurrence_campaign_v2/"
    bundle["admission"].update(hard_deadline_seconds=4500)
    bundle["admission"]["dispatch_hashes"].pop(prefix+"CPU_CAUSAL_REPLAY_DESIGN_DRAFT_002.md")
    bundle["admission"]["dispatch_hashes"][prefix+"CPU_CAUSAL_REPLAY_DESIGN_DRAFT_003.md"] = analysis.DESIGN_4500
    bundle["registration"].update(design_sha256=analysis.DESIGN_4500, parent_deadline_seconds=4500,
                                  body_deadline_seconds=4320, internal_reserve_seconds=180)
    bundle["forecast"].update(parent_deadline_seconds=4500, body_deadline_seconds=4320)
    bundle["io_checks"]["4500_human_scope"] = True
    return bundle


def test_4500_exact_bound_changes_budget_only_not_endpoints_or_units():
    original = analysis.analyze_bundle(fixture())
    extended = analysis.analyze_bundle(extended_fixture())
    assert original["errors"] == extended["errors"] == []
    assert original["primary"] == extended["primary"]
    assert original["EOS_secondary"] == extended["EOS_secondary"]


def test_actual_3778_forecast_refuses3600_and_fits_only_new_approved4500():
    bundles = [fixture(), extended_fixture()]
    g, v, elapsed = 1.7388156999986677, .9941931000030308, 64.91039009999804
    for bundle in bundles:
        for event in bundle["events"]:
            if event.get("event") == "REPLAY_FORWARD_COMPLETED" and event.get("label") in ("qualification_maximum256_answer", "qualification_maximum256_EOS"):
                event["elapsed_seconds"] = g if event["state_gradient_evaluations"] == 1 else v
        bundle["forecast"].update(elapsed_seconds=elapsed, maximum_whole_forward_adjoint_seconds=g,
                                  maximum_value_forward_seconds=v, remaining_forecast_seconds=632*(g+2*v)*1.5)
    assert analysis.analyze_bundle(bundles[0])["primary"] is None
    assert analysis.analyze_bundle(bundles[1])["primary"] is not None


@pytest.mark.parametrize("defect", ["permit", "design", "body", "factor", "reserve", "5400"])
def test_extra_time_never_weakens_permission_source_forecast_or_reserve_gates(defect):
    bundle = extended_fixture()
    if defect == "permit":
        bundle["io_checks"]["4500_human_scope"] = False
    elif defect == "design":
        bundle["registration"]["design_sha256"] = analysis.DESIGN
    elif defect == "body":
        bundle["forecast"]["body_deadline_seconds"] = 4500
    elif defect == "factor":
        bundle["forecast"]["factor"] = 1.
    elif defect == "reserve":
        bundle["admission"]["native_finalization_reserve_seconds"] = 0
    else:
        bundle["admission"]["hard_deadline_seconds"] = 5400
    assert analysis.analyze_bundle(bundle)["primary"] is None


def test_scientific_and_numerical_definitions_are_ast_identical_to_frozen3600():
    package = Path(analysis.__file__).resolve().parent
    old = ast.parse((package/"versions/cpu_causal_replay_analysis_3600_v1/cpu_causal_replay_analysis.py").read_text(encoding="utf-8"))
    new = ast.parse(Path(analysis.__file__).read_text(encoding="utf-8"))
    definitions = lambda tree: {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef)}
    before, after = definitions(old), definitions(new)
    for name in ("numeric", "same_number", "sign", "rank", "correlation", "_number", "_triplet", "_resources", "_rms", "_journal", "_direction_summary", "_population_summaries", "analyze_bundle", "_read_npy", "_vectors"):
        assert before[name] == after[name], name
