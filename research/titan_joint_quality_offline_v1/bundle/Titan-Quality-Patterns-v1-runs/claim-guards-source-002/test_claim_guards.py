"""Targeted guard mechanics on the existing reviewed open PILOT only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

import claim_guards as guards


@pytest.fixture(scope="module")
def open_pilot():
    return guards.load_open_pilot_fixture()


def _case(open_pilot, scenario="measurable"):
    prompts, references = deepcopy(open_pilot)
    with guards._bound_v2() as modules:
        data, analysis, common = (modules[name] for name in ("data", "analysis", "common"))
        refs, depths = guards._bind_inputs(prompts, references, modules)
        predictions = {model: [{"case_id": row["case_id"], "response": row["target"],
                               "completed": True, "truncated": False} for row in references]
                       for model in guards.MODELS}
        reason = [i for i, row in enumerate(references) if row["domain"] == "reasoning"]
        arithmetic = [i for i, row in enumerate(references) if row["domain"] == "arithmetic"]
        def wrong(model, index):
            row = references[index]
            predictions[model][index]["response"] = data.format_target("999999999", row["requested_format"])
        if scenario == "shallow_only":
            shallow = [index for index in reason if depths[references[index]["case_id"]] == 1]
            for model in ("sparse", "control", "joint"):
                for index in shallow[1:]: wrong(model, index)
            for model in ("control", "joint"):
                for index in shallow[1:5]: predictions[model][index]["response"] = references[index]["target"]
        else:
            for index in reason[:32]: wrong("sparse", index)
        if scenario == "all_non_R_wrong":
            for model in guards.MODELS:
                for index, row in enumerate(references):
                    if row["domain"] != "reasoning": wrong(model, index)
            wrong("control", reason[-1])
        elif scenario == "empty_non_R_recovery":
            wrong("control", arithmetic[0])
        else:
            for index in arithmetic[:2]:
                wrong("sparse", index)
                wrong("control", index)
        if scenario == "empty_supa_transition":
            for index, row in enumerate(references):
                if row["domain"] == "transition": wrong("supa", index)
        if scenario == "empty_sparse_transition":
            for index, row in enumerate(references):
                if row["domain"] == "transition": wrong("sparse", index)
        freeze = analysis.freeze_baselines(references,
            {model: predictions[model] for model in ("supa", "sparse", "transition")},
            {"supa": common.SUPA_SHA, "sparse": common.PARENT_SHA, "transition": analysis.TRANSITION_SHA},
            evidence_kind="fixture")
        arm = {"parent_sha256": common.PARENT_SHA, "selected_case_ids": ["fixture.a", "fixture.b"],
               "sequence_digest": "fixture.sequence", "reasoning_labels_digest": "fixture.reasoning",
               "groups": [[0, 1]], "planned_denominators": [4], "processed_tokens": 8, "updates": 1,
               "plan_sha256": "fixture.plan", "status": "COMPLETE", "frozen_parameters_unchanged": True,
               "gradients_cleared": True, "final_checkpoint_sha256": "a" * 64}
        proof = {"evidence_kind": "fixture", "control": deepcopy(arm), "joint": deepcopy(arm)}
        expected_legacy = analysis.analyze_fixture(references, predictions, freeze, proof, reasoning_shortcut_floor=.5)
    return prompts, references, predictions, freeze, proof, expected_legacy


def _inspect(case):
    return guards.inspect_fixture_claim(*case[:5], reasoning_shortcut_floor=.5)


def fixture_demo_inputs(scenario="measurable"):
    """Root demo kwargs using only reviewed open data and synthetic predictions.

    This constructs no corpus or file and imports no model. Supported examples
    are measurable, all_non_R_wrong, shallow_only and empty_non_R_recovery.
    """
    if scenario not in ("measurable", "all_non_R_wrong", "shallow_only", "empty_non_R_recovery"):
        raise ValueError("unknown named fixture demonstration")
    case = _case(guards.load_open_pilot_fixture(), scenario)
    return {"prompts": case[0], "references": case[1], "predictions": case[2], "freeze": case[3],
            "pair_proof": case[4], "reasoning_shortcut_floor": .5, "evidence_kind": "fixture"}


def test_exact_vacuous_all_wrong_non_R_counterexample_is_blocked(open_pilot):
    case = _case(open_pilot, "all_non_R_wrong")
    result = _inspect(case)
    assert all(case[-1]["gates"].values()) and case[-1]["release_full_recovery_satisfied_fixture_only"]
    assert result["legacy_diagnostics"] == case[-1]
    assert not result["guarded_fixture_quality_gate_pass"]
    assert result["retention_metrics"] is result["recovery_metrics"] is None
    assert not result["full_recovery_claim_satisfied_fixture_only"]
    for panels in result["eligibility_by_non_R_domain"].values():
        assert all(panel["case_count"] == panel["parent_count"] == 0 for panel in panels.values())
    assert not result["release_permitted"] and not result["model_quality_established"]


def test_four_shallow_wins_pass_original_margin_without_proving_deep_gain(open_pilot):
    case = _case(open_pilot, "shallow_only")
    result = _inspect(case)
    assert all(result["legacy_diagnostics"]["gates"].values())
    accuracy = result["legacy_diagnostics"]["reasoning_accuracy"]
    assert accuracy["joint"] - accuracy["sparse"] == .0625
    deep = result["deep_reasoning"]
    assert deep["deep_case_count"] == 31 and deep["deep_parent_count"] == 27
    assert deep["reasoning_parent_count"] == 32 and deep["reasoning_family_count"] == 1
    assert {depth: value["case_count"] for depth, value in deep["queried_depth_strata"].items()} == {"1": 33, "2": 12, "3": 19}
    assert deep["comparisons"]["sparse"]["equal_weight_parent_delta"] == 0
    assert not deep["positive_deep_parent_gain_vs_sparse"]
    assert not result["additional_claim_rules"]["gates"]["NEW_positive_deep_parent_gain_vs_sparse"]
    assert "PENDING_OWNER_REGISTRATION" in result["additional_claim_rules"]["scope"]
    assert not result["guarded_fixture_quality_gate_pass"]


def test_measurable_all_domain_fixture_remains_mechanics_with_legacy_unchanged(open_pilot):
    case = _case(open_pilot)
    result = _inspect(case)
    assert result["legacy_diagnostics"] == case[-1]
    assert result["legacy_gates_unchanged"] and result["guarded_fixture_quality_gate_pass"]
    assert result["full_recovery_claim_satisfied_fixture_only"]
    for panels in result["eligibility_by_non_R_domain"].values():
        for name in ("sparse_correct", "supa_correct"):
            assert panels[name]["case_count"] > 0 and panels[name]["parent_count"] > 0
    assert result["recovery_domain_status"]["transition"] == "NO_OBSERVED_DEFICIT"
    assert result["recovery_metrics"]["transition"]["metrics"] is None
    assert result["deep_reasoning"]["comparisons"]["sparse"]["gained_parent_ids"]
    assert result["deep_reasoning"]["comparisons"]["control"]["equal_weight_parent_delta"] == 0
    assert not result["native_winner_declared"] and not result["release_permitted"]
    assert not result["native_authorized"] and result["native_model_calls"] == 0
    assert not result["model_quality_established"] and result["evidence_kind"] == "fixture"


def test_empty_non_R_recovery_never_becomes_restoration(open_pilot):
    case = _case(open_pilot, "empty_non_R_recovery")
    result = _inspect(case)
    assert result["legacy_diagnostics"]["release_full_recovery_satisfied_fixture_only"]
    assert result["additional_claim_rules"]["gates"]["required_non_R_source_panels_nonempty"]
    assert result["retention_metrics"] is not None
    assert result["non_R_recovery_panel"]["case_count"] == 0
    assert result["recovery_status"] == "UNMEASURABLE_NO_OBSERVED_NON_R_DEFICIT"
    assert result["recovery_metrics"] is None
    assert not result["full_recovery_claim_satisfied_fixture_only"]
    assert not result["guarded_fixture_quality_gate_pass"]


def test_retention_and_recovery_source_measurements_are_distinct(open_pilot):
    sparse_present = _inspect(_case(open_pilot, "empty_supa_transition"))
    assert sparse_present["retention_metrics"] is not None
    assert sparse_present["recovery_metrics"] is None
    assert "transition:supa_correct" in sparse_present["missing_required_source_panels"]
    assert not sparse_present["guarded_fixture_quality_gate_pass"]
    supa_present = _inspect(_case(open_pilot, "empty_sparse_transition"))
    assert supa_present["retention_metrics"] is None
    assert supa_present["recovery_metrics"] is not None
    assert "transition:sparse_correct" in supa_present["missing_required_source_panels"]
    assert not supa_present["guarded_fixture_quality_gate_pass"]
    assert not supa_present["release_permitted"]


def test_mismatched_prompt_ids_reject_before_claim(open_pilot):
    case = list(_case(open_pilot))
    case[0][0]["case_id"] = "foreign.open.case"
    with pytest.raises(ValueError, match="prompt ID"):
        _inspect(case)


def test_missing_parent_pair_rejects_even_with_full_row_count(open_pilot):
    case = list(_case(open_pilot))
    references = case[1]
    other = next(row for row in references if row["parent_id"] != references[0]["parent_id"])
    references[0]["parent_id"] = other["parent_id"]
    with pytest.raises(ValueError, match="paired parent"):
        _inspect(case)


def test_dependency_cycle_in_public_graph_rejects(open_pilot):
    case = list(_case(open_pilot))
    with guards._bound_v2() as modules:
        data = modules["data"]
        reference = next(row for row in case[1] if row["domain"] == "reasoning")
        prompt = next(row for row in case[0] if row["case_id"] == reference["case_id"])
        task = data.parse_prompt(prompt["messages"])
        task["expressions"]["unused"] = ["unused", "+", "a"]
        prompt["messages"] = data.render_messages(task, reference["requested_format"])
    with pytest.raises(ValueError, match="cycle"):
        _inspect(case)


def test_native_import_closed_before_loading_review(monkeypatch):
    monkeypatch.setattr(guards, "_verified_review", lambda: (_ for _ in ()).throw(AssertionError("review import occurred")))
    with pytest.raises(PermissionError, match="closed"):
        guards.inspect_fixture_claim(None, None, None, None, None, reasoning_shortcut_floor=.5, evidence_kind="native")


def test_independent_review_checksum_mismatch_rejects(monkeypatch):
    monkeypatch.setattr(guards, "REVIEW_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="review checksum"):
        guards.load_open_pilot_fixture()


def test_verified_source_buffers_are_not_reopened_for_execution(monkeypatch):
    original = Path.read_bytes
    watched = {guards.REVIEW_PATH.resolve()} | {(guards.V2_ROOT / name).resolve() for name in guards.BOUND_SOURCES}
    counts = {}
    def read_once(path):
        resolved = path.resolve()
        if resolved in watched:
            counts[resolved] = counts.get(resolved, 0) + 1
            if counts[resolved] > 1: raise AssertionError("verified bytes were reopened")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read_once)
    before = {name for name in sys.modules if name.startswith("_titan_claim_guard_v2_")}
    with guards._bound_v2() as modules:
        assert modules["analysis"].MODELS == guards.MODELS
    assert all(counts[path] == 1 for path in watched)
    assert {name for name in sys.modules if name.startswith("_titan_claim_guard_v2_")} == before


def test_verified_open_artifact_buffers_are_not_reopened_for_parse(monkeypatch):
    original = Path.read_bytes
    watched = {guards.REVIEW_PATH.resolve()} | {(guards.V2_ROOT / name).resolve() for name in guards.BOUND_SOURCES}
    watched |= {(guards.V2_RUNS / "data-001" / name).resolve() for name in ("pilot.prompts.jsonl", "pilot.references.jsonl")}
    counts = {}
    def read_once(path):
        resolved = path.resolve()
        if resolved in watched:
            counts[resolved] = counts.get(resolved, 0) + 1
            if counts[resolved] > 1: raise AssertionError("hash-validated artifact bytes were reopened")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read_once)
    prompts, references = guards.load_open_pilot_fixture()
    assert len(prompts) == len(references) == 384
    assert all(counts[path] == 1 for path in watched)


def test_actual_caller_inputs_are_bound_separately_from_source_review(open_pilot):
    case = _case(open_pilot)
    result = _inspect(case)
    binding = result["caller_fixture_input_bindings"]
    values = dict(zip(("prompts", "references", "predictions", "baseline_freeze", "pair_proof"), case[:5]))
    values["reasoning_shortcut_floor"] = .5
    def independent_sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    assert binding["sha256"] == {key: independent_sha(value) for key, value in values.items()}
    assert binding["combined_inputs_sha256"] == independent_sha(values)
    assert not binding["reviewed_open_artifact_membership_attested"] and not binding["native_inputs_attested"]
    changed = deepcopy(case)
    changed[2]["control"][0]["response"] = "999999999"
    other = _inspect(changed)["caller_fixture_input_bindings"]
    assert other["combined_inputs_sha256"] != binding["combined_inputs_sha256"]
    assert other["sha256"]["predictions"] != binding["sha256"]["predictions"]
    assert other["sha256"]["references"] == binding["sha256"]["references"]
