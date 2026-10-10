from copy import deepcopy
import pytest
from titan_joint_quality_v1.common import PARENT_SHA, SUPA_SHA
from titan_joint_quality_v1.common import digest
from titan_joint_quality_v1.data import generate_bank
from titan_joint_quality_v1.analysis import (TRANSITION_SHA, score_response, freeze_baselines,
    analyze_fixture, score_predictions)

def fixture():
    refs = [{k: v for k, v in row.items() if k != "messages"} for row in generate_bank("pilot")]
    predictions = {m: [{"case_id": r["case_id"], "response": r["target"], "completed": True, "truncated": False}
                       for r in refs] for m in ("supa", "sparse", "transition", "control", "joint")}
    reason = [i for i, r in enumerate(refs) if r["domain"] == "reasoning"]
    arithmetic = [i for i, r in enumerate(refs) if r["domain"] == "arithmetic"]
    for i in reason[:32] + arithmetic[:2]:
        predictions["sparse"][i]["response"] = "999999999"
    for i in arithmetic[:2]:
        predictions["control"][i]["response"] = "999999999"
    freeze = freeze_baselines(refs, {k: predictions[k] for k in ("supa", "sparse", "transition")},
        {"supa": SUPA_SHA, "sparse": PARENT_SHA, "transition": TRANSITION_SHA}, evidence_kind="fixture")
    arm = {"parent_sha256": PARENT_SHA, "selected_case_ids": ["a", "b"], "sequence_digest": "s",
        "reasoning_labels_digest": "r", "groups": [[0, 1]], "planned_denominators": [4],
        "processed_tokens": 8, "updates": 1, "plan_sha256": "p", "status": "COMPLETE",
        "frozen_parameters_unchanged": True, "gradients_cleared": True, "final_checkpoint_sha256": "a" * 64}
    proof = {"evidence_kind": "fixture", "control": deepcopy(arm), "joint": deepcopy(arm)}
    return refs, predictions, freeze, proof

def test_complete_paired_gates_are_only_fixture_evidence():
    refs, pred, freeze, proof = fixture()
    result = analyze_fixture(refs, pred, freeze, proof, reasoning_shortcut_floor=.1)
    assert all(result["gates"].values()), result
    assert result["release_full_recovery_satisfied_fixture_only"]
    assert not result["model_quality_established"] and not result["advance_to_replication"]
    assert result["parent_units"] == 192 and result["case_count"] == 384

@pytest.mark.parametrize("defect", ["missing", "duplicate", "failed", "truncated", "extra"])
def test_no_survivor_comparison(defect):
    refs, pred, _, _ = fixture(); rows = pred["joint"]
    if defect == "missing": rows.pop()
    elif defect == "duplicate": rows.append(deepcopy(rows[0]))
    elif defect == "failed": rows[0]["completed"] = False
    elif defect == "truncated": rows[0]["truncated"] = True
    else: rows[0]["case_id"] = "foreign-case"
    with pytest.raises(ValueError): score_predictions(refs, rows)

@pytest.mark.parametrize("field,value", [("processed_tokens", 54173), ("planned_denominators", [0]),
    ("updates", 0), ("groups", []), ("parent_sha256", "wrong"), ("status", "PARTIAL"),
    ("gradients_cleared", False), ("frozen_parameters_unchanged", False)])
def test_source_exposure_checkpoint_failures_leave_gates_failed(field, value):
    refs, pred, freeze, proof = fixture(); proof["joint"][field] = value
    result = analyze_fixture(refs, pred, freeze, proof, reasoning_shortcut_floor=.1)
    assert not result["gates"]["complete_matched_pair"] and result["errors"]

def test_new_source_correct_loss_is_not_offset_by_new_reasoning_wins():
    refs, pred, freeze, proof = fixture()
    i = next(i for i, r in enumerate(refs) if r["case_id"] in freeze["sparse_correct_cases"])
    pred["joint"][i]["response"] = "999999999"
    result = analyze_fixture(refs, pred, freeze, proof, reasoning_shortcut_floor=.1)
    assert not result["gates"]["zero_new_sparse_correct_losses"]
    assert refs[i]["case_id"] in result["correct_case_losses"]["joint"]

def test_freeze_cannot_change_after_candidate_scoring():
    refs, pred, freeze, proof = fixture(); pred["supa"][0]["response"] = "different"
    result = analyze_fixture(refs, pred, freeze, proof, reasoning_shortcut_floor=.1)
    assert any("changed after freeze" in error for error in result["errors"])

def test_native_import_and_native_freeze_are_closed():
    refs, pred, freeze, proof = fixture(); proof["evidence_kind"] = "native"
    with pytest.raises(PermissionError): analyze_fixture(refs, pred, freeze, proof, reasoning_shortcut_floor=.1)
    with pytest.raises(PermissionError): freeze_baselines(refs, {}, {}, evidence_kind="native")

def test_format_failure_is_not_renamed_content_failure():
    r = {"canonical_answer": "7", "target": '{"answer":"7"}', "requested_format": "json", "domain": "format"}
    s = score_response("7.", r)
    assert s["semantic_correct"] and not s["strict_correct"] and not s["format_valid"]
    assert not score_response("The answer is 7", r)["semantic_correct"]
    assert score_response('<answer>7</answer>', r)["semantic_correct"]

def test_unsupported_answer_and_false_abstention_are_separate():
    r = {"canonical_answer": "UNKNOWN", "target": "UNKNOWN", "requested_format": "plain", "domain": "grounding"}
    assert score_response("7", r)["unsupported_answer"]
    assert not score_response("7", r)["false_abstention"]
    r = {**r, "canonical_answer": "7", "target": "7"}
    assert score_response("UNKNOWN", r)["false_abstention"]
    assert score_response("Please clarify", r)["requires_blind_review"]

def test_duplicate_json_key_and_wrong_requested_field_type_fail_format():
    r = {"canonical_answer": "7", "target": '{"answer":"7"}', "requested_format": "json", "domain": "format"}
    assert not score_response('{"answer":"wrong","answer":"7"}', r)["semantic_correct"]
    s = score_response('{"answer":7}', r)
    assert s["semantic_correct"] and not s["format_valid"]

def test_rehashed_fabricated_supa_correct_set_is_rejected():
    refs, pred, freeze, proof = fixture()
    freeze["supa_correct_cases"] = freeze["supa_correct_cases"][:-1]
    freeze["freeze_digest"] = digest({k: v for k, v in freeze.items() if k != "freeze_digest"})
    result = analyze_fixture(refs, pred, freeze, proof, reasoning_shortcut_floor=.1)
    assert not result["gates"]["complete_matched_pair"]
    assert any("membership differs" in e for e in result["errors"])

def test_complete_numeric_truth_normalizes_without_prose_or_symbolic_extraction():
    r = {"canonical_answer": "7", "target": "7", "requested_format": "plain", "domain": "reasoning"}
    assert score_response("+7", r)["semantic_correct"]
    assert score_response("007", r)["semantic_correct"]
    assert not score_response("The answer is +7", r)["semantic_correct"]
    r["domain"] = "format"
    assert not score_response("007", r)["semantic_correct"]
