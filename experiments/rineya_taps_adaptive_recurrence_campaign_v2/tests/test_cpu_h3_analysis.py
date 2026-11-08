import json
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.cpu_h3_analysis import analyze

STATE = "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551"


def bank(tmp_path, missing=False, bad_fd=False):
    result = tmp_path / "result"
    result.mkdir()
    (tmp_path / "terminal_receipt.json").write_text(json.dumps({"status": "WORKER_TERMINAL", "native_worker_seconds": 12, "exit_code": 0}))
    (result / "receipt.json").write_text(json.dumps({"status": "CPU_H3_COMPLETE_NOT_SEMANTIC_EFFICACY",
        "lane": "CPU_H3_STATE_SENSITIVITY_FP32", "loaded_state_sha256": STATE, "source_unchanged": True, "checkpoint_files_unchanged": True,
        "before_state_sha256": STATE, "after_state_sha256": STATE,
        "parameter_gradients": "NONE", "optimizer_steps": 0, "training": False, "native_test": False, "generation_pool_access": False}))
    (result / "qualification.json").write_text(json.dumps({"status": "PASS_CPU_H3_NATIVE_MECHANICS",
        "lane": "CPU_H3_STATE_SENSITIVITY_FP32", "device": "cpu",
        "before_state_sha256": STATE, "after_state_sha256": STATE, "source_unchanged": True, "parameter_gradients": "NONE"}))
    (result / "matrix_forecast.json").write_text(json.dumps({"admitted": True, "device": "cpu", "GPU_timing_used": False}))
    population = [{"case_id": str(i), "status": "ELIGIBLE" if i < 7 else "NOT_APPLICABLE_INHERITED_TARGET"} for i in range(16)]
    (result / "population.json").write_text(json.dumps(population))
    outcomes = []
    for i in range(16):
        if missing and i == 6:
            continue
        row = {"case_id": str(i), "physical_run": i < 7, "scenario_id": "shared" if i < 3 else str(i)}
        if i < 7:
            derivative = 0 if i == 0 else (1 if i < 4 else -1)
            row.update(nominal={"sensitivity": derivative, "balance": .5, "loss": 2},
                       plus={"loss": 2+.01*derivative+(1 if bad_fd and i == 1 else 0)},
                       minus={"loss": 2-.01*derivative})
        outcomes.append(row)
    (result / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes)+"\n")
    events = []
    positions = [84]*11+[256]*12+[84,37,30,34,30,27,32]*3
    for i, length in enumerate(positions):
        events.extend([{"event": "VALUE_FORWARD_STARTED"}, {"event": "VALUE_FORWARD_COMPLETED", "prefix_tokens": length, "state_gradient_evaluations": int(i < 9)}])
    events.extend([{"event": "VALUE_FORWARD_STARTED"}, {"event": "VALUE_FORWARD_FAILED", "error": "RuntimeError: EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION"}])
    for _ in range(2):
        events.extend([{"event": "CACHE_REJECTION_STARTED"}, {"event": "CACHE_REJECTED"}])
    (result / "attempts.jsonl").write_text("\n".join(json.dumps(event) for event in events)+"\n")
    return tmp_path


def test_complete_direction_counts_do_not_treat_neutral_or_related_scenarios_as_independent(tmp_path):
    report = analyze(bank(tmp_path))
    m = report["mechanism"]
    assert m["nonneutral_cases"] == 6 and m["neutral_cases"] == 1
    assert m["coherence_sign_agreement_count"] == 3
    assert m["constant_increase_sign_agreement_count"] == 3
    assert m["constant_decrease_sign_agreement_count"] == 3
    assert m["scenario_clusters"] == 5
    assert m["spearman_balance_vs_negative_derivative"] is None
    assert report["primary_semantic_efficacy"] is None


def test_claimed_complete_receipt_cannot_promote_missing_case(tmp_path):
    report = analyze(bank(tmp_path, missing=True))
    assert report["mechanism"] is None
    assert report["status"] == "INCOMPLETE_NO_PRIMARY_MECHANISM_COMPARISON"
    assert report["unmeasured_case_ids"] == ["6"]


def test_recorded_complete_case_must_pass_unchanged_FD_gate(tmp_path):
    with pytest.raises(ValueError, match="numerical gate"):
        analyze(bank(tmp_path, bad_fd=True))


def test_receipt_cannot_substitute_for_missing_cost_and_qualification_journals(tmp_path):
    run = bank(tmp_path)
    (run / "result/attempts.jsonl").write_text("")
    report = analyze(run)
    assert report["mechanism"] is None and report["status"].startswith("INCOMPLETE")


def test_missing_state_proof_does_not_pass_as_none_equals_none(tmp_path):
    run = bank(tmp_path)
    path = run / "result/receipt.json"
    receipt = json.loads(path.read_text())
    for key in ("loaded_state_sha256", "before_state_sha256", "after_state_sha256"):
        receipt.pop(key)
    path.write_text(json.dumps(receipt))
    assert analyze(run)["mechanism"] is None
