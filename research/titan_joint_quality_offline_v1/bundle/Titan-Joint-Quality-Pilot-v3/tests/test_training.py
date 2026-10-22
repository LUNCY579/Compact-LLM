"""Tiny CPU mathematical/checkpoint fixtures, never Titan quality measurements."""
from copy import deepcopy
from pathlib import Path

import pytest

from titan_joint_quality_v1.common import PARENT_SHA, digest
from titan_joint_quality_v1.exposure import build_exposure_plan, encode_rows
from titan_joint_quality_v1.training import (
    common_denominator_loss, cpu_fixture_checkpoint_digest, load_cpu_fixture_mq_adapter,
    make_cpu_fixture_checkpoint, restore_cpu_fixture_checkpoint, run_cpu_fixture,
    run_native, validate_resource_forecast, verify_masked_reasoning_gradient,
)


def fixture_plan():
    class Adapter:
        def encode_case(self, messages, target):
            return {"ids": [1] * 252 + [2, 3, 5, 6], "response_start": 252,
                    "response_end": 255, "eos_position": 254, "eos_token_id": 5}
    roles = ["reasoning"] * 10 + ["preservation"] * 5 + ["grounding"] * 3 + ["format"] * 2
    domains = {"reasoning": "reasoning", "preservation": "transition", "grounding": "grounding", "format": "format"}
    rows = [{"case_id": f"fixture-{index}", "parent_id": f"fixture-parent-{index}",
             "family_id": f"fixture-{role}", "domain": domains[role], "role": role,
             "messages": [{"role": "user", "content": "opaque CPU fixture"}],
             "canonical_answer": 7, "target": "7", "requested_format": "integer"}
            for index, role in enumerate(roles)]
    return build_exposure_plan(encode_rows(rows, Adapter()),
        {"parent_sha256": PARENT_SHA, "data_sha256": digest(rows),
         "tokenizer_sha256": digest("opaque fixture tokenizer")})


def test_loss_shift_common_denominator_and_masked_newline():
    torch = pytest.importorskip("torch")
    logits = torch.arange(36, dtype=torch.float32).reshape(6, 6).requires_grad_()
    ids = [0, 1, 2, 3, 4, 5]
    mask = [False, False, True, True, True, False]
    actual = common_denominator_loss(logits, ids, mask, 9)
    log_probabilities = logits.log_softmax(-1)
    expected = -(log_probabilities[1, 2] + log_probabilities[2, 3] + log_probabilities[3, 4]) / 9
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)
    actual.backward()
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert torch.count_nonzero(logits.grad[4:]) == 0


def test_all_masked_row_forwards_but_has_zero_loss_and_gradient():
    torch = pytest.importorskip("torch")
    logits = torch.randn(6, 7, dtype=torch.float32, requires_grad=True)
    loss = common_denominator_loss(logits, [1, 2, 3, 4, 5, 6], [False] * 6, 12)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(logits.grad).item() == 0


def test_invalid_denominator_mask_and_native_modes_refuse():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(6, 7, dtype=torch.float32, requires_grad=True)
    for denominator in (0, True, 2.0):
        with pytest.raises(ValueError):
            common_denominator_loss(logits, [1, 2, 3, 4, 5, 6], [False] * 6, denominator)
    with pytest.raises(ValueError):
        common_denominator_loss(logits, [1, 2, 3, 4, 5, 6], [True] * 6, 6)
    with pytest.raises(PermissionError):
        common_denominator_loss(logits, [1, 2, 3, 4, 5, 6], [False] * 6, 6, fixture_only=False)
    with pytest.raises(PermissionError):
        run_native(native_authorized=True, training_authorized=True)
    with pytest.raises(PermissionError):
        run_cpu_fixture(fixture_plan(), fixture_only=False)


def test_masked_reasoning_gradient_is_invariant_when_nonreasoning_targets_change():
    pytest.importorskip("torch")
    proof = verify_masked_reasoning_gradient(fixture_plan())
    assert proof["reasoning_gradients_invariant"]
    assert proof["masked_nonreasoning_gradients_zero"]
    assert not proof["capability_evidence"] and not proof["native_authorized"]


def test_both_arms_keep_exact_exposure_and_full_resume_is_bitwise_equivalent():
    pytest.importorskip("torch")
    plan = fixture_plan()
    joint = run_cpu_fixture(plan, "joint", checkpoint_after=1)
    control = run_cpu_fixture(plan, "reasoning_only", checkpoint_after=1)
    for key in ("parent_sha256", "selected_case_ids", "sequence_digest", "reasoning_labels_digest",
                "groups", "planned_denominators", "processed_tokens", "updates", "plan_digest"):
        assert joint[key] == control[key]
    for receipt in (joint, control):
        assert receipt["status"] == "COMPLETE"
        assert receipt["evidence_class"] == "CPU_FIXTURE_MECHANICS_ONLY"
        assert receipt["full_checkpoint_resume_bitwise_equal"] is True
        assert receipt["frozen_parameters_unchanged"] and receipt["gradients_cleared"]
        assert receipt["final_gradients"] == "NONE"
        assert len(receipt["final_checkpoint_sha256"]) == 64
        assert not receipt["native_authorized"] and not receipt["training_authorized"]
        assert receipt["native_model_calls"] == 0
    assert joint["active_label_tokens"] > control["active_label_tokens"]


def test_checkpoint_requires_all_state_and_cleared_gradients_before_restore():
    torch = pytest.importorskip("torch")
    adapter = load_cpu_fixture_mq_adapter(fixture_only=True)
    design = adapter["Design"]()
    model = torch.nn.Linear(2, 2, dtype=torch.float32)
    optimizer = adapter["CpuAdamW"](model.named_parameters(), design)
    model.weight.grad = torch.zeros_like(model.weight)
    with pytest.raises(ValueError, match="cleared"):
        make_cpu_fixture_checkpoint(model, optimizer, cursor=0, completed_tokens=0, plan_sha256=digest("fixture"))
    optimizer.zero_grad()
    checkpoint = make_cpu_fixture_checkpoint(model, optimizer, cursor=0, completed_tokens=0,
                                             plan_sha256=digest("fixture"))
    assert len(cpu_fixture_checkpoint_digest(checkpoint)) == 64
    altered = deepcopy(checkpoint)
    del altered["torch_cpu_rng_state"]
    with pytest.raises(ValueError, match="incomplete"):
        restore_cpu_fixture_checkpoint(model, optimizer, altered, expected_plan_sha256=digest("fixture"))
    altered = deepcopy(checkpoint)
    altered["optimizer_state"]["weight"][1].fill_(-1)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="moments"):
        restore_cpu_fixture_checkpoint(model, optimizer, altered, expected_plan_sha256=digest("fixture"))
    assert torch.equal(before, model.weight)


def test_hash_changed_mq_source_is_rejected_before_primitive_execution(monkeypatch):
    original = Path.read_bytes
    def changed(path):
        return b"changed fixture source" if path.name == "optimizer.py" else original(path)
    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="dependency changed"):
        load_cpu_fixture_mq_adapter(fixture_only=True)


def test_real_tokenizer_ids_cannot_enter_the_tiny_cpu_fixture():
    plan = fixture_plan()
    plan["groups"][0]["rows"][0]["ids"][0] = 151645
    unsigned = dict(plan)
    del unsigned["plan_sha256"]
    plan["plan_sha256"] = digest(unsigned)
    with pytest.raises(ValueError, match="real tokenizer IDs"):
        run_cpu_fixture(plan)


def forecast_inputs():
    rates = {name: .001 for name in ("model_load_seconds", "max_forward_seconds", "training_seconds_per_token",
             "failed_forward_seconds", "checkpoint_save_seconds", "checkpoint_restore_seconds", "source_hash_seconds",
             "failed_cleanup_seconds", "artifact_finalization_seconds")}
    rates["complete_measured_upper_bounds"] = True
    counts = {"evaluation_configurations": 5, "evaluation_rows_per_configuration": 384,
              "max_forward_calls_per_evaluation": 48,
              "processed_tokens_per_arm": 54172, "qualification_forward_calls": 12,
              "qualification_update_tokens": 128, "failed_forward_calls": 2,
              "failed_update_tokens": 128, "checkpoint_saves": 4, "checkpoint_restores": 2,
              "source_hashes": 9, "failed_attempts": 1}
    return rates, counts


def test_complete_forecast_charges_five_configs_pair_and_failed_work_plus_reserve():
    rates, counts = forecast_inputs()
    forecast = validate_resource_forecast(rates, counts)
    assert forecast["status"] == "PROSPECTIVE_COST_FITS_ONLY"
    assert forecast["evaluation_configurations"] == 5 and forecast["training_arms"] == 2
    assert forecast["complete_seconds"] == forecast["body_seconds"] + 1800
    assert forecast["cost_components"]["failed_forward_work"] > 0
    assert forecast["cost_components"]["failed_update_work"] > 0
    assert not forecast["native_authorized"] and not forecast["source_objective_qualified"]


def test_missing_unknown_nonfinite_or_overcap_measurements_block():
    rates, counts = forecast_inputs()
    for bad in (None, float("nan"), float("inf"), False, 0):
        changed = dict(rates, failed_forward_seconds=bad)
        assert validate_resource_forecast(changed, counts)["status"] == "BLOCKED_RESOURCE_FORECAST"
    assert validate_resource_forecast({}, counts)["status"] == "BLOCKED_RESOURCE_FORECAST"
    rates["max_forward_seconds"] = 100
    assert validate_resource_forecast(rates, counts)["status"] == "BLOCKED_COMPLETE_COST_EXCEEDS_CAP"


def test_understated_eval_population_forward_cap_and_unbound_training_budget_block():
    rates, counts = forecast_inputs()
    for key, value in (("evaluation_configurations", 1), ("evaluation_rows_per_configuration", 1),
                       ("max_forward_calls_per_evaluation", 1), ("processed_tokens_per_arm", 1)):
        changed = dict(counts, **{key: value})
        assert validate_resource_forecast(rates, changed)["status"] == "BLOCKED_RESOURCE_FORECAST"
    conservative = validate_resource_forecast(rates, counts)
    assert conservative["status"] == "PROSPECTIVE_COST_FITS_ONLY"
    assert conservative["training_exposure_basis"] == "CONSERVATIVE_FULL_FROZEN_TOKEN_CAP"


def test_plan_bound_forecast_requires_exact_rows_updates_and_positions():
    plan = fixture_plan()
    rates, counts = forecast_inputs()
    counts.update({"processed_tokens_per_arm": plan["processed_tokens_per_arm"],
                   "training_rows_per_arm": plan["selected_rows"], "updates_per_arm": plan["updates"]})
    result = validate_resource_forecast(rates, counts, exposure_plan=plan)
    assert result["status"] == "PROSPECTIVE_COST_FITS_ONLY"
    assert result["exposure_plan_sha256"] == plan["plan_sha256"]
    for key in ("processed_tokens_per_arm", "training_rows_per_arm", "updates_per_arm"):
        changed = dict(counts, **{key: 1})
        assert validate_resource_forecast(rates, changed, exposure_plan=plan)["status"] == "BLOCKED_RESOURCE_FORECAST"
    changed_plan = deepcopy(plan)
    changed_plan["groups"][0]["processed_tokens"] -= 1
    assert validate_resource_forecast(rates, counts, exposure_plan=changed_plan)["status"] == "BLOCKED_RESOURCE_FORECAST"
    missing_qualification = dict(counts)
    del missing_qualification["qualification_forward_calls"]
    assert validate_resource_forecast(rates, missing_qualification, exposure_plan=plan)["status"] == "BLOCKED_RESOURCE_FORECAST"
