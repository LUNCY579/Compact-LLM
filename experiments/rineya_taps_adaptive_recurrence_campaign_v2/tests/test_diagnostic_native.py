import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.diagnostic_native import (
    bind_population, discrepancy_gate, loss_diagnostic,
)
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_sensitivity import FixtureModel


def population_fixture():
    cases = [{"case_id": f"case-{i}", "family": "fixture", "scenario_id": f"parent-{i}",
              "scorer": {"type": "exact", "answers": ["2"]} if i < 7 else {"type": "choice", "value": "YES"}}
             for i in range(16)]
    metadata = [{"case_id": case["case_id"], "split": "DEV", "input_token_ids": [1, 2],
                 "input_tokens": 2237 if index >= 14 else 2, "rendered_prompt_sha256": "prefix"}
                for index, case in enumerate(cases)]
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "2" and add_special_tokens is False
            return [2]
    calls = []
    def canonical(case):
        calls.append(case["case_id"])
        return case["scorer"]["answers"][0] if case["scorer"]["type"] == "exact" else None
    return cases, metadata, Tokenizer(), canonical, calls


def test_population_shape_exclusion_precedes_target_access_and_choice_is_not_guessed():
    cases, metadata, tokenizer, canonical, calls = population_fixture()
    bound = bind_population(cases, metadata, tokenizer, canonical)
    assert len(bound) == 16 and len(calls) == 14
    assert sum(row["status"] == "ELIGIBLE" for row in bound) == 7
    assert sum(row["status"] == "EXCLUDED_SHAPE" for row in bound) == 2
    assert sum(row["status"] == "NOT_APPLICABLE_INHERITED_TARGET" for row in bound) == 7
    assert all(row["target_token_id"] is None for row in bound[7:])
    assert not any(row["gold_enters_forward_input"] for row in bound)


def test_population_count_change_and_guessed_target_convention_are_refused():
    cases, metadata, tokenizer, canonical, calls = population_fixture()
    with pytest.raises(RuntimeError): bind_population(cases[:-1], metadata, tokenizer, canonical)
    with pytest.raises(RuntimeError): bind_population(cases, metadata, tokenizer, lambda _case: "2")


def test_registered_fd_tolerance_fails_unsafe_and_nonfinite_results():
    assert discrepancy_gate({"sensitivity": 2.}, {"loss": 1.02}, {"loss": .98})["absolute_error"] < 1e-12
    with pytest.raises(FloatingPointError): discrepancy_gate({"sensitivity": 0.}, {"loss": 2.}, {"loss": 1.})
    with pytest.raises(FloatingPointError): discrepancy_gate({"sensitivity": float("nan")}, {"loss": 1.}, {"loss": 1.})


def test_diagnostic_journals_three_recomputed_forwards_one_state_adjoint_without_parameter_grads():
    model = FixtureModel()
    ids = torch.tensor([[3, 5, 7]], dtype=torch.long)
    events = []
    result = loss_diagnostic(model, ids, 0, device="cpu", guard=lambda: {},
                             synchronize=lambda: None, sink=events.append, label="fixture")
    assert result["full_model_calls"] == model.forward_calls == 3
    assert result["state_gradient_evaluations"] == 1 and result["absolute_error"] <= result["registered_tolerance"]
    completed = [event for event in events if event["event"] == "DIAGNOSTIC_FORWARD_COMPLETED"]
    assert len(completed) == 3 and sum(event["state_gradient_evaluations"] for event in completed) == 1
    assert all(torch.equal(prefix, ids) for prefix in model.seen_inputs)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    assert not result["generation_pool_access"] and not result["gold_enters_forward_input"]


def test_failed_suffix_is_journaled_as_unknown_partial_compute_and_restores():
    model, events = FixtureModel(), []
    model.fail_suffix = True
    with pytest.raises(RuntimeError, match="suffix failure"):
        loss_diagnostic(model, torch.tensor([[1, 2]]), 0, device="cpu", guard=lambda: {},
                        synchronize=lambda: None, sink=events.append, label="failed")
    assert [event["event"] for event in events] == ["DIAGNOSTIC_FORWARD_STARTED", "DIAGNOSTIC_FORWARD_FAILED"]
    assert events[-1]["partial_compute"] == "UNKNOWN" and events[-1]["state_gradient_evaluations"] is None
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert all("_sensitivity_owner" not in layer.__dict__ for layer in model.model.layers)
