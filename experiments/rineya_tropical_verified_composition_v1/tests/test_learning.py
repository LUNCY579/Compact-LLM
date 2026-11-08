from dataclasses import replace
import copy
import math
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.rineya_tropical_verified_composition_v1.archive import Archive
from experiments.rineya_tropical_verified_composition_v1.domain import Task, root
from experiments.rineya_tropical_verified_composition_v1.metrics import paired, pass_at_k, summarize
from experiments.rineya_tropical_verified_composition_v1.model_bridge import CausalActor, EncodedAction
from experiments.rineya_tropical_verified_composition_v1.objectives import (
    causal_response_logps, path_loss, protected_loss, reference_kl)
from experiments.rineya_tropical_verified_composition_v1.runtime import Ledger, Sample, ScriptedPolicy
from experiments.rineya_tropical_verified_composition_v1.study import (
    TinyActor, actor_hash, evaluate, train_fixed_bank)


def fixture():
    task = Task("task", "parent", "train", (0, 0), (1, 1), 3)
    archive = Archive(task)
    for i, path in enumerate((("inc_x", "inc_y"), ("inc_y", "inc_x"))):
        archive.collect(ScriptedPolicy(path), random.Random(0), Ledger(), str(i))
    return task, archive


def vector():
    return torch.tensor([-1.0, -2.0, -4.0], dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize("temperature", [0.01, 0.25, 1.0, 2.0])
def test_soft_gradient_is_negative_softmax(temperature):
    values = vector()
    loss = path_loss(values, torch.ones(3), "soft", temperature)
    loss.backward()
    assert torch.allclose(values.grad, -torch.softmax(values.detach() / temperature, 0))
    assert (values.grad < 0).all()


def test_soft_t1_is_log_mass_of_distinct_archive_paths():
    values = torch.tensor([math.log(0.2), math.log(0.3)], dtype=torch.float64)
    loss = path_loss(values, torch.ones(2), "soft", 1.0)
    assert math.isclose(float(loss), -math.log(0.5))


def test_hard_gradient_and_tie_rule():
    values = torch.tensor([-1.0, -1.0, -3.0], requires_grad=True)
    path_loss(values, torch.ones(3), "hard_max").backward()
    assert values.grad.tolist() == [-1.0, 0.0, 0.0]


def test_tiny_temperature_stable():
    values = vector()
    result = path_loss(values, torch.ones(3), "soft", 1e-300)
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(values.grad).all()
    assert math.isclose(float(result.detach()), 1.0)


def test_max_improvement_can_reduce_pass1_in_shared_parameter_actor():
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    starting = torch.tensor([0.31, 0.29, 0.40], dtype=torch.float64).log()
    directions = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float64)
    old = (starting + theta * directions).log_softmax(0)
    loss = path_loss(old[:2], torch.ones(2), "hard_max")
    loss.backward()
    updated = theta.detach() - 0.01 * theta.grad
    new = (starting + updated * directions).softmax(0)
    assert new[0] > 0.31 and new[:2].sum() < 0.60


def test_uniform_coverage_and_length_normalization_are_distinct():
    values = torch.tensor([-2.0, -6.0])
    lengths = torch.tensor([1, 3])
    assert float(path_loss(values, lengths, "uniform")) == 4.0
    assert float(path_loss(values, lengths, "coverage", selected=(0, 1))) == 4.0
    assert float(path_loss(values, lengths, "paper_token_mean", selected=(0, 1))) == 2.0


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        path_loss(vector(), torch.ones(3), "soft", temperature)


@pytest.mark.parametrize("values,lengths", [([], []), ([float("nan")], [1]),
    ([0.2], [1]), ([-1], [0]), ([-1], [float("nan")])])
def test_invalid_path_scores(values, lengths):
    with pytest.raises(ValueError):
        path_loss(torch.tensor(values), torch.tensor(lengths), "uniform")


@pytest.mark.parametrize("selected", [None, (), (0, 0), (3,), (-1,), (True,)])
def test_invalid_coverage_indices(selected):
    with pytest.raises(ValueError):
        path_loss(vector(), torch.ones(3), "coverage", selected=selected)


def encoded():
    # ID 0 acts as both EOS and padding; only the explicit masks disambiguate.
    ids = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    attention = torch.tensor([[True, True, True, True, False]])
    response = torch.tensor([[False, False, True, True, False]])
    return EncodedAction(ids, attention, response)


def test_single_shift_and_eos_mask():
    e = encoded()
    logits = torch.zeros(1, 5, 5, requires_grad=True)
    values, lengths = causal_response_logps(logits, e.input_ids, e.attention_mask, e.response_mask)
    assert lengths.tolist() == [2]
    assert torch.allclose(values, torch.tensor([-2 * math.log(5)]))
    values.sum().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 1:3].abs().sum() > 0
    assert logits.grad[0, 3:].abs().sum() == 0


@pytest.mark.parametrize("problem", ["first", "padding", "empty", "predecessor", "dtype", "ids", "nan"])
def test_invalid_masks_and_logits(problem):
    e = encoded()
    ids, attention, response = e.input_ids.clone(), e.attention_mask.clone(), e.response_mask.clone()
    logits = torch.zeros(1, 5, 5)
    if problem == "first": response[0, 0] = True
    if problem == "padding": response[0, 4] = True
    if problem == "empty": response[:] = False
    if problem == "predecessor": attention[0, 1] = False
    if problem == "dtype": response = response.long()
    if problem == "ids": ids[0, 1] = 6
    if problem == "nan": logits[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        causal_response_logps(logits, ids, attention, response)


def test_reference_kl_detaches_reference_and_masks():
    torch.manual_seed(4)
    student = torch.randn(1, 3, 4, requires_grad=True)
    reference = torch.randn(1, 3, 4, requires_grad=True)
    mask = torch.tensor([[False, True, True]])
    loss = reference_kl(student, reference, mask)
    loss.backward()
    assert loss >= 0 and reference.grad is None
    assert student.grad[0, 0].abs().sum() == 0
    assert student.grad[0, 1:].abs().sum() > 0


def test_protection_missing_reservoir_fails():
    with pytest.raises(ValueError):
        protected_loss(torch.tensor(1.0), acquired_weight=1.0)
    result = protected_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), 0.5, 0.25)
    assert float(result) == 2.75


class MockLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(5, 5))

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        return SimpleNamespace(logits=self.weight[input_ids])


def test_object_bound_native_bridge_gradient_and_accounting():
    t, _ = fixture()
    actor = CausalActor(MockLM(), lambda *_: encoded(), lambda *_: Sample("inc_x", 2, 3))
    ledger = Ledger()
    values, lengths = actor.paths(t, [("inc_x", "inc_y")], ledger)
    (-values.mean()).backward()
    assert lengths.tolist() == [4]
    assert ledger.used["supervised_tokens"] == 4
    assert actor.model.weight.grad.abs().sum() > 0
    assert actor.sample(t, root(t), random.Random(0), 2).generated_tokens == 2
    actor.score(t, root(t), "inc_x", ledger)
    assert ledger.used["scored_tokens"] == 2


def test_native_bridge_rejects_eval_and_duplicate_paths():
    t, _ = fixture()
    actor = CausalActor(MockLM(), lambda *_: encoded(), lambda *_: Sample("inc_x"))
    for task, paths in ((replace(t, role="dev"), [("inc_x", "inc_y")]),
                        (t, [("inc_x", "inc_y"), ("inc_x", "inc_y")])):
        with pytest.raises(ValueError):
            actor.paths(task, paths, Ledger())


def test_same_training_engine_accepts_mock_causal_actor():
    _, archive = fixture()
    actor = CausalActor(MockLM(), lambda *_: encoded(), lambda *_: Sample("inc_x"))
    result = train_fixed_bank(actor, [archive], "coverage", 2)
    assert result["ledger"]["used"]["optimizer_steps"] == 2
    assert result["ledger"]["used"]["scored_tokens"] > 0
    assert result["ledger"]["used"]["supervised_tokens"] > 0


def row(identity="x", parent="p", successes=(True, False)):
    return {"task_id": identity, "parent_id": parent, "task_key": identity,
            "contract_id": "fixed", "successes": list(successes)}


@pytest.mark.parametrize("c,n,k,expected", [(1, 4, 1, .25), (1, 4, 4, 1), (0, 4, 2, 0), (4, 4, 2, 1), (2, 4, 2, 5 / 6)])
def test_passk(c, n, k, expected):
    assert math.isclose(pass_at_k(c, n, k), expected)


@pytest.mark.parametrize("counts", [(0, 0, 1), (3, 2, 1), (0, 2, 3), (1, 2, 0), (True, 2, 1)])
def test_invalid_passk(counts):
    with pytest.raises(ValueError):
        pass_at_k(*counts)


def test_retention_zero_denominator_and_loss_accounting():
    result = paired([row(successes=(False, False))], [row(successes=(True, False))])
    assert result["retention"] is None and result["gained_first_draw"] == 1
    result = paired([row(successes=(True, False))], [row(successes=(False, True))])
    assert result["lost"] == 1 and result["retention"] == 0


@pytest.mark.parametrize("change", [{"task_key": "changed"}, {"contract_id": "other"}, {"parent_id": "other"}, {"successes": [True]}])
def test_pair_contract_mismatch(change):
    changed = {**row(), **change}
    with pytest.raises(ValueError):
        paired([row()], [changed])


def test_duplicate_evaluation_and_parent_macro():
    with pytest.raises(ValueError):
        summarize([row(), row()])
    before = [row("a", "p", (False,)), row("b", "p", (False,)), row("c", "q", (True,))]
    after = [row("a", "p", (True,)), row("b", "p", (True,)), row("c", "q", (False,))]
    assert paired(before, after)["parent_macro_delta"] == 0


def test_tiny_training_and_model_only_evaluation_reproducible():
    torch.set_num_threads(1)
    task, archive = fixture()
    a, b = TinyActor(7), TinyActor(7)
    original = actor_hash(a)
    for actor in (a, b):
        result = train_fixed_bank(actor, [copy.deepcopy(archive)], "soft_1", 3)
        assert result["ledger"]["used"]["optimizer_steps"] == 3
        assert result["ledger"]["used"]["generated_tokens"] == 0
    assert actor_hash(a) == actor_hash(b) != original
    dev = replace(task, role="dev")
    left, right = evaluate(a, [dev], 5), evaluate(b, [dev], 5)
    assert left == right and len(left[0][0]["actions"]) == 4
    assert left[1]["used"]["restart_attempts"] == 0


def test_no_success_does_not_create_an_update():
    result = train_fixed_bank(TinyActor(1), [Archive(fixture()[0])], "uniform", 2)
    assert result["status"] == "NO_VERIFIED_TRAINING_PATHS"
    assert result["ledger"]["used"]["optimizer_steps"] == 0


def test_candidate_mass_detects_double_counting():
    with pytest.raises(ValueError, match="mass"):
        path_loss(torch.tensor([-0.1, -0.2]), torch.ones(2), "soft")


def test_temperature_underflow_fails_explicitly():
    with pytest.raises(ValueError, match="underflows"):
        path_loss(torch.tensor([-1.0, -2.0], dtype=torch.float32), torch.ones(2), "soft", 1e-300)


def test_fractional_response_length_rejected():
    with pytest.raises(ValueError):
        path_loss(torch.tensor([-1.0]), torch.tensor([0.5]), "uniform")


def test_candidate_diversity_gate_is_not_native_authority():
    from experiments.rineya_tropical_verified_composition_v1.study import bank_diagnostics
    _, archive = fixture()
    result = bank_diagnostics([archive])
    assert result["status"] == "OBJECTIVE_DISCRIMINATION_READY"
    assert result["multiple_path_fraction"] == 1.0 and not result["native_execution_authorized"]


def test_singleton_bank_fails_discrimination_gate():
    from experiments.rineya_tropical_verified_composition_v1.study import bank_diagnostics
    task, _ = fixture()
    archive = Archive(task)
    archive.collect(ScriptedPolicy(("inc_x", "inc_y")), random.Random(0), Ledger(), "one")
    result = bank_diagnostics([archive])
    assert result["status"] == "INSUFFICIENT_CANDIDATE_DIVERSITY"


def test_bank_digest_ignores_training_history_not_path_content():
    from experiments.rineya_tropical_verified_composition_v1.study import bank_diagnostics
    _, archive = fixture()
    before = bank_diagnostics([archive])["source_bank_sha256"]
    archive.mark_trained(list(archive.candidates.values()))
    assert bank_diagnostics([archive])["source_bank_sha256"] == before
