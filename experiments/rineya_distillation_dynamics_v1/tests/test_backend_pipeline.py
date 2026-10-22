from dataclasses import replace

import pytest
import torch

from experiments.rineya_distillation_dynamics_v1.backend import tensor_hash
from experiments.rineya_distillation_dynamics_v1.contracts import ContractError, digest, matrix
from experiments.rineya_distillation_dynamics_v1.engine import run_arm, schedule, teacher_bank
from experiments.rineya_distillation_dynamics_v1.fixtures import fixture_backend, fixture_plan, fixture_tasks
from experiments.rineya_distillation_dynamics_v1.integration import peft_names
from experiments.rineya_distillation_dynamics_v1.metrics import evaluate, paired, pass_at_k
from experiments.rineya_distillation_dynamics_v1.qualification import qualify


def test_paired_initialization():
    a, b = fixture_backend(), fixture_backend()
    assert tensor_hash(a.snapshot()) == tensor_hash(b.snapshot())
    assert a.frozen_hashes() == b.frozen_hashes()


def test_generation_deterministic():
    b = fixture_backend()
    p = fixture_tasks()[0].public()
    assert b.sample(p, "student", 6) == b.sample(p, "student", 6)


def test_public_dto_enforced():
    with pytest.raises(ContractError): fixture_backend().sample(fixture_tasks()[0], "student", 6)


@pytest.mark.parametrize("kind", ["stale", "wrong_source", "wrong_prompt", "wrong_text", "wrong_termination"])
def test_rollout_audit(kind):
    b = fixture_backend()
    t = next(t for t in fixture_tasks() if t.role == "train")
    r = b.sample(t.public(), "student", 4)
    if kind == "stale": r = replace(r, policy_step=7)
    if kind == "wrong_source": r = replace(r, source="teacher")
    if kind == "wrong_prompt": r = replace(r, prompt_ids=(3,))
    if kind == "wrong_text": r = replace(r, text="invented")
    if kind == "wrong_termination": r = replace(r, terminated=not r.terminated)
    with pytest.raises(ContractError): b.distill(t, r, "forward", "student")


@pytest.mark.parametrize("objective", ["forward", "reverse", "ce"])
def test_no_heldout_gradient(objective):
    b = fixture_backend()
    t = next(t for t in fixture_tasks() if t.role == "dev_acquisition")
    with pytest.raises(ContractError):
        if objective == "ce": b.supervised(t)
        else: b.distill(t, b.sample(t.public(), "student", 5), objective, "student")


def test_no_silent_context_truncation():
    b = fixture_backend()
    p = replace(fixture_tasks()[0].public(), prompt="x"*200)
    with pytest.raises(ContractError): b.sample(p, "student", 1)


def test_response_causal_shift():
    b = fixture_backend()
    p, r = (1, 2, 3), (4, 5)
    h = b.hidden(b.student, p, r, True, "student")
    full = b.student.model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).last_hidden_state[0]
    assert torch.equal(h, full[2:4])


def test_no_padding_eos_interior():
    with pytest.raises(ContractError): fixture_backend().response((3, 0, 4))


def test_teacher_bank_cache_and_binding(tmp_path):
    b, ts = fixture_backend(), fixture_tasks()
    path = tmp_path / "bank.json"
    bank, cost, created = teacher_bank(b, ts, 17, path, "binding")
    again, cost2, created2 = teacher_bank(b, ts, 17, path, "binding")
    assert bank == again and cost == cost2 and created and not created2
    with pytest.raises(ContractError): teacher_bank(b, ts, 29, path, "binding")


@pytest.mark.parametrize("index", list(range(10)))
def test_all_factorial_cells_and_sft(tmp_path, index):
    plan, ts = fixture_plan(), fixture_tasks()
    b = fixture_backend(plan)
    before = b.frozen_hashes()
    result = run_arm(b, ts, matrix(plan)[index], tmp_path / "run", identity="fixture",
                     bank_path=tmp_path / "bank.json")
    assert result["status"] == "COMPLETE" and len(result["checkpoints"]) == 2
    assert b.frozen_hashes() == before and b.version == 0
    assert tensor_hash(b.snapshot()) == tensor_hash(b.initial)
    assert result["checkpoints"][-1]["geometry"]["l2"] > 0
    assert result["resume_supported"] is False and result["native_efficacy_claim"] is False


def test_schedule_is_arm_independent():
    ts = fixture_tasks()
    assert schedule(ts, 2, 17, 3, "train") == schedule(ts, 2, 17, 3, "train")
    assert schedule(ts, 2, 17, 3, "train") != schedule(ts, 2, 17, 3, "replay")


def test_failed_run_is_not_reused(tmp_path):
    b = fixture_backend()
    t = fixture_tasks(per_role=1)
    b.plan = replace(b.plan, batch_tasks=2)
    with pytest.raises(ContractError):
        run_arm(b, t, matrix(b.plan)[0], tmp_path / "run", identity="f", bank_path=tmp_path / "bank.json")
    assert (tmp_path / "run" / "FAILED.json").is_file()
    with pytest.raises(FileExistsError):
        run_arm(b, t, matrix(b.plan)[0], tmp_path / "run", identity="f", bank_path=tmp_path / "bank.json")


def test_qualification_restores_state(tmp_path):
    p = replace(fixture_plan(), teacher_min_accuracy=0, teacher_min_advantage=0)
    b = fixture_backend(p)
    before = tensor_hash(b.snapshot())
    result = qualify(b, fixture_tasks(), tmp_path / "q")
    assert result["save_reload_bitwise"] and not result["efficacy_claim"]
    assert tensor_hash(b.snapshot()) == before and b.version == 0


def test_teacher_competence_gate(tmp_path):
    b = fixture_backend()
    with pytest.raises(ContractError, match="teacher competence"):
        qualify(b, fixture_tasks(), tmp_path / "q")


def test_peft_tensor_name_mapping():
    assert set(peft_names({"model.layers.0.self_attn.q_proj.lora_A": torch.ones(1, 2)})) == {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"}


@pytest.mark.parametrize("c,n,k,expected", [(0, 4, 1, 0), (4, 4, 2, 1), (1, 4, 1, .25), (1, 4, 2, .5), (1, 4, 4, 1)])
def test_pass_at_k(c, n, k, expected):
    assert pass_at_k(c, n, k) == expected


def test_complete_evaluation_denominators():
    b = fixture_backend()
    ts = [t for t in fixture_tasks() if t.role.startswith("dev_")]
    rows = evaluate(b, ts, "student", 17)
    assert set(rows) == {t.key for t in ts}
    summary = paired(rows, rows, bootstrap=20)
    assert summary["all_tasks_paired_gain_ci95"] == [0, 0]
    assert all(g["gain"] == 0 and g["damaged"] == 0 for g in summary["groups"].values())
    rows2 = dict(rows)
    rows2.pop(next(iter(rows2)))
    with pytest.raises(ContractError): paired(rows, rows2)
