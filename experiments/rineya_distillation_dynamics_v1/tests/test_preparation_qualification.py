"""Preparation checks: mock teacher outputs/host dispatch, never native updates."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.rineya_distillation_dynamics_v1.contracts import ContractError, read_json, write_new
from experiments.rineya_distillation_dynamics_v1.fixtures import fixture_plan, fixture_tasks
from experiments.rineya_distillation_dynamics_v1.host import launch
from experiments.rineya_distillation_dynamics_v1.qualification import teacher_screen


def test_teacher_screen_includes_capped_wrong_answers():
    tasks = [t for t in fixture_tasks() if t.role == "teacher_dev"]
    answers = {t.key: t.answer for t in tasks}
    calls = []

    def sample(problem, role, seed, *, greedy=False, purpose=None):
        calls.append((problem, role, greedy, purpose))
        text = answers[problem.key] if role == "teacher" else "wrong"
        return SimpleNamespace(text=text, terminated=False, response_ids=(1, 2, 3))

    backend = SimpleNamespace(plan=fixture_plan(), sample=sample)
    result = teacher_screen(backend, tasks)
    assert result["status"] == "PASS_TEACHER_SCREEN"
    assert result["teacher_accuracy"] == 1 and result["student_accuracy"] == 0
    assert result["teacher_dev_n"] == len(tasks) and result["capped_attempts_included"]
    assert all(row["teacher_terminated"] is False for row in result["outputs"])
    assert len(calls) == len(tasks) * 2
    assert all(greedy and purpose == "teacher_competence_screen" for _, _, greedy, purpose in calls)
    assert all(not hasattr(problem, "answer") for problem, *_ in calls)


def test_teacher_screen_failure_keeps_full_denominator():
    tasks = [t for t in fixture_tasks() if t.role == "teacher_dev"]
    backend = SimpleNamespace(plan=replace(fixture_plan(), teacher_min_accuracy=.6),
                              sample=lambda *a, **k: SimpleNamespace(text="wrong", terminated=True,
                                                                    response_ids=(0,)))
    result = teacher_screen(backend, tasks)
    assert result["status"] == "FAIL_TEACHER_SCREEN"
    assert result["teacher_dev_n"] == len(tasks) and len(result["outputs"]) == len(tasks)


@pytest.mark.parametrize("kwargs,reason", [
    ({}, "model-load"),
    ({"allow_model_load": True, "ack_host_available": True}, "explicit authorization"),
    ({"allow_model_load": True, "ack_host_available": True, "allow_training": True,
      "arm": "A-lr0-s17"}, "one run mode"),
    ({"allow_model_load": True, "ack_host_available": True, "allow_training": True,
      "confirmation": "unopened.json"}, "cannot open confirmation"),
])
def test_qualification_permissions_before_registration_read(tmp_path, kwargs, reason):
    with pytest.raises(ContractError, match=reason):
        launch("missing-registration", tmp_path / "q", qualification_only=True, **kwargs)
    assert not (tmp_path / "q").exists()


def test_qualification_dispatch_stops_before_treatment(tmp_path, monkeypatch):
    from experiments.rineya_distillation_dynamics_v1 import host, registry
    commands = []
    monkeypatch.setattr(registry, "check_registration", lambda _: ({}, None, []))
    monkeypatch.setattr(host, "admission", lambda _: object())

    def check_output(command, **kwargs):
        return "" if command[1] == "status" else str(tmp_path)

    monkeypatch.setattr(host.subprocess, "check_output", check_output)

    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(host.subprocess, "Popen", popen)
    out = tmp_path / "qualification-only"
    launch("missing-registration", out, qualification_only=True, allow_model_load=True,
           allow_training=True, ack_host_available=True)
    assert len(commands) == 1 and "--qualification-only" in commands[0]
    assert "--arm" not in commands[0] and "--selection" not in commands[0]
    assert read_json(out / "SUPERVISOR_STARTED.json")["mode"] == "qualification_only"
    assert not (tmp_path / "rineya-central-gpu0.lock").exists()
    assert not (out / "arm").exists()


def test_qualification_finalization_failure_keeps_receipt(tmp_path, monkeypatch):
    from experiments.rineya_distillation_dynamics_v1 import qualification
    out = tmp_path / "failed-finalization"

    def fake_qualify(backend, tasks, output):
        output.mkdir()
        raise ContractError("restoration failed")

    monkeypatch.setattr(qualification, "_qualify", fake_qualify)
    backend = SimpleNamespace(write_prompt_manifest=lambda path, **kwargs: write_new(path, {"complete": False}))
    with pytest.raises(ContractError, match="restoration failed"):
        qualification.qualify(backend, [], out)
    assert read_json(out / "FAILED.json")["restoration_receipt_present"] is False
    assert (out / "CONSUMED_PROMPTS_INCOMPLETE.json").exists()


def test_qualification_keeps_primary_failure_when_cleanup_fails(tmp_path, monkeypatch):
    from experiments.rineya_distillation_dynamics_v1 import qualification
    out = tmp_path / "primary-and-cleanup"

    def fake_qualify(backend, tasks, output):
        output.mkdir()
        write_new(output / "FAILED.json", {"message": "primary failure"})
        raise ContractError("cleanup failure")

    monkeypatch.setattr(qualification, "_qualify", fake_qualify)
    backend = SimpleNamespace(write_prompt_manifest=lambda path, **kwargs: write_new(path, {"complete": False}))
    with pytest.raises(ContractError, match="cleanup failure"):
        qualification.qualify(backend, [], out)
    assert read_json(out / "FAILED.json")["message"] == "primary failure"
    assert read_json(out / "FINALIZATION_FAILED.json")["message"] == "cleanup failure"


def test_native_completion_cannot_ignore_failed_cleanup(tmp_path):
    from experiments.rineya_distillation_dynamics_v1.registry import check_completed_native_run
    arm = tmp_path / "invocation" / "arm"
    write_new(arm / "RESULT.json", {"status": "COMPLETE", "mechanical_fixture": True})
    write_new(arm / "RESTORE_FAILED.json", {"message": "mock cleanup failure"})
    with pytest.raises(ContractError, match="failed native"):
        check_completed_native_run(arm / "RESULT.json", "fixture")
