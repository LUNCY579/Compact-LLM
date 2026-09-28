"""Guard/controller fixtures, including actual unmodified MQ child supervision.

Journal tests isolate the byte-identical RRP budget module from its unrelated
training-data preflight imports: an empty default-RRP LIMITS import fixture is used,
then the real register() receives SCP-NQ-001's complete ceiling. These tests do not
claim a complete RRP preflight regression or a pretrained/native qualification.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from experiments.model_quality_gpu_v1 import resources, supervisor
from experiments.model_quality_gpu_v1.integrity import digest, file_hash, read_json, seal, write_new
from experiments.sequential_complementary_native_v1 import contracts as c, run


@pytest.fixture
def budget_module(monkeypatch):
    path = c.ROOT / "experiments/reasoning_posttraining_v1/budget.py"
    assert c.git_blob(path.read_bytes().replace(b"\r\n", b"\n")) == c.INHERITED["reasoning_posttraining_v1/budget.py"]
    preflight = ModuleType("experiments.reasoning_posttraining_v1.preflight")
    preflight.LIMITS = {}  # unused original-RRP default; NOT the SCP ceiling
    spec = importlib.util.spec_from_file_location("experiments.reasoning_posttraining_v1._scp_test_budget", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, preflight.__name__, preflight)
        spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "experiments.reasoning_posttraining_v1.budget", module)
    return module


def make_journal(tmp_path):
    binding = {"journal_path": str(tmp_path / "journal"), "lease_path": str(tmp_path / "gpu.lock"), "fixture": True}
    grant = c.grant_request(binding)
    grant.update(decision="APPROVED", inference_qualification_permission=True,
                 approved_by="CPU_FIXTURE_NOT_APPROVAL", authorization_reference="pytest")
    c.validate_approval(grant, binding)
    return binding, grant, run.journal_for(binding, grant, create=True)


def test_real_journal_preserves_active_charge_across_reopen(tmp_path, budget_module):
    binding, grant, journal = make_journal(tmp_path)
    journal.start("student-001", "student", 30)
    journal.reserve({"evaluation_and_qualification_model_calls": 1})
    reopened = run.journal_for(binding, grant)
    assert reopened.read()["active_reservations"] == ["student-001"]
    assert reopened.used["evaluation_and_qualification_model_calls"] == 1
    with pytest.raises(RuntimeError, match="unsettled"):
        reopened.start("mr-001", "mr", 30)
    with pytest.raises(ValueError): reopened.reserve({"processed_training_tokens_all_attempts": True})
    with pytest.raises(RuntimeError): reopened.reserve({"processed_training_tokens_all_attempts": 1})


def test_journal_no_retry_no_reuse_and_no_training_phase(tmp_path, budget_module):
    binding, grant, journal = make_journal(tmp_path)
    with pytest.raises(ValueError): journal.start("train-001", "train", 30)
    journal.start("student-001", "student", 30)
    terminal = {"cleanup": None, "returncode": 2, "child_pid": 123, "wall_seconds": 1.0}
    journal.settle("student-001", terminal, "a" * 64, 1.1)
    with pytest.raises(ValueError, match="reused"): journal.start("student-001", "student", 30)
    with pytest.raises(RuntimeError): journal.start("student-002", "student", 30)
    assert not journal.read()["budget_breached"]


@pytest.mark.parametrize("terminal", [
    {"cleanup": {"child_exited": False}, "returncode": None, "child_pid": 12, "wall_seconds": 1},
    {"cleanup": None, "returncode": None, "child_pid": 12, "wall_seconds": 1},
])
def test_unobserved_exit_cannot_release_reservation(tmp_path, budget_module, terminal):
    _, _, journal = make_journal(tmp_path)
    journal.start("student-001", "student", 30)
    with pytest.raises(RuntimeError): journal.settle("student-001", terminal, "a" * 64, 1)
    assert journal.read()["active_reservations"] == ["student-001"]


def test_successor_cannot_overwrite_existing_registration(tmp_path, budget_module):
    make_journal(tmp_path)
    with pytest.raises(ValueError):
        budget_module.register(c.EXPERIMENT, ceiling={**c.CEILING, "retries": 10}, phases=c.PHASES, excludes=c.EXCLUDES)


def test_budget_chain_edit_and_grant_switch_are_detected(tmp_path, budget_module):
    binding, grant, journal = make_journal(tmp_path)
    journal.start("student-001", "student", 30)
    other = copy.deepcopy(grant); other["approved_by"] = "different"
    with pytest.raises(ValueError): run.journal_for(binding, other)
    event = journal.directory / "events/000000.json"
    event.write_text(event.read_text().replace('"model_worker_launches":1', '"model_worker_launches":0'))
    with pytest.raises(ValueError): journal.read()


def student_artifact(journal, binding, *, mutate=lambda x: None, terminal_status="EXITED"):
    journal.start("student-001", "student", 30)
    directory = journal.directory / "runs/student-001"; directory.mkdir()
    launch = {"fixture": "not a native launch"}
    state = {"parameters": "fixture-parameters", "buffers": "fixture-buffers"}
    value = {"schema": "SCP_NATIVE_RESULT_V1", "experiment": c.EXPERIMENT, "phase": "student",
             "status": "PASS_QUALIFICATION_ONLY", "evidence_class": "NATIVE_QUALIFICATION_ONLY",
             "binding_sha256": digest(binding), "launch_sha256": digest(launch), "worker_pid": 123,
             "loading": {"subject_id": c.SUBJECT, "subject_weight_sha256": c.SUBJECT_SHA, "subject_state": state},
             "checks": {"zero_update": True, "backward_calls": 0, "optimizer_steps": 0,
                        "currently_trainable_tensors": 0, "qualified_sequence_tokens": 2048,
                        "state_before": state, "state_after": state},
             "runtime": {"fixture": True}, **{k: False for k in c.CLOSED}}
    mutate(value)
    write_new(directory / "RESULT.json", seal(value))
    write_new(directory / "launch.json", launch)
    write_new(directory / "worker_claim.json", {"pid": 123, "launch_sha256": digest(launch)})
    terminal = {"status": terminal_status, "returncode": 0, "cleanup": None, "child_pid": 123, "wall_seconds": 1}
    write_new(directory / "supervisor.json", terminal)
    write_new(directory / "TERMINAL_BINDING.json", {"supervisor_sha256": file_hash(directory / "supervisor.json"),
                                                   "result_sha256": file_hash(directory / "RESULT.json")})
    journal.settle("student-001", terminal, file_hash(directory / "TERMINAL_BINDING.json"), 1)
    return directory


def test_student_receipt_requires_real_files_and_exact_chain(tmp_path, budget_module):
    binding, _, journal = make_journal(tmp_path)
    with pytest.raises(ValueError): run.student_receipt(journal, "a" * 64, binding)
    directory = student_artifact(journal, binding)
    value = run.student_receipt(journal, "student-001", binding, {"fixture": True})
    assert value["training_permission"] is False
    (directory / "RESULT.json").write_text((directory / "RESULT.json").read_text() + " ")
    with pytest.raises(ValueError, match="bytes changed"):
        run.student_receipt(journal, "student-001", binding)


@pytest.mark.parametrize("mutation", ["fixture", "subject", "binding", "gradient", "shape", "training", "state", "runtime", "terminal"])
def test_mr_cannot_borrow_stale_or_wrong_student_qualification(tmp_path, budget_module, mutation):
    binding, _, journal = make_journal(tmp_path)
    def mutate(value):
        if mutation == "fixture": value["evidence_class"] = "SOFTWARE_FIXTURE_ONLY"
        if mutation == "subject": value["loading"]["subject_weight_sha256"] = c.SOURCE_SHA
        if mutation == "binding": value["binding_sha256"] = "a" * 64
        if mutation == "gradient": value["checks"]["backward_calls"] = 1
        if mutation == "shape": value["checks"]["qualified_sequence_tokens"] = 64
        if mutation == "training": value["training_permission"] = True
        if mutation == "state": value["checks"]["state_after"] = {"parameters": "changed"}
    student_artifact(journal, binding, mutate=mutate, terminal_status="START_FAILED" if mutation == "terminal" else "EXITED")
    with pytest.raises(ValueError):
        run.student_receipt(journal, "student-001", binding, {"fixture": False} if mutation == "runtime" else None)


def test_actual_supervisor_claim_and_terminal_receipt(tmp_path):
    out = tmp_path / "child"
    script = ("from pathlib import Path; import sys; "
              "from experiments.model_quality_gpu_v1.supervisor import claim_worker; "
              "assert 'torch' not in sys.modules; claim_worker(Path(sys.argv[1])); "
              "assert 'torch' not in sys.modules")
    assert supervisor.bounded_child([sys.executable, "-c", script, str(out)], out, 10) == 0
    assert read_json(out / "supervisor.json")["status"] == "EXITED"
    assert read_json(out / "worker_claim.json")["launch_sha256"] == digest(read_json(out / "launch.json"))
    with pytest.raises(FileExistsError): supervisor.bounded_child([sys.executable, "-c", "pass"], out, 10)


def test_actual_supervisor_timeout_retains_owned_child_cleanup(tmp_path):
    out = tmp_path / "timeout"
    assert supervisor.bounded_child([sys.executable, "-c", "import time; time.sleep(5)"], out, 1) == 2
    value = read_json(out / "supervisor.json")
    assert value["status"] == "TIMEOUT_OR_INTERRUPTED" and value["cleanup"]["child_exited"]
    assert (out / "worker.log").is_file()


def test_direct_worker_refuses_before_runtime_or_output(tmp_path):
    out = tmp_path / "not-created"
    script = ("from pathlib import Path; from types import SimpleNamespace; import sys; "
              "from experiments.sequential_complementary_native_v1.run import worker; "
              "\ntry: worker(SimpleNamespace(out=Path(sys.argv[1])))"
              "\nexcept ValueError as e: assert 'supervisor context absent' in str(e)"
              "\nelse: raise AssertionError('unsupervised worker accepted')"
              "\nassert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", script, str(out)], cwd=c.ROOT, check=True, timeout=10)
    assert not out.exists()


def test_legacy_native_flag_does_not_import_runtime_unsupervised():
    script = ("import sys; from experiments.sequential_complementary_v1 import reasoning; "
              "from experiments.sequential_complementary_v1.data import PublicCase; "
              "\ntry: reasoning.run_case(PublicCase('a','w','s','u'),'MR',object(),None,lambda x:None,native=True)"
              "\nexcept ValueError: pass"
              "\nelse: raise AssertionError('native accepted')"
              "\nassert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", script], cwd=c.ROOT, check=True, timeout=10)


def test_shared_lease_never_deletes_another_owners_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    with resources.lease(path):
        value = read_json(path)
        with pytest.raises(FileExistsError):
            with resources.lease(path): pass
        assert read_json(path) == value
    assert not path.exists()


@pytest.mark.parametrize("host,gpu,phase", [(4.5, 8, "qualify"), (8, 2.9, "qualify"), (5, 8, "unknown"), (8, None, "qualify")])
def test_inherited_preload_minima_are_not_bypassed(monkeypatch, host, gpu, phase):
    monkeypatch.setattr(resources, "snapshot", lambda: {"available_host_bytes": int(host * 2**30),
                    "gpu0": None if gpu is None else {"free": int(gpu * 2**30)}})
    with pytest.raises(RuntimeError): resources.guard(before_load=True, phase=phase)
    assert resources.PRELOAD_MINIMA == {"qualify": (4.6, 3.0), "train": (6.0, 6.0)}
