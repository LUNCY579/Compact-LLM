"""Exercise worker control flow with mocks; no models, optimizer or banks opened."""
import sys
from pathlib import Path

from experiments.rineya_distillation_dynamics_v1 import engine, host, native, qualification, registry
from experiments.rineya_distillation_dynamics_v1.contracts import Plan, digest, read_json, sealed, unseal, write_new


def test_mocked_qualification_worker_returns_without_treatment_or_bank_reads(tmp_path, monkeypatch):
    plan = Plan.load(Path(__file__).parents[1] / "configs" / "selected-pilot.json")
    registration = {"evidence_kind": "mechanical_fixture", "description": "control flow only"}
    tasks, backend, calls = [], object(), []
    output = tmp_path / "mocked-worker"
    output.mkdir()
    write_new(output / "SUPERVISOR_STARTED.json", {"mode": "qualification_only", "mechanical_fixture": True})
    monkeypatch.setenv("RDD001_WORKER_TOKEN", "fixture-worker-token")
    monkeypatch.setattr(sys, "argv", ["native.py", "--registration", "not-opened.json",
                                      "--output", str(output), "--token", "fixture-worker-token",
                                      "--qualification-only"])
    monkeypatch.setattr(host, "runtime_check", lambda: calls.append("mocked_runtime"))
    monkeypatch.setattr(registry, "check_registration", lambda _: (registration, plan, tasks))

    def forbidden(*args, **kwargs):
        raise AssertionError("qualification-only worker must not train or open a bank")

    monkeypatch.setattr(engine, "run_arm", forbidden)
    monkeypatch.setattr(native, "read_json", forbidden)

    def mocked_load(bound, selected, catalog, seed, guard, *, evidence_output=None):
        assert bound is registration and selected is plan and catalog is tasks
        assert seed == 17 and evidence_output == output
        calls.append("mocked_load")
        return backend

    def mocked_qualify(loaded, catalog, directory):
        assert loaded is backend and catalog is tasks
        calls.append("mocked_qualify")
        write_new(directory / "RESULT.json", sealed({"status": "MOCK_NO_MODEL_OR_UPDATES",
                                                     "evidence_kind": "mechanical_fixture"}))

    monkeypatch.setattr(native, "load_backend", mocked_load)
    monkeypatch.setattr(qualification, "qualify", mocked_qualify)
    native.main()
    assert calls == ["mocked_runtime", "mocked_load", "mocked_qualify"]
    result = unseal(read_json(output / "NATIVE_RESULT.json"))
    assert result["status"] == "QUALIFICATION_ONLY_NO_TREATMENT"
    assert result["identity"] == digest(registration)
    assert result["treatment_updates"] == 0 and result["confirmation_reads"] == 0
    assert result["deployment_approved"] is False
    assert not (output / "arm").exists() and not (output / "teacher_banks").exists()
