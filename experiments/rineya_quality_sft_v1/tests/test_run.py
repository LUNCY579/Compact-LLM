import copy
from pathlib import Path
from types import SimpleNamespace
import pytest

from experiments.rineya_quality_sft_v1 import run, supervisor
from experiments.rineya_quality_sft_v1.integrity import digest, read_json, external_caches, seal, write_new
from experiments.rineya_quality_sft_v1.reuse import REQUEST_FIELDS, reuse_prediction


def test_native_flags_refuse_before_read_load_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "read_json", lambda *a: pytest.fail("No file read before opt-in"))
    args = SimpleNamespace(command="train", execute_training=False, output=str(tmp_path / "out"))
    with pytest.raises(PermissionError): run.public_native(args)
    assert not (tmp_path / "out").exists()


def test_unowned_worker_refuses_before_native_import(tmp_path, monkeypatch):
    monkeypatch.delenv(supervisor.CONTEXT_ENV, raising=False)
    with pytest.raises(PermissionError): supervisor.claim_worker(tmp_path)


def test_exposure_review_cannot_import_foreign_or_pending_grant():
    plan = {"sha256": "a" * 64}
    review = run.exposure_review_template(plan)
    with pytest.raises(PermissionError): run.validate_exposure_review(review, plan)
    review.update(decision="APPROVED", reviewer="independent fixture reviewer", rationale="fixture only")
    for key, value in list(review.items()):
        if type(value) is bool: review[key] = True
    run.validate_exposure_review(review, plan)
    review["experiment"] = "MQ-001"
    with pytest.raises(PermissionError): run.validate_exposure_review(review, plan)


def test_external_cache_overrides_and_tempfile_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "outside-old-value")
    with external_caches(tmp_path):
        import os, tempfile
        assert Path(os.environ["HF_HUB_CACHE"]).is_relative_to(tmp_path)
        assert Path(tempfile.gettempdir()).is_relative_to(tmp_path)
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    import os
    assert os.environ["HF_HUB_CACHE"] == "outside-old-value"


def test_work_guard_caps_are_phase_specific_and_charge_attempts(tmp_path):
    guard = run.WorkGuard("train", tmp_path, 1, snapshot=lambda stage: {"fixture_only": True})
    guard("forward_before", input_tokens=25000, forward_calls=1)
    with pytest.raises(RuntimeError, match="ceiling"):
        guard("forward_before", input_tokens=1, forward_calls=1)
    assert guard.counts["processed_input_tokens"] == 25001
    assert read_json(tmp_path / "resources" / "00001.json")["counts"]["processed_input_tokens"] == 25000
    evaluation = run.WorkGuard("baseline", tmp_path, 1, snapshot=lambda stage: {"fixture_only": True})
    evaluation.calls = 25000  # Previous training sample numbering, not this phase's cap.
    evaluation.phase_calls = 98303
    evaluation("forward_after")
    assert evaluation.phase_calls == 98304 and evaluation.calls == 25001


def test_all_exact_reuse_bindings_required_and_origin_separate():
    request = {key: [1, 2] if key.endswith("ids") else "fixed-fixture" for key in REQUEST_FIELDS}
    old = {"material_role": "OPEN_DEV", "ancestry_roles": [], "completed": True, "source_unchanged": True,
           "request": copy.deepcopy(request), "prediction": {"text": "fixture", "correct": True}, "artifact_sha256": "a" * 64}
    result = reuse_prediction(old, request)
    assert result["new_model_calls"] == 0 and not result["qualification_transferred"]
    assert "correct" not in result["prediction"] and not result["judgments_reused"]
    changed = copy.deepcopy(request); changed["runtime"] = "changed"
    with pytest.raises(ValueError): reuse_prediction(old, changed)
    old["ancestry_roles"] = ["SEALED"]
    with pytest.raises(PermissionError): reuse_prediction(old, request)


def test_foreground_supervisor_preserves_stdout_and_terminal_receipt(tmp_path):
    import sys
    out = tmp_path / "supervised-toy"
    assert supervisor.bounded_child([sys.executable, "-B", "-c", "print('CPU_TOY_ONLY')"], out, 20) == 0
    assert "CPU_TOY_ONLY" in (out / "worker.log").read_text()
    assert read_json(out / "supervisor.json")["status"] == "EXITED"
    with pytest.raises(FileExistsError):
        supervisor.bounded_child([sys.executable, "-B", "-c", "print('not-run')"], out, 20)


def test_failed_outer_stage_or_changed_inner_artifact_cannot_advance(tmp_path):
    path = tmp_path / "predictions.json"
    write_new(path, {"fixture": "no model"})
    identity = {"fixture": True}
    stage = {"experiment": "RQSFT-001", "phase": "baseline", "identity": identity,
        "status": "NATIVE_STAGE_COMPLETED_QUALITY_UNDECIDED", "source_unchanged": True,
        "inputs_unchanged": True, "native_model_loaded": True,
        "artifact_hashes": {"predictions.json": run.file_hash(path)}}
    write_new(tmp_path / "stage.json", seal(stage))
    write_new(tmp_path / "supervisor.json", {"status": "EXITED", "returncode": 0})
    assert run.completed_stage(path, identity, "baseline")["phase"] == "baseline"
    path.write_text('{"changed":true}')
    with pytest.raises(PermissionError, match="bytes differ"):
        run.completed_stage(path, identity, "baseline")


def test_campaign_failure_consumes_stage_slot_without_refill(tmp_path):
    from experiments.rineya_quality_sft_v1.campaign import claim, settle
    reservation = claim("a" * 64, "b" * 64, "train", "whole_1e-5", tmp_path / "attempt", root=tmp_path / "campaign")
    settle(reservation, 2, tmp_path / "attempt")
    with pytest.raises(FileExistsError):
        claim("a" * 64, "b" * 64, "train", "whole_1e-5", tmp_path / "retry", root=tmp_path / "campaign")
    with pytest.raises(ValueError, match="digests"):
        claim("../source", "b" * 64, "train", "whole_1e-5", tmp_path / "bad", root=tmp_path / "campaign")


def test_owner_template_is_not_admission_before_payload_read(tmp_path):
    from experiments.rineya_quality_sft_v1.bank import owner_request_template, read_owner_bundle
    manifest = tmp_path / "owner-request.json"
    write_new(manifest, owner_request_template())
    with pytest.raises(ValueError, match="request/template"):
        read_owner_bundle(manifest)
