from datetime import datetime, timezone, timedelta
import json
import pytest
import torch

from experiments.rineya_length_generalization_regularization_v1.provenance import (
    checkpoint_fingerprint, file_sha256, write_json_new, exclusive_run, verify_receipt, binding)
from experiments.rineya_length_generalization_regularization_v1.config import Plan
from experiments.rineya_length_generalization_regularization_v1.native import native_run, _qualify
from experiments.rineya_length_generalization_regularization_v1.fixtures import TinyLM
from experiments.rineya_length_generalization_regularization_v1.adapters import inject_lora, adapter_state


def receipt(now):
    return {"status": "READY_FOR_EXPLICIT_BOUNDED_TRAINING", "binding": {"key": "value"},
            "created_at": now.isoformat(), "checks": {k: True for k in (
                "identity_parity", "finite_backward", "serialization_parity", "merge_parity")}}


def test_receipt_bound_expiring_and_complete():
    now = datetime.now(timezone.utc)
    good = receipt(now)
    verify_receipt(good, {"key": "value"}, now=now)
    with pytest.raises(ValueError):
        verify_receipt(good, {"key": "changed"}, now=now)
    for delta in (-2, 25):
        with pytest.raises(ValueError):
            verify_receipt(good, {"key": "value"}, now=now + timedelta(hours=delta))
    good["checks"]["finite_backward"] = False
    with pytest.raises(ValueError):
        verify_receipt(good, {"key": "value"}, now=now)


def test_content_hashes_and_eligibility(tmp_path):
    model = tmp_path / "model"; model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "model.safetensors").write_bytes(b'fixture-not-a-checkpoint')
    original = checkpoint_fingerprint(model)
    (model / "config.json").write_text('{"changed":true}')
    assert checkpoint_fingerprint(model) != original
    bank = tmp_path / "bank.jsonl"; bank.write_text('{}\n')
    eligible = tmp_path / "eligible.json"
    payload = {"bank_sha256": file_sha256(bank), "train_eligible": True, "quarantine_excluded": True,
               "cross_split_parent_and_prompt_audit_passed": True}
    write_json_new(eligible, payload)
    result = binding(model, bank, Plan(), eligible, {"fixture": True})
    assert result["bank_sha256"] == file_sha256(bank)
    bank.write_text('{}\n{}\n')
    with pytest.raises(ValueError):
        binding(model, bank, Plan(), eligible, {})


def test_symlinks_no_overwrite_and_spent_runs(tmp_path):
    target = tmp_path / "target"; target.write_text('x')
    alias = tmp_path / "alias"; alias.symlink_to(target)
    with pytest.raises(ValueError):
        file_sha256(alias)
    run = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with exclusive_run(run):
            raise RuntimeError("fixture failure")
    assert (run / "RUN_LOCK.json").exists()
    with pytest.raises(FileExistsError):
        with exclusive_run(run):
            pass


def test_native_denied_before_any_load():
    kwargs = dict(checkpoint="missing", rows=[], bank_path="missing", plan=Plan(), eligibility="missing", output_dir="missing")
    with pytest.raises(PermissionError):
        native_run("qualify", **kwargs)
    with pytest.raises(PermissionError):
        native_run("train", allow_native_load=True, **kwargs)


def test_actual_qualification_logic_on_cpu_fixture():
    model = TinyLM()
    plan = Plan(expected_layers=2, rank=2, lora_alpha=4, dtype="float32", mode="vpad", probability=0.1)
    inject_lora(model, plan)
    state = adapter_state(model)
    row = {"prompt_ids": [3, 5, 7, 9], "target_ids": [4, 2]}
    result = _qualify(model, row, plan, lambda: None)
    assert all(result[k] is True for k in ("identity_parity", "finite_backward", "serialization_parity", "merge_parity"))
    assert result["qualification_optimizer_steps"] == 0
    assert all(torch.equal(v, adapter_state(model)[k]) for k, v in state.items())
