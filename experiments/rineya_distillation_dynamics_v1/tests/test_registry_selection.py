from dataclasses import asdict, replace
from pathlib import Path

import pytest

from experiments.rineya_distillation_dynamics_v1.contracts import (
    ContractError, digest, file_hash, matrix, read_json, sealed, unseal, write_new)
from experiments.rineya_distillation_dynamics_v1.fixtures import fixture_plan
from experiments.rineya_distillation_dynamics_v1.host import launch
from experiments.rineya_distillation_dynamics_v1.registry import check_e0, inventory, register
from experiments.rineya_distillation_dynamics_v1.selection import select, spend_confirmation


def fake_runs(root, plan, *, damage=0, gain=.1):
    """Deliberately fabricated score metadata for state-machine tests, NOT model evidence."""
    paths = []
    for arm in matrix(plan, "screen"):
        directory = root / arm.name
        ckpt = directory / "checkpoints" / "000001"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter.safetensors").write_bytes(b"STATE_MACHINE_FIXTURE_NOT_WEIGHTS")
        receipt = {"adapter_sha256": file_hash(ckpt / "adapter.safetensors"), "step": 1}
        write_new(ckpt / "receipt.json", sealed(receipt))
        group = lambda g, d: {"gain": g, "n": 10, "damaged": d, "correct": 7}
        result = {"status": "COMPLETE", "plan": plan.fingerprint, "identity": "fixture-id", "arm": asdict(arm),
                  "initial_adapter": str(arm.seed), "frozen": {"teacher": "fixture", "student": "fixture"},
                  "evidence_kind": "mechanical_fixture", "checkpoints": [{"step": 1, "receipt": receipt,
                      "meter": {}, "summary": {"groups": {"dev_acquisition/s": group(gain, 0),
                          "dev_retention/s": group(-damage/10, damage), "dev_hard/s": group(0, 0)}}}]}
        path = directory / "RESULT.json"
        write_new(path, sealed(result))
        paths.append(path)
    return paths


def test_seed_complete_selection_and_spent_confirmation(tmp_path):
    plan = replace(fixture_plan(), seeds=(17, 29), steps=1)
    paths = fake_runs(tmp_path / "runs", plan)
    out = tmp_path / "selection"
    chosen = select(plan, paths, out, stage="screen", identity="fixture-id")
    assert chosen["status"] == "DEVELOPMENT_SELECTED"
    assert set(chosen["selected_seeds"]) == {"17", "29"}
    assert chosen["selected_configuration"] == ["A-lr0", 1]
    spend_confirmation(out / "SELECTION.json", "fixture-id", "f"*64)
    with pytest.raises(FileExistsError): spend_confirmation(out / "SELECTION.json", "fixture-id", "f"*64)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unpaired", "mixed_evidence", "tampered_weights", "changed_identity"])
def test_selection_rejects_invalid_campaign(tmp_path, kind):
    plan = replace(fixture_plan(), steps=1)
    paths = fake_runs(tmp_path / "runs", plan)
    if kind == "missing": paths.pop()
    if kind == "duplicate": paths[-1] = paths[0]
    if kind in {"unpaired", "mixed_evidence", "changed_identity"}:
        record = unseal(read_json(paths[-1]))
        field = {"unpaired": "initial_adapter", "mixed_evidence": "evidence_kind", "changed_identity": "identity"}[kind]
        record[field] = "different"
        paths[-1].write_text(__import__('json').dumps(sealed(record)))
    if kind == "tampered_weights":
        (paths[-1].parent / "checkpoints" / "000001" / "adapter.safetensors").write_bytes(b"changed")
    with pytest.raises(ContractError): select(plan, paths, tmp_path / "s", stage="screen", identity="fixture-id")


@pytest.mark.parametrize("gain,damage", [(0, 0), (.1, 1)])
def test_no_gain_or_harm_cannot_win(tmp_path, gain, damage):
    plan = replace(fixture_plan(), steps=1)
    paths = fake_runs(tmp_path / "runs", plan, gain=gain, damage=damage)
    chosen = select(plan, paths, tmp_path / "selection", stage="screen", identity="fixture-id")
    assert chosen["status"] == "NO_ELIGIBLE_CONFIGURATION"
    with pytest.raises(ContractError): spend_confirmation(tmp_path / "selection" / "SELECTION.json", "fixture-id", "a"*64)


def test_permission_gate_before_any_model_or_registration_access(tmp_path):
    with pytest.raises(ContractError, match="model-load"):
        launch("absent", tmp_path / "out", arm="A-lr0-s17")
    assert not (tmp_path / "out").exists()


def test_training_permission_separate(tmp_path):
    with pytest.raises(ContractError, match="training includes"):
        launch("absent", tmp_path / "out", arm="A-lr0-s17", allow_model_load=True, ack_host_available=True)


def test_confirm_permission_separate(tmp_path):
    with pytest.raises(ContractError, match="confirmation permission"):
        launch("absent", tmp_path / "out", selection="absent", allow_model_load=True, ack_host_available=True)


def test_existing_e0_gate(tmp_path):
    config = tmp_path / "cfg.json"
    write_new(config, {"fixture": True})
    result = tmp_path / "result.json"
    keys = ("exact_native_load", "save_reload_resume_bitwise", "frozen_base_unchanged", "finite_scoped_backward", "eos_pad_mask")
    write_new(result, {"status": "PASS_E0_SCOPED_NO_EFFICACY_CLAIM", "config_sha256": file_hash(config),
                       "checks": {k: True for k in keys}})
    check_e0(config, result)
    config.write_text("{}")
    with pytest.raises(ContractError): check_e0(config, result)


@pytest.mark.parametrize("kind", ["missing", "pickle", "custom", "quantized", "wrong_arch", "missing_shard"])
def test_model_inventory_fails_closed(tmp_path, kind):
    root = tmp_path / "model"
    root.mkdir()
    cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    if kind == "custom": cfg["auto_map"] = {"AutoModel": "remote.Model"}
    if kind == "quantized": cfg["quantization_config"] = {"bits": 4}
    if kind == "wrong_arch": cfg["architectures"] = ["OtherModel"]
    write_new(root / "config.json", cfg)
    write_new(root / "tokenizer.json", {})
    write_new(root / "tokenizer_config.json", {})
    if kind != "missing": (root / "model.safetensors").write_bytes(b"INVENTORY_FIXTURE")
    if kind == "pickle": (root / "pytorch_model.bin").write_bytes(b"not loaded")
    if kind == "missing_shard": write_new(root / "model.safetensors.index.json", {"weight_map": {"w": "../outside.safetensors"}})
    with pytest.raises(ContractError): inventory(root)


def test_model_inventory_hashes_without_loading_weights(tmp_path):
    write_new(tmp_path / "config.json", {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    write_new(tmp_path / "tokenizer.json", {})
    write_new(tmp_path / "tokenizer_config.json", {})
    (tmp_path / "model.safetensors").write_bytes(b"INVENTORY_FIXTURE_NOT_REAL_WEIGHTS")
    before = inventory(tmp_path)["sha256"]
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    assert inventory(tmp_path)["sha256"] != before


@pytest.mark.parametrize("field", ["teacher_min_accuracy", "teacher_min_advantage", "min_acquisition_gain"])
def test_native_registration_rejects_zero_evidence_thresholds(tmp_path, field):
    plan = replace(fixture_plan(), **{field: 0})
    path = tmp_path / "plan.json"
    write_new(path, asdict(plan))
    with pytest.raises(ContractError, match="positive native"):
        register(path, "nonexistent-binding", tmp_path / "registration.json")
    assert not (tmp_path / "registration.json").exists()
