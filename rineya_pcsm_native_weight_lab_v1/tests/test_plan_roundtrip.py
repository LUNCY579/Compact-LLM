from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rineya_pcsm.hashing import mask_hash
from rineya_pcsm.persistence import (
    canonical_mask_hash,
    fingerprint_state_dict,
    load_plan,
    save_plan,
)
from rineya_pcsm.plan import PCSMPlan, make_tensor_plan


def _example_plan() -> PCSMPlan:
    base = fingerprint_state_dict(
        {"layer.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    )
    entry = make_tensor_plan(
        name="layers.1.mlp.down_proj.weight",
        role="mlp.down_proj",
        scores=torch.tensor([[0.0, 5.0], [2.0, 1.0]]),
        stabilizer=torch.tensor([[0.25, 0.5], [0.75, 1.0]]),
        k=2,
        peer_names=("layers.0.mlp.down_proj.weight", "layers.2.mlp.down_proj.weight"),
        attention_alphas=torch.tensor([0.6, 0.4]),
        module_type="Linear",
        layer_depth=1,
        support_fraction=0.5,
        feature_names=("task_gradient_magnitude", "peer_joint_leverage"),
        feature_weights=torch.tensor([0.7, 0.3]),
        generator_config={"version": "pcsm-generator-v1", "hidden_dimension": 64},
    )
    return PCSMPlan(
        base_fingerprint=base,
        descriptor_dimension=24,
        entries=(entry,),
        metadata={"seed": 17, "split": "mask_cal"},
    )


def test_plan_roundtrip_preserves_masks_a_alphas_and_counts(tmp_path: Path) -> None:
    path = save_plan(_example_plan(), tmp_path / "frozen-plan")
    plan = _example_plan()
    loaded = load_plan(path, expected_base_fingerprint=plan.base_fingerprint)
    assert loaded.base_fingerprint == plan.base_fingerprint
    assert loaded.support_stats() == plan.support_stats()
    assert torch.equal(loaded.entries[0].support_mask, plan.entries[0].support_mask)
    assert torch.equal(loaded.entries[0].flat_indices, plan.entries[0].flat_indices)
    assert torch.equal(loaded.entries[0].stabilizer, plan.entries[0].stabilizer)
    assert torch.equal(
        loaded.entries[0].attention_alphas, plan.entries[0].attention_alphas
    )
    assert loaded.entries[0].module_type == "Linear"
    assert loaded.entries[0].layer_depth == 1
    assert loaded.entries[0].support_fraction == 0.5
    assert loaded.entries[0].feature_names == plan.entries[0].feature_names
    assert torch.equal(
        loaded.entries[0].feature_weights, plan.entries[0].feature_weights
    )
    assert dict(loaded.entries[0].generator_config) == dict(
        plan.entries[0].generator_config
    )
    manifest = json.loads((path / "plan.json").read_text("utf-8"))
    record = manifest["metadata"]["target_plan_records"][plan.entries[0].name]
    assert record["module_type"] == "Linear"
    assert record["layer_depth"] == 1
    assert record["exact_k"] == 2
    assert record["requested_fraction"] == 0.5
    assert record["feature_weights"] == pytest.approx([0.7, 0.3])
    assert record["generator_config"]["version"] == "pcsm-generator-v1"
    assert canonical_mask_hash(loaded.entries[0].support_mask) == canonical_mask_hash(
        plan.entries[0].support_mask
    )
    assert canonical_mask_hash(plan.entries[0].support_mask) == mask_hash(
        plan.entries[0].support_mask
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        save_plan(plan, path)


def test_plan_load_fails_closed_on_base_mismatch(tmp_path: Path) -> None:
    plan = _example_plan()
    path = save_plan(plan, tmp_path / "frozen-plan")
    with pytest.raises(ValueError, match="base fingerprint mismatch"):
        load_plan(path, expected_base_fingerprint="0" * 64)


def test_plan_load_rejects_tampered_canonical_mask_hash(tmp_path: Path) -> None:
    plan = _example_plan()
    path = save_plan(plan, tmp_path / "frozen-plan")
    manifest_path = path / "plan.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["entries"][0]["mask_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), "utf-8"
    )
    with pytest.raises(ValueError, match="canonical mask hash mismatch"):
        load_plan(path, expected_base_fingerprint=plan.base_fingerprint)


def test_base_fingerprint_changes_with_one_weight_bit() -> None:
    first = torch.zeros(2, 2)
    second = first.clone()
    second.view(torch.int32)[0, 0] = 1
    assert fingerprint_state_dict({"w": first}) != fingerprint_state_dict({"w": second})


def test_base_fingerprint_supports_scalar_state_entries() -> None:
    fingerprint = fingerprint_state_dict({"step": torch.tensor(1, dtype=torch.int64)})
    assert len(fingerprint) == 64


def test_tensor_plan_rejects_fraction_or_feature_weight_mismatch() -> None:
    common = {
        "name": "layers.1.mlp.down_proj.weight",
        "role": "mlp.down_proj",
        "scores": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "stabilizer": torch.ones(2, 2),
        "k": 2,
        "peer_names": ("peer",),
        "attention_alphas": torch.ones(1),
    }
    with pytest.raises(ValueError, match="support fraction"):
        make_tensor_plan(**common, support_fraction=0.25)
    with pytest.raises(ValueError, match="every feature"):
        make_tensor_plan(
            **common,
            feature_names=("gradient", "peer"),
            feature_weights=torch.ones(1),
        )
