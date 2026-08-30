from __future__ import annotations

import torch

from safetensors import safe_open

from rineya_cross_scale.checkpoint import CheckpointError
from rineya_cross_scale.projection import (
    ProjectionPlan,
    build_projection_plan,
    donor_key_for_target,
    project_tensor,
    project_checkpoint,
)
from rineya_cross_scale.hashing import sha256_file

from helpers import tiny_config, tiny_tensors, write_checkpoint


def configs() -> tuple[dict, dict]:
    target = {
        "model_type": "qwen3",
        "vocab_size": 16,
        "head_dim": 2,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 3,
    }
    donor = {
        "model_type": "qwen3",
        "vocab_size": 16,
        "head_dim": 2,
        "hidden_size": 8,
        "intermediate_size": 10,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 5,
    }
    return target, donor


def plan() -> ProjectionPlan:
    target, donor = configs()
    return build_projection_plan(target, donor, depth_strategy="D2", width_strategy="W2")


def test_plan_and_layer_mapping() -> None:
    value = plan()
    assert value.layer_mapping == (0, 2, 4)
    assert donor_key_for_target("model.layers.1.self_attn.q_proj.weight", value) == (
        "model.layers.2.self_attn.q_proj.weight"
    )
    assert donor_key_for_target("model.embed_tokens.weight", value) == (
        "model.embed_tokens.weight"
    )


def test_attention_projection_preserves_complete_heads() -> None:
    value = plan()
    donor = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    projected = project_tensor(
        "model.layers.0.self_attn.q_proj.weight", donor, (4, 4), value
    )
    assert torch.equal(projected, donor[:4, :4])
    output = project_tensor(
        "model.layers.0.self_attn.o_proj.weight", donor, (4, 4), value
    )
    assert torch.equal(output, donor[:4, :4])


def test_kv_and_mlp_projection_shapes() -> None:
    value = plan()
    kv = project_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        torch.zeros(4, 8),
        (2, 4),
        value,
    )
    gate = project_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        torch.zeros(10, 8),
        (6, 4),
        value,
    )
    down = project_tensor(
        "model.layers.0.mlp.down_proj.weight",
        torch.zeros(8, 10),
        (4, 6),
        value,
    )
    assert kv.shape == (2, 4)
    assert gate.shape == (6, 4)
    assert down.shape == (4, 6)


def test_reject_incompatible_head_dimension() -> None:
    target, donor = configs()
    donor["head_dim"] = 4
    try:
        build_projection_plan(target, donor, depth_strategy="D2", width_strategy="W2")
    except CheckpointError as error:
        assert "head dimension mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("incompatible head dimensions were accepted")


def test_untied_donor_output_head_policy() -> None:
    target, donor = configs()
    target["tie_word_embeddings"] = True
    donor["tie_word_embeddings"] = False
    value = build_projection_plan(
        target,
        donor,
        depth_strategy="D2",
        width_strategy="W2",
        tied_projection="output_head",
    )
    assert donor_key_for_target("model.embed_tokens.weight", value) == "lm_head.weight"
    assert donor_key_for_target("lm_head.weight", value) == "lm_head.weight"


def test_untied_donor_embedding_policy_ties_both_target_keys() -> None:
    target, donor = configs()
    target["tie_word_embeddings"] = True
    donor["tie_word_embeddings"] = False
    value = build_projection_plan(
        target,
        donor,
        depth_strategy="D2",
        width_strategy="W2",
        tied_projection="embedding",
    )
    assert donor_key_for_target("model.embed_tokens.weight", value) == (
        "model.embed_tokens.weight"
    )
    assert donor_key_for_target("lm_head.weight", value) == (
        "model.embed_tokens.weight"
    )


def test_project_checkpoint_is_shape_exact_and_deterministic(tmp_path) -> None:
    target_config = tiny_config(hidden=4, mlp=6, q_heads=2, kv_heads=1, layers=3)
    donor_config = tiny_config(hidden=8, mlp=10, q_heads=4, kv_heads=2, layers=5)
    target = write_checkpoint(
        tmp_path / "target", target_config, tiny_tensors(target_config)
    )
    donor_tensors = tiny_tensors(donor_config, offset=1000.0)
    donor = write_checkpoint(tmp_path / "donor", donor_config, donor_tensors)
    first = tmp_path / "projected-first"
    second = tmp_path / "projected-second"
    for output in (first, second):
        manifest = project_checkpoint(
            template_path=target,
            donor_path=donor,
            destination=output,
            depth_strategy="D2",
            width_strategy="W2",
            output_dtype="float32",
            max_shard_bytes=1024 * 1024,
        )
        assert manifest["plan"]["layer_mapping"] == (0, 2, 4)
    assert sha256_file(first / "model.safetensors") == sha256_file(
        second / "model.safetensors"
    )
    with safe_open(str(first / "model.safetensors"), framework="pt") as handle:
        assert set(handle.keys()) == set(tiny_tensors(target_config))
        projected = handle.get_tensor("model.layers.1.self_attn.q_proj.weight")
    expected = donor_tensors["model.layers.2.self_attn.q_proj.weight"][:4, :4]
    assert torch.equal(projected, expected)
