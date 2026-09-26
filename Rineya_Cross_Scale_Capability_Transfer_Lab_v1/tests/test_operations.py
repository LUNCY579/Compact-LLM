from __future__ import annotations

import json

import torch
from safetensors import safe_open

from helpers import tiny_config, tiny_tensors, write_checkpoint
from rineya_cross_scale.operations import (
    average_checkpoints,
    build_delta,
    merge_absolute,
    merge_task_vector,
)


def tensor(path, key):
    with safe_open(str(path / "model.safetensors"), framework="pt") as handle:
        return handle.get_tensor(key)


def test_delta_and_task_vector_round_trip(tmp_path) -> None:
    config = tiny_config(hidden=4, mlp=6, q_heads=2, kv_heads=1, layers=3)
    reference_values = tiny_tensors(config, offset=0.0)
    projected_values = tiny_tensors(config, offset=2.0)
    base_values = tiny_tensors(config, offset=10.0)
    reference = write_checkpoint(tmp_path / "reference", config, reference_values)
    projected = write_checkpoint(tmp_path / "projected", config, projected_values)
    base = write_checkpoint(
        tmp_path / "base", config, base_values, abstain_bias=5.15625
    )
    delta = tmp_path / "delta"
    build_delta(
        projected_path=projected,
        reference_path=reference,
        destination=delta,
        max_shard_bytes=1024 * 1024,
    )
    key = "model.layers.1.self_attn.q_proj.weight"
    assert torch.equal(tensor(delta, key), torch.full_like(reference_values[key], 2.0))

    candidate = tmp_path / "candidate"
    merge_task_vector(
        base_path=base,
        delta_path=delta,
        destination=candidate,
        alpha=0.25,
        output_dtype="float32",
        max_shard_bytes=1024 * 1024,
        overwrite=False,
    )
    assert torch.equal(tensor(candidate, key), base_values[key] + 0.5)
    candidate_config = json.loads((candidate / "config.json").read_text())
    assert candidate_config["abstain_logit_bias"] == 0.0
    manifest = json.loads((candidate / "candidate_manifest.json").read_text())
    assert manifest["storage_summary"]["stored_nonzero_coordinates"] > 0


def test_absolute_interpolation(tmp_path) -> None:
    config = tiny_config(hidden=4, mlp=6, q_heads=2, kv_heads=1, layers=3)
    base_values = tiny_tensors(config, offset=0.0)
    projected_values = tiny_tensors(config, offset=4.0)
    base = write_checkpoint(tmp_path / "base", config, base_values)
    projected = write_checkpoint(tmp_path / "projected", config, projected_values)
    candidate = tmp_path / "candidate"
    merge_absolute(
        base_path=base,
        projected_path=projected,
        destination=candidate,
        alpha=0.125,
        output_dtype="float32",
        max_shard_bytes=1024 * 1024,
        overwrite=False,
    )
    key = "model.layers.0.mlp.down_proj.weight"
    assert torch.equal(tensor(candidate, key), base_values[key] + 0.5)


def test_average_checkpoints(tmp_path) -> None:
    config = tiny_config(hidden=4, mlp=6, q_heads=2, kv_heads=1, layers=2)
    first_values = tiny_tensors(config, offset=0.0)
    second_values = tiny_tensors(config, offset=4.0)
    first = write_checkpoint(tmp_path / "first", config, first_values)
    second = write_checkpoint(tmp_path / "second", config, second_values)
    averaged = tmp_path / "averaged"
    average_checkpoints(
        source_paths=[first, second],
        destination=averaged,
        max_shard_bytes=1024 * 1024,
    )
    key = "model.layers.0.self_attn.q_proj.weight"
    assert torch.allclose(
        tensor(averaged, key), (first_values[key] + second_values[key]) / 2
    )
