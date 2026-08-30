from __future__ import annotations

import pytest
import torch

from rineya_cross_scale.advanced import (
    dare_delta,
    dare_ties_consensus,
    fisher_weighted_parameters,
    ties_consensus,
    trim_magnitude,
    merge_ties_candidate,
    merge_fisher_candidate,
)
from rineya_cross_scale.interventions import tensor_selected
from rineya_cross_scale.fisher import accumulate_squared_gradients
from helpers import tiny_config, tiny_tensors, write_checkpoint
from safetensors import safe_open


def test_trim_magnitude_exact_count_and_tie_break() -> None:
    value = torch.tensor([1.0, -3.0, 2.0, -3.0])
    trimmed, retained = trim_magnitude(value, 0.5)
    assert retained == 2
    assert torch.equal(trimmed, torch.tensor([0.0, -3.0, 0.0, -3.0]))


def test_ties_elects_sign_and_averages_agreement() -> None:
    first = torch.tensor([2.0, -2.0, 1.0])
    second = torch.tensor([4.0, 1.0, -3.0])
    merged, stats = ties_consensus([first, second], 1.0)
    assert merged[0].item() == 3.0
    assert merged[1].item() == -2.0
    assert merged[2].item() == -3.0
    assert stats["conflict_coordinates"] == 2


def test_dare_is_seeded_and_rescaled() -> None:
    value = torch.ones(1000)
    first, stats = dare_delta(value, drop_probability=0.2, seed=17, key="x")
    second, _ = dare_delta(value, drop_probability=0.2, seed=17, key="x")
    assert torch.equal(first, second)
    assert set(first.unique().tolist()) <= {0.0, 1.25}
    assert 0.7 < stats["actual_keep_fraction"] < 0.9


def test_dare_ties_is_deterministic() -> None:
    deltas = [torch.arange(20).float(), -torch.arange(20).float()]
    first, first_stats = dare_ties_consensus(
        deltas,
        drop_probability=0.1,
        seed=7,
        key="x",
        keep_fraction=0.5,
    )
    second, second_stats = dare_ties_consensus(
        deltas,
        drop_probability=0.1,
        seed=7,
        key="x",
        keep_fraction=0.5,
    )
    assert torch.equal(first, second)
    assert first_stats == second_stats


def test_fisher_weighting_and_zero_denominator_fallback() -> None:
    base = torch.tensor([0.0, 2.0])
    donor = torch.tensor([2.0, 10.0])
    merged, stats = fisher_weighted_parameters(
        base,
        donor,
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        donor_lambda=1.0,
    )
    assert merged.tolist() == pytest.approx([1.0, 2.0])
    assert stats["zero_or_small_denominator"] == 1


def test_layer_and_module_selection() -> None:
    key = "model.layers.4.self_attn.q_proj.weight"
    assert tensor_selected(key, layers={4}, modules={"attention"})
    assert tensor_selected(key, layers={4}, modules={"attention.q"})
    assert not tensor_selected(key, layers={3}, modules={"attention"})
    assert not tensor_selected(key, layers={4}, modules={"mlp"})


def test_fisher_gradient_accumulation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    (parameter.square().sum()).backward()
    accumulator = {}
    used = accumulate_squared_gradients([("x", parameter)], accumulator)
    assert used == 1
    assert torch.equal(accumulator["x"], torch.tensor([16.0, 36.0]))


def test_advanced_candidates_serialize_end_to_end(tmp_path) -> None:
    config = tiny_config(hidden=4, mlp=6, q_heads=2, kv_heads=1, layers=3)
    base_values = tiny_tensors(config, offset=0.0)
    base = write_checkpoint(tmp_path / "base", config, base_values)
    delta1 = write_checkpoint(
        tmp_path / "delta1",
        config,
        {key: torch.ones_like(value) for key, value in base_values.items()},
    )
    delta2 = write_checkpoint(
        tmp_path / "delta2",
        config,
        {key: torch.full_like(value, 2.0) for key, value in base_values.items()},
    )
    ties_output = tmp_path / "ties"
    ties_manifest = merge_ties_candidate(
        base_path=base,
        delta_paths=[delta1, delta2],
        destination=ties_output,
        alpha=0.5,
        keep_fraction=1.0,
        output_dtype="float32",
        max_shard_bytes=1024 * 1024,
    )
    assert ties_manifest["advanced_tensor_stats"]["tensor_count"] == len(base_values)
    assert (ties_output / "advanced_tensor_stats.json").is_file()

    donor = write_checkpoint(
        tmp_path / "donor", config, tiny_tensors(config, offset=2.0)
    )
    fisher_base = write_checkpoint(
        tmp_path / "fisher_base",
        config,
        {key: torch.ones_like(value) for key, value in base_values.items()},
    )
    fisher_donor = write_checkpoint(
        tmp_path / "fisher_donor",
        config,
        {key: torch.ones_like(value) for key, value in base_values.items()},
    )
    fisher_output = tmp_path / "fisher_output"
    merge_fisher_candidate(
        base_path=base,
        donor_path=donor,
        fisher_base_path=fisher_base,
        fisher_donor_path=fisher_donor,
        destination=fisher_output,
        donor_lambda=1.0,
        output_dtype="float32",
        max_shard_bytes=1024 * 1024,
    )
    key = "model.layers.0.self_attn.q_proj.weight"
    with safe_open(str(fisher_output / "model.safetensors"), framework="pt") as handle:
        assert torch.equal(handle.get_tensor(key), base_values[key] + 1.0)
