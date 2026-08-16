from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from rineya_soup_lab.model.checkpoint import CheckpointReader
from rineya_soup_lab.model.compatibility import compare_checkpoints
from rineya_soup_lab.model.layer_map import MergeRecipe
from rineya_soup_lab.model.merge import merge_checkpoints


def test_streaming_layerwise_merge(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=3)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=10.0, layers=3)
    output = tmp_path / "merged"
    recipe = MergeRecipe(global_alpha=0.2, layer_alphas={1: 0.8}, label="test")
    merge_checkpoints(
        anchor,
        greedy,
        output,
        recipe,
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        max_shard_size_gb=0.000001,
    )
    with CheckpointReader(output) as merged:
        embedding = merged.get_tensor("model.embed_tokens.weight")
        layer0 = merged.get_tensor("model.layers.0.self_attn.q_proj.weight")
        layer1 = merged.get_tensor("model.layers.1.self_attn.q_proj.weight")
    assert torch.allclose(embedding, torch.full_like(embedding, 2.0))
    assert torch.allclose(layer0, torch.full_like(layer0, 3.0))
    # Anchor layer 1 value is 2; greedy value is 12; alpha 0.8 => 10.
    assert torch.allclose(layer1, torch.full_like(layer1, 10.0))


def test_checkpoint_compatibility(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=3)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=3)
    report = compare_checkpoints(
        anchor,
        greedy,
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        expected_layers=3,
        allow_quantized_sources=False,
        require_non_floating_equal=True,
        lineage_acknowledged=True,
        tokenizer_path=anchor,
    )
    assert report["compatible"] is True
    assert report["layer_indices"] == [0, 1, 2]


def test_compatibility_rejects_nonfinite_weights(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    tensors = load_file(str(greedy / "model.safetensors"))
    tensors["model.layers.0.self_attn.q_proj.weight"][0, 0] = float("nan")
    save_file(tensors, str(greedy / "model.safetensors"))
    report = compare_checkpoints(
        anchor,
        greedy,
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        expected_layers=2,
        allow_quantized_sources=False,
        require_non_floating_equal=True,
        lineage_acknowledged=True,
        tokenizer_path=anchor,
    )
    assert report["compatible"] is False
    assert report["non_finite_tensors"]


def test_compatibility_rejects_float_integer_kind_mismatch(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    tensors = load_file(str(greedy / "model.safetensors"))
    key = "model.layers.0.self_attn.q_proj.weight"
    tensors[key] = tensors[key].to(torch.int64)
    save_file(tensors, str(greedy / "model.safetensors"))
    report = compare_checkpoints(
        anchor,
        greedy,
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        expected_layers=2,
        allow_quantized_sources=False,
        require_non_floating_equal=True,
        lineage_acknowledged=True,
        tokenizer_path=anchor,
    )
    assert report["compatible"] is False
    assert report["floating_kind_mismatches"]


def test_checkpoint_index_rejects_shard_path_escape(tmp_path: Path) -> None:
    import json
    import pytest

    from rineya_soup_lab.errors import CompatibilityError

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside.safetensors"
    save_file({"weight": torch.ones(1)}, str(outside))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(CompatibilityError, match="Unsafe shard path"):
        CheckpointReader(checkpoint)


def test_merge_overwrite_replaces_existing_file(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    output = tmp_path / "merged"
    output.write_text("stale", encoding="utf-8")
    merge_checkpoints(
        anchor,
        greedy,
        output,
        MergeRecipe(global_alpha=0.5, label="overwrite_file"),
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        overwrite=True,
    )
    assert output.is_dir()
    assert (output / "merge_manifest.json").is_file()
