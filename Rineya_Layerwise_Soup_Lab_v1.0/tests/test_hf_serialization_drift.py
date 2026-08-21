import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from rineya_soup_lab.model.compatibility import compare_checkpoints
from rineya_soup_lab.model.layer_map import MergeRecipe
from rineya_soup_lab.model.merge import merge_checkpoints


def _compare(anchor: Path, greedy: Path, *, drift: bool) -> dict:
    return compare_checkpoints(
        anchor,
        greedy,
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        expected_layers=2,
        allow_quantized_sources=False,
        require_non_floating_equal=True,
        lineage_acknowledged=True,
        tokenizer_path=anchor,
        allow_hf_serialization_drift=drift,
    )


def _rewrite_config(path: Path, **updates) -> None:
    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    payload.update(updates)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_hf_config_rewrite_requires_drift_flag(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    payload = json.loads((greedy / "config.json").read_text(encoding="utf-8"))
    theta = payload.pop("rope_theta")
    payload["rope_parameters"] = {"rope_theta": theta, "rope_type": "default"}
    payload["layer_types"] = ["full_attention", "full_attention"]
    (greedy / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    strict = _compare(anchor, greedy, drift=False)
    assert strict["compatible"] is False
    drifted = _compare(anchor, greedy, drift=True)
    assert drifted["compatible"] is True
    assert "rope_parameters" in drifted["equivalent_config_fields"]
    assert "layer_types" in drifted["equivalent_config_fields"]


def test_tied_lm_head_alias_and_merge(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=10.0, layers=2)
    _rewrite_config(anchor, tie_word_embeddings=True)
    _rewrite_config(greedy, tie_word_embeddings=True)
    tensors = load_file(str(anchor / "model.safetensors"))
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, str(anchor / "model.safetensors"))
    greedy_tensors = load_file(str(greedy / "model.safetensors"))
    del greedy_tensors["lm_head.weight"]
    save_file(greedy_tensors, str(greedy / "model.safetensors"))

    strict = _compare(anchor, greedy, drift=False)
    assert strict["compatible"] is False
    drifted = _compare(anchor, greedy, drift=True)
    assert drifted["compatible"] is True
    assert drifted["aliased_tied_lm_head"] is True

    output = tmp_path / "merged"
    merge_checkpoints(
        anchor,
        greedy,
        output,
        MergeRecipe(global_alpha=1.0, label="alias"),
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        allow_tied_lm_head_alias=True,
    )
    merged = load_file(str(next(output.glob("*.safetensors"))))
    assert torch.allclose(merged["lm_head.weight"], greedy_tensors["model.embed_tokens.weight"])


def test_tokenizer_drift_is_warning_when_configured(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    (greedy / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 256, "chat_template": "other"}),
        encoding="utf-8",
    )
    strict = _compare(anchor, greedy, drift=False)
    assert strict["compatible"] is False
    drifted = _compare(anchor, greedy, drift=True)
    assert drifted["compatible"] is True
    assert any("tokenizer" in warning.lower() for warning in drifted["warnings"])


def test_unrelated_missing_key_still_fails(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    tensors = load_file(str(greedy / "model.safetensors"))
    del tensors["model.norm.weight"]
    save_file(tensors, str(greedy / "model.safetensors"))
    drifted = _compare(anchor, greedy, drift=True)
    assert drifted["compatible"] is False
