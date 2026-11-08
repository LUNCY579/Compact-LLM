from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rineya_tgsa.historical_soup import (
    build_candidate,
    preflight,
    sha256_file,
    verify_candidate,
)


def _config(architecture: str, *, abstention: bool) -> dict:
    value = {
        "architectures": [architecture],
        "model_type": "qwen3",
        "vocab_size": 16,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "sliding_window": None,
        "use_sliding_window": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "layer_types": ["full_attention"],
    }
    if abstention:
        value.update(
            {
                "auto_map": {"AutoModelForCausalLM": "modeling_supa_a11.Qwen3AbstainForCausalLM"},
                "abstain_token_id": 3,
                "abstain_logit_bias": 5.156249985098839,
            }
        )
    return value


def _write_checkpoint(path: Path, *, parent: bool, offset: float) -> None:
    path.mkdir(parents=True)
    config = _config("Qwen3AbstainForCausalLM" if parent else "Qwen3ForCausalLM", abstention=parent)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name, content in {
        "generation_config.json": "{}",
        "tokenizer_config.json": "{}",
        "tokenizer.json": '{"version":"1.0"}',
        "vocab.json": "{}",
        "merges.txt": "",
    }.items():
        (path / name).write_text(content, encoding="utf-8")
    if parent:
        (path / "modeling_supa_a11.py").write_text("# test runtime\n", encoding="utf-8")
        (path / "CORE.json").write_text("{}", encoding="utf-8")
        (path / "FILES.json").write_text("{}", encoding="utf-8")
        (path / "evidence").mkdir()
        (path / "evidence" / "old.json").write_text("{}", encoding="utf-8")
    state = {
        "model.embed_tokens.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4)
        + offset,
        "model.layers.0.self_attn.q_proj.weight": torch.arange(
            16, dtype=torch.float32
        ).reshape(4, 4)
        + offset,
    }
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})


def _registration(path: Path, parent: Path, ingredient: Path) -> Path:
    parent_weight = parent / "model.safetensors"
    ingredient_weight = ingredient / "model.safetensors"
    value = {
        "schema_version": "rineya.historical-transfer-trial.v1",
        "trial_id": "test-soup",
        "registration_status": "FROZEN_BEFORE_OUTCOME",
        "registration_base_commit": "1" * 40,
        "parent": {
            "weight_sha256": sha256_file(parent_weight),
            "weight_bytes": parent_weight.stat().st_size,
            "config_sha256": sha256_file(parent / "config.json"),
            "tokenizer_sha256": sha256_file(parent / "tokenizer.json"),
            "required_class": "Qwen3AbstainForCausalLM",
            "abstain_token_id": 3,
            "abstain_logit_bias": 5.156249985098839,
        },
        "ingredient": {
            "weight_sha256": sha256_file(ingredient_weight),
            "weight_bytes": ingredient_weight.stat().st_size,
            "config_sha256": sha256_file(ingredient / "config.json"),
            "tokenizer_sha256": sha256_file(ingredient / "tokenizer.json"),
        },
        "compatibility_freeze": {"tensor_count_each": 2},
        "recipe": {
            "ingredient_coefficient": 0.5,
            "parent_coefficient": 0.5,
            "accumulation_dtype": "float32",
            "output_dtype": "float16",
        },
        "candidate": {
            "name": "test-candidate",
            "required_runtime_assets": [
                "config.json",
                "generation_config.json",
                "modeling_supa_a11.py",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
            ],
            "forbidden_inherited_identity_assets": ["CORE.json", "FILES.json", "evidence"],
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_preflight_and_build_strip_identity_and_average_weights(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    ingredient = tmp_path / "ingredient"
    _write_checkpoint(parent, parent=True, offset=0.0)
    _write_checkpoint(ingredient, parent=False, offset=2.0)
    registration = _registration(tmp_path / "trial.json", parent, ingredient)

    report = preflight(registration, parent, ingredient)
    assert report["compatibility"]["passed"] is True
    assert report["compatibility"]["tensor_count"] == 2

    destination = tmp_path / "candidate"
    evidence = tmp_path / "trial-evidence"
    artifact = build_candidate(registration, parent, ingredient, destination, evidence)
    assert artifact["verification"]["passed"] is True
    assert not (destination / "CORE.json").exists()
    assert not (destination / "FILES.json").exists()
    assert not (destination / "evidence").exists()
    assert (destination / "TRIAL_ARTIFACT.json").is_file()
    assert verify_candidate(registration, destination)["required_class"] == "Qwen3AbstainForCausalLM"

    weight_file = next(destination.glob("model*.safetensors"))
    with safe_open(weight_file, framework="pt", device="cpu") as handle:
        merged = handle.get_tensor("model.layers.0.self_attn.q_proj.weight")
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1.0
    torch.testing.assert_close(merged.float(), expected, atol=0, rtol=0)


def test_preflight_rejects_weight_hash_drift(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    ingredient = tmp_path / "ingredient"
    _write_checkpoint(parent, parent=True, offset=0.0)
    _write_checkpoint(ingredient, parent=False, offset=2.0)
    registration = _registration(tmp_path / "trial.json", parent, ingredient)
    value = json.loads(registration.read_text(encoding="utf-8"))
    value["ingredient"]["weight_sha256"] = "0" * 64
    registration.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="registered"):
        preflight(registration, parent, ingredient)
