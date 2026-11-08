from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from rineya_tgsa.historical_dose import build_dose_trial, sha256_file


def _assets(path: Path, *, custom: bool) -> None:
    config = {
        "architectures": ["Qwen3AbstainForCausalLM" if custom else "Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "tie_word_embeddings": True,
        "abstain_token_id": 3 if custom else None,
        "abstain_logit_bias": 5.0 if custom else None,
    }
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name, value in {
        ".gitattributes": "",
        "generation_config.json": "{}",
        "LICENSE": "test",
        "modeling_supa_a11.py": "# test\n",
        "tokenizer_config.json": "{}",
        "tokenizer.json": '{"version":"1.0"}',
        "vocab.json": "{}",
        "merges.txt": "",
    }.items():
        (path / name).write_text(value, encoding="utf-8")


def test_control_reconstruction_is_exact_before_candidate_build(tmp_path: Path) -> None:
    base = tmp_path / "base"
    parent = tmp_path / "parent"
    adapter = tmp_path / "adapter"
    _assets(base, custom=False)
    _assets(parent, custom=True)
    adapter.mkdir()

    base_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).bfloat16()
    base_state = {
        "model.layers.0.self_attn.q_proj.weight": base_weight,
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4).bfloat16(),
        "lm_head.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4).bfloat16(),
    }
    save_file(base_state, base / "model.safetensors", metadata={"format": "pt"})
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    b = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    adapter_state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
    }
    save_file(adapter_state, adapter / "adapter_model.safetensors", metadata={"format": "pt"})
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 1,
                "lora_alpha": 2,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    delta = b @ a
    parent_state = {
        "model.layers.0.self_attn.q_proj.weight": (
            base_weight.float() + delta * 2 * 0.8125
        ).half(),
        "model.embed_tokens.weight": base_state["model.embed_tokens.weight"].half(),
    }
    save_file(parent_state, parent / "model.safetensors", metadata={"format": "pt"})

    registration = {
        "schema_version": "rineya.historical-dose-transfer-trial.v1",
        "trial_id": "dose-test",
        "registration_status": "FROZEN_BEFORE_OUTCOME",
        "parent": {
            "weight_sha256": sha256_file(parent / "model.safetensors"),
            "config_sha256": sha256_file(parent / "config.json"),
            "tokenizer_sha256": sha256_file(parent / "tokenizer.json"),
        },
        "base": {
            "weight_sha256": sha256_file(base / "model.safetensors"),
            "weight_bytes": (base / "model.safetensors").stat().st_size,
            "config_sha256": sha256_file(base / "config.json"),
            "tokenizer_sha256": sha256_file(base / "tokenizer.json"),
        },
        "adapter": {
            "weights_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "weights_bytes": (adapter / "adapter_model.safetensors").stat().st_size,
            "config_sha256": sha256_file(adapter / "adapter_config.json"),
            "rank": 1,
            "lora_alpha": 2,
            "target_modules": ["q_proj"],
            "expected_lora_slots": 1,
        },
        "recipe": {
            "control_scale": 0.8125,
            "candidate_scale": 1.0625,
            "merge_formula": "test",
        },
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")

    result = build_dose_trial(
        registration_path,
        parent,
        base,
        adapter,
        tmp_path / "control",
        tmp_path / "candidate",
        tmp_path / "evidence",
    )
    assert result["control_fidelity"]["passed"] is True
    assert result["control"]["adapted_tensor_count"] == 1
    assert result["candidate"]["scale"] == 1.0625
