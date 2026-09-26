from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def write_checkpoint(
    path: Path,
    config: dict,
    tensors: dict[str, torch.Tensor],
    *,
    abstain_bias: float | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    resolved_config = dict(config)
    if abstain_bias is not None:
        resolved_config["abstain_logit_bias"] = abstain_bias
    (path / "config.json").write_text(
        json.dumps(resolved_config, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        str(path / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return path


def tiny_config(*, hidden: int, mlp: int, q_heads: int, kv_heads: int, layers: int):
    return {
        "model_type": "qwen3",
        "vocab_size": 16,
        "head_dim": 2,
        "hidden_size": hidden,
        "intermediate_size": mlp,
        "num_attention_heads": q_heads,
        "num_key_value_heads": kv_heads,
        "num_hidden_layers": layers,
        "tie_word_embeddings": True,
    }


def tiny_tensors(config: dict, *, offset: float = 0.0) -> dict[str, torch.Tensor]:
    hidden = config["hidden_size"]
    mlp = config["intermediate_size"]
    q_features = config["num_attention_heads"] * config["head_dim"]
    kv_features = config["num_key_value_heads"] * config["head_dim"]
    counter = 0

    def value(shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal counter
        count = 1
        for dimension in shape:
            count *= dimension
        start = counter
        counter += count
        return torch.arange(start, start + count, dtype=torch.float32).reshape(shape) + offset

    result = {"model.embed_tokens.weight": value((16, hidden))}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        result[prefix + "input_layernorm.weight"] = value((hidden,))
        result[prefix + "post_attention_layernorm.weight"] = value((hidden,))
        result[prefix + "self_attn.q_proj.weight"] = value((q_features, hidden))
        result[prefix + "self_attn.k_proj.weight"] = value((kv_features, hidden))
        result[prefix + "self_attn.v_proj.weight"] = value((kv_features, hidden))
        result[prefix + "self_attn.o_proj.weight"] = value((hidden, q_features))
        result[prefix + "mlp.gate_proj.weight"] = value((mlp, hidden))
        result[prefix + "mlp.up_proj.weight"] = value((mlp, hidden))
        result[prefix + "mlp.down_proj.weight"] = value((hidden, mlp))
    result["model.norm.weight"] = value((hidden,))
    return result
