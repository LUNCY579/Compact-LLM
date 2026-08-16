from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def tiny_checkpoint_factory():
    def create(path: Path, *, offset: float, layers: int = 3) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.full((8, 4), offset, dtype=torch.float32),
            "model.norm.weight": torch.full((4,), offset + 0.25, dtype=torch.float32),
            "lm_head.weight": torch.full((8, 4), offset + 0.5, dtype=torch.float32),
        }
        for layer in range(layers):
            tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.full(
                (4, 4), offset + layer + 1.0, dtype=torch.float32
            )
            tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = torch.full(
                (4, 6), offset + layer + 1.5, dtype=torch.float32
            )
        save_file(tensors, str(path / "model.safetensors"))
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "rope_theta": 10000.0,
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 128, "chat_template": "dummy"}), encoding="utf-8"
        )
        return path

    return create


def write_dataset(path: Path, count: int = 14, *, prefix: str = "dev") -> Path:
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"{prefix}_{index:03d}",
                "category": "simple_fact" if index % 2 == 0 else "casual",
                "messages": [{"role": "user", "content": f"Question {prefix} {index}?"}],
                "reference": "Okay.",
                "expected": {"min_tokens": 1, "max_tokens": 12},
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path
