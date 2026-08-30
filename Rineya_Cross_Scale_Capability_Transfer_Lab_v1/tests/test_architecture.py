from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from rineya_cross_scale.architecture import (
    compatibility_report,
    center_preserving_depth_map,
    inspect_tensor_headers,
    read_safetensors_header,
    uniform_depth_map,
)


def write_header(path: Path, header: dict) -> None:
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(payload) % 8) % 8
    payload += b" " * padding
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_uniform_depth_map_28_to_36() -> None:
    mapping = uniform_depth_map(28, 36)
    assert mapping[0] == 0
    assert mapping[-1] == 35
    assert len(mapping) == 28
    assert mapping == sorted(mapping)


def test_center_preserving_map_keeps_edge_quarters() -> None:
    mapping = center_preserving_depth_map(28, 36)
    assert mapping[:7] == list(range(7))
    assert mapping[-7:] == list(range(29, 36))
    assert len(mapping) == 28
    assert mapping == sorted(mapping)


def test_read_and_inspect_safetensors_header(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    header = {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [10, 4],
            "data_offsets": [0, 80],
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "BF16",
            "shape": [8, 4],
            "data_offsets": [80, 144],
        },
        "__metadata__": {"format": "pt"},
    }
    write_header(path, header)
    assert read_safetensors_header(path)["__metadata__"]["format"] == "pt"
    report = inspect_tensor_headers([path], None)
    assert report["tensor_count"] == 2
    assert report["parameter_count_serialized"] == 72
    assert report["layer_ids"] == [0]
    assert report["bias_keys"] == []


def test_reject_truncated_header(tmp_path: Path) -> None:
    path = tmp_path / "broken.safetensors"
    path.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(RuntimeError, match="truncated"):
        read_safetensors_header(path)


def test_compatibility_accepts_verified_tied_head_alias() -> None:
    def audit(name: str, hidden: int, layers: int, q_heads: int, include_head: bool):
        shapes = {
            "model.embed_tokens.weight": [16, hidden],
            "model.norm.weight": [hidden],
            "model.layers.0.input_layernorm.weight": [hidden],
        }
        if include_head:
            shapes["lm_head.weight"] = [16, hidden]
        layout = [
            {"key": key, "dtype": "BF16", "shape": shape}
            for key, shape in sorted(shapes.items())
        ]
        return {
            "config": {
                "model_type": "qwen3",
                "vocab_size": 16,
                "head_dim": 2,
                "hidden_size": hidden,
                "intermediate_size": hidden * 3,
                "num_hidden_layers": layers,
                "num_attention_heads": q_heads,
                "num_key_value_heads": 1,
                "tie_word_embeddings": True,
            },
            "tokenizer": {"bundle_sha256": "same"},
            "weights": {
                "state_dict_keys": sorted(shapes),
                "state_dict_tensor_layout": layout,
                "layer_schema": {"input_layernorm.weight": {}},
            },
            "tied_embedding_verification": {
                "status": "verified_redundant" if include_head else "verified_implicit"
            },
            "release_manifest_verification": (
                {"status": "verified"} if name == "supa" else None
            ),
        }

    audits = {
        "qwen06": audit("qwen06", 4, 3, 2, True),
        "supa": audit("supa", 4, 3, 2, False),
        "qwen4b": audit("qwen4b", 8, 5, 4, False),
        "qwen8b": audit("qwen8b", 8, 5, 4, True),
    }
    report = compatibility_report(audits)
    assert report["projection_gate"] == "GO"
    assert report["small_reference"]["exact_state_dict_layout_match"] is True
