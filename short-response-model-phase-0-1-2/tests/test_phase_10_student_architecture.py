from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.common.model_introspection import count_parameters
from scripts.phase_10_student_architecture.phase10_common import (
    apply_attention_group_slicing,
    apply_hidden_coordinate_slicing,
    apply_mlp_channel_slicing,
    candidate_assignment,
    load_candidate_specs,
    retained_layer_indices,
    round_retained_width,
)


class ToyNorm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight


class ToyAttention(nn.Module):
    def __init__(self, hidden: int, q_heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = q_heads
        self.num_attention_heads = q_heads
        self.num_key_value_heads = kv_heads
        self.num_key_value_groups = q_heads // kv_heads
        self.hidden_size = hidden
        self.q_proj = nn.Linear(hidden, q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)


class ToyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = torch.nn.functional.silu


class ToyLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int, q_heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.input_layernorm = ToyNorm(hidden)
        self.post_attention_layernorm = ToyNorm(hidden)
        self.self_attn = ToyAttention(hidden, q_heads, kv_heads, head_dim)
        self.mlp = ToyMLP(hidden, intermediate)


class ToyBackbone(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int, intermediate: int, q_heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [ToyLayer(hidden, intermediate, q_heads, kv_heads, head_dim) for _ in range(layers)]
        )
        self.norm = ToyNorm(hidden)


class ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vocab = 40
        hidden = 16
        layers = 3
        intermediate = 32
        q_heads = 4
        kv_heads = 2
        head_dim = 4
        self.model = ToyBackbone(vocab, hidden, layers, intermediate, q_heads, kv_heads, head_dim)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=q_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            tie_word_embeddings=True,
        )
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, module: nn.Embedding) -> None:
        self.model.embed_tokens = module

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, module: nn.Linear) -> None:
        self.lm_head = module

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight


def test_phase10_candidate_config_and_runner_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/phase_10_student_architecture.json").read_text(encoding="utf-8"))
    assert load_candidate_specs(config, "pilot")
    runner = (root / "scripts/phase_10_student_architecture/run_phase_10.ps1").read_text(encoding="utf-8")
    assert "Invoke-PythonStep" in runner
    assert "create_structural_candidate.py" in runner
    assert "train_student.py" in runner
    assert "check_phase_10_gate.py" in runner
    assert '$RawCandidateSpecs.Add("$Name=$RawSummary")' in runner


def test_retained_layer_and_width_planning() -> None:
    assert retained_layer_indices(6, drop_count=1, mode="drop-last") == [0, 1, 2, 3, 4]
    retained = retained_layer_indices(6, drop_count=2, mode="uniform-drop")
    assert len(retained) == 4
    assert round_retained_width(32, 0.25, 4) == 24
    assert candidate_assignment("x=path/summary.json")[0] == "x"


def test_structural_slicing_reduces_real_tensor_shapes() -> None:
    model = ToyCausalLM()
    before = count_parameters(model)["total"]
    layer_ids = [0, 1, 2]
    mlp_importance = {index: torch.arange(32, dtype=torch.float32) for index in layer_ids}
    head_importance = {index: torch.tensor([1.0, 2.0, 3.0, 4.0]) for index in layer_ids}

    mlp = apply_mlp_channel_slicing(
        model,
        importance_by_original_layer=mlp_importance,
        current_original_layer_ids=layer_ids,
        prune_ratio=0.25,
        multiple=4,
    )
    assert mlp["retained_width"] == 24
    assert model.model.layers[0].mlp.gate_proj.out_features == 24
    assert model.model.layers[0].mlp.down_proj.in_features == 24

    attention = apply_attention_group_slicing(
        model,
        importance_by_original_layer=head_importance,
        current_original_layer_ids=layer_ids,
        prune_ratio=0.5,
    )
    assert attention["retained_kv_heads"] == 1
    assert model.model.layers[0].self_attn.q_proj.out_features == 8
    assert model.model.layers[0].self_attn.o_proj.in_features == 8

    hidden = apply_hidden_coordinate_slicing(
        model,
        hidden_importance=torch.arange(16, dtype=torch.float32),
        prune_ratio=0.25,
        multiple=4,
    )
    assert hidden["retained_hidden_size"] == 12
    assert model.get_input_embeddings().embedding_dim == 12
    assert model.get_output_embeddings().in_features == 12
    assert model.model.layers[0].self_attn.q_proj.in_features == 12
    assert model.model.layers[0].self_attn.o_proj.out_features == 12
    assert model.model.layers[0].mlp.gate_proj.in_features == 12
    assert model.model.layers[0].mlp.down_proj.out_features == 12
    assert model.model.layers[0].input_layernorm.weight.numel() == 12
    assert model.model.norm.weight.numel() == 12
    assert count_parameters(model)["total"] < before


def test_phase10_full_config_contains_declared_size_bands() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/phase_10_student_architecture.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in config["candidate_sets"]["full"]}
    assert by_name["combined_intermediate_target"]["target_parameter_min"] == 180_000_000
    assert by_name["combined_small_target"]["target_parameter_max"] == 180_000_000
    assert by_name["combined_deployment_target"]["target_parameter_max"] == 120_000_000


def test_phase10_phase9_handoff_accepts_hybrid_candidate_names() -> None:
    from scripts.phase_10_student_architecture.finalize_phase_9_selection import (
        candidate_is_research_control,
        validate_candidate_handoff,
    )

    assessment = {
        "candidates": {
            "sft_warm_344m": "artifacts/phase_9/evaluation_sft/summary.json",
            "kd_warm_344m": "artifacts/phase_9/evaluation_kd/summary.json",
            "kd_narrow_130m": "artifacts/phase_9/evaluation_narrow/summary.json",
        },
        "research_controls": ["kd_narrow_130m"],
        "recommended_candidate": "kd_warm_344m",
    }
    assert validate_candidate_handoff(
        assessment,
        candidate="kd_warm_344m",
        role="intermediate_teacher",
        allow_research_control_promotion=False,
    ).endswith("summary.json")
    assert candidate_is_research_control(assessment, "kd_narrow_130m")
    with pytest.raises(ValueError, match="research control"):
        validate_candidate_handoff(
            assessment,
            candidate="kd_narrow_130m",
            role="deployment_candidate",
            allow_research_control_promotion=False,
        )
    assert validate_candidate_handoff(
        assessment,
        candidate="kd_narrow_130m",
        role="research_control",
        allow_research_control_promotion=False,
    ).endswith("summary.json")
    with pytest.raises(ValueError, match="Available candidates"):
        validate_candidate_handoff(
            assessment,
            candidate="kd_missing_profile",
            role="intermediate_teacher",
            allow_research_control_promotion=False,
        )
