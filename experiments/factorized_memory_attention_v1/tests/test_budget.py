from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.factorized_memory_attention_v1.fma.budget import parameter_budget
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, load_shape
from experiments.factorized_memory_attention_v1.fma.memory import FactorizedMemoryBank
from experiments.factorized_memory_attention_v1.fma.toy import tiny_shape


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/config.json"


def test_registered_shape_and_exact_budget() -> None:
    shape = load_shape(CONFIG)
    assert shape.vocab_size == json.loads(CONFIG.read_text())["vocab_size"]
    assert shape.value_width == shape.num_key_value_heads * shape.head_dim
    budget = parameter_budget(shape, FMAConfig(rank=32), baseline_params=596_049_920)
    assert budget.shared_token_table_params == 4_861_952
    assert budget.layer_projection_params == 917_504
    assert budget.normalization_params == 3_584
    assert budget.added_fma_params == 5_783_040
    assert budget.full_ma_params == 4_356_312_576
    assert budget.full_ma_bf16_bytes == 8_712_625_152
    assert budget.hard_replacement_model_total_params == 572_472_832


def test_rank_changes_only_expected_factorized_terms() -> None:
    shape = tiny_shape()
    low = parameter_budget(shape, FMAConfig(rank=2, layers=(0,)))
    high = parameter_budget(shape, FMAConfig(rank=5, layers=(0,)))
    assert high.added_fma_params - low.added_fma_params == (shape.vocab_size + shape.value_width) * 3


def test_parameter_matched_control_is_within_preregistered_tolerance() -> None:
    shape = load_shape(CONFIG)
    budget = parameter_budget(shape, FMAConfig(rank=32), baseline_params=596_049_920)
    assert abs(budget.control_budget_delta_fraction) <= 0.002


def test_real_dense_materialization_is_guarded() -> None:
    shape = load_shape(CONFIG)
    bank = FactorizedMemoryBank(shape, FMAConfig(rank=32))
    with pytest.raises(MemoryError, match="refusing to materialize"):
        bank.materialize_dense_layer(0)
