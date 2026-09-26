import pytest

from experiments.factorized_memory_attention_v1.fma.budget import parameter_budget
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, ModelShape


def supa_shape():
    return ModelShape(151936, 1024, 28, 16, 8, 128, 1e-6, "qwen3")


def test_shape_derives_value_and_query_widths():
    shape = supa_shape()
    assert shape.value_width == 1024
    assert shape.query_width == 2048


def test_rank32_exact_parameter_budget():
    budget = parameter_budget(supa_shape(), FMAConfig(rank=32), baseline_params=596_049_920)
    assert budget.shared_token_table_params == 4_861_952
    assert budget.layer_projection_params == 917_504
    assert budget.normalization_params == 3_584
    assert budget.added_fma_params == 5_783_040
    assert budget.removed_standard_value_params == 29_360_128
    assert budget.transition_model_total_params == 601_832_960
    assert budget.hard_replacement_model_total_params == 572_472_832


def test_literal_full_ma_cost_is_calculated_not_instantiated():
    budget = parameter_budget(supa_shape(), FMAConfig(rank=32))
    assert budget.full_ma_params == 4_356_312_576
    assert budget.full_ma_bf16_bytes == 8_712_625_152


def test_rank_changes_parameter_count_monotonically():
    shape = supa_shape()
    counts = [parameter_budget(shape, FMAConfig(rank=r)).added_fma_params for r in (16, 32, 64)]
    assert counts[0] < counts[1] < counts[2]


def test_parameter_matched_control_is_within_point_two_percent():
    budget = parameter_budget(supa_shape(), FMAConfig(rank=32))
    assert budget.control_rank == 101
    assert abs(budget.control_budget_delta_fraction) < 0.002


def test_invalid_head_grouping_is_rejected():
    with pytest.raises(ValueError):
        ModelShape(10, 24, 2, 5, 3, 8, 1e-6)
