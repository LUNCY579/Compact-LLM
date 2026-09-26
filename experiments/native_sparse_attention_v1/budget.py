"""Analytic storage/parameter arithmetic, not measured GPU timing or peak memory."""
from __future__ import annotations

from .core import NSAConfig


def parameter_budget(model_config: dict, nsa: NSAConfig, converted_layers: int) -> dict:
    w, h, g, d, total = (int(model_config[k]) for k in (
        "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "num_hidden_layers"))
    if min(w, h, g, d, total) <= 0 or h % g or not 0 <= converted_layers <= total:
        raise ValueError("invalid model geometry/layer count")
    if model_config.get("model_type") != "qwen3" or model_config.get("attention_bias", False):
        raise ValueError("budget is for bias-free Qwen3 geometry")
    l, r, a = nsa.block_size, nsa.compression_hidden, nsa.gate_hidden
    compressors = 2 * (l * d + l * d * r + r + r * d + d)
    gates = w * a + a + a * (3 * h) + 3 * h
    projections = 4 * w * g * d if nsa.kv_mode == "independent" else 0
    norms = 2 * d if nsa.kv_mode == "independent" else 0
    per_layer = compressors + gates + projections + norms
    return {"query_heads_per_kv_group": h // g, "compressor_parameters_per_layer": compressors,
            "gate_parameters_per_layer": gates, "extra_kv_parameters_per_layer": projections,
            "extra_norm_parameters_per_layer": norms, "added_parameters_per_layer": per_layer,
            "converted_layers": converted_layers, "added_parameters": converted_layers * per_layer,
            "added_bf16_parameter_bytes": converted_layers * per_layer * 2,
            "measured_peak_memory_bytes": None, "measured_speedup": None}


def access_budget(length: int, nsa: NSAConfig) -> dict:
    if type(length) is not int or length <= 0:
        raise ValueError("positive sequence length required")
    coarse = length // nsa.block_size
    exact = min(length, nsa.selected_blocks * nsa.block_size)
    local = min(length, nsa.window_size)
    return {"length": length, "late_query_compressed_positions_upper_bound": coarse,
            "late_query_selected_positions_upper_bound": exact,
            "late_query_local_positions_upper_bound": local,
            "three_path_position_sum_upper_bound": coarse + exact + local,
            "dense_late_query_positions": length,
            "resident_kv_evicted": False,
            "caveat": "Not a speedup: extra projections/compression/routing/gates/gathers and overlapping paths remain."}
