#!/usr/bin/env python3
"""Write auditable kept-layer α recipes for the Halo-donor cross-depth stitch."""

from __future__ import annotations

import json
from pathlib import Path


LAB = Path(__file__).resolve().parents[1]
LAYER_MAP = LAB / "configs" / "layer_map.json"
OUT = LAB / "configs" / "alpha_recipes.json"

# Phase 2B v4 stage_09 raw-deletion quality delta vs 19L parent
# (PARTIAL_RUN_REPORT.md). More negative = more quality lost if that Qwen3 id is dropped.
STAGE09_RAW_DELTA = {
    0: -0.41982017827119544,
    1: -0.36834270658424484,
    2: -0.2525109481765265,
    3: 0.02980465620567621,
    4: 0.01998831294224232,
    5: 0.06419759410422748,
    11: -0.01149037721864496,
    12: 0.0830207298122933,
    13: 0.08944368690248297,
    15: 0.10028516149533964,
    16: 0.02679015274787344,
    17: 0.00910229668028445,
    19: 0.03346853867008637,
    20: 0.07804887270092462,
    21: 0.02327203290334126,
    22: 0.07571745432533222,
    25: 0.05657709438843947,
    26: 0.03916814251107348,
    27: -0.22915604070284473,
}


def round_vec(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


def affine_mean_clip(
    values: list[float],
    target: float = 0.5,
    lo: float = 0.25,
    hi: float = 0.75,
) -> list[float]:
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [target] * len(values)
    scaled = [lo + (hi - lo) * (value - vmin) / (vmax - vmin) for value in values]
    shifted = [value + (target - mean(scaled)) for value in scaled]
    clipped = [min(hi, max(lo, value)) for value in shifted]
    drift = target - mean(clipped)
    if abs(drift) > 1e-9:
        clipped = [min(hi, max(lo, value + drift)) for value in clipped]
    return round_vec(clipped)


def uniform(value: float, count: int) -> list[float]:
    return round_vec([value] * count)


def main() -> None:
    layer_map = json.loads(LAYER_MAP.read_text(encoding="utf-8"))
    n_rineya = int(layer_map["rineya_layers"])
    mapping = {int(src): int(dst) for src, dst in layer_map["rineya_to_qwen3"].items()}
    rineya_ids = list(range(n_rineya))
    qwen_ids = [mapping[index] for index in rineya_ids]

    late = round_vec(linspace(0.25, 0.75, n_rineya))
    early = round_vec(linspace(0.75, 0.25, n_rineya))
    qwen_ramp = affine_mean_clip([float(qid) for qid in qwen_ids])
    keep_scores = [-STAGE09_RAW_DELTA[qid] for qid in qwen_ids]
    keep_importance = affine_mean_clip(keep_scores)

    recipes = [
        {
            "name": "uniform_0.25",
            "family": "uniform_dose",
            "reuse_report": None,
            "mean_alpha": 0.25,
            "rineya_alphas": uniform(0.25, n_rineya),
            "notes": "Uniform donor weight 0.25 on all 19 kept layers.",
        },
        {
            "name": "uniform_0.375",
            "family": "uniform_dose",
            "reuse_report": None,
            "mean_alpha": 0.375,
            "rineya_alphas": uniform(0.375, n_rineya),
            "notes": "Uniform donor weight 0.375 on all 19 kept layers.",
        },
        {
            "name": "uniform_0.5",
            "family": "uniform_dose",
            "reuse_report": "outputs/qwen3_x_v1_5e4_v2_report.json",
            "recorded_as": "Halo",
            "mean_alpha": 0.5,
            "rineya_alphas": uniform(0.5, n_rineya),
            "notes": "Recorded Halo. Do not restitch; copy the existing 64-example showcase report.",
        },
        {
            "name": "uniform_0.625",
            "family": "uniform_dose",
            "reuse_report": None,
            "mean_alpha": 0.625,
            "rineya_alphas": uniform(0.625, n_rineya),
            "notes": "Uniform donor weight 0.625 on all 19 kept layers.",
        },
        {
            "name": "uniform_0.75",
            "family": "uniform_dose",
            "reuse_report": None,
            "mean_alpha": 0.75,
            "rineya_alphas": uniform(0.75, n_rineya),
            "notes": "Uniform donor weight 0.75 on all 19 kept layers.",
        },
        {
            "name": "uniform_1.0",
            "family": "uniform_dose",
            "reuse_report": "outputs/qwen3_replace_v1_5e4_v2_report.json",
            "recorded_as": "Halo Uncensored",
            "mean_alpha": 1.0,
            "rineya_alphas": uniform(1.0, n_rineya),
            "notes": "Recorded Halo Uncensored. Do not restitch; copy the existing showcase report.",
        },
        {
            "name": "late_rineya",
            "family": "shape_mean_0.5",
            "reuse_report": None,
            "mean_alpha": round(mean(late), 6),
            "rineya_alphas": late,
            "notes": "α rises with Rineya depth (0.25→0.75). Early kept blocks more Qwen3; late more donor. Mean 0.5.",
        },
        {
            "name": "early_rineya",
            "family": "shape_mean_0.5",
            "reuse_report": None,
            "mean_alpha": round(mean(early), 6),
            "rineya_alphas": early,
            "notes": "α falls with Rineya depth (0.75→0.25). Inverse of late_rineya. Mean 0.5.",
        },
        {
            "name": "qwen_index_ramp",
            "family": "shape_mean_0.5",
            "reuse_report": None,
            "mean_alpha": round(mean(qwen_ramp), 6),
            "rineya_alphas": qwen_ramp,
            "qwen3_kept_ids": qwen_ids,
            "notes": "α rises with Qwen3 layer id on kept blocks only, then affine-mapped to mean 0.5 in [0.25, 0.75]. Dropped Qwen3 layers stay untouched.",
        },
        {
            "name": "keep_importance",
            "family": "shape_mean_0.5",
            "reuse_report": None,
            "mean_alpha": round(mean(keep_importance), 6),
            "rineya_alphas": keep_importance,
            "qwen3_kept_ids": qwen_ids,
            "keep_score_neg_delta": keep_scores,
            "notes": "α proportional to −(stage_09 raw deletion quality delta) on the kept Qwen3 id. Hypothesis: more donor on layers that hurt most to drop. Affine-mapped to mean 0.5 in [0.25, 0.75].",
        },
    ]

    payload = {
        "name": "Halo-donor kept-layer α recipes",
        "donor": "v1_5e4_v2",
        "members": ["v1_lr5e4", "v2_lr1e4"],
        "qwen3_model": "Qwen3-0.6B",
        "pulse_base": "short-response-model-phase-0-1-2/artifacts/phase_4_model_soups_phase2c_balanced_training/accepted_model",
        "layer_map": "configs/layer_map.json",
        "halo_hits": 30,
        "halo_items": 64,
        "keep_weights_min_hits": 30,
        "clip": [0.25, 0.75],
        "eval": {
            "input": "Rineya_Long_Context_Demo_v1/data/showcase_prompts.jsonl",
            "key": "Rineya_Long_Context_Demo_v1/data/showcase_answer_key.jsonl",
            "max_new_tokens": 24,
            "max_context": 4096,
            "seed": 17,
            "disable_thinking": True,
            "torch_dtype": "bfloat16",
        },
        "selection": {
            "primary": "hits / 64 strict exact",
            "ties": ["counterfactual_pair_both_correct_and_changed", "unanswerable_accuracy"],
            "noise_floor_hits": 2,
            "record_core": False,
        },
        "recipes": recipes,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(recipes)} recipes)")


if __name__ == "__main__":
    main()
