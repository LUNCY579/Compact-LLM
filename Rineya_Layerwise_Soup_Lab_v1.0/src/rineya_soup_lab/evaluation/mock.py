from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..model.layer_map import MergeRecipe

from ..utils import read_json, write_json, write_jsonl
from .dataset import load_dataset


def run_mock_evaluator(
    *,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    protocol_name: str,
    recipe: MergeRecipe | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministic pipeline test backend. Never use for research claims."""

    recipe_path = model_path / "merge_recipe.json"
    recipe_payload = (recipe.to_dict() if recipe is not None else (read_json(recipe_path) if recipe_path.exists() else {}))
    label = str(recipe_payload.get("label") or model_path.name).lower()
    if recipe_payload.get("kind") == "endpoint":
        alpha = 0.0 if recipe_payload.get("endpoint") == "anchor" else 1.0
    elif "anchor" in label:
        alpha = 0.0
    elif "greedy" in label:
        alpha = 1.0
    else:
        alpha = float(recipe_payload.get("global_alpha", 0.5))
    layer_alphas = {int(key): float(value) for key, value in (recipe_payload.get("layer_alphas") or {}).items()}
    layer_bonus = sum((index + 1) * (value - alpha) for index, value in layer_alphas.items()) * 0.0005
    quality = max(0.0, min(1.0, 0.80 + 0.08 * (1.0 - ((alpha - 0.62) / 0.62) ** 2) + layer_bonus))
    records = load_dataset(dataset_path)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        perturb = ((index % 7) - 3) * 0.002
        case_quality = max(0.0, min(1.0, quality + perturb))
        rows.append(
            {
                "id": record["id"],
                "category": record["category"],
                "messages": record["messages"],
                "reference": record.get("reference"),
                "reply": "Okay.",
                "visible_generated_tokens": 2,
                "generated_tokens": 2,
                "elapsed_seconds": 0.001,
                "peak_vram_gb": 0.0,
                "complete_utterance": True,
                "semantic_pass": True,
                "observable_quality_pass": True,
                "quality_pass": True,
                "critical_failures": [],
                "major_issues": [],
                "semantic_warnings": [],
                "quality_score": case_quality,
                "quality_dimensions": {
                    "task_adequacy": case_quality,
                    "factual_correctness": case_quality,
                    "context_consistency": case_quality,
                },
            }
        )
    ordered = sorted(float(row["quality_score"]) for row in rows)
    tail_n = max(1, math.ceil(len(ordered) * 0.10))
    dimensions = {
        key: sum(float(row["quality_dimensions"][key]) for row in rows) / len(rows)
        for key in ("task_adequacy", "factual_correctness", "context_consistency")
    }
    summary = {
        "case_count": len(rows),
        "quality_evaluation_version": "4.5",
        "protocol_name": protocol_name,
        "mean_quality_score": sum(ordered) / len(ordered),
        "worst_decile_mean_quality_score": sum(ordered[:tail_n]) / tail_n,
        "p10_quality_score": ordered[tail_n - 1],
        "complete_utterance_rate": 1.0,
        "observable_quality_pass_rate": 1.0,
        "semantic_pass_rate": 1.0,
        "quality_pass_rate": 1.0,
        "critical_failure_count": 0,
        "major_issue_count": 0,
        "median_reply_tokens": 2.0,
        "p90_reply_tokens": 2.0,
        "duplicate_reply_rate": max(0.0, (len(rows) - 1) / len(rows)),
        "mean_quality_dimensions": dimensions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "results.jsonl", rows)
    return summary, rows
