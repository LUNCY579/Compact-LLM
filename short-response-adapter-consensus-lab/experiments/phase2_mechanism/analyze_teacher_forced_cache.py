from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[2]
CACHE = LAB_ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl"
MANIFEST = LAB_ROOT / "artifacts/phase2_mechanism/teacher_forced_dev_capture_topk64_manifest.json"
OUTPUT_ROOT = LAB_ROOT / "results/phase2_mechanism"
PARTIAL_JSON = OUTPUT_ROOT / "teacher_forced_dev_analysis_partial.json"
FINAL_JSON = OUTPUT_ROOT / "teacher_forced_dev_analysis.json"
COMMITTEE_CSV = OUTPUT_ROOT / "teacher_forced_committee_results.csv"
SCALING_CSV = OUTPUT_ROOT / "teacher_forced_committee_scaling.csv"
DIVERSITY_CSV = OUTPUT_ROOT / "teacher_forced_expert_diversity.csv"
REPORT = LAB_ROOT / "reports/PHASE2_TEACHER_FORCED_DEV.md"
PARTIAL_REPORT = LAB_ROOT / "reports/PHASE2_TEACHER_FORCED_DEV_PARTIAL.md"

EXPERTS = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]
N_EXPERTS = len(EXPERTS)
FULL_MASK = (1 << N_EXPERTS) - 1
METHODS = [
    "hard_plurality",
    "probability_mean",
    "calibrated_probability_mean",
    "shared_temperature_probability_mean",
    "logit_mean",
    "normalized_logit_mean",
    "top2_approval",
    "borda_top5",
    "reciprocal_rank_top5",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def subset_inventory() -> tuple[list[int], np.ndarray, list[list[str]]]:
    masks = list(range(1, FULL_MASK + 1))
    matrix = np.asarray(
        [[1.0 if mask & (1 << index) else 0.0 for index in range(N_EXPERTS)] for mask in masks],
        dtype=np.float64,
    )
    members = [
        [expert for index, expert in enumerate(EXPERTS) if mask & (1 << index)]
        for mask in masks
    ]
    return masks, matrix, members


def stable_argmax(scores: np.ndarray) -> np.ndarray:
    # Candidate IDs are sorted ascending, so NumPy's first-maximum behavior
    # implements the final smallest-token-ID tie rule.
    return np.argmax(scores, axis=1)


def tie_break_primary(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    maximum = primary.max(axis=1, keepdims=True)
    tied = np.isclose(primary, maximum, rtol=1e-12, atol=1e-12)
    scores = np.where(tied, secondary, -np.inf)
    return stable_argmax(scores)


def fit_temperatures(manifest: dict[str, Any]) -> dict[str, Any]:
    grid = [float(value) for value in manifest["temperature_grid"]]
    losses = np.zeros((N_EXPERTS, len(grid)), dtype=np.float64)
    token_count = 0
    example_count = 0
    calibration_values: dict[str, dict[str, list[float]]] = {
        expert: defaultdict(list) for expert in EXPERTS
    }
    category_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in iter_jsonl(CACHE):
        example_count += 1
        for token in row["tokens"]:
            token_count += 1
            for expert_index, expert in enumerate(EXPERTS):
                metrics = token["expert_metrics"][expert_index]
                gold_logit = float(metrics["gold_logit"])
                lse = np.asarray(metrics["logsumexp_by_temperature"], dtype=np.float64)
                losses[expert_index] += -gold_logit / np.asarray(grid) + lse
                values = calibration_values[expert]
                values["confidence"].append(float(metrics["top1_probability"]))
                values["correct"].append(float(bool(metrics["top1_correct"])))
                values["nll"].append(-float(metrics["gold_log_probability"]))
                values["brier"].append(float(metrics["brier_score"]))
                values["entropy"].append(float(metrics["entropy"]))
                values["margin"].append(float(metrics["margin"]))
                category = str(row.get("reasoning_family") or "unknown")
                category_values[f"{expert}\x1f{category}"]["confidence"].append(
                    float(metrics["top1_probability"])
                )
                category_values[f"{expert}\x1f{category}"]["correct"].append(
                    float(bool(metrics["top1_correct"]))
                )
    if token_count == 0:
        raise RuntimeError("Teacher-forced cache is empty")
    best_indices = np.argmin(losses, axis=1)
    shared_losses = losses.sum(axis=0) / (token_count * N_EXPERTS)
    shared_index = int(np.argmin(shared_losses))
    experts = {}
    for expert_index, expert in enumerate(EXPERTS):
        values = calibration_values[expert]
        confidence = np.asarray(values["confidence"])
        correct = np.asarray(values["correct"])
        bins = []
        ece = 0.0
        for low in np.linspace(0.0, 1.0, 16)[:-1]:
            high = low + 1 / 15
            mask = (confidence >= low) & (confidence <= high if high >= 1.0 else confidence < high)
            count = int(mask.sum())
            accuracy = float(correct[mask].mean()) if count else None
            mean_confidence = float(confidence[mask].mean()) if count else None
            if count:
                ece += count / len(confidence) * abs(float(accuracy) - float(mean_confidence))
            bins.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "count": count,
                    "accuracy": accuracy,
                    "mean_confidence": mean_confidence,
                }
            )
        experts[expert] = {
            "temperature": grid[int(best_indices[expert_index])],
            "best_temperature_index": int(best_indices[expert_index]),
            "token_count": token_count,
            "top1_accuracy": float(correct.mean()),
            "mean_confidence": float(confidence.mean()),
            "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
            "ece_15": float(ece),
            "mean_nll": statistics.mean(values["nll"]),
            "mean_brier": statistics.mean(values["brier"]),
            "mean_entropy": statistics.mean(values["entropy"]),
            "mean_margin": statistics.mean(values["margin"]),
            "mean_nll_by_temperature": {
                str(temperature): float(losses[expert_index, temp_index] / token_count)
                for temp_index, temperature in enumerate(grid)
            },
            "reliability_bins": bins,
        }
    categories = []
    for key, values in sorted(category_values.items()):
        expert, category = key.split("\x1f", maxsplit=1)
        categories.append(
            {
                "expert": expert,
                "task_category": category,
                "token_count": len(values["correct"]),
                "top1_accuracy": statistics.mean(values["correct"]),
                "mean_confidence": statistics.mean(values["confidence"]),
                "confidence_minus_accuracy": statistics.mean(values["confidence"])
                - statistics.mean(values["correct"]),
            }
        )
    return {
        "example_count": example_count,
        "token_count": token_count,
        "temperature_grid": grid,
        "experts": experts,
        "shared_temperature": grid[shared_index],
        "shared_temperature_index": shared_index,
        "shared_mean_nll_by_temperature": {
            str(temperature): float(shared_losses[index]) for index, temperature in enumerate(grid)
        },
        "category_profiles": categories,
    }


def expert_error_diversity() -> dict[str, Any]:
    """Gold-prefix expert complementarity without using free-running outcomes."""
    pair_counts: dict[tuple[int, int], Counter[str]] = {
        pair: Counter() for pair in combinations(range(N_EXPERTS), 2)
    }
    token_count = 0
    oracle_token_correct = 0
    unanimous_correct = 0
    unanimous_wrong = 0
    expert_token_correct = np.zeros(N_EXPERTS, dtype=np.int64)
    expert_example_correct = np.zeros(N_EXPERTS, dtype=np.int64)
    oracle_example_correct = 0
    example_count = 0
    for row in iter_jsonl(CACHE):
        example_count += 1
        example_ok = np.ones(N_EXPERTS, dtype=bool)
        oracle_ok = True
        for token in row["tokens"]:
            token_count += 1
            correct = np.asarray(
                [bool(metrics["top1_correct"]) for metrics in token["expert_metrics"]], dtype=bool
            )
            top_ids = np.asarray(
                [int(metrics["top1_token_id"]) for metrics in token["expert_metrics"]], dtype=np.int64
            )
            expert_token_correct += correct
            example_ok &= correct
            any_correct = bool(correct.any())
            oracle_token_correct += int(any_correct)
            oracle_ok &= any_correct
            unanimous_correct += int(bool(correct.all()))
            unanimous_wrong += int(not any_correct)
            for left, right in pair_counts:
                counts = pair_counts[(left, right)]
                left_correct = bool(correct[left])
                right_correct = bool(correct[right])
                counts[f"correct_{int(left_correct)}{int(right_correct)}"] += 1
                counts["top1_disagreement"] += int(top_ids[left] != top_ids[right])
        expert_example_correct += example_ok
        oracle_example_correct += int(oracle_ok)

    pairs = []
    for (left, right), counts in pair_counts.items():
        n11 = counts["correct_11"]
        n10 = counts["correct_10"]
        n01 = counts["correct_01"]
        n00 = counts["correct_00"]
        q_denominator = n11 * n00 + n10 * n01
        phi_denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        pairs.append(
            {
                "expert_a": EXPERTS[left],
                "expert_b": EXPERTS[right],
                "token_count": token_count,
                "both_correct": n11,
                "a_only_correct": n10,
                "b_only_correct": n01,
                "both_wrong": n00,
                "top1_disagreement_rate": counts["top1_disagreement"] / token_count,
                "correctness_phi": ((n11 * n00 - n10 * n01) / phi_denominator if phi_denominator else None),
                "q_statistic": ((n11 * n00 - n10 * n01) / q_denominator if q_denominator else None),
                "double_fault_rate": n00 / token_count,
                "complementary_correct_rate": (n10 + n01) / token_count,
            }
        )
    return {
        "analysis_role": "DEV MECHANISTIC / EXPLORATORY",
        "token_count": token_count,
        "example_count": example_count,
        "expert_token_accuracy": {
            expert: float(expert_token_correct[index] / token_count)
            for index, expert in enumerate(EXPERTS)
        },
        "expert_all_token_example_accuracy": {
            expert: float(expert_example_correct[index] / example_count)
            for index, expert in enumerate(EXPERTS)
        },
        "dynamic_token_oracle_accuracy": oracle_token_correct / token_count,
        "dynamic_token_oracle_all_token_example_accuracy": oracle_example_correct / example_count,
        "oracle_token_gain_over_best_individual": (
            oracle_token_correct / token_count - float(expert_token_correct.max() / token_count)
        ),
        "unanimous_correct_rate": unanimous_correct / token_count,
        "unanimous_wrong_rate": unanimous_wrong / token_count,
        "pairwise": pairs,
        "interpretation": (
            "The dynamic oracle is an unattainable upper bound that may choose a different expert at every gold-prefix token. "
            "Pairwise correctness statistics are associations, not causal specialization controls."
        ),
    }


def attach_committee_diversity(
    committee_rows: list[dict[str, Any]], diversity: dict[str, Any]
) -> list[dict[str, Any]]:
    pair_by_names = {
        frozenset((row["expert_a"], row["expert_b"])): row for row in diversity["pairwise"]
    }
    for row in committee_rows:
        member_pairs = [pair_by_names[frozenset(pair)] for pair in combinations(row["members"], 2)]
        row["mean_pairwise_top1_disagreement"] = (
            statistics.mean(pair["top1_disagreement_rate"] for pair in member_pairs)
            if member_pairs
            else None
        )
        row["mean_pairwise_double_fault"] = (
            statistics.mean(pair["double_fault_rate"] for pair in member_pairs)
            if member_pairs
            else None
        )

    associations = []
    for method in METHODS:
        for k in range(2, N_EXPERTS):
            group = [row for row in committee_rows if row["method"] == method and row["k"] == k]
            utility = np.asarray([row["example_accuracy"] for row in group], dtype=np.float64)
            disagreement = np.asarray(
                [row["mean_pairwise_top1_disagreement"] for row in group], dtype=np.float64
            )
            double_fault = np.asarray([row["mean_pairwise_double_fault"] for row in group], dtype=np.float64)

            def correlation(values: np.ndarray) -> float | None:
                return (
                    float(np.corrcoef(values, utility)[0, 1])
                    if len(values) > 1 and np.std(values) > 0 and np.std(utility) > 0
                    else None
                )

            associations.append(
                {
                    "method": method,
                    "k": k,
                    "committee_count": len(group),
                    "top1_disagreement_vs_example_accuracy_pearson": correlation(disagreement),
                    "double_fault_vs_example_accuracy_pearson": correlation(double_fault),
                    "analysis_role": "DEV MECHANISTIC / EXPLORATORY",
                }
            )
    return associations


def rank_matrix(token: dict[str, Any], candidate_ids: np.ndarray, *, mode: str, depth: int) -> np.ndarray:
    scores = np.zeros((N_EXPERTS, len(candidate_ids)), dtype=np.float64)
    index_by_token = {int(token_id): index for index, token_id in enumerate(candidate_ids.tolist())}
    for expert_index, metrics in enumerate(token["expert_metrics"]):
        for rank, token_id in enumerate(metrics["topk_token_ids"][:depth], start=1):
            candidate_index = index_by_token[int(token_id)]
            if mode == "approval":
                scores[expert_index, candidate_index] = 1.0
            elif mode == "borda":
                scores[expert_index, candidate_index] = depth - rank + 1
            elif mode == "reciprocal":
                scores[expert_index, candidate_index] = 1.0 / rank
            else:
                raise ValueError(mode)
    return scores


def initialize_metric_arrays(subset_count: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        method: {
            "token_correct": np.zeros(subset_count, dtype=np.int64),
            "token_total": np.zeros(subset_count, dtype=np.int64),
            "example_correct": np.zeros(subset_count, dtype=np.int64),
            "example_total": np.zeros(subset_count, dtype=np.int64),
            "first_token_correct": np.zeros(subset_count, dtype=np.int64),
            "first_token_total": np.zeros(subset_count, dtype=np.int64),
            "answer_token_correct": np.zeros(subset_count, dtype=np.int64),
            "answer_token_total": np.zeros(subset_count, dtype=np.int64),
            "eos_correct": np.zeros(subset_count, dtype=np.int64),
            "eos_total": np.zeros(subset_count, dtype=np.int64),
            "certified": np.zeros(subset_count, dtype=np.int64),
            "certification_total": np.zeros(subset_count, dtype=np.int64),
        }
        for method in METHODS
    }


def aggregate_all_subsets(
    calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks, subset_matrix, members = subset_inventory()
    subset_sizes = subset_matrix.sum(axis=1)
    metrics = initialize_metric_arrays(len(masks))
    grid = calibration["temperature_grid"]
    raw_temp_index = grid.index(1.0)
    fitted_indices = np.asarray(
        [calibration["experts"][expert]["best_temperature_index"] for expert in EXPERTS],
        dtype=np.int64,
    )
    shared_index = int(calibration["shared_temperature_index"])
    shared_intervention = {
        str(temperature): {
            "token_correct": 0,
            "token_total": 0,
            "example_correct": 0,
            "example_total": 0,
            "certified": 0,
        }
        for temperature in grid
    }
    expert_intervention = {
        expert: {
            str(temperature): {
                "token_correct": 0,
                "token_total": 0,
                "example_correct": 0,
                "example_total": 0,
                "certified": 0,
            }
            for temperature in grid
        }
        for expert in EXPERTS
    }
    full_index = masks.index(FULL_MASK)
    for row in iter_jsonl(CACHE):
        example_ok = {method: np.ones(len(masks), dtype=bool) for method in METHODS}
        shared_ok = {str(temperature): True for temperature in grid}
        expert_ok = {
            expert: {str(temperature): True for temperature in grid} for expert in EXPERTS
        }
        for token_index, token in enumerate(row["tokens"]):
            candidate_ids = np.asarray(token["candidate_token_ids"], dtype=np.int64)
            logits = np.asarray(token["candidate_logits_by_expert"], dtype=np.float64)
            gold = int(token["gold_token_id"])
            lse = np.asarray(
                [item["logsumexp_by_temperature"] for item in token["expert_metrics"]],
                dtype=np.float64,
            )
            kth_logits = np.asarray(
                [float(item["topk_logits"][-1]) for item in token["expert_metrics"]],
                dtype=np.float64,
            )
            top1_ids = np.asarray(
                [int(item["top1_token_id"]) for item in token["expert_metrics"]],
                dtype=np.int64,
            )
            raw_log_probabilities = logits - lse[:, raw_temp_index][:, None]
            raw_probabilities = np.exp(raw_log_probabilities)
            hard_votes = (top1_ids[:, None] == candidate_ids[None, :]).astype(np.float64)
            hard_primary = subset_matrix @ hard_votes
            hard_secondary = subset_matrix @ raw_log_probabilities
            winner_indices: dict[str, np.ndarray] = {
                "hard_plurality": tie_break_primary(hard_primary, hard_secondary),
            }
            probability_scores = (subset_matrix @ raw_probabilities) / subset_sizes[:, None]
            winner_indices["probability_mean"] = stable_argmax(probability_scores)
            fitted_probabilities = np.exp(
                logits / np.asarray([grid[index] for index in fitted_indices])[:, None]
                - lse[np.arange(N_EXPERTS), fitted_indices][:, None]
            )
            calibrated_scores = (subset_matrix @ fitted_probabilities) / subset_sizes[:, None]
            winner_indices["calibrated_probability_mean"] = stable_argmax(calibrated_scores)
            shared_temperature = float(grid[shared_index])
            shared_probabilities = np.exp(
                logits / shared_temperature - lse[:, shared_index][:, None]
            )
            shared_scores = (subset_matrix @ shared_probabilities) / subset_sizes[:, None]
            winner_indices["shared_temperature_probability_mean"] = stable_argmax(shared_scores)
            logit_scores = (subset_matrix @ logits) / subset_sizes[:, None]
            winner_indices["logit_mean"] = stable_argmax(logit_scores)
            means = np.asarray([float(item["logit_mean"]) for item in token["expert_metrics"]])
            stds = np.asarray([float(item["logit_std"]) for item in token["expert_metrics"]])
            normalized_logits = (logits - means[:, None]) / np.maximum(stds[:, None], 1e-12)
            normalized_scores = (subset_matrix @ normalized_logits) / subset_sizes[:, None]
            winner_indices["normalized_logit_mean"] = stable_argmax(normalized_scores)
            approval = rank_matrix(token, candidate_ids, mode="approval", depth=2)
            approval_primary = subset_matrix @ approval
            winner_indices["top2_approval"] = tie_break_primary(approval_primary, hard_secondary)
            borda = rank_matrix(token, candidate_ids, mode="borda", depth=5)
            borda_primary = subset_matrix @ borda
            winner_indices["borda_top5"] = tie_break_primary(borda_primary, hard_secondary)
            reciprocal = rank_matrix(token, candidate_ids, mode="reciprocal", depth=5)
            reciprocal_primary = subset_matrix @ reciprocal
            winner_indices["reciprocal_rank_top5"] = tie_break_primary(reciprocal_primary, hard_secondary)

            probability_outside = (
                subset_matrix @ np.exp(kth_logits - lse[:, raw_temp_index])
            ) / subset_sizes
            calibrated_outside = (
                subset_matrix
                @ np.exp(
                    kth_logits / np.asarray([grid[index] for index in fitted_indices])
                    - lse[np.arange(N_EXPERTS), fitted_indices]
                )
            ) / subset_sizes
            shared_outside = (
                subset_matrix @ np.exp(kth_logits / shared_temperature - lse[:, shared_index])
            ) / subset_sizes
            logit_outside = (subset_matrix @ kth_logits) / subset_sizes
            normalized_outside = (
                subset_matrix @ ((kth_logits - means) / np.maximum(stds, 1e-12))
            ) / subset_sizes
            certification = {
                "hard_plurality": np.ones(len(masks), dtype=bool),
                "probability_mean": probability_scores[np.arange(len(masks)), winner_indices["probability_mean"]]
                > probability_outside + 1e-15,
                "calibrated_probability_mean": calibrated_scores[
                    np.arange(len(masks)), winner_indices["calibrated_probability_mean"]
                ]
                > calibrated_outside + 1e-15,
                "shared_temperature_probability_mean": shared_scores[
                    np.arange(len(masks)), winner_indices["shared_temperature_probability_mean"]
                ]
                > shared_outside + 1e-15,
                "logit_mean": logit_scores[np.arange(len(masks)), winner_indices["logit_mean"]]
                > logit_outside + 1e-12,
                "normalized_logit_mean": normalized_scores[
                    np.arange(len(masks)), winner_indices["normalized_logit_mean"]
                ]
                > normalized_outside + 1e-12,
                "top2_approval": np.ones(len(masks), dtype=bool),
                "borda_top5": np.ones(len(masks), dtype=bool),
                "reciprocal_rank_top5": np.ones(len(masks), dtype=bool),
            }
            is_eos = token_index == len(row["tokens"]) - 1
            for method in METHODS:
                selected = candidate_ids[winner_indices[method]]
                correct = selected == gold
                metric = metrics[method]
                metric["token_correct"] += correct
                metric["token_total"] += 1
                metric["certified"] += certification[method]
                metric["certification_total"] += 1
                example_ok[method] &= correct
                if token_index == 0:
                    metric["first_token_correct"] += correct
                    metric["first_token_total"] += 1
                if is_eos:
                    metric["eos_correct"] += correct
                    metric["eos_total"] += 1
                else:
                    metric["answer_token_correct"] += correct
                    metric["answer_token_total"] += 1

            full_subset = np.ones(N_EXPERTS, dtype=np.float64)
            for temperature_index, temperature in enumerate(grid):
                key = str(temperature)
                probabilities = np.exp(logits / temperature - lse[:, temperature_index][:, None])
                scores = probabilities.mean(axis=0)
                winner = int(np.argmax(scores))
                selected = int(candidate_ids[winner])
                correct = selected == gold
                outside = float(np.exp(kth_logits / temperature - lse[:, temperature_index]).mean())
                shared_intervention[key]["token_correct"] += int(correct)
                shared_intervention[key]["token_total"] += 1
                shared_intervention[key]["certified"] += int(scores[winner] > outside + 1e-15)
                shared_ok[key] &= correct
            for perturbed_index, expert in enumerate(EXPERTS):
                for temperature_index, temperature in enumerate(grid):
                    key = str(temperature)
                    probabilities = raw_probabilities.copy()
                    probabilities[perturbed_index] = np.exp(
                        logits[perturbed_index] / temperature - lse[perturbed_index, temperature_index]
                    )
                    scores = probabilities.mean(axis=0)
                    winner = int(np.argmax(scores))
                    selected = int(candidate_ids[winner])
                    correct = selected == gold
                    outside_by_expert = np.exp(kth_logits - lse[:, raw_temp_index])
                    outside_by_expert[perturbed_index] = math.exp(
                        kth_logits[perturbed_index] / temperature - lse[perturbed_index, temperature_index]
                    )
                    outside = float(outside_by_expert.mean())
                    expert_intervention[expert][key]["token_correct"] += int(correct)
                    expert_intervention[expert][key]["token_total"] += 1
                    expert_intervention[expert][key]["certified"] += int(scores[winner] > outside + 1e-15)
                    expert_ok[expert][key] &= correct
        for method in METHODS:
            metrics[method]["example_correct"] += example_ok[method]
            metrics[method]["example_total"] += 1
        for key in shared_ok:
            shared_intervention[key]["example_correct"] += int(shared_ok[key])
            shared_intervention[key]["example_total"] += 1
        for expert in EXPERTS:
            for key in expert_ok[expert]:
                expert_intervention[expert][key]["example_correct"] += int(expert_ok[expert][key])
                expert_intervention[expert][key]["example_total"] += 1

    committee_rows = []
    for method in METHODS:
        metric = metrics[method]
        for index, mask in enumerate(masks):
            committee_rows.append(
                {
                    "method": method,
                    "mask": mask,
                    "bitmask": format(mask, f"0{N_EXPERTS}b")[::-1],
                    "members": members[index],
                    "k": int(subset_sizes[index]),
                    "token_correct": int(metric["token_correct"][index]),
                    "token_total": int(metric["token_total"][index]),
                    "token_accuracy": float(metric["token_correct"][index] / metric["token_total"][index]),
                    "example_correct": int(metric["example_correct"][index]),
                    "example_total": int(metric["example_total"][index]),
                    "example_accuracy": float(metric["example_correct"][index] / metric["example_total"][index]),
                    "first_token_accuracy": float(
                        metric["first_token_correct"][index] / metric["first_token_total"][index]
                    ),
                    "answer_token_accuracy": float(
                        metric["answer_token_correct"][index] / metric["answer_token_total"][index]
                    ),
                    "eos_accuracy": float(metric["eos_correct"][index] / metric["eos_total"][index]),
                    "candidate_argmax_certification_rate": float(
                        metric["certified"][index] / metric["certification_total"][index]
                    ),
                    "expert_token_forward_proxy": int(subset_sizes[index] * metric["token_total"][index]),
                    "analysis_role": "DEV MODEL-SELECTION / CALIBRATION",
                }
            )
    for collection in (shared_intervention,):
        for values in collection.values():
            values["token_accuracy"] = values["token_correct"] / values["token_total"]
            values["example_accuracy"] = values["example_correct"] / values["example_total"]
            values["certification_rate"] = values["certified"] / values["token_total"]
    for expert in EXPERTS:
        for values in expert_intervention[expert].values():
            values["token_accuracy"] = values["token_correct"] / values["token_total"]
            values["example_accuracy"] = values["example_correct"] / values["example_total"]
            values["certification_rate"] = values["certified"] / values["token_total"]
    return committee_rows, {
        "shared_temperature": shared_intervention,
        "one_expert_perturbed": expert_intervention,
        "hard_positive_temperature_top1_invariant": True,
        "full_committee_index": full_index,
    }


def committee_scaling(committee_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packets = []
    for method in METHODS:
        method_rows = [row for row in committee_rows if row["method"] == method]
        for k in range(1, N_EXPERTS + 1):
            group = [row for row in method_rows if row["k"] == k]
            best = max(group, key=lambda row: (row["example_accuracy"], row["token_accuracy"], -row["mask"]))
            packets.append(
                {
                    "method": method,
                    "k": k,
                    "committee_count": len(group),
                    "mean_example_accuracy": statistics.mean(row["example_accuracy"] for row in group),
                    "min_example_accuracy": min(row["example_accuracy"] for row in group),
                    "max_example_accuracy": max(row["example_accuracy"] for row in group),
                    "mean_token_accuracy": statistics.mean(row["token_accuracy"] for row in group),
                    "best_mask": best["mask"],
                    "best_members": best["members"],
                    "best_example_accuracy": best["example_accuracy"],
                    "best_token_accuracy": best["token_accuracy"],
                    "analysis_role": "DEV MODEL-SELECTION / CALIBRATION",
                }
            )
    return packets


def shapley_values(committee_rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = {0: 0.0}
    for row in committee_rows:
        if row["method"] == method:
            values[int(row["mask"])] = float(row["example_accuracy"])
    factorial = math.factorial
    shapley = {}
    for expert_index, expert in enumerate(EXPERTS):
        contribution = 0.0
        bit = 1 << expert_index
        for subset in range(0, FULL_MASK + 1):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = factorial(size) * factorial(N_EXPERTS - size - 1) / factorial(N_EXPERTS)
            contribution += weight * (values[subset | bit] - values[subset])
        shapley[expert] = contribution
    standalone = {expert: values[1 << index] for index, expert in enumerate(EXPERTS)}
    full_complement_interactions = []
    for left_index, right_index in combinations(range(N_EXPERTS), 2):
        left_bit = 1 << left_index
        right_bit = 1 << right_index
        interaction = (
            values[FULL_MASK]
            - values[FULL_MASK ^ left_bit]
            - values[FULL_MASK ^ right_bit]
            + values[FULL_MASK ^ left_bit ^ right_bit]
        )
        full_complement_interactions.append(
            {
                "expert_a": EXPERTS[left_index],
                "expert_b": EXPERTS[right_index],
                "interaction": interaction,
            }
        )
    standalone_vector = np.asarray([standalone[expert] for expert in EXPERTS])
    shapley_vector = np.asarray([shapley[expert] for expert in EXPERTS])
    correlation = (
        float(np.corrcoef(standalone_vector, shapley_vector)[0, 1])
        if np.std(standalone_vector) > 0 and np.std(shapley_vector) > 0
        else None
    )
    return {
        "method": method,
        "utility": "teacher-forced all-gold-token example accuracy",
        "empty_set_utility": 0.0,
        "full_set_utility": values[FULL_MASK],
        "standalone": standalone,
        "shapley": shapley,
        "standalone_shapley_pearson": correlation,
        "full_complement_pair_interactions": full_complement_interactions,
    }


def derive_ablation_rows(committee_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["method"], row["mask"]): row for row in committee_rows}
    packets = []
    for method in METHODS:
        full = by_key[(method, FULL_MASK)]
        for index, expert in enumerate(EXPERTS):
            reduced = by_key[(method, FULL_MASK ^ (1 << index))]
            packets.append(
                {
                    "method": method,
                    "removed_expert": expert,
                    "full_example_accuracy": full["example_accuracy"],
                    "leave_one_out_example_accuracy": reduced["example_accuracy"],
                    "full_minus_leave_one_out": full["example_accuracy"] - reduced["example_accuracy"],
                    "full_token_accuracy": full["token_accuracy"],
                    "leave_one_out_token_accuracy": reduced["token_accuracy"],
                    "full_minus_leave_one_out_token": full["token_accuracy"] - reduced["token_accuracy"],
                }
            )
    return packets


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_report(packet: dict[str, Any], path: Path) -> None:
    calibration = packet["calibration"]
    committee_rows = packet["committee_results"]
    scaling = packet["committee_scaling"]
    by_key = {(row["method"], row["mask"]): row for row in committee_rows}
    diversity = packet["expert_error_diversity"]
    lines = [
        "# Phase 2 Teacher-Forced DEV Analysis",
        "",
        f"**Status: {packet['status']}**",
        "",
        f"This analysis uses **{packet['example_count']:,} DEV examples / {packet['token_count']:,} gold next-token decisions**. All temperatures, committee choices, and comparator rankings here are `DEV MODEL-SELECTION / CALIBRATION`; they are not TEST confirmation.",
        "",
        "## Expert calibration",
        "",
        "| Expert | Fitted T | Top-1 acc. | Mean confidence | Conf.−acc. | ECE-15 | NLL | Brier | Entropy | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for expert in EXPERTS:
        row = calibration["experts"][expert]
        lines.append(
            f"| {expert} | {row['temperature']:.2f} | {percent(row['top1_accuracy'])} | {percent(row['mean_confidence'])} | "
            f"{100 * row['confidence_minus_accuracy']:+.2f} pp | {row['ece_15']:.4f} | {row['mean_nll']:.4f} | "
            f"{row['mean_brier']:.4f} | {row['mean_entropy']:.4f} | {row['mean_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"The shared NLL-optimal grid temperature is **{calibration['shared_temperature']:.2f}**. Per-expert temperatures are fitted independently on the same DEV capture and must be frozen before any new TEST evaluation.",
            "",
            "## Full K=7 local-decision comparators",
            "",
            "| Method | Token acc. | All-token example acc. | First-token acc. | Answer-token acc. | EOS acc. | Candidate certification |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = by_key[(method, FULL_MASK)]
        lines.append(
            f"| {method} | {percent(row['token_accuracy'])} | {percent(row['example_accuracy'])} | "
            f"{percent(row['first_token_accuracy'])} | {percent(row['answer_token_accuracy'])} | "
            f"{percent(row['eos_accuracy'])} | {percent(row['candidate_argmax_certification_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Candidate certification proves that the selected probability/logit winner beats the maximum possible score of every token omitted from the top-k union. A rate below 100% marks positions whose soft/logit winner is approximate and requires a larger cache top-k before confirmation.",
            "",
            "## Gold-prefix complementarity and error dependence",
            "",
            f"- Dynamic per-token oracle accuracy: **{percent(diversity['dynamic_token_oracle_accuracy'])}** (gain over best individual: **{100 * diversity['oracle_token_gain_over_best_individual']:+.2f} pp**).",
            f"- Dynamic oracle all-token example accuracy: **{percent(diversity['dynamic_token_oracle_all_token_example_accuracy'])}**.",
            f"- All seven wrong at the same gold-prefix token: **{percent(diversity['unanimous_wrong_rate'])}**; all seven correct: **{percent(diversity['unanimous_correct_rate'])}**.",
            "",
            "| Expert pair | Top-1 disagreement | Complementary correctness | Double fault | Correctness phi | Q statistic |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        diversity["pairwise"], key=lambda item: (-item["top1_disagreement_rate"], item["expert_a"], item["expert_b"])
    ):
        phi = "NA" if row["correctness_phi"] is None else f"{row['correctness_phi']:.3f}"
        q_value = "NA" if row["q_statistic"] is None else f"{row['q_statistic']:.3f}"
        lines.append(
            f"| {row['expert_a']} / {row['expert_b']} | {percent(row['top1_disagreement_rate'])} | "
            f"{percent(row['complementary_correct_rate'])} | {percent(row['double_fault_rate'])} | {phi} | {q_value} |"
        )
    lines.extend(
        [
            "",
            "The dynamic oracle is an unattainable switching upper bound. Pairwise disagreement can reflect useful complementarity or shared low quality; committee-diversity correlations are therefore stored by fixed K and treated as exploratory, not as a causal specialization test.",
            "",
            "## Committee-size scaling",
            "",
            "| Method | K | Committees | Mean example acc. | Best example acc. | Best members | Mean token acc. |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for method in ("hard_plurality", "probability_mean", "calibrated_probability_mean", "borda_top5"):
        for row in (item for item in scaling if item["method"] == method):
            lines.append(
                f"| {method} | {row['k']} | {row['committee_count']} | {percent(row['mean_example_accuracy'])} | "
                f"{percent(row['best_example_accuracy'])} | {', '.join(row['best_members'])} | {percent(row['mean_token_accuracy'])} |"
            )
    lines.extend(
        [
            "",
            "## Hard-K7 leave-one-out",
            "",
            "| Removed expert | Full example acc. | K6 example acc. | Full−K6 | Full token acc. | K6 token acc. | Full−K6 token |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in packet["leave_one_out"]:
        if row["method"] != "hard_plurality":
            continue
        lines.append(
            f"| {row['removed_expert']} | {percent(row['full_example_accuracy'])} | "
            f"{percent(row['leave_one_out_example_accuracy'])} | {100 * row['full_minus_leave_one_out']:+.2f} pp | "
            f"{percent(row['full_token_accuracy'])} | {percent(row['leave_one_out_token_accuracy'])} | "
            f"{100 * row['full_minus_leave_one_out_token']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Exact Shapley values on cached committees",
            "",
            "| Expert | Hard standalone | Hard Shapley | Probability standalone | Probability Shapley |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    hard_shapley = packet["shapley"]["hard_plurality"]
    probability_shapley = packet["shapley"]["probability_mean"]
    for expert in EXPERTS:
        lines.append(
            f"| {expert} | {percent(hard_shapley['standalone'][expert])} | {100 * hard_shapley['shapley'][expert]:+.2f} pp | "
            f"{percent(probability_shapley['standalone'][expert])} | {100 * probability_shapley['shapley'][expert]:+.2f} pp |"
        )
    raw_full = by_key[("probability_mean", FULL_MASK)]
    calibrated_full = by_key[("calibrated_probability_mean", FULL_MASK)]
    hard_full = by_key[("hard_plurality", FULL_MASK)]
    lines.extend(
        [
            "",
            "## Mechanistic reading permitted by this DEV cache",
            "",
            f"- Hard K=7 local token accuracy: **{percent(hard_full['token_accuracy'])}**; raw probability mean: **{percent(raw_full['token_accuracy'])}**; calibrated probability mean: **{percent(calibrated_full['token_accuracy'])}**.",
            f"- Hard K=7 all-gold-token example accuracy: **{percent(hard_full['example_accuracy'])}**; raw probability mean: **{percent(raw_full['example_accuracy'])}**; calibrated probability mean: **{percent(calibrated_full['example_accuracy'])}**.",
            "- Positive-temperature hard top-1 voting is mathematically invariant. The stored temperature interventions quantify how soft fusion changes when either all experts or one expert is rescaled.",
            "- This cache establishes local next-token behavior only. A trajectory-correction conclusion still requires same-split free-running DEV evaluation of frozen methods.",
            "",
            "## Provenance",
            "",
            f"- Cache SHA256: `{packet['provenance']['cache_sha256']}`",
            f"- Capture manifest SHA256: `{packet['provenance']['manifest_sha256']}`",
            f"- Analysis script SHA256: `{packet['provenance']['analysis_script_sha256']}`",
            f"- Top-k: **{packet['top_k']}**",
            f"- Aggregated soft/logit decisions certified: **{packet['all_selected_soft_positions_certified']}**",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the compressed Phase 2 teacher-forced DEV cache.")
    parser.add_argument("--expected-examples", type=int, default=4050)
    args = parser.parse_args()
    manifest = read_json(MANIFEST)
    rows_seen: set[str] = set()
    top_k_values: set[int] = set()
    config_hashes: set[str] = set()
    for row in iter_jsonl(CACHE):
        example_id = str(row["example_id"])
        if example_id in rows_seen:
            raise RuntimeError(f"Duplicate cache example: {example_id}")
        rows_seen.add(example_id)
        top_k_values.add(int(row["top_k"]))
        config_hashes.add(str(row["capture_config_sha256"]))
    if config_hashes != {manifest["capture_config_sha256"]} or len(top_k_values) != 1:
        raise RuntimeError("Cache rows do not match the frozen capture manifest")
    calibration = fit_temperatures(manifest)
    committee_rows, interventions = aggregate_all_subsets(calibration)
    diversity = expert_error_diversity()
    diversity_associations = attach_committee_diversity(committee_rows, diversity)
    scaling = committee_scaling(committee_rows)
    leave_one_out = derive_ablation_rows(committee_rows)
    shapley = {
        method: shapley_values(committee_rows, method)
        for method in ("hard_plurality", "probability_mean", "calibrated_probability_mean", "borda_top5")
    }
    selected_soft_methods = {
        "probability_mean",
        "calibrated_probability_mean",
        "shared_temperature_probability_mean",
        "logit_mean",
        "normalized_logit_mean",
    }
    all_certified = all(
        math.isclose(float(row["candidate_argmax_certification_rate"]), 1.0, abs_tol=1e-15)
        for row in committee_rows
        if row["method"] in selected_soft_methods
    )
    if len(rows_seen) > args.expected_examples:
        raise RuntimeError(
            f"Cache has {len(rows_seen)} examples, exceeding --expected-examples={args.expected_examples}"
        )
    # A smoke-sized expectation is useful for exercising the analyzer, but it
    # must never produce a paper-facing "complete" artifact.
    complete = len(rows_seen) == args.expected_examples == 4_050
    status = (
        "DEV_TEACHER_FORCED_ANALYSIS_COMPLETE"
        if complete and all_certified
        else "DEV_TEACHER_FORCED_ANALYSIS_COMPLETE_WITH_UNCERTIFIED_SOFT_POSITIONS"
        if complete
        else "DEV_TEACHER_FORCED_ANALYSIS_PARTIAL"
    )
    packet = {
        "schema_version": "1.0",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MODEL-SELECTION / CALIBRATION",
        "example_count": len(rows_seen),
        "expected_examples": args.expected_examples,
        "token_count": calibration["token_count"],
        "top_k": next(iter(top_k_values)),
        "all_selected_soft_positions_certified": all_certified,
        "calibration": calibration,
        "committee_results": committee_rows,
        "committee_scaling": scaling,
        "expert_error_diversity": diversity,
        "committee_diversity_associations": diversity_associations,
        "leave_one_out": leave_one_out,
        "shapley": shapley,
        "temperature_interventions": interventions,
        "provenance": {
            "cache": str(CACHE),
            "cache_sha256": sha256_file(CACHE),
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "capture_config_sha256": manifest["capture_config_sha256"],
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": [
            "DEV results select calibration and confirmatory candidates; they are not TEST confirmation.",
            "Tokens within an example are correlated; point estimates are reported for committee exploration.",
            "All-token example accuracy under teacher forcing is not free-running exact-match accuracy.",
            "Soft/logit decisions are exact only where the top-k union certificate passes.",
            "Same-task random-seed diversity remains unavailable.",
        ],
    }
    output = FINAL_JSON if complete else PARTIAL_JSON
    report = REPORT if complete else PARTIAL_REPORT
    output.write_text(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(COMMITTEE_CSV if complete else OUTPUT_ROOT / "teacher_forced_committee_results_partial.csv", committee_rows)
    write_csv(SCALING_CSV if complete else OUTPUT_ROOT / "teacher_forced_committee_scaling_partial.csv", scaling)
    write_csv(
        DIVERSITY_CSV if complete else OUTPUT_ROOT / "teacher_forced_expert_diversity_partial.csv",
        diversity["pairwise"],
    )
    make_report(packet, report)
    by_key = {(row["method"], row["mask"]): row for row in committee_rows}
    print(
        json.dumps(
            {
                "status": status,
                "examples": len(rows_seen),
                "tokens": calibration["token_count"],
                "all_selected_soft_positions_certified": all_certified,
                "fitted_temperatures": {
                    expert: calibration["experts"][expert]["temperature"] for expert in EXPERTS
                },
                "full_k7": {
                    method: {
                        "token_accuracy": by_key[(method, FULL_MASK)]["token_accuracy"],
                        "example_accuracy": by_key[(method, FULL_MASK)]["example_accuracy"],
                    }
                    for method in METHODS
                },
                "output": str(output),
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
