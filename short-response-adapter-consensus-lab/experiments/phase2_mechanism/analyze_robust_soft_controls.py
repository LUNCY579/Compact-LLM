from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl"
TEACHER = ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json"
OUTPUT = ROOT / "results/phase2_mechanism/robust_soft_controls_dev.json"
SUMMARY_CSV = ROOT / "results/phase2_mechanism/robust_soft_controls_dev.csv"
REPORT = ROOT / "reports/PHASE2_ROBUST_SOFT_CONTROLS_DEV.md"
SOURCE_CONFIG = ROOT / "configs/phase2_selected_dev_free_run.yaml"

EXPECTED_EXAMPLES = 4050
EXPECTED_TOKENS = 18760
EXPERT_COUNT = 7
CLIP_CAPS = (0.5, 0.7, 0.9)
METHODS = (
    "hard_plurality",
    "probability_mean",
    "validation_weighted_hard",
    "entropy_normalized_probability_mean",
    "trimmed_probability_mean",
    "median_probability",
    *(f"confidence_clipped_probability_mean_{cap:g}" for cap in CLIP_CAPS),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_winner(scores: np.ndarray, candidate_ids: np.ndarray) -> int:
    """Return the smallest token ID among exact score maxima."""
    maximum = float(np.max(scores))
    indices = np.flatnonzero(scores == maximum)
    return int(indices[np.argmin(candidate_ids[indices])])


def hard_winner(
    candidate_ids: np.ndarray,
    top1_ids: np.ndarray,
    raw_log_probabilities: np.ndarray,
    weights: np.ndarray | None = None,
) -> int:
    vote_matrix = (top1_ids[:, None] == candidate_ids[None, :]).astype(np.float64)
    votes = vote_matrix.sum(axis=0) if weights is None else weights @ vote_matrix
    maximum = float(np.max(votes))
    tied = np.isclose(votes, maximum, rtol=1e-12, atol=1e-12)
    secondary = np.where(tied, raw_log_probabilities.sum(axis=0), -np.inf)
    return stable_winner(secondary, candidate_ids)


def clipped_candidate_probabilities(
    probabilities: np.ndarray,
    candidate_ids: np.ndarray,
    top1_ids: np.ndarray,
    top1_probabilities: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip each expert's top-1 mass and exactly renormalize its distribution.

    Caps are at least 0.5, so at most one vocabulary item can exceed a cap.
    The returned normalizers also apply to every outside-union probability.
    """
    if not 0.5 <= cap < 1.0:
        raise ValueError("cap must be in [0.5, 1.0)")
    transformed = probabilities.copy()
    normalizers = 1.0 - top1_probabilities + np.minimum(top1_probabilities, cap)
    index_by_id = {int(token_id): index for index, token_id in enumerate(candidate_ids)}
    for expert_index, token_id in enumerate(top1_ids):
        transformed[expert_index, index_by_id[int(token_id)]] = min(
            float(top1_probabilities[expert_index]), cap
        )
    transformed /= normalizers[:, None]
    return transformed, normalizers


def paired_comparison(
    reference: list[bool], other: list[bool], reference_method: str, other_method: str
) -> dict[str, Any]:
    left = np.asarray(reference, dtype=bool)
    right = np.asarray(other, dtype=bool)
    other_only = int(np.sum(~left & right))
    reference_only = int(np.sum(left & ~right))
    discordant = other_only + reference_only
    p_value = (
        float(binomtest(min(other_only, reference_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "reference_method": reference_method,
        "other_method": other_method,
        "paired_examples": len(left),
        "both_correct": int(np.sum(left & right)),
        "other_only_correct": other_only,
        "reference_only_correct": reference_only,
        "both_wrong": int(np.sum(~left & ~right)),
        "difference_other_minus_reference": float(right.mean() - left.mean()),
        "mcnemar_exact_p": p_value,
        "inference_role": "exploratory DEV; unadjusted descriptive p-value",
    }


def summarize(
    counts: dict[str, dict[str, int]], example_correct: dict[str, list[bool]]
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        item = counts[method]
        rows.append(
            {
                "method": method,
                "token_correct": item["token_correct"],
                "token_total": item["token_total"],
                "token_accuracy": item["token_correct"] / item["token_total"],
                "example_correct": sum(example_correct[method]),
                "example_total": len(example_correct[method]),
                "example_accuracy": sum(example_correct[method]) / len(example_correct[method]),
                "first_token_accuracy": item["first_token_correct"] / item["first_token_total"],
                "answer_token_accuracy": item["answer_token_correct"] / item["answer_token_total"],
                "eos_accuracy": item["eos_correct"] / item["eos_total"],
                "candidate_argmax_certification_rate": item["certified"] / item["token_total"],
                "uncertified_token_count": item["token_total"] - item["certified"],
                "analysis_role": "DEV MECHANISTIC / EXPLORATORY ROBUSTNESS CONTROL",
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float) -> str:
    return f"{100 * value:.3f}%"


def main() -> None:
    teacher = json.loads(TEACHER.read_text(encoding="utf-8-sig"))
    if teacher.get("status") not in {
        "DEV_TEACHER_FORCED_COMPLETE",
        "DEV_TEACHER_FORCED_ANALYSIS_COMPLETE_WITH_UNCERTIFIED_SOFT_POSITIONS",
    }:
        raise RuntimeError("The frozen teacher-forced analysis is not complete")
    baseline_rows = {
        row["method"]: row
        for row in teacher["committee_results"]
        if int(row["mask"]) == 127 and row["method"] in {"hard_plurality", "probability_mean"}
    }
    if set(baseline_rows) != {"hard_plurality", "probability_mean"}:
        raise RuntimeError("Could not find the two full-committee reproduction baselines")
    source_config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source_experts = [str(item["name"]) for item in source_config["experts"]]
    validation_weights = np.asarray(
        [float(item["validation_weight"]) for item in source_config["experts"]],
        dtype=np.float64,
    )
    if np.any(validation_weights <= 0):
        raise RuntimeError("Validation-weighted hard voting requires positive frozen weights")

    counts: dict[str, dict[str, int]] = {
        method: defaultdict(int) for method in METHODS
    }
    examples: dict[str, list[bool]] = {method: [] for method in METHODS}
    example_count = 0
    raw_temp_index: int | None = None
    for row in iter_jsonl(CACHE):
        example_count += 1
        example_ok = {method: True for method in METHODS}
        if len(row["expert_order"]) != EXPERT_COUNT:
            raise RuntimeError(f"Unexpected expert count on {row.get('example_id')}")
        if list(row["expert_order"]) != source_experts:
            raise RuntimeError(f"Expert order differs from the frozen validation weights on {row.get('example_id')}")
        for token_index, token in enumerate(row["tokens"]):
            candidate_ids = np.asarray(token["candidate_token_ids"], dtype=np.int64)
            logits = np.asarray(token["candidate_logits_by_expert"], dtype=np.float64)
            if logits.shape != (EXPERT_COUNT, len(candidate_ids)):
                raise RuntimeError("Candidate-logit shape mismatch")
            metrics = token["expert_metrics"]
            temperature_grid = [float(value) for value in row["temperature_grid"]]
            if raw_temp_index is None:
                raw_temp_index = temperature_grid.index(1.0)
            elif temperature_grid[raw_temp_index] != 1.0:
                raise RuntimeError("Temperature grid changed within the frozen cache")
            lse = np.asarray(
                [item["logsumexp_by_temperature"][raw_temp_index] for item in metrics],
                dtype=np.float64,
            )
            raw_log_probabilities = logits - lse[:, None]
            probabilities = np.exp(raw_log_probabilities)
            top1_ids = np.asarray([item["top1_token_id"] for item in metrics], dtype=np.int64)
            top1_probabilities = np.asarray(
                [item["top1_probability"] for item in metrics], dtype=np.float64
            )
            kth_probabilities = np.exp(
                np.asarray([item["topk_logits"][-1] for item in metrics], dtype=np.float64) - lse
            )
            entropies = np.asarray([item["entropy"] for item in metrics], dtype=np.float64)
            gold = int(token["gold_token_id"])

            winners: dict[str, int] = {
                "hard_plurality": hard_winner(candidate_ids, top1_ids, raw_log_probabilities),
                "probability_mean": stable_winner(probabilities.mean(axis=0), candidate_ids),
                "validation_weighted_hard": hard_winner(
                    candidate_ids, top1_ids, raw_log_probabilities, validation_weights
                ),
            }
            winner_scores: dict[str, float] = {
                "hard_plurality": math.inf,
                "probability_mean": float(np.max(probabilities.mean(axis=0))),
                "validation_weighted_hard": math.inf,
            }
            outside_bounds: dict[str, float] = {
                "hard_plurality": -math.inf,
                "probability_mean": float(kth_probabilities.mean()),
                "validation_weighted_hard": -math.inf,
            }

            inverse_entropy = 1.0 / np.maximum(entropies, 1e-12)
            entropy_weights = inverse_entropy / inverse_entropy.sum()
            entropy_scores = entropy_weights @ probabilities
            entropy_method = "entropy_normalized_probability_mean"
            winners[entropy_method] = stable_winner(entropy_scores, candidate_ids)
            winner_scores[entropy_method] = float(np.max(entropy_scores))
            outside_bounds[entropy_method] = float(entropy_weights @ kth_probabilities)

            trimmed_method = "trimmed_probability_mean"
            trimmed_scores = np.sort(probabilities, axis=0)[1:-1].mean(axis=0)
            winners[trimmed_method] = stable_winner(trimmed_scores, candidate_ids)
            winner_scores[trimmed_method] = float(np.max(trimmed_scores))
            outside_bounds[trimmed_method] = float(np.sort(kth_probabilities)[1:-1].mean())

            median_method = "median_probability"
            median_scores = np.median(probabilities, axis=0)
            winners[median_method] = stable_winner(median_scores, candidate_ids)
            winner_scores[median_method] = float(np.max(median_scores))
            outside_bounds[median_method] = float(np.median(kth_probabilities))

            for cap in CLIP_CAPS:
                method = f"confidence_clipped_probability_mean_{cap:g}"
                clipped, normalizers = clipped_candidate_probabilities(
                    probabilities, candidate_ids, top1_ids, top1_probabilities, cap
                )
                scores = clipped.mean(axis=0)
                winners[method] = stable_winner(scores, candidate_ids)
                winner_scores[method] = float(np.max(scores))
                outside_bounds[method] = float(np.mean(kth_probabilities / normalizers))

            is_eos = token_index == len(row["tokens"]) - 1
            for method in METHODS:
                selected = int(candidate_ids[winners[method]])
                correct = selected == gold
                item = counts[method]
                item["token_total"] += 1
                item["token_correct"] += int(correct)
                item["certified"] += int(winner_scores[method] > outside_bounds[method] + 1e-15)
                example_ok[method] &= correct
                if token_index == 0:
                    item["first_token_total"] += 1
                    item["first_token_correct"] += int(correct)
                if is_eos:
                    item["eos_total"] += 1
                    item["eos_correct"] += int(correct)
                else:
                    item["answer_token_total"] += 1
                    item["answer_token_correct"] += int(correct)
        for method in METHODS:
            examples[method].append(bool(example_ok[method]))

    if example_count != EXPECTED_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_EXAMPLES} examples, observed {example_count}")
    if any(counts[method]["token_total"] != EXPECTED_TOKENS for method in METHODS):
        raise RuntimeError("Unexpected teacher-forced token count")
    for method in ("hard_plurality", "probability_mean"):
        expected = baseline_rows[method]
        observed_token = counts[method]["token_correct"]
        observed_example = sum(examples[method])
        if observed_token != int(expected["token_correct"]) or observed_example != int(expected["example_correct"]):
            raise RuntimeError(
                f"Independent {method} reproduction mismatch: tokens {observed_token} vs "
                f"{expected['token_correct']}; examples {observed_example} vs {expected['example_correct']}"
            )

    summaries = summarize(counts, examples)
    uncertified = {
        row["method"]: row["uncertified_token_count"]
        for row in summaries
        if row["uncertified_token_count"]
    }
    if set(uncertified) - {"median_probability"}:
        raise RuntimeError(f"Unexpected candidate-union certification failures: {uncertified}")
    comparisons = []
    for method in METHODS[2:]:
        comparisons.append(paired_comparison(examples["hard_plurality"], examples[method], "hard_plurality", method))
        comparisons.append(paired_comparison(examples["probability_mean"], examples[method], "probability_mean", method))

    packet = {
        "schema_version": 1,
        "status": "DEV_ROBUST_SOFT_CONTROLS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MECHANISTIC / EXPLORATORY ROBUSTNESS CONTROL",
        "definition": {
            "entropy_normalization": "Per-token inverse-entropy expert weighting: w_i=(1/H_i)/sum_j(1/H_j), followed by weighted arithmetic probability pooling.",
            "validation_weighted_hard": "Sum the frozen positive validation_weight attached to experts proposing each top-1 token; resolve tied weighted leaders with the deployed all-expert raw-log-probability sum and then smallest token ID.",
            "trimmed_probability_mean": "For each token, discard the largest and smallest of the seven expert probabilities and average the middle five.",
            "median_probability": "For each token, take the median of the seven expert probabilities.",
            "confidence_clipping": "For cap c in {0.5,0.7,0.9}, replace each expert's top-1 probability p with min(p,c), leave other masses unchanged, exactly renormalize by 1-p+min(p,c), then arithmetic-pool experts.",
        },
        "provenance": {
            "cache_path": str(CACHE.relative_to(ROOT)),
            "cache_sha256": sha256(CACHE),
            "teacher_analysis_path": str(TEACHER.relative_to(ROOT)),
            "teacher_analysis_sha256": sha256(TEACHER),
            "validation_weight_config_path": str(SOURCE_CONFIG.relative_to(ROOT)),
            "validation_weight_config_sha256": sha256(SOURCE_CONFIG),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "cache_reused_without_new_model_forwards": True,
        },
        "example_count": example_count,
        "token_count": EXPECTED_TOKENS,
        "baseline_reproduction": {
            method: {
                "token_correct": counts[method]["token_correct"],
                "example_correct": sum(examples[method]),
                "matches_frozen_full_committee_result": True,
            }
            for method in ("hard_plurality", "probability_mean")
        },
        "candidate_union": {
            "top_k_per_expert": 64,
            "all_winners_certified": not uncertified,
            "all_nonmedian_winners_certified": set(uncertified) <= {"median_probability"},
            "uncertified_token_counts": uncertified,
            "outside_bound": "Method-specific monotone aggregation of each expert's 64th-token probability upper bound (mean, weighted mean, trimmed mean, or median); rank/top-1-only winners are in the candidate union by construction. Reported robust-soft winners strictly exceed their bounds.",
        },
        "method_summaries": summaries,
        "paired_example_comparisons": comparisons,
        "limitations": [
            "These controls were added after inspecting DEV and are exploratory, not a new confirmatory family.",
            "Entropy normalization is operationalized specifically as inverse-entropy expert weighting; other entropy transforms were not searched.",
            "Trimmed and median pooling are post-DEV full-K=7 robustness definitions, not a tuned family or a free-running promotion.",
            "Validation weights were pre-existing configuration metadata and are not interpreted as independently held-out causal evidence.",
            "Median pooling has two uncertified candidate-union decisions; its displayed accuracy is the top-64-union approximation and is not promoted or used for an exact claim.",
            "Clip caps are a small robustness grid, not tuned or promoted into the frozen free-running selection.",
            "All outcomes use gold-prefix teacher forcing and do not establish free-running sequence accuracy.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    write_csv(summaries)

    by_method = {row["method"]: row for row in summaries}
    lines = [
        "# Phase 2 robust soft-voting controls on DEV",
        "",
        "Status: **DEV_ROBUST_SOFT_CONTROLS_COMPLETE**",
        "",
        "These are cache-only, full-K=7, gold-prefix DEV robustness controls. They were added after DEV inspection and are not promoted into the frozen free-running selection.",
        "",
        "## Definitions",
        "",
        "- Entropy normalization means inverse-entropy expert weighting, followed by arithmetic probability pooling.",
        "- Validation-weighted hard voting weights top-1 proposals by the frozen per-expert validation weights and retains the deployed confidence tie-break.",
        "- Trimmed probability pooling drops the largest and smallest expert probability per candidate; median pooling takes the seven-expert median.",
        "- Confidence clipping caps each expert's top-1 mass at 0.5, 0.7, or 0.9, exactly renormalizes that expert distribution, then arithmetic-pools experts.",
        "",
        "## Results",
        "",
        "| Method | Token accuracy | All-token example accuracy | First token | Answer token | EOS | Candidate certified |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = by_method[method]
        lines.append(
            f"| {method} | {percent(item['token_accuracy'])} | {percent(item['example_accuracy'])} | "
            f"{percent(item['first_token_accuracy'])} | {percent(item['answer_token_accuracy'])} | "
            f"{percent(item['eos_accuracy'])} | {percent(item['candidate_argmax_certification_rate'])} |"
        )
    lines.extend(
        [
            "",
            "The hard-plurality and raw-probability rows exactly reproduce the frozen full-committee analysis at both token and all-token-example levels. Every winner except two median-pooling positions is certified against tokens outside the cached top-64 union; the median row is therefore an explicitly approximate exploratory result.",
            "",
            "## Paired all-token-example comparisons",
            "",
            "| Reference | Other | Other-only | Reference-only | Difference | Exact McNemar p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['reference_method']} | {item['other_method']} | {item['other_only_correct']} | "
            f"{item['reference_only_correct']} | {100*item['difference_other_minus_reference']:+.3f} pp | "
            f"{item['mcnemar_exact_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            "P-values are unadjusted descriptive diagnostics for exploratory DEV controls. No robust-control variant changes the locked free-running manifest.",
            "",
            "## Provenance and limitations",
            "",
            f"- Frozen cache SHA-256: `{packet['provenance']['cache_sha256']}`.",
            f"- Frozen teacher-analysis SHA-256: `{packet['provenance']['teacher_analysis_sha256']}`.",
            "- No new model forwards were executed.",
            "- Gold-prefix outcomes are not free-running sequence outcomes.",
            "- Median pooling is uncertified at 2/18,760 positions and is not used for exact inference.",
            "- The tested entropy, trimmed, median, validation-weighted, and clipping definitions are explicitly bounded; no post-hoc winner was promoted.",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": packet["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
