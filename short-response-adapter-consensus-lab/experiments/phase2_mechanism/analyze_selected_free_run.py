from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

try:
    from experiments.phase2_mechanism.make_selected_runtime_boundaries import (
        OUTPUT as RUNTIME_BOUNDARIES,
        build_manifest as build_runtime_boundaries,
    )
except ModuleNotFoundError:  # Direct ``python experiments/...`` execution.
    from make_selected_runtime_boundaries import (
        OUTPUT as RUNTIME_BOUNDARIES,
        build_manifest as build_runtime_boundaries,
    )


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "runs" / "phase2_selected_dev_free_run_dev_full.jsonl"
SELECTION = ROOT / "artifacts" / "phase2_mechanism" / "selected_dev_free_run_manifest.json"
TEACHER = ROOT / "results" / "phase2_mechanism" / "teacher_forced_dev_analysis.json"
OUTPUT = ROOT / "results" / "phase2_mechanism" / "selected_dev_free_run_analysis.json"
PAIR_CSV = ROOT / "results" / "phase2_mechanism" / "selected_dev_free_run_pairs.csv"
SUMMARY_CSV = ROOT / "results" / "phase2_mechanism" / "selected_dev_free_run_summary.csv"
REPORT = ROOT / "reports" / "PHASE2_FREE_RUNNING_DEV.md"
RANK_PARITY = ROOT / "results" / "phase2_mechanism" / "rank_topk_parity_verification.json"

EXPECTED_EXAMPLES = 4050
METHODS = [
    "hard_plurality",
    "probability_mean",
    "calibrated_probability_mean",
    "reciprocal_rank_top5",
]
METHOD_LABELS = {
    "hard_plurality": "Hard plurality",
    "probability_mean": "Raw probability mean",
    "calibrated_probability_mean": "Calibrated probability mean",
    "reciprocal_rank_top5": "Reciprocal rank top-5",
}
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    estimates: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, n, size=(batch, n))
        estimates.append(values[indices].mean(axis=1))
        remaining -= batch
    samples = np.concatenate(estimates)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def cluster_bootstrap_ci(
    values: np.ndarray, clusters: list[str], rng: np.random.Generator
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(float(value))
    keys = sorted(grouped)
    sums = np.array([sum(grouped[key]) for key in keys], dtype=float)
    counts = np.array([len(grouped[key]) for key in keys], dtype=float)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for start in range(0, BOOTSTRAP_REPLICATES, 2000):
        batch = min(2000, BOOTSTRAP_REPLICATES - start)
        sampled = rng.integers(0, len(keys), size=(batch, len(keys)))
        estimates[start : start + batch] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[index] = running
    return adjusted


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answerable = [row for row in rows if row["decision"] == "answer"]
    abstention = [row for row in rows if row["decision"] == "abstain"]
    return {
        "method": rows[0]["method"],
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "answerable_n": len(answerable),
        "answerable_correct": sum(bool(row["correct"]) for row in answerable),
        "answerable_accuracy": sum(bool(row["correct"]) for row in answerable) / len(answerable),
        "abstention_n": len(abstention),
        "abstention_correct": sum(bool(row["correct"]) for row in abstention),
        "abstention_accuracy": sum(bool(row["correct"]) for row in abstention) / len(abstention),
        "mean_output_tokens": float(np.mean([row["generated_tokens"] for row in rows])),
        "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in rows])),
        "mean_ttft_seconds": float(np.mean([row["time_to_first_final_token_seconds"] for row in rows])),
        "mean_forward_passes": float(np.mean([row["forward_passes"] for row in rows])),
        "mean_adapter_evaluations": float(np.mean([row["adapter_evaluations"] for row in rows])),
        "mean_ties_per_example": float(np.mean([row["tie_count"] for row in rows])),
        "early_eos_disagreement_rate": float(np.mean([bool(row["early_eos_disagreement"]) for row in rows])),
        "late_eos_disagreement_rate": float(np.mean([bool(row["late_eos_disagreement"]) for row in rows])),
        "eos_completion_rate": float(np.mean([row["stop_reason"] == "eos" for row in rows])),
        "peak_gpu_vram_bytes": max(row["peak_gpu_vram_bytes"] or 0 for row in rows),
        "peak_cpu_rss_bytes": max(row["peak_cpu_rss_bytes"] or 0 for row in rows),
        "cuda_oom_retry_examples": sum(int(row.get("cuda_oom_retry_count") or 0) > 0 for row in rows),
        "cuda_oom_retry_total": sum(int(row.get("cuda_oom_retry_count") or 0) for row in rows),
    }


def paired_comparison(
    hard: dict[str, dict[str, Any]],
    other: dict[str, dict[str, Any]],
    other_method: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = sorted(hard)
    differences = np.array(
        [int(other[example_id]["correct"]) - int(hard[example_id]["correct"]) for example_id in ids],
        dtype=float,
    )
    hard_correct = np.array([bool(hard[example_id]["correct"]) for example_id in ids])
    other_correct = np.array([bool(other[example_id]["correct"]) for example_id in ids])
    both_correct = int(np.sum(hard_correct & other_correct))
    other_only = int(np.sum(~hard_correct & other_correct))
    hard_only = int(np.sum(hard_correct & ~other_correct))
    both_wrong = int(np.sum(~hard_correct & ~other_correct))
    low, high = bootstrap_mean_ci(differences, rng)
    clusters = [hard[example_id]["reasoning_family"] for example_id in ids]
    cluster_low, cluster_high = cluster_bootstrap_ci(differences, clusters, rng)
    discordant = other_only + hard_only
    p_value = float(binomtest(min(other_only, hard_only), discordant, 0.5).pvalue) if discordant else 1.0
    strata: dict[str, Any] = {}
    for decision in ("answer", "abstain"):
        selected = [example_id for example_id in ids if hard[example_id]["decision"] == decision]
        local = np.array(
            [int(other[example_id]["correct"]) - int(hard[example_id]["correct"]) for example_id in selected],
            dtype=float,
        )
        strata[decision] = {
            "n": len(selected),
            "hard_correct": sum(bool(hard[example_id]["correct"]) for example_id in selected),
            "other_correct": sum(bool(other[example_id]["correct"]) for example_id in selected),
            "difference_other_minus_hard": float(local.mean()),
        }
    positions: Counter[str] = Counter()
    outcome_states: Counter[str] = Counter()
    exact_matches = 0
    case_records: list[dict[str, Any]] = []
    for example_id in ids:
        left = list(hard[example_id]["output_token_ids"])
        right = list(other[example_id]["output_token_ids"])
        divergence = first_divergence(left, right)
        if divergence is None:
            exact_matches += 1
            bucket = "same"
        elif divergence == 0:
            bucket = "t=0"
        elif divergence == 1:
            bucket = "t=1"
        elif divergence == 2:
            bucket = "t=2"
        elif divergence <= 5:
            bucket = "t=3-5"
        elif divergence <= 10:
            bucket = "t=6-10"
        else:
            bucket = "t>10"
        positions[bucket] += 1
        state = (
            "both_correct"
            if hard[example_id]["correct"] and other[example_id]["correct"]
            else "hard_only_correct"
            if hard[example_id]["correct"]
            else "other_only_correct"
            if other[example_id]["correct"]
            else "both_wrong"
        )
        outcome_states[state] += 1
        if divergence is not None and state != "both_wrong":
            case_records.append(
                {
                    "example_id": example_id,
                    "decision": hard[example_id]["decision"],
                    "reasoning_family": hard[example_id]["reasoning_family"],
                    "gold": hard[example_id]["gold"],
                    "hard_output": hard[example_id]["output"],
                    "other_output": other[example_id]["output"],
                    "hard_correct": bool(hard[example_id]["correct"]),
                    "other_correct": bool(other[example_id]["correct"]),
                    "first_divergence": divergence,
                    "outcome_state": state,
                }
            )
    case_records.sort(key=lambda row: (row["outcome_state"], row["first_divergence"], row["example_id"]))
    return {
        "comparison": f"{other_method}_vs_hard_plurality",
        "other_method": other_method,
        "n": len(ids),
        "hard_accuracy": float(hard_correct.mean()),
        "other_accuracy": float(other_correct.mean()),
        "difference_other_minus_hard": float(differences.mean()),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "reasoning_family_cluster_ci95_low": cluster_low,
        "reasoning_family_cluster_ci95_high": cluster_high,
        "both_correct": both_correct,
        "other_only_correct": other_only,
        "hard_only_correct": hard_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": p_value,
        "strata": strata,
        "exact_sequence_matches": exact_matches,
        "exact_sequence_match_rate": exact_matches / len(ids),
        "first_divergence_buckets": dict(positions),
        "outcome_states": dict(outcome_states),
        "qualitative_candidates_first_40": case_records[:40],
    }


def validate_and_load(allow_partial: bool) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    selection = load_json(SELECTION)
    expected_hash = selection["output_config_canonical_sha256"]
    records: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in METHODS}
    duplicate_keys: list[tuple[str, str]] = []
    config_hashes: set[str] = set()
    with RUN.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL line {line_number}: {exc}") from exc
            method = row["method"]
            if method not in records:
                raise RuntimeError(f"Unexpected method {method!r} at line {line_number}")
            example_id = row["example_id"]
            if example_id in records[method]:
                duplicate_keys.append((method, example_id))
            records[method][example_id] = row
            config_hashes.add(row["config_sha256"])
    counts = {method: len(rows) for method, rows in records.items()}
    if duplicate_keys:
        raise RuntimeError(f"Duplicate method-example keys: {duplicate_keys[:10]}")
    if config_hashes != {expected_hash}:
        raise RuntimeError(f"Config hash mismatch: found {sorted(config_hashes)}, expected {expected_hash}")
    if not allow_partial and any(count != EXPECTED_EXAMPLES for count in counts.values()):
        raise RuntimeError(f"Run is incomplete: {counts}; expected {EXPECTED_EXAMPLES} per method")
    if allow_partial:
        common = set.intersection(*(set(records[method]) for method in METHODS))
        records = {method: {example_id: records[method][example_id] for example_id in common} for method in METHODS}
    else:
        id_sets = {method: set(rows) for method, rows in records.items()}
        if len({frozenset(ids) for ids in id_sets.values()}) != 1:
            raise RuntimeError("Method example-ID sets differ")
    for example_id in records["hard_plurality"]:
        golds = {records[method][example_id]["gold"] for method in METHODS}
        decisions = {records[method][example_id]["decision"] for method in METHODS}
        prompts = {records[method][example_id]["prompt_hash"] for method in METHODS}
        if len(golds) != 1 or len(decisions) != 1 or len(prompts) != 1:
            raise RuntimeError(f"Cross-method row identity mismatch for {example_id}")
    return records, selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true", help="For code smoke only; no paper artifacts are written")
    args = parser.parse_args()
    records, selection = validate_and_load(args.allow_partial)
    if args.allow_partial:
        print(json.dumps({method: len(rows) for method, rows in records.items()}, indent=2))
        return

    build_runtime_boundaries()
    runtime_boundaries = load_json(RUNTIME_BOUNDARIES)
    teacher = load_json(TEACHER)
    rank_parity = load_json(RANK_PARITY)
    rules_path = ROOT / "src" / "consensus_lab" / "voting" / "rules.py"
    direct_path = ROOT / "src" / "consensus_lab" / "evaluation" / "direct.py"
    decoder_path = ROOT / "src" / "consensus_lab" / "decoding" / "decoder.py"
    if rank_parity.get("status") != "RANK_TOPK_IMPLEMENTATION_PARITY_VERIFIED":
        raise RuntimeError("Rank top-k implementation parity is not verified")
    if rank_parity.get("provenance", {}).get("voting_rules_sha256") != sha256(rules_path):
        raise RuntimeError("Rank top-k parity artifact does not bind the current voting implementation")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries = [summarize_method([records[method][example_id] for example_id in sorted(records[method])]) for method in METHODS]
    hard = records["hard_plurality"]
    comparisons = [paired_comparison(hard, records[method], method, rng) for method in METHODS[1:]]
    adjusted = holm_adjust([comparison["mcnemar_exact_p"] for comparison in comparisons])
    for comparison, p_value in zip(comparisons, adjusted, strict=True):
        comparison["holm_adjusted_mcnemar_p"] = p_value

    teacher_rows = selection["selected_teacher_forced_rows"]
    summary_by_method = {row["method"]: row for row in summaries}
    teacher_free = []
    for method in METHODS:
        teacher_row = teacher_rows[method]
        free_row = summary_by_method[method]
        teacher_free.append(
            {
                "method": method,
                "teacher_forced_token_accuracy": teacher_row["token_accuracy"],
                "teacher_forced_all_token_example_accuracy": teacher_row["example_accuracy"],
                "free_running_exact_accuracy": free_row["accuracy"],
                "teacher_example_advantage_vs_hard": teacher_row["example_accuracy"] - teacher_rows["hard_plurality"]["example_accuracy"],
                "free_sequence_advantage_vs_hard": free_row["accuracy"] - summary_by_method["hard_plurality"]["accuracy"],
                "difference_of_advantages": (
                    free_row["accuracy"] - summary_by_method["hard_plurality"]["accuracy"]
                ) - (
                    teacher_row["example_accuracy"] - teacher_rows["hard_plurality"]["example_accuracy"]
                ),
            }
        )

    analysis = {
        "schema_version": "1.0",
        "status": "DEV_SELECTED_FREE_RUNNING_ANALYSIS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MECHANISM CONFIRMATION / SAME-SPLIT",
        "provenance": {
            "run": str(RUN),
            "run_sha256": sha256(RUN),
            "selection_manifest": str(SELECTION),
            "selection_manifest_sha256": sha256(SELECTION),
            "teacher_forced_analysis": str(TEACHER),
            "teacher_forced_analysis_sha256": sha256(TEACHER),
            "canonical_config_sha256": selection["output_config_canonical_sha256"],
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "voting_rules": str(rules_path),
            "voting_rules_sha256": sha256(rules_path),
            "direct_evaluator": str(direct_path),
            "direct_evaluator_sha256": sha256(direct_path),
            "reference_decoder": str(decoder_path),
            "reference_decoder_sha256": sha256(decoder_path),
            "rank_topk_parity": str(RANK_PARITY),
            "rank_topk_parity_sha256": sha256(RANK_PARITY),
            "runtime_boundaries": str(RUNTIME_BOUNDARIES),
            "runtime_boundaries_sha256": sha256(RUNTIME_BOUNDARIES),
        },
        "example_count": EXPECTED_EXAMPLES,
        "methods": METHODS,
        "method_summaries": summaries,
        "paired_family": {
            "definition": "each selected comparator versus hard plurality; exact McNemar with Holm correction",
            "comparisons": comparisons,
        },
        "teacher_forced_vs_free_running": teacher_free,
        "limitations": [
            "DEV was used to fit temperatures and select the rank representative; these are mechanism-confirmation, not untouched-test, results.",
            "Teacher-forced all-gold-token example accuracy and free-running exact sequence accuracy answer different questions.",
            "Difference-of-advantages is descriptive and should not be interpreted as a common-scale causal decomposition.",
            "Latency is hardware- and reference-decoder-specific.",
            "CUDA OOM retries, when present, are recomputations before durable append; successful-generation latency excludes failed allocation attempts.",
        ],
        "runtime_boundaries": runtime_boundaries["boundaries"],
        "oom_retry_telemetry": runtime_boundaries["oom_retry_telemetry"],
    }
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    flat_pairs = []
    for comparison in comparisons:
        flat_pairs.append({key: value for key, value in comparison.items() if not isinstance(value, (dict, list))})
    with PAIR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_pairs[0]))
        writer.writeheader()
        writer.writerows(flat_pairs)

    summary_lines = []
    for row in summaries:
        summary_lines.append(
            f'| {METHOD_LABELS[row["method"]]} | {row["correct"]}/{row["n"]} | {100*row["accuracy"]:.2f}% | '
            f'{100*row["answerable_accuracy"]:.2f}% | {100*row["abstention_accuracy"]:.2f}% | '
            f'{row["mean_latency_seconds"]:.3f} | {row["mean_forward_passes"]:.2f} |'
        )
    pair_lines = []
    for row in comparisons:
        pair_lines.append(
            f'| {METHOD_LABELS[row["other_method"]]} − Hard | {100*row["difference_other_minus_hard"]:+.2f} pp | '
            f'[{100*row["bootstrap_ci95_low"]:+.2f}, {100*row["bootstrap_ci95_high"]:+.2f}] | '
            f'{row["other_only_correct"]}/{row["hard_only_correct"]} | {row["mcnemar_exact_p"]:.3e} | '
            f'{row["holm_adjusted_mcnemar_p"]:.3e} |'
        )
    trajectory_lines = []
    for row in teacher_free:
        trajectory_lines.append(
            f'| {METHOD_LABELS[row["method"]]} | {100*row["teacher_forced_token_accuracy"]:.2f}% | '
            f'{100*row["teacher_forced_all_token_example_accuracy"]:.2f}% | '
            f'{100*row["free_running_exact_accuracy"]:.2f}% | '
            f'{100*row["teacher_example_advantage_vs_hard"]:+.2f} pp | '
            f'{100*row["free_sequence_advantage_vs_hard"]:+.2f} pp |'
        )
    report = "\n".join(
        [
            "# Phase 2 Selected Free-Running DEV Analysis",
            "",
            "Role: **DEV MECHANISM CONFIRMATION / SAME-SPLIT**. Methods and temperatures were frozen before these free-running outputs were inspected.",
            "",
            "## Free-running results",
            "",
            "| Method | Correct | Exact acc. | Answerable | Abstention | Latency (s) | Forwards |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *summary_lines,
            "",
            "## Paired family versus hard plurality",
            "",
            "| Difference | Effect | Paired bootstrap 95% CI | Other-only/Hard-only | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
            *pair_lines,
            "",
            "## Teacher-forced versus free-running",
            "",
            "| Method | Local token | Local all-token example | Free exact | Local example gap vs Hard | Free gap vs Hard |",
            "|---|---:|---:|---:|---:|---:|",
            *trajectory_lines,
            "",
            "The two regimes use the same DEV examples but different estimands. The comparison tests whether the local ordering survives path construction; it is not a subtraction of commensurate accuracy metrics.",
            "",
            "Exact provenance and detailed divergence summaries are in `results/phase2_mechanism/selected_dev_free_run_analysis.json`.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
