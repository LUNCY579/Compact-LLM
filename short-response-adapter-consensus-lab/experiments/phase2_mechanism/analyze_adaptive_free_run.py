from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
HARD_RUN = ROOT / "results" / "runs" / "phase2_selected_dev_free_run_dev_full.jsonl"
ADAPTIVE_RUN = ROOT / "results" / "runs" / "phase2_selected_adaptive_dev_free_run_dev_full.jsonl"
HARD_MANIFEST = ROOT / "artifacts" / "phase2_mechanism" / "selected_dev_free_run_manifest.json"
ADAPTIVE_MANIFEST = ROOT / "artifacts" / "phase2_mechanism" / "selected_adaptive_dev_manifest.json"
OUTPUT = ROOT / "results" / "phase2_mechanism" / "adaptive_free_run_dev_analysis.json"
SUMMARY_CSV = ROOT / "results" / "phase2_mechanism" / "adaptive_free_run_dev_summary.csv"
REPORT = ROOT / "reports" / "PHASE2_ADAPTIVE_FREE_RUNNING_DEV.md"

EXPECTED_EXAMPLES = 4050
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260813


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_method_rows(
    path: Path,
    *,
    method: str,
    expected_config_hash: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL line {line_number} in {path}: {exc}") from exc
            if row.get("config_sha256") != expected_config_hash:
                raise RuntimeError(
                    f"Config hash mismatch at {path}:{line_number}: "
                    f"{row.get('config_sha256')} != {expected_config_hash}"
                )
            if row.get("method") != method:
                continue
            example_id = str(row["example_id"])
            if example_id in records:
                raise RuntimeError(f"Duplicate {method}/{example_id} in {path}")
            records[example_id] = row
    return records


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        batch = min(1000, BOOTSTRAP_REPLICATES - start)
        sampled = rng.integers(0, len(values), size=(batch, len(values)))
        estimates[start : start + batch] = values[sampled].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: list[str],
    rng: np.random.Generator,
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


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def summarize(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(int(row["generated_tokens"]) for row in rows)
    total_forwards = sum(int(row["forward_passes"]) for row in rows)
    total_triggers = sum(int(row.get("adaptive_trigger_count") or 0) for row in rows)
    answer = [row for row in rows if row["decision"] == "answer"]
    abstain = [row for row in rows if row["decision"] == "abstain"]
    return {
        "condition": label,
        "n": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "answerable_n": len(answer),
        "answerable_accuracy": float(np.mean([bool(row["correct"]) for row in answer])),
        "abstention_n": len(abstain),
        "abstention_accuracy": float(np.mean([bool(row["correct"]) for row in abstain])),
        "mean_output_tokens": total_tokens / len(rows),
        "mean_latency_seconds": float(np.mean([float(row["latency_seconds"]) for row in rows])),
        "mean_ttft_seconds": float(
            np.mean([float(row["time_to_first_final_token_seconds"]) for row in rows])
        ),
        "total_forward_passes": total_forwards,
        "mean_forward_passes": total_forwards / len(rows),
        "mean_forwards_per_generated_token": total_forwards / total_tokens,
        "adaptive_trigger_count": total_triggers,
        "adaptive_trigger_token_rate": total_triggers / total_tokens,
        "adaptive_trigger_example_rate": float(
            np.mean([int(row.get("adaptive_trigger_count") or 0) > 0 for row in rows])
        ),
        "peak_gpu_vram_bytes": max(int(row.get("peak_gpu_vram_bytes") or 0) for row in rows),
        "peak_cpu_rss_bytes": max(int(row.get("peak_cpu_rss_bytes") or 0) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Validate available rows and print counts; do not write paper artifacts.",
    )
    args = parser.parse_args()

    hard_manifest = load_json(HARD_MANIFEST)
    adaptive_manifest = load_json(ADAPTIVE_MANIFEST)
    hard = read_method_rows(
        HARD_RUN,
        method="hard_plurality",
        expected_config_hash=hard_manifest["output_config_canonical_sha256"],
    )
    adaptive = read_method_rows(
        ADAPTIVE_RUN,
        method="adaptive_hard_plurality",
        expected_config_hash=adaptive_manifest["output_config_canonical_sha256"],
    )
    if args.allow_partial:
        common = set(hard) & set(adaptive)
        print(json.dumps({"hard_plurality": len(hard), "adaptive_hard_plurality": len(adaptive), "common": len(common)}, indent=2))
        return
    if len(hard) != EXPECTED_EXAMPLES or len(adaptive) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Runs are incomplete: hard={len(hard)}, adaptive={len(adaptive)}; "
            f"expected {EXPECTED_EXAMPLES} each"
        )
    if set(hard) != set(adaptive):
        raise RuntimeError("Hard and adaptive example-ID sets differ")

    ids = sorted(hard)
    for example_id in ids:
        identity = ("gold", "decision", "prompt_hash", "reasoning_family")
        if any(hard[example_id].get(key) != adaptive[example_id].get(key) for key in identity):
            raise RuntimeError(f"Cross-run identity mismatch for {example_id}")

    hard_rows = [hard[example_id] for example_id in ids]
    adaptive_rows = [adaptive[example_id] for example_id in ids]
    summaries = [summarize("hard_plurality", hard_rows), summarize("adaptive_k3_to_k7", adaptive_rows)]
    hard_summary, adaptive_summary = summaries

    hard_correct = np.array([bool(row["correct"]) for row in hard_rows])
    adaptive_correct = np.array([bool(row["correct"]) for row in adaptive_rows])
    differences = adaptive_correct.astype(float) - hard_correct.astype(float)
    adaptive_only = int(np.sum(adaptive_correct & ~hard_correct))
    hard_only = int(np.sum(hard_correct & ~adaptive_correct))
    discordant = adaptive_only + hard_only
    p_value = float(binomtest(min(adaptive_only, hard_only), discordant, 0.5).pvalue) if discordant else 1.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ci_low, ci_high = bootstrap_mean_ci(differences, rng)
    cluster_low, cluster_high = cluster_bootstrap_ci(
        differences,
        [str(row["reasoning_family"]) for row in hard_rows],
        rng,
    )

    divergence_positions: dict[str, int] = defaultdict(int)
    changed = 0
    for hard_row, adaptive_row in zip(hard_rows, adaptive_rows, strict=True):
        divergence = first_divergence(hard_row["output_token_ids"], adaptive_row["output_token_ids"])
        if divergence is None:
            divergence_positions["same"] += 1
        else:
            changed += 1
            bucket = "t=0" if divergence == 0 else "t=1" if divergence == 1 else "t=2" if divergence == 2 else "t=3-5" if divergence <= 5 else "t=6-10" if divergence <= 10 else "t>10"
            divergence_positions[bucket] += 1

    strata: dict[str, Any] = {}
    for decision in ("answer", "abstain"):
        selected = [index for index, row in enumerate(hard_rows) if row["decision"] == decision]
        strata[decision] = {
            "n": len(selected),
            "hard_correct": int(np.sum(hard_correct[selected])),
            "adaptive_correct": int(np.sum(adaptive_correct[selected])),
            "difference_adaptive_minus_hard": float(np.mean(differences[selected])),
        }

    paired = {
        "n": EXPECTED_EXAMPLES,
        "hard_accuracy": hard_summary["accuracy"],
        "adaptive_accuracy": adaptive_summary["accuracy"],
        "difference_adaptive_minus_hard": float(np.mean(differences)),
        "paired_bootstrap_ci95_low": ci_low,
        "paired_bootstrap_ci95_high": ci_high,
        "reasoning_family_cluster_ci95_low": cluster_low,
        "reasoning_family_cluster_ci95_high": cluster_high,
        "both_correct": int(np.sum(hard_correct & adaptive_correct)),
        "adaptive_only_correct": adaptive_only,
        "hard_only_correct": hard_only,
        "both_wrong": int(np.sum(~hard_correct & ~adaptive_correct)),
        "mcnemar_exact_p": p_value,
        "strata": strata,
        "changed_sequence_count": changed,
        "changed_sequence_rate": changed / EXPECTED_EXAMPLES,
        "first_divergence_buckets": dict(divergence_positions),
    }
    efficiency = {
        "forward_pass_reduction_vs_hard": 1.0 - adaptive_summary["total_forward_passes"] / hard_summary["total_forward_passes"],
        "mean_forward_pass_difference": adaptive_summary["mean_forward_passes"] - hard_summary["mean_forward_passes"],
        "latency_ratio_adaptive_over_hard": adaptive_summary["mean_latency_seconds"] / hard_summary["mean_latency_seconds"],
        "latency_difference_seconds": adaptive_summary["mean_latency_seconds"] - hard_summary["mean_latency_seconds"],
        "teacher_forced_selected_policy": adaptive_manifest["selected_teacher_forced_policy"],
    }
    analysis = {
        "schema_version": "1.0",
        "status": "DEV_ADAPTIVE_FREE_RUNNING_ANALYSIS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV EFFICIENCY CONFIRMATION / SAME-SPLIT",
        "policy": {
            "core": adaptive_manifest["core"],
            "threshold": adaptive_manifest["selected_threshold"],
            "threshold_semantics": adaptive_manifest["threshold_semantics"],
            "selection_rule": adaptive_manifest["selection_rule"],
        },
        "provenance": {
            "hard_run": str(HARD_RUN),
            "hard_run_sha256": sha256(HARD_RUN),
            "adaptive_run": str(ADAPTIVE_RUN),
            "adaptive_run_sha256": sha256(ADAPTIVE_RUN),
            "hard_manifest": str(HARD_MANIFEST),
            "hard_manifest_sha256": sha256(HARD_MANIFEST),
            "adaptive_manifest": str(ADAPTIVE_MANIFEST),
            "adaptive_manifest_sha256": sha256(ADAPTIVE_MANIFEST),
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "summaries": summaries,
        "paired_accuracy": paired,
        "efficiency": efficiency,
        "limitations": [
            "The threshold was selected on teacher-forced DEV and is evaluated free-running on the same split.",
            "The paired accuracy test is descriptive mechanism confirmation, not untouched-TEST inference.",
            "Latency is hardware-, decoder-, process-order-, and thermal-state-specific; forward passes are the cleaner compute proxy.",
            "Reasoning-family cluster intervals resample a small set of observed families and are descriptive.",
        ],
    }
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report = "\n".join(
        [
            "# Phase 2 Adaptive Free-Running DEV Analysis",
            "",
            "Role: **DEV EFFICIENCY CONFIRMATION / SAME-SPLIT**. The K3 core, K7 escalation set, and threshold were frozen before these outputs were inspected.",
            "",
            "| Condition | Correct | Exact acc. | Mean forwards | Forwards/token | Mean latency (s) |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row['condition']} | {row['correct']}/{row['n']} | {100*row['accuracy']:.2f}% | {row['mean_forward_passes']:.2f} | {row['mean_forwards_per_generated_token']:.2f} | {row['mean_latency_seconds']:.3f} |"
                for row in summaries
            ],
            "",
            f"Adaptive minus hard accuracy: **{100*paired['difference_adaptive_minus_hard']:+.2f} pp** "
            f"(paired bootstrap 95% CI [{100*ci_low:+.2f}, {100*ci_high:+.2f}] pp; "
            f"adaptive-only/hard-only {adaptive_only}/{hard_only}; exact McNemar p={p_value:.3e}).",
            "",
            f"The adaptive policy reduced measured expert-token forwards by **{100*efficiency['forward_pass_reduction_vs_hard']:.2f}%**. "
            f"Consultation was triggered on {100*adaptive_summary['adaptive_trigger_token_rate']:.2f}% of generated token decisions and in "
            f"{100*adaptive_summary['adaptive_trigger_example_rate']:.2f}% of examples.",
            "",
            "Latency is reported as an environment-specific observation; forward-pass reduction is the primary efficiency result.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
