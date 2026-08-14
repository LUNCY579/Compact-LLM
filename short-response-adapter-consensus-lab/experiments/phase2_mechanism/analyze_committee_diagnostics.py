from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase2_mechanism.analyze_teacher_forced_cache import (
    EXPERTS,
    FULL_MASK,
    METHODS,
    rank_matrix,
    stable_argmax,
    subset_inventory,
    tie_break_primary,
)


CACHE = ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl"
TEACHER = ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json"
OUTPUT = ROOT / "results/phase2_mechanism/teacher_forced_committee_diagnostics.json"
CSV = ROOT / "results/phase2_mechanism/teacher_forced_committee_diagnostics.csv"
REPORT = ROOT / "reports/PHASE2_COMMITTEE_DIAGNOSTICS_DEV.md"
EXPECTED_EXAMPLES = 4050
EXPECTED_TOKENS = 18760


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


def tie_flags(primary: np.ndarray) -> np.ndarray:
    maximum = primary.max(axis=1, keepdims=True)
    return np.isclose(primary, maximum, rtol=1e-12, atol=1e-12).sum(axis=1) > 1


def normal_mean_ci95(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    radius = 1.959963984540054 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean - radius, mean + radius


def wilson_ci95(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def main() -> None:
    teacher = json.loads(TEACHER.read_text(encoding="utf-8-sig"))
    if teacher.get("example_count") != EXPECTED_EXAMPLES or teacher.get("token_count") != EXPECTED_TOKENS:
        raise RuntimeError("Frozen teacher analysis is not complete")
    baseline = {
        (int(row["mask"]), row["method"]): row for row in teacher["committee_results"]
    }
    masks, subset_matrix, members = subset_inventory()
    subset_sizes = subset_matrix.sum(axis=1)
    subset_count = len(masks)
    grid = [float(value) for value in teacher["calibration"]["temperature_grid"]]
    raw_index = grid.index(1.0)
    fitted_indices = np.asarray(
        [teacher["calibration"]["experts"][expert]["best_temperature_index"] for expert in EXPERTS],
        dtype=np.int64,
    )
    shared_index = int(teacher["calibration"]["shared_temperature_index"])

    token_correct = {method: np.zeros(subset_count, dtype=np.int64) for method in METHODS}
    tie_counts = {method: np.zeros(subset_count, dtype=np.int64) for method in METHODS}
    example_correct = {
        method: {
            "answer": np.zeros(subset_count, dtype=np.int64),
            "abstain": np.zeros(subset_count, dtype=np.int64),
        }
        for method in METHODS
    }
    example_totals = {"answer": 0, "abstain": 0}
    example_outcomes = {
        method: np.zeros((EXPECTED_EXAMPLES, subset_count), dtype=bool) for method in METHODS
    }
    vote_entropy_sum = np.zeros(subset_count, dtype=np.float64)
    examples = 0
    tokens = 0

    for row in iter_jsonl(CACHE):
        examples += 1
        decision = str(row["decision"])
        if decision not in example_totals:
            raise RuntimeError(f"Unexpected decision class {decision!r}")
        example_totals[decision] += 1
        example_ok = {method: np.ones(subset_count, dtype=bool) for method in METHODS}
        for token in row["tokens"]:
            tokens += 1
            candidate_ids = np.asarray(token["candidate_token_ids"], dtype=np.int64)
            logits = np.asarray(token["candidate_logits_by_expert"], dtype=np.float64)
            gold = int(token["gold_token_id"])
            lse = np.asarray(
                [item["logsumexp_by_temperature"] for item in token["expert_metrics"]],
                dtype=np.float64,
            )
            top1_ids = np.asarray(
                [int(item["top1_token_id"]) for item in token["expert_metrics"]],
                dtype=np.int64,
            )
            raw_log_probabilities = logits - lse[:, raw_index][:, None]
            raw_probabilities = np.exp(raw_log_probabilities)
            hard_votes = (top1_ids[:, None] == candidate_ids[None, :]).astype(np.float64)
            hard_primary = subset_matrix @ hard_votes
            hard_secondary = subset_matrix @ raw_log_probabilities
            probabilities = hard_primary / subset_sizes[:, None]
            safe_probabilities = np.where(probabilities > 0, probabilities, 1.0)
            vote_entropy_sum += -np.sum(
                np.where(
                    probabilities > 0,
                    probabilities * np.log(safe_probabilities),
                    0.0,
                ),
                axis=1,
            )

            primaries: dict[str, np.ndarray] = {"hard_plurality": hard_primary}
            winners: dict[str, np.ndarray] = {
                "hard_plurality": tie_break_primary(hard_primary, hard_secondary)
            }
            probability_scores = (subset_matrix @ raw_probabilities) / subset_sizes[:, None]
            primaries["probability_mean"] = probability_scores
            winners["probability_mean"] = stable_argmax(probability_scores)

            fitted_temperatures = np.asarray([grid[index] for index in fitted_indices])
            fitted_probabilities = np.exp(
                logits / fitted_temperatures[:, None]
                - lse[np.arange(len(EXPERTS)), fitted_indices][:, None]
            )
            fitted_scores = (subset_matrix @ fitted_probabilities) / subset_sizes[:, None]
            primaries["calibrated_probability_mean"] = fitted_scores
            winners["calibrated_probability_mean"] = stable_argmax(fitted_scores)

            shared_temperature = grid[shared_index]
            shared_probabilities = np.exp(
                logits / shared_temperature - lse[:, shared_index][:, None]
            )
            shared_scores = (subset_matrix @ shared_probabilities) / subset_sizes[:, None]
            primaries["shared_temperature_probability_mean"] = shared_scores
            winners["shared_temperature_probability_mean"] = stable_argmax(shared_scores)

            logit_scores = (subset_matrix @ logits) / subset_sizes[:, None]
            primaries["logit_mean"] = logit_scores
            winners["logit_mean"] = stable_argmax(logit_scores)
            means = np.asarray([float(item["logit_mean"]) for item in token["expert_metrics"]])
            stds = np.asarray([float(item["logit_std"]) for item in token["expert_metrics"]])
            normalized = (logits - means[:, None]) / np.maximum(stds[:, None], 1e-12)
            normalized_scores = (subset_matrix @ normalized) / subset_sizes[:, None]
            primaries["normalized_logit_mean"] = normalized_scores
            winners["normalized_logit_mean"] = stable_argmax(normalized_scores)

            for method, mode, depth in (
                ("top2_approval", "approval", 2),
                ("borda_top5", "borda", 5),
                ("reciprocal_rank_top5", "reciprocal", 5),
            ):
                primary = subset_matrix @ rank_matrix(token, candidate_ids, mode=mode, depth=depth)
                primaries[method] = primary
                winners[method] = tie_break_primary(primary, hard_secondary)

            if set(winners) != set(METHODS):
                raise RuntimeError("Diagnostic method family differs from the frozen analysis")
            for method in METHODS:
                selected = candidate_ids[winners[method]]
                correct = selected == gold
                token_correct[method] += correct
                example_ok[method] &= correct
                tie_counts[method] += tie_flags(primaries[method])
        for method in METHODS:
            example_correct[method][decision] += example_ok[method]
            example_outcomes[method][examples - 1] = example_ok[method]

    if examples != EXPECTED_EXAMPLES or tokens != EXPECTED_TOKENS:
        raise RuntimeError(f"Unexpected cache extent: {examples} examples, {tokens} tokens")
    if example_totals != {"answer": 3510, "abstain": 540}:
        raise RuntimeError(f"Unexpected answer-class counts: {example_totals}")

    records = []
    for index, mask in enumerate(masks):
        for method in METHODS:
            source = baseline[(mask, method)]
            observed_example_correct = int(
                example_correct[method]["answer"][index]
                + example_correct[method]["abstain"][index]
            )
            if int(token_correct[method][index]) != int(source["token_correct"]):
                raise RuntimeError(f"Token reproduction mismatch for {mask}/{method}")
            if observed_example_correct != int(source["example_correct"]):
                raise RuntimeError(f"Example reproduction mismatch for {mask}/{method}")
            answerable_accuracy = float(
                example_correct[method]["answer"][index] / example_totals["answer"]
            )
            abstention_accuracy = float(
                example_correct[method]["abstain"][index] / example_totals["abstain"]
            )
            records.append(
                {
                    "mask": mask,
                    "bitmask": f"{mask:07b}",
                    "members": members[index],
                    "k": int(subset_sizes[index]),
                    "method": method,
                    "token_accuracy": source["token_accuracy"],
                    "all_token_example_accuracy": source["example_accuracy"],
                    "answerable_example_accuracy": answerable_accuracy,
                    "abstention_example_accuracy": abstention_accuracy,
                    "balanced_answer_class_accuracy": 0.5
                    * (answerable_accuracy + abstention_accuracy),
                    "mean_vote_entropy": float(vote_entropy_sum[index] / tokens),
                    "primary_tie_rate": float(tie_counts[method][index] / tokens),
                    "expert_token_forward_proxy": int(subset_sizes[index] * tokens),
                    "measured_latency_seconds": None,
                    "measured_peak_gpu_vram_bytes": None,
                    "systems_measurement_boundary": (
                        "not measured for cached counterfactual subsets; selected free-running conditions only"
                    ),
                    "analysis_role": "DEV MECHANISTIC / CACHED COMMITTEE DIAGNOSTICS",
                }
            )
    if len(records) != 127 * len(METHODS):
        raise RuntimeError("Committee diagnostic inventory is incomplete")

    balanced_scaling = []
    for method in METHODS:
        for k in range(1, 8):
            group = [
                row for row in records if row["method"] == method and int(row["k"]) == k
            ]
            values = np.asarray(
                [row["balanced_answer_class_accuracy"] for row in group], dtype=float
            )
            best = sorted(
                group,
                key=lambda row: (
                    -row["balanced_answer_class_accuracy"],
                    -row["all_token_example_accuracy"],
                    row["mask"],
                ),
            )[0]
            balanced_scaling.append(
                {
                    "method": method,
                    "k": k,
                    "committee_count": len(group),
                    "mean_balanced_answer_class_accuracy": float(values.mean()),
                    "min_balanced_answer_class_accuracy": float(values.min()),
                    "max_balanced_answer_class_accuracy": float(values.max()),
                    "best_balanced_mask": best["mask"],
                    "best_balanced_members": best["members"],
                    "best_balanced_answer_class_accuracy": best[
                        "balanced_answer_class_accuracy"
                    ],
                    "analysis_role": "DEV MODEL-SELECTION / BALANCED ANSWER-CLASS SUMMARY",
                }
            )

    scaling_intervals = []
    for source in teacher["committee_scaling"]:
        method = source["method"]
        k = int(source["k"])
        indices = np.flatnonzero(subset_sizes == k)
        mean_per_example = example_outcomes[method][:, indices].mean(axis=1)
        mean_low, mean_high = normal_mean_ci95(mean_per_example)
        best_index = masks.index(int(source["best_mask"]))
        best_correct = int(example_outcomes[method][:, best_index].sum())
        best_low, best_high = wilson_ci95(best_correct, EXPECTED_EXAMPLES)
        scaling_intervals.append(
            {
                "method": method,
                "k": k,
                "committee_count": len(indices),
                "mean_example_accuracy": source["mean_example_accuracy"],
                "mean_example_accuracy_ci95_low": mean_low,
                "mean_example_accuracy_ci95_high": mean_high,
                "mean_ci_definition": (
                    "normal 95% interval over per-example mean correctness across the exhaustive committees at fixed K"
                ),
                "best_mask": int(source["best_mask"]),
                "best_members": source["best_members"],
                "best_example_accuracy": source["best_example_accuracy"],
                "best_example_accuracy_ci95_low": best_low,
                "best_example_accuracy_ci95_high": best_high,
                "best_ci_definition": (
                    "Wilson 95% interval conditional on the DEV-selected best committee; not selection-adjusted"
                ),
                "analysis_role": "DEV MODEL-SELECTION / DESCRIPTIVE INTERVAL",
            }
        )

    packet = {
        "schema_version": "1.0",
        "status": "DEV_COMMITTEE_DIAGNOSTICS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MECHANISTIC / CACHED COMMITTEE DIAGNOSTICS",
        "example_count": examples,
        "token_count": tokens,
        "committee_count": len(masks),
        "method_count": len(METHODS),
        "record_count": len(records),
        "answer_class_counts": example_totals,
        "records": records,
        "balanced_committee_scaling": balanced_scaling,
        "committee_scaling_intervals": scaling_intervals,
        "provenance": {
            "cache_path": str(CACHE.relative_to(ROOT)).replace("\\", "/"),
            "cache_sha256": sha256(CACHE),
            "teacher_analysis_path": str(TEACHER.relative_to(ROOT)).replace("\\", "/"),
            "teacher_analysis_sha256": sha256(TEACHER),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "cache_reused_without_new_model_forwards": True,
            "all_frozen_token_and_example_counts_reproduced": True,
        },
        "limitations": [
            "All accuracy values are gold-prefix teacher-forced local results, not free-running sequence accuracy.",
            "Vote entropy summarizes the active experts' raw top-1 proposal histogram and is shared across aggregation rules for a fixed committee/token.",
            "Primary ties mean ties before each method's deterministic confidence/token-ID tie-break.",
            "Counterfactual cached subsets have no measured latency or VRAM; those fields remain null rather than imputed.",
            "Best-at-K Wilson intervals are conditional on selection using the same DEV outcomes and are not selection-adjusted confirmatory intervals.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat = [{**row, "members": ";".join(row["members"])} for row in records]
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    full_hard = next(
        row for row in records if row["mask"] == FULL_MASK and row["method"] == "hard_plurality"
    )
    REPORT.write_text(
        "\n".join(
            [
                "# DEV cached committee diagnostics",
                "",
                "Status: **DEV_COMMITTEE_DIAGNOSTICS_COMPLETE**",
                "",
                "This supplement adds answer/abstain all-token accuracy, proposal vote entropy, and pre-tie-break tie rates to all 127 committees under all nine frozen cached aggregators (1,143 records). It exactly reproduces every frozen token and example count.",
                "",
                "## Full-K=7 hard reference",
                "",
                f"- Answerable all-token example accuracy: {100*full_hard['answerable_example_accuracy']:.3f}%.",
                f"- Abstention all-token example accuracy: {100*full_hard['abstention_example_accuracy']:.3f}%.",
                f"- Mean proposal vote entropy: {full_hard['mean_vote_entropy']:.4f} nats.",
                f"- Hard primary tie rate: {100*full_hard['primary_tie_rate']:.3f}%.",
                "",
                "## Systems boundary",
                "",
                "Latency and VRAM are intentionally null for cached counterfactual subsets. They are measured only for selected free-running conditions; no systems value is imputed from forward proxies.",
                "",
                f"Cache SHA-256: `{packet['provenance']['cache_sha256']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": packet["status"], "records": len(records), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
