from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from consensus_lab.io import write_json_atomic


SUMMARY_PROJECTION_KEYS = {
    "run_name",
    "dataset",
    "committee",
    "method",
    "experts",
    "correct",
    "decision",
    "abstained",
    "generated_tokens",
    "latency_seconds",
    "reasoning_family",
    "domain",
    "stop_reason",
    "repetition_failure",
    "output",
    "peak_gpu_vram_bytes",
    "peak_cpu_rss_bytes",
    "forward_passes",
    "adapter_evaluations",
    "time_to_first_final_token_seconds",
    "adapter_storage_bytes",
    "backbone_parameter_count",
    "active_parameter_count",
    "committee_adapter_parameter_count",
    "mean_agreement",
    "mean_vote_entropy",
    "mean_pairwise_js",
    "tie_count",
}


def summary_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Discard token traces and text metadata not used by aggregate summaries."""
    return {key: row.get(key) for key in SUMMARY_PROJECTION_KEYS}


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def _mean(values: list[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(kept) if kept else None


def _uncertainty_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def optional_array(key: str, *, transform: Callable[[float], float] | None = None) -> np.ndarray:
        values: list[float] = []
        for row in rows:
            raw = row.get(key)
            if raw is None:
                values.append(float("nan"))
                continue
            value = float(raw)
            values.append(transform(value) if transform is not None else value)
        return np.asarray(values, dtype=float)

    errors = np.asarray([not bool(row["correct"]) for row in rows], dtype=int)
    disagreement = optional_array("mean_agreement", transform=lambda value: 1.0 - value)
    entropy = optional_array("mean_vote_entropy")
    js = optional_array("mean_pairwise_js")
    def auc(scores: np.ndarray) -> float | None:
        finite = np.isfinite(scores)
        if finite.sum() < 2 or len(set(errors[finite].tolist())) < 2:
            return None
        return float(roc_auc_score(errors[finite], scores[finite]))

    finite_disagreement = np.isfinite(disagreement)
    disagreement_auprc = None
    if finite_disagreement.sum() >= 2 and len(set(errors[finite_disagreement].tolist())) >= 2:
        disagreement_auprc = float(
            average_precision_score(errors[finite_disagreement], disagreement[finite_disagreement])
        )
    calibration_bins: list[dict[str, float | int | None]] = []
    if finite_disagreement.any():
        for low in np.linspace(0.0, 1.0, 6)[:-1]:
            high = low + 0.2
            if high >= 1.0:
                mask = finite_disagreement & (disagreement >= low) & (disagreement <= high)
            else:
                mask = finite_disagreement & (disagreement >= low) & (disagreement < high)
            calibration_bins.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "count": int(mask.sum()),
                    "mean_predicted_error": float(disagreement[mask].mean()) if mask.any() else None,
                    "observed_error_rate": float(errors[mask].mean()) if mask.any() else None,
                }
            )
    finite_count = int(finite_disagreement.sum())
    brier = (
        float(np.mean((disagreement[finite_disagreement] - errors[finite_disagreement]) ** 2))
        if finite_count
        else None
    )
    ece = None
    if finite_count:
        ece = float(
            sum(
                int(item["count"])
                / finite_count
                * abs(float(item["mean_predicted_error"]) - float(item["observed_error_rate"]))
                for item in calibration_bins
                if int(item["count"]) > 0
            )
        )
    return {
        "disagreement_error_auroc": auc(disagreement),
        "disagreement_error_auprc": disagreement_auprc,
        "disagreement_error_brier": brier,
        "disagreement_error_ece": ece,
        "disagreement_calibration_bins": calibration_bins,
        "vote_entropy_error_auroc": auc(entropy),
        "pairwise_js_error_auroc": auc(js),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("run_name") or "unknown_run"),
                str(row["dataset"]),
                str(row["committee"]),
                str(row["method"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (run_name, dataset, committee, method), group in sorted(groups.items()):
        tokens = [int(row["generated_tokens"]) for row in group]
        latencies = [float(row["latency_seconds"]) for row in group]
        category_values: dict[str, list[bool]] = defaultdict(list)
        for row in group:
            category_values[str(row.get("reasoning_family") or row.get("domain") or "uncategorized")].append(
                bool(row["correct"])
            )
        summary: dict[str, Any] = {
            "run_name": run_name,
            "dataset": dataset,
            "committee": committee,
            "method": method,
            "committee_size": len(group[0]["experts"]),
            "case_count": len(group),
            "correctness": _rate([bool(row["correct"]) for row in group]),
            "answerable_correctness": _rate(
                [bool(row["correct"]) for row in group if row.get("decision") == "answer"]
            ),
            "abstention_correctness": _rate(
                [bool(row["correct"]) for row in group if row.get("decision") != "answer"]
            ),
            "false_abstention_rate": _rate(
                [bool(row.get("abstained")) for row in group if row.get("decision") == "answer"]
            ),
            "mean_output_tokens": statistics.mean(tokens) if tokens else None,
            "median_output_tokens": statistics.median(tokens) if tokens else None,
            "p90_output_tokens": _quantile([float(value) for value in tokens], 0.90),
            "p95_output_tokens": _quantile([float(value) for value in tokens], 0.95),
            "max_output_tokens": max(tokens) if tokens else None,
            "eos_rate": _rate([row.get("stop_reason") == "eos" for row in group]),
            "repetition_rate": _rate([bool(row.get("repetition_failure")) for row in group]),
            "nonempty_rate": _rate([bool(str(row.get("output") or "").strip()) for row in group]),
            "completion_rate": _rate(
                [bool(str(row.get("output") or "").strip()) and row.get("stop_reason") == "eos" for row in group]
            ),
            "short_response_rate_le_16_tokens": _rate([value <= 16 for value in tokens]),
            "generation_limit_compliance_rate": _rate([value <= 32 for value in tokens]),
            "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
            "p50_latency_seconds": _quantile(latencies, 0.50),
            "p90_latency_seconds": _quantile(latencies, 0.90),
            "p95_latency_seconds": _quantile(latencies, 0.95),
            "tokens_per_second": sum(tokens) / sum(latencies) if sum(latencies) else None,
            "peak_gpu_vram_bytes": max(
                (int(row["peak_gpu_vram_bytes"]) for row in group if row.get("peak_gpu_vram_bytes") is not None),
                default=None,
            ),
            "peak_cpu_rss_bytes": max(int(row["peak_cpu_rss_bytes"]) for row in group),
            "mean_forward_passes": statistics.mean(int(row["forward_passes"]) for row in group),
            "mean_adapter_evaluations": statistics.mean(int(row["adapter_evaluations"]) for row in group),
            "mean_time_to_first_final_token_seconds": _mean(
                [row.get("time_to_first_final_token_seconds") for row in group]
            ),
            "adapter_storage_bytes": max(int(row.get("adapter_storage_bytes") or 0) for row in group),
            "backbone_parameter_count": max(int(row.get("backbone_parameter_count") or 0) for row in group),
            "active_parameter_count": max(int(row.get("active_parameter_count") or 0) for row in group),
            "committee_adapter_parameter_count": max(
                int(row.get("committee_adapter_parameter_count") or 0) for row in group
            ),
            "mean_agreement": _mean([row.get("mean_agreement") for row in group]),
            "mean_vote_entropy": _mean([row.get("mean_vote_entropy") for row in group]),
            "mean_pairwise_js": _mean([row.get("mean_pairwise_js") for row in group]),
            "tie_example_rate": _rate([int(row.get("tie_count") or 0) > 0 for row in group]),
            "category_correctness": {key: _rate(values) for key, values in sorted(category_values.items())},
        }
        summary.update(_uncertainty_metrics(group))
        summaries.append(summary)
    return summaries


def write_summary_files(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    summaries = summarize_rows(rows)
    write_json_atomic(json_path, {"schema_version": "1.0", "summaries": summaries})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = sorted(
        {key for summary in summaries for key, value in summary.items() if not isinstance(value, (dict, list))}
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in scalar_keys})
    return summaries
