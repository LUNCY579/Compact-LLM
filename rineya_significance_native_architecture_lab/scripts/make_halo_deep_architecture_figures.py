#!/usr/bin/env python3
"""Generate measured B-F and prospective-validation figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.integrity import content_tree_sha256  # noqa: E402
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _json(path: str) -> dict[str, Any]:
    return json.loads((LAB_ROOT / path).read_text(encoding="utf-8"))


def _jsonl(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (LAB_ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bar(
    output: Path,
    filename: str,
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    color: str = "#4c78a8",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3g}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output / filename, dpi=160, metadata={"Software": "Rineya Significance Lab"})
    plt.close(fig)


def build(output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise RuntimeError(f"refusing to overwrite non-empty figure directory: {output}")
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []

    def record(
        filename: str,
        title: str,
        csv_name: str,
        sources: list[str],
        limit: str,
    ) -> None:
        figures.append(
            {
                "file": filename,
                "title": title,
                "data": f"data/{csv_name}",
                "sources": sources,
                "claim_limit": limit,
                "sha256": sha256_file(output / filename),
            }
        )

    # Locally registered prospective endpoint validation.
    endpoint_summary_path = "results/runs/halo_endpoint_validation_v2_registered/summary.json"
    endpoint_stats_path = "results/runs/halo_endpoint_validation_v2_registered/statistics.json"
    endpoint_registration_path = "results/registrations/halo_endpoint_validation_v2.json"
    endpoint_summary = _json(endpoint_summary_path)
    endpoint_stats = _json(endpoint_stats_path)
    endpoint_rows = [
        {
            "condition": condition,
            "clusters": values["clusters"],
            "correct": values["correct"],
            "accuracy": values["correct"] / values["clusters"],
            "endpoint": values["mean_layer_fact_exact_completion_rate"],
            "registration_scope": "local_12_cluster_v2",
        }
        for condition, values in endpoint_summary["by_condition"].items()
    ]
    _csv(data / "registered_endpoint.csv", endpoint_rows)
    _bar(
        output,
        "01_registered_endpoint_retention.png",
        [row["condition"] for row in endpoint_rows],
        [row["endpoint"] for row in endpoint_rows],
        title="Registered endpoint validation v2",
        ylabel="Mean exact layer x fact completion",
    )
    record(
        "01_registered_endpoint_retention.png",
        "Registered endpoint validation v2",
        "registered_endpoint.csv",
        [endpoint_summary_path, endpoint_registration_path],
        "Local 12-cluster registration; not a public preregistration or population claim.",
    )
    _bar(
        output,
        "02_registered_endpoint_accuracy.png",
        [row["condition"] for row in endpoint_rows],
        [row["accuracy"] for row in endpoint_rows],
        title="Registered endpoint validation answers",
        ylabel="Correct / cluster",
        ylim=(0, 1.0),
        color="#e45756",
    )
    record(
        "02_registered_endpoint_accuracy.png",
        "Registered endpoint validation answers",
        "registered_endpoint.csv",
        [endpoint_summary_path],
        "B3 retains more evidence but has one fewer correct answer than B2.",
    )
    primary = endpoint_stats["comparisons"]["B3_minus_B2_primary_endpoint"]
    endpoint_delta_rows = [
        {"cluster": key, "delta": value}
        for key, value in primary["cluster_differences"].items()
    ]
    _csv(data / "registered_endpoint_deltas.csv", endpoint_delta_rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    deltas = [row["delta"] for row in endpoint_delta_rows]
    ax.scatter(range(len(deltas)), deltas, color="#4c78a8")
    ax.axhline(0, color="0.35", linewidth=1)
    ax.axhline(primary["mean_difference"], color="#e45756", label="cluster mean")
    ax.set_title("Registered B3-B2 endpoint deltas")
    ax.set_xlabel("Independent cluster")
    ax.set_ylabel("Endpoint difference")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(
        output / "03_registered_endpoint_cluster_deltas.png",
        dpi=160,
        metadata={"Software": "Rineya Significance Lab"},
    )
    plt.close(fig)
    record(
        "03_registered_endpoint_cluster_deltas.png",
        "Registered B3-B2 endpoint deltas",
        "registered_endpoint_deltas.csv",
        [endpoint_stats_path],
        "Primary locally registered contrast; 12 clusters only.",
    )

    # Clean eligible-pool count controls.
    count_path = "results/runs/halo_count_causal_controls_v2/summary.json"
    count_manifest_path = "results/runs/halo_count_causal_controls_v2/manifest.json"
    count_summary = _json(count_path)
    count_rows = [
        {
            "condition": condition,
            "accuracy": values["correct"] / values["clusters"],
            "endpoint": values["mean_layer_fact_exact_completion_rate"],
            "paired_execution_scope": "same_process",
            "cross_process_trace_identity": "not_established",
        }
        for condition, values in count_summary["by_condition"].items()
    ]
    _csv(data / "eligible_count_v2.csv", count_rows)
    for index, key, title, ylabel, color in (
        (4, "endpoint", "Eligible-pool count controls", "Mean frozen endpoint", "#4c78a8"),
        (5, "accuracy", "Eligible-pool count-control answers", "Correct / cluster", "#e45756"),
    ):
        filename = f"{index:02d}_eligible_count_{key}.png"
        _bar(
            output,
            filename,
            [row["condition"].replace("_count_v2", "") for row in count_rows],
            [row[key] for row in count_rows],
            title=title,
            ylabel=ylabel,
            ylim=(0, 1.0) if key == "accuracy" else None,
            color=color,
        )
        record(
            filename,
            title,
            "eligible_count_v2.csv",
            [count_path, count_manifest_path],
            "Exploratory process-local pairs; cross-process trace reproducibility is not claimed.",
        )

    # Experiment B closed-loop heterogeneous allocation.
    b_path = "results/runs/halo_heterogeneous_budget_quality_screen_v1/summary.json"
    b_summary = _json(b_path)
    b_rows = [
        {
            "condition": condition,
            "accuracy": values["correct"] / values["clusters"],
            "endpoint": values["mean_layer_fact_exact_completion_rate"],
            "logical_mib": values["mean_logical_payload_bytes_after_decode"] / 2**20,
            "physical_mib": values["mean_physical_total_bytes_after_decode"] / 2**20,
            "overhead_mib": values["mean_representation_overhead_bytes"] / 2**20,
        }
        for condition, values in b_summary["by_condition"].items()
    ]
    _csv(data / "experiment_b_quality.csv", b_rows)
    for index, key, title, ylabel in (
        (6, "endpoint", "Experiment B retention", "Mean frozen endpoint"),
        (7, "accuracy", "Experiment B answers", "Correct / cluster"),
        (8, "physical_mib", "Experiment B padded storage", "Mean physical MiB after decode"),
    ):
        filename = f"{index:02d}_experiment_b_{key}.png"
        _bar(
            output,
            filename,
            [row["condition"].replace("B3_", "") for row in b_rows],
            [row[key] for row in b_rows],
            title=title,
            ylabel=ylabel,
            ylim=(0, 1.0) if key == "accuracy" else None,
            color="#72b7b2" if key == "physical_mib" else "#4c78a8",
        )
        record(
            filename,
            title,
            "experiment_b_quality.csv",
            [b_path],
            "Exploratory same-process screen; shuffled retention exceeds entropy and physical padding is disclosed.",
        )

    # Experiment C attention prior.
    c_path = "results/runs/halo_attention_prior_quality_screen_v1/summary.json"
    c_config_path = "results/runs/halo_attention_prior_quality_screen_v1/resolved_config.yaml"
    c_summary = _json(c_path)
    c_rows = [
        {
            "condition": condition,
            "accuracy": values["correct"] / values["clusters"],
            "endpoint": values["mean_primary_endpoint"],
            "attention_entropy": values["mean_attention_entropy_normalized"],
            "max_bias": values["maximum_abs_bias"],
            "lambda": 0.0 if condition == "C0_B3_NO_PRIOR" else 0.01,
        }
        for condition, values in c_summary["by_condition"].items()
    ]
    _csv(data / "experiment_c_attention.csv", c_rows)
    for index, key, title, ylabel in (
        (9, "endpoint", "Experiment C retention", "Mean frozen endpoint"),
        (10, "accuracy", "Experiment C answers", "Correct / cluster"),
        (11, "attention_entropy", "Experiment C attention entropy", "Normalized attention entropy"),
    ):
        filename = f"{index:02d}_experiment_c_{key}.png"
        _bar(
            output,
            filename,
            [row["condition"].replace("_SC", "") for row in c_rows],
            [row[key] for row in c_rows],
            title=title,
            ylabel=ylabel,
            ylim=(0, 1.0) if key == "accuracy" else None,
        )
        record(
            filename,
            title,
            "experiment_c_attention.csv",
            [c_path, c_config_path],
            "Lambda 0.01 exploratory screen; normalized equals raw/shuffled on endpoint and all answers tie.",
        )

    # Experiment D fixed memory slots.
    d_path = "results/runs/halo_significance_memory_quality_screen_v1/summary.json"
    d_summary = _json(d_path)
    d_rows = [
        {
            "condition": condition,
            "slots": values["memory_slots_per_layer_kv_head"],
            "ordinary_budget": values["ordinary_retained_history_budget"],
            "accuracy": values["correct"] / values["clusters"],
            "endpoint": values["mean_layer_fact_exact_completion_rate"],
        }
        for condition, values in d_summary["by_condition"].items()
    ]
    _csv(data / "experiment_d_memory.csv", d_rows)
    for index, key, title, ylabel in (
        (12, "endpoint", "Experiment D retention", "Mean frozen endpoint"),
        (13, "accuracy", "Experiment D answers", "Correct / cluster"),
    ):
        filename = f"{index:02d}_experiment_d_{key}.png"
        _bar(
            output,
            filename,
            [row["condition"].replace("M4_", "") for row in d_rows],
            [row[key] for row in d_rows],
            title=title,
            ylabel=ylabel,
            ylim=(0, 1.0) if key == "accuracy" else None,
            color="#b279a2",
        )
        record(
            filename,
            title,
            "experiment_d_memory.csv",
            [d_path],
            "Total-byte-matched exploratory screen; blended slots cannot satisfy the exact-token endpoint directly.",
        )
    normalized = next(row for row in d_rows if row["condition"] == "M4_NORMALIZED_S_OVER_C")
    hard = next(row for row in d_rows if row["condition"] == "M0_HARD")
    memory_count_rows = [hard, normalized]
    _csv(data / "memory_slot_count.csv", memory_count_rows)
    _bar(
        output,
        "14_memory_slot_count_vs_accuracy.png",
        [f"M={row['slots']}" for row in memory_count_rows],
        [row["accuracy"] for row in memory_count_rows],
        title="Memory-slot count vs answer accuracy",
        ylabel="Correct / cluster",
        ylim=(0, 1.0),
        color="#b279a2",
    )
    record(
        "14_memory_slot_count_vs_accuracy.png",
        "Memory-slot count vs answer accuracy",
        "memory_slot_count.csv",
        [d_path],
        "Only M=0 and normalized M=4 were tested; both score 5/12.",
    )
    budget = d_summary["total_byte_budget"]
    memory_budget_rows = [
        {"condition": "M0 hard", "bytes": budget["baseline_total_bytes"], "endpoint": hard["endpoint"]},
        {
            "condition": "M4 normalized",
            "bytes": budget["candidate_total_bytes"],
            "endpoint": normalized["endpoint"],
        },
    ]
    _csv(data / "memory_quality_bytes.csv", memory_budget_rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for row in memory_budget_rows:
        ax.scatter(row["bytes"] / 2**20, row["endpoint"], s=70)
        ax.annotate(row["condition"], (row["bytes"] / 2**20, row["endpoint"]), xytext=(4, 4), textcoords="offset points")
    ax.set_title("Memory retention vs matched total bytes")
    ax.set_xlabel("Boundary capacity (MiB)")
    ax.set_ylabel("Mean frozen endpoint")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "15_memory_quality_vs_bytes.png", dpi=160, metadata={"Software": "Rineya Significance Lab"})
    plt.close(fig)
    record(
        "15_memory_quality_vs_bytes.png",
        "Memory retention vs matched total bytes",
        "memory_quality_bytes.csv",
        [d_path],
        "M4 uses 108,416 fewer represented bytes but has lower exact-token retention; no deployment saving or quality advantage is supported.",
    )

    # Experiment E risk/coverage.
    recovery_path = "results/runs/halo_recovery_controller_screen_v1/risk_coverage.jsonl"
    recovery_summary_path = "results/runs/halo_recovery_controller_screen_v1/summary.json"
    recovery_summary = _json(recovery_summary_path)
    recovery_rows = _jsonl(recovery_path)
    wall_interpretation = recovery_summary["cost_accounting"]["wall_time_interpretation"]
    for row in recovery_rows:
        row["wall_time_interpretation"] = wall_interpretation
    _csv(data / "recovery_risk_coverage.csv", recovery_rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ordered = sorted(recovery_rows, key=lambda row: row["coverage"])
    ax.plot([row["coverage"] for row in ordered], [row["system_accuracy"] for row in ordered], marker="o", label="system accuracy")
    ax.plot([row["coverage"] for row in ordered], [row["selective_risk"] if row["selective_risk"] is not None else np.nan for row in ordered], marker="s", label="selective risk")
    ax.set_title("Experiment E risk and coverage")
    ax.set_xlabel("Accepted coverage")
    ax.set_ylabel("Rate")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "16_recovery_risk_coverage.png", dpi=160, metadata={"Software": "Rineya Significance Lab"})
    plt.close(fig)
    record(
        "16_recovery_risk_coverage.png",
        "Experiment E risk and coverage",
        "recovery_risk_coverage.csv",
        [recovery_path, recovery_summary_path],
        "Open-loop resubstitution curve on 11 selected clusters; deterministic risk AUC is 0.50.",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot([row["coverage"] for row in ordered], [row["wall_multiplier_vs_compressed"] for row in ordered], marker="o")
    ax.set_title("Recovery coverage vs recorded wall multiplier")
    ax.set_xlabel("Accepted coverage")
    ax.set_ylabel("Wall multiplier")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "17_recovery_coverage_cost.png", dpi=160, metadata={"Software": "Rineya Significance Lab"})
    plt.close(fig)
    record(
        "17_recovery_coverage_cost.png",
        "Recovery coverage vs recorded wall multiplier",
        "recovery_risk_coverage.csv",
        [recovery_path],
        "Source schedules differ; cost is descriptive and not a latency benchmark.",
    )

    # Experiment F tiny controller.
    f_path = "results/experiment_f/halo_tiny_controller_v1/summary.json"
    f_summary = _json(f_path)
    evaluation_support = f_summary["dataset"]["class_support"]["evaluation"]
    evaluation_clusters = int(f_summary["dataset"]["evaluation_clusters"])
    auc_pairs = int(evaluation_support["compressed_error"]) * int(
        evaluation_support["compressed_success"]
    )
    f_rows = [
        {
            "method": method,
            "auc": values["error_roc_auc"],
            "balanced_accuracy": values["balanced_accuracy"],
            "coverage": values["coverage"],
            "system_accuracy": values["system_accuracy"],
            "evaluation_clusters": evaluation_clusters,
            "positive_negative_auc_pairs": auc_pairs,
        }
        for method, values in f_summary["held_out_comparison"].items()
    ]
    _csv(data / "tiny_controller.csv", f_rows)
    for index, key, title, ylabel in (
        (18, "auc", "Experiment F held-out error AUC", "AUC (n=3)"),
        (19, "balanced_accuracy", "Experiment F balanced accuracy", "Balanced accuracy (n=3)"),
        (20, "coverage", "Experiment F accepted coverage", "Coverage (n=3)"),
    ):
        filename = f"{index:02d}_tiny_controller_{key}.png"
        _bar(
            output,
            filename,
            [row["method"].replace("deterministic_", "") for row in f_rows],
            [row[key] for row in f_rows],
            title=title,
            ylabel=ylabel,
            ylim=(0, 1.08),
            color="#54a24b",
        )
        record(
            filename,
            title,
            "tiny_controller.csv",
            [f_path],
            "Underpowered overfit screen: evaluation has three clusters and only two positive-negative AUC pairs.",
        )

    # Balanced R7 v2 capability and prospective gate redesign.
    r7_path = "results/runs/halo_r7_v2_capability_screen/cases.jsonl"
    r7_cases = _jsonl(r7_path)
    r7_rows = []
    for answer in ("YES", "NO"):
        selected = [row for row in r7_cases if row["true_answer"] == answer]
        r7_rows.append(
            {
                "true_answer": answer,
                "clusters": len(selected),
                "correct": sum(bool(row["correct"]) for row in selected),
                "accuracy": float(np.mean([bool(row["correct"]) for row in selected])),
                "predicted_yes": sum(str(row["prediction"]).strip().upper().startswith("YES") for row in selected),
            }
        )
    _csv(data / "r7_v2_capability.csv", r7_rows)
    _bar(
        output,
        "21_r7_v2_answer_balance.png",
        [row["true_answer"] for row in r7_rows],
        [row["accuracy"] for row in r7_rows],
        title="R7 v2 Halo capability by true answer",
        ylabel="Full-cache accuracy",
        ylim=(0, 1.08),
        color="#f58518",
    )
    record(
        "21_r7_v2_answer_balance.png",
        "R7 v2 Halo capability by true answer",
        "r7_v2_capability.csv",
        [r7_path],
        "Balanced valid design; Halo predicted YES on all eight and failed the per-answer gate.",
    )
    gate_rows = []
    for version, path in (
        ("v1", "results/runs/halo_endpoint_validation_capability_v1/summary.json"),
        ("v2", "results/runs/halo_endpoint_validation_v2_capability/summary.json"),
    ):
        summary = _json(path)
        for family, values in summary["by_family"].items():
            gate_rows.append(
                {
                    "version": version,
                    "family": family,
                    "accuracy": values["correct"] / values["cases"],
                }
            )
    _csv(data / "prospective_capability_gates.csv", gate_rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    families = ["R1", "R3", "R4"]
    x = np.arange(len(families))
    width = 0.36
    for offset, version in ((-width / 2, "v1"), (width / 2, "v2")):
        values = [
            next(row["accuracy"] for row in gate_rows if row["version"] == version and row["family"] == family)
            for family in families
        ]
        ax.bar(x + offset, values, width=width, label=version)
    ax.set_xticks(x, families)
    ax.set_ylim(0, 1.08)
    ax.set_title("Prospective endpoint B0 gate redesign")
    ax.set_ylabel("Full-cache accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "22_prospective_capability_gates.png", dpi=160, metadata={"Software": "Rineya Significance Lab"})
    plt.close(fig)
    record(
        "22_prospective_capability_gates.png",
        "Prospective endpoint B0 gate redesign",
        "prospective_capability_gates.csv",
        [
            "results/runs/halo_endpoint_validation_capability_v1/summary.json",
            "results/runs/halo_endpoint_validation_v2_capability/summary.json",
            "PROSPECTIVE_ENDPOINT_VALIDATION_PROTOCOL.md",
            "ENDPOINT_VALIDATION_V2_RESULT.md",
        ],
        "V2 difficulty was redesigned after v1 capability failure and before v2 compression.",
    )

    artifacts = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "figure_manifest.json"
    }
    figures.sort(key=lambda row: str(row["file"]))
    manifest = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "MEASURED_DEEP_ARCHITECTURE_FIGURES",
        "figure_count": len(figures),
        "figures": figures,
        "source_hashes": {
            source: sha256_file(LAB_ROOT / source)
            for source in sorted({item for figure in figures for item in figure["sources"]})
        },
        "integrity": {
            "algorithm": "sha256-canonical-path-map-v1",
            "artifact_files": artifacts,
            "artifact_tree_sha256": content_tree_sha256(artifacts),
            "generator": "scripts/make_halo_deep_architecture_figures.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_limit": "All panels are local exploratory or bounded validation artifacts; individual limits apply.",
    }
    atomic_write_json(output / "figure_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "results" / "figures" / "halo_deep_architecture_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build(args.output, overwrite=args.overwrite)
    print(json.dumps({"status": result["status"], "figures": result["figure_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
