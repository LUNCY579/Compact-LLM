#!/usr/bin/env python3
"""Generate evidence-backed Halo control figures and source data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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

from rineya_significance_lab.evaluation.evidence_endpoints import (  # noqa: E402
    score_serialized_evidence_retention,
)
from rineya_significance_lab.integrity import content_tree_sha256  # noqa: E402
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402

CORES = (
    ("Halo", "halo_cpu_smoke_20260823_v2"),
    ("Halo Unc.", "halo_uncensored_cpu_smoke_20260823"),
    ("Lyra", "lyra_cpu_smoke_20260823"),
    ("Nimbus", "nimbus_cpu_smoke_20260823"),
    ("Pulse", "pulse_cpu_smoke_20260823"),
)
COUNT_ORDER = (
    "true_realized_count",
    "constant_count",
    "age_derived_count",
    "shuffled_count",
    "wrong_token_count",
    "delayed_count",
    "noisy_count",
    "head_averaged_count",
)
COUNT_LABELS = {
    "true_realized_count": "true C",
    "constant_count": "constant",
    "age_derived_count": "age",
    "shuffled_count": "shuffled",
    "wrong_token_count": "wrong-token",
    "delayed_count": "delayed",
    "noisy_count": "noisy",
    "head_averaged_count": "head-avg",
}
METHOD_COLORS = {"B2": "#d95f02", "B3": "#1b9e77"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty figure data: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.4),
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "font.size": 10,
        }
    )


def _bar(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    path: Path,
    ylim: tuple[float, float] | None = None,
    color: str = "#4c78a8",
) -> None:
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=28)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3g}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": "Rineya Significance Lab"})
    plt.close(fig)


def _line_by_method(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots()
    for method in ("B2", "B3"):
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda row: row["rounds"]
        )
        ax.plot(
            [row["rounds"] for row in selected],
            [row[value_key] for row in selected],
            marker="o",
            label=f"{method} (n shown in data)",
            color=METHOD_COLORS[method],
        )
    ax.set_title(title)
    ax.set_xlabel("Realized compression rounds")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": "Rineya Significance Lab"})
    plt.close(fig)


def build(output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise RuntimeError(f"refusing to overwrite non-empty figure directory: {output}")
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _style()
    figures: list[dict[str, Any]] = []

    def record(name: str, title: str, data: str, sources: list[str], claim_limit: str) -> None:
        figures.append(
            {
                "file": name,
                "title": title,
                "data": f"data/{data}",
                "sources": sources,
                "claim_limit": claim_limit,
                "sha256": sha256_file(output / name),
            }
        )

    # Published-family engineering smokes.
    family_rows: list[dict[str, Any]] = []
    for label, run_id in CORES:
        path = LAB_ROOT / "results" / "runs" / run_id / "summary.json"
        summary = _json(path)
        identity = dict(summary["identity"])
        runtime = dict(summary["compressed_runtime"])
        family_rows.append(
            {
                "core": label,
                "run_id": run_id,
                "argmax_agreement": identity["argmax_agreement"],
                "max_abs_logit_delta": identity["max_abs_logit_delta"],
                "runtime_wall_seconds": runtime["wall_seconds"],
                "kv_bytes": runtime["kv_bytes"],
                "significance_matrix_bytes": runtime["significance_matrix_bytes"],
                "alignment_metadata_bytes": runtime["alignment_metadata_bytes"],
                "effective_evictions": runtime["effective_evictions"],
            }
        )
    _write_csv(data_dir / "family_smokes.csv", family_rows)
    labels = [row["core"] for row in family_rows]
    _bar(
        labels,
        [row["argmax_agreement"] for row in family_rows],
        title="Published-family disabled-path identity",
        ylabel="Argmax agreement",
        path=output / "01_family_smoke_identity.png",
        ylim=(0, 1.08),
    )
    record(
        "01_family_smoke_identity.png",
        "Published-family disabled-path identity",
        "family_smokes.csv",
        [f"results/runs/{run_id}/summary.json" for _, run_id in CORES],
        "One bounded case per core; compatibility only.",
    )
    _bar(
        labels,
        [row["runtime_wall_seconds"] for row in family_rows],
        title="Published-family bounded CPU runtime",
        ylabel="Wall seconds (local CPU)",
        path=output / "02_family_smoke_cpu_seconds.png",
    )
    record(
        "02_family_smoke_cpu_seconds.png",
        "Published-family bounded CPU runtime",
        "family_smokes.csv",
        [f"results/runs/{run_id}/summary.json" for _, run_id in CORES],
        "Local smoke timing; not deployment latency.",
    )
    fig, ax = plt.subplots()
    bottom = np.zeros(len(family_rows))
    for key, label, color in (
        ("kv_bytes", "K/V", "#4c78a8"),
        ("significance_matrix_bytes", "S+C", "#f58518"),
        ("alignment_metadata_bytes", "metadata", "#54a24b"),
    ):
        values = np.asarray([row[key] / 2**20 for row in family_rows])
        ax.bar(labels, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_title("Published-family persistent-state accounting")
    ax.set_ylabel("MiB at smoke endpoint")
    ax.tick_params(axis="x", rotation=28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        output / "03_family_smoke_persistent_bytes.png",
        metadata={"Software": "Rineya Significance Lab"},
    )
    plt.close(fig)
    record(
        "03_family_smoke_persistent_bytes.png",
        "Published-family persistent-state accounting",
        "family_smokes.csv",
        [f"results/runs/{run_id}/summary.json" for _, run_id in CORES],
        "Exact tensor bytes; excludes allocator/object overhead.",
    )

    # Full-cache capability gates.
    capability_rows: list[dict[str, Any]] = []
    ladder_validity = _json(
        LAB_ROOT
        / "results"
        / "runs"
        / "halo_r5_r8_capability_screen_v1"
        / "validity_audit.json"
    )
    invalid_ladder_families = set(ladder_validity["invalid_families"])
    for suite, rel in (
        ("R1-R4 calibration", "results/runs/halo_capability_calibration_v2/summary.json"),
        ("R5-R8 screen", "results/runs/halo_r5_r8_capability_screen_v1/summary.json"),
    ):
        summary = _json(LAB_ROOT / rel)
        for family, values in summary["by_family"].items():
            count = int(values.get("cases", values.get("clusters")))
            capability_rows.append(
                {
                    "suite": suite,
                    "family": family,
                    "correct": values["correct"],
                    "clusters": count,
                    "accuracy": values["correct"] / count,
                    "valid_for_capability_inference": (
                        family not in invalid_ladder_families
                        if suite == "R5-R8 screen"
                        else True
                    ),
                }
            )
    _write_csv(data_dir / "capability.csv", capability_rows)
    for index, (suite, filename) in enumerate(
        (
            ("R1-R4 calibration", "04_halo_b0_r1_r4_capability.png"),
            ("R5-R8 screen", "05_halo_b0_r5_r8_capability.png"),
        ),
        start=4,
    ):
        rows = [
            row
            for row in capability_rows
            if row["suite"] == suite and row["valid_for_capability_inference"]
        ]
        displayed_suite = (
            "R5/R6/R8 valid cells (n=2/family; R7 invalid)"
            if suite == "R5-R8 screen"
            else suite
        )
        _bar(
            [row["family"] for row in rows],
            [row["accuracy"] for row in rows],
            title=f"Halo B0 capability: {displayed_suite}",
            ylabel="Full-cache exact accuracy",
            path=output / filename,
            ylim=(0, 1.08),
            color="#72b7b2",
        )
        record(
            filename,
            f"Halo B0 capability: {displayed_suite}",
            "capability.csv",
            (
                ["results/runs/halo_capability_calibration_v2/summary.json"]
                if index == 4
                else [
                    "results/runs/halo_r5_r8_capability_screen_v1/summary.json",
                    "results/runs/halo_r5_r8_capability_screen_v1/validity_audit.json",
                ]
            ),
            "Capability gate only; no compression-policy conclusion.",
        )

    # Endpoint calibration and original pilot.
    pilot_cases = _jsonl(LAB_ROOT / "results" / "runs" / "halo_retention_pilot_v2" / "cases.jsonl")
    endpoint_values = [
        score_serialized_evidence_retention(row["evidence"]).primary_value for row in pilot_cases
    ]
    endpoint_rows = [{"row": index, "value": value} for index, value in enumerate(endpoint_values)]
    _write_csv(data_dir / "endpoint_calibration.csv", endpoint_rows)
    fig, ax = plt.subplots()
    ax.hist(endpoint_values, bins=np.linspace(0, 0.5, 11), color="#4c78a8", edgecolor="white")
    ax.set_title("Outcome-blind endpoint calibration distribution")
    ax.set_xlabel("Exact layer x fact completion rate")
    ax.set_ylabel("Condition-blind rows")
    fig.tight_layout()
    fig.savefig(
        output / "06_endpoint_calibration_distribution.png",
        metadata={"Software": "Rineya Significance Lab"},
    )
    plt.close(fig)
    record(
        "06_endpoint_calibration_distribution.png",
        "Outcome-blind endpoint calibration distribution",
        "endpoint_calibration.csv",
        [
            "results/runs/halo_retention_pilot_v2/cases.jsonl",
            "results/statistics/halo_evidence_endpoint_calibration_v1.json",
        ],
        "Calibration reuse; not an independent result.",
    )
    pilot_rows = []
    for condition in ("B2", "B3", "B4"):
        selected = [row for row in pilot_cases if row["condition"] == condition]
        pilot_rows.append(
            {
                "condition": condition,
                "clusters": len(selected),
                "correct": sum(bool(row["correct"]) for row in selected),
                "accuracy": sum(bool(row["correct"]) for row in selected) / len(selected),
                "endpoint": float(
                    np.mean(
                        [
                            score_serialized_evidence_retention(row["evidence"]).primary_value
                            for row in selected
                        ]
                    )
                ),
            }
        )
    _write_csv(data_dir / "pilot.csv", pilot_rows)
    _bar(
        [row["condition"] for row in pilot_rows],
        [row["endpoint"] for row in pilot_rows],
        title="Halo pilot frozen retention endpoint",
        ylabel="Mean exact layer x fact completion",
        path=output / "07_pilot_frozen_endpoint.png",
    )
    record(
        "07_pilot_frozen_endpoint.png",
        "Halo pilot frozen retention endpoint",
        "pilot.csv",
        ["results/runs/halo_retention_pilot_v2/cases.jsonl"],
        "Post-hoc application to endpoint-calibration rows.",
    )
    _bar(
        [row["condition"] for row in pilot_rows],
        [row["accuracy"] for row in pilot_rows],
        title="Halo pilot answer accuracy",
        ylabel="Correct / cluster",
        path=output / "08_pilot_accuracy.png",
        ylim=(0, 1.0),
    )
    record(
        "08_pilot_accuracy.png",
        "Halo pilot answer accuracy",
        "pilot.csv",
        ["results/runs/halo_retention_pilot_v2/cases.jsonl"],
        "Eleven B0-selected clusters; exploratory.",
    )

    # Count causal controls.
    count_summary = _json(
        LAB_ROOT / "results" / "runs" / "halo_count_causal_controls_v1" / "summary.json"
    )
    count_stats = _json(
        LAB_ROOT / "results" / "runs" / "halo_count_causal_controls_v1" / "statistics.json"
    )
    count_rows = []
    for condition in COUNT_ORDER:
        values = count_summary["by_condition"][condition]
        equivalence = (
            11
            if condition == "true_realized_count"
            else count_stats["count_causal"]["comparisons"][condition][
                "exact_equivalence_case_counts"
            ]["complete_boundary_diagnostic_exact"]
        )
        count_rows.append(
            {
                "condition": condition,
                "label": COUNT_LABELS[condition],
                "clusters": values["clusters"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["clusters"],
                "endpoint": values["mean_layer_fact_exact_completion_rate"],
                "exact_reference_trajectories": equivalence,
                "partial_retention_labels": count_stats["failure_distribution"]["by_condition"]
                .get(condition, {})
                .get("EVIDENCE_PARTIALLY_RETAINED", 0),
            }
        )
    _write_csv(data_dir / "count_conditions.csv", count_rows)
    _bar(
        [row["label"] for row in count_rows],
        [row["endpoint"] for row in count_rows],
        title="Count controls: frozen retention endpoint",
        ylabel="Mean exact layer x fact completion",
        path=output / "09_count_endpoint_by_condition.png",
    )
    record(
        "09_count_endpoint_by_condition.png",
        "Count controls: frozen retention endpoint",
        "count_conditions.csv",
        ["results/runs/halo_count_causal_controls_v1/summary.json"],
        "Exploratory B0-selected clusters.",
    )
    delta_rows = []
    for condition in COUNT_ORDER[1:]:
        metric = count_stats["count_causal"]["comparisons"][condition]["paired_metrics"][
            "layer_fact_exact_completion_rate"
        ]
        delta_rows.append(
            {
                "condition": condition,
                "label": COUNT_LABELS[condition],
                "mean_delta": metric["mean_difference"],
                "ci_low": metric["confidence_interval"][0],
                "ci_high": metric["confidence_interval"][1],
                "p_value": metric["p_value_two_sided"],
                "holm_p": metric["holm_adjusted_p_value_across_count_controls"],
            }
        )
    _write_csv(data_dir / "count_deltas.csv", delta_rows)
    fig, ax = plt.subplots()
    y = np.arange(len(delta_rows))
    means = np.asarray([row["mean_delta"] for row in delta_rows])
    low = means - np.asarray([row["ci_low"] for row in delta_rows])
    high = np.asarray([row["ci_high"] for row in delta_rows]) - means
    ax.errorbar(means, y, xerr=[low, high], fmt="o", color="#4c78a8", capsize=3)
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_yticks(y, [row["label"] for row in delta_rows])
    ax.set_title("Count controls minus true C")
    ax.set_xlabel("Paired frozen-endpoint difference")
    fig.tight_layout()
    fig.savefig(
        output / "10_count_endpoint_delta_vs_true.png",
        metadata={"Software": "Rineya Significance Lab"},
    )
    plt.close(fig)
    record(
        "10_count_endpoint_delta_vs_true.png",
        "Count controls minus true C",
        "count_deltas.csv",
        ["results/runs/halo_count_causal_controls_v1/statistics.json"],
        "Cluster bootstrap intervals; exploratory and multiplicity-adjusted values remain in CSV.",
    )
    _bar(
        [row["label"] for row in count_rows],
        [row["accuracy"] for row in count_rows],
        title="Count controls: Halo answer accuracy",
        ylabel="Correct / cluster",
        path=output / "11_count_accuracy.png",
        ylim=(0, 1.0),
        color="#e45756",
    )
    record(
        "11_count_accuracy.png",
        "Count controls: Halo answer accuracy",
        "count_conditions.csv",
        ["results/runs/halo_count_causal_controls_v1/summary.json"],
        "Eleven B0-selected clusters; no answer contrast is confirmatory.",
    )
    _bar(
        [row["label"] for row in count_rows],
        [row["exact_reference_trajectories"] for row in count_rows],
        title="Exact trajectory identity with true C",
        ylabel="Cases exact at every diagnostic array",
        path=output / "12_count_trajectory_equivalence.png",
        ylim=(0, 11.8),
        color="#54a24b",
    )
    record(
        "12_count_trajectory_equivalence.png",
        "Exact trajectory identity with true C",
        "count_conditions.csv",
        ["results/runs/halo_count_causal_controls_v1/statistics.json"],
        "Exact byte/value equality for the recorded boundary diagnostics.",
    )
    boundary_data = _jsonl(
        LAB_ROOT / "results" / "runs" / "halo_count_causal_controls_v1" / "boundary_evidence.jsonl"
    )
    boundary_rows = []
    grouped_boundary: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in boundary_data:
        evidence = row["boundary_evidence"]["evidence"]
        if evidence is not None:
            grouped_boundary[(row["condition"], int(row["boundary_index"]))].append(
                float(evidence["layer_fact_exact_completion_rate"])
            )
    for (condition, boundary), values in sorted(grouped_boundary.items()):
        boundary_rows.append(
            {
                "condition": condition,
                "label": COUNT_LABELS[condition],
                "boundary": boundary,
                "scored_rows": len(values),
                "endpoint": float(np.mean(values)),
            }
        )
    _write_csv(data_dir / "count_boundaries.csv", boundary_rows)
    fig, ax = plt.subplots()
    for condition in COUNT_ORDER:
        selected = [row for row in boundary_rows if row["condition"] == condition]
        ax.plot(
            [row["boundary"] for row in selected],
            [row["endpoint"] for row in selected],
            marker="o",
            label=COUNT_LABELS[condition],
        )
    ax.set_title("Observed-fact retention across boundaries")
    ax.set_xlabel("Compression boundary index")
    ax.set_ylabel("Mean exact layer x fact completion")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        output / "13_count_boundary_retention.png", metadata={"Software": "Rineya Significance Lab"}
    )
    plt.close(fig)
    record(
        "13_count_boundary_retention.png",
        "Observed-fact retention across boundaries",
        "count_boundaries.csv",
        ["results/runs/halo_count_causal_controls_v1/boundary_evidence.jsonl"],
        "Only fully observed facts are scored; future facts are excluded.",
    )
    _bar(
        [row["label"] for row in count_rows],
        [row["partial_retention_labels"] for row in count_rows],
        title="Count-control final-state labels",
        ylabel="EVIDENCE_PARTIALLY_RETAINED labels",
        path=output / "14_count_failure_distribution.png",
        color="#b279a2",
    )
    record(
        "14_count_failure_distribution.png",
        "Count-control final-state labels",
        "count_conditions.csv",
        ["results/runs/halo_count_causal_controls_v1/statistics.json"],
        "Descriptive final retained-state labels; they do not assert causal mediation.",
    )

    # Held-constant schedule interaction.
    schedule_cases = _jsonl(
        LAB_ROOT / "results" / "runs" / "halo_schedule_control_v2" / "cases.jsonl"
    )
    schedule_stats = _json(
        LAB_ROOT / "results" / "runs" / "halo_schedule_control_v2" / "statistics.json"
    )
    schedule_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in schedule_cases:
        schedule_groups[(row["method"], int(row["schedule"]["compression_rounds"]))].append(row)
    schedule_rows = []
    for (method, rounds), values in sorted(schedule_groups.items()):
        schedule_rows.append(
            {
                "method": method,
                "rounds": rounds,
                "n": len(values),
                "endpoint": float(
                    np.mean([row["evidence"]["layer_fact_exact_completion_rate"] for row in values])
                ),
                "accuracy": float(np.mean([bool(row["correct"]) for row in values])),
                "wall_seconds": float(
                    np.mean([row["efficiency"]["wall_seconds"] for row in values])
                ),
                "persistent_mib": float(
                    np.mean([row["efficiency"]["total_persistent_bytes"] for row in values])
                )
                / 2**20,
                "query_token_evaluations": float(
                    np.mean(
                        [row["query_access"]["query_probe_token_evaluations"] for row in values]
                    )
                ),
                "final_prefill_entries": float(
                    np.mean([row["schedule"]["final_resident_entries"] for row in values])
                ),
            }
        )
    _write_csv(data_dir / "schedule_rows.csv", schedule_rows)
    contrast_rows = []
    for outcome, label in (
        ("frozen_layer_fact_exact_completion", "endpoint"),
        ("answer_accuracy", "accuracy"),
    ):
        by_round = schedule_stats["outcomes"][outcome]["method_contrast_by_common_round_count"]
        for rounds, values in sorted(by_round.items(), key=lambda item: int(item[0])):
            contrast_rows.append(
                {
                    "outcome": label,
                    "rounds": int(rounds),
                    "mean_delta": values["mean"],
                    "ci_low": values["confidence_interval"][0],
                    "ci_high": values["confidence_interval"][1],
                    "p_value": values["p_value_two_sided"],
                    "clusters": values["cluster_count"],
                }
            )
    _write_csv(data_dir / "schedule_contrasts.csv", contrast_rows)
    for outcome, filename, title, ylabel in (
        (
            "endpoint",
            "15_schedule_endpoint_contrast.png",
            "B3-B2 retention contrast by rounds",
            "Frozen-endpoint difference",
        ),
        (
            "accuracy",
            "17_schedule_accuracy_contrast.png",
            "B3-B2 answer contrast by rounds",
            "Accuracy difference",
        ),
    ):
        rows = [row for row in contrast_rows if row["outcome"] == outcome]
        x = np.asarray([row["rounds"] for row in rows])
        y = np.asarray([row["mean_delta"] for row in rows])
        lo = y - np.asarray([row["ci_low"] for row in rows])
        hi = np.asarray([row["ci_high"] for row in rows]) - y
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", capsize=3, color="#4c78a8")
        ax.axhline(0, color="0.35", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Common realized compression rounds")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(output / filename, metadata={"Software": "Rineya Significance Lab"})
        plt.close(fig)
        record(
            filename,
            title,
            "schedule_contrasts.csv",
            ["results/runs/halo_schedule_control_v2/statistics.json"],
            "Cluster-level exploratory contrast; rows are not independent replicates.",
        )
    slopes = schedule_stats["outcomes"]["frozen_layer_fact_exact_completion"][
        "linear_interaction_per_additional_compression_round"
    ]
    slope_rows = [
        {"cluster": key, "slope": value} for key, value in slopes["cluster_values"].items()
    ]
    _write_csv(data_dir / "schedule_slopes.csv", slope_rows)
    fig, ax = plt.subplots()
    values = np.asarray([row["slope"] for row in slope_rows])
    ax.scatter(np.arange(len(values)), values, color="#4c78a8")
    ax.axhline(slopes["mean"], color="#e45756", label=f"mean {slopes['mean']:.4f}")
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_title("Per-cluster B3-B2 retention interaction slopes")
    ax.set_xlabel("Independent cluster (sorted)")
    ax.set_ylabel("Endpoint delta per added round")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        output / "16_schedule_endpoint_slopes.png", metadata={"Software": "Rineya Significance Lab"}
    )
    plt.close(fig)
    record(
        "16_schedule_endpoint_slopes.png",
        "Per-cluster B3-B2 retention interaction slopes",
        "schedule_slopes.csv",
        ["results/runs/halo_schedule_control_v2/statistics.json"],
        "Exploratory B0-selected clusters.",
    )
    _line_by_method(
        schedule_rows,
        value_key="endpoint",
        title="Frozen retention by realized rounds",
        ylabel="Mean exact layer x fact completion",
        path=output / "18_schedule_absolute_endpoint.png",
    )
    record(
        "18_schedule_absolute_endpoint.png",
        "Frozen retention by realized rounds",
        "schedule_rows.csv",
        ["results/runs/halo_schedule_control_v2/cases.jsonl"],
        "Descriptive cell means; n varies by realized round.",
    )
    _line_by_method(
        schedule_rows,
        value_key="accuracy",
        title="Answer accuracy by realized rounds",
        ylabel="Mean exact accuracy",
        path=output / "19_schedule_absolute_accuracy.png",
    )
    record(
        "19_schedule_absolute_accuracy.png",
        "Answer accuracy by realized rounds",
        "schedule_rows.csv",
        ["results/runs/halo_schedule_control_v2/cases.jsonl"],
        "Descriptive B0-selected cell means; sample size and family composition vary by round.",
    )
    _line_by_method(
        schedule_rows,
        value_key="wall_seconds",
        title="Local CPU time by realized rounds",
        ylabel="Mean wall seconds / cell",
        path=output / "20_schedule_cpu_time.png",
    )
    record(
        "20_schedule_cpu_time.png",
        "Local CPU time by realized rounds",
        "schedule_rows.csv",
        ["results/runs/halo_schedule_control_v2/cases.jsonl"],
        "Local CPU cell means; sample size and family composition vary by round; not deployment or GPU latency.",
    )
    fig, ax = plt.subplots()
    for method in ("B2", "B3"):
        selected = [row for row in schedule_rows if row["method"] == method]
        ax.scatter(
            [row["wall_seconds"] for row in selected],
            [row["endpoint"] for row in selected],
            label=method,
            color=METHOD_COLORS[method],
        )
        for row in selected:
            ax.annotate(
                str(row["rounds"]),
                (row["wall_seconds"], row["endpoint"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title("Retention vs local CPU time")
    ax.set_xlabel("Mean local wall seconds / cell")
    ax.set_ylabel("Mean exact layer x fact completion")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        output / "21_schedule_quality_vs_cpu_time.png",
        metadata={"Software": "Rineya Significance Lab"},
    )
    plt.close(fig)
    record(
        "21_schedule_quality_vs_cpu_time.png",
        "Retention vs local CPU time",
        "schedule_rows.csv",
        ["results/runs/halo_schedule_control_v2/cases.jsonl"],
        "Descriptive tradeoff scatter, not a Pareto frontier; round cells have changing n/family composition.",
    )
    _line_by_method(
        schedule_rows,
        value_key="query_token_evaluations",
        title="Query-probe cost by realized rounds",
        ylabel="Mean query-token evaluations",
        path=output / "22_schedule_query_cost.png",
    )
    record(
        "22_schedule_query_cost.png",
        "Query-probe cost by realized rounds",
        "schedule_rows.csv",
        ["results/runs/halo_schedule_control_v2/cases.jsonl"],
        "Exact query-access accounting.",
    )
    _line_by_method(
        schedule_rows,
        value_key="final_prefill_entries",
        title="Matched final-prefill cache entries",
        ylabel="Resident entries after prefill",
        path=output / "23_schedule_persistent_bytes.png",
    )
    record(
        "23_schedule_persistent_bytes.png",
        "Matched final-prefill cache entries",
        "schedule_rows.csv",
        ["results/runs/halo_schedule_control_v2/cases.jsonl"],
        "The prefill entry budget is matched at 192; post-decode tensor bytes may vary with generated length.",
    )
    count_cases = _jsonl(
        LAB_ROOT / "results" / "runs" / "halo_count_causal_controls_v1" / "cases.jsonl"
    )
    r4_rows = []
    for condition in COUNT_ORDER:
        selected = [
            row
            for row in count_cases
            if row["condition"] == condition and row["task_family"] == "R4"
        ]
        r4_rows.append(
            {
                "condition": condition,
                "label": COUNT_LABELS[condition],
                "clusters": len(selected),
                "accuracy": float(np.mean([bool(row["correct"]) for row in selected])),
            }
        )
    _write_csv(data_dir / "count_r4.csv", r4_rows)
    _bar(
        [row["label"] for row in r4_rows],
        [row["accuracy"] for row in r4_rows],
        title="R4 correction-task accuracy by count control",
        ylabel="Correct / R4 cluster",
        path=output / "24_count_r4_correction_accuracy.png",
        ylim=(0, 1.08),
        color="#b279a2",
    )
    record(
        "24_count_r4_correction_accuracy.png",
        "R4 correction-task accuracy by count control",
        "count_r4.csv",
        ["results/runs/halo_count_causal_controls_v1/cases.jsonl"],
        "Four B0-selected R4 clusters; not a direct stale-token survival measure.",
    )

    # Boundary-local adaptive layer/head allocation screen.
    adaptive_rows_raw = _jsonl(
        LAB_ROOT
        / "results"
        / "runs"
        / "halo_adaptive_budget_screen_v1"
        / "rows.jsonl"
    )
    adaptive_stats = _json(
        LAB_ROOT
        / "results"
        / "runs"
        / "halo_adaptive_budget_screen_v1"
        / "statistics.json"
    )
    adaptive_primary = [
        row for row in adaptive_rows_raw if int(row["boundary_index"]) == 2
    ]
    adaptive_allocation_rows: list[dict[str, Any]] = []
    adaptive_endpoint_rows: list[dict[str, Any]] = []
    for strategy in sorted({str(row["strategy"]) for row in adaptive_primary}):
        selected = [row for row in adaptive_primary if row["strategy"] == strategy]
        layer_values = np.mean(
            np.asarray([row["allocation_by_layer_mean"] for row in selected], dtype=float),
            axis=0,
        )
        head_values = np.mean(
            np.asarray([row["allocation_by_kv_head_mean"] for row in selected], dtype=float),
            axis=0,
        )
        adaptive_allocation_rows.extend(
            {
                "strategy": strategy,
                "unit": "layer",
                "index": index,
                "mean_entries": value,
            }
            for index, value in enumerate(layer_values)
        )
        adaptive_allocation_rows.extend(
            {
                "strategy": strategy,
                "unit": "kv_head",
                "index": index,
                "mean_entries": value,
            }
            for index, value in enumerate(head_values)
        )
        adaptive_endpoint_rows.append(
            {
                "strategy": strategy,
                "clusters": len(selected),
                "endpoint": float(
                    np.mean([row["layer_fact_exact_completion_rate"] for row in selected])
                ),
            }
        )
    _write_csv(data_dir / "adaptive_allocations.csv", adaptive_allocation_rows)
    _write_csv(data_dir / "adaptive_endpoints.csv", adaptive_endpoint_rows)
    for unit, filename, title, xlabel in (
        (
            "layer",
            "25_adaptive_layer_allocation.png",
            "Adaptive allocation by layer",
            "Layer index",
        ),
        (
            "kv_head",
            "26_adaptive_head_allocation.png",
            "Adaptive allocation by KV head",
            "KV-head index",
        ),
    ):
        fig, ax = plt.subplots()
        for strategy in sorted({row["strategy"] for row in adaptive_allocation_rows}):
            selected = sorted(
                (
                    row
                    for row in adaptive_allocation_rows
                    if row["strategy"] == strategy and row["unit"] == unit
                ),
                key=lambda row: row["index"],
            )
            ax.plot(
                [row["index"] for row in selected],
                [row["mean_entries"] for row in selected],
                marker="o" if unit == "kv_head" else None,
                label=strategy,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean allocated entries")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, metadata={"Software": "Rineya Significance Lab"})
        plt.close(fig)
        record(
            filename,
            title,
            "adaptive_allocations.csv",
            ["results/runs/halo_adaptive_budget_screen_v1/rows.jsonl"],
            "Boundary-local open-loop allocation; sparse physical storage and closed-loop outputs are unmeasured.",
        )
    adaptive_delta_rows = []
    for strategy, values in adaptive_stats["comparisons"].items():
        adaptive_delta_rows.append(
            {
                "strategy": strategy,
                "mean_delta": values["mean_difference"],
                "ci_low": values["confidence_interval"][0],
                "ci_high": values["confidence_interval"][1],
                "p_value": values["p_value_two_sided"],
                "holm_p": values["holm_adjusted_p_value"],
            }
        )
    _write_csv(data_dir / "adaptive_deltas.csv", adaptive_delta_rows)
    fig, ax = plt.subplots()
    y = np.arange(len(adaptive_delta_rows))
    means = np.asarray([row["mean_delta"] for row in adaptive_delta_rows])
    ax.errorbar(
        means,
        y,
        xerr=[
            means - np.asarray([row["ci_low"] for row in adaptive_delta_rows]),
            np.asarray([row["ci_high"] for row in adaptive_delta_rows]) - means,
        ],
        fmt="o",
        capsize=3,
        color="#4c78a8",
    )
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_yticks(y, [row["strategy"] for row in adaptive_delta_rows])
    ax.set_title("Adaptive allocation minus uniform")
    ax.set_xlabel("Boundary-local frozen-endpoint difference")
    fig.tight_layout()
    fig.savefig(
        output / "27_adaptive_endpoint_delta.png",
        metadata={"Software": "Rineya Significance Lab"},
    )
    plt.close(fig)
    record(
        "27_adaptive_endpoint_delta.png",
        "Adaptive allocation minus uniform",
        "adaptive_deltas.csv",
        ["results/runs/halo_adaptive_budget_screen_v1/statistics.json"],
        "Open-loop boundary-local endpoint; shuffled-statistic control is strongest, so specificity gate fails.",
    )
    _bar(
        [row["strategy"] for row in adaptive_endpoint_rows],
        [row["endpoint"] for row in adaptive_endpoint_rows],
        title="Boundary-local retention by allocation strategy",
        ylabel="Mean exact layer x fact completion",
        path=output / "28_adaptive_endpoint_by_strategy.png",
    )
    record(
        "28_adaptive_endpoint_by_strategy.png",
        "Boundary-local retention by allocation strategy",
        "adaptive_endpoints.csv",
        ["results/runs/halo_adaptive_budget_screen_v1/rows.jsonl"],
        "Open-loop, B0-selected, boundary-local evidence only; no answer-quality claim.",
    )

    artifact_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "figure_manifest.json"
    }
    figures.sort(key=lambda row: str(row["file"]))
    manifest = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "VALID_HALO_EXPLORATORY_FIGURES",
        "figure_count": len(figures),
        "figures": figures,
        "source_hashes": {
            rel: sha256_file(LAB_ROOT / rel)
            for rel in sorted({source for figure in figures for source in figure["sources"]})
        },
        "integrity": {
            "algorithm": "sha256-canonical-path-map-v1",
            "artifact_files": artifact_hashes,
            "artifact_tree_sha256": content_tree_sha256(artifact_hashes),
            "generator": "scripts/make_halo_control_figures.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_limit": "Figures are exploratory unless their individual claim limit says engineering-only; no confirmatory claim.",
    }
    atomic_write_json(output / "figure_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "results" / "figures" / "halo_controls_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build(args.output, overwrite=args.overwrite)
    print(
        json.dumps({"status": result["status"], "figure_count": result["figure_count"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
