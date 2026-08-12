from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.analysis.statistics import holm_adjust, paired_binary_statistics
from consensus_lab.evaluation.summary import summarize_rows
from consensus_lab.io import read_jsonl, write_json_atomic
from consensus_lab.paths import LabPaths


RUNS = {
    "hard_sweep": LAB_ROOT / "results/runs/v22_hard_vote_primary_smoke.jsonl",
    "fusion": LAB_ROOT / "results/runs/v22_fusion_comparators_smoke.jsonl",
    "negative": LAB_ROOT / "results/runs/v22_negative_controls_smoke.jsonl",
    "eos_strict": LAB_ROOT / "results/runs/v22_eos_strict_majority_smoke.jsonl",
    "eos_confidence": LAB_ROOT / "results/runs/v22_eos_confidence_threshold_smoke.jsonl",
    "eos_minimum": LAB_ROOT / "results/runs/v22_eos_minimum_tokens_smoke.jsonl",
}

PLOT_CONDITIONS = (
    ("v22_hard_vote_primary", "backbone_k1", "hard_plurality", "Backbone (n=16)", "K=1 controls"),
    ("v22_hard_vote_primary", "track_a_k1", "hard_plurality", "Track A (n=16)", "K=1 controls"),
    ("v22_hard_vote_primary", "abstention_k1", "hard_plurality", "Abstention specialist (n=16)", "K=1 controls"),
    ("v22_hard_vote_primary", "complement_k2", "hard_plurality", "Complement K=2 hard (n=16)", "Hard committees"),
    ("v22_hard_vote_primary", "specialist_k3", "hard_plurality", "Specialist K=3 hard (n=16)", "Hard committees"),
    ("v22_hard_vote_primary", "specialist_k5", "hard_plurality", "Specialist K=5 hard (n=16)", "Hard committees"),
    ("v22_hard_vote_primary", "full_k7", "hard_plurality", "Full K=7 hard (n=16)", "Hard committees"),
    ("v22_fusion_comparators", "full_k7", "probability_mean", "K=7 probability mean (n=8)", "Fusion comparators"),
    ("v22_fusion_comparators", "full_k7", "calibrated_logit_mean", "K=7 calibrated logits (n=8)", "Fusion comparators"),
    ("v22_fusion_comparators", "full_k7", "sequence_medoid", "K=7 sequence medoid (n=8)", "Fusion comparators"),
    ("v22_negative_controls", "fixed_random_proxy_k3", "hard_plurality", "Random-proxy K=3 hard (n=8)", "Negative controls"),
    ("v22_negative_controls", "weak_injection_k3", "hard_plurality", "Weak-injection K=3 hard (n=8)", "Negative controls"),
)


def _indexed(rows: list[dict[str, Any]], committee: str, method: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["example_id"]): row
        for row in rows
        if row["committee"] == committee and row["method"] == method
    }


def _comparison(
    *,
    label: str,
    rows_a: dict[str, dict[str, Any]],
    rows_b: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    ids = sorted(set(rows_a) & set(rows_b))
    if not ids:
        raise ValueError(f"No paired rows for {label}")
    stats = paired_binary_statistics(
        [bool(rows_a[value]["correct"]) for value in ids],
        [bool(rows_b[value]["correct"]) for value in ids],
        seed=seed,
    ).to_dict()
    return {"label": label, "paired_example_ids": ids, **stats}


def _selected_plot_points(summaries: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, str]]:
    index = {
        (str(row["run_name"]), str(row["committee"]), str(row["method"])): row
        for row in summaries
    }
    return [
        (index[(run_name, committee, method)], display_name, family)
        for run_name, committee, method, display_name, family in PLOT_CONDITIONS
    ]


def _plot_frontier(
    summaries: list[dict[str, Any]],
    x_key: str,
    label: str,
    output: Path,
    *,
    scale: float = 1.0,
) -> None:
    points = [
        (row, display_name, family)
        for row, display_name, family in _selected_plot_points(summaries)
        if row.get(x_key) is not None and row.get("correctness") is not None
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = plt.get_cmap("tab20").colors
    markers = {"K=1 controls": "o", "Hard committees": "s", "Fusion comparators": "^", "Negative controls": "X"}
    for index, (row, display_name, family) in enumerate(points):
        ax.scatter(
            float(row[x_key]) / scale,
            float(row["correctness"]),
            s=70,
            color=colors[index % len(colors)],
            marker=markers[family],
            edgecolor="white",
            linewidth=0.6,
            label=display_name,
            zorder=3,
        )
    ax.set_xlabel(label)
    ax.set_ylabel("Exact correctness")
    fig.suptitle(f"DEV pilot quality vs {label.lower()}", y=0.98, fontsize=16)
    fig.text(
        0.38,
        0.925,
        "Curated conditions for legibility; n=16 hard sweep, n=8 fusion/control packets",
        ha="center",
        fontsize=8,
        color="dimgray",
    )
    ax.set_ylim(-0.025, 0.55)
    ax.grid(alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, title="Condition")
    fig.tight_layout(rect=(0, 0, 0.76, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_committee_scaling(summaries: list[dict[str, Any]], output: Path) -> None:
    index = {
        (str(row["run_name"]), str(row["committee"]), str(row["method"])): row
        for row in summaries
    }
    curve = [
        (1, "abstention_k1", "Best K=1\n(abstention)"),
        (2, "complement_k2", "Complement\nK=2"),
        (3, "specialist_k3", "Specialist\nK=3"),
        (5, "specialist_k5", "Specialist\nK=5"),
        (7, "full_k7", "Full\nK=7"),
    ]
    rates = [
        float(index[("v22_hard_vote_primary", committee, "hard_plurality")]["correctness"])
        for _, committee, _ in curve
    ]
    sizes = [size for size, _, _ in curve]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(sizes, rates, color="#355c9a", marker="o", linewidth=2, markersize=7, label="Constructed hard-vote curve")
    offsets = [(8, -28), (0, 12), (0, -30), (0, 12), (-18, 12)]
    for (size, _, name), rate, offset in zip(curve, rates, offsets, strict=True):
        ax.annotate(name, (size, rate), xytext=offset, textcoords="offset points", fontsize=8, ha="center")
    for committee, label, marker, color in (
        ("track_a_k1", "Track A K=1", "D", "#e6862e"),
        ("backbone_k1", "Backbone K=1", "X", "#777777"),
    ):
        rate = float(index[("v22_hard_vote_primary", committee, "hard_plurality")]["correctness"])
        ax.scatter(1, rate, s=70, marker=marker, color=color, label=label, zorder=3)
    ax.set_xlabel("Committee size K")
    ax.set_ylabel("Exact correctness")
    fig.suptitle("DEV hard-vote quality vs committee size", y=0.98, fontsize=16)
    fig.text(0.5, 0.925, "Paired balanced 16-example packet", ha="center", fontsize=8, color="dimgray")
    ax.set_xticks([1, 2, 3, 5, 7])
    ax.set_ylim(-0.04, 0.58)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    paths = LabPaths.discover(LAB_ROOT)
    loaded = {name: read_jsonl(path) for name, path in RUNS.items()}
    hard = loaded["hard_sweep"]
    fusion = loaded["fusion"]
    individuals = [
        "backbone_k1",
        "track_a_k1",
        "track_b_k1",
        "general_k1",
        "grounding_k1",
        "procedure_k1",
        "abstention_k1",
        "social_k1",
    ]
    hard_maps = {name: _indexed(hard, name, "hard_plurality") for name in individuals + ["full_k7", "specialist_k5"]}
    individual_rates = {
        name: sum(bool(row["correct"]) for row in values.values()) / len(values)
        for name, values in hard_maps.items()
        if name in individuals
    }
    adapter_rates = {name: rate for name, rate in individual_rates.items() if name != "backbone_k1"}
    best_individual = max(individual_rates, key=lambda name: (individual_rates[name], name))
    comparisons = [
        _comparison(
            label="full_k7 hard vs backbone_k1",
            rows_a=hard_maps["full_k7"],
            rows_b=hard_maps["backbone_k1"],
            seed=20260811,
        ),
        _comparison(
            label=f"full_k7 hard vs best pilot individual ({best_individual})",
            rows_a=hard_maps["full_k7"],
            rows_b=hard_maps[best_individual],
            seed=20260812,
        ),
    ]
    fusion_hard = _indexed(fusion, "full_k7", "hard_plurality")
    for offset, method in enumerate(
        [
            "weighted_hard",
            "probability_mean",
            "logit_mean",
            "calibrated_logit_mean",
            "confidence_weighted_probability",
            "confidence_weighted_logit",
            "sequence_medoid",
        ],
        start=10,
    ):
        comparisons.append(
            _comparison(
                label=f"full_k7 hard vs {method}",
                rows_a=fusion_hard,
                rows_b=_indexed(fusion, "full_k7", method),
                seed=20260811 + offset,
            )
        )
    adjusted = holm_adjust([float(row["mcnemar_exact_p"]) for row in comparisons])
    for row, p_value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_mcnemar_p"] = p_value
    negative = loaded["negative"]
    duplicate_checks = []
    for method in ("hard_plurality", "probability_mean", "logit_mean"):
        left = _indexed(negative, "general_k1", method)
        right = _indexed(negative, "duplicate_general_k3", method)
        ids = sorted(set(left) & set(right))
        duplicate_checks.append(
            {
                "method": method,
                "paired_n": len(ids),
                "exact_token_sequence_match_rate": sum(
                    left[value]["output_token_ids"] == right[value]["output_token_ids"] for value in ids
                )
                / len(ids),
            }
        )
    all_rows = [row for rows in loaded.values() for row in rows]
    summaries = summarize_rows(all_rows)
    packet = {
        "schema_version": "1.0",
        "status": "BOUNDED_DEV_PILOT_NOT_FINAL_INFERENCE",
        "source_runs": {name: str(path) for name, path in RUNS.items()},
        "hard_sweep_case_count": len(hard_maps["full_k7"]),
        "fusion_case_count": len(fusion_hard),
        "best_individual_on_hard_packet": best_individual,
        "individual_correctness": individual_rates,
        "mean_adapter_correctness": sum(adapter_rates.values()) / len(adapter_rates),
        "paired_comparisons": comparisons,
        "duplicate_expert_invariants": duplicate_checks,
        "warnings": [
            "All intervals and tests are underpowered pilot estimates.",
            "Best-individual identification and method selection used DEV and cannot be validated on the same packet.",
            "Holm adjustment covers the paired comparisons listed in this artifact.",
        ],
    }
    output = paths.assert_output_path(LAB_ROOT / "results/summaries/pilot_statistics.json")
    write_json_atomic(output, packet)
    csv_path = paths.assert_output_path(LAB_ROOT / "results/summaries/pilot_paired_comparisons.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [key for key in comparisons[0] if key != "paired_example_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in comparisons)
    plots = LAB_ROOT / "reports/plots"
    for key, label, filename, scale in (
        ("mean_latency_seconds", "Mean latency (seconds/example)", "quality_vs_latency.png", 1.0),
        ("peak_gpu_vram_bytes", "Peak allocated VRAM (MiB)", "quality_vs_vram.png", 1024**2),
        ("adapter_storage_bytes", "Adapter storage (MiB)", "quality_vs_adapter_storage.png", 1024**2),
        ("mean_forward_passes", "Mean expert-token forwards", "quality_vs_forward_passes.png", 1.0),
    ):
        _plot_frontier(summaries, key, label, plots / filename, scale=scale)
    _plot_committee_scaling(summaries, plots / "quality_vs_committee_size.png")
    print(json.dumps({"output": str(output), "comparisons": len(comparisons), "plots": 5}, indent=2))


if __name__ == "__main__":
    main()
