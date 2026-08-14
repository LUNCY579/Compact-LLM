from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402


LAB_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from consensus_lab.io import read_json, read_jsonl, sha256_file, write_json_atomic  # noqa: E402


EXISTING = LAB_ROOT / "results/phase2_mechanism/existing_results_analysis.json"
TEACHER = LAB_ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json"
HARD_TIE_TEMP = LAB_ROOT / "results/phase2_mechanism/hard_temperature_tie_sensitivity_dev.json"
COMMITTEE_DIAGNOSTICS = LAB_ROOT / "results/phase2_mechanism/teacher_forced_committee_diagnostics.json"
FREE_RUN = LAB_ROOT / "results/phase2_mechanism/selected_dev_free_run_analysis.json"
SELECTED_COMMITTEE = LAB_ROOT / "results/phase2_mechanism/selected_committee_dev_analysis.json"
PILOT = LAB_ROOT / "results/archive/sealed_n32/v22_frozen_finalists.jsonl"
OUTPUT = LAB_ROOT / "paper/figures"
METADATA = OUTPUT / "figure_metadata.json"

COLORS = {
    "blue": "#386CB0",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#8E6CBE",
    "teal": "#2A9D8F",
    "gray": "#7A7A7A",
    "light": "#D8D8D8",
    "dark": "#222222",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    svg = OUTPUT / f"{stem}.svg"
    png = OUTPUT / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "svg": str(svg.relative_to(LAB_ROOT)),
        "svg_sha256": sha256_file(svg),
        "png": str(png.relative_to(LAB_ROOT)),
        "png_sha256": sha256_file(png),
    }


def add_bar_labels(ax: plt.Axes, bars: Any, *, fmt: Callable[[float], str] | None = None) -> None:
    formatter = fmt or (lambda value: f"{value:.1f}")
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def architecture_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.3)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str, fontsize: float = 8.5) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color, edgecolor=COLORS["dark"], linewidth=0.8,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=COLORS["gray"]))

    box(0.2, 3.0, 1.25, 1.0, "Prompt +\naccepted prefix", "#EAF0F8")
    box(1.85, 3.0, 1.3, 1.0, "Shared frozen\n344M backbone", "#EAF0F8")
    arrow(1.45, 3.5, 1.85, 3.5)
    experts = ["Track A", "Track B", "General", "Grounding", "Procedure", "Abstention", "Social"]
    y_values = np.linspace(6.25, 0.45, len(experts))
    for index, (expert, y) in enumerate(zip(experts, y_values, strict=True)):
        box(3.75, y, 1.25, 0.62, f"{expert}\nLoRA", "#F3EAF8" if expert not in {"Abstention"} else "#FCECD9", 7.5)
        arrow(3.15, 3.5, 3.75, y + 0.31)
        box(5.55, y, 1.1, 0.62, "logits → top-1", "#E8F4EC", 7.5)
        arrow(5.0, y + 0.31, 5.55, y + 0.31)
        arrow(6.65, y + 0.31, 7.25, 3.5)
    box(7.18, 2.85, 1.5, 1.3, "Hard plurality\nconfidence tie-break\n(or comparator)", "#FFF2CC", 7.4)
    box(9.0, 3.0, 0.8, 1.0, "Next\ntoken", "#EAF0F8")
    arrow(8.6, 3.5, 9.0, 3.5)
    ax.plot([9.4, 9.4, 0.85], [3.0, 0.18, 0.18], color=COLORS["gray"], lw=1.0)
    arrow(0.85, 0.18, 0.85, 2.98)
    ax.text(
        5.1,
        0.18,
        "Selected token is appended; all experts condition on the same new prefix",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.set_title("Hard token-level multi-adapter consensus architecture", pad=10, fontweight="bold")
    return fig


def primary_accuracy_figure(existing: dict[str, Any]) -> plt.Figure:
    behavior = existing["behavior"]
    keys = ["backbone", "track_a", "abstention", "probability_k7", "hard_k7"]
    labels = ["Backbone\nK=1", "Track A\nK=1", "Abstention\nK=1", "Probability\nK=7", "Hard\nK=7"]
    values = [100 * behavior[key]["accuracy"] for key in keys]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["green"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    add_bar_labels(ax, bars, fmt=lambda value: f"{value:.2f}%")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_title("Frozen v2.2 TEST cornerstone (N=4,050)", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.2)
    return fig


def hard_vs_soft_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    hard_votes = [2, 1]
    probability_means = [37, 63]
    labels = ["Token X", "Token Y"]
    colors = [COLORS["blue"], COLORS["orange"]]
    bars = axes[0].bar(labels, hard_votes, color=colors)
    add_bar_labels(
        axes[0],
        bars,
        fmt=lambda value: f"{int(value)} vote" if int(value) == 1 else f"{int(value)} votes",
    )
    axes[0].set_ylim(0, 2.7)
    axes[0].set_ylabel("Equal expert votes")
    axes[0].set_title("Hard plurality selects X", fontweight="bold")
    axes[0].text(0.5, 2.52, "A: X@0.55   B: X@0.55   C: Y@0.99", ha="center", fontsize=8)
    soft_bars = axes[1].bar(labels, probability_means, color=colors)
    add_bar_labels(axes[1], soft_bars, fmt=lambda value: f"{value:.0f}% mean mass")
    axes[1].set_ylim(0, 75)
    axes[1].set_ylabel("Mean probability (%)")
    axes[1].set_title("Probability mean selects Y", fontweight="bold")
    axes[1].text(0.5, 70, "Confidence magnitude outweighs 2-to-1 support", ha="center", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Synthetic hard-versus-soft fusion example", fontweight="bold")
    fig.tight_layout()
    return fig


def pilot_full_figure(existing: dict[str, Any]) -> plt.Figure:
    pilot_rows = read_jsonl(PILOT)
    conditions = {
        ("backbone_k1", "hard_plurality"): "Backbone",
        ("track_a_k1", "hard_plurality"): "Track A",
        ("abstention_k1", "hard_plurality"): "Abstention",
        ("full_k7", "probability_mean"): "Probability K=7",
        ("full_k7", "hard_plurality"): "Hard K=7",
    }
    pilot_values = {}
    for key, label in conditions.items():
        rows = [row for row in pilot_rows if (row["committee"], row["method"]) == key]
        pilot_values[label] = 100 * sum(bool(row["correct"]) for row in rows) / len(rows)
    behavior = existing["behavior"]
    full_values = {
        "Backbone": 100 * behavior["backbone"]["accuracy"],
        "Track A": 100 * behavior["track_a"]["accuracy"],
        "Abstention": 100 * behavior["abstention"]["accuracy"],
        "Probability K=7": 100 * behavior["probability_k7"]["accuracy"],
        "Hard K=7": 100 * behavior["hard_k7"]["accuracy"],
    }
    labels = list(pilot_values)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.3))
    width = 0.36
    left = ax.bar(x - width / 2, [pilot_values[label] for label in labels], width, label="Pilot n=32", color=COLORS["orange"])
    right = ax.bar(x + width / 2, [full_values[label] for label in labels], width, label="Full N=4,050", color=COLORS["blue"])
    add_bar_labels(ax, left, fmt=lambda value: f"{value:.1f}")
    add_bar_labels(ax, right, fmt=lambda value: f"{value:.1f}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_title("Pilot ranking reversal after full-corpus evaluation", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return fig


def committee_scaling_figure(
    teacher: dict[str, Any],
    committee_diagnostics: dict[str, Any],
    selected_committee: dict[str, Any] | None = None,
) -> plt.Figure:
    panel_count = 3 if selected_committee is not None else 2
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=((15.4, 4.5) if selected_committee is not None else (11.2, 4.5)),
        sharex=True,
    )
    ax = axes[0]
    methods = [
        ("hard_plurality", "Hard: mean", COLORS["green"], "o"),
        ("probability_mean", "Probability: mean", COLORS["purple"], "s"),
        ("calibrated_probability_mean", "Calibrated probability: mean", COLORS["blue"], "^"),
    ]
    for method, label, color, marker in methods:
        rows = sorted((row for row in teacher["committee_scaling"] if row["method"] == method), key=lambda row: row["k"])
        ax.plot([row["k"] for row in rows], [100 * row["mean_example_accuracy"] for row in rows], marker=marker, color=color, label=label)
        if method == "hard_plurality":
            ax.plot([row["k"] for row in rows], [100 * row["best_example_accuracy"] for row in rows], marker="D", linestyle="--", color=COLORS["orange"], label="Hard: best subset")
    ax.set_xlabel("Committee size K")
    ax.set_ylabel("All-gold-token example accuracy (%)")
    ax.set_xticks(range(1, 8))
    ax.set_title("Overall all-token accuracy", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    balanced_ax = axes[1]
    balanced_rows = sorted(
        (
            row
            for row in committee_diagnostics["balanced_committee_scaling"]
            if row["method"] == "hard_plurality"
        ),
        key=lambda row: row["k"],
    )
    balanced_ax.plot(
        [row["k"] for row in balanced_rows],
        [100 * row["mean_balanced_answer_class_accuracy"] for row in balanced_rows],
        marker="o",
        color=COLORS["green"],
        label="Hard: mean balanced",
    )
    balanced_ax.plot(
        [row["k"] for row in balanced_rows],
        [100 * row["best_balanced_answer_class_accuracy"] for row in balanced_rows],
        marker="D",
        linestyle="--",
        color=COLORS["orange"],
        label="Hard: best balanced subset",
    )
    balanced_ax.set_xlabel("Committee size K")
    balanced_ax.set_ylabel("Macro answer-class accuracy (%)")
    balanced_ax.set_xticks(range(1, 8))
    balanced_ax.set_title("Balanced answer/abstention capability", fontweight="bold")
    balanced_ax.legend(frameon=False)
    balanced_ax.grid(alpha=0.2)
    if selected_committee is not None:
        free_ax = axes[2]
        rows = sorted(
            selected_committee["selected_best_at_k_scaling"], key=lambda row: row["k"]
        )
        free_ax.plot(
            [row["k"] for row in rows],
            [100 * row["accuracy"] for row in rows],
            marker="X",
            linewidth=2.0,
            color=COLORS["dark"],
            label="Overall exact",
        )
        free_ax.plot(
            [row["k"] for row in rows],
            [100 * row["balanced_answer_class_accuracy"] for row in rows],
            marker="P",
            linewidth=2.0,
            color=COLORS["teal"],
            label="Balanced answer class",
        )
        free_ax.set_xlabel("Committee size K")
        free_ax.set_ylabel("Free-running sequence accuracy (%)")
        free_ax.set_xticks(range(1, 8))
        free_ax.set_title("Selected free-running systems", fontweight="bold")
        free_ax.legend(frameon=False)
        free_ax.grid(alpha=0.2)
    title = "DEV committee-size scaling across all 127 cached subsets"
    if selected_committee is not None:
        title += " and selected free-running systems"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def calibration_figure(teacher: dict[str, Any]) -> plt.Figure:
    experts = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]
    values = teacher["calibration"]["experts"]
    y = np.arange(len(experts))
    accuracy = [100 * values[expert]["top1_accuracy"] for expert in experts]
    confidence = [100 * values[expert]["mean_confidence"] for expert in experts]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    width = 0.36
    ax.barh(y + width / 2, accuracy, width, label="Gold-token top-1 accuracy", color=COLORS["blue"])
    ax.barh(y - width / 2, confidence, width, label="Mean top-1 confidence", color=COLORS["orange"])
    ax.set_yticks(y, [expert.replace("_", " ").title() for expert in experts])
    ax.invert_yaxis()
    ax.set_xlabel("Percent")
    ax.set_title("DEV gold-prefix confidence–accuracy mismatch", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    for index, expert in enumerate(experts):
        ax.text(max(accuracy[index], confidence[index]) + 1, index, f"T={values[expert]['temperature']:.3g}", va="center", fontsize=8)
    ax.set_xlim(0, 105)
    return fig


def calibration_robustness_figure(teacher: dict[str, Any]) -> plt.Figure:
    experts = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]
    values = teacher["calibration"]["experts"]
    labels = {expert: expert.replace("_", " ").title() for expert in experts}
    palette = [
        COLORS["blue"], COLORS["purple"], COLORS["teal"], COLORS["green"],
        COLORS["gray"], COLORS["orange"], COLORS["red"],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))

    mismatch_ax = axes[0, 0]
    for expert, color in zip(experts, palette, strict=True):
        accuracy = 100 * values[expert]["top1_accuracy"]
        confidence = 100 * values[expert]["mean_confidence"]
        mismatch_ax.scatter(accuracy, confidence, s=55, color=color)
        mismatch_ax.annotate(
            f'{labels[expert]}\nT={values[expert]["temperature"]:.3g}',
            (accuracy, confidence), xytext=(4, 4), textcoords="offset points", fontsize=6.6,
        )
    limits = [20, 100]
    mismatch_ax.plot(limits, limits, linestyle="--", color=COLORS["dark"], linewidth=0.9)
    mismatch_ax.set_xlim(limits)
    mismatch_ax.set_ylim(limits)
    mismatch_ax.set_xlabel("Gold-token top-1 accuracy (%)")
    mismatch_ax.set_ylabel("Mean top-1 confidence (%)")
    mismatch_ax.set_title("A. Confidence–accuracy mismatch", fontweight="bold")
    mismatch_ax.grid(alpha=0.2)

    nll_ax = axes[0, 1]
    for expert, color in zip(experts, palette, strict=True):
        points = sorted(
            (float(temperature), float(nll))
            for temperature, nll in values[expert]["mean_nll_by_temperature"].items()
        )
        nll_ax.plot(
            [point[0] for point in points], [point[1] for point in points],
            marker="o", markersize=3, color=color, label=labels[expert],
        )
        fitted = float(values[expert]["temperature"])
        fitted_nll = next(value for temperature, value in points if temperature == fitted)
        nll_ax.scatter(
            [fitted], [fitted_nll], marker="*", s=70, color=color,
            edgecolor="white", linewidth=0.4, zorder=4,
        )
    nll_ax.set_xscale("log", base=2)
    nll_ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    nll_ax.set_xlabel("Temperature")
    nll_ax.set_ylabel("Mean gold-token NLL")
    nll_ax.set_title("B. Expert calibration curves (stars: fitted)", fontweight="bold")
    nll_ax.legend(frameon=False, ncol=2, fontsize=6.6)
    nll_ax.grid(alpha=0.2)

    interventions = teacher["temperature_interventions"]
    hard = 100 * next(
        row for row in teacher["committee_results"]
        if row["mask"] == 127 and row["method"] == "hard_plurality"
    )["example_accuracy"]

    shared_ax = axes[1, 0]
    shared_points = sorted(
        (float(key), 100 * value["example_accuracy"])
        for key, value in interventions["shared_temperature"].items()
    )
    shared_ax.plot(
        [point[0] for point in shared_points], [point[1] for point in shared_points],
        marker="o", color=COLORS["purple"], label="Probability mean: shared T",
    )
    shared_ax.axhline(
        hard, color=COLORS["red"], linestyle="--", linewidth=1.0,
        label="Hard raw reference",
    )
    shared_ax.set_xscale("log", base=2)
    shared_ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    shared_ax.set_xlabel("Shared temperature")
    shared_ax.set_ylabel("All-gold-token example accuracy (%)")
    shared_ax.set_title("C. Shared-temperature soft fusion", fontweight="bold")
    shared_ax.legend(frameon=False)
    shared_ax.grid(alpha=0.2)

    perturb_ax = axes[1, 1]
    for expert, color in zip(experts, palette, strict=True):
        points = sorted(
            (float(key), 100 * value["example_accuracy"])
            for key, value in interventions["one_expert_perturbed"][expert].items()
        )
        perturb_ax.plot(
            [point[0] for point in points], [point[1] for point in points],
            marker="o", markersize=3, color=color, label=labels[expert],
        )
    perturb_ax.axhline(
        hard, color=COLORS["dark"], linestyle="--", linewidth=1.0,
        label="Hard raw reference",
    )
    perturb_ax.set_xscale("log", base=2)
    perturb_ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    perturb_ax.set_xlabel("One-expert temperature")
    perturb_ax.set_ylabel("All-gold-token example accuracy (%)")
    perturb_ax.set_title("D. One-expert perturbations", fontweight="bold")
    perturb_ax.legend(frameon=False, ncol=2, fontsize=6.4)
    perturb_ax.grid(alpha=0.2)

    fig.suptitle(
        "DEV calibration and temperature robustness of K=7 aggregation",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def rescue_harm_figure(existing: dict[str, Any]) -> plt.Figure:
    global_rows = {
        row["comparison"]: row
        for row in existing["rescue_harm"]
        if row["dimension"] == "global"
    }
    specifications = [
        ("hard_vs_track_a", "A. Track A to Hard K=7"),
        ("hard_vs_probability", "B. Probability K=7 to Hard K=7"),
    ]
    cell_colors = {
        "both_correct": "#D8E5F2",
        "harm": "#F4C7C7",
        "rescue": "#CDE8C8",
        "both_wrong": "#E4E4E4",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, (key, title) in zip(axes, specifications, strict=True):
        row = global_rows[key]
        cells = [
            (0, 0, "both_correct", row["both_correct"], "Both correct"),
            (1, 0, "harm", row["harm"], "Harm"),
            (0, 1, "rescue", row["rescue"], "Rescue"),
            (1, 1, "both_wrong", row["both_wrong"], "Both wrong"),
        ]
        for x, y, state, count, label in cells:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=cell_colors[state],
                    edgecolor="white",
                    linewidth=2,
                )
            )
            ax.text(
                x + 0.5,
                y + 0.5,
                f"{label}\n{count:,} ({100 * count / row['n']:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if state in {"rescue", "harm"} else "normal",
            )
        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.5], ["Hard correct", "Hard wrong"])
        ax.set_yticks([0.5, 1.5], ["Baseline correct", "Baseline wrong"])
        ax.set_title(title, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.legend(
        handles=[
            Patch(facecolor=cell_colors["rescue"], label="Rescue: baseline wrong, Hard correct"),
            Patch(facecolor=cell_colors["harm"], label="Harm: baseline correct, Hard wrong"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Paired v2.2 TEST state transitions: net gain is rescue minus harm",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


def shapley_figure(teacher: dict[str, Any]) -> plt.Figure:
    values = teacher["shapley"]["hard_plurality"]["shapley"]
    ordered = sorted(values, key=lambda expert: values[expert])
    contributions = [100 * values[expert] for expert in ordered]
    colors = [COLORS["red"] if value < 0 else COLORS["green"] for value in contributions]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.barh([expert.replace("_", " ").title() for expert in ordered], contributions, color=colors)
    ax.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xlabel("Exact Shapley contribution (percentage points)")
    ax.set_title("Hard-vote marginal utility over all 127 DEV committees", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, contributions, strict=True):
        if value >= 0:
            x, alignment, color = value + 0.18, "left", COLORS["dark"]
        else:
            x, alignment, color = -0.08, "right", "white"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            ha=alignment,
            va="center",
            fontsize=8,
            color=color,
        )
    return fig


def coalition_heatmap_figure(existing: dict[str, Any]) -> plt.Figure:
    summary = existing["coalitions"]["coalition_summary"]
    top_masks = [row["mask"] for row in summary[:10]]
    detail = existing["coalitions"]["coalition_detail"]
    categories = sorted({row["task_category"] for row in detail})
    experts = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]
    labels = {expert: expert.replace("_", " ").title() for expert in experts}
    mask_counts = defaultdict(Counter)
    expert_counts = defaultdict(Counter)
    totals = Counter()
    for row in detail:
        category = row["task_category"]
        count = int(row["token_count"])
        totals[category] += count
        for expert in row["members"]:
            expert_counts[expert][category] += count
        if row["mask"] in top_masks:
            mask_counts[row["mask"]][category] += count
    expert_matrix = np.asarray(
        [
            [100 * expert_counts[expert][category] / totals[category] for category in categories]
            for expert in experts
        ]
    )
    mask_matrix = np.asarray(
        [
            [100 * mask_counts[mask][category] / totals[category] for category in categories]
            for mask in top_masks
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.2), width_ratios=[0.78, 1.05])
    tick_labels = [value.replace("_", " ") for value in categories]

    expert_ax = axes[0]
    expert_image = expert_ax.imshow(
        expert_matrix,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=float(np.percentile(expert_matrix, 98)),
    )
    expert_ax.set_xticks(range(len(categories)), tick_labels, rotation=52, ha="right")
    expert_ax.set_yticks(range(len(experts)), [labels[expert] for expert in experts])
    expert_ax.set_xlabel("Reasoning family")
    expert_ax.set_ylabel("Expert")
    expert_ax.set_title("A. Expert support for the winning token", fontweight="bold")
    fig.colorbar(
        expert_image,
        ax=expert_ax,
        label="Within-family winner-support rate (%)",
        fraction=0.03,
        pad=0.02,
    )

    mask_ax = axes[1]
    mask_image = mask_ax.imshow(
        mask_matrix,
        aspect="auto",
        cmap="Purples",
        vmin=0,
        vmax=float(np.percentile(mask_matrix, 98)),
    )
    mask_ax.set_xticks(range(len(categories)), tick_labels, rotation=52, ha="right")
    mask_ax.set_yticks(range(len(top_masks)), top_masks)
    mask_ax.set_xlabel("Reasoning family")
    mask_ax.set_ylabel("Winner-support mask (Track A to Social)")
    mask_ax.set_title("B. Ten most common coalition masks", fontweight="bold")
    fig.colorbar(
        mask_image,
        ax=mask_ax,
        label="Within-family coalition share (%)",
        fraction=0.03,
        pad=0.02,
    )
    fig.suptitle(
        "TEST winner-support coalitions by reasoning family (descriptive, noncausal)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def divergence_figure(existing: dict[str, Any]) -> plt.Figure:
    records = existing["free_running_divergence"]["records"]
    outcomes = ["hard_only", "probability_only", "both_wrong"]
    labels = ["Hard only correct", "Probability only correct", "Both wrong"]
    buckets = ["t=0", "t=1", "t=2", "t=3-5", "t=6-10"]

    def bucket(t: int) -> str:
        if t == 0:
            return "t=0"
        if t == 1:
            return "t=1"
        if t == 2:
            return "t=2"
        if t <= 5:
            return "t=3-5"
        return "t=6-10"

    values = {
        outcome: Counter(bucket(int(row["first_divergence_t"])) for row in records if row["outcome_state"] == outcome)
        for outcome in outcomes
    }
    x = np.arange(len(buckets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for index, (outcome, label, color) in enumerate(
        zip(outcomes, labels, [COLORS["green"], COLORS["purple"], COLORS["gray"]], strict=True)
    ):
        ax.bar(x + (index - 1) * width, [values[outcome][name] for name in buckets], width, label=label, color=color)
    ax.set_xticks(x, buckets)
    ax.set_xlabel("First shared-prefix hard/soft divergence position")
    ax.set_ylabel("Divergent TEST examples")
    ax.set_title("Hard/soft differences usually begin at the first token", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return fig


def temperature_figure(teacher: dict[str, Any]) -> plt.Figure:
    interventions = teacher["temperature_interventions"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    series = [
        ("Shared", interventions["shared_temperature"], COLORS["gray"], "o"),
        ("Track A only", interventions["one_expert_perturbed"]["track_a"], COLORS["blue"], "s"),
        ("Grounding only", interventions["one_expert_perturbed"]["grounding"], COLORS["green"], "^"),
        ("Abstention only", interventions["one_expert_perturbed"]["abstention"], COLORS["orange"], "D"),
    ]
    for label, mapping, color, marker in series:
        points = sorted((float(key), 100 * value["example_accuracy"]) for key, value in mapping.items())
        ax.plot([point[0] for point in points], [point[1] for point in points], marker=marker, label=label, color=color)
    hard = 100 * next(row for row in teacher["committee_results"] if row["mask"] == 127 and row["method"] == "hard_plurality")["example_accuracy"]
    ax.axhline(
        hard,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.0,
        label="Hard raw reference (unique pluralities invariant)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("Temperature multiplier")
    ax.set_ylabel("All-gold-token example accuracy (%)")
    ax.set_title("Probability fusion changes under calibration interventions", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    return fig


def diversity_figure(teacher: dict[str, Any]) -> plt.Figure:
    pairs = teacher["expert_error_diversity"]["pairwise"]
    experts = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]
    index = {expert: i for i, expert in enumerate(experts)}
    matrix = np.zeros((len(experts), len(experts)))
    for row in pairs:
        left, right = index[row["expert_a"]], index[row["expert_b"]]
        matrix[left, right] = matrix[right, left] = 100 * row["top1_disagreement_rate"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(70, float(matrix.max())))
    labels = [expert.replace("_", " ").title() for expert in experts]
    ax.set_xticks(range(len(experts)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(experts)), labels)
    for i in range(len(experts)):
        for j in range(len(experts)):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] > 45 else COLORS["dark"])
    fig.colorbar(image, ax=ax, label="Top-1 disagreement (%)", fraction=0.046, pad=0.04)
    ax.set_title("Pairwise DEV gold-prefix expert disagreement", fontweight="bold")
    return fig


def quality_compute_figure(existing: dict[str, Any]) -> plt.Figure:
    behavior = existing["behavior"]
    keys = ["backbone", "track_a", "abstention", "probability_k7", "hard_k7"]
    labels = ["Backbone", "Track A", "Abstention", "Probability K=7", "Hard K=7"]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["green"]]
    y = [100 * behavior[key]["accuracy"] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    panel_specs = [
        (
            axes[0],
            [behavior[key]["mean_forward_passes"] for key in keys],
            "Mean expert-token forward proxy",
            "A. Quality versus expert-token forwards",
        ),
        (
            axes[1],
            [behavior[key]["mean_latency_seconds"] for key in keys],
            "Mean latency per example (s)",
            "B. Quality versus measured latency",
        ),
    ]
    for ax, x_values, x_label, title in panel_specs:
        for x_value, y_value, label, color in zip(x_values, y, labels, colors, strict=True):
            ax.scatter(
                [x_value],
                [y_value],
                s=65,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel(x_label)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Exact accuracy (%)")
    fig.suptitle(
        "Frozen v2.2 TEST quality–compute frontier (hardware descriptive)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def teacher_free_figure(free_run: dict[str, Any]) -> plt.Figure:
    rows = free_run["teacher_forced_vs_free_running"]
    labels = {
        "hard_plurality": "Hard",
        "probability_mean": "Raw prob.",
        "calibrated_probability_mean": "Calibrated prob.",
        "reciprocal_rank_top5": "Reciprocal rank",
    }
    x = np.arange(len(rows))
    width = 0.36
    local = [100 * row["teacher_example_advantage_vs_hard"] for row in rows]
    free = [100 * row["free_sequence_advantage_vs_hard"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    local_bars = ax.bar(x - width / 2, local, width, color=COLORS["blue"], label="Gold-prefix all-token example gap")
    free_bars = ax.bar(x + width / 2, free, width, color=COLORS["orange"], label="Free-running exact gap")
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xticks(x, [labels[row["method"]] for row in rows])
    ax.set_ylabel("Difference from hard plurality (percentage points)")
    ax.set_title("Same-split DEV local versus free-running ordering", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for bars in (local_bars, free_bars):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.12 if value >= 0 else -0.12),
                f"{value:+.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )
    return fig


def main() -> None:
    configure()
    existing = read_json(EXISTING)
    teacher = read_json(TEACHER)
    committee_diagnostics = read_json(COMMITTEE_DIAGNOSTICS)
    if teacher.get("example_count") != 4_050:
        raise RuntimeError("Paper figures require the complete 4,050-example teacher-forced analysis")
    selected_committee = None
    if SELECTED_COMMITTEE.is_file():
        selected_committee = read_json(SELECTED_COMMITTEE)
        if selected_committee.get("status") != "DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE":
            raise RuntimeError("Figure 3 selected committee panel is incomplete")
    figures: dict[str, Any] = {}
    specifications = [
        ("figure01_architecture", architecture_figure, [], "Conceptual architecture; no empirical values."),
        ("figure02_hard_vs_soft", hard_vs_soft_figure, [], "Synthetic mechanism illustration; no empirical performance values."),
        ("figure03_committee_scaling", committee_scaling_figure, [teacher, committee_diagnostics, selected_committee], "E-TF-ANALYSIS / E-COMMITTEE-DIAG / E-COMMITTEE-FREERUN; DEV model selection"),
        ("figure04_calibration", calibration_robustness_figure, [teacher], "E-TF-ANALYSIS; DEV calibration and temperature intervention"),
        ("figure05_rescue_harm", rescue_harm_figure, [existing], "E-RESCUE; paired TEST"),
        ("figure06_coalition_heatmap", coalition_heatmap_figure, [existing], "E-RESCUE; TEST descriptive/noncausal"),
        ("figure07_quality_compute", quality_compute_figure, [existing], "E-CORNERSTONE-RAW; hardware descriptive"),
        ("figure08_pilot_vs_full", pilot_full_figure, [existing], "E-PILOT-RAW / E-CORNERSTONE-RAW"),
        ("figure10_divergence_timing", divergence_figure, [existing], "E-RESCUE; TEST descriptive"),
        ("figure_primary_test_accuracy", primary_accuracy_figure, [existing], "E-CORNERSTONE-RAW / E-RESCUE"),
        ("appendix_shapley", shapley_figure, [teacher], "E-TF-ANALYSIS; DEV exploratory"),
        ("appendix_temperature_interventions", temperature_figure, [teacher], "E-TF-ANALYSIS / E-HARD-TIE-TEMP; DEV intervention"),
        ("appendix_diversity", diversity_figure, [teacher], "E-TF-ANALYSIS; DEV exploratory"),
    ]
    free_run = None
    if FREE_RUN.is_file():
        free_run = read_json(FREE_RUN)
        if free_run.get("status") != "DEV_SELECTED_FREE_RUNNING_ANALYSIS_COMPLETE":
            raise RuntimeError("Figure 9 requires a complete selected free-running DEV analysis")
        specifications.insert(
            8,
            ("figure09_teacher_vs_free", teacher_free_figure, [free_run], "E-FREERUN-DEV / E-TF-ANALYSIS; same-split DEV"),
        )
    for stem, function, args, role in specifications:
        artifact = save_figure(function(*args), stem)
        artifact["evidence_role"] = role
        figures[stem] = artifact
    packet = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "sources": {
            EXISTING.relative_to(LAB_ROOT).as_posix(): sha256_file(EXISTING),
            TEACHER.relative_to(LAB_ROOT).as_posix(): sha256_file(TEACHER),
            HARD_TIE_TEMP.relative_to(LAB_ROOT).as_posix(): sha256_file(HARD_TIE_TEMP),
            COMMITTEE_DIAGNOSTICS.relative_to(LAB_ROOT).as_posix(): sha256_file(
                COMMITTEE_DIAGNOSTICS
            ),
            PILOT.relative_to(LAB_ROOT).as_posix(): sha256_file(PILOT),
        },
        "figures": figures,
    }
    if free_run is not None:
        packet["sources"][FREE_RUN.relative_to(LAB_ROOT).as_posix()] = sha256_file(FREE_RUN)
    if selected_committee is not None:
        packet["sources"][SELECTED_COMMITTEE.relative_to(LAB_ROOT).as_posix()] = sha256_file(
            SELECTED_COMMITTEE
        )
    write_json_atomic(METADATA, packet)
    print(json.dumps({"status": "PAPER_FIGURES_COMPLETE", "count": len(figures), "metadata": str(METADATA)}, indent=2))


if __name__ == "__main__":
    main()
