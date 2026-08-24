#!/usr/bin/env python3
"""Generate deterministic figures for one completed synthetic Tier-A run.

The script intentionally rejects model-derived or closed-loop runs. It reads
only committed Tier-A CSV/JSON summaries and does not infer missing results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend("Agg")


FIGURE_NAMES = (
    "exponent_sweep.png",
    "compression_round_interaction.png",
    "baseline_controls.png",
    "count_ablations.png",
    "rank_trajectories.png",
    "byte_comparison.png",
)

COLORS = {
    "blue": "#31688e",
    "green": "#35b779",
    "gold": "#fde725",
    "purple": "#440154",
    "red": "#c44e52",
    "gray": "#7f7f7f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "Rineya deterministic Tier-A figure generator"},
    )
    plt.close(fig)


def exponent_sweep(rows: list[dict[str, str]], output: Path) -> None:
    x = [float(row["exponent"]) for row in rows]
    recall = [float(row["evidence_recall"]) for row in rows]
    all_survive = [float(row["all_required_evidence_survival"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, recall, marker="o", color=COLORS["blue"], label="Evidence recall")
    ax.plot(x, all_survive, marker="s", linestyle="--", color=COLORS["green"], label="All evidence survives")
    ax.set(xlabel="Normalization exponent e", ylabel="Synthetic survival metric", ylim=(-0.05, 1.05))
    ax.set_title("Toy exposure trace: exponent sweep")
    ax.legend(loc="lower right")
    ax.text(
        0.01,
        0.96,
        "OPEN_LOOP / SYNTHETIC",
        transform=ax.transAxes,
        color=COLORS["gray"],
        fontsize=8,
        va="top",
    )
    save(fig, output)


def compression_rounds(rows: list[dict[str, str]], output: Path) -> None:
    policies = sorted({row["policy"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    styles = {
        "B2_raw_accumulation": (COLORS["red"], "o", "B2 raw accumulation"),
        "B3_observation_normalized": (COLORS["green"], "s", "B3 S/C normalization"),
    }
    for policy in policies:
        selected = sorted((row for row in rows if row["policy"] == policy), key=lambda row: int(row["rounds"]))
        color, marker, label = styles[policy]
        ax.plot(
            [int(row["rounds"]) for row in selected],
            [float(row["evidence_recall"]) for row in selected],
            marker=marker,
            linewidth=2,
            color=color,
            label=label,
        )
    ax.set(xticks=[3, 6, 9], xlabel="Repeated scoring rounds", ylabel="Evidence recall", ylim=(-0.05, 1.05))
    ax.set_title("Specified toy process: compression-round contrast (n=3 traces)")
    ax.legend(loc="center right")
    save(fig, output)


def baseline_controls(rows: list[dict[str, str]], output: Path) -> None:
    labels = [row["baseline"] for row in rows]
    values = [float(row["evidence_recall"]) for row in rows]
    colors = [COLORS["green"] if label == "B3" else COLORS["blue"] for label in labels]
    colors[labels.index("B8")] = COLORS["gold"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    bars[labels.index("B8")].set_hatch("///")
    ax.set(ylabel="Evidence recall", ylim=(0, 1.12))
    ax.set_title("B0-B8 controls on the r6 synthetic trace\nB8 is oracle; B0 selects 12 entries versus 2")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    save(fig, output)


def count_ablations(rows: list[dict[str, str]], output: Path) -> None:
    labels = [row["count_transform"].replace("head_averaged", "head avg") for row in rows]
    recall = [float(row["evidence_recall"]) for row in rows]
    all_survive = [float(row["all_required_evidence_survival"]) for row in rows]
    positions = list(range(len(rows)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar([x - width / 2 for x in positions], recall, width, label="Evidence recall", color=COLORS["blue"])
    ax.bar([x + width / 2 for x in positions], all_survive, width, label="All evidence survives", color=COLORS["green"])
    ax.set(xticks=positions, xticklabels=labels, ylabel="Synthetic survival metric", ylim=(0, 1.12))
    ax.tick_params(axis="x", rotation=28)
    ax.set_title("Count perturbations on one full-candidate toy trace")
    ax.legend(loc="upper center", ncols=2)
    save(fig, output)


def rank_trajectories(rows: list[dict[str, str]], output: Path) -> None:
    early_id = "10000"
    late_id = next(row["entry_id"] for row in rows if row["is_late_high_utility"].casefold() == "true")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    for ax, field, title in (
        (axes[0], "raw_rank", "Raw cumulative S"),
        (axes[1], "normalized_rank", "Exposure-normalized S/C"),
    ):
        for entry_id, label, color, marker in (
            (early_id, "Early unit-utility entry", COLORS["red"], "o"),
            (late_id, "Late high-utility entry", COLORS["green"], "s"),
        ):
            selected = sorted((row for row in rows if row["entry_id"] == entry_id), key=lambda row: int(row["round"]))
            ax.plot(
                [int(row["round"]) for row in selected],
                [float(row[field]) for row in selected],
                marker=marker,
                linewidth=2,
                color=color,
                label=label,
            )
        ax.set(xlabel="Scoring round", title=title, xticks=[1, 2, 3, 4, 5, 6])
    axes[0].set_ylabel("Ranking value recorded by toy trace")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Exposure changes raw score while normalization removes pure longevity")
    save(fig, output)


def byte_comparison(payload: dict[str, Any], output: Path) -> None:
    reports = [
        ("Slot matched\nno S/C", payload["slot_matched"]["without_significance"]),
        ("Slot matched\nwith S/C", payload["slot_matched"]["with_significance"]),
        ("Byte matched\nno S/C", payload["total_byte_matched"]["without_significance"]),
        ("Byte matched\nwith S/C", payload["total_byte_matched"]["with_significance"]),
    ]
    labels = [label for label, _ in reports]
    kv = [row["kv_bytes"] / 2**20 for _, row in reports]
    auxiliary = [row["auxiliary_bytes"] / 2**20 for _, row in reports]
    positions = list(range(len(reports)))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(positions, kv, label="K/V", color=COLORS["blue"])
    ax.bar(positions, auxiliary, bottom=kv, label="Persistent auxiliary state", color=COLORS["gold"])
    ax.set(xticks=positions, xticklabels=labels, ylabel="Persistent state (MiB)")
    ax.set_title("R2 geometry byte ledger (formula, no model allocation measured)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    for index, (_, row) in enumerate(reports):
        total = row["total_persistent_bytes"] / 2**20
        ax.text(index, total + 0.25, f"{total:.2f}", ha="center", fontsize=8)
    save(fig, output)


def require_supported_run(manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    if manifest.get("status") != "COMPLETE" or summary.get("status") != "PASS":
        raise ValueError("figure generation requires a COMPLETE/PASS run")
    if manifest.get("experiment", {}).get("execution_mode") != "OPEN_LOOP_TRACE_SIMULATION":
        raise ValueError("this script only supports OPEN_LOOP_TRACE_SIMULATION")
    if summary.get("tier") != "A" or summary.get("model_inference_performed") is not False:
        raise ValueError("this script only supports synthetic Tier-A runs without model inference")


def generate(run_root: Path) -> list[Path]:
    run_root = run_root.expanduser().resolve()
    manifest = load_json(run_root / "manifest.json")
    summary = load_json(run_root / "summary.json")
    require_supported_run(manifest, summary)
    data = run_root / "figures" / "data"
    output = run_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    inputs = {
        "exponent_sweep.csv": data / "exponent_sweep.csv",
        "compression_round_interaction.csv": data / "compression_round_interaction.csv",
        "baseline_policy_comparison.csv": data / "baseline_policy_comparison.csv",
        "count_ablation.csv": data / "count_ablation.csv",
        "exposure_rank_trajectories.csv": data / "exposure_rank_trajectories.csv",
        "budget_ledger.json": run_root / "statistics" / "budget_ledger.json",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing figure inputs: {missing}")

    exponent_sweep(load_csv(inputs["exponent_sweep.csv"]), output / FIGURE_NAMES[0])
    compression_rounds(load_csv(inputs["compression_round_interaction.csv"]), output / FIGURE_NAMES[1])
    baseline_controls(load_csv(inputs["baseline_policy_comparison.csv"]), output / FIGURE_NAMES[2])
    count_ablations(load_csv(inputs["count_ablation.csv"]), output / FIGURE_NAMES[3])
    rank_trajectories(load_csv(inputs["exposure_rank_trajectories.csv"]), output / FIGURE_NAMES[4])
    byte_comparison(load_json(inputs["budget_ledger.json"]), output / FIGURE_NAMES[5])

    generated = [output / name for name in FIGURE_NAMES]
    figure_manifest = {
        "schema_version": "1.0",
        "run_id": manifest["run_id"],
        "generator": {
            "script": "scripts/make_figures.py",
            "script_sha256": sha256(Path(__file__)),
            "matplotlib": matplotlib.__version__,
        },
        "execution_mode": "OPEN_LOOP_TRACE_SIMULATION",
        "scope": "SYNTHETIC_TIER_A_ONLY",
        "model_inference_performed": False,
        "inputs": {name: sha256(path) for name, path in sorted(inputs.items())},
        "figures": {path.name: sha256(path) for path in generated},
        "not_run": [
            "model accuracy or evidence survival",
            "attention-prior strength versus quality",
            "memory-slot count versus quality",
            "controller risk/coverage",
            "comparative/deployment CPU or GPU latency and GPU VRAM",
        ],
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return generated


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed Tier-A run directory")
    args = parser.parse_args(list(argv) if argv is not None else None)
    generated = generate(args.run)
    print(json.dumps({"status": "PASS", "figures": [str(path) for path in generated]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
