"""Render frozen Track B calibration mechanism metrics; never loads a model.

Run with the canonical programme Python. Outputs are exclusively created in this
directory. Both figures display all sixteen depth pairs and all six objectives.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
PROGRAMME = HERE.parent.parent
REPOSITORY = PROGRAMME.parent.parent
AGGREGATE = PROGRAMME / "cross_scale/evidence/01_alignment_matrix/alignment-cpu-v2-002/aggregate/attempt_000"
INPUTS = [AGGREGATE / "matrix.json", AGGREGATE / "selection.json"]
SUPA = [8, 12, 16, 19]
DONOR = [12, 16, 20, 24]
OBJECTIVES = ["B0", "B1", "B2", "B3", "BS", "BR"]
LABELS = {
    "B0": "Absolute ridge",
    "B1": "Token-normalized state",
    "B2": "Delta-aware",
    "B3": "Token-normalized + delta-aware",
    "BS": "Shuffled-pair control",
    "BR": "Random bridge control",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY), *arguments], text=True
    ).strip()


def write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def main() -> None:
    inputs = {}
    for path in INPUTS:
        relative = path.relative_to(REPOSITORY).as_posix()
        committed = subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "show", f"HEAD:{relative}"]
        )
        assert committed == path.read_bytes(), f"Uncommitted input bytes: {path}"
        inputs[relative] = sha256(path)
    matrix = json.loads(INPUTS[0].read_text(encoding="utf-8"))
    selection = json.loads(INPUTS[1].read_text(encoding="utf-8"))
    selected, = selection["selected"]
    selected_key = (selected["objective"], selected["supa_layer"], selected["donor_layer"])
    assert selected_key == ("B3", 19, 24)
    assert selection["status"] == "MECHANISM_GO"
    assert selection["large_generation_licensed"] is False
    pairs = {(p["supa_layer"], p["donor_layer"]): p for p in matrix["pairs"]}
    assert len(matrix["pairs"]) == len(pairs) == 16
    assert set(pairs) == {(s, d) for s in SUPA for d in DONOR}
    values = {key: {} for key in ("mean_token_cosine", "mean_token_relative_mse")}
    rows = []
    for objective in OBJECTIVES:
        for metric in values:
            values[metric][objective] = np.empty((4, 4), dtype=float)
        for i, supa in enumerate(SUPA):
            for j, donor in enumerate(DONOR):
                pair = pairs[supa, donor]
                assert set(pair["objectives"]) == set(OBJECTIVES)
                metrics = pair["objectives"][objective]["heldout"]["delta"]["ordinary_tokens"]
                assert metrics["tokens"] == 2395
                for metric in values:
                    value = metrics[metric]
                    assert np.isfinite(value)
                    values[metric][objective][i, j] = value
                rows.append({"objective": objective, "supa_layer": supa, "donor_layer": donor,
                             "mean_token_cosine": metrics["mean_token_cosine"],
                             "mean_token_relative_mse": metrics["mean_token_relative_mse"],
                             "tokens": metrics["tokens"],
                             "selected": (objective, supa, donor) == selected_key})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.titleweight": "semibold",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})
    specs = [
        ("mean_token_cosine", "ordinary_delta_cosine", "Transformation direction",
         "Mean per-token cosine  |  higher is better", Normalize(0, 1), "viridis",
         [0, .2, .4, .6, .8, 1]),
        ("mean_token_relative_mse", "ordinary_delta_relative_mse", "Transformation error",
         "Mean per-token relative MSE  |  lower is better  |  logarithmic color scale",
         LogNorm(.25, 128), "viridis_r", [.25, 1, 4, 16, 64, 128]),
    ]
    artifacts = []
    for metric, name, heading, label, norm, cmap_name, ticks in specs:
        all_values = np.concatenate([v.ravel() for v in values[metric].values()])
        assert all_values.min() >= norm.vmin and all_values.max() <= norm.vmax
        figure = plt.figure(figsize=(11.7, 8.3))
        grid = figure.add_gridspec(2, 3, left=.075, right=.86, bottom=.17, top=.80,
                                  wspace=.32, hspace=.38)
        cmap = matplotlib.colormaps[cmap_name]
        for index, objective in enumerate(OBJECTIVES):
            ax = figure.add_subplot(grid[index // 3, index % 3])
            data = values[metric][objective]
            artist = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
            ax.set_title(f"{objective}  {LABELS[objective]}", pad=8)
            ax.set_xticks(range(4), DONOR)
            ax.set_yticks(range(4), SUPA)
            ax.tick_params(length=0, pad=6)
            if index >= 3:
                ax.set_xlabel("Qwen3-4B donor layer", labelpad=7)
            if index % 3 == 0:
                ax.set_ylabel("Supa insertion after layer", labelpad=8)
            ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.6, alpha=.65)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for i, supa in enumerate(SUPA):
                for j, donor in enumerate(DONOR):
                    value = data[i, j]
                    rgba = cmap(norm(value))
                    luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                    text_color = "#151515" if luminance > .53 else "white"
                    rendered = f"{value:.3f}" if metric == "mean_token_cosine" else f"{value:.2f}"
                    ax.text(j, i, rendered, ha="center", va="center", color=text_color, fontsize=10)
                    if (objective, supa, donor) == selected_key:
                        ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False,
                                               edgecolor="white", linewidth=3.4))
                        ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False,
                                               edgecolor="#b42446", linewidth=1.6))
        cax = figure.add_axes([.90, .24, .021, .47])
        colorbar = figure.colorbar(artist, cax=cax, ticks=ticks)
        colorbar.ax.set_yticklabels([str(tick) for tick in ticks])
        colorbar.ax.tick_params(length=3)
        colorbar.outline.set_visible(False)
        figure.text(.075, .943, "TRACK B  /  CALIBRATION MECHANISM SCREEN", fontsize=10,
                    color="#555e6b", weight="semibold")
        figure.text(.075, .896, heading, fontsize=22, weight="semibold", color="#152030")
        figure.text(.075, .857, label, fontsize=11, color="#354052")
        figure.text(.075, .108,
                    "Held-out calibration: 32 scenarios, 2,395 ordinary tokens per cell. "
                    "All 16 depth pairs × 6 objectives shown.", fontsize=10)
        figure.text(.075, .079,
                    "Outlined cell: frozen selected B3 / Supa L19 / donor L24. "
                    "Layer indices are zero-based.", fontsize=10)
        figure.text(.075, .046,
                    "Calibration fidelity only; these figures do not establish generation or capability gains.",
                    fontsize=10, weight="semibold", color="#a51e3b")
        for extension in ("png", "pdf"):
            buffer = io.BytesIO()
            kwargs = {"dpi": 220} if extension == "png" else {"metadata": {
                "Title": f"Track B calibration mechanism screen: {heading}",
                "Subject": "All 96 frozen alignment cells; not efficacy evidence.",
                "Creator": "Matplotlib; render_alignment_matrix.py",
                "CreationDate": None, "ModDate": None}}
            figure.savefig(buffer, format=extension, **kwargs)
            output = HERE / f"{name}.{extension}"
            write_new(output, buffer.getvalue())
            artifacts.append(output)
        plt.close(figure)

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path = HERE / "ordinary_token_metrics.csv"
    write_new(csv_path, csv_buffer.getvalue().encode("utf-8"))
    artifacts.append(csv_path)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Static visualization of complete frozen Track B held-out calibration mechanism metrics; not efficacy.",
        "source_head": git("rev-parse", "HEAD"),
        "input_sha256": inputs,
        "input_bytes_match_committed_HEAD": True,
        "render_script_sha256": sha256(Path(__file__)),
        "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                    "numpy": np.__version__, "backend": matplotlib.get_backend()},
        "coverage": {"pairs": 16, "objectives": OBJECTIVES, "cells_per_figure": 96,
                     "heldout_scenarios": 32, "ordinary_tokens_per_cell": 2395,
                     "ordinary_token_metrics_csv_rows": len(rows)},
        "selected": selected,
        "scales": {"cosine": {"minimum": 0, "maximum": 1, "scale": "linear"},
                   "relative_mse": {"minimum": .25, "maximum": 128, "scale": "logarithmic"},
                   "all_values_in_scale": True, "common_across_all_six_objectives": True},
        "artifact_sha256": {p.name: sha256(p) for p in artifacts},
    }
    write_new(HERE / "input_sha256_receipt.json", (json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"artifacts": [str(p) for p in artifacts], "cells": len(rows),
                      "input_sha256": inputs}, indent=2))


if __name__ == "__main__":
    main()
