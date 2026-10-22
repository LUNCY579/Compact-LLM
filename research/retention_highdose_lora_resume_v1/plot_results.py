"""Create a six-arm scientific plot only from COMPLETE_COMPARABLE RESULTS.json.

Uses matplotlib already installed in Chat-Bubble/.venv. No model runtime, live
probe, raw partial output, missing-arm imputation, or model selection is involved.
PNG, SVG and a hash receipt are create-only and remain beside RESULTS.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("a025", "a050", "a075", "a100", "norm", "lora")
OLD_ARMS = ARMS[:4]
COLORS = {"a025": "#5799c6", "a050": "#2979a8", "a075": "#155272", "a100": "#303943",
          "norm": "#b86a28", "lora": "#8b3f82"}
MARKERS = {**{name: "o" for name in OLD_ARMS}, "norm": "D", "lora": "*"}
LABELS = {"a025": "α=0.25", "a050": "α=0.50", "a075": "α=0.75", "a100": "α=1.00",
          "norm": "Norm scalar", "lora": "LoRA"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def count(value, maximum: int, name: str) -> int:
    require(type(value) is int and 0 <= value <= maximum, f"invalid complete-report count: {name}")
    return value


def validate_report(report: dict) -> dict:
    """Validate the completed schema before loading any plotting dependency."""
    require(isinstance(report, dict) and report.get("status") == "COMPLETE_COMPARABLE" and
            report.get("experiment") == "DOSELORA-001", "plot requires a completed comparable DOSELORA result")
    endpoints = report.get("endpoints")
    require(isinstance(endpoints, dict) and set(endpoints) == set(ARMS), "plot requires exactly all six registered arms")
    norm = report.get("norm_match", {})
    beta, discrepancy = norm.get("beta"), norm.get("relative_norm_discrepancy")
    require(type(beta) in (float, int) and math.isfinite(beta) and 0 <= beta <= 2,
            "invalid registered norm-matching coefficient")
    require(type(discrepancy) in (float, int) and math.isfinite(discrepancy) and 0 <= discrepancy <= .005,
            "matched scalar did not meet the registered norm tolerance")
    points = {}
    bank_hashes = {}
    for arm in ARMS:
        endpoint = endpoints[arm]
        retention = endpoint["retention"]
        require(retention.get("registered_policy") == "permissive", "plot policy must remain registered permissive")
        policy = retention["policies"]["permissive"]
        require(policy["registered_reference_n"] == 47 and policy["panel_n"] == 150,
                "plot requires fixed 47 source references and the complete 150-row arithmetic panel")
        point = {"retained": count(policy["references_retained"], 47, arm + "/retained"),
                 "arithmetic": count(policy["panel_correct"], 150, arm + "/arithmetic")}
        for bank in ("dose", "select"):
            data = endpoint[bank]
            require(data["n"] == 384, "plot requires complete 384-row acquisition banks")
            bank_sha = data.get("views_sha256")
            require(isinstance(bank_sha, str) and len(bank_sha) == 64 and
                    all(c in "0123456789abcdef" for c in bank_sha), "missing acquisition bank hash")
            require(bank_hashes.setdefault(bank, bank_sha) == bank_sha, "arms use different acquisition banks")
            point[bank] = count(data["correct"], 384, arm + "/" + bank)
        points[arm] = point
    require(bank_hashes["dose"] != bank_hashes["select"], "the two banks were duplicated or mislabeled")
    return {"points": points, "beta": beta, "relative_norm_discrepancy": discrepancy, "bank_sha256": bank_hashes}


def _annotate(ax, grouped, renderer):
    """Place six short labels without data jitter; use leaders when displaced."""
    occupied = []
    plot_box = ax.get_window_extent(renderer).padded(-3)
    offsets = ((0, 13), (20, 13), (-20, 13), (0, -22), (25, -22), (-25, -22),
               (34, 0), (-34, 0), (0, 28), (0, -36), (42, 24), (-42, 24))
    for (x, y), labels in grouped.items():
        label = " / ".join(labels)
        candidates = []
        for dx, dy in offsets:
            annotation = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                                     ha="center", va="bottom", fontsize=8.6, color="#26313c",
                                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": .8},
                                     zorder=5, annotation_clip=False)
            box = annotation.get_window_extent(renderer).padded(3)
            overlap = sum(box.overlaps(other) for other in occupied)
            outside = not (plot_box.contains(box.x0, box.y0) and plot_box.contains(box.x1, box.y1))
            candidates.append((overlap * 1000 + int(outside) * 100 + abs(dx) + abs(dy), dx, dy, box))
            annotation.remove()
        _, dx, dy, box = min(candidates, key=lambda item: item[0])
        occupied.append(box)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.6, color="#26313c",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": .8},
                    arrowprops={"arrowstyle": "-", "color": "#9aa4af", "linewidth": .6},
                    zorder=5, annotation_clip=False)


def build_figure(report: dict):
    data = validate_report(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    points = data["points"]
    values = [points[arm][bank] for bank in ("dose", "select") for arm in ARMS]
    xlo = max(0, min([284, *values]) - 18)
    xhi = min(384, max([284, *values]) + 18)
    if xhi - xlo < 80:
        xlo, xhi = max(0, xlo - 40), min(384, xhi + 40)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.068, right=.982, bottom=.34, top=.80, wspace=.15)
    for ax, bank, title in zip(axes, ("dose", "select"), ("Dose bank · 384 rows", "SELECT bank · 384 rows")):
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xlim(xlo, xhi); ax.set_ylim(-1, 49)
        ax.set_yticks([0, 10, 20, 30, 40, 47])
        ax.grid(axis="y", color="#e7ebef", linewidth=.7); ax.set_axisbelow(True)
        ax.add_patch(Rectangle((284, 40), 100, 7, facecolor="#e8f0e9", edgecolor="none", zorder=0))
        ax.axvline(284, color="#799783", linestyle="--", linewidth=1)
        ax.axhline(40, color="#799783", linestyle="--", linewidth=1)
        old = [points[name] for name in OLD_ARMS]
        ax.plot([p[bank] for p in old], [p["retained"] for p in old], color="#aab9c5", linewidth=1.0, zorder=1)
        grouped = {}
        for arm in ARMS:
            p = points[arm]
            ax.scatter(p[bank], p["retained"], marker=MARKERS[arm], color=COLORS[arm],
                       s=165 if arm == "lora" else 80 if arm == "norm" else 55,
                       edgecolors="white", linewidth=.8, zorder=4 if arm == "lora" else 3)
            grouped.setdefault((p[bank], p["retained"]), []).append(LABELS[arm])
        ax.set_xlabel("Acquisition answers correct / 384", labelpad=8)
        ax._dose_grouped_labels = grouped
    axes[0].set_ylabel("Frozen source answers retained / 47", labelpad=8)
    fig.suptitle("High-dose training: six measured artifacts", x=.068, y=.968, ha="left", fontsize=16, fontweight="bold")
    fig.text(.068, .916, f"FP32 merged LoRA and its scalar norm control (β={data['beta']:.6g}) alongside four prior native α arms",
             fontsize=10.5, color="#44515f")
    legend = [Line2D([], [], color=COLORS[name], marker=MARKERS[name], linestyle="none",
                     markersize=10 if name == "lora" else 7,
                     label=LABELS[name] + (f" (β={data['beta']:.4g})" if name == "norm" else "")) for name in ARMS]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.525, .211), ncol=6, frameon=False,
               columnspacing=1.45, handletextpad=.35, fontsize=9)
    fig.text(.068, .175, "Shading marks acquisition ≥284 and retained ≥40 on each bank. Arithmetic ≥62/150 is also required.",
             fontsize=9, color="#44515f")
    panel_counts = "  ·  ".join(f"{LABELS[name]}: {points[name]['arithmetic']}" for name in ARMS)
    fig.text(.068, .139, "Arithmetic panel /150 — " + panel_counts, fontsize=9, color="#44515f")
    fig.text(.068, .097, "Banks are separate. The line connects only the four prior α observations from one native training draw.",
             fontsize=9, color="#44515f")
    fig.text(.068, .058, "Positions are descriptive; they establish neither a winner nor equivalence. Paired statistics and the ≥20 cross-draw floor are in RESULTS.md.",
             fontsize=8.7, color="#44515f")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        _annotate(ax, ax._dose_grouped_labels, renderer)
        del ax._dose_grouped_labels
    return fig, data, matplotlib.__version__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", default="FRONTIER_SIX_ARMS")
    args = parser.parse_args()
    require(args.name and Path(args.name).name == args.name and args.name not in (".", ".."),
            "figure name must be one filename component")
    source = args.root / "RESULTS.json"
    content = source.read_bytes()
    report = json.loads(content.decode("utf-8-sig"))
    validate_report(report)
    paths = [args.root / (args.name + suffix) for suffix in (".png", ".svg", "_FIGURE.json")]
    require(not any(p.exists() or p.is_symlink() for p in paths), "figure outputs are create-only")
    fig, data, version = build_figure(report)
    source_sha = hashlib.sha256(content).hexdigest()
    try:
        require(hashlib.sha256(source.read_bytes()).hexdigest() == source_sha, "completed result changed while plotting")
        for path, format_ in zip(paths[:2], ("png", "svg")):
            with path.open("xb") as stream:
                fig.savefig(stream, format=format_, dpi=180, facecolor="white")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == source_sha, "completed result changed before figure receipt")
        metadata = {"schema": 1, "status": "COMPLETE", "source_results_sha256": source_sha,
                    "source_results_path": str(source.resolve()), "matplotlib": version,
                    "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:2]},
                    "plotted_values": data, "engineering_thresholds": {"each_acquisition_bank": 284, "retained": 40, "arithmetic": 62},
                    "quality_claim": False, "selection_permitted": False, "release_permitted": False}
        with paths[2].open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write("\n")
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(json.dumps({"status": "COMPLETE", "outputs": [str(p) for p in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
