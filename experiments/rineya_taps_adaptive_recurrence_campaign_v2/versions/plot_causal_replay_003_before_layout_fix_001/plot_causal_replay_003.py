"""Render descriptive primary counts from the reviewed native003 report; no model."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / "runs/causal-plot-cache-001")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "analyses/cpu-causal-replay-003-fp32-reviewed-001/REPORT.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETE_CPU_CAUSAL_PREFIX_SURROGATE_ONLY_EOS_ORDERING_UNVERIFIED"
    assert report["errors"] == []
    rows = report["primary"]["case_rows"]
    assert len(rows) == 192 and len({r["parent_id"] for r in rows}) == 96
    assert all(r["balance"] > 0 and not r["numerically_uncertain"] for r in rows)
    groups = []
    for name in ("answerable", "unanswerable"):
        subset = [r for r in rows if r["answerability"] == name]
        groups.append([
            sum(not r["neutral"] and r["useful_direction"] == 1 for r in subset),
            sum(not r["neutral"] and r["useful_direction"] == -1 for r in subset),
            sum(r["neutral"] for r in subset),
        ])
    assert groups == [[91, 52, 1], [0, 45, 3]]
    out = ROOT / "analyses/cpu-causal-replay-003-figures-001"
    out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=.19, right=.97, top=.70, bottom=.30)
    colors = ["#276DA5", "#C56B27", "#A7ABB1"]
    labels = ["Increase", "Decrease", "Neutral"]
    for y, counts in zip((1, 0), groups):
        left = 0
        for n, color in zip(counts, colors):
            ax.barh(y, n, left=left, height=.48, color=color)
            if n >= 7:
                ax.text(left+n/2, y, str(n), color="white", ha="center", va="center", fontweight="bold")
            elif n:
                ax.annotate(str(n), (left+n/2, y), xytext=(left+n+7, y+.23),
                            ha="left", va="center", arrowprops={"arrowstyle": "-", "color": "#777777"})
            left += n
    ax.set_yticks([1, 0], ["Answerable\n144 cases", "Unanswerable\n48 cases"])
    ax.set_xlim(0, 160)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel("Cases")
    ax.set_ylim(-.5, 1.5)
    ax.grid(axis="x", color="#E5E7EB", linewidth=.7)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, labels, title="Direction indicated by the local answer-loss derivative",
               loc="lower center", bbox_to_anchor=(.57, .12), ncol=3, frameon=False)
    fig.text(.19, .92, "Local scale preferences differ by answerability", fontsize=17, fontweight="bold")
    fig.text(.19, .83, "Coherence predicts increase in all 192 cases.", fontsize=12)
    fig.text(.19, .075, "192 exposed DEV cases; 96 paired parents; six shared generators. Four neutral cases retained.", fontsize=9)
    fig.text(.19, .025, "Canonical teacher-prefix likelihood at one extra block; not generated correctness. EOS is separate.", fontsize=9)
    fig.savefig(out / "answerability_direction.png", dpi=180)
    fig.savefig(out / "answerability_direction.svg")
    plt.close(fig)
    metadata = {"source_report_sha256": sha(source), "generator_sha256": sha(Path(__file__)),
                "counts_increase_decrease_neutral": dict(zip(("answerable", "unanswerable"), groups)),
                "model_calls": 0, "interpretation": "descriptive primary derivative counts only"}
    (out / "MANIFEST.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "model_calls": 0}))


if __name__ == "__main__":
    main()
