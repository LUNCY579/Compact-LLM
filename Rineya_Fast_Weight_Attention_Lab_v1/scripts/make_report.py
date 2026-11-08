"""Turn the P2b run into the tables and figures the brief asks for (section 13).

Reads artifacts/.../p2b_longctx/summary.json plus the per-arm eval.json files and
the resource-scaling diagnostics, and writes:

  reports/RESULTS_P2B.md          aggregate, by-position, by-length, interference,
                                  and the preregistered paired contrasts
  reports/figures/*.png           accuracy vs position, accuracy vs context length,
                                  peak VRAM vs length, prefill throughput vs length

Position effects are never averaged away: every table is reported per position
and per length as well as in aggregate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.eval_longctx import wilson  # noqa: E402
from rineya_fwa.paths import ARTIFACTS, REPORTS  # noqa: E402

POSITIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
ARM_ORDER = ["arm_A", "arm_B", "arm_C", "arm_D", "arm_Cshort"]


def arm_label(tag: str) -> str:
    return tag.replace("arm_", "").replace("/seed_", " s")


def load(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {}
    inter: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(run_dir.rglob("eval.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rows[rec["tag"]] = rec.get("position_rows", [])
        inter[rec["tag"]] = rec.get("interference_rows", [])
    return {"summary": summary, "rows": rows, "interference": inter}


def sort_tags(tags: list[str]) -> list[str]:
    def key(t: str) -> tuple[int, str]:
        base = t.split("/")[0]
        return (ARM_ORDER.index(base) if base in ARM_ORDER else 99, t)

    return sorted(tags, key=key)


def fmt_ci(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k/n:.3f} [{lo:.2f},{hi:.2f}]"


def table_aggregate(data: dict[str, Any]) -> str:
    out = ["| Arm | n | exact [95% CI] | middle | edge | middle−edge | interference |",
           "|---|---:|---|---:|---:|---:|---|"]
    for tag in sort_tags(list(data["rows"])):
        r = data["rows"][tag]
        if not r:
            continue
        k, n = sum(x["exact"] for x in r), len(r)
        mid = [x["exact"] for x in r if x["position"] in (0.25, 0.50, 0.75)]
        edge = [x["exact"] for x in r if x["position"] in (0.05, 0.10, 0.90, 0.95)]
        m = sum(mid) / len(mid) if mid else float("nan")
        e = sum(edge) / len(edge) if edge else float("nan")
        ir = data["interference"].get(tag, [])
        ik, inn = sum(x["exact"] for x in ir), len(ir)
        out.append(
            f"| {arm_label(tag)} | {n} | {fmt_ci(k, n)} | {m:.3f} | {e:.3f} | {m-e:+.3f} | {fmt_ci(ik, inn)} |"
        )
    return "\n".join(out)


def table_by(data: dict[str, Any], field: str, values: list[Any], header: str) -> str:
    cols = " | ".join(str(v) for v in values)
    out = [f"| Arm | {cols} |", "|---" * (len(values) + 1) + "|"]
    for tag in sort_tags(list(data["rows"])):
        r = data["rows"][tag]
        if not r:
            continue
        cells = []
        for v in values:
            cell = [x["exact"] for x in r if x[field] == v]
            cells.append(f"{sum(cell)/len(cell):.3f}" if cell else "—")
        out.append(f"| {arm_label(tag)} | " + " | ".join(cells) + " |")
    return f"**{header}**\n\n" + "\n".join(out)


def table_interference(data: dict[str, Any]) -> str:
    cats = ["A", "B", "C", "D", "E", "F", "G"]
    out = ["| Arm | " + " | ".join(cats) + " |", "|---" * 8 + "|"]
    for tag in sort_tags(list(data["interference"])):
        r = data["interference"][tag]
        if not r:
            continue
        cells = []
        for c in cats:
            cell = [x["exact"] for x in r if x["category"] == c]
            cells.append(f"{sum(cell)/len(cell):.2f}" if cell else "—")
        out.append(f"| {arm_label(tag)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_contrasts(summary: dict[str, Any]) -> str:
    out = ["| Contrast | n paired | b better | a better | discordant | p (two-sided exact) |",
           "|---|---:|---:|---:|---:|---:|"]
    for name, c in (summary.get("contrasts") or {}).items():
        out.append(
            f"| {name} | {c['n_paired']} | {c['b_better']} | {c['a_better']} | "
            f"{c['discordant']} | {c['p_two_sided']:.4f} |"
        )
    return "\n".join(out)


def table_training(summary: dict[str, Any]) -> str:
    out = ["| Run | steps | first loss | last loss | final alpha | GPU-h | peak VRAM |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary.get("training") or []:
        peak = r.get("peak_vram_bytes") or 0
        out.append(
            f"| {r.get('tag')} | {r.get('optimizer_steps')} | {r.get('first_loss'):.4f} | "
            f"{r.get('last_loss'):.4f} | {r.get('final_alpha'):.5f} | "
            f"{r.get('gpu_hours', 0):.3f} | {peak/2**30:.2f} GiB |"
        )
    return "\n".join(out)


def figures(data: dict[str, Any], fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # accuracy vs position
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag in sort_tags(list(data["rows"])):
        r = data["rows"][tag]
        if not r:
            continue
        ys = []
        for p in POSITIONS:
            cell = [x["exact"] for x in r if x["position"] == p]
            ys.append(sum(cell) / len(cell) if cell else float("nan"))
        ax.plot(POSITIONS, ys, marker="o", label=arm_label(tag))
    ax.set_xlabel("relative position of the decisive evidence")
    ax.set_ylabel("exact match")
    ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy vs needle position (all context lengths pooled)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "accuracy_vs_position.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # accuracy vs context length
    lengths = sorted({x["target_length"] for r in data["rows"].values() for x in r})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for tag in sort_tags(list(data["rows"])):
        r = data["rows"][tag]
        if not r:
            continue
        ys = []
        for L in lengths:
            cell = [x["exact"] for x in r if x["target_length"] == L]
            ys.append(sum(cell) / len(cell) if cell else float("nan"))
        ax.plot(lengths, ys, marker="s", label=arm_label(tag))
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(x) for x in lengths])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("exact match")
    ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy vs context length")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "accuracy_vs_length.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # resources, from the scaling diagnostics
    scal: dict[str, list[dict[str, Any]]] = {}
    for fname in ("scaling.json", "scaling_long.json"):
        fp = ARTIFACTS / "diagnostics" / fname
        if not fp.is_file():
            continue
        blob = json.loads(fp.read_text(encoding="utf-8"))
        for arm, rowlist in (blob.get("arms") or {}).items():
            scal.setdefault(arm, []).extend(rowlist)
    if scal:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for arm, rowlist in sorted(scal.items()):
            rows = sorted((r for r in rowlist if not r.get("oom")), key=lambda r: r["seq_len"])
            if not rows:
                continue
            xs = [r["seq_len"] for r in rows]
            axes[0].plot(xs, [r["peak_vram_bytes"] / 2**30 for r in rows], marker="o", label=f"arm {arm}")
            axes[1].plot(xs, [r["prefill_tokens_per_sec"] for r in rows], marker="o", label=f"arm {arm}")
        # KV cache that would be needed, vs the bounded recurrent state
        anyrows = sorted(
            (r for rl in scal.values() for r in rl if not r.get("oom")), key=lambda r: r["seq_len"]
        )
        if anyrows:
            xs = sorted({r["seq_len"] for r in anyrows})
            kv = {r["seq_len"]: r["kv_cache_bytes_if_used"] / 2**30 for r in anyrows}
            axes[0].plot(xs, [kv[x] for x in xs], linestyle="--", color="grey", label="KV cache if used")
        for ax, ylab, title in (
            (axes[0], "GiB", "Peak VRAM vs context length"),
            (axes[1], "tokens/sec", "Prefill throughput vs context length"),
        ):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("context length (tokens)")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = fig_dir / "resources_vs_length.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(p.name)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=str(ARTIFACTS / "p2b_longctx"))
    ap.add_argument("--out", type=str, default=str(REPORTS / "RESULTS_P2B.md"))
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not (run_dir / "summary.json").is_file():
        print(f"no summary.json under {run_dir}; has the run finished?")
        return 1
    data = load(run_dir)
    summary = data["summary"]
    lengths = sorted({x["target_length"] for r in data["rows"].values() for x in r})

    figs = figures(data, REPORTS / "figures")

    cfg = summary.get("config", {})
    body = f"""# Results — P2b (long-context, non-ceilinged suite)

Classification: `EXPLORATORY_NOT_CONFIRMATORY`. Preregistered in
`PREREGISTRATION.md` Amendment 1. Generated by `scripts/make_report.py`.

Suite: two-hop `vault -> unit -> code`, {cfg.get('n_decoys')} confusable decoys,
token-exact lengths {lengths}, {cfg.get('n_per_cell')} items per
(length, position) cell, 32 candidates from a 552-code single-token answer
space, scored by candidate argmax in one forward pass.
Seeds {cfg.get('seeds')}. Layer {cfg.get('layer')}, rank {cfg.get('rank')}.

## Aggregate

{table_aggregate(data)}

## Accuracy by needle position

{table_by(data, 'position', list(POSITIONS), 'exact match at each relative position')}

![accuracy vs position](figures/accuracy_vs_position.png)

## Accuracy by context length

{table_by(data, 'target_length', lengths, 'exact match at each context length')}

![accuracy vs length](figures/accuracy_vs_length.png)

## Interference battery (A–G)

{table_interference(data)}

## Preregistered contrasts (paired two-sided exact McNemar)

In each row, "b better" counts items the second arm got right and the first got
wrong. `C_vs_B` has a=B, b=C, so "b better" is arm C winning.

{table_contrasts(summary)}

## Training

{table_training(summary)}

Total training GPU-hours: {summary.get('training_gpu_hours', 0):.3f}

## Resources

![resources](figures/resources_vs_length.png)

Figures written: {', '.join(figs)}
"""
    Path(args.out).write_text(body, encoding="utf-8")
    print(body)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
