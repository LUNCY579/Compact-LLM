"""SMALLVD-001 -- layer-ranking report and the registered decisions, computed from the stage files.

Applies the rules exactly as registered in SMALLVD001_PREREG.md, and nothing else:
  §4  A2/A3 layer set at 18 blocks = the candidate set with the LOWEST HOLDOUT objective
      (category-mean answer KL). The SELECT objective chooses nothing.
  §4  protected-skill flag: a category whose HOLDOUT generation flip rate is >= 0.90 at the selected
      set is CATASTROPHIC for that skill, whatever the average says.
  §5  V2/18 stays the recommended first student unless another frontier configuration has HOLDOUT
      answer bpb at least 1 % lower AND no additional CATASTROPHIC category.

Also reports the diagnostics (update norm, cosine, CKA, block influence) next to the causal
single-ablation damage, with rank correlations, so the reader can see how far "looks redundant"
is from "is redundant" on this model. Written create-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CATASTROPHIC_FLIP = 0.90
FRONTIER_MARGIN = 0.01
FRONTIER = {("V1_98304", 16), ("V2_65536", 18), ("V3_49152", 19), ("V4_32768", 20), ("V0", 12)}


def load(d: Path, name: str):
    return json.loads((d / f"{name}.json").read_text(encoding="utf-8"))


def spearman(xs, ys) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def teacher_accuracy(panels, teacher_rows):
    import layer_ablation as la
    by_cat = {}
    for r in panels:
        if r["kind"] != "chat" or r["split"] != "holdout" or r["id"] not in teacher_rows:
            continue                      # only rows the study actually generated
        ok = la.score_row(teacher_rows[r["id"]]["text"], r)
        if ok is None:
            continue
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += ok
        c[1] += 1
    return {c: {"correct": a, "scored": n, "accuracy": a / n} for c, (a, n) in sorted(by_cat.items())}


def catastrophic(gen: dict) -> list[str]:
    return sorted(c for c, v in gen["by_category"].items() if v["flip_rate_vs_teacher"] >= CATASTROPHIC_FLIP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path)
    ap.add_argument("--select-depth", type=int, default=18)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    a = ap.parse_args(argv)
    panels = [json.loads(l) for l in a.panels.read_text(encoding="utf-8").splitlines() if l.strip()]
    S1, S2, S3 = load(a.study, "S1_teacher_outputs"), load(a.study, "S2_diagnostics"), load(a.study, "S3_single_ablation")
    S4, S5 = load(a.study, "S4_greedy_backward"), load(a.study, "S5_depth_strategies")
    S6 = load(a.study, "S6_vocab_frontier") if (a.study / "S6_vocab_frontier.json").exists() else None
    tacc = teacher_accuracy(panels, S1["rows"])
    L = len(S3["layers"])
    diag = {b["layer"]: b for b in S2["per_block"]}

    # ---- single-layer causal table + diagnostics
    single = []
    for l in range(L):
        s = S3["layers"][str(l)]
        h = s["holdout"]
        cats = h["by_category"]
        single.append({
            "layer": l, "select_objective": s["select_objective"],
            "holdout_objective": sum(c["kl_mean_pos"] for c in cats.values()) / len(cats),
            "holdout_kl_p95_row": h["all"]["kl_p95_row"], "holdout_kl_max_row": h["all"]["kl_max_row"],
            "holdout_top1": h["all"]["top1"], "holdout_delta_nll": h["all"].get("delta_nll"),
            "flip_rate": s["holdout_generation"]["all"]["flip_rate_vs_teacher"],
            "worst_category": max(cats, key=lambda c: cats[c]["kl_mean_pos"]),
            "worst_category_kl": max(c["kl_mean_pos"] for c in cats.values()),
            "rel_update_norm": diag[l]["rel_update_norm"], "cos_in_out": diag[l]["cos_in_out"],
            "cka_in_out": diag[l]["cka_in_out"], "block_influence": diag[l]["block_influence"],
            "catastrophic_categories": catastrophic(s["holdout_generation"]),
        })
    causal = [r["holdout_objective"] for r in single]
    corr = {k: spearman(causal, [r[k] for r in single])
            for k in ("select_objective", "rel_update_norm", "cos_in_out", "cka_in_out", "block_influence", "flip_rate")}

    # ---- strategies per depth + registered selection
    depths = {}
    for n, strat in S5["depths"].items():
        rows = []
        for name, v in strat.items():
            g = v["holdout_generation"]
            rows.append({"strategy": name, "kept": v["kept"], "removed": [l for l in range(L) if l not in v["kept"]],
                         "holdout_objective": v["holdout_objective"],
                         "flip_rate": g["all"]["flip_rate_vs_teacher"], "accuracy": g["all"]["accuracy"],
                         "by_category_flip": {c: x["flip_rate_vs_teacher"] for c, x in g["by_category"].items()},
                         "by_category_accuracy": {c: x["accuracy"] for c, x in g["by_category"].items()},
                         "by_category_kl": {c: x["kl_mean_pos"] for c, x in v["holdout"]["by_category"].items()},
                         "catastrophic_categories": catastrophic(g)})
        rows.sort(key=lambda r: r["holdout_objective"])
        depths[int(n)] = rows
    sel = depths[a.select_depth][0]
    greedy_sets = {t["n_layers"]: t["kept"] for t in S4["trajectory"]}
    single_sets = {n: next(r["kept"] for r in depths[n] if r["strategy"] == "single_ablation") for n in depths}
    jaccard = {n: len(set(greedy_sets[n]) & set(single_sets[n])) / len(set(greedy_sets[n]) | set(single_sets[n]))
               for n in depths if n in greedy_sets}

    # ---- frontier rule
    frontier = None
    if S6:
        cfgs = S6["configs"]
        byname = {(v["vocab"], v["n_layers"]): v for v in cfgs.values()}
        pts = []
        for (vn, n), v in sorted(byname.items(), key=lambda kv: -kv[1]["unique_params"]):
            pts.append({"vocab": vn, "n_layers": n, "unique_params": v["unique_params"],
                        "answer_bpb": v["answer_bpb"]["all"]["bpb"],
                        "bpb_by_category": {c: x["bpb"] for c, x in v["answer_bpb"]["by_category"].items()},
                        "flip_rate": v["holdout_generation"]["all"]["flip_rate_vs_teacher"],
                        "catastrophic_categories": catastrophic(v["holdout_generation"]),
                        "on_registered_frontier": (vn, n) in FRONTIER})
        base = next((p for p in pts if p["vocab"] == "V2_65536" and p["n_layers"] == 18), None)
        better = []
        if base:
            for p in pts:
                if p["on_registered_frontier"] and p is not base and \
                        p["answer_bpb"] <= base["answer_bpb"] * (1 - FRONTIER_MARGIN) and \
                        set(p["catastrophic_categories"]) <= set(base["catastrophic_categories"]):
                    better.append(p)
        frontier = {"points": pts, "baseline_V2_18": base,
                    "safer_alternatives": sorted(better, key=lambda p: p["answer_bpb"]),
                    "decision": ("UNDETERMINED: V2_65536/18 was not measured in S6" if base is None else
                                 "V2/18 stays the recommended first student" if not better else
                                 f"SAFER NEARBY CONFIGURATION: {min(better, key=lambda p: p['answer_bpb'])['vocab']}/"
                                 f"{min(better, key=lambda p: p['answer_bpb'])['n_layers']}")}

    res = {"schema": 1, "record": "SMALLVD_LAYER_REPORT", "experiment": "SMALLVD-001",
           "teacher_holdout_accuracy": tacc, "single_layer": single, "diagnostic_rank_correlation_with_causal": corr,
           "greedy_trajectory": [{k: t[k] for k in ("n_layers", "removed", "objective")} for t in S4["trajectory"]],
           "strategies_by_depth": depths, "greedy_vs_single_jaccard": jaccard,
           "selected_layer_set": {"depth": a.select_depth, "strategy": sel["strategy"], "kept": sel["kept"],
                                  "removed": sel["removed"], "holdout_objective": sel["holdout_objective"],
                                  "catastrophic_categories": sel["catastrophic_categories"]},
           "frontier": frontier, "quality_claim": False, "release_permitted": False}
    with open(a.out_json, "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, allow_nan=False)

    md = [f"# SMALLVD-001 layer study — report (generated by `report_layer_study.py`)\n",
          "**Surgery-only. Every number is immediate structural damage before any recovery.**\n",
          "## Teacher reference accuracy on the HOLDOUT generation panel\n",
          "| category | correct / scored |", "|---|---:|"]
    md += [f"| {c} | {v['correct']} / {v['scored']} ({v['accuracy']:.3f}) |" for c, v in tacc.items()]
    md += ["\n## Single-block bypass (causal) vs diagnostics\n",
           "| L | holdout obj | P95 row KL | top-1 | flip | worst category | rel ‖ΔF‖ | cos | CKA | BI |",
           "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in single:
        md.append(f"| {r['layer']} | {r['holdout_objective']:.4f} | {r['holdout_kl_p95_row']:.4f} | {r['holdout_top1']:.3f} | "
                  f"{r['flip_rate']:.3f} | {r['worst_category']} ({r['worst_category_kl']:.3f}) | {r['rel_update_norm']:.3f} | "
                  f"{r['cos_in_out']:.4f} | {r['cka_in_out']:.4f} | {r['block_influence']:.4f} |")
    md += ["\nSpearman rank correlation with the causal HOLDOUT objective: " +
           ", ".join(f"{k} {v:+.3f}" for k, v in corr.items()) + "\n",
           "## Registered selection (§4): lowest HOLDOUT objective at "
           f"{a.select_depth} blocks → **{sel['strategy']}**, removed {sel['removed']}, "
           f"objective {sel['holdout_objective']:.4f}, catastrophic categories: "
           f"{sel['catastrophic_categories'] or 'none'}\n",
           "## Strategies by depth (HOLDOUT)\n",
           "| depth | strategy | objective | flip | accuracy | removed | catastrophic |", "|---:|---|---:|---:|---:|---|---|"]
    for n in sorted(depths, reverse=True):
        for r in depths[n]:
            acc = "—" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
            md.append(f"| {n} | {r['strategy']} | {r['holdout_objective']:.4f} | {r['flip_rate']:.3f} | {acc} | "
                      f"{r['removed']} | {', '.join(r['catastrophic_categories']) or '—'} |")
    md.append("\nGreedy vs single-ablation set overlap (Jaccard): " +
              ", ".join(f"{n}L {j:.2f}" for n, j in sorted(jaccard.items(), reverse=True)) + "\n")
    if frontier:
        md += ["## Iso-parameter frontier (§5, tokenizer-independent)\n",
               "| config | unique params | answer bpb | flip | catastrophic | registered frontier |",
               "|---|---:|---:|---:|---|:--:|"]
        for p in frontier["points"]:
            md.append(f"| {p['vocab']}/{p['n_layers']} | {p['unique_params']:,} | {p['answer_bpb']:.4f} | "
                      f"{p['flip_rate']:.3f} | {', '.join(p['catastrophic_categories']) or '—'} | "
                      f"{'yes' if p['on_registered_frontier'] else ''} |")
        md.append(f"\n**§5 decision:** {frontier['decision']}\n")
    with open(a.out_md, "x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[-12:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
