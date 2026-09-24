"""SMALLVD-001 -- layer-ranking report and the registered decisions, computed from the stage files.

Applies the rules exactly as registered in SMALLVD001_PREREG.md (with Amendment 1), and nothing else:
  §4  A2/A3 layer set at 18 blocks = the candidate set with the LOWEST SELECT2 objective
      (category-mean answer KL). SELECT chooses nothing among strategies; REPORT chooses nothing.
  §4  protected-skill flag: a category whose REPORT generation flip rate is >= 0.90 at the selected
      set is CATASTROPHIC for that skill, whatever the average says.
  §5  V2/18 stays the recommended first student unless another frontier configuration has SELECT2
      answer bpb at least 1 % lower AND no additional SELECT2 CATASTROPHIC category. Every figure
      reported for the chosen configuration is its REPORT figure.
  §7  G4 (the brief's definition): damage measured separately for A1 (vocabulary only), A2 (depth
      only) and A3 (combined), here on the REPORT panels; additivity Δ(A3) vs Δ(A1)+Δ(A2) on answer
      bpb with a paired per-row bootstrap interval, descriptive only.
  §10 V-SAFE: A1 has no CATASTROPHIC category on REPORT (a HOLD condition; A2/A3 are not gated on it).
  §10 D-COLLAPSE (Amendment 3): the recommended student keeps a POSITIVE share of the teacher's
      advantage over knowing nothing, in every chat category, on REPORT answer bpb (HOLD otherwise);
      categories below half, plus its REPORT-CATASTROPHIC ones, form `recovery_protected_categories`.

Also reports the diagnostics (update norm, cosine, CKA, block influence) next to the causal
single-ablation damage, with rank correlations, so the reader can see how far "looks redundant"
is from "is redundant" on this model. Written create-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer_select as ls  # noqa: E402

CATASTROPHIC_FLIP = 0.90
FRONTIER_MARGIN = 0.01
FRONTIER = {("V1_98304", 16), ("V2_65536", 18), ("V3_49152", 19), ("V4_32768", 20), ("V0", 12)}
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 2000, 17
COLLAPSE_RETAINED, SEVERE_RETAINED = 0.0, 0.5          # Amendment 3


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


def teacher_accuracy(panels, teacher_rows, sub=None, part=None):
    import layer_ablation as la
    by_cat = {}
    for r in panels:
        if r["kind"] != "chat" or r["split"] != "holdout" or r["id"] not in teacher_rows:
            continue                      # only rows the study actually generated
        if part is not None and sub.get(r["id"]) != part:
            continue
        ok = la.score_row(teacher_rows[r["id"]]["text"], r)
        if ok is None:
            continue
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += ok
        c[1] += 1
    return {c: {"correct": a, "scored": n, "accuracy": a / n} for c, (a, n) in sorted(by_cat.items())}


def catastrophic(gen: dict) -> list[str]:
    return sorted(c for c, v in gen["by_category"].items() if v["flip_rate_vs_teacher"] >= CATASTROPHIC_FLIP)


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def bpb_of(rows) -> float:
    return sum(r["nats"] for r in rows) / math.log(2) / sum(r["bytes"] for r in rows)


def additivity(per_arm: dict, ids: list[str]) -> dict:
    """Δk = bpb(Ak) - bpb(A0) on the same rows; interaction = Δ3 - Δ1 - Δ2. Paired bootstrap over
    rows (the same resample for every arm). Descriptive only -- no significance claim (§7)."""
    rows = {k: {r["id"]: r for r in v} for k, v in per_arm.items()}

    def stats(sample):
        b = {k: bpb_of([rows[k][i] for i in sample]) for k in rows}
        d = {k: b[k] - b["A0"] for k in ("A1", "A2", "A3")}
        return b, d, d["A3"] - d["A1"] - d["A2"]
    b, d, inter = stats(ids)
    rng = random.Random(BOOTSTRAP_SEED)
    boots = sorted(stats([rng.choice(ids) for _ in ids])[2] for _ in range(BOOTSTRAP_RESAMPLES))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    return {"rows": len(ids), "bpb": b, "delta_vs_A0": d, "interaction_A3_minus_A1_minus_A2": inter,
            "interaction_bootstrap_95": [lo, hi], "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            "reading": "descriptive only (§7); interaction > 0 means the combined damage exceeds the sum"}


def retained_fraction(entry: dict, a0: dict, sub: dict) -> dict:
    """Amendment 3. Per chat category on REPORT: the share of the teacher's advantage over KNOWING
    NOTHING that the student keeps, in answer bits per byte -- (u - s) / (u - t), with u = uniform
    over the student's OWN vocabulary under its OWN tokenisation (tokens * log2 V / bytes), s the
    student's bpb and t the teacher's (A0) bpb on the same rows and bytes. 1 = teacher level,
    0 = no better than uniform, < 0 = worse. Answer bpb does not saturate the way a 64-token
    full-string flip does, and it exists in every chat category (open-ended rows carry reference
    answers; only their SCORER is absent)."""
    V = entry["vocab_rows"]
    t_rows = {r["id"]: r for r in a0["answer_bpb"]["holdout_per_item"]}
    by = {}
    for r in entry["answer_bpb"]["holdout_per_item"]:
        if sub.get(r["id"]) == "report":
            by.setdefault(r["category"], []).append(r)
    out = {}
    for c, rows in sorted(by.items()):
        byts = sum(r["bytes"] for r in rows)
        s_ = sum(r["nats"] for r in rows) / math.log(2) / byts
        t_ = sum(t_rows[r["id"]]["nats"] for r in rows) / math.log(2) / byts
        u_ = sum(r["tokens"] for r in rows) * math.log2(V) / byts
        out[c] = {"rows": len(rows), "student_bpb": s_, "teacher_bpb": t_, "uniform_bpb": u_,
                  "retained": (u_ - s_) / (u_ - t_) if u_ > t_ else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path)
    ap.add_argument("--select-depth", type=int, default=18)
    ap.add_argument("--a-vocab", default="V2_65536", help="the A1/A3 vocabulary (its S6 name)")
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    a = ap.parse_args(argv)
    panels = [json.loads(l) for l in a.panels.read_text(encoding="utf-8").splitlines() if l.strip()]
    S1, S2, S3 = load(a.study, "S1_teacher_outputs"), load(a.study, "S2_diagnostics"), load(a.study, "S3_single_ablation")
    S4, S5 = load(a.study, "S4_greedy_backward"), load(a.study, "S5_depth_strategies")
    S6 = load(a.study, "S6_vocab_frontier") if (a.study / "S6_vocab_frontier.json").exists() else None
    L = len(S3["layers"])
    diag = {b["layer"]: b for b in S2["per_block"]}

    # the HOLDOUT rows the study actually used (all of them, unless a dry run subsampled), and the
    # SAME subsplit the study computed from them
    used = {x["id"] for x in S3["layers"]["0"]["holdout_per_item"]}
    hold_rows = [r for r in panels if r["split"] == "holdout" and r["id"] in used]
    sub = ls.holdout_subsplit(hold_rows)
    split_sha = hashlib.sha256(json.dumps(sorted(sub.items())).encode("utf-8")).hexdigest()
    for name, st in (("S1", S1), ("S3", S3), ("S5", S5), ("S6", S6)):
        if st is not None and st.get("_holdout_subsplit_sha256") != split_sha:
            raise SystemExit(f"{name} was computed under HOLDOUT subsplit {st.get('_holdout_subsplit_sha256')}, "
                             f"but the current definition gives {split_sha}: the REPORT figures would not be REPORT")
    tacc = teacher_accuracy(panels, S1["rows"], sub, "report")
    tacc_all = teacher_accuracy(panels, S1["rows"])

    # ---- single-layer causal table + diagnostics (all HOLDOUT rows: nothing is selected here)
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
    corr = {k: (None if math.isnan(v) else v) for k, v in corr.items()}

    # ---- strategies per depth + registered selection (Amendment 1: SELECT2 chooses, REPORT reports)
    depths = {}
    for n, strat in S5["depths"].items():
        rows = []
        for name, v in strat.items():
            g = v["holdout_generation"]["report"]
            rows.append({"strategy": name, "kept": v["kept"], "removed": [l for l in range(L) if l not in v["kept"]],
                         "select2_objective": v["select2_objective"], "report_objective": v["report_objective"],
                         "holdout_objective": v["holdout_objective"],
                         "flip_rate": g["all"]["flip_rate_vs_teacher"], "accuracy": g["all"]["accuracy"],
                         "by_category_flip": {c: x["flip_rate_vs_teacher"] for c, x in g["by_category"].items()},
                         "by_category_accuracy": {c: x["accuracy"] for c, x in g["by_category"].items()},
                         "by_category_kl": {c: x["kl_mean_pos"] for c, x in v["report"]["by_category"].items()},
                         "catastrophic_categories": catastrophic(g),
                         "catastrophic_categories_select2": catastrophic(v["holdout_generation"]["select2"])})
        rows.sort(key=lambda r: r["select2_objective"])
        depths[int(n)] = rows
    sel = depths[a.select_depth][0]
    greedy_sets = {t["n_layers"]: t["kept"] for t in S4["trajectory"]}
    single_sets = {n: next(r["kept"] for r in depths[n] if r["strategy"] == "single_ablation") for n in depths}
    jaccard = {n: len(set(greedy_sets[n]) & set(single_sets[n])) / len(set(greedy_sets[n]) | set(single_sets[n]))
               for n in depths if n in greedy_sets}

    # ---- frontier rule (SELECT2 decides; REPORT is what gets reported)
    frontier = None
    if S6:
        byname = {(v["vocab"], v["n_layers"]): v for v in S6["configs"].values()}
        pts = []
        for (vn, n), v in sorted(byname.items(), key=lambda kv: -kv[1]["unique_params"]):
            ab, hg = v["answer_bpb"], v["holdout_generation"]
            pts.append({"vocab": vn, "n_layers": n, "unique_params": v["unique_params"],
                        "select2_answer_bpb": ab["select2"]["all"]["bpb"],
                        "report_answer_bpb": ab["report"]["all"]["bpb"],
                        "report_bpb_by_category": {c: x["bpb"] for c, x in ab["report"]["by_category"].items()},
                        "report_flip_rate": hg["report"]["all"]["flip_rate_vs_teacher"],
                        "catastrophic_categories_select2": catastrophic(hg["select2"]),
                        "catastrophic_categories": catastrophic(hg["report"]),
                        "on_registered_frontier": (vn, n) in FRONTIER})
        base = next((p for p in pts if p["vocab"] == "V2_65536" and p["n_layers"] == 18), None)
        better = []
        if base:
            for p in pts:
                if p["on_registered_frontier"] and p is not base and \
                        p["select2_answer_bpb"] <= base["select2_answer_bpb"] * (1 - FRONTIER_MARGIN) and \
                        set(p["catastrophic_categories_select2"]) <= set(base["catastrophic_categories_select2"]):
                    better.append(p)
        pick = min(better, key=lambda p: p["select2_answer_bpb"]) if better else None
        frontier = {"points": pts, "baseline_V2_18": base,
                    "safer_alternatives": sorted(better, key=lambda p: p["select2_answer_bpb"]),
                    "decision": ("UNDETERMINED: V2_65536/18 was not measured in S6" if base is None else
                                 "V2/18 stays the recommended first student" if not better else
                                 f"SAFER NEARBY CONFIGURATION: {pick['vocab']}/{pick['n_layers']}")}

    # ---- G4 (brief: damage measured separately for A1, A2, A3) + V-SAFE + additivity, on REPORT
    g4 = None
    if S6:
        cfgs = {(v["vocab"], v["n_layers"]): v for v in S6["configs"].values()}
        arms = {"A0": ("V0", L), "A1": (a.a_vocab, L), "A2": ("V0", a.select_depth), "A3": (a.a_vocab, a.select_depth)}
        present = {k: cfgs.get(key) for k, key in arms.items()}
        missing = [f"{k} ({arms[k][0]}/{arms[k][1]}L) not measured in S6" for k, v in present.items() if v is None]
        for k in ("A2", "A3"):
            if present[k] is not None and present[k]["kept"] != sel["kept"]:
                missing.append(f"{k}: S6 layer set {present[k]['kept']} is not the registered selection {sel['kept']}")
        if not missing:
            limbs = {}
            for k, v in present.items():
                ab, hg = v["answer_bpb"]["report"], v["holdout_generation"]["report"]
                limbs[k] = {"report_answer_bpb": ab["all"]["bpb"],
                            "report_bpb_by_category": {c: x["bpb"] for c, x in ab["by_category"].items()},
                            "report_flip_rate": hg["all"]["flip_rate_vs_teacher"],
                            "report_accuracy": hg["all"]["accuracy"],
                            "report_by_category_flip": {c: x["flip_rate_vs_teacher"] for c, x in hg["by_category"].items()},
                            "report_by_category_accuracy": {c: x["accuracy"] for c, x in hg["by_category"].items()},
                            "catastrophic_categories": catastrophic(hg),
                            "finite": finite(ab["all"]["bpb"]) and all(finite(x["bpb"]) for x in ab["by_category"].values())}
            limbs["A2"]["report_kl_objective"] = sel["report_objective"]
            per_arm = {k: [r for r in v["answer_bpb"]["holdout_per_item"] if sub.get(r["id"]) == "report"]
                       for k, v in present.items()}
            add_ids = sorted(set.intersection(*[{r["id"] for r in x} for x in per_arm.values()]))
            measured = all(l["finite"] for l in limbs.values()) and finite(sel["report_objective"]) and bool(add_ids)
            # Amendment 3: D-COLLAPSE and the protected-category list, for the RECOMMENDED student
            rec_key = (pick["vocab"], pick["n_layers"]) if pick else arms["A3"]
            rec = cfgs[rec_key]
            ret = retained_fraction(rec, present["A0"], sub)
            undefined = sorted(c for c, x in ret.items() if x["retained"] is None)
            collapsed = sorted(c for c, x in ret.items() if x["retained"] is not None and x["retained"] <= COLLAPSE_RETAINED)
            protected = sorted(set(catastrophic(rec["holdout_generation"]["report"])) |
                               {c for c, x in ret.items() if x["retained"] is None or x["retained"] < SEVERE_RETAINED})
            g4 = {"arms": {k: f"{vn}/{n}L" for k, (vn, n) in arms.items()}, "limbs": limbs,
                  "additivity_report": additivity(per_arm, add_ids) if add_ids else None,
                  "G4_calibration_panels": "PASS" if measured else "FAIL",
                  "G4_full": "PENDING: no instrument shim is qualified yet, and G3 comes from the surgery arms",
                  "V_SAFE_A1_no_catastrophic": "PASS" if not limbs["A1"]["catastrophic_categories"] else "HOLD",
                  "recommended_student": f"{rec_key[0]}/{rec_key[1]}L",
                  "retained_fraction_report": ret,
                  "retained_fraction_report_A2": retained_fraction(present["A2"], present["A0"], sub),
                  "D_COLLAPSE": "HOLD" if (collapsed or undefined) else "PASS",
                  "D_COLLAPSE_categories": collapsed, "retained_undefined_categories": undefined,
                  "recovery_protected_categories": protected}
        else:
            g4 = {"G4_calibration_panels": "FAIL", "missing": missing}

    res = {"schema": 2, "record": "SMALLVD_LAYER_REPORT", "experiment": "SMALLVD-001", "amendment": 1,
           "holdout_subsplit": {"salt": ls.SUBSPLIT_SALT, "rows": len(sub), "sha256": split_sha,
                                "report_rows": sum(1 for v in sub.values() if v == "report")},
           "teacher_report_accuracy": tacc, "teacher_holdout_accuracy": tacc_all,
           "single_layer": single, "diagnostic_rank_correlation_with_causal": corr,
           "greedy_trajectory": [{k: t[k] for k in ("n_layers", "removed", "objective")} for t in S4["trajectory"]],
           "strategies_by_depth": depths, "greedy_vs_single_jaccard": jaccard,
           "selected_layer_set": {"depth": a.select_depth, "strategy": sel["strategy"], "kept": sel["kept"],
                                  "removed": sel["removed"], "chosen_on": "SELECT2",
                                  "select2_objective_SELECTED_ON": sel["select2_objective"],
                                  "report_objective": sel["report_objective"],
                                  "catastrophic_categories": sel["catastrophic_categories"]},
           "frontier": frontier, "G4": g4, "quality_claim": False, "release_permitted": False}
    with open(a.out_json, "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, allow_nan=False)

    md = [f"# SMALLVD-001 layer study — report (generated by `report_layer_study.py`)\n",
          "**Surgery-only. Every number is immediate structural damage before any recovery.**\n",
          "**Amendment 1:** SELECT2 chooses among strategies and frontier configurations; REPORT chooses "
          "nothing. Every figure reported for a selected configuration is its REPORT figure.\n",
          "## Teacher reference accuracy on the REPORT generation panel\n",
          "| category | correct / scored |", "|---|---:|"]
    md += [f"| {c} | {v['correct']} / {v['scored']} ({v['accuracy']:.3f}) |" for c, v in tacc.items()]
    md += ["\n## Single-block bypass (causal, all HOLDOUT rows; selects nothing) vs diagnostics\n",
           "| L | holdout obj | P95 row KL | top-1 | flip | worst category | rel ‖ΔF‖ | cos | CKA | BI |",
           "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in single:
        md.append(f"| {r['layer']} | {r['holdout_objective']:.4f} | {r['holdout_kl_p95_row']:.4f} | {r['holdout_top1']:.3f} | "
                  f"{r['flip_rate']:.3f} | {r['worst_category']} ({r['worst_category_kl']:.3f}) | {r['rel_update_norm']:.3f} | "
                  f"{r['cos_in_out']:.4f} | {r['cka_in_out']:.4f} | {r['block_influence']:.4f} |")
    md += ["\nSpearman rank correlation with the causal HOLDOUT objective: " +
           ", ".join(f"{k} {'n/a' if v is None else f'{v:+.3f}'}" for k, v in corr.items()) + "\n",
           "## Registered selection (§4, Amendment 1): lowest SELECT2 objective at "
           f"{a.select_depth} blocks → **{sel['strategy']}**, removed {sel['removed']}. "
           f"REPORT objective {sel['report_objective']:.4f} (SELECT2 {sel['select2_objective']:.4f}, "
           f"selected on); REPORT catastrophic categories: {sel['catastrophic_categories'] or 'none'}\n",
           "## Strategies by depth (SELECT2 chooses, REPORT reports; generation columns are REPORT)\n",
           "| depth | strategy | SELECT2 obj | REPORT obj | flip | accuracy | removed | catastrophic |",
           "|---:|---|---:|---:|---:|---:|---|---|"]
    for n in sorted(depths, reverse=True):
        for r in depths[n]:
            acc = "—" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
            md.append(f"| {n} | {r['strategy']} | {r['select2_objective']:.4f} | {r['report_objective']:.4f} | "
                      f"{r['flip_rate']:.3f} | {acc} | {r['removed']} | {', '.join(r['catastrophic_categories']) or '—'} |")
    md.append("\nGreedy vs single-ablation set overlap (Jaccard): " +
              ", ".join(f"{n}L {j:.2f}" for n, j in sorted(jaccard.items(), reverse=True)) + "\n")
    if frontier:
        md += ["## Iso-parameter frontier (§5, tokenizer-independent; SELECT2 decides, REPORT reports)\n",
               "| config | unique params | SELECT2 bpb | REPORT bpb | REPORT flip | REPORT catastrophic | registered frontier |",
               "|---|---:|---:|---:|---:|---|:--:|"]
        for p in frontier["points"]:
            md.append(f"| {p['vocab']}/{p['n_layers']} | {p['unique_params']:,} | {p['select2_answer_bpb']:.4f} | "
                      f"{p['report_answer_bpb']:.4f} | {p['report_flip_rate']:.3f} | "
                      f"{', '.join(p['catastrophic_categories']) or '—'} | {'yes' if p['on_registered_frontier'] else ''} |")
        md.append(f"\n**§5 decision:** {frontier['decision']}\n")
    if g4:
        md.append("## G4 — damage measured separately for A1 / A2 / A3 (REPORT panels)\n")
        if "limbs" in g4:
            md += ["| arm | config | REPORT answer bpb | REPORT flip | REPORT accuracy | catastrophic |",
                   "|---|---|---:|---:|---:|---|"]
            for k, l in g4["limbs"].items():
                acc = "—" if l["report_accuracy"] is None else f"{l['report_accuracy']:.3f}"
                md.append(f"| {k} | {g4['arms'][k]} | {l['report_answer_bpb']:.4f} | {l['report_flip_rate']:.3f} | "
                          f"{acc} | {', '.join(l['catastrophic_categories']) or '—'} |")
            ad = g4["additivity_report"]
            if ad:
                md.append(f"\nAdditivity (descriptive, {ad['rows']} REPORT rows): Δ(A1) {ad['delta_vs_A0']['A1']:+.4f}, "
                          f"Δ(A2) {ad['delta_vs_A0']['A2']:+.4f}, Δ(A3) {ad['delta_vs_A0']['A3']:+.4f} bpb; interaction "
                          f"{ad['interaction_A3_minus_A1_minus_A2']:+.4f} (paired bootstrap 95% "
                          f"[{ad['interaction_bootstrap_95'][0]:+.4f}, {ad['interaction_bootstrap_95'][1]:+.4f}])\n")
        md.append(f"**G4 (calibration panels):** {g4['G4_calibration_panels']} · **V-SAFE (A1 not catastrophic):** "
                  f"{g4.get('V_SAFE_A1_no_catastrophic', '—')} · full G4: {g4.get('G4_full', g4.get('missing'))}\n")
        if "retained_fraction_report" in g4:
            md += [f"## D-COLLAPSE (Amendment 3) — recommended student {g4['recommended_student']}, REPORT answer bpb\n",
                   "| category | student bpb | teacher bpb | uniform bpb | retained share |", "|---|---:|---:|---:|---:|"]
            for c, x in g4["retained_fraction_report"].items():
                rs = "undefined" if x["retained"] is None else f"{x['retained']:.3f}"
                md.append(f"| {c} | {x['student_bpb']:.4f} | {x['teacher_bpb']:.4f} | {x['uniform_bpb']:.4f} | {rs} |")
            md.append(f"\n**D-COLLAPSE:** {g4['D_COLLAPSE']} · **recovery_protected_categories:** "
                      f"{', '.join(g4['recovery_protected_categories']) or 'none'}\n")
    with open(a.out_md, "x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[-14:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
