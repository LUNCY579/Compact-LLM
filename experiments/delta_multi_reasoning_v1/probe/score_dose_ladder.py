"""ALIAS-002 — score the dose ladder against the predictions registered in ALIAS002_DOSE_PREREG.md.

    python probe/score_dose_ladder.py --root <records>/alias001-20260918 \
        --out <records>/alias001-20260918/ALIAS_DOSE_RESULT.json

**Written while the arms were still generating**, so the analysis exists before the numbers do.

Primary, as registered: **Spearman rank correlation between alpha and alias accuracy**, pooled
over conditions, predicted negative and monotone. With five doses the permutation null has only
120 orderings, so the p-value is computed EXACTLY by enumeration rather than from a normal
approximation that does not hold at n = 5.

Secondary, and the part that discriminates capacity from a generic dose effect: the three
per-condition curves. Capacity predicts roughly parallel declines — no dose x salience
interaction. Damage is always conditioned on the rows the **alpha = 0 source** got right, since
that is the only set on which a loss was possible.

`high` absolute rates remain unreadable by construction (the answer is the most frequent person
there; extraction floor 1.000). The paired contrasts are readable at every dose.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import fisher_exact_2x2, paired  # noqa: E402

CONDITIONS = ["low", "equal", "high"]
#: (label, alpha, generation directory). alpha = 0 is the frozen source, already scored.
LADDER = [("B", 0.00, "gen-B"), ("a025", 0.25, "gen-a025"), ("a05", 0.50, "gen-a05"),
          ("a075", 0.75, "gen-a075"), ("a10", 1.00, "gen-a10")]


def norm(v):
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def spearman(xs, ys):
    """Rho plus an EXACT two-sided permutation p. n = 5 here; 120 orderings, so enumerate."""
    n = len(xs)

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    rho = num / den if den else 0.0
    hits = 0
    total = 0
    for perm in permutations(ry):
        total += 1
        nu = sum((a - mx) * (b - my) for a, b in zip(rx, perm))
        if abs(nu) >= abs(num) - 1e-12:
            hits += 1
    return rho, hits / total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    preds, missing = {}, []
    for lab, alpha, d in LADDER:
        p = a.root / d / "predictions.jsonl"
        if not p.exists():
            missing.append(lab)
            continue
        preds[lab] = {r["id"]: r.get("prediction", "") for r in
                      (json.loads(l) for l in
                       p.read_text(encoding="utf-8").splitlines() if l.strip())}
    if missing:
        raise SystemExit(f"no predictions for {missing} -- the ladder is incomplete")

    base = preds["B"]
    per_dose, per_cond = {}, {c: {"alpha": [], "acc": [], "dmg_rate": []} for c in CONDITIONS}
    for lab, alpha, _ in LADDER:
        pr = preds[lab]
        d = {"alpha": alpha, "conditions": {}}
        tot_ok = tot_n = 0
        for c in CONDITIONS:
            sub = [r for r in rows if r["condition"] == c]
            cb = {r["id"]: norm(base.get(r["id"], "")) == r["reference_normalised"] for r in sub}
            ca = {r["id"]: norm(pr.get(r["id"], "")) == r["reference_normalised"] for r in sub}
            pp = paired("alpha0", lab, cb, ca)
            ok = sum(ca.values())
            tot_ok += ok
            tot_n += len(sub)
            d["conditions"][c] = {
                "n": len(sub), "correct": ok, "acc": round(ok / len(sub), 4),
                "repair": pp.repair, "damage": pp.damage,
                "damage_rate_given_alpha0_correct": (
                    round(pp.damage / pp.baseline_correct, 4) if pp.baseline_correct else None),
            }
            per_cond[c]["alpha"].append(alpha)
            per_cond[c]["acc"].append(ok / len(sub))
            per_cond[c]["dmg_rate"].append(
                d["conditions"][c]["damage_rate_given_alpha0_correct"] or 0.0)
        d["pooled"] = {"n": tot_n, "correct": tot_ok, "acc": round(tot_ok / tot_n, 4)}
        per_dose[lab] = d

    alphas = [al for _, al, _ in LADDER]
    pooled_acc = [per_dose[lab]["pooled"]["acc"] for lab, _, _ in LADDER]
    rho, pval = spearman(alphas, pooled_acc)
    monotone = all(pooled_acc[i] >= pooled_acc[i + 1] for i in range(len(pooled_acc) - 1))

    cond_stats = {}
    for c in CONDITIONS:
        r_, p_ = spearman(per_cond[c]["alpha"], per_cond[c]["acc"])
        cond_stats[c] = {"spearman_rho": round(r_, 4), "p_exact": round(p_, 4),
                         "acc_by_dose": [round(x, 4) for x in per_cond[c]["acc"]],
                         "damage_rate_by_dose": per_cond[c]["dmg_rate"],
                         "drop_alpha0_to_alpha1": round(per_cond[c]["acc"][0]
                                                        - per_cond[c]["acc"][-1], 4)}

    # Dose x salience interaction at the extreme dose: does alpha=1 damage differently across
    # conditions than a mid dose does? Conditioned on the alpha=0 correct set throughout.
    hi = per_dose["a10"]["conditions"]
    mid = per_dose["a05"]["conditions"]

    def _cell(x, dose):
        n0 = per_dose["B"]["conditions"][x]["correct"]
        return dose[x]["damage"], n0 - dose[x]["damage"]

    # Two DIFFERENT questions, and only the second is the interaction. Naming them apart
    # because the first version printed both under "dose x salience" and the within-condition
    # contrast is not an interaction at all.
    inter = {}
    for x in ("low", "high"):
        # within one condition: is damage larger at the top of the ladder than mid-ladder?
        inter[f"within_{x}__a10_vs_a05"] = fisher_exact_2x2(*_cell(x, hi), *_cell(x, mid))
    # THE INTERACTION: at the top dose, does damage differ BETWEEN conditions?
    inter["INTERACTION_low_vs_high_at_a10"] = fisher_exact_2x2(*_cell("low", hi),
                                                               *_cell("high", hi))

    drops = [cond_stats[c]["drop_alpha0_to_alpha1"] for c in CONDITIONS]
    parallel = (max(drops) - min(drops)) <= 0.15

    if monotone and rho < 0 and pval <= 0.05:
        verdict = "MONOTONE_DOSE_RESPONSE"
    elif rho < 0 and not monotone:
        verdict = "DECLINING_BUT_NOT_MONOTONE"
    elif abs(rho) < 0.3:
        verdict = "FLAT -- not a dose effect"
    else:
        verdict = "UNEXPECTED"

    out = {
        "record": "ALIAS_002_DOSE_RESULT",
        "class": "MECHANISM SCOUT -- not a confirmation; no read budget was spent",
        "registered_primary": "Spearman(alpha, pooled alias accuracy), predicted negative "
                              "and monotone; exact permutation p at n=5",
        "alphas": alphas,
        "pooled_accuracy_by_dose": pooled_acc,
        "spearman_rho": round(rho, 4), "p_exact_permutation": round(pval, 4),
        "monotone_non_increasing": monotone,
        "verdict": verdict,
        "per_dose": per_dose,
        "per_condition": cond_stats,
        "dose_by_salience": {
            "fisher": inter,
            "drops_alpha0_to_alpha1": dict(zip(CONDITIONS, drops)),
            "roughly_parallel": parallel,
            "reading": ("capacity predicts three roughly parallel declines. A significant "
                        "dose x salience interaction would mean magnitude and salience trade "
                        "against each other, which no current account predicts."),
        },
        "reference_points_not_on_the_alpha_grid": {},
        "caveats": [
            "monotone decline is NECESSARY for the capacity account and NOT SUFFICIENT for it: "
            "magnitude is already the single operative lever across mask, rank and scalar "
            "parameterisations, so any mechanism that scales with magnitude predicts this too",
            "`high` absolute rates are unreadable by construction (extraction floor 1.000); "
            "the paired contrasts are readable at every dose",
            "one training draw of one delta; 120 rows per condition per dose",
        ],
    }
    # Place the promoted k=0.25 core on this axis if its predictions are here. It is a
    # DIFFERENT parameterisation -- magnitude MASKING, not scaling -- so it is a reference
    # point rather than a rung, and where it lands is a cross-parameterisation check on the
    # programme's finding that magnitude is the single operative lever.
    dpath = a.root / "gen-D" / "predictions.jsonl"
    if dpath.exists():
        prd = {r["id"]: r.get("prediction", "") for r in
               (json.loads(l) for l in
                dpath.read_text(encoding="utf-8").splitlines() if l.strip())}
        ok = sum(1 for r in rows if norm(prd.get(r["id"], "")) == r["reference_normalised"])
        acc = ok / len(rows)
        between = None
        for i in range(len(alphas) - 1):
            if pooled_acc[i] >= acc >= pooled_acc[i + 1]:
                between = [alphas[i], alphas[i + 1]]
        out["reference_points_not_on_the_alpha_grid"]["k0.25_promoted_core"] = {
            "correct": ok, "n": len(rows), "acc": round(acc, 4),
            "lands_between_alphas": between,
            "note": ("magnitude MASKING, not scaling: k=0.25 keeps the largest quarter of the "
                     "update's elements and so preserves most of its norm, where alpha=0.25 "
                     "scales every element to a quarter. The two `0.25`s are very different "
                     "doses, and this is where the masked one actually sits."),
        }
        print(f"\nreference point -- promoted k=0.25 core: {ok}/{len(rows)} = {acc:.4f}, "
              f"lands between alpha {between}")

    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'alpha':>6}{'pooled acc':>12}  |  " + "  ".join(f"{c:>18}" for c in CONDITIONS))
    for i, (lab, al, _) in enumerate(LADDER):
        cells = []
        for c in CONDITIONS:
            v = per_dose[lab]["conditions"][c]
            dr = v["damage_rate_given_alpha0_correct"]
            cells.append(f"{v['acc']:.3f} dmg {v['damage']:>2}"
                         f"{'' if dr is None else f'/{dr:.2f}'}")
        print(f"{al:>6.2f}{pooled_acc[i]:>12.4f}  |  " + "  ".join(f"{x:>18}" for x in cells))
    print(f"\nSpearman(alpha, pooled acc) = {rho:+.4f}, exact permutation p = {pval:.4f}, "
          f"monotone non-increasing = {monotone}")
    print("per condition:")
    for c, v in cond_stats.items():
        print(f"  {c:<7} rho {v['spearman_rho']:+.3f} (p {v['p_exact']:.3f})  "
              f"drop alpha0->alpha1 {v['drop_alpha0_to_alpha1']:+.3f}")
    print(f"drops roughly parallel: {parallel}  {dict(zip(CONDITIONS, drops))}")
    print("Fisher:")
    for k, v in inter.items():
        print(f"  {k:<32} {v:.4f}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
