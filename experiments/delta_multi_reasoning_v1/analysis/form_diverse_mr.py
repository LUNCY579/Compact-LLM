"""Form-diverse multi-reasoning: aggregate across SURFACE VARIATION, not across roles.

    python analysis/form_diverse_mr.py --root <records>/rci002-20260918 \
        --gen B=<...>/gen-rci2_calib_B/predictions.jsonl \
        --gen D=<...>/gen-rci2_calib_D/predictions.jsonl \
        --out <...>/FORM_DIVERSE_MR.json

**Zero new model calls.** Every generation this needs already exists.

## Why this and not the multi-reasoning we built

DMRF-001 measured our multi-reasoning implementation as the **identity**: 0 repairs and 0 damage
over 128 items on the delta candidate, -4 on the frozen source, role-oracle headroom of 2, and
`uniquely_solved_by` empty for all four roles. The roles diverged textually -- 66% non-identical
outputs -- and produced correlated outcomes anyway. So the machinery worked and there was
nothing to select between.

The decomposition of RCI-001 says why, and it is not that ensembling cannot work here. Those
roles vary **persona and instruction**. The errors vary with the **question's surface form**:
the same model scored 25/25 on "Which role ..." and 2/25 on "Who ...", emitting a person on
23/25 of the latter and 0/25 of the former, for the same fact in the same context. Role
diversity is diversity along a dimension the errors do not live on. Form diversity is diversity
along the dimension they do.

## The test, which is free

Each pack renders one generated world three times as answerable rows -- `base`, `paraphrase`,
`counterfactual` -- with different wording and independently shuffled block order. On the five
strata where the counterfactual is a no-op (the carve-out family and the exclusion count) all
three carry the **same reference**, so they are three surface forms of one question.

    single form   accuracy of each variant alone
    MAJORITY      the modal answer over the three, ties broken by abstaining
    UNANIMOUS     answer only when all three agree, else abstain

Reported per system, and the contrast that matters is **B versus D**: if the delta made answers
more determined by surface form, D should be the one form-diversity rescues more of. That is a
directional prediction, and it can fail.

**This is a DEV measurement on a spent calibration twin.** It is not a confirmation, it does not
touch a sealed bank, and it may not be cited as a result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired import paired  # noqa: E402

SENTINEL = "unknown"
ARTICLE = re.compile(r"^(the|a|an)\s+")

#: Strata whose `counterfactual` variant re-renders the same world with the same answer, so all
#: three answerable variants are surface forms of one question.
SAME_ANSWER_STRATA = {
    "carve_out_role_congruent", "carve_out_role_incongruent",
    "carve_out_badge_congruent", "carve_out_badge_incongruent",
    "exclusion_count_number",
}
FORMS = ("base", "paraphrase", "counterfactual")


def normalise(v):
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    s = " ".join(s.lower().replace(".", " ").replace(",", " ").split())
    return ARTICLE.sub("", s)


def aggregate(answers, rule):
    """answers: the normalised prediction for each form. Returns the aggregated answer."""
    votes = Counter(a for a in answers if a)
    if not votes:
        return SENTINEL
    top, n = votes.most_common(1)[0]
    if rule == "unanimous":
        return top if n == len(answers) else SENTINEL
    # MAJORITY: a strict plurality only. A three-way split is not a majority, and answering
    # anyway would make this a first-form-wins rule wearing an ensemble's name.
    if n * 2 > len(answers):
        return top
    return SENTINEL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--bank", default="RCI_CALIBRATION.jsonl")
    ap.add_argument("--gen", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / a.bank).read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = {}
    for spec in a.gen:
        lab, _, path = spec.partition("=")
        preds[lab] = {r["id"]: r.get("prediction", "") for r in
                      (json.loads(l) for l in
                       Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}

    # pack -> variant -> row, restricted to strata where all three forms share a reference
    packs = defaultdict(dict)
    for r in rows:
        if r["answerable"] and r["task_type"] in SAME_ANSWER_STRATA:
            packs[r["scenario_id"]][r["variant"]] = r
    packs = {k: v for k, v in packs.items() if all(f in v for f in FORMS)}

    # The invariant this whole design rests on, ASSERTED rather than assumed.
    bad = [k for k, v in packs.items()
           if len({v[f]["reference_normalised"] for f in FORMS}) != 1]
    if bad:
        raise SystemExit(f"{len(bad)} packs do not share a reference across forms, e.g. "
                         f"{bad[:3]} -- form-diverse aggregation is meaningless there")

    out = {
        "what_this_is": "DEV measurement on a SPENT calibration twin. Zero new model calls. "
                        "Not a confirmation and not citable as a result.",
        "packs": len(packs), "strata": sorted(SAME_ANSWER_STRATA), "forms": list(FORMS),
        "systems": {},
    }

    per_system_correct = {}
    for lab, pr in preds.items():
        single = {f: 0 for f in FORMS}
        agg = {"majority": 0, "unanimous": 0}
        agree3 = agree2 = agree0 = 0
        correct_flags = {}
        for sid, v in packs.items():
            ref = v[FORMS[0]]["reference_normalised"]
            ans = [normalise(pr.get(v[f]["id"], "")) for f in FORMS]
            for f, x in zip(FORMS, ans):
                single[f] += x == ref
            for rule in agg:
                agg[rule] += aggregate(ans, rule) == ref
            distinct = len(set(ans))
            agree3 += distinct == 1
            agree2 += distinct == 2
            agree0 += distinct == 3
            correct_flags[sid] = {
                "best_single": max(x == ref for x in ans),
                "mean_single": sum(x == ref for x in ans) / len(ans),
                **{f"single_{f}": (x == ref) for f, x in zip(FORMS, ans)},
                **{rule: (aggregate(ans, rule) == ref) for rule in agg},
            }
        per_system_correct[lab] = correct_flags
        n = len(packs)

        # PER STRATUM, because an aggregate repair can be one stratum wearing a disguise. The
        # peer session's point, and it is the same habit that exposed a single-family
        # composition collapse hiding inside a global 60-row loss: a repair confined to one
        # stratum is much weaker evidence than one spread across them, and you cannot tell
        # which you have without splitting.
        by_str = {}
        for sid, v in packs.items():
            t = v[FORMS[0]]["task_type"]
            ref = v[FORMS[0]]["reference_normalised"]
            ans = [normalise(pr.get(v[f]["id"], "")) for f in FORMS]
            d = by_str.setdefault(t, {"n": 0, "mean_single": 0.0, "majority": 0,
                                      "disagree": 0})
            d["n"] += 1
            d["mean_single"] += sum(x == ref for x in ans) / len(ans)
            d["majority"] += aggregate(ans, "majority") == ref
            d["disagree"] += len(set(ans)) > 1
        for t, d in by_str.items():
            d["mean_single_rate"] = round(d.pop("mean_single") / d["n"], 4)
            d["majority_rate"] = round(d["majority"] / d["n"], 4)
            d["gain"] = round(d["majority_rate"] - d["mean_single_rate"], 4)
            d["disagreement_rate"] = round(d.pop("disagree") / d["n"], 4)
        out["systems"][lab] = {
            "n_packs": n,
            "single_form": {f: {"correct": single[f], "rate": round(single[f] / n, 4)}
                            for f in FORMS},
            "mean_single_form_rate": round(sum(single.values()) / (n * len(FORMS)), 4),
            "majority": {"correct": agg["majority"], "rate": round(agg["majority"] / n, 4)},
            "unanimous": {"correct": agg["unanimous"], "rate": round(agg["unanimous"] / n, 4)},
            "gain_majority_over_mean_single": round(
                agg["majority"] / n - sum(single.values()) / (n * len(FORMS)), 4),
            "form_agreement": {"all_three_agree": agree3, "two_distinct": agree2,
                               "all_three_differ": agree0,
                               "disagreement_rate": round((agree2 + agree0) / n, 4)},
            "per_stratum": by_str,
            "strata_with_positive_gain": sorted(t for t, d in by_str.items() if d["gain"] > 0),
        }

    # The directional prediction: the delta is the more form-sensitive system, so it should be
    # the one form-diversity rescues more of.
    if sorted(preds) == ["B", "D"]:
        gb = out["systems"]["B"]["gain_majority_over_mean_single"]
        gd = out["systems"]["D"]["gain_majority_over_mean_single"]
        pb_, pd_ = per_system_correct["B"], per_system_correct["D"]
        # Paired, per system: does majority repair what single-form got wrong?
        rep = {}
        for lab, flags in (("B", pb_), ("D", pd_)):
            p = paired(f"{lab}:base", f"{lab}:majority",
                       {k: v["single_base"] for k, v in flags.items()},
                       {k: v["majority"] for k, v in flags.items()})
            rep[lab] = {k: val for k, val in p.as_dict().items()
                        if k not in ("repair_ids", "damage_ids")}
        out["majority_vs_base_paired"] = rep
        out["directional_prediction"] = {
            "claim": "the delta is more form-sensitive, so form diversity should rescue more "
                     "of it than of the frozen source",
            "gain_B": gb, "gain_D": gd,
            "holds": gd > gb,
            "disagreement_rate_B": out["systems"]["B"]["form_agreement"]["disagreement_rate"],
            "disagreement_rate_D": out["systems"]["D"]["form_agreement"]["disagreement_rate"],
            "note": ("disagreement rate is the precondition -- a system that answers every form "
                     "identically has nothing for any aggregator to work with, and for it "
                     "form-diverse MR is the identity for the same reason role-diverse MR was"),
        }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{len(packs)} packs, {len(FORMS)} surface forms each, same reference across forms\n")
    print(f"{'system':<8}{'base':>8}{'para':>8}{'cf':>8}{'mean':>8}"
          f"{'MAJORITY':>10}{'UNANIM':>9}{'gain':>8}{'disagree':>10}")
    for lab, v in out["systems"].items():
        s = v["single_form"]
        print(f"{lab:<8}{s['base']['rate']:>8.3f}{s['paraphrase']['rate']:>8.3f}"
              f"{s['counterfactual']['rate']:>8.3f}{v['mean_single_form_rate']:>8.3f}"
              f"{v['majority']['rate']:>10.3f}{v['unanimous']['rate']:>9.3f}"
              f"{v['gain_majority_over_mean_single']:>+8.3f}"
              f"{v['form_agreement']['disagreement_rate']:>10.3f}")
    print("\nper stratum (mean single -> majority, gain, disagreement)")
    for lab, v in out["systems"].items():
        for t, d in sorted(v["per_stratum"].items()):
            print(f"  {lab}  {t:<32} n={d['n']:<3} {d['mean_single_rate']:.3f} -> "
                  f"{d['majority_rate']:.3f}  gain {d['gain']:+.3f}  "
                  f"disagree {d['disagreement_rate']:.3f}")
    if "directional_prediction" in out:
        d = out["directional_prediction"]
        print(f"\nprediction -- form diversity rescues D more than B: "
              f"gain_B {d['gain_B']:+.3f} vs gain_D {d['gain_D']:+.3f}  "
              f"-> {'HOLDS' if d['holds'] else 'FAILS'}")
        for lab, r in out["majority_vs_base_paired"].items():
            print(f"  {lab}: majority vs base -- repair {r['repair']}, damage {r['damage']}, "
                  f"p {r['mcnemar_exact_two_sided_p']:.3e}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
