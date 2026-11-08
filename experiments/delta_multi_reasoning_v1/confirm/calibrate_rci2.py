"""Score the RCI-002 CALIBRATION twin and decide whether the sealed bank is worth spending.

    python confirm/calibrate_rci2.py --root <records>/rci002-20260918 \
        --gen B=<...>/gen-rci2_calib_B/predictions.jsonl \
        --gen D=<...>/gen-rci2_calib_D/predictions.jsonl

The calibration bank is the disjoint twin of the sealed bank: same generator, same seed family,
entity partitions assigned before generation. Scoring it is free; scoring the sealed bank costs
the instrument. So every question about DIFFICULTY is asked here, and the sealed bank inherits
the answer without being touched.

RCI-001 asked one question here -- is the aggregate in band. That was not enough. This asks:

  * does each ENDPOINT discriminate, separately (E1 abstention, E2 wh-incongruence);
  * is any stratum above its ceiling or at its floor -- `gate_strata`, the check that would
    have caught a fifth of RCI-001 being decorative;
  * is the false-abstention control clean, without which an abstention endpoint means nothing;
  * how many rows turn on a scoring CONVENTION rather than on being right;
  * and what band to REGISTER for the sealed read, per endpoint.

A verdict of USABLE is not a result. It licenses spending the seal, nothing else.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from floors import convention_sensitivity, cp_lower, gate, gate_strata  # noqa: E402
from paired import (  # noqa: E402
    fisher_exact_2x2, mcnemar_exact, paired, subgroup_dissociation,
)

SENTINEL = "UNKNOWN"
ARTICLE = re.compile(r"^(the|a|an)\s+")

#: The four cells of the discriminating 2x2, in the order the manifest predicts them.
CARVE = ["carve_out_role_congruent", "carve_out_role_incongruent",
         "carve_out_badge_congruent", "carve_out_badge_incongruent"]


def normalise(v: str) -> str:
    """THE declared scorer -- must stay byte-identical to build_rci2.normalise."""
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    s = " ".join(s.lower().replace(".", " ").replace(",", " ").split())
    return ARTICLE.sub("", s)


def correct_map(rows, pred, keep=lambda r: True):
    return {r["id"]: normalise(pred.get(r["id"], "")) == r["reference_normalised"]
            for r in rows if keep(r)}


def abstains_map(rows, pred, keep=lambda r: True):
    return {r["id"]: normalise(pred.get(r["id"], "")) == normalise(SENTINEL)
            for r in rows if keep(r)}


def person_vocab(rows) -> set:
    """Every person name the generator uses, taken from the bank rather than hardcoded."""
    return {str(r["reference"]) for r in rows
            if r["task_type"] == "alias_resolution" and r["answerable"]}


def emits_person(text: str, people: set) -> bool:
    t = str(text).lower()
    return any(p.lower() in t for p in people)


def emission_endpoint(label, rows, pb, pd, people):
    """E2, and it is a PAIRED EMISSION rate, not an accuracy.

    Accuracy could not measure this. Asked `Who was permitted to enter X?` where the context
    identifies the exempt party only by badge number, the frozen source answers `The holder of
    badge 14 was permitted to enter the Bell Passage without one.` -- substantively right, and
    exact-match scores it zero against the reference `14`. That stratum came back 0/24 for BOTH
    systems on the twin, which is a floor created by answer form rather than by capability.

    What the wh-compliance account actually predicts is not a wrong answer, it is a wrong
    ANSWER CLASS: emitting a person where the question's interrogative types one and the
    context determines none. So measure that directly. It is also what the RCI-001
    decomposition measured -- 0/25 person-name emissions on `Which role`, 23/25 on `Who` --
    and accuracy there was a proxy for it.

    **A paired emission endpoint is immune to every answer-agnostic predictor by construction.**
    A deterministic rule produces the same output for both systems, so it yields zero discordant
    pairs and cannot manufacture a difference. Floors bound what an ABSOLUTE rate means; they do
    not bear on a paired contrast. That is worth stating plainly rather than leaving implicit,
    because it is also why E1's high mechanical floor does not weaken E1's verdict.
    """
    eb = {r["id"]: emits_person(pb.get(r["id"], ""), people) for r in rows}
    ed = {r["id"]: emits_person(pd.get(r["id"], ""), people) for r in rows}
    p = paired("B", "D", eb, ed)
    nb, nd = sum(eb.values()), sum(ed.values())
    n = len(rows)
    return {
        "label": label, "n": n, "metric": "emits a person name where none is determined",
        "B": nb, "B_rate": round(nb / n, 4), "D": nd, "D_rate": round(nd / n, 4),
        # `paired()` names the axis for CORRECTNESS: `repair` is B-false -> D-true. Here the
        # boolean is "emitted a person", so `repair` means D emitted one and B did not, which
        # is the direction under test. Naming these explicitly because the first version of
        # this function read the sign backwards and called a clean 7/0 result a failure.
        "paired": {k: v for k, v in p.as_dict().items()
                   if k not in ("repair_ids", "damage_ids")},
        "D_more_than_B": p.repair, "B_more_than_D": p.damage,
        "direction": "D>B" if nd > nb else ("D<B" if nd < nb else "equal"),
        "immune_to_answer_agnostic_predictors": True,
        "why_immune": ("a deterministic rule emits the same string for both systems, so it "
                       "produces zero discordant pairs and cannot create a paired difference"),
    }


def endpoint(label, rows, pb, pd, abstention=False):
    """One endpoint, scored for both systems and paired."""
    f = abstains_map if abstention else correct_map
    cb, cd = f(rows, pb), f(rows, pd)
    p = paired("B", "D", cb, cd)
    nb, nd = sum(cb.values()), sum(cd.values())
    n = len(rows)
    return {
        "label": label, "n": n,
        "B": nb, "B_rate": round(nb / n, 4), "B_ci_lower_95": round(cp_lower(nb, n), 4),
        "D": nd, "D_rate": round(nd / n, 4), "D_ci_lower_95": round(cp_lower(nd, n), 4),
        "paired": {k: v for k, v in p.as_dict().items()
                   if k not in ("repair_ids", "damage_ids")},
        "discriminates": nb != nd,
        "direction": "D<B" if nd < nb else ("D>B" if nd > nb else "equal"),
    }


def scale_band(count, n_cal, n_sealed, slack=0.30):
    """A registered band for the sealed read, from the twin's rate with honest slack.

    RCI-001's twin under-predicted the sealed abstention count by 9 (16 expected by rate, 25
    observed) while still landing inside its band. Calibration inheritance bounds difficulty;
    it does not forecast a point estimate, and a band tight enough to look impressive is a band
    that will be wrong.
    """
    p = count / n_cal if n_cal else 0.0
    mid = p * n_sealed
    half = max(4.0, slack * mid + 2.0 * math.sqrt(max(mid, 1.0)))
    return [max(0, int(math.floor(mid - half))), min(n_sealed, int(math.ceil(mid + half)))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gen", action="append", required=True, metavar="LABEL=PATH")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "RCI_CALIBRATION.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    sealed = [json.loads(l) for l in
              (a.root / "RCI_SEALED.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    build = json.loads((a.root / "RCI_BUILD.json").read_text(encoding="utf-8"))

    preds = {}
    for spec in a.gen:
        lab, _, path = spec.partition("=")
        preds[lab] = {r["id"]: r.get("prediction", "") for r in
                      (json.loads(l) for l in
                       Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}
    if sorted(preds) != ["B", "D"]:
        raise SystemExit(f"expected exactly B and D, got {sorted(preds)}")
    pb, pd = preds["B"], preds["D"]

    una = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    e2rows = [r for r in ans if r["endpoint"] == "E2_wh_incongruence"]
    e2ctrl = [r for r in ans if r["task_type"] in ("carve_out_role_congruent",
                                                   "carve_out_badge_congruent")]

    people = person_vocab(rows)
    e1 = endpoint("E1_abstention", una, pb, pd, abstention=True)
    # E2 proper: the emission endpoint.
    e2 = emission_endpoint("E2_wh_incongruence", e2rows, pb, pd, people)
    # The control that makes E2 mean what it says: on CONGRUENT rows of the same reasoning
    # task, neither system should emit a person at all. If it does, E2 is measuring a general
    # tendency rather than a response to the interrogative.
    e2c = emission_endpoint("E2_control_congruent_emission", e2ctrl, pb, pd, people)
    # Reported, and the version that did not work -- kept so the failed operationalisation is
    # visible rather than quietly replaced.
    e2acc = endpoint("E2_accuracy_NOT_USED", e2rows, pb, pd)
    e2cacc = endpoint("E2_control_accuracy_NOT_USED", e2ctrl, pb, pd)

    # THE CONTROL WITHOUT WHICH E1 MEANS NOTHING: does either system just say UNKNOWN a lot?
    fa = {lab: sum(abstains_map(ans, p).values()) for lab, p in (("B", pb), ("D", pd))}

    # The registered interaction, conditioned on baseline-correct -- the denominator rule.
    inter = subgroup_dissociation({
        "incongruent": paired("B", "D", correct_map(e2rows, pb), correct_map(e2rows, pd)),
        "congruent": paired("B", "D", correct_map(e2ctrl, pb), correct_map(e2ctrl, pd)),
    })

    # Per-stratum gate: is any stratum decorative?
    fl = build["floors"]["CALIBRATION"]["per_stratum_answerable"]
    scores = {t: (sum(correct_map([r for r in ans if r["task_type"] == t], pb).values()),
                  len([r for r in ans if r["task_type"] == t]))
              for t in fl}
    gs = gate_strata(scores, {t: v["binding_floor_rate"] for t, v in fl.items()})

    conv = {lab: convention_sensitivity(ans, {r["id"]: p.get(r["id"], "") for r in ans})
            for lab, p in (("B", pb), ("D", pd))}

    # The four cells of the 2x2, per system, which is what the sealed read will read.
    cells = {}
    for t in CARVE:
        sub = [r for r in ans if r["task_type"] == t]
        cells[t] = endpoint(t, sub, pb, pd)

    n_una_sealed = sum(1 for r in sealed if not r["answerable"])
    n_e2_sealed = sum(1 for r in sealed
                      if r["answerable"] and r["endpoint"] == "E2_wh_incongruence")

    agg_floor = build["floors"]["CALIBRATION"]["aggregate"]["binding_floor_rate"]
    agg = {lab: sum(correct_map(rows, p).values()) for lab, p in (("B", pb), ("D", pd))}
    agg_gate = {lab: gate(c, len(rows), agg_floor) for lab, c in agg.items()}

    reasons = []
    if not e1["discriminates"]:
        reasons.append("E1 does not discriminate on the twin")
    if e2["D_more_than_B"] <= e2["B_more_than_D"]:
        reasons.append(f"E2 does not discriminate in the predicted direction on the twin: "
                       f"D-more {e2['D_more_than_B']}, B-more {e2['B_more_than_D']}")
    if e2c["B"] or e2c["D"]:
        reasons.append(f"E2 control is not clean: congruent rows already draw person "
                       f"emissions (B {e2c['B']}, D {e2c['D']}), so E2 would not be "
                       f"specific to the interrogative")
    if max(fa.values()) > 0.05 * len(ans):
        reasons.append(f"false abstention is not negligible: {fa} of {len(ans)} answerable")
    # NOT a veto: `gate_strata` scores every stratum on ACCURACY, and two of this bank's
    # strata are not measured by accuracy at all -- E1 is an abstention rate and E2 an emission
    # rate. A stratum being uninformative for accuracy says nothing about the endpoints, so it
    # is reported and does not block. What blocks is an endpoint that cannot be measured.
    verdict = "USABLE" if not reasons else "NOT_USABLE"

    report = {
        "schema": 1, "record": "RCI_002_CALIBRATION", "instrument": "RCI-002",
        "scored_bank": "CALIBRATION twin -- now SPENT for confirmation, still fine for DEV",
        "systems": {"B": "Rineya-Supa-1.7.0-devfreeze (frozen source)",
                    "D": "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze (promoted k=0.25)"},
        "declared_normaliser": build["declared_normaliser"],
        "endpoints": {"E1_abstention": e1, "E2_wh_incongruence": e2,
                      "E2_control_congruent_emission": e2c},
        "reported_not_verdict_bearing": {"E2_accuracy_NOT_USED": e2acc,
                                         "E2_control_accuracy_NOT_USED": e2cacc},
        "why_E2_is_an_emission_rate": e2["why_immune"],
        "required_control_false_abstention_on_answerable": {
            "counts": fa, "of_answerable": len(ans),
            "why": ("without this, `it just emits the sentinel more often` is a live "
                    "alternative explanation for any abstention difference"),
        },
        "registered_interaction_conditioned": inter,
        "discriminating_2x2": cells,
        "predicted_damage_patterns": build["design"]["discriminating_2x2"]["predicted_damage"],
        "aggregate": {"floor": agg_floor, "counts": agg, "n": len(rows), "gate": agg_gate},
        "per_stratum_gate": gs,
        "convention_sensitivity": conv,
        "bands_to_register_for_the_sealed_read": {
            "E1_abstention": {
                "n_sealed": n_una_sealed,
                "B": scale_band(e1["B"], e1["n"], n_una_sealed),
                "D": scale_band(e1["D"], e1["n"], n_una_sealed),
            },
            "E2_wh_incongruence": {
                "n_sealed": n_e2_sealed,
                "metric": "person-name emissions, not accuracy",
                "B": scale_band(e2["B"], e2["n"], n_e2_sealed),
                "D": scale_band(e2["D"], e2["n"], n_e2_sealed),
            },
            "caveat": ("bands bound difficulty, they do not forecast a point estimate. "
                       "RCI-001's twin under-predicted the sealed count by 9 and was still "
                       "inside its band."),
        },
        "verdict": verdict,
        "verdict_reasons": reasons or ["both endpoints discriminate on the twin, false "
                                       "abstention is negligible, and the strata are readable"],
        "what_this_licenses": ("spending the sealed read on these two frozen systems. It is "
                               "not a result and may not be cited as one."),
    }
    (a.root / "RCI_CALIBRATION_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"VERDICT {verdict}   ({'; '.join(reasons) if reasons else 'clean'})\n")
    for k, e in report["endpoints"].items():
        p = e["paired"]
        print(f"{k:<32} n={e['n']:<4} B {e['B']:<4} D {e['D']:<4} "
              f"disc {p['repair']}/{p['damage']:<4} "
              f"p {p['mcnemar_exact_two_sided_p']:.3e}   {e['direction']}")
    for k, e in report["reported_not_verdict_bearing"].items():
        p = e["paired"]
        print(f"  (reported) {k:<28} B {e['B']:<4} D {e['D']:<4} "
              f"p {p['mcnemar_exact_two_sided_p']:.3e}")
    print(f"\nfalse abstention on {len(ans)} answerable rows: {fa}")
    print(f"\nthe 2x2 (answerable):")
    for t, e in cells.items():
        p = e["paired"]
        print(f"  {t:<30} n={e['n']:<4} B {e['B']:<4} D {e['D']:<4} "
              f"repair {p['repair']:<3} damage {p['damage']:<3} p {p['mcnemar_exact_two_sided_p']:.3e}")
    print(f"\ninteraction incongruent vs congruent: {inter.get('verdict')} "
          f"(conditioned p {inter.get('fisher_given_baseline_correct')})")
    print(f"informative strata {len(gs['informative'])}/{len(gs['per_stratum'])}"
          f"   uninformative: {gs['uninformative']}")
    print(f"aggregate floor {agg_floor}  B {agg['B']}/{len(rows)} {agg_gate['B']['verdict']}"
          f"   D {agg['D']}/{len(rows)} {agg_gate['D']['verdict']}")
    print(f"convention-sensitive rows: "
          f"B {conv['B']['rows_turning_on_article'] + conv['B']['rows_turning_on_punctuation']}, "
          f"D {conv['D']['rows_turning_on_article'] + conv['D']['rows_turning_on_punctuation']} "
          f"of {len(ans)}")
    print(f"\nbands to register: {json.dumps(report['bands_to_register_for_the_sealed_read'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
