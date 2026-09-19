"""Score RCI-002's single authorised confirmation read, exactly as registered.

    python confirm/read_rci2.py --root <records>/rci002-20260918 \
        --gen B=<...>/gen-rci2_confirm_B/predictions.jsonl \
        --gen D=<...>/gen-rci2_confirm_D/predictions.jsonl \
        --out <...>/RCI2_READ_001.json

**Written and committed BEFORE the budget is claimed.** The analysis cannot be shaped by the
result if it exists before the result does, and that ordering is the only thing separating a
pre-registered test from a story told about numbers.

Everything here comes from `RCI002_CONFIRM_PREREG.md` and nothing is added:

  E2  emits a person name on incongruent answerable rows, where the context determines none.
      Direction D > B. Exact two-sided McNemar on discordant pairs, confirmation at p <= 0.05
      AND the registered direction. ONE endpoint, so no multiplicity correction -- Holm over a
      family of one is the identity and applying it would be a gesture at rigour.

  E1  correct literal UNKNOWN. COMPUTED AND REPORTED, never verdict-bearing. Calibration showed
      the abstention task here is cue-driven: across four constructions of the same unanswerable
      item the frozen source scored 0.153, 0.547, 0.234 and 0.141 as each surface cue was
      removed. The lexically-matched construction that shipped leaves both systems near 0.14
      and one discordant pair on the twin, which `gate()` calls INCONCLUSIVE_UNDERPOWERED.
      Registering a criterion it cannot meet would be theatre.

Secondaries, reported and barred from the verdict: the congruent control for E2, false
abstention on answerable rows, the four cells of the 2x2, distractor density on E1, per-stratum
accuracy, convention sensitivity, power compression, and the registered bands.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from floors import convention_sensitivity, cp_lower, gate_strata  # noqa: E402
from paired import holm_bonferroni, paired, subgroup_dissociation  # noqa: E402

SENTINEL = "UNKNOWN"
ARTICLE = re.compile(r"^(the|a|an)\s+")
CARVE = ["carve_out_role_congruent", "carve_out_role_incongruent",
         "carve_out_badge_congruent", "carve_out_badge_incongruent"]

#: The registered CONFIRMATION endpoint. E1 is computed and reported but is not verdict-bearing:
#: calibration showed its abstention task is cue-driven, and with the cue matched away both
#: systems sit near 14% and differ by one row. See the prereg for the four-construction ladder.
DIRECTION = {"E2_wh_incongruence": "D>B"}
REPORTED_ONLY = {"E1_abstention": "D<B"}


def normalise(v):
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    s = " ".join(s.lower().replace(".", " ").replace(",", " ").split())
    return ARTICLE.sub("", s)


def person_vocab(rows):
    return {str(r["reference"]) for r in rows
            if r["task_type"] == "alias_resolution" and r["answerable"]}


def emits(text, people):
    t = str(text).lower()
    return any(p.lower() in t for p in people)


def score(rows, pb, pd, mode, people=None):
    if mode == "abstain":
        f = lambda p, r: normalise(p) == normalise(SENTINEL)          # noqa: E731
    elif mode == "emit":
        f = lambda p, r: emits(p, people)                             # noqa: E731
    else:
        f = lambda p, r: normalise(p) == r["reference_normalised"]    # noqa: E731
    cb = {r["id"]: f(pb.get(r["id"], ""), r) for r in rows}
    cd = {r["id"]: f(pd.get(r["id"], ""), r) for r in rows}
    p = paired("B", "D", cb, cd)
    nb, nd = sum(cb.values()), sum(cd.values())
    n = len(rows)
    d = p.as_dict()
    return {
        "n": n, "B": nb, "D": nd,
        "B_rate": round(nb / n, 4) if n else None,
        "D_rate": round(nd / n, 4) if n else None,
        "B_ci_lower_95": round(cp_lower(nb, n), 4) if n else None,
        "D_ci_lower_95": round(cp_lower(nd, n), 4) if n else None,
        # Named for the metric rather than for correctness, because two of the three modes are
        # not correctness and calling a person-emission a "repair" would invert the reading.
        "D_higher_than_B_on": p.repair, "B_higher_than_D_on": p.damage,
        "discordant": p.repair + p.damage,
        "p_mcnemar_exact": d["mcnemar_exact_two_sided_p"],
        "observed_direction": "D<B" if nd < nb else ("D>B" if nd > nb else "equal"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gen", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "RCI_SEALED.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    build = json.loads((a.root / "RCI_BUILD.json").read_text(encoding="utf-8"))
    seal = json.loads((a.root / "RCI_SEAL.json").read_text(encoding="utf-8"))
    if seal["read_budget"]["reads_used"] < 1:
        raise SystemExit("REFUSING: the read budget was never claimed. This script must run "
                         "AFTER seal_rci.py --claim-read, not before.")

    preds = {}
    for spec in a.gen:
        lab, _, path = spec.partition("=")
        preds[lab] = {r["id"]: r.get("prediction", "") for r in
                      (json.loads(l) for l in
                       Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}
    if sorted(preds) != ["B", "D"]:
        raise SystemExit(f"expected exactly B and D, got {sorted(preds)}")
    pb, pd = preds["B"], preds["D"]
    missing = [r["id"] for r in rows if r["id"] not in pb or r["id"] not in pd]
    if missing:
        raise SystemExit(f"{len(missing)} sealed rows have no prediction, e.g. {missing[:3]}")

    people = person_vocab(rows)
    una = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    inc = [r for r in ans if r["wh_congruence"] == "incongruent"]
    con = [r for r in ans if r["task_type"] in ("carve_out_role_congruent",
                                                "carve_out_badge_congruent")]

    e1 = score(una, pb, pd, "abstain")
    e2 = score(inc, pb, pd, "emit", people)
    primaries = {"E2_wh_incongruence": e2}
    # One endpoint, so no multiplicity correction. Holm over a family of one is the identity
    # and applying it would be a gesture at rigour rather than rigour.
    adj = {k: {"p_adjusted": v["p_mcnemar_exact"]} for k, v in primaries.items()}

    verdicts = {}
    for k, v in primaries.items():
        p_adj = adj[k]["p_adjusted"] if isinstance(adj[k], dict) else adj[k]
        ok_dir = v["observed_direction"] == DIRECTION[k]
        verdicts[k] = {
            "registered_direction": DIRECTION[k],
            "observed_direction": v["observed_direction"],
            "p_raw": v["p_mcnemar_exact"], "p_holm": p_adj,
            "direction_as_registered": ok_dir,
            "verdict": "CONFIRMED" if (ok_dir and p_adj <= 0.05) else "NOT CONFIRMED",
        }

    bands = json.loads((a.root / "RCI_CALIBRATION_REPORT.json").read_text(
        encoding="utf-8"))["bands_to_register_for_the_sealed_read"]

    out = {
        "schema": 1, "record": "RCI_002_READ_001", "instrument": "RCI-002",
        "sealed_sha256": seal["sealed_sha256"],
        "reads_used": seal["read_budget"]["reads_used"],
        "systems": seal.get("calibration_inherited", {}).get("systems"),
        "primary_endpoint": primaries,
        "reported_not_verdict_bearing_E1": {
            "endpoint": e1,
            "why_demoted": ("abstention on this family is cue-driven. Across four constructions "
                            "of the same unanswerable task the frozen source scored 0.153, "
                            "0.547, 0.234 and 0.141 as each surface cue was removed; the "
                            "lexically-matched construction shipped here leaves both systems "
                            "near 0.14 with one discordant pair on the twin."),
        },
        "verdicts": verdicts,
        "multiplicity": "none applied -- one confirmation endpoint",
        "band_check": {
            k: {"registered": bands[k], "observed": {"B": primaries[k]["B"],
                                                     "D": primaries[k]["D"]},
                "B_in_band": bands[k]["B"][0] <= primaries[k]["B"] <= bands[k]["B"][1],
                "D_in_band": bands[k]["D"][0] <= primaries[k]["D"] <= bands[k]["D"][1]}
            for k in primaries},
        "secondaries_barred_from_the_verdict": {
            "E2_control_congruent_emission": score(con, pb, pd, "emit", people),
            "false_abstention_on_answerable": {
                lab: sum(normalise(p.get(r["id"], "")) == normalise(SENTINEL) for r in ans)
                for lab, p in (("B", pb), ("D", pd))},
            "the_2x2_emission": {t: score([r for r in ans if r["task_type"] == t],
                                          pb, pd, "emit", people) for t in CARVE},
            "the_2x2_accuracy": {t: score([r for r in ans if r["task_type"] == t],
                                          pb, pd, "accuracy") for t in CARVE},
            "E1_by_distractor_density": {
                d: score([r for r in una if r["distractor_density"] == d], pb, pd, "abstain")
                for d in ("low", "high")},
            "per_stratum_accuracy": {t: score([r for r in ans if r["task_type"] == t],
                                              pb, pd, "accuracy")
                                     for t in sorted({r["task_type"] for r in ans})},
            "convention_sensitivity": {
                lab: convention_sensitivity(ans, {r["id"]: p.get(r["id"], "") for r in ans})
                for lab, p in (("B", pb), ("D", pd))},
        },
        "registered_interaction_conditioned": subgroup_dissociation({
            "incongruent": paired("B", "D",
                                  {r["id"]: emits(pb.get(r["id"], ""), people) for r in inc},
                                  {r["id"]: emits(pd.get(r["id"], ""), people) for r in inc}),
            "congruent": paired("B", "D",
                                {r["id"]: emits(pb.get(r["id"], ""), people) for r in con},
                                {r["id"]: emits(pd.get(r["id"], ""), people) for r in con}),
        }),
        "standing_prohibitions": [
            "E1 may not be read as a competence measurement: its mechanical floor is 1.000 "
            "(trigger_absence_abstainer). The PAIRED contrast is unaffected; no absolute claim "
            "may be made from it.",
            "No arm other than B and D was read. Reading one would have been selection.",
            "One training draw of one delta. REP-001 measured single-draw spreads of 7 and 13 "
            "rows on other endpoints.",
        ],
    }

    # Per-stratum power compression, using the candidate scores the gate can now take.
    fl = build["floors"]["SEALED"]["per_stratum_answerable"]
    acc = out["secondaries_barred_from_the_verdict"]["per_stratum_accuracy"]
    out["secondaries_barred_from_the_verdict"]["power_compression"] = gate_strata(
        {t: (acc[t]["B"], acc[t]["n"]) for t in acc},
        {t: fl[t]["binding_floor_rate"] for t in acc if t in fl},
        candidate_scores={t: (acc[t]["D"], acc[t]["n"]) for t in acc})["power_compression"]

    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("PRIMARY ENDPOINTS\n")
    for k, v in primaries.items():
        w = verdicts[k]
        print(f"{k:<22} n={v['n']:<4} B {v['B']:<4} D {v['D']:<4} "
              f"D-higher {v['D_higher_than_B_on']:<3} B-higher {v['B_higher_than_D_on']:<3} "
              f"p {v['p_mcnemar_exact']:.3e} holm {w['p_holm']:.3e}")
        print(f"{'':<22} registered {w['registered_direction']}, observed "
              f"{w['observed_direction']}  ->  **{w['verdict']}**")
        b = out["band_check"][k]
        print(f"{'':<22} band B {b['registered']['B']} obs {b['observed']['B']} "
              f"{'IN' if b['B_in_band'] else 'OUT'}; "
              f"D {b['registered']['D']} obs {b['observed']['D']} "
              f"{'IN' if b['D_in_band'] else 'OUT'}\n")
    sec = out["secondaries_barred_from_the_verdict"]
    c = sec["E2_control_congruent_emission"]
    print(f"E2 control (congruent): n={c['n']} B {c['B']} D {c['D']} "
          f"discordant {c['discordant']}  <- must be ~0 for E2 to mean what it says")
    print(f"false abstention on {len(ans)} answerable rows: {sec['false_abstention_on_answerable']}")
    print("\nthe 2x2, person emission:")
    for t, v in sec["the_2x2_emission"].items():
        print(f"  {t:<32} n={v['n']:<4} B {v['B']:<4} D {v['D']:<4} p {v['p_mcnemar_exact']:.3e}")
    print("\nE1 by distractor density:")
    for d, v in sec["E1_by_distractor_density"].items():
        print(f"  {d:<6} n={v['n']:<4} B {v['B']:<4} D {v['D']:<4} p {v['p_mcnemar_exact']:.3e}")
    if sec["power_compression"]:
        print(f"\npower compression flagged: {sorted(sec['power_compression'])}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
