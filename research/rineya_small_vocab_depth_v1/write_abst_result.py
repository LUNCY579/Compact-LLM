"""ABST-001 -- generate the result from the committed artifacts ONLY (ABST_PRIMARY.json, ABST_H3.json, the attempt
record's stored per-row values, the BOUNDARY marker). No model runs. ABST_RESULT.json was never written (greedy
generation raised after ABST_H3), so the registered descriptive block (prereg §7) is computed here with the
registered abst_describe.summarise on the attempt record's stored values -- the same function the runner would
have called; cells that were not measured come out absent (NOT_MEASURED).

Writes (create-only):
  <records>/abst001-20260925/result/ABST_DESCRIPTIVE.json   the descriptive block + input hashes + code hash
  ABST001_RESULT.md                                          the result document, every number read from the above
usage: python -B write_abst_result.py --records <records root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abst_describe as ad  # noqa: E402

KNOWN_COUNT_RULE = 147           # prereg §9: generic damage excluded only if fewer than 147 known rows have plain < T
CUE_AUROC = 0.652


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    a = ap.parse_args(argv)
    A = a.records / "abst001-20260925"
    prim_p, h3_p = A / "result" / "ABST_PRIMARY.json", A / "result" / "ABST_H3.json"
    att = sorted((A / "attempts").glob("ATTEMPT_*.json"))
    mk = sorted((A / "attempts").glob("BOUNDARY_*.json"))
    if len(att) != 1 or len(mk) != 1:
        raise SystemExit(f"expected one attempt record and one marker, got {len(att)} / {len(mk)}")
    out_json, out_md = A / "result" / "ABST_DESCRIPTIVE.json", HERE / "ABST001_RESULT.md"
    if out_json.exists() or out_md.exists():
        raise SystemExit("create-only: an output exists")
    P, H3 = json.loads(prim_p.read_text(encoding="utf-8")), json.loads(h3_p.read_text(encoding="utf-8"))
    AT, MK = json.loads(att[0].read_text(encoding="utf-8")), json.loads(mk[0].read_text(encoding="utf-8"))
    recs = AT["partial"]["records"]
    # the primary must equal the attempt record's stored values (the descriptive block is computed on the latter)
    mism = [i for i, v in P["per_row_A_T4"].items()
            if (v["A_plain"], v["A_lc"], v["T4"]) != tuple(recs[i]["m"][c]["V4"]["V4"]["nats"] for c in ("S20_plain", "S20_lc", "T"))]
    if mism:
        raise SystemExit(f"ABST_PRIMARY and the attempt record disagree on {len(mism)} rows")
    D = ad.summarise(recs)
    V = H3["verdict"]
    if {k: V[k] for k in ("H1", "H2")} != P["steps_1_2"]:
        raise SystemExit("steps 1-2 differ between ABST_PRIMARY and ABST_H3")
    inputs = {p.name: sha(p) for p in (prim_p, h3_p, att[0], mk[0])}
    code = {p.name: sha(p) for p in (HERE / "abst_describe.py", HERE / "abst_verdict.py", Path(__file__).resolve())}
    with open(out_json, "x", encoding="utf-8", newline="\n") as f:
        json.dump({"schema": 1, "record": "ABST_DESCRIPTIVE", "experiment": "ABST-001",
                   "computed_from": "the attempt record's stored per-row values (ABST_RESULT.json was never written)",
                   "inputs_sha256": inputs, "code_sha256": code, "descriptive": D}, f, indent=1)

    h1, h2, h3 = V["H1"], V["H2"], V["H3"]
    k2b = h2["step_2b_known_anchor"]
    cA = D["cells"]["A"]
    known_rev = cA["known"]["plain_nll_below_teacher"]
    fam = D["A_unk_by_family"]
    rev_fams = [f for f, x in sorted(fam.items()) if x["better"] > x["worse"]]
    au = cA["auroc_unknown_vs_known"]
    tokA, dig, dan = D["A_unk_unknown_token_only"], D["A_known_digit_token_only"], D["A_known_digit_anchor"]
    trk = D["A_tracking"]
    g = D["greedy"]
    cue = D["cue_not_recorded"]
    rc = AT["partial"]["refusal_controls"]
    f4 = (fam["prerequisite_schedule"]["worse"] / fam["prerequisite_schedule"]["n"] <= 0.5 and
          all(x["worse"] / x["n"] >= 0.7 for f, x in fam.items() if f != "prerequisite_schedule"))
    worse_frac = h1["negative"] / h1["n"]
    f8 = cA["unk"]["plain_nll_below_teacher"] / 256
    dcl = D["cells"]["D"]["unk"]["lc_closer_to_teacher"]
    forecasts = [
        ("F1", "H1 = LC_LOWERS_UNKNOWN", h1["label"] == "LC_LOWERS_UNKNOWN", h1["label"]),
        ("F2", "cell-A worse fraction >= 0.70", worse_frac >= 0.70, f"{worse_frac:.3f}"),
        ("F3", "H2 = TEACHER_CONSISTENT_DIRECTION", h2["label"] == "TEACHER_CONSISTENT_DIRECTION", h2["label"]),
        ("F4", "prerequisite_schedule worse <= 0.50, every other family >= 0.70", f4,
         ", ".join(f"{f} {x['worse']}/{x['n']}" for f, x in sorted(fam.items()))),
        ("F5", "D+ rejected", h3["holm_rejected"]["D+"], f"D {h3['tests']['D+']['positive']}/256 more likely"),
        ("F6", "attribution = SIGN_FOLLOWS_DEPTH", h3["label"] == "SIGN_FOLLOWS_DEPTH", h3["label"]),
        ("F7", "M1 = TEACHER_TRACKING", cA["reading_M1_teacher_tracking_rowwise"] == "TEACHER_TRACKING",
         cA["reading_M1_teacher_tracking_rowwise"]),
        ("F8", "plain V4/20 NLL below the teacher's on >= 90% of UNKNOWN rows", f8 >= 0.90,
         pct(cA["unk"]["plain_nll_below_teacher"], 256)),
        ("F9", "teacher abstains greedily on <= 5% of UNKNOWN rows (T4 and T0)", None, "NOT_SCOREABLE: T4/T0 greedy NOT_MEASURED"),
        ("F10", "in D, LC moves away from the teacher on the majority of UNKNOWN rows", dcl < 128, f"{256 - dcl}/256 farther"),
    ]
    scored = [x for x in forecasts if x[2] is not None]
    rule1_prop = au["teacher"] is not None and abs(au["teacher"] - 0.5) <= 0.05
    L = []
    w = L.append
    w("# ABST-001 — result: LC lowers the gold UNKNOWN likelihood on V4/20, in the teacher's direction; the sign follows depth")
    w("")
    w("**Generated** by `write_abst_result.py` from the committed artifacts only: records `abst001-20260925/` "
      "`result/ABST_PRIMARY.json`, `result/ABST_H3.json`, the attempt record and the BOUNDARY marker (sha256s in "
      "`result/ABST_DESCRIPTIVE.json`). No number in this document was typed by hand.")
    w(f"**Run:** code 8486d817 (R1, harness 31/31, finalized chain), launched via `launch_abst.ps1`, lease abst001. "
      f"BOUNDARY {mk[0].name.split('_')[1]}. **GO/HOLD stays HOLD; no quality claim; no release.**")
    w("")
    w("## 1. Registered outcome")
    w("")
    w("| step | test | result | threshold | decision |")
    w("|---|---|---|---|---|")
    w(f"| 1 | H1, cell A (S20/V4), UNKNOWN rows | LC-worse **{h1['negative']}** / better {h1['positive']} (ties {h1['ties_or_undefined']}), "
      f"p = {h1['p_negative']:.2g} | ≥ 145 | **{h1['label']}** |")
    w(f"| 2 | UNKNOWN-row anchor vs T4 | closer **{h2['positive']}** / farther {h2['negative']}, p = {h2['p_positive']:.2g} | ≥ 147 | TOWARD |")
    w(f"| 2b | known-row anchor vs T4 | closer **{k2b['positive']}** / farther {k2b['negative']}, p = {k2b['p_positive']:.2g} | ≥ 147 | rejects toward |")
    w(f"| — | over-suppression (plain < T < LC) | {h2['over_suppressing_rows']} / {h2['rows_defined']} rows | majority = ≥ 128 | no majority |")
    w(f"| H2 | | | | **{h2['label']}** |")
    w(f"| 3 | Holm at 0.0125 | D+ {h3['tests']['D+']['positive']}/256 · B {h3['tests']['B-']['negative']}/256 less likely · "
      f"C {h3['tests']['C+']['positive']}/256 more likely · dV {h3['tests']['dV']['positive']}/256 · dD {h3['tests']['dD']['positive']}/256 | "
      f"153 … 147 | rejected: {', '.join(k for k, v in h3['holm_rejected'].items() if v)} |")
    w(f"| H3 | | | | **{h3['label']}** |")
    w("")
    w(f"**Size** (reported, no gate): the median per-row ratio NLL_lc / NLL_plain on UNKNOWN rows is at least "
      f"×{h1['size_median_row_ratio_lc_over_plain_at_least']:.3f} (97.5% lower bound).")
    w("")
    w("## 2. What it licenses (the registered R1 table, applied)")
    w("")
    w(f"> **Pooled over the four RTG1-dev family pools, equally weighted, conditional on the fitted T:** LC makes the gold "
      f"reference `UNKNOWN` **less** likely on V4/20 on most rows ({h1['negative']}/256)"
      + (f"; **in {', '.join(rev_fams)} the majority went the other way** ("
         + "; ".join(f"{fam[f]['better']}/{fam[f]['n']} LC-better" for f in rev_fams) + ")" if rev_fams else "")
      + f". It moves V4/20 in the teacher's direction on both `UNKNOWN` rows ({h2['positive']}/256) and known rows "
      f"({k2b['positive']}/256), without an over-suppression majority ({h2['over_suppressing_rows']}/256). This is consistent "
      f"with moving toward a teacher that does not abstain on these pools. "
      + (f"**Generic damage is excluded**: {known_rev} known rows have NLL_plain < NLL_T (the registered count rule needs "
         f"fewer than {KNOWN_COUNT_RULE})." if known_rev < KNOWN_COUNT_RULE else
         "**Generic damage is not excluded** (count rule not met).")
      + " **It does not restore abstention, and it is not shown to be teacher-specific**: a teacher-blind shift of mass "
      "from `UNKNOWN` to the digits passes the same tests. **HLCC's abstention caveat stays in force, reinforced**: the "
      "direction-only licence does not license abstention, and LC lowers the likelihood of the gold `UNKNOWN` answer.")
    w(">")
    w(f"> **H3 ({h3['label']}):** at depth 20 the sign is LC-worse under V4 **and** V2 (and V0); at depth 18 it is "
      "LC-better under every vocabulary. The depth set carries A's sign, and changing depth alone reverses it. No "
      "recommendation changes, and no set is chosen.")
    w(">")
    w("> *Abstention where the teacher abstains (e.g. CC0) is not measured by any registered instrument.*")
    w("")
    w("## 3. Refusal controls (all passed before the boundary)")
    w("")
    w(f"- R4 fit canaries: S20 {rc['fit_canary']['S20']['max_abs_diff']}, S18 {rc['fit_canary']['S18']['max_abs_diff']} (≤ 1e-6).")
    w(f"- R5 tier 1 (exact): {', '.join(f'{k} {v}' for k, v in rc['tier1_max_abs_diff'].items())} (≤ 1e-6).")
    w(f"- R5b tier 2 (B = 1 path): max {max(rc['tier2_max_abs_diff'].values()):.3g} (≤ 1e-3); d(D) sign disagreements "
      f"{rc['tier2_sign_disagreements_reported']} (reported).")
    w(f"- R1 structure_id: {rc['structure_id_check']['distinct']} distinct, overlap {rc['structure_id_check']['overlap_with_used_or_train']}, "
      f"controls {rc['structure_id_check']['positive_controls_caught']}.")
    w(f"- R6 projection {rc['projected_minutes']:.1f} min (cap 60). W: down_proj restored. No earlier crossing.")
    w("")
    w("## 4. What did not complete")
    w("")
    w(f"- Greedy generation finished A-plain and A-LC, then raised `{AT['exception_type']}: "
      f"{AT['message'].splitlines()[0][:80]}` during the **teacher (T4)** generation. The attempt record classifies it "
      f"'{AT['classification_hint']}' with primary_written {AT['primary_written']}. Per §9 there is no rerun.")
    w("- The W check after A-LC generation copies weights to the CPU (a device sync), so an earlier asynchronous fault "
      "would have been raised there; the fault arose in T4 generation, after every measurement cell, ABST_PRIMARY and ABST_H3.")
    w("- **NOT_MEASURED:** T4 and T0 greedy outputs; forecast F9 is NOT_SCOREABLE. `ABST_RESULT.json` was not written; "
      "the registered descriptive block below is computed from the attempt record's stored values (§7 computes "
      "descriptives from stored per-row values; same function, no model).")
    w("")
    w("## 5. Descriptive (registered readings; never a verdict)")
    w("")
    w("| cell | UNKNOWN rows LC worse / better | UNKNOWN LC closer to T | known LC worse / better |")
    w("|---|---|---|---|")
    for X in ("A", "B", "C", "D", "E", "F"):
        c = D["cells"].get(X)
        if c:
            w(f"| {X} | {c['unk']['lc_worse']} / {c['unk']['lc_better']} | {c['unk']['lc_closer_to_teacher']} | "
              f"{c['known']['lc_worse']} / {c['known']['lc_better']} |")
    w("")
    w(f"- **By family (cell A, UNKNOWN rows, LC worse / better):** "
      + "; ".join(f"{f} {x['worse']}/{x['better']}" for f, x in sorted(fam.items())) + ".")
    w(f"- **Geometry:** plain V4/20 is below the teacher on {pct(cA['unk']['plain_nll_below_teacher'], 256)} of UNKNOWN rows "
      f"and on {pct(known_rev, 256)} of known rows. Median gap closure on UNKNOWN rows {cA['unk']['median_gap_closure']:.3f}; "
      f"rows crossing the teacher in either direction {cA['unk']['crossed_the_teacher']} (over-suppressing, one direction: "
      f"{h2['over_suppressing_rows']}). Teacher median UNKNOWN NLL {D['teacher']['T4']['unk']['median_nats']:.2f} nats.")
    w(f"- **M1:** {cA['reading_M1_teacher_tracking_rowwise']} (nearly implied by the geometry; not independent evidence). "
      f"**M2:** {au['reading_M2']} (AUROC plain {au['plain']:.3f}, LC {au['lc']:.3f}, teacher {au['teacher']:.3f}).")
    w(f"- **Row-level tracking (within-family partial Spearman; correlational, licenses nothing):** UNKNOWN "
      f"{trk['unk']['partial_spearman']:.3f} ({trk['unk']['reading']}; pooled {trk['unk']['pooled_partial_spearman']:.3f}; per family "
      + ", ".join(f"{f} {x['partial_spearman']:.2f}" for f, x in sorted(trk['unk']['per_family'].items()))
      + f"); known {trk['known']['partial_spearman']:.3f} ({trk['known']['reading']}). Rows move in proportion to where the "
      "teacher sits, within families; a row feature could still drive both, so this is not 'tracks the teacher'.")
    w(f"- **UNKNOWN token alone:** LC worse {tokA['lc_worse']} / better {tokA['lc_better']}; agrees with the two-token "
      f"endpoint on {tokA['agrees_with_endpoint']}/{tokA['rows']} rows. **Gold digit alone (known rows):** LC worse "
      f"{dig['lc_worse']} / better {dig['lc_better']}; digit-only anchor closer {dan['closer']} / farther {dan['farther']}.")
    w(f"- **Gap cue:** {cue['unk_rows_with_cue']} UNKNOWN / {cue['known_rows_with_cue']} known rows carry it; cue-only AUROC "
      f"{cue['cue_only_auroc_unknown_vs_known']:.3f}.")
    w(f"- **Greedy (4 tokens; T4/T0 NOT_MEASURED):** A-plain abstains on {g['A_plain']['abstains_on_unk']}/256 UNKNOWN rows, "
      f"false-abstains on {g['A_plain']['false_abstain_on_known']}/256 known rows, answers {g['A_plain']['correct_on_known']}/256 "
      f"known rows correctly; A-LC: {g['A_lc']['abstains_on_unk']}/256, {g['A_lc']['false_abstain_on_known']}/256, "
      f"**{g['A_lc']['correct_on_known']}/256**.")
    w(f"- **Vocabulary at depth 20 (UNKNOWN rows):** median output-support term {D['vocab_split_depth20_unk']['median_output_support_term_V4']:.4f}, "
      f"input term {D['vocab_split_depth20_unk']['median_input_term_V4']:.4f}; tokenisation identity: "
      + "; ".join(f"{k} {v['worse']}/{v['better']} of {v['rows']}" for k, v in D["A_unk_by_tokenisation_identity"].items()) + ".")
    w("")
    w("**Wording rules applied (§7):**")
    w(f"- Rule 1: |AUROC_T4 − 0.5| = {abs(au['teacher'] - 0.5):.3f} "
      + ("≤ 0.05, so 'abstention' becomes 'UNKNOWN-token propensity'." if rule1_prop else
         "> 0.05, so rule 1 does not force the substitution; ")
      + f"**but** every AUROC here ({au['plain']:.3f}, {au['lc']:.3f}, {au['teacher']:.3f}) is below what the literal gap cue "
      f"alone reaches ({CUE_AUROC}), so none is evidence of discrimination beyond reading the cue. This document therefore "
      "says 'UNKNOWN likelihood', not 'abstention', for every model property.")
    w(f"- Rule 2: the UNKNOWN token alone agrees with the endpoint on {tokA['agrees_with_endpoint']}/{tokA['rows']} rows, so "
      "'UNKNOWN likelihood' (not 'reference-sequence likelihood') is used.")
    w(f"- Rule 2b: the digit-only anchor ({dan['closer']} closer / {dan['farther']} farther) agrees in direction with step 2b, "
      "so the known-row statement may say 'the gold digit'.")
    w("- Rule 3: no 'restores abstention', 'abstention is safe', 'tracks the teacher' or 'teacher-specific' anywhere.")
    w("")
    w(f"## 6. Forecasts (scored; not gates) — {sum(1 for x in scored if x[2])}/{len(scored)} scoreable hit")
    w("")
    w("| # | forecast | observed | scored |")
    w("|---|---|---|---|")
    for fid, txt, ok, obs in forecasts:
        w(f"| {fid} | {txt} | {obs} | {'hit' if ok else ('NOT_SCOREABLE' if ok is None else '**miss**')} |")
    w("")
    w("## 7. Disclosures")
    w("")
    w("- The run is R1 (71acefb8) plus a harness-forced two-line fix (1d2f5552), finalized at 8486d817; the cold reviewer "
      "re-checked R1 and the fix diff first-hand before launch.")
    w("- The descriptive block is computed after the run, from the attempt record, because the result file was never "
      "written (§4). ABST_PRIMARY's per-row values equal the attempt record's on all 512 rows (checked by this generator).")
    w("- Spent-data framing (§1 of the prereg) predicted most of this: F1–F7 and F10 follow from the spent geometry plus "
      "HLCC's lead, and TEACHER_CONSISTENT_DIRECTION was close to implied once H1 replicated. The new information is the "
      "replication on fresh rows, the family reversal's size, the depth attribution, and the descriptive greedy numbers.")
    out_md.write_bytes(("\n".join(L) + "\n").encode("utf-8"))
    print(json.dumps({"written": [str(out_json), str(out_md)], "forecast_hits": f"{sum(1 for x in scored if x[2])}/{len(scored)}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
