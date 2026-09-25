"""SMALLVD-001 deliverable 17 -- the GO / HOLD decision for recovery, computed from the artifacts.

Written BEFORE the layer study finished, so the decision is a function of the files, not of a reading
of the numbers. Implements SMALLVD001_PREREG.md §10 as amended (A1.3 V-SAFE, A3 D-COLLAPSE and the
protected-category superset, A1.5 teacher specificity, A6 rule 1''). Every limb is PASS, FAIL or
MISSING, with the evidence path it was read from. GO requires every limb PASS; anything else is HOLD,
with the missing and failing limbs NAMED.

Limbs outside this run's reach (G5, a qualified training lane for the student geometry, the S1/S2
plan, the operator's teacher decision, an operator grant without architectural_change) are MISSING
unless evidence is supplied on the command line -- never assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p is not None and p.exists() else None


def limb(state: str, evidence, detail: str) -> dict:
    assert state in ("PASS", "FAIL", "MISSING")
    return {"state": state, "evidence": None if evidence is None else str(evidence), "detail": detail}


def decide(records: Path, layer_report: Path, arms: Path, lc: Path | None, s1s2_plan: Path | None = None,
           g5: Path | None = None, training_lane: Path | None = None, teacher_decision: str | None = None,
           grant: Path | None = None, g4_instruments: Path | None = None) -> dict:
    L = {}
    # G0 -- identity
    g0p = records / "G0_TEACHER_LINEAGE.json"
    g0 = _load(g0p)
    if g0 is None:
        L["G0_lineage"] = limb("MISSING", g0p, "no lineage record")
    else:
        bad = [k for k, v in g0["recorded_hash_comparison"].items() if not v["match"]]
        L["G0_lineage"] = limb("FAIL" if bad else "PASS", g0p,
                               f"hash mismatches: {bad}" if bad else f"all {len(g0['recorded_hash_comparison'])} recorded hashes match")
    # G1 -- vocabulary surgery (A6: rule 1''; the registered-rule REJECT stays on the record)
    vc, rc = _load(records / "vocab" / "VOCAB_CANDIDATES.json"), _load(records / "vocab" / "VOCAB_RULE1_CORRECTIONS.json")
    logs = sorted((records / "vocab").glob("G1_TOKENIZER_TESTS_*.log")) if (records / "vocab").exists() else []
    if vc is None or rc is None or not logs:
        L["G1_vocabulary"] = limb("MISSING", records / "vocab", "candidates, rule-1 corrections or G1 log absent")
    else:
        v2 = vc["candidates"]["V2"]
        only_rule1 = all(x.startswith("round-trip failures") for x in v2["REJECT"])
        rule1pp = rc["candidates"]["V2_65536"]["rule1_double_prime"] == "PASS"
        tail = logs[-1].read_text(encoding="utf-8", errors="replace")
        tests_ok = " passed" in tail and " failed" not in tail
        ok = only_rule1 and rule1pp and tests_ok and not v2["FLAG"]
        L["G1_vocabulary"] = limb("PASS" if ok else "FAIL", records / "vocab",
                                  f"registered verdict {v2['verdict']} (REJECT stays on record), rule 1'' "
                                  f"{'PASS' if rule1pp else 'FAIL'}, other REJECT reasons "
                                  f"{'none' if only_rule1 else v2['REJECT']}, G1 tests {'pass' if tests_ok else 'NOT pass'}")
    # G2 / G3 -- surgery arms
    gates = {a: _load(arms / a / "ARM_GATES.json") for a in ("A1", "A2", "A3")}
    a0 = sorted(arms.glob("A0_THREE_COUNTS_*.json")) if arms.exists() else []
    if any(g is None for g in gates.values()) or not a0:
        L["G2_G3_surgery"] = limb("MISSING", arms, f"missing: {[a for a, g in gates.items() if g is None] + ([] if a0 else ['A0 counts'])}")
    else:
        failing = [a for a, g in gates.items() if g["G1_G2_G3"] != "PASS"]
        L["G2_G3_surgery"] = limb("FAIL" if failing else "PASS", arms,
                                  f"failing arms {failing}" if failing else "A1 A2 A3 PASS; A0 header = analytic = torch = by-storage")
    # G4, V-SAFE, D-COLLAPSE, protected list -- the layer report
    rep = _load(layer_report)
    g4 = rep.get("G4") if rep else None
    if not g4 or "limbs" not in g4:
        for k in ("G4_calibration_panels", "G4_full", "V_SAFE", "D_COLLAPSE", "protected_categories_in_plan"):
            L[k] = limb("MISSING", layer_report, "layer report or its G4 block absent")
        protected = None
    else:
        L["G4_calibration_panels"] = limb("PASS" if g4["G4_calibration_panels"] == "PASS" else "FAIL", layer_report,
                                          "A1/A2/A3 measured separately on REPORT")
        # full G4 also needs >= 1 instrument whose shim replay of A0 is exact (A1.2) -- reachable
        # only with that evidence; never PASS merely because no instrument qualified
        inst = _load(g4_instruments)
        ok_inst = bool(inst) and inst.get("measured_A1_A2_A3") is True and len(inst.get("qualified_instruments", [])) >= 1
        L["G4_full"] = limb("PASS" if ok_inst else "MISSING", g4_instruments or layer_report,
                            f"instruments {inst.get('qualified_instruments')}" if ok_inst
                            else g4.get("G4_full", "instrument shim not qualified"))
        L["V_SAFE"] = limb("PASS" if g4["V_SAFE_A1_no_catastrophic"] == "PASS" else "FAIL", layer_report,
                           "A1 has no REPORT-CATASTROPHIC category")
        L["D_COLLAPSE"] = limb("PASS" if g4["D_COLLAPSE"] == "PASS" else "FAIL", layer_report,
                               f"recommended student {g4['recommended_student']}; collapsed {g4.get('D_COLLAPSE_categories')}")
        protected = g4["recovery_protected_categories"]
        plan = _load(s1s2_plan)
        if plan is None:
            L["protected_categories_in_plan"] = limb("MISSING", s1s2_plan, f"no S1/S2 plan; it must cover {protected}")
        else:
            missing = sorted(set(protected) - set(plan.get("protected_skill_categories", [])))
            L["protected_categories_in_plan"] = limb("FAIL" if missing else "PASS", s1s2_plan,
                                                     f"not covered: {missing}" if missing else "superset of the fixed list")
    # outside this run
    L["G5_recovery_readiness"] = limb("PASS" if _load(g5) and _load(g5).get("G5") == "PASS" else "MISSING", g5,
                                      "dataset / tokenization / training smoke on the student geometry")
    L["qualified_training_lane"] = limb("PASS" if _load(training_lane) and _load(training_lane).get("qualified") is True
                                        else "MISSING", training_lane,
                                        "native lane hard-requires 28 blocks / 151,936 rows / 596,049,920 params")
    L["operator_teacher_decision"] = limb("PASS" if teacher_decision else "MISSING", None,
                                          teacher_decision or "the operator has not named a distillation teacher")
    g = _load(grant)
    L["grant_allows_architectural_change"] = limb(
        "PASS" if g and "architectural_change" not in g.get("excludes", ["architectural_change"]) else "MISSING", grant,
        "95 of 100 grants exclude architectural_change")
    if teacher_decision and "supa" not in teacher_decision.lower():
        L["layer_study_rerun_for_other_teacher"] = limb("MISSING", None,
                                                        f"A1.5: §4 must be re-run on {teacher_decision} before Phase 11")
    # recommended first student (deliverable 16), and the exploratory H-LC reading
    rec = None
    if rep and rep.get("frontier"):
        rec = {"frontier_decision": rep["frontier"]["decision"],
               "recommended_student": g4.get("recommended_student") if g4 and "limbs" in g4 else None,
               "selected_layer_set": rep["selected_layer_set"]["kept"]}
    lc_rec = _load(lc)
    if lc_rec:
        h = lc_rec["H_LC"]
        rec = {**(rec or {}), "H_LC_verdict": h["verdict"], "H_LC_r": h["relative_reduction_r"],
               "H_LC_harm_detectability": h.get("harm_detectability"),
               "H_LC_follow_up": (f"name the LC variant ({lc_rec['follow_up_choice_by_SELECT2']}) as the candidate "
                                  "STARTING POINT for recovery -- its own registration required")
               if h["verdict"] == "SUPPORTED" else "no LC follow-up (A2.7 applies only if SUPPORTED)"}
    bad = {k: v for k, v in L.items() if v["state"] != "PASS"}
    return {"schema": 1, "record": "SMALLVD_GO_HOLD", "experiment": "SMALLVD-001",
            "decision": "GO" if not bad else "HOLD",
            "failing": sorted(k for k, v in bad.items() if v["state"] == "FAIL"),
            "missing": sorted(k for k, v in bad.items() if v["state"] == "MISSING"),
            "limbs": L, "recommendation": rec, "quality_claim": False, "release_permitted": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--layer-report", type=Path)
    ap.add_argument("--arms", type=Path)
    ap.add_argument("--lc", type=Path)
    ap.add_argument("--s1s2-plan", type=Path)
    ap.add_argument("--g5", type=Path)
    ap.add_argument("--training-lane", type=Path)
    ap.add_argument("--teacher-decision")
    ap.add_argument("--grant", type=Path)
    ap.add_argument("--g4-instruments", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    r = decide(a.records, a.layer_report or a.records / "layer_study" / "LAYER_REPORT.json",
               a.arms or a.records / "arms", a.lc or a.records / "lc_arms" / "LC_ARMS.json",
               a.s1s2_plan, a.g5, a.training_lane, a.teacher_decision, a.grant, a.g4_instruments)
    with open(a.out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(r, f, indent=1)
    print(r["decision"], "| failing:", r["failing"], "| missing:", r["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
