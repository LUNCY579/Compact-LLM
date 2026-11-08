"""go_hold.decide is a FUNCTION of the artifacts: GO must be reachable, and every limb must be able to
move the decision to HOLD on its own (a decision procedure's state space, enumerated)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import go_hold as gh  # noqa: E402


def w(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def fixture(tmp: Path, **over):
    o = dict(g0_match=True, v2_reject=["round-trip failures: 1"], v2_flag=[], rule1pp="PASS", g1_log="16 passed in 30s",
             arm_states=("PASS", "PASS", "PASS"), a0=True, g4="PASS", vsafe="PASS", dcollapse="PASS",
             protected=["conversation", "instruction_writing"], lc_verdict="INCONCLUSIVE")
    o.update(over)
    R = tmp / "records"
    w(R / "G0_TEACHER_LINEAGE.json", {"recorded_hash_comparison": {"model.safetensors": {"match": o["g0_match"]},
                                                                   "config.json": {"match": True}}})
    w(R / "vocab" / "VOCAB_CANDIDATES.json", {"candidates": {"V2": {"verdict": "REJECT" if o["v2_reject"] else "PASS",
                                                                    "REJECT": o["v2_reject"], "FLAG": o["v2_flag"]}}})
    w(R / "vocab" / "VOCAB_RULE1_CORRECTIONS.json", {"candidates": {"V2_65536": {"rule1_double_prime": o["rule1pp"]}}})
    (R / "vocab" / "G1_TOKENIZER_TESTS_20260925T000000Z.log").write_text(o["g1_log"], encoding="utf-8")
    A = R / "arms"
    for a, st in zip(("A1", "A2", "A3"), o["arm_states"]):
        if st is not None:
            w(A / a / "ARM_GATES.json", {"G1_G2_G3": st})
    if o["a0"]:
        w(A / "A0_THREE_COUNTS_1.json", {"ok": True})
    rep = w(R / "layer_study" / "LAYER_REPORT.json", {
        "G4": {"limbs": {}, "G4_calibration_panels": o["g4"], "G4_full": "PENDING: no shim",
               "V_SAFE_A1_no_catastrophic": o["vsafe"], "D_COLLAPSE": o["dcollapse"],
               "D_COLLAPSE_categories": [], "recommended_student": "V2_65536/18L",
               "recovery_protected_categories": o["protected"]},
        "frontier": {"decision": "V2/18 stays the recommended first student"},
        "selected_layer_set": {"kept": list(range(18))}})
    lc = w(R / "lc_arms" / "LC_ARMS.json", {"H_LC": {"verdict": o["lc_verdict"], "relative_reduction_r": 0.1,
                                                     "harm_detectability": "YES"},
                                            "follow_up_choice_by_SELECT2": "selected"})
    outside = dict(s1s2_plan=w(tmp / "plan.json", {"protected_skill_categories": ["conversation", "instruction_writing", "x"]}),
                   g5=w(tmp / "g5.json", {"G5": "PASS"}), training_lane=w(tmp / "lane.json", {"qualified": True}),
                   teacher_decision="Supa 1.7.0", grant=w(tmp / "grant.json", {"excludes": ["data_change"]}),
                   g4_instruments=w(tmp / "inst.json", {"measured_A1_A2_A3": True, "qualified_instruments": ["RTG-1 dev"]}))
    return R, rep, A, lc, outside


def test_go_is_reachable_when_every_limb_passes(tmp_path):
    R, rep, A, lc, out = fixture(tmp_path)
    d = gh.decide(R, rep, A, lc, **out)
    assert d["decision"] == "GO", (d["failing"], d["missing"])


def test_default_this_run_is_hold_with_the_outside_limbs_named(tmp_path):
    R, rep, A, lc, _ = fixture(tmp_path)
    d = gh.decide(R, rep, A, lc)
    assert d["decision"] == "HOLD" and not d["failing"]
    assert set(d["missing"]) == {"G4_full", "G5_recovery_readiness", "qualified_training_lane", "operator_teacher_decision",
                                 "grant_allows_architectural_change", "protected_categories_in_plan"}


def test_every_in_run_limb_can_fail_on_its_own(tmp_path):
    cases = {"G0_lineage": dict(g0_match=False), "G1_vocabulary": dict(rule1pp="FAIL"),
             "G2_G3_surgery": dict(arm_states=("PASS", "FAIL", "PASS")),
             "G4_calibration_panels": dict(g4="FAIL"), "V_SAFE": dict(vsafe="HOLD"), "D_COLLAPSE": dict(dcollapse="HOLD")}
    for i, (name, over) in enumerate(cases.items()):
        R, rep, A, lc, out = fixture(tmp_path / str(i), **over)
        d = gh.decide(R, rep, A, lc, **out)
        assert d["decision"] == "HOLD" and d["failing"] == [name], (name, d["failing"], d["missing"])


def test_g1_fails_on_another_reject_reason_a_flag_or_failed_tests(tmp_path):
    for i, over in enumerate((dict(v2_reject=["round-trip failures: 1", "special-token violations: 2"]),
                              dict(v2_flag=["overall token ratio 1.2 > 1.10"]), dict(g1_log="1 failed, 15 passed"))):
        R, rep, A, lc, out = fixture(tmp_path / str(i), **over)
        assert gh.decide(R, rep, A, lc, **out)["failing"] == ["G1_vocabulary"]


def test_plan_must_cover_the_fixed_protected_list(tmp_path):
    R, rep, A, lc, out = fixture(tmp_path, protected=["conversation", "long_context"])
    d = gh.decide(R, rep, A, lc, **out)
    assert d["failing"] == ["protected_categories_in_plan"]


def test_missing_arms_and_a_non_supa_teacher_add_named_missing_limbs(tmp_path):
    R, rep, A, lc, out = fixture(tmp_path, arm_states=("PASS", None, "PASS"), a0=False)
    out["teacher_decision"] = "Titan-Experimental"
    d = gh.decide(R, rep, A, lc, **out)
    assert "G2_G3_surgery" in d["missing"] and "layer_study_rerun_for_other_teacher" in d["missing"]


def test_lc_follow_up_only_when_supported(tmp_path):
    R, rep, A, lc, out = fixture(tmp_path, lc_verdict="SUPPORTED")
    assert "STARTING POINT" in gh.decide(R, rep, A, lc, **out)["recommendation"]["H_LC_follow_up"]
    R2, rep2, A2, lc2, out2 = fixture(tmp_path / "b", lc_verdict="REAL_BUT_SMALL")
    assert "no LC follow-up" in gh.decide(R2, rep2, A2, lc2, **out2)["recommendation"]["H_LC_follow_up"]
