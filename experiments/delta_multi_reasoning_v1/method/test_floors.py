"""Fixtures for the instrument gate, built from the real defects it exists to catch.

Run:  python experiments/delta_multi_reasoning_v1/method/test_floors.py

Each case reconstructs an actual failure from this programme rather than an abstract one, so a
regression in the gate shows up as "it would no longer have caught RCI-001" instead of as an
assertion number.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floors import (  # noqa: E402
    convention_sensitivity, cp_lower, extraction, floors, gate, gate_strata, immunity,
    min_k_to_clear, require_pack_structure, structural,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import paired, subgroup_dissociation  # noqa: E402

FAIL: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAIL.append(name)


def test_rci001_recency_defect_is_caught():
    """THE ONE THAT MATTERS. RCI-001 published 0.2773 from four distributional predictors;
    a recency rule scored 0.3711. The gate must now catch that at build time."""
    print("RCI-001's recency defect is caught by the extraction class")
    rows = []
    # 40 items where the correct answer is the LAST candidate mentioned -- the real defect:
    # the correction is emitted after the thing it corrects.
    for i in range(40):
        rows.append({"id": f"t{i}", "stratum": "temporal", "reference": f"Day{i % 6}",
                     "context": f"An earlier page gives Day{(i + 1) % 6}. That is an error. "
                                f"The examination took place on Day{i % 6}.",
                     "question": "On which day?"})
    # 40 items an extractor cannot touch
    for i in range(40):
        rows.append({"id": f"a{i}", "stratum": "alias", "reference": f"Person{i % 8}",
                     "context": f"The steward that week was Person{i % 8}. "
                                f"Later, Person{(i + 3) % 8} filed the report.",
                     "question": "Who was the steward?"})
    f = floors(rows)
    check("extraction class is computed at all", "last_candidate_in_context" in f["extraction"])
    check("recency rule scores the confounded stratum",
          f["extraction"]["last_candidate_in_context"] >= 40,
          str(f["extraction"]["last_candidate_in_context"]))
    check("binding floor comes from the EXTRACTION class",
          f["binding_class"] == "extraction", f["binding_class"])
    d_only = max(f["distributional"].values())
    check("extraction floor exceeds anything distributional",
          f["binding_floor_count"] > d_only, f"{f['binding_floor_count']} vs {d_only}")


def test_abstention_stratum_is_immune():
    """RCI-001's confirmation survived because its primary endpoint had a zero shortcut floor.
    `immunity()` must find that deliberately rather than by luck."""
    print("an unanswerable stratum is immune to every extractor")
    rows = []
    for i in range(30):
        rows.append({"id": f"a{i}", "stratum": "lc", "reference": f"Seal{i}",
                     "context": f"The seal number is Seal{i}.", "question": "Seal number?"})
    for i in range(10):
        rows.append({"id": f"u{i}", "stratum": "lc", "reference": "UNKNOWN",
                     "context": "No seal number appears in this account.",
                     "question": "Seal number?", "unanswerable": True})
    im = immunity(rows, lambda r: r.get("unanswerable"), abstain="UNKNOWN")
    check("unanswerable subset found", im["n"] == 10)
    # The structural half: no extractor can emit the sentinel, so it scores zero here.
    check("EXTRACTION class scores zero on it", im["extraction_max"] == 0,
          str(im["extraction"]))
    check("flagged extraction-immune", im["extraction_immune"] is True)
    # The half endpoint choice does NOT buy: always-abstain wins an all-UNKNOWN subset.
    check("but it is NOT distributionally immune -- always_abstain takes it",
          im["distributional_immune"] is False and im["distributional"]["always_abstain"] == 10,
          str(im["distributional"]))
    check("so the function demands the false-abstention control",
          "false abstention" in im["required_control_if_not_distributionally_immune"])
    full = floors(rows, abstain="UNKNOWN")
    check("the WHOLE bank is not immune either", full["binding_floor_count"] > 0)


def test_ceiling_blocks_a_saturated_instrument():
    """DMRF-SCOUT-001's long-context stratum was 32/32 on both cores: cleared its floor and
    could not show a regression."""
    print("a saturated stratum is refused even though it clears the floor")
    g = gate(correct=32, n=32, floor_rate=0.25)
    check("32/32 is ABOVE_CEILING, not CLEARS", g["verdict"] == "ABOVE_CEILING", g["verdict"])
    check("reason names the confirmation problem", "regression" in g["why"])
    check("a mid-band score clears", gate(20, 32, 0.25)["verdict"] == "CLEARS")


def test_underpowered_is_not_failure():
    """CAL n=6: a stratum with a real 17% signal was reported FAILS."""
    print("underpowered is distinguished from below-floor")
    g = gate(correct=1, n=6, floor_rate=0.0526)
    check("1/6 above a 0.0526 floor is INCONCLUSIVE, not BELOW_FLOOR",
          g["verdict"] == "INCONCLUSIVE_UNDERPOWERED", g["verdict"])
    check("it reports how many it would need", g["min_correct_to_clear_at_this_n"] == 3,
          str(g["min_correct_to_clear_at_this_n"]))
    check("a genuine zero IS below floor", gate(0, 6, 0.0526)["verdict"] == "BELOW_FLOOR")
    check("min_k_to_clear matches at n=6", min_k_to_clear(6, 0.0526) == 3)


def test_always_abstain_floor():
    """Scoring unanswerable rows as their own stratum gives a floor of 1.000 -- the first
    version of DMRF-SCOUT-001 did exactly this."""
    print("an all-UNKNOWN stratum has a floor of 1.000")
    rows = [{"id": f"u{i}", "stratum": "u", "reference": "UNKNOWN",
             "context": "nothing here", "question": "?"} for i in range(16)]
    f = floors(rows, abstain="UNKNOWN")
    check("always_abstain is perfect", f["distributional"]["always_abstain"] == 16)
    check("binding floor is 1.0", f["binding_floor_rate"] == 1.0, str(f["binding_floor_rate"]))
    check("gate refuses it at any score", gate(16, 16, 1.0)["verdict"] == "ABOVE_CEILING")


def test_pack_structure_is_required():
    """The one safeguard RCI-001 had by habit rather than by design."""
    print("per-pack structure is demanded, not hoped for")
    good = [{"scenario_id": f"S{i}", "variant": v}
            for i in range(20) for v in ("base", "para", "counter", "unans")]
    bad = [{"scenario_id": f"S{i}", "variant": "base"} for i in range(80)]
    check("a four-variant bank satisfies it", require_pack_structure(good)["satisfied"])
    r = require_pack_structure(bad)
    check("a flat bank does not", r["satisfied"] is False)
    check("and it says what to do", "do not ship" in r["action_if_unsatisfied"])


def test_convention_sensitivity_finds_the_article_rows():
    """52 of RCI-001's 384 answerable rows turned on a leading definite article."""
    print("rows that turn on a convention are counted, not argued about")
    rows = [{"id": f"r{i}", "reference": "the standing inspector"} for i in range(10)]
    rows += [{"id": f"q{i}", "reference": "Yolanda Pais"} for i in range(10)]
    pred = {r["id"]: ("standing inspector" if r["id"].startswith("r") else "Yolanda Pais.")
            for r in rows}
    cs = convention_sensitivity(rows, pred)
    check("article-only misses are found", cs["rows_turning_on_article"] == 10,
          str(cs["rows_turning_on_article"]))
    check("trailing periods are found separately", cs["rows_turning_on_punctuation"] == 10,
          str(cs["rows_turning_on_punctuation"]))
    check("strict scorer sees none of them right", cs["strict"] == 0, str(cs["strict"]))


def test_per_stratum_gate_catches_the_decorative_stratum():
    """`temporal_correction` answerable: 78/78 for both systems AND 78/78 for the recency
    rule. A fifth of RCI-001 carried no information and the aggregate gate did not notice."""
    print("a stratum that is above ceiling and at its extraction floor is refused")
    g = gate_strata(
        {"temporal_correction": (78, 78), "alias_resolution": (65, 78),
         "count_with_exclusions": (30, 75)},
        {"temporal_correction": 1.000, "alias_resolution": 0.141,
         "count_with_exclusions": 0.000})
    check("the saturated stratum is not informative",
          "temporal_correction" in g["uninformative"], str(g["uninformative"]))
    check("the readable ones are", g["informative"] ==
          ["alias_resolution", "count_with_exclusions"], str(g["informative"]))
    check("share is reported", abs(g["informative_share"] - 0.6667) < 0.001,
          str(g["informative_share"]))


def test_dissociation_needs_the_right_denominator():
    """THE ONE THAT CHANGED A CONCLUSION. On RCI-001's unanswerable rows the binding group
    looked spared; conditioned on what the baseline got right it is damaged at the same rate."""
    print("a dissociation is tested against the baseline's correct set, not the row count")
    # binding, unanswerable: n=51, baseline correct 3, 1 lost.
    b_base = {f"b{i}": i < 3 for i in range(51)}
    b_cand = {f"b{i}": i < 2 for i in range(51)}
    # non-binding, unanswerable: n=77, baseline correct 37, 14 lost.
    n_base = {f"n{i}": i < 37 for i in range(77)}
    n_cand = {f"n{i}": i < 23 for i in range(77)}
    d = subgroup_dissociation({"binding": paired("B", "D", b_base, b_cand),
                               "non_binding": paired("B", "D", n_base, n_cand)})
    check("the naive test calls it a dissociation", d["fisher_vs_all_rows"] < 0.05,
          f"{d['fisher_vs_all_rows']:.2e}")
    check("the conditioned test does not", d["fisher_given_baseline_correct"] > 0.5,
          f"{d['fisher_given_baseline_correct']:.3f}")
    check("verdict names it an artifact", d["verdict"] == "ARTIFACT_OF_BASELINE", d["verdict"])
    check("loss rates are in fact near-identical",
          abs(d["groups"]["binding"]["damage_rate_given_baseline_correct"]
              - d["groups"]["non_binding"]["damage_rate_given_baseline_correct"]) < 0.05,
          str(d["groups"]))
    # The answerable half of the same read IS a real dissociation and must survive.
    a_base = {f"a{i}": i < 97 for i in range(153)}
    a_cand = {f"a{i}": 31 <= i < 97 for i in range(153)}
    c_base = {f"c{i}": i < 128 for i in range(231)}
    c_cand = {f"c{i}": 3 <= i < 128 for i in range(231)}
    d2 = subgroup_dissociation({"binding": paired("B", "D", a_base, a_cand),
                                "non_binding": paired("B", "D", c_base, c_cand)})
    check("the real one survives conditioning", d2["verdict"] == "DISSOCIATION", d2["verdict"])
    check("at p < 1e-8", d2["admissible_p"] < 1e-8, f"{d2['admissible_p']:.2e}")


def test_structural_class_catches_the_complement_leak():
    """THE ONE A RED TEAM FOUND. RCI-002's first draft listed three of four roles in a decoy
    sentence and always omitted the exempt one; set difference scored 1.000 while every
    published floor read 0.20-0.29 and saw nothing."""
    print("set-complement between two listings is caught")
    rows = []
    roles = ["night steward", "records clerk", "duty warden", "survey lead"]
    for i in range(24):
        ex = roles[i % 4]
        others = [r for r in roles if r != ex]
        rows.append({"id": f"c{i}", "stratum": "carve", "reference": ex,
                     "context": (f"Duty badges were issued as follows: the {roles[0]}; "
                                 f"the {roles[1]}; the {roles[2]}; the {roles[3]}. "
                                 f"One standing exception applied: the {ex} could enter. "
                                 f"Duty rosters list the {', the '.join(others)} on site."),
                     "question": "Which role was exempt?"})
    st = structural(rows)
    check("set_complement_singleton solves it outright",
          st["set_complement_singleton"] == 24, str(st["set_complement_singleton"]))
    f = floors(rows)
    check("and the binding floor now comes from the STRUCTURAL class",
          f["binding_class"] == "structural", f["binding_class"])
    check("while extraction alone would have missed it",
          max(f["extraction"].values()) < 24, str(f["extraction"]))
    # Listing ALL four -- the fix -- removes it.
    for r in rows:
        r["context"] = r["context"].replace(
            "Duty rosters list the " + ", the ".join(
                [x for x in roles if x != r["reference"]]) + " on site.",
            "Duty rosters list the " + ", the ".join(roles) + " on site.")
    check("listing every member closes it",
          structural(rows)["set_complement_singleton"] == 0,
          str(structural(rows)["set_complement_singleton"]))


def test_list_arity_attack():
    """`inspected_list_arity`: when exactly one item is always withdrawn, the answer is the
    listing's length minus one and the withdrawal clause is never read."""
    print("list arity is caught when the exclusion count is constant")
    rows = [{"id": f"n{i}", "stratum": "count", "reference": str(3 + i % 3),
             "context": ("The items inspected were "
                         + ", ".join(f"item{j}" for j in range(4 + i % 3))
                         + ". Of those, item0 was withdrawn."),
             "question": "How many count as inspected?"} for i in range(24)]
    st = structural(rows)
    check("list_arity_minus_1 reads it straight off",
          st["list_arity_minus_1"] == 24, str(st["list_arity_minus_1"]))
    check("and minus_0 does not", st["list_arity_minus_0"] == 0)


def test_binomial_matches_published_values():
    print("exact binomial reproduces the programme's published bounds")
    check("cp_lower(63,128) ~ 0.4028", abs(cp_lower(63, 128) - 0.4028) < 0.001,
          f"{cp_lower(63, 128):.4f}")
    check("cp_lower(56,128) ~ 0.3500", abs(cp_lower(56, 128) - 0.3500) < 0.001,
          f"{cp_lower(56, 128):.4f}")


def main() -> int:
    for fn in (test_rci001_recency_defect_is_caught, test_abstention_stratum_is_immune,
               test_ceiling_blocks_a_saturated_instrument, test_underpowered_is_not_failure,
               test_always_abstain_floor, test_pack_structure_is_required,
               test_structural_class_catches_the_complement_leak,
               test_list_arity_attack,
               test_convention_sensitivity_finds_the_article_rows,
               test_per_stratum_gate_catches_the_decorative_stratum,
               test_dissociation_needs_the_right_denominator,
               test_binomial_matches_published_values):
        fn()
    print()
    if FAIL:
        print(f"FIXTURE FAIL: {len(FAIL)}")
        for f in FAIL:
            print("  -", f)
        return 1
    print("FIXTURE PASS — the gate catches every defect that actually happened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
