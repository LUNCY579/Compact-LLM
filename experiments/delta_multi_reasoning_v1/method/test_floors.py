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
from floors import cp_lower, extraction, floors, gate, immunity, min_k_to_clear  # noqa: E402

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


def test_binomial_matches_published_values():
    print("exact binomial reproduces the programme's published bounds")
    check("cp_lower(63,128) ~ 0.4028", abs(cp_lower(63, 128) - 0.4028) < 0.001,
          f"{cp_lower(63, 128):.4f}")
    check("cp_lower(56,128) ~ 0.3500", abs(cp_lower(56, 128) - 0.3500) < 0.001,
          f"{cp_lower(56, 128):.4f}")


def main() -> int:
    for fn in (test_rci001_recency_defect_is_caught, test_abstention_stratum_is_immune,
               test_ceiling_blocks_a_saturated_instrument, test_underpowered_is_not_failure,
               test_always_abstain_floor, test_binomial_matches_published_values):
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
