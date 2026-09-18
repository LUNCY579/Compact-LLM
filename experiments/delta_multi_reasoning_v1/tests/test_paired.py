"""Fixture tests for the paired/interaction/selector analysis.

Run:  python experiments/delta_multi_reasoning_v1/tests/test_paired.py
      (or under pytest; both work)

FIXTURE PASS IS NOT NATIVE PASS. These prove the arithmetic that turns recorded outcomes into
the report's tables. They say nothing about any model's behaviour. That distinction is the
reason this file exists separately from any run receipt.

Several cases are built from THIS PROGRAMME's own recorded numbers, so a regression in the
analysis code shows up as a disagreement with a published table rather than as an abstract
assertion failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.paired import (  # noqa: E402
    REP001_SINGLE_DRAW_SPREAD, cluster_bootstrap_ci, cost_per_net_correct, holm_bonferroni,
    interaction, mcnemar_exact, noise_verdict, paired, preservation_verdict,
    role_complementarity, roles_perform_different_operations, selector_utilisation,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


# ------------------------------------------------------------------------------------- #
def test_paired_basic() -> None:
    print("paired(): repair/damage/net")
    base = {"a": True, "b": True, "c": False, "d": False, "e": True}
    cand = {"a": True, "b": False, "c": True, "d": False, "e": False}
    p = paired("B", "C", base, cand)
    check("both_correct", p.both_correct == 1)
    check("repair == 1 (c)", p.repair == 1 and p.repair_ids == ("c",))
    check("damage == 2 (b,e)", p.damage == 2 and p.damage_ids == ("b", "e"))
    check("both_wrong == 1", p.both_wrong == 1)
    check("net_gain == -1", p.net_gain == -1)
    check("baseline_correct == 3", p.baseline_correct == 3)
    check("candidate_correct == 2", p.candidate_correct == 2)
    check("counts sum to n", p.both_correct + p.repair + p.damage + p.both_wrong == p.n)
    check("not zero_observed_damage", p.zero_observed_damage is False)


def test_paired_refuses_partial_overlap() -> None:
    print("paired(): refuses a partial id overlap")
    try:
        paired("B", "C", {"a": True, "b": True}, {"a": True})
    except ValueError as e:
        check("raises ValueError", "id sets differ" in str(e))
        return
    check("raises ValueError", False, "no exception raised")


def test_zero_damage_is_the_only_preservation_pass() -> None:
    print("preservation_verdict(): a large p-value is not equivalence")
    # 7 lost, 10 gained: SPARSE-001's recorded long-context decomposition at p = 0.63.
    base = {f"i{n}": True for n in range(7)}
    cand = {f"i{n}": False for n in range(7)}
    for n in range(10):
        base[f"j{n}"] = False
        cand[f"j{n}"] = True
    for n in range(239):
        base[f"k{n}"] = True
        cand[f"k{n}"] = True
    p = paired("Supa", "k0.25", base, cand)
    check("repair == 10", p.repair == 10)
    check("damage == 7", p.damage == 7)
    check("net_gain == +3", p.net_gain == 3)
    v = preservation_verdict(p)
    check("strict criterion FAILS despite net +3",
          v["strict_criterion"] == "FAIL_OBSERVED_DAMAGE")
    mc = mcnemar_exact(10, 7)
    check("mcnemar p is large (>0.4) yet strict still fails", mc > 0.4, f"p={mc:.4f}")
    check("noninferiority NOT_REGISTERED by default",
          v["noninferiority"] == "NOT_REGISTERED")

    clean = {f"k{n}": True for n in range(50)}
    p2 = paired("Supa", "identity", clean, dict(clean))
    v2 = preservation_verdict(p2)
    check("zero damage PASSES strict", v2["strict_criterion"] == "PASS_ZERO_OBSERVED_DAMAGE")


def test_underpowered_noninferiority_is_inconclusive() -> None:
    print("preservation_verdict(): underpowered means inconclusive, not pass")
    base = {f"k{n}": True for n in range(100)}
    cand = dict(base)
    cand["k0"] = False                       # 1 discordant pair only
    p = paired("B", "C", base, cand)
    v = preservation_verdict(p, margin=5)
    check("within margin but underpowered -> INCONCLUSIVE",
          v["noninferiority"] == "INCONCLUSIVE_UNDERPOWERED", str(v["noninferiority"]))


def test_noise_floor_against_recorded_differences() -> None:
    print("noise_verdict(): the report's own margins against REP-001 spreads")
    # These are the exact differences the report leans on.
    check("k0.25 vs Titan long context (164-162=+2) is WITHIN_NOISE",
          noise_verdict("long_context", 2) == "WITHIN_NOISE")
    check("k0.25 vs Titan acquisition (232-234=-2) is WITHIN_NOISE",
          noise_verdict("acquisition", -2) == "WITHIN_NOISE")
    check("4B vs best core long context (180-164=+16) EXCEEDS_NOISE",
          noise_verdict("long_context", 16) == "EXCEEDS_NOISE")
    check("4B vs best core acquisition (270-232=+38) EXCEEDS_NOISE",
          noise_verdict("acquisition", 38) == "EXCEEDS_NOISE")
    check("boundary: exactly the spread is WITHIN_NOISE",
          noise_verdict("long_context", REP001_SINGLE_DRAW_SPREAD["long_context"]) == "WITHIN_NOISE")
    check("unregistered endpoint says so",
          noise_verdict("abstention", 4) == "NO_REGISTERED_SPREAD")


def test_interaction_contrast() -> None:
    print("interaction(): difference in differences")
    cells = {("D", "MR"): 240.0, ("D", "O"): 232.0, ("B", "MR"): 20.0, ("B", "O"): 14.0}
    r = interaction(lambda s, i: cells[(s, i)], "D", "B", "MR", "O")
    check("simple effect on D == +8", r["simple_effect_on"]["D"] == 8.0)
    check("simple effect on B == +6", r["simple_effect_on"]["B"] == 6.0)
    check("interaction == +2", r["interaction"] == 2.0)
    # A large main effect with zero interaction must report zero.
    cells2 = {("D", "MR"): 250.0, ("D", "O"): 232.0, ("B", "MR"): 32.0, ("B", "O"): 14.0}
    r2 = interaction(lambda s, i: cells2[(s, i)], "D", "B", "MR", "O")
    check("equal simple effects -> interaction 0", r2["interaction"] == 0.0)


def test_selector_utilisation() -> None:
    print("selector_utilisation(): denominator is cases with a correct candidate")
    cc = {
        "i1": {"R0": True, "R1": False},     # correct available, selector gets it
        "i2": {"R0": False, "R1": True},     # correct available, selector misses
        "i3": {"R0": False, "R1": False},    # nothing correct: cannot test the selector
        "i4": {"R0": True, "R1": True},      # correct available, selector abstains
    }
    selected = {"i1": "R0", "i2": "R0", "i3": "R0", "i4": None}
    final = {"i1": True, "i2": False, "i3": False, "i4": False}
    u = selector_utilisation(cc, selected, final)
    check("n_with_correct_candidate == 3", u["n_with_correct_candidate"] == 3)
    check("n_with_no_correct_candidate == 1", u["n_with_no_correct_candidate"] == 1)
    check("n_selected_correctly == 1", u["n_selected_correctly"] == 1)
    check("utilisation == 1/3", abs(u["utilisation"] - 1 / 3) < 1e-12)
    check("abstention despite correct candidate counted",
          u["n_abstained_despite_correct_candidate"] == 1)
    check("shortfall vs oracle == 2", u["selector_shortfall_vs_oracle"] == 2)


def test_role_complementarity_detects_redundancy() -> None:
    print("role_complementarity(): redundant roles show zero headroom")
    same = {f"i{n}": {"R0": n % 2 == 0, "R1": n % 2 == 0, "R2": n % 2 == 0} for n in range(10)}
    r = role_complementarity(same)
    check("identical roles -> oracle == best single", r["oracle_minus_best_single"] == 0)
    check("no unique solves", all(v == 0 for v in r["uniquely_solved_by"].values()))
    comp = {
        "i1": {"R0": True, "R1": False},
        "i2": {"R0": False, "R1": True},
        "i3": {"R0": False, "R1": False},
    }
    r2 = role_complementarity(comp)
    check("complementary roles -> headroom +1", r2["oracle_minus_best_single"] == 1)
    check("each solves one uniquely",
          r2["uniquely_solved_by"] == {"R0": 1, "R1": 1})
    check("correlated failure counted", r2["correlated_failure"] == 1)


def test_roles_collapsed_detection() -> None:
    print("roles_perform_different_operations(): name-only differentiation is caught")
    collapsed = {f"i{n}": {"R0": "X", "R1": "X", "R2": "X"} for n in range(10)}
    r = roles_perform_different_operations(collapsed)
    check("collapsed roles flagged", r["verdict"] == "ROLES_COLLAPSED")
    check("all-identical count == 10", r["items_where_all_roles_emitted_identical_text"] == 10)
    diverse = {f"i{n}": {"R0": f"a{n}", "R1": f"b{n}", "R2": f"c{n}"} for n in range(10)}
    check("diverse roles pass", roles_perform_different_operations(diverse)["verdict"] == "ROLES_DIVERGE")


def test_cost_refuses_ratio_without_gain() -> None:
    print("cost_per_net_correct(): no attractive ratio for a non-positive gain")
    r = cost_per_net_correct(0, {"model_calls": 1920.0})
    check("zero gain -> None", r["cost_per_net_correct"] is None)
    check("zero gain -> NOT_REPORTABLE", r["verdict"].startswith("NOT_REPORTABLE"))
    r2 = cost_per_net_correct(-3, {"model_calls": 1920.0})
    check("negative gain -> NOT_REPORTABLE", r2["verdict"].startswith("NOT_REPORTABLE"))
    r3 = cost_per_net_correct(8, {"model_calls": 1920.0})
    check("positive gain -> ratio", abs(r3["cost_per_net_correct"]["model_calls"] - 240.0) < 1e-9)


def test_holm_and_bootstrap_determinism() -> None:
    print("holm_bonferroni() and cluster_bootstrap_ci()")
    h = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.6}, alpha=0.05)
    check("smallest p rejected", h["results"]["a"]["reject"] is True)
    check("step-down stops after a failure", h["results"]["c"]["reject"] is False)
    check("family size recorded", h["family_size"] == 3)

    by_world = {f"w{w}": [1.0 if (w + i) % 3 else 0.0 for i in range(4)] for w in range(12)}
    c1 = cluster_bootstrap_ci(by_world, iters=500)
    c2 = cluster_bootstrap_ci(by_world, iters=500)
    check("bootstrap is deterministic under a fixed seed",
          (c1["lo"], c1["hi"], c1["point"]) == (c2["lo"], c2["hi"], c2["point"]))
    check("interval brackets the point", c1["lo"] <= c1["point"] <= c1["hi"])
    check("unit is the world", c1["unit"].startswith("world"))
    check("n_worlds == 12", c1["n_worlds"] == 12)


def test_mcnemar_edges() -> None:
    print("mcnemar_exact(): edges")
    check("no discordant pairs -> 1.0", mcnemar_exact(0, 0) == 1.0)
    check("symmetric -> 1.0", mcnemar_exact(5, 5) == 1.0)
    check("all one way is small", mcnemar_exact(10, 0) < 0.01)
    check("bounded above by 1", mcnemar_exact(1, 1) <= 1.0)


def main() -> int:
    for fn in (test_paired_basic, test_paired_refuses_partial_overlap,
               test_zero_damage_is_the_only_preservation_pass,
               test_underpowered_noninferiority_is_inconclusive,
               test_noise_floor_against_recorded_differences, test_interaction_contrast,
               test_selector_utilisation, test_role_complementarity_detects_redundancy,
               test_roles_collapsed_detection, test_cost_refuses_ratio_without_gain,
               test_holm_and_bootstrap_determinism, test_mcnemar_edges):
        fn()
    print()
    if FAILURES:
        print(f"FIXTURE FAIL: {len(FAILURES)} check(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("FIXTURE PASS (analysis arithmetic only; not a native or behavioural pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
