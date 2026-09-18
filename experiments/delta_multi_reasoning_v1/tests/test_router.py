"""Fixture tests for the router: information boundary, fixed dispatch, cache equivalence.

Run:  python experiments/delta_multi_reasoning_v1/tests/test_router.py

FIXTURE PASS IS NOT NATIVE PASS. No model is called and no backend is loaded. In particular
nothing here measures routed latency: `route_cached` is scored from recorded outcomes by
construction, and the tests assert that it says so.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.router import (  # noqa: E402
    LARGE, SMALL, assert_input_only, dispatch, features, route_cached, rule,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


def msg(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


STATE_REQ = msg('before: {"A":-10,"B":16,"C":9,"D":-3}; event: add A -5; status: posted')
LONGDOC_REQ = msg("Here is a report.\n" + ("Sales rose in March. " * 120) +
                  "\nWhich month did sales rise?")
SHORTCUT_REQ = msg("Background.\n" + ("Context line. " * 120) +
                   '\nNow apply: before: {"A":1,"B":2,"C":3}; event: scale B 2; status: posted')
AMBIG_REQ = msg('The value is {"x":1}. What is x?')


def test_information_boundary_is_structural() -> None:
    print("the router decides from the request alone")
    sig = inspect.signature(dispatch)
    check("dispatch() takes only messages", list(sig.parameters) == ["messages"],
          str(list(sig.parameters)))
    sigf = inspect.signature(features)
    check("features() takes only messages", list(sigf.parameters) == ["messages"])

    item = {"id": "i1", "family": "ledger_replay", "reference": '{"A":-15}',
            "instrument": "SELECT-VIEWS-001", "variant": "unanswerable"}
    try:
        assert_input_only(STATE_REQ, item)
        check("clean request passes", True)
    except AssertionError as e:
        check("clean request passes", False, str(e))

    for field, val in (("family", "ledger_replay"),
                       ("instrument", "SELECT-VIEWS-001"),
                       ("variant", "unanswerable")):
        leaky = msg(f"solve this {val} item: before: {{\"A\":1,\"B\":2}}")
        try:
            assert_input_only(leaky, item)
            check(f"leaked {field} is caught", False, "no AssertionError")
        except AssertionError:
            check(f"leaked {field} is caught", True)

    check("forbidden list names suite and answerability fields",
          all(f in rule()["forbidden_information"]
              for f in ("instrument", "family", "accepted_answers", "is_correct", "variant")))


def test_dispatch_rule() -> None:
    print("the frozen dispatch rule")
    check("structured state -> small specialist", dispatch(STATE_REQ).component == SMALL)
    check("long document, no state -> large general", dispatch(LONGDOC_REQ).component == LARGE)
    check("SHORTCUT: long request that still carries a state -> small",
          dispatch(SHORTCUT_REQ).component == SMALL,
          "length must be the tie-breaker, not the primary signal")
    check("single-key object is not a multi-key state -> large",
          dispatch(AMBIG_REQ).component == LARGE)
    check("dispatch is deterministic",
          dispatch(STATE_REQ).component == dispatch(STATE_REQ).component)
    check("dispatch records a reason", bool(dispatch(STATE_REQ).reason))
    check("dispatch records the features it used",
          dispatch(STATE_REQ).features["widest_json_object_keys"] == 4)


def test_route_cached_reports_every_branch() -> None:
    print("route_cached(): per-branch accuracy, misrouting loss, and the labels")
    items = [
        {"id": "s1", "prompt": STATE_REQ},      # -> SMALL
        {"id": "s2", "prompt": SHORTCUT_REQ},   # -> SMALL
        {"id": "l1", "prompt": LONGDOC_REQ},    # -> LARGE
        {"id": "l2", "prompt": AMBIG_REQ},      # -> LARGE
    ]
    # Complementary components, the profile QWEN-001 actually measured.
    small = {"s1": True, "s2": True, "l1": False, "l2": False}
    large = {"s1": False, "s2": False, "l1": True, "l2": False}
    cc = {"D": small, "G0": large}
    r = route_cached(items, cc, small_system="D", large_system="G0")

    check("labelled cached_functional", r["evaluation_label"] == "cached_functional")
    check("runtime is NOT qualified", r["runtime_qualified"] is False)
    check("latency caveat present", "not routed-model latency" in r["latency_note"])
    check("dispatch counts split 2/2",
          r["dispatch_counts"] == {SMALL: 2, LARGE: 2}, str(r["dispatch_counts"]))
    check("routed beats both components", r["routed_correct"] == 3
          and r["always_small_correct"] == 2 and r["always_large_correct"] == 1)
    check("net vs always-small == +1", r["net_vs_always_small"] == 1)
    check("net vs always-large == +2", r["net_vs_always_large"] == 2)
    check("small branch conditional accuracy == 1.0",
          r["branches"][SMALL]["conditional_accuracy"] == 1.0)
    check("large branch conditional accuracy == 0.5",
          r["branches"][LARGE]["conditional_accuracy"] == 0.5)
    check("no misrouting in this fixture", r["total_misrouted_count"] == 0)
    check("zero observed misrouting flag set", r["zero_observed_misrouting"] is True)

    # Now a genuine misroute: the large model would have been right on a state item.
    large2 = dict(large); large2["s1"] = True
    small2 = dict(small); small2["s1"] = False
    r2 = route_cached(items, {"D": small2, "G0": large2}, "D", "G0")
    check("a compensated misroute is still counted, not netted away",
          r2["branches"][SMALL]["misrouted_count"] == 1)
    check("the offsetting rescue is reported separately",
          r2["branches"][SMALL]["rescued_count"] == 1)
    check("net branch advantage is 0 while one item was misrouted",
          r2["branches"][SMALL]["net_branch_advantage"] == 0)
    check("zero_observed_misrouting is False once any item is misrouted",
          r2["zero_observed_misrouting"] is False)
    check("misrouted ids are listed", "s1" in r2["branches"][SMALL]["misrouted_ids"])


def test_cache_equivalence_and_fixed_rule() -> None:
    print("cache equivalence, and the rule is fixed across component substitutions")
    items = [{"id": "s1", "prompt": STATE_REQ}, {"id": "l1", "prompt": LONGDOC_REQ}]
    small = {"s1": True, "l1": False}
    large = {"s1": False, "l1": True}

    r = route_cached(items, {"D": small, "G0": large}, "D", "G0")
    # The routed answer on each item must be exactly the dispatched component's answer.
    for iid, comp in r["per_item_dispatch"].items():
        src = small if comp == SMALL else large
        check(f"cache equivalence on {iid}", r["routed_per_item_correct"][iid] == src[iid])

    # Substituting L for D, and G1 for G0, must not change any dispatch.
    lcur = {"s1": True, "l1": True}
    g1 = {"s1": True, "l1": True}
    r_l0 = route_cached(items, {"L": lcur, "G0": large}, "L", "G0")
    r_d1 = route_cached(items, {"D": small, "G1": g1}, "D", "G1")
    check("dispatch identical across R_D0 / R_L0",
          r["per_item_dispatch"] == r_l0["per_item_dispatch"])
    check("dispatch identical across R_D0 / R_D1",
          r["per_item_dispatch"] == r_d1["per_item_dispatch"])
    check("rule declares it is frozen across all four routed systems",
          set(rule()["frozen_identically_across"]) == {"R_D0", "R_L0", "R_D1", "R_L1"})

    try:
        route_cached(items, {"D": {"s1": True}, "G0": large}, "D", "G0")
        check("missing recorded outcome refused", False, "no KeyError")
    except KeyError:
        check("missing recorded outcome refused", True)


def test_rule_declares_its_own_limits() -> None:
    print("the rule states what it is not")
    rl = rule()
    check("not an output-dependent cascade", "cascade" in rl["not"])
    check("end_to_end_native marked NOT YET RUN",
          "NOT YET RUN" in rl["evaluation_labels"]["end_to_end_native"])
    check("oracle gets no deployment credit", "no deployment credit" in rl["oracle_note"])
    for req in ("ambiguous cases", "shortcut checks"):
        check(f"scout requirement recorded: {req}", req in rl["required_of_any_scout"])


def main() -> int:
    for fn in (test_information_boundary_is_structural, test_dispatch_rule,
               test_route_cached_reports_every_branch, test_cache_equivalence_and_fixed_rule,
               test_rule_declares_its_own_limits):
        fn()
    print()
    if FAILURES:
        print(f"FIXTURE FAIL: {len(FAILURES)} check(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("FIXTURE PASS (dispatch logic and cache equivalence only; no model, no latency)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
