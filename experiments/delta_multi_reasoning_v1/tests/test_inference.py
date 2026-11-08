"""Fixture tests for label isolation, the frozen selector policy, and role independence.

Run:  python experiments/delta_multi_reasoning_v1/tests/test_inference.py

FIXTURE PASS IS NOT NATIVE PASS. These prove the prompt construction and the selector's
degenerate-case handling. No model is called: `generate` is a stub, and every stub asserts on
what it was handed, which is how the label-isolation guarantees are checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.roles import (  # noqa: E402
    ABSTAIN, CONTRACT, MR_SOLVER_ROLES, R0, R4, REP_PARAPHRASES, assert_label_isolated, build,
    role_roster, selector_prompt,
)
from inference.selector import Candidate, policy, s0_agreement, select  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


ITEM = {
    "id": "RRP-select:RRFR2-deadbeef-step0-valid",
    "family": "ledger_replay",
    "reference": '{"A":-15,"B":16,"C":9,"D":-3}',
    "reference_text": '{"A":-15,"B":16,"C":9,"D":-3}',
    "source_split": "train",
    "view_role": "ADMITTED_FOR_SELECTION",
    "gold_verified_by": "oracle.replay_transition",
    "prompt": [
        {"role": "system", "content": "Apply one event to the supplied account balances."},
        {"role": "user", "content": "before: A=-10 B=16 C=9 D=-3; event: add A -5; status: posted"},
    ],
}


# ------------------------------------------------------------------------------------- #
def test_contract_is_identical_across_conditions() -> None:
    print("the final-answer contract is identical in every condition")
    o = build(ITEM["prompt"])
    t = build(ITEM["prompt"])
    reps = [build(ITEM["prompt"], paraphrase=p) for p in REP_PARAPHRASES]
    mrs = [build(ITEM["prompt"], role=r) for r in MR_SOLVER_ROLES]
    for label, msgs in [("O", o), ("T", t)] + \
                       [(f"REP{i}", m) for i, m in enumerate(reps)] + \
                       [(r.rid, m) for r, m in zip(MR_SOLVER_ROLES, mrs)]:
        text = "\n".join(m["content"] for m in msgs)
        check(f"{label} carries the contract", CONTRACT in text)
    check("O and T are byte-identical prompts (T differs only in token allowance)", o == t)
    check("MR's R0 is byte-identical to condition O",
          build(ITEM["prompt"], role=R0) == o,
          "R0 must be the ordinary solve so the shared generation is reusable")


def test_label_isolation() -> None:
    print("label isolation: a reference in a prompt fails loudly")
    good = build(ITEM["prompt"])
    try:
        assert_label_isolated(good, ITEM)
        check("clean prompt passes", True)
    except AssertionError as e:
        check("clean prompt passes", False, str(e))

    # A template regression that INTERPOLATES the reference into the constructed prompt.
    # Simulated by handing the checker a built prompt containing the reference while the
    # item's own prompt does not -- which is exactly "construction introduced it".
    leaked = build([{"role": "user",
                     "content": f"solve this. the answer is {ITEM['reference']}"}])
    try:
        assert_label_isolated(leaked, ITEM)
        check("reference introduced by construction is caught", False, "no AssertionError")
    except AssertionError as e:
        check("reference introduced by construction is caught", "label leak" in str(e))

    # A retrieval item legitimately contains its own answer in the evidence. That is the task,
    # not a leak, and it must NOT fire -- the first version refused every such item.
    needle_item = {"id": "lc1", "reference": "963692",
                   "prompt": [{"role": "system", "content": "Answer using only the document."},
                              {"role": "user",
                               "content": "... the seal number is 963692 ... What is it?"}]}
    try:
        assert_label_isolated(build(needle_item["prompt"]), needle_item)
        check("a needle item's own answer in its evidence is NOT a leak", True)
    except AssertionError as e:
        check("a needle item's own answer in its evidence is NOT a leak", False, str(e)[:120])

    # The family label is hidden suite metadata, not part of the user's request.
    fam = build([{"role": "user", "content": "this is a ledger_replay item, solve it"}])
    try:
        assert_label_isolated(fam, ITEM)
        check("leaked family label is caught", False, "no AssertionError raised")
    except AssertionError:
        check("leaked family label is caught", True)

    check("forbidden list covers reference and metadata",
          all(f in role_roster()["forbidden_fields_checked_at_call_time"]
              for f in ("reference", "family", "accepted_answers", "gold")))


def test_roles_cannot_see_each_other() -> None:
    print("R0-R3 are independent perspectives, structurally")
    import inspect
    sig = inspect.signature(build)
    check("build() has no parameter for candidates or other roles' outputs",
          set(sig.parameters) == {"prompt", "role", "paraphrase"}, str(set(sig.parameters)))
    check("no solver role is flagged as seeing other outputs",
          all(not r.sees_other_role_outputs for r in MR_SOLVER_ROLES))
    check("only the selector sees other outputs", R4.sees_other_role_outputs is True)
    check("four solver roles registered", len(MR_SOLVER_ROLES) == 4)
    check("role operations are distinct",
          len({r.operation for r in MR_SOLVER_ROLES}) == 4)
    check("REP has four paraphrases, matching MR's four solver calls",
          len(REP_PARAPHRASES) == len(MR_SOLVER_ROLES))


def test_selector_sees_no_gold() -> None:
    print("the selector prompt carries candidates but never a reference")
    # One candidate is exactly right, so it is byte-identical to the reference. That must be
    # allowed -- a selector has to see its candidates -- but only via `allow`.
    cands = [{"id": "R0", "answer": ITEM["reference"]}, {"id": "R1", "answer": '{"A":-99}'}]
    msgs = selector_prompt(ITEM["prompt"], cands, include_checker_findings=False)
    try:
        assert_label_isolated(msgs, ITEM, allow=[c["answer"] for c in cands])
        check("a correct candidate is allowed in the selector prompt", True)
    except AssertionError as e:
        check("a correct candidate is allowed in the selector prompt", False, str(e)[:120])
    try:
        assert_label_isolated(msgs, ITEM)   # same prompt, candidates NOT declared
        check("gold in a selector prompt is caught when not declared as a candidate",
              False, "no AssertionError")
    except AssertionError:
        check("gold in a selector prompt is caught when not declared as a candidate", True)
    text = "\n".join(m["content"] for m in msgs)
    check("candidate ids are offered", "R0" in text and "R1" in text)

    withheld = selector_prompt(ITEM["prompt"],
                               [{"id": "R2", "answer": "x", "finding": "premise missing"}],
                               include_checker_findings=False)
    shown = selector_prompt(ITEM["prompt"],
                            [{"id": "R2", "answer": "x", "finding": "premise missing"}],
                            include_checker_findings=True)
    check("S1 withholds checker findings",
          "premise missing" not in "\n".join(m["content"] for m in withheld))
    check("S2 shows checker findings",
          "premise missing" in "\n".join(m["content"] for m in shown))
    check("findings are a separate labelled field from answers",
          "note:" in "\n".join(m["content"] for m in shown))


def test_s0_degenerate_paths() -> None:
    print("S0: frozen degenerate paths")
    empty = s0_agreement([])
    check("empty candidate set -> abstention", empty.answer == ABSTAIN)
    check("empty candidate set flagged", empty.flags.get("empty_candidate_set") is True)
    check("empty candidate set costs no call", empty.extra_model_calls == 0)

    allbad = s0_agreement([Candidate("R0", "junk", None), Candidate("R1", "junk", None)])
    check("all malformed -> abstention", allbad.answer == ABSTAIN)
    check("all malformed flagged", allbad.flags.get("all_malformed") is True)

    maj = s0_agreement([Candidate("R0", "", "a"), Candidate("R1", "", "b"),
                        Candidate("R2", "", "a")])
    check("majority wins", maj.answer == "a" and maj.selected_rid == "R0")
    check("no tie flag on a clear majority", not maj.flags.get("tie_broken"))
    check("S0 costs zero extra model calls", maj.extra_model_calls == 0)

    tie = s0_agreement([Candidate("R1", "", "b"), Candidate("R0", "", "a")])
    check("tie broken to lowest-indexed role", tie.selected_rid == "R0" and tie.answer == "a")
    check("tie flagged", tie.flags.get("tie_broken") is True)
    tie2 = s0_agreement([Candidate("R0", "", "a"), Candidate("R1", "", "b")])
    check("tie break is independent of input order",
          (tie2.selected_rid, tie2.answer) == (tie.selected_rid, tie.answer))

    mixed = s0_agreement([Candidate("R0", "", "a"), Candidate("R1", "junk", None)])
    check("partial malformed is flagged but still selects",
          mixed.answer == "a" and mixed.flags.get("some_malformed") is True)

    absts = s0_agreement([Candidate("R0", "", ABSTAIN), Candidate("R1", "", ABSTAIN)])
    check("unanimous abstention selects abstention",
          absts.answer == ABSTAIN and absts.flags.get("abstained") is True)
    check("abstention is not auto-ranked down",
          s0_agreement([Candidate("R0", "", ABSTAIN), Candidate("R1", "", ABSTAIN),
                        Candidate("R2", "", "a")]).answer == ABSTAIN)


def test_model_selector_paths() -> None:
    print("S2: valid pick, malformed output, non-candidate id")
    cands = [Candidate("R0", "", "a"), Candidate("R1", "", "b"), Candidate("R2", "", "b")]
    calls: list[int] = []

    def gen_ok(msgs, mnt):
        calls.append(mnt)
        return "R1"

    s = select(cands, "S2", generate=gen_ok, prompt=ITEM["prompt"])
    check("valid id is honoured", s.selected_rid == "R1" and s.answer == "b")
    check("S2 costs exactly one extra call", s.extra_model_calls == 1)
    check("selector token cap is small and frozen", calls and calls[0] == R4.max_new_tokens)
    check("agreement is recorded alongside", s.agreement.get("b") == 2)

    s2 = select(cands, "S2", generate=lambda m, n: "I think the answer is probably b",
                prompt=ITEM["prompt"])
    check("prose without an id -> S0 fallback", s2.flags.get("selector_malformed") is True)
    check("fallback still answers", s2.answer == "b", "S0 majority is b")
    check("fallback still charges the wasted call", s2.extra_model_calls == 1)

    s3 = select(cands, "S2", generate=lambda m, n: "R9", prompt=ITEM["prompt"])
    check("non-candidate id -> malformed fallback", s3.flags.get("selector_malformed") is True)

    s4 = select(cands, "S2", generate=lambda m, n: "R0 or R1", prompt=ITEM["prompt"])
    check("several ids -> malformed fallback", s4.flags.get("selector_malformed") is True)

    try:
        select(cands, "S3", generate=gen_ok, prompt=ITEM["prompt"])
        check("unregistered variant refused", False, "no ValueError")
    except ValueError as e:
        check("unregistered variant refused", "unregistered" in str(e))

    try:
        select(cands, "S2")
        check("S2 without a backend refused", False, "no ValueError")
    except ValueError as e:
        check("S2 without a backend refused", "needs a native" in str(e))


def test_policy_is_frozen_and_complete() -> None:
    print("the frozen policy names every degenerate path")
    p = policy()
    check("main selector is S2", p["main_selector"] == "S2")
    check("selection is an ID, not a synthesised answer",
          "candidate ID" in p["selection_output"])
    for k in ("empty_candidate_set", "all_candidates_malformed", "s0_tie",
              "malformed_selector_output", "selector_names_non_candidate",
              "abstention_as_candidate"):
        check(f"policy fixes {k}", k in p["frozen_degenerate_paths"])
    check("oracle gets no deployment credit", "no deployment credit" in p["oracle_note"])
    r = role_roster()
    check("roster records independence",
          "never each other's outputs" in r["mr_independence"])
    check("roster records greedy decoding", "greedy" in r["decoding"])


def main() -> int:
    for fn in (test_contract_is_identical_across_conditions, test_label_isolation,
               test_roles_cannot_see_each_other, test_selector_sees_no_gold,
               test_s0_degenerate_paths, test_model_selector_paths,
               test_policy_is_frozen_and_complete):
        fn()
    print()
    if FAILURES:
        print(f"FIXTURE FAIL: {len(FAILURES)} check(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("FIXTURE PASS (prompt construction and selector policy only; no model was called)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
