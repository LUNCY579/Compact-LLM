"""Fixture gates for the adversarial verifier audit.

Hermetic: no release tree, no model, no network. `attacks()` is a pure function, and
`audit()` is exercised against stub verifiers whose behaviour is chosen to be either
correct or deliberately broken. The point is that the audit CATCHES a broken verifier,
which is the only property that makes a passing audit worth anything.
"""
import json

import pytest

from experiments.reasoning_posttraining_v1 import verifier_audit as va

VALUE_GOLD = {"value": -28}
STATUS_GOLD = {"status": "UNKNOWN"}
PLAN_GOLD = {"cost": 20, "path": ["N3", "N1", "N4", "N2"]}
WORLD = {"family": "ledger_replay", "query": {}, "records": []}


def names(reference, world=WORLD):
    return [c["name"] for c in va.attacks(reference, world)]


# ---------------------------------------------------------------- attack construction

def test_every_answer_shape_gets_the_universal_attacks():
    universal = {"gold_witness", "malformed_not_json", "malformed_markdown_fence",
                 "malformed_trailing_text", "malformed_empty", "malformed_json_array",
                 "malformed_two_objects", "manipulation_empty_object",
                 "manipulation_injection_key", "manipulation_reward_key",
                 "manipulation_padded_gold", "manipulation_all_statuses"}
    for ref in (VALUE_GOLD, STATUS_GOLD, PLAN_GOLD):
        assert universal <= set(names(ref)), ref


def test_abstention_attacks_point_the_right_way():
    # An answerable world is attacked with false abstention...
    answerable = names(VALUE_GOLD)
    assert "false_unknown_on_answerable" in answerable
    assert "false_impossible_on_answerable" in answerable
    assert "wrong_status_sentinel" not in answerable
    # ...and an unanswerable one with a false answer and the opposite sentinel.
    unanswerable = names(STATUS_GOLD)
    assert "false_answer_on_unanswerable" in unanswerable
    assert "wrong_status_sentinel" in unanswerable
    assert "false_unknown_on_answerable" not in unanswerable
    swap = [c for c in va.attacks(STATUS_GOLD, WORLD) if c["name"] == "wrong_status_sentinel"][0]
    assert json.loads(swap["response"]) == {"status": "IMPOSSIBLE"}


def test_plan_attacks_only_for_plan_shapes_and_are_really_different():
    assert "nonoptimal_cost_claim" not in names(VALUE_GOLD)
    assert "invalid_action_token" not in names(STATUS_GOLD)
    # Only the plan-specific attacks; the malformed ones are deliberately not JSON.
    wanted = {"reversed_plan_consistency", "truncated_plan", "duplicated_step",
              "invalid_action_token", "nonoptimal_cost_claim"}
    plan = {c["name"]: json.loads(c["response"])
            for c in va.attacks(PLAN_GOLD, WORLD) if c["name"] in wanted}
    assert set(plan) == wanted
    assert plan["reversed_plan_consistency"]["path"] == ["N2", "N4", "N1", "N3"]
    assert plan["truncated_plan"]["path"] == ["N3", "N1", "N4"]
    assert plan["duplicated_step"]["path"][-2:] == ["N2", "N2"]
    assert plan["invalid_action_token"]["path"][-1] == "__NOT_A_REAL_STEP__"
    assert plan["nonoptimal_cost_claim"]["cost"] == 21
    assert "off_by_one_value" not in plan


def test_off_by_one_only_for_integer_values():
    assert "off_by_one_value" in names(VALUE_GOLD)
    assert "off_by_one_value" not in names({"answer": "S0"})
    case = [c for c in va.attacks(VALUE_GOLD, WORLD) if c["name"] == "off_by_one_value"][0]
    assert json.loads(case["response"]) == {"value": -27}


def test_padded_gold_is_expected_to_still_be_correct():
    case = [c for c in va.attacks(VALUE_GOLD, WORLD)
            if c["name"] == "manipulation_padded_gold"][0]
    assert case["expect"]["correct"] is True
    assert json.loads(case["response"]) == VALUE_GOLD      # padding only, same object


def test_no_attack_response_is_unbounded_or_nonstring():
    for ref in (VALUE_GOLD, STATUS_GOLD, PLAN_GOLD):
        for case in va.attacks(ref, WORLD):
            assert isinstance(case["response"], str)
            assert len(case["response"]) <= 100_000       # the parser's own bound


# ---------------------------------------------------------------- audit behaviour

class Stub:
    """A stub `rewards` module. `mode` selects correct or broken verifier behaviour."""

    def __init__(self, mode):
        self.mode = mode

    def score_response(self, world, response, certificate=None):
        out = {"reward": 0.0, "correct": False, "format_valid": False, "schema_valid": False,
               "feasible": False, "reason": "parse_failure"}
        if self.mode == "crash":
            raise RuntimeError("verifier exploded")
        try:
            obj = json.loads(response)
            if not isinstance(obj, dict):
                raise ValueError
        except Exception:
            return out
        out["format_valid"] = True
        gold = world["_gold"]
        if self.mode == "credulous":                       # credits anything parseable
            return {**out, "schema_valid": True, "feasible": True, "correct": True,
                    "reward": 1.0, "reason": "exact_match"}
        if self.mode == "stingy" and obj == gold:          # refuses its own gold
            return {**out, "schema_valid": True, "reason": "incorrect_value"}
        if self.mode == "inconsistent" and obj == gold:    # correct with a failing reason
            return {**out, "schema_valid": True, "feasible": True, "correct": True,
                    "reward": 1.0, "reason": "invalid_witness"}
        if self.mode == "lowreward" and obj == gold:       # correct but under-rewarded
            return {**out, "schema_valid": True, "feasible": True, "correct": True,
                    "reward": 0.10, "reason": "exact_match"}
        if obj == gold:
            return {**out, "schema_valid": True, "feasible": True, "correct": True,
                    "reward": 1.0, "reason": "exact_match"}
        out["schema_valid"] = True
        out["reason"] = "incorrect_value"
        return out


def worlds_for(gold):
    return [{"id": "w1", "world": {**WORLD, "_gold": gold}, "reference": gold,
             "certificate": None}]


def run(mode, gold=VALUE_GOLD):
    return va.audit(worlds_for(gold), {"rewards": Stub(mode)})


def test_a_correct_verifier_passes_the_audit():
    rep = run("good")
    assert rep["status"] == "PASS", rep["first_failures"]
    assert rep["violations"] == 0
    assert rep["credit_leaks"] == {}
    assert rep["attacks_that_ever_earned_credit"] == {"gold_witness": 1,
                                                      "manipulation_padded_gold": 1}


@pytest.mark.parametrize("mode,expected", [
    ("credulous", "credits attacks that must never be credited"),
    ("stingy", "refuses its own gold witness"),
    ("inconsistent", "pairs correct=True with a failing reason"),
    ("lowreward", "reports correct with reward below 0.95"),
    ("crash", "raises instead of returning a verdict"),
])
def test_the_audit_catches_a_broken_verifier(mode, expected):
    rep = run(mode)
    assert rep["status"] == "FAIL", f"audit missed a verifier that {expected}"
    assert rep["violations"] > 0


def test_credulous_verifier_is_reported_as_a_credit_leak():
    rep = run("credulous")
    # Every non-permitted attack class that earned credit must be named.
    assert rep["credit_leaks"], "a verifier crediting everything must leak"
    assert "malformed_not_json" not in rep["credit_leaks"]   # unparseable, so not credited
    assert "manipulation_empty_object" in rep["credit_leaks"]
    assert "false_unknown_on_answerable" in rep["credit_leaks"]


def test_audit_reports_the_reason_vocabulary_it_actually_saw():
    rep = run("good")
    assert set(rep["reasons_observed"]) <= set(va.REASONS)
    assert "exact_match" in rep["reasons_observed"]
    assert set(rep["reasons_never_exercised"]) == set(va.REASONS) - set(rep["reasons_observed"])


def test_feasible_is_not_required_for_status_only_answers():
    """The real verifier reports feasible=False on a correct abstention."""
    class StatusStub(Stub):
        def score_response(self, world, response, certificate=None):
            got = super().score_response(world, response, certificate)
            if got["correct"]:
                got["feasible"] = False                     # not applicable, as upstream does
                got["reason"] = "correct_status"
            return got

    rep = va.audit(worlds_for(STATUS_GOLD), {"rewards": StatusStub("good")})
    assert rep["status"] == "PASS", rep["first_failures"]
    assert "feasible" not in str(rep["first_failures"])


def test_plan_answers_still_require_feasible_when_correct():
    class BadPlan(Stub):
        def score_response(self, world, response, certificate=None):
            got = super().score_response(world, response, certificate)
            if got["correct"]:
                got["feasible"] = False                     # illegitimate for a plan answer
            return got

    rep = va.audit(worlds_for(PLAN_GOLD), {"rewards": BadPlan("good")})
    assert rep["status"] == "FAIL"
    assert any("failed channel" in str(f) for f in rep["first_failures"])


def test_audit_never_claims_quality_or_spends_budget():
    rep = run("good")
    assert rep["model_calls"] == 0 and rep["optimizer_steps"] == 0
    assert rep["quality_claim"] is False
    assert rep["semantic_and_format_separately_observable"] is True
