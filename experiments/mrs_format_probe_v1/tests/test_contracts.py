"""Synthetic-fixture validation of the probe's strict parsers.

These are software checks. They are not a model result and not evidence of efficacy.
"""
import json

import pytest

from experiments.mrs_format_probe_v1 import contracts
from experiments.mrs_format_probe_v1.contracts import CONTRACTS, parse, messages

ALL = list(CONTRACTS)


def ok(cid, text, role="R0"):
    return parse(cid, text, role, True)


# --- the control must still reject exactly what MRS-1 rejected ------------------


def test_c0_rejects_the_observed_bare_answer():
    # This is the literal output both native candidates produced under MRS-1.
    for token in ("0", "1", "2", "3"):
        r = ok("C0_JSON_MRS1", token)
        assert r["format_failure"], token
        assert "invalid answer schema" in r["parse_error"]


def test_c0_accepts_a_well_formed_object():
    r = ok("C0_JSON_MRS1", '{"answer": "2"}')
    assert not r["format_failure"]
    assert r["parsed"]["answer"] == "2"


def test_c0_bare_unknown_is_not_json_at_all():
    r = ok("C0_JSON_MRS1", "UNKNOWN")
    assert r["format_failure"]


# --- bare ----------------------------------------------------------------------


@pytest.mark.parametrize("token", ["0", "1", "2", "3", "UNKNOWN"])
def test_c1_accepts_every_vocabulary_token(token):
    r = ok("C1_BARE", token)
    assert not r["format_failure"]
    assert r["parsed"]["answer"] == token


@pytest.mark.parametrize("bad", ["4", "-1", "unknown", "The answer is 2", "2.", '{"answer": "2"}', ""])
def test_c1_rejects_out_of_contract_text(bad):
    assert ok("C1_BARE", bad)["format_failure"]


# --- line forms ----------------------------------------------------------------


def test_c2_solver_line():
    r = ok("C2_LINE", "ANSWER: 3")
    assert not r["format_failure"] and r["parsed"] == {"answer": "3"}


def test_c2_checker_requires_its_own_field_name():
    good = parse("C2_LINE", "ANSWER: 3\nSUFFICIENT: YES", "R2", True)
    assert not good["format_failure"]
    assert good["parsed"] == {"answer": "3", "sufficient_evidence": True}
    # R3's field name must not be accepted for R2.
    assert parse("C2_LINE", "ANSWER: 3\nVALID: YES", "R2", True)["format_failure"]
    r3 = parse("C2_LINE", "ANSWER: 1\nVALID: NO", "R3", True)
    assert not r3["format_failure"]
    assert r3["parsed"] == {"answer": "1", "candidate_valid": False}


def test_c2_checker_rejects_a_missing_flag_line():
    assert parse("C2_LINE", "ANSWER: 3", "R2", True)["format_failure"]


@pytest.mark.parametrize("bad", ["answer: 3", "ANSWER:3\nextra", "ANSWER: 7", "ANSWER: 3 because x"])
def test_c2_rejects_near_misses(bad):
    assert ok("C2_LINE", bad)["format_failure"]


def test_c2_accepts_no_space_variant():
    # The registered regex allows an optional single space; state that explicitly.
    assert not ok("C2_LINE", "ANSWER:3")["format_failure"]


# --- prefilled contracts parse the PREFILL + COMPLETION concatenation -----------


def test_c3_parses_prefill_concatenation():
    prefill = CONTRACTS["C3_JSON_PREFILL"]["prefill"]
    assert prefill == '{"answer": "'
    r = ok("C3_JSON_PREFILL", prefill + '2"}')
    assert not r["format_failure"] and r["parsed"]["answer"] == "2"


def test_c3_checker_still_needs_its_boolean():
    prefill = CONTRACTS["C3_JSON_PREFILL"]["prefill"]
    assert parse("C3_JSON_PREFILL", prefill + '2"}', "R2", True)["format_failure"]
    good = parse("C3_JSON_PREFILL", prefill + '2", "sufficient_evidence": true}', "R2", True)
    assert not good["format_failure"]
    assert good["parsed"]["sufficient_evidence"] is True


def test_c3_rejects_non_boolean_flag():
    prefill = CONTRACTS["C3_JSON_PREFILL"]["prefill"]
    bad = parse("C3_JSON_PREFILL", prefill + '2", "sufficient_evidence": "yes"}', "R2", True)
    assert bad["format_failure"]


def test_c4_parses_prefill_concatenation():
    prefill = CONTRACTS["C4_LINE_PREFILL"]["prefill"]
    assert prefill == "ANSWER:"
    assert not ok("C4_LINE_PREFILL", prefill + " 1")["format_failure"]


# --- universal invariants -------------------------------------------------------


@pytest.mark.parametrize("cid", ALL)
def test_nontermination_always_fails_closed(cid):
    # Even a perfectly formed body is rejected if the model never emitted EOS.
    body = {"C0_JSON_MRS1": '{"answer": "1"}', "C1_BARE": "1", "C2_LINE": "ANSWER: 1",
            "C3_JSON_PREFILL": '{"answer": "1"}', "C4_LINE_PREFILL": "ANSWER: 1"}[cid]
    r = parse(cid, body, "R0", False)
    assert r["format_failure"] and r["nontermination"]


@pytest.mark.parametrize("cid", ALL)
def test_no_contract_leaks_gold_or_repairs_silently(cid):
    r = ok(cid, "complete nonsense that matches nothing")
    assert r["format_failure"]
    assert r["parsed"] == {}
    assert "parse_error" in r


@pytest.mark.parametrize("cid", ALL)
def test_parse_never_raises(cid):
    for junk in ("", "   ", "\x00", "{", "[]", "null", "true", '{"answer": null}', "ANSWER:"):
        parse(cid, junk, "R0", True)


@pytest.mark.parametrize("cid", ALL)
def test_answer_vocabulary_is_closed(cid):
    for bad in ("4", "5", "-1", "None", "maybe"):
        body = {"C0_JSON_MRS1": json.dumps({"answer": bad}),
                "C3_JSON_PREFILL": json.dumps({"answer": bad}),
                "C1_BARE": bad, "C2_LINE": "ANSWER: " + bad,
                "C4_LINE_PREFILL": "ANSWER: " + bad}[cid]
        assert ok(cid, body)["format_failure"], (cid, bad)


# --- prompt construction --------------------------------------------------------


@pytest.mark.parametrize("cid", ALL)
@pytest.mark.parametrize("role", ["R0", "R2", "R3"])
def test_messages_are_system_plus_user_only(cid, role):
    m = messages(cid, role, "PROBLEM BODY")
    assert [x["role"] for x in m] == ["system", "user"]
    assert "PROBLEM BODY" in m[1]["content"]


@pytest.mark.parametrize("cid", ALL)
@pytest.mark.parametrize("role", ["R0", "R2", "R3"])
def test_no_gold_or_answer_key_can_enter_a_prompt(cid, role):
    # The probe never passes gold anywhere; assert the prompt is a pure function of
    # the contract, the role and the public problem text.
    m = messages(cid, role, "PROBLEM BODY")
    blob = m[0]["content"] + m[1]["content"]
    for forbidden in ("gold", "GOLD", "correct_answer", "label"):
        assert forbidden not in blob


def test_role_instructions_match_mrs1_verbatim():
    from experiments.multi_reasoning_scout_v1.prompts import ROLES as MRS1_ROLES
    # R0 is carried byte-identical. R2/R3 drop only the MRS-1 JSON-field sentence,
    # which is the contract-specific part the probe is varying by design.
    assert contracts.ROLES["R0"] == MRS1_ROLES["R0"]
    for role in ("R2", "R3"):
        assert MRS1_ROLES[role].startswith(contracts.ROLES[role])


def test_control_system_prompt_is_mrs1_verbatim():
    from experiments.multi_reasoning_scout_v1.prompts import SYSTEM as MRS1_SYSTEM
    assert CONTRACTS["C0_JSON_MRS1"]["system"] == MRS1_SYSTEM
    assert contracts.MRS1_SYSTEM == MRS1_SYSTEM


def test_prefill_is_item_independent():
    # The same fixed bytes for every item: no item-specific information injected.
    for cid in ALL:
        p = CONTRACTS[cid]["prefill"]
        assert isinstance(p, str)
        assert "{" not in p.replace('{"answer": "', "")
