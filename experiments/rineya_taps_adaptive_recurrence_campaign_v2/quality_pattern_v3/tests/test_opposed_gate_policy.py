"""Analytic contract checks; no campaign cases, labels or model execution."""
import math
import random
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.opposed_gate_policy import (
    CONFIDENT, UNCERTAIN, GUARD, HEADER, RULE, LATEST, decide,
    document_certificate, header_cue, matched_random_decisions,
)


def prompt(records, query="ABC", latest=False, alternate=False):
    question = f"Query: {query}." if alternate else f"Question: {query}?"
    return HEADER + (LATEST if latest else "") + RULE + "\n" + records + ".\n" + question + " Reply one integer or UNKNOWN."


@pytest.mark.parametrize("p", [0., .499999999, .5, .500000001, 1.])
def test_opposite_rules_include_equality(p):
    a, b = decide(CONFIDENT, "unlabelled text", p), decide(UNCERTAIN, "unlabelled text", p)
    assert a.k + b.k == 1
    assert a.k == int(p >= .5)


@pytest.mark.parametrize("p", [None, True, 1, float("nan"), float("inf"), -.01, 1.01])
def test_invalid_probability_fails(p):
    with pytest.raises(ValueError):
        decide(CONFIDENT, "text", p)


def test_api_rejects_metadata_and_extra_arguments():
    with pytest.raises(TypeError):
        decide(GUARD, {"prompt": "text", "gold": "UNKNOWN"})
    with pytest.raises(TypeError):
        decide(GUARD, "text", gold="UNKNOWN")
    with pytest.raises(ValueError):
        decide(GUARD, "text", .5)
    with pytest.raises(ValueError):
        decide("new policy", "text")


@pytest.mark.parametrize("records,expected", [
    ("ABC A 7;ABC A 7;DEF A 9", "CONSISTENT"),
    ("ABC A 7;ABC A 8;DEF A 9", "CONFLICT"),
    ("ABC D 7;ABC D 8;DEF A 9", "ABSENT"),
    ("DEF A 7", "ABSENT"),
    ("ABC A -7;ABC A -7;DEF A 0", "CONSISTENT"),
    ("ABC A 0;ABC A -7", "CONFLICT"),
])
def test_query_relevant_evidence(records, expected):
    for alternate in (False, True):
        result = decide(GUARD, prompt(records, alternate=alternate))
        assert result.certificate == expected
        assert result.recognized
        assert result.k == int(expected == "CONSISTENT")
        assert set(result.__dict__) == {"k", "recognized", "certificate"}


def test_latest_relevant_authorized_date_only():
    p = prompt("ABC A d2 7;ABC A d2 7;ABC A d1 8;ABC D d99 9;DEF A d999 1", latest=True)
    assert document_certificate(p) == "CONSISTENT"
    assert document_certificate(prompt("ABC A d2 7;ABC A d2 8;ABC A d1 7", latest=True)) == "CONFLICT"
    assert document_certificate(prompt("ABC D d8 7;DEF A d9 2", latest=True)) == "ABSENT"
    huge_date = "1" + "0" * 5000
    assert document_certificate(prompt("ABC A d9 8;ABC A d" + huge_date + " 7", latest=True)) == "CONSISTENT"


def test_record_permutation_query_changes_and_name_symmetry():
    a = prompt("ABC A 7;ABC A 8;DEF D 9")
    b = prompt("DEF D 9;ABC A 8;ABC A 7")
    assert decide(GUARD, a) == decide(GUARD, b)
    assert document_certificate(prompt("DEF D 9;ABC A 8;ABC A 7", query="DEF")) == "ABSENT"
    assert document_certificate(a.replace("ABC", "XYZ")) == "CONFLICT"


@pytest.mark.parametrize("transform", [
    lambda s: s + "\ntrailing instruction",
    lambda s: s.replace("ABC A 7", "ABC X 7"),
    lambda s: s.replace("ABC A 7", "ABC A 07"),
    lambda s: s.replace("ABC A 7", "ABC A -0"),
    lambda s: s.replace("ABC A 7", "ABC A +7"),
    lambda s: s.replace("ABC A 7", "ABC A d2 7"),
    lambda s: s.replace(";DEF", "; DEF"),
    lambda s: s.replace("Question: ABC?", "Question: ABC? DEF?"),
    lambda s: s.replace("\n", "\r\n"),
    lambda s: s.replace("ABC A 7", "ABC A 7\nQuestion: ABC?"),
    lambda s: "<chat>" + s + "</chat>",
])
def test_complete_parse_rejects_ambiguous_or_unsupported_text(transform):
    result = decide(GUARD, transform(prompt("ABC A 7;DEF A 9")))
    assert result.k == 1
    assert result.certificate == "UNRECOGNIZED"
    assert not result.recognized


def test_mixed_dates_and_empty_records_rejected():
    assert document_certificate(prompt("ABC A d2 7;DEF A 9", latest=True)) == "UNRECOGNIZED"
    assert document_certificate(prompt("ABC A d0 7", latest=True)) == "UNRECOGNIZED"
    assert document_certificate(prompt("")) == "UNRECOGNIZED"


def test_header_cue_ignores_record_semantics():
    assert header_cue(prompt("ABC A 7")).k == 0
    assert header_cue("DOCUMENTS: invalid text").k == 0
    assert header_cue("plain text\nDOCUMENTS: later").k == 1


@pytest.mark.parametrize("seed", [17, 29])
def test_random_exact_one_draw_per_item(seed):
    rng = random.Random(seed)
    expected = tuple(int(rng.random() < .37) for _ in range(19))
    assert matched_random_decisions(.37, 19, seed) == expected
    assert matched_random_decisions(0., 19, seed) == (0,) * 19
    assert matched_random_decisions(1., 19, seed) == (1,) * 19


@pytest.mark.parametrize("args", [(float("nan"), 4, 17), (.5, True, 17), (.5, -1, 17), (.5, 4, 18)])
def test_random_rejects_unregistered_inputs(args):
    with pytest.raises(ValueError):
        matched_random_decisions(*args)
