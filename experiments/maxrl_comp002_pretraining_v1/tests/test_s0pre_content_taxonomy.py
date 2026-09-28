"""Amendment 1 §8.4 a-d: content repairs, strict-failure subtypes and RS classes.

Data-backed tests read only DEV and guard_dev canonical/oracle files (plus the
manifest that binds them) and skip when the dataset is absent.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from itertools import combinations
from pathlib import Path

import pytest

from experiments.maxrl_comp002_pretraining_v1 import content_scoring, failure_taxonomy
from experiments.maxrl_comp002_pretraining_v1.content_scoring import (
    MULTIPLE, UNPARSEABLE, is_expression, is_int_list, score, score_legacy, score_row)
from experiments.maxrl_comp002_pretraining_v1.failure_taxonomy import (
    KEY_VARIANTS, RS_CLASSES, SUBTYPES, classify_rs, subtype_strict_failure)
from experiments.maxrl_composable_skills_v2.verifier import concrete
from experiments.maxrl_verifiable_data_v1.specs import FAMILIES

PACKAGE = Path(content_scoring.__file__).resolve().parent
R_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "lookup": {"requested": ["x"], "links": {"x": 1}}}
IMPOSSIBLE = {"initial": [1, 2, 3, 4, 5, 6], "lookup": {"requested": ["x"], "links": {"x": "y", "y": "x"}}}
S_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "selection": {"k": 2, "target": 11}}
# Value triples {0,1,2}=5 and {3,4,5}=7; resolved {0,1,3,4}; pass = one 5 and one 7, both resolved.
RS_SPEC = {"initial": [5, 5, 5, 7, 7, 7],
           "lookup": {"requested": ["a", "b", "c", "d"],
                      "links": {"a": 0, "b": 1, "c": 3, "d": 4, "e": 2, "f": 5}},
           "selection": {"k": 2, "target": 12}}
RS_CYCLE = {"initial": [5, 5, 5, 7, 7, 7], "lookup": {"requested": ["a"], "links": {"a": "b", "b": "a"}},
            "selection": {"k": 2, "target": 12}}
# validate() admits k up to 5: with k = 4 the full resolved set {0,1,3,4} (5+5+7+7 = 24) is also a pass.
RS_K4 = dict(RS_SPEC, selection={"k": 4, "target": 24})
# Targets that only a malformed lookup-blind list reaches: [2,5,5] sums 5+7+7 = 19; [2,2] sums 5+5 = 10.
RS_T19 = dict(RS_SPEC, selection={"k": 2, "target": 19})
RS_T10 = dict(RS_SPEC, selection={"k": 2, "target": 10})
SUBSET = {"values": [1, 2, 3, 4, 5], "k": 2, "target": 9}
COUNTDOWN = {"numbers": [3, 4, 5], "target": 27}


def comp_row(spec):
    return {"id": "fixture", "kind": "composition", "data_source": "rineya_maxrl_comp_v2",
            "verifier": {"version": "2.0.0", "kind": "composition", "spec": spec}}


def legacy_row(family, problem):
    return {"id": "fixture", "kind": "legacy", "data_source": "rineya_maxrl_comp_v2/legacy/" + family,
            "verifier": {"version": "2.0.0", "kind": "legacy", "spec": {"family": family, "problem": problem}}}


def check(row, response, category, strict, content):
    result = score_row(row, response)
    assert result == {"strict_pass": strict, "content_pass": content, "normalization": category,
                      "format_only_failure": content and not strict}, (response, result)


# ---- §8.4(b): the rejection clause -------------------------------------------------------------

def test_amendment_contract_fixtures():
    assert score(R_SPEC, '{"answer":[1],"alt":[2]}') == {
        "strict_pass": False, "content_pass": False, "normalization": MULTIPLE, "format_only_failure": False}
    assert score(R_SPEC, '{"answer":[1],"reason":"ok"}') == {
        "strict_pass": False, "content_pass": True, "normalization": "extra_object_fields+answer_object",
        "format_only_failure": True}


@pytest.mark.parametrize("spec,response,category,strict,content", [
    (R_SPEC, '{"answer":[1]}', "answer_object", True, True),
    (R_SPEC, "[1]", "bare_list", False, True),
    (IMPOSSIBLE, "UNKNOWN", "bare_unknown", False, True),
    (R_SPEC, '```json\n{"answer":[1]}\n```', "code_fence+answer_object", False, True),
    (R_SPEC, '{"answer":[1]}.', "trailing_period+answer_object", False, True),
    (R_SPEC, '{"answer":[1],"reason":"ok"}', "extra_object_fields+answer_object", False, True),
    (R_SPEC, '{"answer":[1],"alt":[2]}', MULTIPLE, False, False),
    (R_SPEC, "Here is the answer: [1]", UNPARSEABLE, False, False),
])
def test_one_fixture_per_composition_normalization(spec, response, category, strict, content):
    check(comp_row(spec), response, category, strict, content)


@pytest.mark.parametrize("response", [
    '{"answer":[1],"why":{"steps":[[0,1]]}}',      # int list at depth
    '{"answer":[1],"alt":"UNKNOWN"}',                # UNKNOWN is a composition candidate
    '{"answer":[1],"notes":["see",{"alt":[-3]}]}',  # negative ints still form an int list
    '```json\n{"answer":[1],"alt":[1]}.\n```',       # wrappers do not bypass the rule
    '{"answer":"UNKNOWN","alt":[1]}',
])
def test_multiple_candidate_rule_rejects(response):
    result = score(R_SPEC, response)
    assert result["normalization"] == MULTIPLE and not result["content_pass"]


@pytest.mark.parametrize("response", [
    '{"answer":[1],"alt":[true]}',      # bools are not ints
    '{"answer":[1],"alt":[]}',          # empty list is not a candidate
    '{"answer":[1],"count":2}',         # a scalar int is not a list
    '{"answer":[1],"alt":"unknown"}',   # case-sensitive
    '{"answer":[1],"UNKNOWN":true}',    # field names are not values
    '{"answer":[1],"alt":["x"],"p":[1.0]}',
])
def test_non_candidate_extra_fields_remain_allowed(response):
    result = score(R_SPEC, response)
    assert result["content_pass"] and result["normalization"] == "extra_object_fields+answer_object"


# ---- C4: one definition of "int list" (F10) ----------------------------------------------------

class _Int(int):
    pass


@pytest.mark.parametrize("value,expected", [
    ([1], True), ([-3, 0, 7], True), ([2, 2], True),
    ([], False),                                    # non-empty (C4): [] is never an int list
    ([True], False), ([1, False], False),           # bools are not ints
    ([1.0], False), (["1"], False), ([None], False), ([[1]], False),
    ([_Int(1)], False),                             # element type must be exactly int
    ((1, 2), False), ({1: 2}, False), (1, False), ("[1]", False), (None, False),
])
def test_int_list_definition(value, expected):
    assert is_int_list(value) is expected


def _int_list_definitions(root: Path) -> list[tuple[str, str]]:
    """Every function (or lambda bound to a name) in the package's modules whose name says int list."""
    named = re.compile(r"int_?list", re.IGNORECASE)
    found = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and named.search(node.name):
                found.append((path.name, node.name))
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                found.extend((path.name, t.id) for t in node.targets
                             if isinstance(t, ast.Name) and named.search(t.id))
    return found


def test_int_list_has_one_definition():
    """C4: content_scoring.is_int_list is the package's only int-list predicate; others import it."""
    assert _int_list_definitions(PACKAGE) == [("content_scoring.py", "is_int_list")]
    assert failure_taxonomy.is_int_list is content_scoring.is_int_list
    # The one definition decides every place the addendum says "int list".
    assert score(R_SPEC, '{"answer":[1],"alt":[]}')["content_pass"]                 # A1: [] is no candidate
    assert classify_rs(RS_SPEC, '{"answer":[]}') == "NOT_INT_LIST"                  # A5: not RESOLVED_WRONG
    assert subtype_strict_failure(comp_row(R_SPEC), '{"answer":[]}', "stop") == "OTHER"  # A4: [] is ill typed


# ---- §8.4(a): the legacy content path -----------------------------------------------------------

@pytest.mark.parametrize("family,problem,response,category,strict,content", [
    ("subset_sum", SUBSET, '{"answer":[3,4]}', "answer_object", True, True),
    ("subset_sum", SUBSET, "[4,3]", "bare_list", False, True),
    ("subset_sum", SUBSET, '```json\n{"answer":[3,4]}\n```', "code_fence+answer_object", False, True),
    ("subset_sum", SUBSET, '{"answer":[3,4]}.', "trailing_period+answer_object", False, True),
    ("subset_sum", SUBSET, '{"answer":[3,4],"note":"ok"}', "extra_object_fields+answer_object", False, True),
    ("subset_sum", SUBSET, '{"answer":[3,4],"expr":"4+5"}', "extra_object_fields+answer_object", False, True),
    ("subset_sum", SUBSET, '{"answer":[3,4],"alt":[0,1]}', MULTIPLE, False, False),
    ("subset_sum", SUBSET, "UNKNOWN", UNPARSEABLE, False, False),
    ("subset_sum", SUBSET, '{"answer":[0,1]}', "answer_object", False, False),
    ("countdown", COUNTDOWN, '{"answer":"(4+5)*3"}', "answer_object", True, True),
    ("countdown", COUNTDOWN, '```\n{"answer":"(4+5)*3"}\n```', "code_fence+answer_object", False, True),
    ("countdown", COUNTDOWN, '{"answer":"(4+5)*3"}.', "trailing_period+answer_object", False, True),
    ("countdown", COUNTDOWN, '{"answer":"(4+5)*3","note":"4 plus 5 is 9"}',
     "extra_object_fields+answer_object", False, True),
    ("countdown", COUNTDOWN, '{"answer":"(4+5)*3","used":[3,4,5]}',
     "extra_object_fields+answer_object", False, True),
    ("countdown", COUNTDOWN, '{"answer":"(4+5)*3","alt":"3*(5+4)"}', MULTIPLE, False, False),
    ("countdown", COUNTDOWN, '"(4+5)*3"', UNPARSEABLE, False, False),
    ("countdown", COUNTDOWN, "(4+5)*3", UNPARSEABLE, False, False),
    ("countdown", COUNTDOWN, '["(4+5)*3"]', UNPARSEABLE, False, False),
])
def test_one_fixture_per_legacy_normalization(family, problem, response, category, strict, content):
    check(legacy_row(family, problem), response, category, strict, content)
    assert score_legacy(family, problem, response) == score_row(legacy_row(family, problem), response)


@pytest.mark.parametrize("text,expected", [
    ("(4+5)*3", True), ("8/2*3", True), ("100-1", True),
    ("45", False), ("4+5=9", False), ("-4+5", False), ("4**2", False), ("101+1", False),
    ("x+1", False), ("4+True", False), ("", False), ("4 plus 5", False),
])
def test_countdown_candidate_shape(text, expected):
    assert is_expression(text) is expected


def test_score_row_refuses_malformed_rows():
    row = comp_row(R_SPEC)
    row["verifier"] = dict(row["verifier"], kind="legacy")
    with pytest.raises(ValueError, match="kind mismatch"):
        score_row(row, "[1]")
    odd = dict(comp_row(R_SPEC), kind="other")
    odd["verifier"] = dict(odd["verifier"], kind="other")
    with pytest.raises(ValueError, match="unknown row kind"):
        score_row(odd, "[1]")
    with pytest.raises(ValueError, match="legacy"):
        score_row(legacy_row("chess", SUBSET), "[1]")
    with pytest.raises(ValueError, match="unknown legacy family"):
        score_legacy("chess", SUBSET, "[1]")


def test_reward_assertion_is_live(monkeypatch):
    monkeypatch.setattr(content_scoring, "strict_reward", lambda row, response: 0)
    with pytest.raises(AssertionError, match="corpus.reward"):
        score_row(comp_row(R_SPEC), '{"answer":[1]}')


def test_legacy_strict_implies_content_is_asserted(monkeypatch):
    raw = '{"answer": [3, 4]}'
    monkeypatch.setattr(content_scoring, "legacy_strict_score", lambda family, problem, text: int(text == raw))
    with pytest.raises(AssertionError, match="strict verifier success lost"):
        score_legacy("subset_sum", SUBSET, raw)


# ---- §8.4(c): strict-failure subtyper ------------------------------------------------------------

SUBTYPE_FIXTURES = [
    (comp_row(R_SPEC), '{"answer":[1', "length", "LENGTH_CAP"),
    (comp_row(R_SPEC), "   ", "length", "LENGTH_CAP"),      # blank and capped: LENGTH_CAP precedes EMPTY
    (comp_row(R_SPEC), "   \n", "stop", "EMPTY"),
    (comp_row(R_SPEC), "", None, "EMPTY"),
    (comp_row(R_SPEC), '{"answer":"[1]"}', "stop", "STRING_ENCODED_ANSWER"),
    (comp_row(R_SPEC), '"{\\"answer\\":[1]}"', "stop", "STRING_ENCODED_ANSWER"),
    (legacy_row("subset_sum", SUBSET), '{"answer":"[3,4]"}', "stop", "STRING_ENCODED_ANSWER"),
    (comp_row(R_SPEC), '{"answer":[1.0]}', "stop", "NON_INT_ELEMENTS"),
    (comp_row(R_SPEC), '{"answer":[true]}', "stop", "NON_INT_ELEMENTS"),
    (comp_row(R_SPEC), '{"answer":["x","zz"]}', "stop", "NON_INT_ELEMENTS"),
    (comp_row(S_SPEC), '{"answer":["x"]}', "stop", "NON_INT_ELEMENTS"),
    (legacy_row("subset_sum", SUBSET), '{"answer":["3","4"]}', "stop", "NON_INT_ELEMENTS"),
    (comp_row(R_SPEC), '{"answer":["x"]}', "stop", "ALIAS_STRINGS"),
    (comp_row(RS_SPEC), '[0,"c"]', "stop", "ALIAS_STRINGS"),
    (comp_row(R_SPEC), '{"Answer":[1]}', "stop", "KEY_VARIANT"),
    (comp_row(R_SPEC), '{"answers":[1]}', "stop", "KEY_VARIANT"),     # one fixture per registered variant
    (comp_row(R_SPEC), '{"output":[1]}', "stop", "KEY_VARIANT"),
    (comp_row(R_SPEC), '```json\n{"final_answer":[1]}\n```', "stop", "KEY_VARIANT"),
    (legacy_row("countdown", COUNTDOWN), '{"result":"(4+5)*3"}', "stop", "KEY_VARIANT"),
    (comp_row(R_SPEC), 'Here is the answer: {"answer":[1]}', "stop", "PROSE_WRAPPED"),
    (comp_row(R_SPEC), '{"answer":[1]} because x links to 1', "stop", "PROSE_WRAPPED"),
    (comp_row(R_SPEC), '```python\n[1]\n```', "stop", "PROSE_WRAPPED"),
    (comp_row(R_SPEC), '{"answer":[1]}\n{"answer":[2]}', "stop", "MULTI_OBJECT"),
    (comp_row(R_SPEC), "[1] [2]", "stop", "MULTI_OBJECT"),
    (comp_row(R_SPEC), '{"answer":null}', "stop", "OTHER"),
    (comp_row(R_SPEC), "I cannot tell.", "stop", "OTHER"),
    (comp_row(R_SPEC), '{"answer":[1],"answer":[2]}', "stop", "OTHER"),
    (legacy_row("countdown", COUNTDOWN), '{"answer":[3,4,5]}', "stop", "OTHER"),
]


@pytest.mark.parametrize("row,response,stop,expected", SUBTYPE_FIXTURES)
def test_one_fixture_per_subtype(row, response, stop, expected):
    assert subtype_strict_failure(row, response, stop) == expected


def test_every_subtype_has_a_positive_control():
    assert {expected for *_, expected in SUBTYPE_FIXTURES} == set(SUBTYPES)


def test_registered_key_variants():
    """A4 type 6 lists exactly these keys; each has a fixture above."""
    assert KEY_VARIANTS == {"answers", "result", "output", "final_answer"}
    shown = set()
    for _, response, _, expected in SUBTYPE_FIXTURES:
        if expected == "KEY_VARIANT":
            text = response.removeprefix("```json\n").removesuffix("\n```")
            shown.update(json.loads(text))
    assert KEY_VARIANTS <= shown


NO_SUBTYPE = [
    (comp_row(R_SPEC), '{"answer":[1]}', "length"),               # strict pass
    (comp_row(R_SPEC), "[1]", "stop"),                            # format-only
    (comp_row(R_SPEC), '{"answer":[2]}', "length"),               # well-typed task failure
    (comp_row(R_SPEC), '{"answer":"UNKNOWN"}', "stop"),
    (comp_row(R_SPEC), '{"answer":[1],"alt":[2]}', "stop"),      # named by its normalization
    (legacy_row("countdown", COUNTDOWN), '{"answer":"4+5"}', "stop"),
    (legacy_row("subset_sum", SUBSET), '{"answer":[0,1]}', "stop"),
]


@pytest.mark.parametrize("row,response,stop", NO_SUBTYPE)
def test_strict_failures_inside_the_normalisations_get_no_subtype(row, response, stop):
    assert subtype_strict_failure(row, response, stop) is None


# C6: a multiple_candidates_in_object row is named by that category and gets no subtype, whatever
# its answer's type and stop reason. Before C6 each of these fell through to the type in the comment.
MULTIPLE_ILL_TYPED = [
    (comp_row(R_SPEC), '{"answer":null,"alt":[2]}', "stop"),                          # OTHER
    (comp_row(R_SPEC), '{"answer":null,"alt":[2]}', "length"),                        # LENGTH_CAP
    (comp_row(R_SPEC), '{"answer":"[1]","alt":[2]}', "stop"),                         # STRING_ENCODED_ANSWER
    (comp_row(R_SPEC), '{"answer":[1.0],"alt":"UNKNOWN"}', "stop"),                   # NON_INT_ELEMENTS
    (comp_row(R_SPEC), '{"answer":["x"],"why":{"steps":[1]}}', "stop"),               # ALIAS_STRINGS
    (comp_row(R_SPEC), '{"answer":[],"alt":[1]}', "stop"),                            # OTHER ([] is ill typed)
    (comp_row(R_SPEC), '```json\n{"answer":null,"alt":[2]}\n```', "stop"),            # OTHER
    (legacy_row("countdown", COUNTDOWN), '{"answer":[3,4,5],"alt":"(4+5)*3"}', "stop"),  # OTHER
    (legacy_row("subset_sum", SUBSET), '{"answer":"[3,4]","alt":[3,4]}', "stop"),     # STRING_ENCODED_ANSWER
]


@pytest.mark.parametrize("row,response,stop", MULTIPLE_ILL_TYPED)
def test_multiple_candidate_objects_get_no_subtype_whatever_the_answer(row, response, stop):
    assert score_row(row, response)["normalization"] == MULTIPLE
    assert subtype_strict_failure(row, response, stop) is None


def test_subtype_scope_is_the_registered_rule():
    """A4 with C6: a row a normalization names gets no type; every unparseable strict failure gets one."""
    corpus = ([(row, response, stop) for row, response, stop, _ in SUBTYPE_FIXTURES] + NO_SUBTYPE
              + MULTIPLE_ILL_TYPED + [(comp_row(spec), response, "stop") for spec, response, _ in RS_FIXTURES])
    kinds = set()
    for row, response, stop in corpus:
        result = score_row(row, response)
        subtype = subtype_strict_failure(row, response, stop)
        if result["strict_pass"] or result["content_pass"] or result["normalization"] == MULTIPLE:
            assert subtype is None, response
            kinds.add("named")
        elif result["normalization"] == UNPARSEABLE:
            assert subtype in SUBTYPES, response
            kinds.add("typed")
        else:  # a single parsed candidate that fails: None exactly when it is well typed
            assert (subtype is None) == failure_taxonomy.well_typed(
                content_scoring.parse_candidate(response, content_scoring.row_family(row))[0],
                content_scoring.row_family(row)), response
            kinds.add("parsed")
    assert kinds == {"named", "typed", "parsed"}


def test_subtyper_input_types():
    with pytest.raises(ValueError):
        subtype_strict_failure(comp_row(R_SPEC), None, "stop")
    with pytest.raises(ValueError):
        subtype_strict_failure(comp_row(R_SPEC), "[2]", 7)


# ---- §8.4(d): RS response classifier -------------------------------------------------------------

RS_FIXTURES = [
    (RS_SPEC, "Here: [0,3]", "UNPARSEABLE"),
    (RS_SPEC, '{"answer":[0,3],"alt":[1,4]}', "UNPARSEABLE"),
    (RS_SPEC, '{"answer":"UNKNOWN","alt":[1]}', "UNPARSEABLE"),   # multiple precedes UNKNOWN_ANSWER
    (RS_SPEC, "UNKNOWN", "UNKNOWN_ANSWER"),
    (RS_CYCLE, "UNKNOWN", "UNKNOWN_ANSWER"),
    (RS_SPEC, '{"answer":["a","c"]}', "NOT_INT_LIST"),
    (RS_SPEC, '{"answer":[]}', "NOT_INT_LIST"),
    (RS_SPEC, '```json\n[0,3]\n```', "RS_PASS"),
    (RS_SPEC, '{"answer":[4,1]}', "RS_PASS"),
    (RS_K4, '{"answer":[0,1,3,4]}', "RS_PASS"),                   # RS_PASS precedes FULL_RESOLVED_SET
    (RS_SPEC, '{"answer":[4,1,3,0]}', "FULL_RESOLVED_SET"),
    (RS_SPEC, "[0,1,3,4].", "FULL_RESOLVED_SET"),
    (RS_SPEC, "[0,1]", "RESOLVED_WRONG"),
    (RS_SPEC, '{"answer":[3,3]}', "RESOLVED_WRONG"),
    (RS_SPEC, '{"answer":[2,5]}', "R_BLIND_S_VALID"),
    (RS_SPEC, '{"answer":[0,5]}', "R_BLIND_S_VALID"),
    (RS_CYCLE, "[0,3]", "R_BLIND_S_VALID"),
    (RS_T10, '{"answer":[0,2]}', "R_BLIND_S_VALID"),              # the T10 spec admits a blind pair
    (RS_SPEC, '{"answer":[2,4,5]}', "OTHER_WRONG"),
    (RS_SPEC, '{"answer":[-1,0]}', "OTHER_WRONG"),   # would sum to target by wrap-around
    (RS_SPEC, '{"answer":[2,6]}', "OTHER_WRONG"),
    # R_BLIND_S_VALID's defining conditions, one fixture each (TA-08):
    (RS_SPEC, '{"answer":[0,2]}', "OTHER_WRONG"),    # length k, distinct, in range, not in R; sums 10 != 12
    (RS_SPEC, '{"answer":[2,5,5]}', "OTHER_WRONG"),  # three entries for k = 2
    (RS_T19, '{"answer":[2,5,5]}', "OTHER_WRONG"),   # k distinct indices and sums to 19, but length 3 != k
    (RS_T10, '{"answer":[2,2]}', "OTHER_WRONG"),     # length k and sums to 10, but one distinct index
]


@pytest.mark.parametrize("spec,response,expected", RS_FIXTURES)
def test_one_fixture_per_rs_class(spec, response, expected):
    assert classify_rs(spec, response) == expected


def test_every_rs_class_has_a_positive_control():
    assert {expected for *_, expected in RS_FIXTURES} == set(RS_CLASSES)


def test_rs_candidate_is_the_h3b_candidate():
    """C5: the first three RS classes are exactly the responses whose H3b candidate is None."""
    from experiments.maxrl_comp002_pretraining_v1.stats_exact import h3b_candidate
    for spec, response, expected in RS_FIXTURES:
        unscored = expected in ("UNPARSEABLE", "UNKNOWN_ANSWER", "NOT_INT_LIST")
        assert (h3b_candidate(response) is None) is unscored, response


@pytest.mark.parametrize("spec", [R_SPEC, S_SPEC, dict(RS_SPEC, updates={"operations": [["add", 0, 1]], "cutoff": 0})])
def test_classifier_refuses_non_rs_specs(spec):
    with pytest.raises(ValueError, match="RS spec"):
        classify_rs(spec, "[0,3]")


# ---- data-backed canary (DEV core R/S/RS; guard_dev legacy) --------------------------------------

DATA = Path(os.environ.get("COMP_RS_DATA", r"C:\Users\clclc\BusinessProject\Rineya-Dataset\data"))
MANIFEST_SHA = "86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44"
DATA_FILES = ("canonical/dev.jsonl", "oracles/dev.jsonl", "canonical/guard_dev.jsonl", "oracles/guard_dev.jsonl")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


@pytest.fixture(scope="module")
def data():
    if not all((DATA / name).is_file() for name in ("manifest.json",) + DATA_FILES):
        pytest.skip("dataset absent")
    assert _sha(DATA / "manifest.json") == MANIFEST_SHA, "dataset manifest drift"
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name in DATA_FILES:
        assert _sha(DATA / name) == manifest["files"][name]["sha256"], "file drift: " + name
    dev = {row["id"]: row for row in _rows(DATA / "canonical/dev.jsonl")
           if row["tag"] == "core" and "".join(row["skills"]) in ("R", "S", "RS")}
    dev_refs = {o["id"]: o["references"] for o in _rows(DATA / "oracles/dev.jsonl") if o["id"] in dev}
    guard = {row["id"]: row for row in _rows(DATA / "canonical/guard_dev.jsonl")}
    guard_oracles = _rows(DATA / "oracles/guard_dev.jsonl")
    assert set(dev_refs) == set(dev) and {o["id"] for o in guard_oracles} == set(guard)
    return dev, dev_refs, guard, guard_oracles


def _wrappers(ref: str, answer, family: str | None = None) -> dict[str, str]:
    out = {"code_fence+answer_object": "```json\n" + ref + "\n```",
           "trailing_period+answer_object": ref + ".",
           "extra_object_fields+answer_object": json.dumps({"answer": answer, "note": "ok"})}
    if type(answer) is list:
        out["bare_list"] = json.dumps(answer)
    elif family is None and answer == "UNKNOWN":
        out["bare_unknown"] = "UNKNOWN"
    return out


def test_dev_core_references_score_strict_and_every_wrapper_scores_content(data):
    dev, dev_refs, _, _ = data
    per_signature = {"R": 0, "S": 0, "RS": 0}
    checked = unknown = 0
    for row_id, row in dev.items():
        per_signature["".join(row["skills"])] += 1
        for ref in dev_refs[row_id]:
            answer = json.loads(ref)["answer"]
            unknown += answer == "UNKNOWN"
            check(row, ref, "answer_object", True, True)
            for category, wrapped in _wrappers(ref, answer).items():
                check(row, wrapped, category, False, True)
                assert subtype_strict_failure(row, wrapped, "stop") is None
            check(row, json.dumps({"answer": answer, "alt": answer}), MULTIPLE, False, False)
            checked += 1
    assert per_signature == {"R": 250, "S": 250, "RS": 250}
    assert unknown == 0  # §8.5: core rows carry no UNKNOWN reference, so bare UNKNOWN never applies
    assert checked >= 750


def test_dev_core_rs_classes(data):
    dev, dev_refs, _, _ = data
    rs = [row for row in dev.values() if row["skills"] == ["R", "S"]]
    blind = 0
    for row in rs:
        spec = row["verifier"]["spec"]
        for ref in dev_refs[row["id"]]:
            assert classify_rs(spec, ref) == "RS_PASS"
        resolved, values = concrete(spec)
        full = sorted(resolved)
        assert classify_rs(spec, json.dumps({"answer": full})) == "FULL_RESOLVED_SET"
        assert classify_rs(spec, "```json\n" + json.dumps(full) + "\n```") == "FULL_RESOLVED_SET"
        k, target = spec["selection"]["k"], spec["selection"]["target"]
        subset = next((list(c) for c in combinations(range(len(values)), k)
                       if sum(values[i] for i in c) == target and not set(c) <= resolved), None)
        if subset is not None:
            blind += 1
            assert classify_rs(spec, json.dumps({"answer": subset})) == "R_BLIND_S_VALID"
            assert classify_rs(spec, json.dumps(subset[::-1]) + ".") == "R_BLIND_S_VALID"
    assert len(rs) == 250 and blind > len(rs) // 2


def test_guard_dev_legacy_references_for_all_families(data):
    _, _, guard, guard_oracles = data
    seen = {family: 0 for family in FAMILIES}
    for oracle in guard_oracles:
        row = guard[oracle["id"]]
        family = row["verifier"]["spec"]["family"]
        for ref in oracle["references"]:
            answer = json.loads(ref)["answer"]
            check(row, ref, "answer_object", True, True)
            wrapped = _wrappers(ref, answer, family)
            assert ("bare_list" in wrapped) == (family != "countdown")
            for category, text in wrapped.items():
                check(row, text, category, False, True)
                assert subtype_strict_failure(row, text, "stop") is None
            check(row, json.dumps({"answer": answer, "alt": answer}), MULTIPLE, False, False)
            if family == "countdown":
                check(row, json.dumps(answer), UNPARSEABLE, False, False)
            seen[family] += 1
        negative = score_row(row, oracle["negative"])
        assert not negative["strict_pass"] and not negative["content_pass"]
    assert all(count > 0 for count in seen.values()), seen
