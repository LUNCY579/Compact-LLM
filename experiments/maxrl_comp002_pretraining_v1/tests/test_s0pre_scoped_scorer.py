"""Amendment 1 section 8.0 and clarification C8: the hash-bound scoped scorer and its scope receipts.

Synthetic fixtures run everywhere. Data-backed tests read only the manifest and the DEV, CALIBRATION
and guard_dev canonical/oracle files (and policy/dev.jsonl through the scope_receipt CLI), each checked
by sha against manifest 86ac8f0d, and skip when the dataset is absent (COMP_RS_DATA overrides the
default root). Every refusal of a sealed or challenge role is exercised on an empty or synthetic tmp
directory, never on the dataset.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from experiments.maxrl_comp002_pretraining_v1 import scope_receipt as sr
from experiments.maxrl_comp002_pretraining_v1 import scoped_scorer as ss
from experiments.maxrl_comp002_pretraining_v1.preflight import MANIFEST, digest
from experiments.maxrl_composable_skills_v2 import behaviour
from experiments.maxrl_composable_skills_v2.corpus import reward

PACKAGE = Path(__file__).resolve().parents[1]
META = {"model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64, "decoding_sha256": "c" * 64}

R_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "lookup": {"requested": ["x"], "links": {"x": 1}}}
S_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "selection": {"k": 2, "target": 11}}
# Value triples {0,1,2}=5 and {3,4,5}=7; resolved {0,1,3,4}; pass = one resolved 5 and one resolved 7.
RS_SPEC = {"initial": [5, 5, 5, 7, 7, 7],
           "lookup": {"requested": ["a", "b", "c", "d"],
                      "links": {"a": 0, "b": 1, "c": 3, "d": 4, "e": 2, "f": 5}},
           "selection": {"k": 2, "target": 12}}
SUBSET = {"values": [1, 2, 3, 4, 5], "k": 2, "target": 9}
COUNTDOWN = {"numbers": [3, 4, 5], "target": 27}
CORPUS = "experiments/maxrl_composable_skills_v2/corpus.py"


def comp_row(ident, spec, sig, tag="core", parent="w1", role="dev"):
    return {"id": ident, "parent_id": parent, "role": role, "kind": "composition", "tag": tag,
            "skills": list(sig), "data_source": "rineya_maxrl_comp_v2",
            "prompt_hash": digest(["prompt", ident]),
            "verifier": {"version": "2.0.0", "kind": "composition", "spec": spec}}


def legacy_row(ident, family, problem, parent):
    return {"id": ident, "parent_id": parent, "role": "guard_dev", "kind": "legacy", "tag": "preservation",
            "skills": ["legacy:" + family], "data_source": "rineya_maxrl_comp_v2/legacy/" + family,
            "prompt_hash": digest(["prompt", ident]),
            "verifier": {"version": "2.0.0", "kind": "legacy", "spec": {"family": family, "problem": problem}}}


def synthetic_bank():
    rows = [comp_row("r1", R_SPEC, "R"), comp_row("s1", S_SPEC, "S"), comp_row("rs1", RS_SPEC, "RS"),
            comp_row("rs2", RS_SPEC, "RS", parent="w2"), comp_row("rsb", RS_SPEC, "RS", tag="boundary", parent="w2")]
    return ss.Bank(rows, role="dev", bank_sha256=digest(rows))


def legacy_bank():
    rows = [legacy_row("g1", "subset_sum", SUBSET, "p1"), legacy_row("g2", "countdown", COUNTDOWN, "p2")]
    return ss.Bank(rows, role="guard_dev", bank_sha256=digest(rows))


SYNTHETIC = {  # id -> (response, stop_reason, generated_tokens)
    "r1": ('{"answer":[1]}', "eos", 6),
    "s1": ('```json\n{"answer":[4,5]}\n```', "eos", 12),
    "rs1": ('{"answer":[0,1,3,4]}', "eos", 10),
    "rs2": ('The answer is {"answer": [0, 3]}', "eos", 14),
    "rsb": ('{"answer": "[0, 3]"}', "length", ss.MAX_NEW_TOKENS),
}


def response(row, text, stop="eos", tokens=None):
    if tokens is None:
        tokens = max(1, min(ss.MAX_NEW_TOKENS, len(text) // 4))
    return {"id": row["id"], "prompt_hash": row["prompt_hash"], "response": text,
            "generated_tokens": tokens, "stop_reason": stop}


def synthetic_responses(bank, ids):
    by_id = {row["id"]: row for row in bank}
    return [response(by_id[i], *SYNTHETIC[i]) for i in ids]


def legacy_responses(bank):
    texts = {"g1": "[3,4]", "g2": '{"answer":"(4+5)*3"}'}
    return [response(row, texts[row["id"]]) for row in bank]


def four(responses):
    """The shipped behaviour schema: the same responses without stop_reason."""
    return [{k: v for k, v in item.items() if k != "stop_reason"} for item in responses]


def frozen_receipt(bank):
    """A section 1 scope receipt of `bank`, built with the scope_receipt CLI's own signature_scope."""
    scope = sr.signature_scope([{key: row[key] for key in sr.FIELDS} for row in bank])
    return {"dataset_manifest_sha256": MANIFEST, "role": bank.role, "stage": "SYNTHETIC_SCOPE",
            "canonical_" + bank.role + "_sha256": bank.bank_sha256, "signatures": scope["signatures"]}


def scored(bank, scope, responses, meta=META, **options):
    """score_scope, with the bank's frozen scope receipt unless the call names one (C8)."""
    if bank.role in ss.RECEIPT_ROLES and "scope_receipt" not in options:
        options["scope_receipt"] = frozen_receipt(bank)
    return ss.score_scope(bank, scope, responses, meta, **options)


def refused(bank, scope, responses, meta=META, match=None, **options):
    with pytest.raises(ValueError, match=match):
        scored(bank, scope, responses, meta, **options)


# ---------------------------------------------------------------- synthetic: scope declaration

def test_declare_scope_filters_and_binds_ids():
    bank = synthetic_bank()
    core_rs = ss.declare_scope(bank, signatures=["RS"])
    assert core_rs["ids"] == ["rs1", "rs2"] and core_rs["n"] == 2 and core_rs["role"] == "dev"
    assert core_rs["ids_sha256"] == digest(["rs1", "rs2"])
    assert core_rs["parents_sha256"] == digest(["w1", "w2"]) and core_rs["parents"] == 2
    assert core_rs["bank_sha256"] == bank.bank_sha256
    assert core_rs["one_row_per_parent_per_signature"] is True
    assert core_rs["filter"] == {"tag": "core", "signatures": ["RS"], "families": None, "ids_declared": False,
                                 "diagnostic": False}
    everything = ss.declare_scope(bank, tag=None, diagnostic=True)
    assert everything["n"] == 5 and everything["rows_by_signature"] == {"R": 1, "RS": 3, "S": 1}
    assert everything["ids"] == ["r1", "rs1", "rs2", "rsb", "s1"]  # sorted, not bank order
    assert everything["ids_sha256"] == digest(everything["ids"])
    assert everything["parents_sha256"] == digest(["w1", "w2"]) and everything["parents"] == 2  # distinct worlds
    assert everything["one_row_per_parent_per_signature"] is False  # rs2 and rsb share w2
    assert everything["filter"]["diagnostic"] is True
    assert ss.declare_scope(bank, tag="boundary", diagnostic=True)["ids"] == ["rsb"]
    chosen = ss.declare_scope(bank, ids=["s1", "r1"])
    assert chosen["ids"] == ["r1", "s1"] and chosen["ids_sha256"] == digest(["r1", "s1"])
    assert ss.declare_scope(list(bank), signatures=("R",))["bank_sha256"] is None  # plain list: unbound


@pytest.mark.parametrize("kwargs,message", [
    ({"signatures": ["SP"]}, "empty scope"),
    ({"tag": "preservation"}, "empty scope"),
    ({"families": ["countdown"]}, "empty scope"),        # composition rows never match a family
    ({"signatures": "RS"}, "collection of strings"),     # a bare string would filter by characters
    ({"signatures": []}, "collection of strings"),
    ({"ids": ["r1", "r1"]}, "duplicate ids"),
    ({"ids": ["r1", "nope"]}, "outside the filtered bank"),
    ({"ids": ["rsb"]}, "outside the filtered bank"),     # boundary row, core filter
    ({"tag": "Core"}, "unknown tag"),
    ({"tag": None}, "boundary rows are never read"),     # F6: every tag, boundary rows included
    ({"tag": "boundary"}, "boundary rows are never read"),
    ({"tag": None, "signatures": ["R"]}, "never read"),  # refused even when no boundary row is selected
    ({"diagnostic": 1}, "diagnostic must be a bool"),
    ({"tag": None, "diagnostic": "yes"}, "diagnostic must be a bool"),
])
def test_declare_scope_refusals(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ss.declare_scope(synthetic_bank(), **kwargs)


def test_declare_scope_refuses_malformed_banks():
    bank = synthetic_bank()
    with pytest.raises(ValueError, match="duplicate id"):
        ss.declare_scope(list(bank) + [copy.deepcopy(bank[0])])
    mixed = copy.deepcopy(list(bank))
    mixed[0]["role"] = "calibration"
    with pytest.raises(ValueError, match="mixes roles"):
        ss.declare_scope(mixed)
    challenge = copy.deepcopy(list(bank))
    for row in challenge:
        row["role"] = "challenge"
    with pytest.raises(ValueError, match="role not admitted"):
        ss.declare_scope(challenge)
    with pytest.raises(ValueError, match="empty or malformed"):
        ss.declare_scope([])


def test_boundary_rows_need_the_diagnostic_flag_and_no_endpoint_uses_it(monkeypatch):
    """F6 / C8: boundary rows are refused unless diagnostic; the flag is recorded; enrollment refuses it."""
    bank = synthetic_bank()
    frozen = frozen_receipt(bank)
    boundary = ss.declare_scope(bank, tag="boundary", diagnostic=True)
    receipt = ss.score_scope(bank, boundary, synthetic_responses(bank, ["rsb"]), META, scope_receipt=frozen)
    assert receipt["diagnostic"] is True and receipt["scope"]["filter"]["diagnostic"] is True
    assert receipt["scope_binding"]["signatures"] == {
        "RS": {"binding": "outside_receipt", "rows": 1, "rows_in_receipt": 0, "receipt_rows": 2}}
    with pytest.raises(ValueError, match="not an endpoint"):
        ss.enrollment_from_receipt(receipt)
    assert ss.enrollment_from_receipt(receipt, diagnostic=True)["diagnostic"] is True
    with pytest.raises(ValueError, match="not an endpoint"):
        ss.enroll_scope(bank, boundary, synthetic_responses(bank, ["rsb"]), META, scope_receipt=frozen)
    # A diagnostic scope relabelled as an endpoint scope fails its own re-declaration.
    relabelled = copy.deepcopy(boundary)
    relabelled["filter"]["diagnostic"] = False
    refused(bank, relabelled, synthetic_responses(bank, ["rsb"]), match="never read", scope_receipt=frozen)
    stripped = copy.deepcopy(boundary)
    del stripped["filter"]["diagnostic"]
    refused(bank, stripped, synthetic_responses(bank, ["rsb"]), match="filter schema", scope_receipt=frozen)
    # A core scope stays an endpoint scope, and its receipt and enrollment say so.
    core = ss.declare_scope(bank, signatures=["RS"])
    endpoint = scored(bank, core, synthetic_responses(bank, core["ids"]))
    assert endpoint["diagnostic"] is False and ss.enrollment_from_receipt(endpoint)["diagnostic"] is False
    # Two core rows on one parent break the section 9 one-row-per-parent property: diagnostic only.
    twin = ss.Bank(list(bank) + [comp_row("rs3", RS_SPEC, "RS", parent="w1")], role="dev", bank_sha256="e" * 64)
    with pytest.raises(ValueError, match="one row per parent per signature"):
        ss.declare_scope(twin, signatures=["RS"])
    assert ss.declare_scope(twin, signatures=["RS"], diagnostic=True)["one_row_per_parent_per_signature"] is False
    # A guard bank holds no composition rows, so tag None needs no flag there.
    guard = legacy_bank()
    assert ss.declare_scope(guard, tag=None)["n"] == 2


# ---------------------------------------------------------------- synthetic: scoring

def test_score_scope_records_counts_and_enrollment():
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, tag=None, diagnostic=True)
    responses = synthetic_responses(bank, scope["ids"])
    frozen = frozen_receipt(bank)
    receipt = ss.score_scope(bank, scope, responses, META, scope_receipt=frozen)
    rows = {r["id"]: r for r in receipt["rows"]}
    assert [r["id"] for r in receipt["rows"]] == scope["ids"]
    assert {i: r["strict"] for i, r in rows.items()} == behaviour.score_responses(list(bank), four(responses))
    expected = {  # id -> (strict, content_pass, format_only, subtype, rs_class)
        "r1": (1, True, False, None, None),
        "s1": (0, True, True, None, None),                    # code fence: a content normalisation
        "rs1": (0, False, False, None, "FULL_RESOLVED_SET"),  # well-typed wrong answer: no subtype
        "rs2": (0, False, False, "PROSE_WRAPPED", "UNPARSEABLE"),
        "rsb": (0, False, False, "LENGTH_CAP", None),         # boundary row: never RS-classified
    }
    for ident, (strict, content, fmt, subtype, rs_class) in expected.items():
        record = rows[ident]
        assert (record["strict"], record["content_pass"], record["format_only_failure"],
                record["subtype"], record["rs_class"]) == (strict, content, fmt, subtype, rs_class), ident
    overall = receipt["counts"]["overall"]
    assert (overall["rows"], overall["strict_correct"], overall["content_correct"],
            overall["format_only_failures"], overall["strict_failures_without_subtype"]) == (5, 1, 2, 1, 2)
    assert overall["subtype"] == {"LENGTH_CAP": 1, "PROSE_WRAPPED": 1}
    assert overall["rs_class"] == {"FULL_RESOLVED_SET": 1, "UNPARSEABLE": 1}
    assert overall["stop_reason"] == {"eos": 4, "length": 1}
    assert receipt["counts"]["by_signature"]["RS"]["rows"] == 3
    assert receipt["strict_correct_ids"] == ["r1"]
    assert receipt["role"] == "dev" and receipt["bank_sha256"] == bank.bank_sha256
    assert receipt["scope"]["ids_sha256"] == scope["ids_sha256"] and "ids" not in receipt["scope"]
    assert receipt["run"] == META and receipt["manifest_sha256"] == MANIFEST
    assert receipt["responses_sha256"] == digest(sorted(responses, key=lambda item: item["id"]))
    assert set(receipt["code_sha256"]) == set(ss.CODE_FILES)
    assert receipt["code_bundle_sha256"] == digest(receipt["code_sha256"])
    assert receipt["diagnostic"] is True and receipt["scope_receipt_sha256"] == digest(frozen)
    assert receipt["scope_binding"]["signatures"] == {
        "R": {"binding": "full", "rows": 1, "rows_in_receipt": 1, "receipt_rows": 1},
        "RS": {"binding": "outside_receipt", "rows": 3, "rows_in_receipt": 2, "receipt_rows": 2},
        "S": {"binding": "full", "rows": 1, "rows_in_receipt": 1, "receipt_rows": 1}}
    # Response order does not change the receipt.
    assert ss.score_scope(bank, scope, list(reversed(responses)), META, scope_receipt=frozen) == receipt
    enrollment = ss.enroll_scope(bank, scope, responses, META, diagnostic=True, scope_receipt=frozen)
    assert enrollment["source_correct_ids"] == ["r1"] and enrollment["source_correct_count"] == 1
    assert enrollment["scoped_rows"] == 5 and enrollment["scoring_receipt_sha256"] == digest(receipt)
    assert enrollment == ss.enrollment_from_receipt(receipt, diagnostic=True)
    assert enrollment["scope_receipt_sha256"] == digest(frozen)
    json.dumps(receipt, allow_nan=False, sort_keys=True)  # serialisable as written


def test_subset_scope_returns_exactly_its_entries():
    bank = synthetic_bank()
    full = scored(bank, ss.declare_scope(bank, tag=None, diagnostic=True),
                  synthetic_responses(bank, [r["id"] for r in bank]), META)
    scope = ss.declare_scope(bank, signatures=["RS"])
    subset = scored(bank, scope, synthetic_responses(bank, scope["ids"]), META)
    assert subset["rows"] == [r for r in full["rows"] if r["id"] in {"rs1", "rs2"}]


def test_legacy_rows_score_through_the_legacy_verifier():
    bank = legacy_bank()
    with pytest.raises(ValueError, match="empty scope"):
        ss.declare_scope(bank)  # guard rows are tagged 'preservation', not 'core'
    scope = ss.declare_scope(bank, tag="preservation")
    responses = legacy_responses(bank)
    receipt = ss.score_scope(bank, scope, responses, META)
    by_id = {r["id"]: r for r in receipt["rows"]}
    assert (by_id["g1"]["strict"], by_id["g1"]["content_pass"], by_id["g1"]["normalization"]) == (0, True, "bare_list")
    assert (by_id["g2"]["strict"], by_id["g2"]["content_pass"]) == (1, True)
    assert ss.declare_scope(bank, tag="preservation", families=["countdown"])["ids"] == ["g2"]
    # C8: a guard bank is bound by its manifest-checked bank sha until guard scope receipts exist.
    assert receipt["scope_binding"]["kind"] == "guard_bank_sha256" and receipt["scope_receipt_sha256"] is None
    binding = receipt["scope_binding"]
    assert binding["bank_sha256"] == bank.bank_sha256 and "future work" in binding["note"]
    for options in ({"scope_receipt": frozen_receipt(synthetic_bank())}, {"expected_scope_receipt_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="guard scope receipts do not exist yet"):
            ss.score_scope(bank, scope, responses, META, **options)
    foreign = ss.Bank(list(bank), role="guard_dev", bank_sha256=bank.bank_sha256, manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="guard bank manifest mismatch"):
        ss.score_scope(foreign, ss.declare_scope(foreign, tag="preservation"), responses, META)


# ---------------------------------------------------------------- synthetic: refusals

def test_response_set_must_equal_the_scope():
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    good = synthetic_responses(bank, scope["ids"])
    refused(bank, scope, good[:1], match="missing 1, extra 0")
    refused(bank, scope, good + synthetic_responses(bank, ["r1"]), match="missing 0, extra 1")
    refused(bank, scope, good + good[:1], match="duplicate")
    refused(bank, scope, (good[0],), match="must be a list")


@pytest.mark.parametrize("field,value,message", [
    ("prompt_hash", "0" * 64, "wrong rendered prompt"),
    ("generated_tokens", 0, "generated_tokens"),
    ("generated_tokens", True, "generated_tokens"),
    ("generated_tokens", ss.MAX_NEW_TOKENS + 1, "generated_tokens"),
    ("generated_tokens", 5.0, "generated_tokens"),
    ("stop_reason", "stop", "unbound stop_reason"),
    ("stop_reason", None, "unbound stop_reason"),
    ("stop_reason", "length", "below the token cap"),
    ("response", None, "type"),
])
def test_response_field_refusals(field, value, message):
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    responses = synthetic_responses(bank, scope["ids"])
    responses[0][field] = value
    refused(bank, scope, responses, match=message)


def test_response_schema_refusals():
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    extra = synthetic_responses(bank, scope["ids"])
    extra[0]["model"] = "x"
    refused(bank, scope, extra, match="response schema")
    refused(bank, scope, four(synthetic_responses(bank, scope["ids"])), match="response schema")


@pytest.mark.parametrize("meta", [
    {"model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64},
    dict(META, extra="d" * 64),
    dict(META, model_sha256="A" * 64),
    dict(META, decoding_sha256="c" * 63),
    dict(META, tokenizer_sha256=None),
    None,
])
def test_run_meta_refusals(meta):
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    refused(bank, scope, synthetic_responses(bank, scope["ids"]), meta=meta)


def test_tampered_or_foreign_scopes_are_refused():
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    responses = synthetic_responses(bank, scope["ids"])
    for key, value in (("ids_sha256", "0" * 64), ("n", 3), ("parents_sha256", "0" * 64), ("role", "calibration")):
        refused(bank, dict(scope, **{key: value}), responses, match="does not match")
    shrunk = dict(scope, ids=["rs1"], ids_sha256=digest(["rs1"]), n=1)
    refused(bank, shrunk, responses[:1], match="does not match")  # the RS filter declares two rows
    other = ss.Bank(list(bank), role="dev", bank_sha256="f" * 64)
    refused(other, scope, responses, match="not declared on this bank")
    refused(bank, ss.declare_scope(list(bank), signatures=["RS"]), responses, match="not declared on this bank")
    with pytest.raises(ValueError, match="Bank from load_role"):
        ss.score_scope(list(bank), scope, responses, META)
    round_trip = json.loads(json.dumps(scope))
    assert scored(bank, round_trip, responses, META)["scope"]["ids_sha256"] == scope["ids_sha256"]


# ---------------------------------------------------------------- synthetic: scope receipts (C8)

@pytest.mark.parametrize("role", ["calibration", "dev", "test"])
def test_scope_receipt_is_mandatory_for_calibration_dev_and_test(role, monkeypatch):
    rows = [comp_row("r1", R_SPEC, "R", role=role), comp_row("s1", S_SPEC, "S", role=role)]
    bank = ss.Bank(rows, role=role, bank_sha256=digest(rows))
    scope = ss.declare_scope(bank, signatures=["S"])
    responses = synthetic_responses(bank, ["s1"])
    calls = []
    real = ss.score_one
    monkeypatch.setattr(ss, "score_one", lambda row, item: calls.append(row["id"]) or real(row, item))
    with pytest.raises(ValueError, match="needs its frozen section 1 scope receipt"):
        ss.score_scope(bank, scope, responses, META)
    with pytest.raises(ValueError, match="needs its frozen section 1 scope receipt"):
        ss.enroll_scope(bank, scope, responses, META)
    assert calls == []  # refused before any response is scored
    receipt = ss.score_scope(bank, scope, responses, META, scope_receipt=frozen_receipt(bank))
    assert receipt["scope_binding"]["bank_key"] == "canonical_" + role + "_sha256"
    assert receipt["scope_binding"]["signatures"] == {
        "S": {"binding": "full", "rows": 1, "rows_in_receipt": 1, "receipt_rows": 1}}
    assert calls == ["s1"]
    legacy = ss.Bank(rows + [dict(legacy_row("g1", "subset_sum", SUBSET, "p1"), role=role)], role=role,
                     bank_sha256="d" * 64)
    with pytest.raises(ValueError, match="composition rows only"):
        ss.score_scope(legacy, ss.declare_scope(legacy, tag="preservation"), [], META,
                       scope_receipt=frozen_receipt(legacy))


def test_scope_receipt_must_carry_and_match_the_bank_and_every_entry():
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    responses = synthetic_responses(bank, scope["ids"])
    frozen = frozen_receipt(bank)
    entry = frozen["signatures"]["RS"]
    for broken, message in (
            ({k: v for k, v in frozen.items() if k != "canonical_dev_sha256"},
             "lacks the bank sha key canonical_dev_sha256"),
            (dict(frozen, canonical_dev_sha256="0" * 64), "bank mismatch"),
            (dict(frozen, dataset_manifest_sha256="0" * 64), "manifest mismatch"),
            ({k: v for k, v in frozen.items() if k != "dataset_manifest_sha256"}, "manifest mismatch"),
            (dict(frozen, role="calibration"), "role mismatch"),
            (dict(frozen, signatures={"RS": dict(entry, rows=3)}), "differs from the frozen scope receipt: RS"),
            (dict(frozen, signatures={"RS": dict(entry, parents=1)}), "differs from the frozen scope receipt: RS"),
            (dict(frozen, signatures={"RS": dict(entry, parents_sha256="0" * 64)}), "differs"),
            (dict(frozen, signatures={"RS": {k: v for k, v in entry.items() if k != "parents"}}), "entry schema"),
            (dict(frozen, signatures={}), "lacks signature RS"),
            (dict(frozen, signatures=[]), "scope receipt schema"),
            ("receipt", "scope receipt schema")):
        refused(bank, scope, responses, match=message, scope_receipt=broken)
    r_scope = ss.declare_scope(bank, signatures=["R"])
    boolean = copy.deepcopy(frozen)
    boolean["signatures"]["R"]["rows"] = True  # True == 1, but a bool is not a row count
    refused(bank, r_scope, synthetic_responses(bank, ["r1"]), match="entry schema", scope_receipt=boolean)
    assert scored(bank, r_scope, synthetic_responses(bank, ["r1"]), scope_receipt=frozen)["counts"]["overall"][
        "strict_correct"] == 1
    moved = ss.Bank(list(bank), role="dev", bank_sha256=bank.bank_sha256, manifest_sha256="0" * 64)
    refused(moved, ss.declare_scope(moved, signatures=["RS"]), responses, match="manifest mismatch",
            scope_receipt=frozen)


def test_scope_receipt_binds_every_signature_of_a_multi_signature_scope():
    """F2 / DS-5: one call binds R, S and RS, each against its own entry; subsets bind as 'subset'."""
    bank = synthetic_bank()
    frozen = frozen_receipt(bank)
    scope = ss.declare_scope(bank, signatures=["R", "S", "RS"])
    responses = synthetic_responses(bank, scope["ids"])
    receipt = ss.score_scope(bank, scope, responses, META, scope_receipt=frozen)
    assert {name: e["binding"] for name, e in receipt["scope_binding"]["signatures"].items()} == {
        "R": "full", "RS": "full", "S": "full"}
    for name in ("R", "S", "RS"):  # every signature is checked, not only the first
        tampered = copy.deepcopy(frozen)
        tampered["signatures"][name]["ids_sha256"] = "3" * 64
        refused(bank, scope, responses, match="differs from the frozen scope receipt: " + name, scope_receipt=tampered)
        lacking = copy.deepcopy(frozen)
        del lacking["signatures"][name]
        refused(bank, scope, responses, match="lacks signature " + name, scope_receipt=lacking)
    part = ss.declare_scope(bank, ids=["rs2", "s1"])
    bound = ss.score_scope(bank, part, synthetic_responses(bank, part["ids"]), META, scope_receipt=frozen)
    assert bound["scope_binding"]["signatures"] == {
        "RS": {"binding": "subset", "rows": 1, "rows_in_receipt": 1, "receipt_rows": 2},
        "S": {"binding": "full", "rows": 1, "rows_in_receipt": 1, "receipt_rows": 1}}
    # A diagnostic scope may hold a signature the receipt does not name; an endpoint scope may not.
    lacking = copy.deepcopy(frozen)
    del lacking["signatures"]["RS"]
    diagnostic = ss.declare_scope(bank, signatures=["R", "RS"], diagnostic=True)
    out = ss.score_scope(bank, diagnostic, synthetic_responses(bank, diagnostic["ids"]), META, scope_receipt=lacking)
    assert out["scope_binding"]["signatures"]["RS"] == {"binding": "outside_receipt", "rows": 2,
                                                        "rows_in_receipt": 0, "receipt_rows": None}
    # The public single-signature assertion binds a whole core scope and needs the bank sha key.
    whole = ss.declare_scope(bank, signatures=["RS"])
    ss.assert_scope_receipt(whole, frozen, "RS")
    ss.assert_scope_receipt(whole, frozen, "RS", expected_sha256=digest(frozen))
    with pytest.raises(ValueError, match="lacks the bank sha key"):
        ss.assert_scope_receipt(whole, {k: v for k, v in frozen.items() if k != "canonical_dev_sha256"}, "RS")
    with pytest.raises(ValueError, match="differs from the pinned one"):
        ss.assert_scope_receipt(whole, frozen, "RS", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="differs from the frozen scope receipt: RS"):
        ss.assert_scope_receipt(ss.declare_scope(bank, ids=["rs1"]), frozen, "RS")  # a part is not the whole
    with pytest.raises(ValueError, match="not the core S scope"):
        ss.assert_scope_receipt(scope, frozen, "S")


def test_scope_receipt_digest_can_be_pinned(tmp_path):
    """DS-5 (3): a receipt forged from the scope itself binds unpinned, never against a pinned digest."""
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    responses = synthetic_responses(bank, scope["ids"])
    frozen = frozen_receipt(bank)
    pinned = scored(bank, scope, responses, scope_receipt=frozen, expected_scope_receipt_sha256=digest(frozen))
    assert pinned["scope_binding"]["pinned"] is True and pinned["scope_receipt_sha256"] == digest(frozen)
    assert pinned["scope_binding"]["stage"] == "SYNTHETIC_SCOPE"
    forged = {"dataset_manifest_sha256": MANIFEST, "canonical_dev_sha256": scope["bank_sha256"],
              "signatures": {"RS": {"rows": scope["n"], "parents": scope["parents"],
                                    "ids_sha256": scope["ids_sha256"], "parents_sha256": scope["parents_sha256"]}}}
    assert scored(bank, scope, responses, scope_receipt=forged)["scope_binding"]["pinned"] is False
    refused(bank, scope, responses, match="differs from the pinned one", scope_receipt=forged,
            expected_scope_receipt_sha256=digest(frozen))
    refused(bank, scope, responses, match="must be a string", scope_receipt=frozen, expected_scope_receipt_sha256=5)
    path = tmp_path / "SCOPE.json"
    path.write_bytes(json.dumps(frozen, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ss.load_scope_receipt(path, expected_file_sha256=file_sha) == frozen
    with pytest.raises(ValueError, match="file sha differs"):
        ss.load_scope_receipt(path, expected_file_sha256="0" * 64)
    duplicate = tmp_path / "DUPLICATE.json"
    duplicate.write_bytes(b'{"signatures": {}, "signatures": {}}')
    with pytest.raises(ValueError, match="duplicate key"):
        ss.load_scope_receipt(duplicate)
    listed = tmp_path / "LIST.json"
    listed.write_bytes(b"[]")
    with pytest.raises(ValueError, match="scope receipt schema"):
        ss.load_scope_receipt(listed)


# ---------------------------------------------------------------- roles and loading

@pytest.mark.parametrize("role", ["challenge", "guard_challenge", "train", "policy", "", None])
def test_refused_roles_never_touch_a_file(tmp_path, role):
    with pytest.raises(ValueError, match="role not admitted"):
        ss.load_role(tmp_path, role)
    with pytest.raises(ValueError, match="role not admitted"):
        ss.load_role(tmp_path, role, at_freeze=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("role", ["test", "guard_test"])
def test_sealed_roles_need_the_freeze_flag(tmp_path, role):
    for flag in (False, 1, "yes"):
        with pytest.raises(ValueError, match="requires at_freeze=True"):
            ss.load_role(tmp_path, role, at_freeze=flag)
    # With the flag the role gate opens, and the empty tmp root fails at the manifest, before any bank file.
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        ss.load_role(tmp_path, role, at_freeze=True)


def write_fake_dataset(root, rows):
    raw = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    (root / "canonical").mkdir(parents=True)
    (root / "canonical" / "dev.jsonl").write_bytes(raw)
    manifest = {"files": {"canonical/dev.jsonl": {"sha256": hashlib.sha256(raw).hexdigest()}},
                "counts": {"dev": len(rows)}}
    (root / "manifest.json").write_bytes(json.dumps(manifest).encode())
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def test_load_role_checks_manifest_file_and_count(tmp_path, monkeypatch):
    rows = list(synthetic_bank())
    root = tmp_path / "fake"
    manifest_sha = write_fake_dataset(root, rows)
    with pytest.raises(ValueError, match="manifest drift"):
        ss.load_role(root, "dev")
    monkeypatch.setattr(ss, "MANIFEST", manifest_sha)
    bank = ss.load_role(root, "dev")
    assert isinstance(bank, ss.Bank) and list(bank) == json.loads(json.dumps(rows))
    assert bank.bank_sha256 == hashlib.sha256((root / "canonical/dev.jsonl").read_bytes()).hexdigest()
    assert bank.manifest_sha256 == manifest_sha and bank.role == "dev"
    (root / "canonical/dev.jsonl").write_bytes((root / "canonical/dev.jsonl").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="data file drift"):
        ss.load_role(root, "dev")
    shutil.rmtree(root)
    manifest_sha = write_fake_dataset(root, rows[:4])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["dev"] = 5
    (root / "manifest.json").write_bytes(json.dumps(manifest).encode())
    monkeypatch.setattr(ss, "MANIFEST", hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="row count mismatch"):
        ss.load_role(root, "dev")


# ---------------------------------------------------------------- code binding and receipts

def copy_code(root):
    for name in ss.CODE_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ss.REPO / name, target)


def repo_file(module: str) -> str | None:
    """The repo-relative file that importing `module` executes, or None outside the repo's packages."""
    if module != "experiments" and not module.startswith("experiments."):
        return None
    base = ss.REPO.joinpath(*module.split("."))
    for candidate in (base.parent / (base.name + ".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate.relative_to(ss.REPO).as_posix()
    return None


def module_level_imports(relative: str) -> set[str]:
    """Module names a file imports when it is executed as a module (function bodies excluded)."""
    parts = Path(relative).with_suffix("").parts
    package = ".".join(parts[:-1])
    tree = ast.parse((ss.REPO / relative).read_text(encoding="utf-8"))
    found, stack = set(), list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(package.split(".")[:len(package.split(".")) - node.level + 1])
                target = base + ("." + node.module if node.module else "")
            else:
                target = node.module
            found.add(target)
            found.update(target + "." + alias.name for alias in node.names)  # a name may be a submodule
        stack.extend(ast.iter_child_nodes(node))
    return found


def import_closure(start: str) -> set[str]:
    """Every repo file executed when `start` is imported: module-level imports, package __init__ files."""
    seen, todo = set(), [start]
    while todo:
        relative = todo.pop()
        if relative in seen:
            continue
        seen.add(relative)
        for name in module_level_imports(relative):
            pieces = name.split(".")
            for end in range(1, len(pieces) + 1):
                found = repo_file(".".join(pieces[:end]))
                if found is not None and found not in seen:
                    todo.append(found)
    return seen


def test_code_files_are_the_scorer_import_closure():
    """F1: CODE_FILES is exactly what importing scoped_scorer executes from the repo."""
    closure = import_closure("experiments/maxrl_comp002_pretraining_v1/scoped_scorer.py")
    assert closure == set(ss.CODE_FILES), (sorted(closure - set(ss.CODE_FILES)), sorted(set(ss.CODE_FILES) - closure))
    for name in (CORPUS, "experiments/maxrl_composable_skills_v2/contracts.py",
                 "experiments/maxrl_verifiable_data_v1/specs.py",
                 "experiments/maxrl_comp002_pretraining_v1/preflight.py",
                 "experiments/maxrl_composable_skills_v2/behaviour.py"):
        assert name in ss.CODE_FILES
    assert len(set(ss.CODE_FILES)) == len(ss.CODE_FILES)


def test_code_files_cover_every_repo_file_score_scope_executes():
    """F1: trace score_scope on composition (core, boundary, RS) and legacy rows; every repo frame is bound."""
    filenames = set()

    def tracer(frame, event, arg):
        if event == "call":
            filenames.add(frame.f_code.co_filename)
        return None

    bank = synthetic_bank()
    scope = ss.declare_scope(bank, tag=None, diagnostic=True)
    frozen = frozen_receipt(bank)
    guard = legacy_bank()
    guard_scope = ss.declare_scope(guard, tag="preservation")
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        ss.score_scope(bank, scope, synthetic_responses(bank, scope["ids"]), META, scope_receipt=frozen)
        ss.score_scope(guard, guard_scope, legacy_responses(guard), META)
    finally:
        sys.settrace(previous)
    executed = set()
    for name in filenames:
        path = Path(name)
        if path.is_absolute() and path.resolve().is_relative_to(ss.REPO):
            relative = path.resolve().relative_to(ss.REPO).as_posix()
            if "tests" not in Path(relative).parts:  # the sealed-read guard's open wrappers (conftest)
                executed.add(relative)
    assert {CORPUS, "experiments/maxrl_verifiable_data_v1/verifier.py",
            "experiments/maxrl_composable_skills_v2/behaviour.py"} <= executed
    assert executed <= set(ss.CODE_FILES), sorted(executed - set(ss.CODE_FILES))


def test_bundle_changes_when_corpus_reward_changes(tmp_path, monkeypatch):
    """F1: a scratch copy whose corpus.reward always returns 1 moves the bundle and is refused at scoring."""
    copy_code(tmp_path)
    assert ss.code_sha256(tmp_path) == ss._IMPORT_CODE
    target = tmp_path / CORPUS
    target.write_bytes(target.read_bytes() + b"\n\ndef reward(row, response):\n    return 1\n")
    changed = ss.code_sha256(tmp_path)
    assert changed[CORPUS] != ss._IMPORT_CODE[CORPUS]
    others = {k: v for k, v in ss._IMPORT_CODE.items() if k != CORPUS}
    assert {k: v for k, v in changed.items() if k != CORPUS} == others
    assert digest(changed) != digest(ss._IMPORT_CODE)
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    responses = synthetic_responses(bank, scope["ids"])
    monkeypatch.setattr(ss, "REPO", tmp_path)
    refused(bank, scope, responses, match="changed since import")
    monkeypatch.setattr(ss, "_IMPORT_CODE", changed)  # as if the edited copy had been imported
    refused(bank, scope, responses, match="differs from source_lock.json: " + CORPUS)


def test_source_locked_scoring_files_equal_the_lock():
    """section 8.0 'the unchanged corpus.reward': every source-locked CODE_FILE equals source_lock.json."""
    lock_sha, locked = ss.source_lock()
    assert (lock_sha, locked) == (ss.SOURCE_LOCK_SHA256, ss.SOURCE_LOCKED)
    in_locked_packages = sorted(name for name in ss.CODE_FILES
                                if name.split("/")[1] in ("maxrl_composable_skills_v2", "maxrl_verifiable_data_v1"))
    assert len(in_locked_packages) == 10 and all(name in locked for name in in_locked_packages)
    assert all(ss._IMPORT_CODE[name] == locked[name] for name in in_locked_packages)
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    receipt = scored(bank, scope, synthetic_responses(bank, scope["ids"]))
    assert receipt["source_lock"] == {"path": ss.SOURCE_LOCK_FILE, "sha256": lock_sha,
                                      "files_checked": in_locked_packages}


def test_code_sha256_tracks_every_scorer_file(tmp_path):
    original = ss.code_sha256()
    assert list(original) == list(ss.CODE_FILES)
    assert all(len(v) == 64 for v in original.values())
    copy_code(tmp_path)
    assert ss.code_sha256(tmp_path) == original
    for name in ss.CODE_FILES:
        path = tmp_path / name
        before = path.read_bytes()
        path.write_bytes(before + b"# changed\n")
        changed = ss.code_sha256(tmp_path)
        assert changed[name] != original[name]
        assert {k: v for k, v in changed.items() if k != name} == {k: v for k, v in original.items() if k != name}
        path.write_bytes(before)
    assert ss.code_sha256(tmp_path) == original


def test_receipt_binds_the_code_that_ran(tmp_path, monkeypatch):
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    responses = synthetic_responses(bank, scope["ids"])
    baseline = scored(bank, scope, responses, META)
    assert scored(bank, scope, responses, META, expected_code_bundle=baseline["code_bundle_sha256"]) == baseline
    refused(bank, scope, responses, match="differs from the expected", expected_code_bundle="0" * 64)
    copy_code(tmp_path)
    name = "experiments/maxrl_comp002_pretraining_v1/content_scoring.py"
    (tmp_path / name).write_bytes((tmp_path / name).read_bytes() + b"# changed\n")
    monkeypatch.setattr(ss, "REPO", tmp_path)
    refused(bank, scope, responses, match="changed since import")
    monkeypatch.setattr(ss, "_IMPORT_CODE", ss.code_sha256())
    moved = scored(bank, scope, responses, META)
    assert moved["code_sha256"][name] != baseline["code_sha256"][name]
    assert moved["code_bundle_sha256"] != baseline["code_bundle_sha256"]
    assert moved["rows"] == baseline["rows"]


def test_write_receipt_is_create_only_lf(tmp_path):
    bank = synthetic_bank()
    scope = ss.declare_scope(bank, signatures=["RS"])
    receipt = scored(bank, scope, synthetic_responses(bank, scope["ids"]), META)
    path = tmp_path / "out" / "SCORED.json"
    written = ss.write_receipt(path, receipt)
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert written == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw) == receipt
    with pytest.raises(FileExistsError):
        ss.write_receipt(path, receipt)
    assert path.read_bytes() == raw


# ---------------------------------------------------------------- scope_receipt CLI (C8, DS-6)

def sha_of(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_scope_dataset(root, role, rows):
    """A synthetic dataset root holding one role's canonical and policy files; returns (manifest, bank) shas."""
    canonical = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    policy = b"".join(json.dumps({"id": row["id"], "prompt": "hidden"}).encode() + b"\n" for row in rows)
    for folder, raw in (("canonical", canonical), ("policy", policy)):
        (root / folder).mkdir(parents=True)
        (root / folder / (role + ".jsonl")).write_bytes(raw)
    manifest = {"files": {"canonical/" + role + ".jsonl": {"sha256": sha_of(canonical)},
                          "policy/" + role + ".jsonl": {"sha256": sha_of(policy)}},
                "counts": {role: len(rows)}}
    (root / "manifest.json").write_bytes(json.dumps(manifest).encode())
    return sha_of((root / "manifest.json").read_bytes()), sha_of(canonical)


def dev_scope_rows():
    rows = []
    for parent in ("w1", "w2"):
        for sig, spec in (("R", R_SPEC), ("S", S_SPEC), ("P", S_SPEC), ("RS", RS_SPEC)):  # P rows: never scored here
            rows.append(comp_row(sig.lower() + parent, spec, sig, parent=parent))
    rows.append(comp_row("sbw1", S_SPEC, "S", tag="boundary", parent="w1"))
    rows.append(comp_row("uw1", S_SPEC, "U", parent="w1"))  # a signature outside every receipt
    return rows


def test_scope_receipt_code_map_is_hashed_at_import_and_rechecked_before_writing(tmp_path, monkeypatch):
    """DS-6: scope_receipt and preflight are hashed at import; an edit before the write refuses it."""
    from experiments.maxrl_comp002_pretraining_v1 import preflight
    assert sr.CODE_FILES == {"scope_receipt.py": Path(sr.__file__).resolve(),
                             "preflight.py": Path(preflight.__file__).resolve()}
    assert sr._IMPORT_CODE == sr.code_sha256()
    probe = tmp_path / "probe.py"
    probe.write_bytes(b"x = 1\n")
    monkeypatch.setattr(sr, "CODE_FILES", {"probe.py": probe})
    monkeypatch.setattr(sr, "_IMPORT_CODE", sr.code_sha256())
    written = tmp_path / "unchanged" / "SCOPE.json"
    sr.write_receipt({"a": 1}, written)
    raw = written.read_bytes()
    assert b"\r" not in raw and json.loads(raw) == {"a": 1}
    with pytest.raises(FileExistsError):
        sr.write_receipt({"a": 2}, written)
    # An edit while the CLI computes: the receipt is not written.
    data = tmp_path / "data"
    manifest_sha, _ = write_scope_dataset(data, "dev", dev_scope_rows())
    monkeypatch.setattr(sr, "MANIFEST", manifest_sha)
    monkeypatch.setitem(sr.EXPECTED, "dev", {"signatures": ("R", "S", "P", "RS"), "rows": 2})
    real = sr.signature_scope

    def editing(rows):
        probe.write_bytes(b"x = 2\n")
        return real(rows)

    monkeypatch.setattr(sr, "signature_scope", editing)
    out = tmp_path / "edited" / "DEV_SCOPE.json"
    with pytest.raises(SystemExit, match="changed since import"):
        sr.main(["--data", str(data), "--role", "dev", "--out", str(out)])
    assert not out.exists()


def test_scope_receipt_cli_freezes_dev_with_p_and_the_scorer_binds_it(tmp_path, monkeypatch):
    """C8: a DEV scope receipt carries R, S, P and RS, and binds a multi-signature scope in the scorer."""
    rows = dev_scope_rows()
    data = tmp_path / "data"
    manifest_sha, canonical_sha = write_scope_dataset(data, "dev", rows)
    monkeypatch.setattr(sr, "MANIFEST", manifest_sha)
    monkeypatch.setitem(sr.EXPECTED, "dev", {"signatures": ("R", "S", "P", "RS"), "rows": 2})
    out = tmp_path / "out" / "DEV_SCOPE.json"
    assert sr.main(["--data", str(data), "--role", "dev", "--out", str(out)]) == 0
    raw = out.read_bytes()
    assert b"\r" not in raw
    receipt = json.loads(raw)
    assert sorted(receipt["signatures"]) == ["P", "R", "RS", "S"] and receipt["absent_signatures"] == []
    assert all(v["rows"] == v["parents"] == 2 for v in receipt["signatures"].values())
    assert receipt["stage"] == "S0PRE_DEV_SCOPE" and receipt["role"] == "dev"
    assert receipt["canonical_dev_sha256"] == canonical_sha and receipt["shared_parent_worlds"] == 2
    assert receipt["code_sha256"] == sr._IMPORT_CODE and receipt["code_bundle_sha256"] == digest(sr._IMPORT_CODE)
    assert receipt["test_challenge_read"] is False and receipt["spec_only_ack"] is False
    with pytest.raises(SystemExit, match="create-only"):
        sr.main(["--data", str(data), "--role", "dev", "--out", str(out)])
    monkeypatch.setattr(ss, "MANIFEST", manifest_sha)
    bank = ss.Bank(json.loads(json.dumps(rows)), role="dev", bank_sha256=canonical_sha, manifest_sha256=manifest_sha)
    frozen = ss.load_scope_receipt(out, expected_file_sha256=sha_of(raw))
    every = ss.declare_scope(bank, signatures=["R", "S", "P", "RS"])
    binding = ss.bind_scope_receipt(bank, every, frozen)
    assert {name: e["binding"] for name, e in binding["signatures"].items()} == dict.fromkeys(("P", "R", "RS", "S"),
                                                                                               "full")
    scope = ss.declare_scope(bank, signatures=["R", "RS"])
    by_id = {row["id"]: row for row in bank}
    texts = {"rw1": '{"answer":[1]}', "rw2": '{"answer":[1]}', "rsw1": '{"answer":[0,3]}', "rsw2": '{"answer":[1,4]}'}
    scoring = ss.score_scope(bank, scope, [response(by_id[i], texts[i]) for i in scope["ids"]], META,
                             scope_receipt=frozen, expected_scope_receipt_sha256=digest(frozen))
    assert scoring["counts"]["overall"]["strict_correct"] == 4 and scoring["scope_binding"]["pinned"] is True
    u_scope = ss.declare_scope(bank, signatures=["U"])
    with pytest.raises(ValueError, match="lacks signature U"):
        ss.bind_scope_receipt(bank, u_scope, frozen)


def test_scope_receipt_cli_role_gates(tmp_path):
    with pytest.raises(SystemExit, match="spec-only-ack"):
        sr.main(["--data", str(tmp_path / "absent"), "--role", "test", "--out", str(tmp_path / "o.json")])
    for role in ("guard_test", "guard_dev", "challenge"):  # guard scope receipts are future work (C8)
        with pytest.raises(SystemExit):
            sr.main(["--data", str(tmp_path), "--role", role, "--out", str(tmp_path / "out" / "o.json")])
    with pytest.raises(SystemExit, match="protected output path"):
        sr.main(["--data", str(tmp_path), "--role", "dev", "--out", str(tmp_path / "o.json")])
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- data-backed (DEV, CALIBRATION, guard_dev)

def verified_jsonl(data: Path, relative: str) -> list[dict]:
    manifest_raw = (data / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == MANIFEST
    raw = (data / relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == json.loads(manifest_raw)["files"][relative]["sha256"]
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def oracle_responses(bank, oracles):
    out = []
    for row in bank:
        oracle = oracles[row["id"]]
        out.append(response(row, oracle["references"][0] if oracle["references"] else oracle["negative"]))
    return out


VARIANTS = ("reference", "negative", "code_fence", "trailing_period", "string_encoded", "prose", "length_cap")


def variant_response(row, oracle, index):
    reference = oracle["references"][0] if oracle["references"] else oracle["negative"]
    kind = VARIANTS[index % len(VARIANTS)]
    if kind == "reference":
        return response(row, reference)
    if kind == "negative":
        return response(row, oracle["negative"])
    if kind == "code_fence":
        return response(row, "```json\n" + reference + "\n```")
    if kind == "trailing_period":
        return response(row, reference + ".")
    if kind == "string_encoded":
        return response(row, json.dumps({"answer": json.dumps(json.loads(reference)["answer"])}))
    if kind == "prose":
        return response(row, "The answer is " + reference)
    return response(row, reference[:len(reference) // 2], "length", ss.MAX_NEW_TOKENS)


def evidence(name: str) -> dict:
    return ss.load_scope_receipt(PACKAGE / "evidence" / name)


@pytest.fixture(scope="module")
def dev(comp_rs_data):
    bank = ss.load_role(comp_rs_data, "dev")
    oracles = {o["id"]: o for o in verified_jsonl(comp_rs_data, "oracles/dev.jsonl")}
    assert set(oracles) == {row["id"] for row in bank}
    return bank, oracles


@pytest.fixture(scope="module")
def dev_scope_receipt(comp_rs_data, tmp_path_factory):
    """The DEV scope receipt with R, S, P and RS, frozen by the scope_receipt CLI into a tmp dir."""
    out = tmp_path_factory.mktemp("dev_scope") / "S0PRE_DEV_SCOPE.json"
    assert sr.main(["--data", str(comp_rs_data), "--role", "dev", "--out", str(out)]) == 0
    return ss.load_scope_receipt(out)


@pytest.fixture(scope="module")
def dev_mixed(dev, dev_scope_receipt):
    bank, oracles = dev
    scope = ss.declare_scope(bank, tag=None, diagnostic=True)
    responses = [variant_response(bank_row, oracles[bank_row["id"]], i)
                 for i, bank_row in enumerate(sorted(bank, key=lambda r: r["id"]))]
    return scope, responses, ss.score_scope(bank, scope, responses, META, scope_receipt=dev_scope_receipt)


def test_real_dev_scope_receipt_adds_p_to_q1(dev_scope_receipt):
    q1 = evidence("Q1_DEV_SCOPE.json")
    for sig in ("R", "S", "RS"):  # canary: the committed Q1 DEV digests, same convention
        assert dev_scope_receipt["signatures"][sig] == q1["signatures"][sig]
    p = dev_scope_receipt["signatures"]["P"]
    assert p["rows"] == p["parents"] == 250 and p["parents_sha256"] == q1["signatures"]["R"]["parents_sha256"]
    assert dev_scope_receipt["canonical_dev_sha256"] == q1["canonical_dev_sha256"]
    assert dev_scope_receipt["policy_dev_sha256"] == q1["policy_dev_sha256"]
    assert dev_scope_receipt["shared_parent_worlds"] == 250 and dev_scope_receipt["absent_signatures"] == []
    assert dev_scope_receipt["role"] == "dev" and dev_scope_receipt["rows_in_bank"] == 4150


def test_full_dev_bank_references_match_behaviour(comp_rs_data, dev, dev_scope_receipt):
    bank, oracles = dev
    with pytest.raises(ValueError, match="never read"):
        ss.declare_scope(bank, tag=None)
    scope = ss.declare_scope(bank, tag=None, diagnostic=True)
    assert scope["n"] == len(bank) == 4150
    responses = oracle_responses(bank, oracles)
    receipt = ss.score_scope(bank, scope, responses, META, scope_receipt=dev_scope_receipt)
    shipped = behaviour.score_responses(list(bank), four(responses))
    assert {r["id"]: r["strict"] for r in receipt["rows"]} == shipped
    assert receipt["counts"]["overall"]["strict_correct"] == 4150  # oracle references score n/n strict
    assert receipt["counts"]["overall"]["content_correct"] == 4150
    binding = receipt["scope_binding"]["signatures"]
    assert binding["R"]["binding"] == binding["P"]["binding"] == "full"
    assert binding["S"] == binding["RS"] == {"binding": "outside_receipt", "rows": 300, "rows_in_receipt": 250,
                                             "receipt_rows": 250}  # 50 boundary rows each
    assert binding["RU"]["binding"] == "outside_receipt" and receipt["diagnostic"] is True
    enrolled = behaviour.enroll(comp_rs_data, "dev", four(responses), META)
    assert receipt["bank_sha256"] == enrolled["bank_sha256"]  # two hash paths, one bank
    with pytest.raises(ValueError, match="not an endpoint"):
        ss.enrollment_from_receipt(receipt)
    mine = ss.enrollment_from_receipt(receipt, diagnostic=True)
    assert mine["row_scores"] == enrolled["row_scores"]
    assert mine["source_correct_ids"] == enrolled["source_correct_ids"]
    assert mine["source_correct_count"] == enrolled["source_correct_count"] == 4150
    assert mine["scoped_rows"] == enrolled["scoped_rows"]


def test_full_dev_bank_mixed_responses_match_behaviour(comp_rs_data, dev, dev_mixed):
    bank, _ = dev
    scope, responses, receipt = dev_mixed
    shipped = behaviour.score_responses(list(bank), four(responses))
    assert {r["id"]: r["strict"] for r in receipt["rows"]} == shipped
    kinds = {item["id"]: VARIANTS[i % len(VARIANTS)] for i, item in enumerate(responses)}
    overall = receipt["counts"]["overall"]
    assert overall["strict_correct"] == sum(k == "reference" for k in kinds.values())
    assert overall["content_correct"] == sum(k in ("reference", "code_fence", "trailing_period")
                                             for k in kinds.values())
    assert overall["format_only_failures"] == sum(k in ("code_fence", "trailing_period") for k in kinds.values())
    assert overall["subtype"]["LENGTH_CAP"] == sum(k == "length_cap" for k in kinds.values())
    assert overall["subtype"]["PROSE_WRAPPED"] == sum(k == "prose" for k in kinds.values())
    assert overall["subtype"]["STRING_ENCODED_ANSWER"] > 0
    core_rs = {row["id"] for row in bank if ss.is_core_rs(row)}
    assert len(core_rs) == 250
    assert {r["id"] for r in receipt["rows"] if r["rs_class"] is not None} == core_rs
    assert sum(receipt["counts"]["overall"]["rs_class"].values()) == 250
    enrolled = behaviour.enroll(comp_rs_data, "dev", four(responses), META)
    assert ss.enrollment_from_receipt(receipt, diagnostic=True)["row_scores"] == enrolled["row_scores"]


def test_declared_core_rs_subset_returns_exactly_its_entries(comp_rs_data, dev, dev_mixed, dev_scope_receipt):
    bank, oracles = dev
    _, responses, full = dev_mixed
    scope = ss.declare_scope(bank, signatures=["RS"])
    assert scope["n"] == 250 and scope["one_row_per_parent_per_signature"] is True
    q1 = evidence("Q1_DEV_SCOPE.json")
    # Canary: the committed Q1 DEV scope receipt's RS digests (same preflight.digest convention).
    assert scope["ids_sha256"] == q1["signatures"]["RS"]["ids_sha256"]
    assert scope["parents_sha256"] == q1["signatures"]["RS"]["parents_sha256"]
    ss.assert_scope_receipt(scope, q1, "RS")
    with pytest.raises(ValueError, match="not the core S scope"):
        ss.assert_scope_receipt(scope, q1, "S")
    wanted = set(scope["ids"])
    subset = [item for item in responses if item["id"] in wanted]
    with pytest.raises(ValueError, match="needs its frozen section 1 scope receipt"):
        ss.score_scope(bank, scope, subset, META)
    receipt = ss.score_scope(bank, scope, subset, META, scope_receipt=q1)
    assert receipt["rows"] == [r for r in full["rows"] if r["id"] in wanted]
    assert receipt["scope_binding"]["signatures"]["RS"]["binding"] == "full"
    enrolled = behaviour.enroll(comp_rs_data, "dev", four(responses), META)
    mine = ss.enroll_scope(bank, scope, subset, META, scope_receipt=q1)
    assert mine["row_scores"] == {i: enrolled["row_scores"][i] for i in scope["ids"]}
    assert mine["source_correct_ids"] == [i for i in enrolled["source_correct_ids"] if i in wanted]
    assert mine["scope_ids_sha256"] == q1["signatures"]["RS"]["ids_sha256"]
    for signature in ("R", "S"):
        other = ss.declare_scope(bank, signatures=[signature])
        ss.assert_scope_receipt(other, q1, signature)
    with pytest.raises(ValueError, match="missing 250, extra 250"):
        ss.score_scope(bank, scope, [item for item in responses if item["id"] in set(other["ids"])], META,
                       scope_receipt=q1)
    few = scope["ids"][:17]
    declared = ss.declare_scope(bank, ids=list(reversed(few)))
    assert declared["ids"] == few and declared["ids_sha256"] == digest(few)
    part = ss.score_scope(bank, declared, [item for item in subset if item["id"] in set(few)], META, scope_receipt=q1)
    assert part["rows"] == [r for r in receipt["rows"] if r["id"] in set(few)]
    assert part["scope_binding"]["signatures"]["RS"] == {"binding": "subset", "rows": 17, "rows_in_receipt": 17,
                                                         "receipt_rows": 250}
    # C8: the H1 P gate needs a DEV receipt with P; Q1's lacks it, the re-frozen one binds it.
    p_scope = ss.declare_scope(bank, signatures=["P"])
    p_rows = [row for row in bank if row["id"] in set(p_scope["ids"])]
    p_responses = oracle_responses(p_rows, oracles)
    with pytest.raises(ValueError, match="lacks signature P"):
        ss.score_scope(bank, p_scope, p_responses, META, scope_receipt=q1)
    p_receipt = ss.score_scope(bank, p_scope, p_responses, META, scope_receipt=dev_scope_receipt)
    assert p_receipt["counts"]["overall"]["strict_correct"] == 250
    endpoint = ss.declare_scope(bank, signatures=["R", "S", "P", "RS"])
    assert {name: e["binding"] for name, e in ss.bind_scope_receipt(bank, endpoint, dev_scope_receipt)[
        "signatures"].items()} == dict.fromkeys(("P", "R", "RS", "S"), "full")


def test_calibration_role_is_accepted(comp_rs_data):
    with pytest.raises(ValueError, match="only exposed evaluation/guard banks"):
        behaviour.bank(comp_rs_data, "calibration")
    bank = ss.load_role(comp_rs_data, "calibration")
    assert len(bank) == 1250
    oracles = {o["id"]: o for o in verified_jsonl(comp_rs_data, "oracles/calibration.jsonl")}
    q0 = json.loads((PACKAGE / "evidence/Q0_RECEIPT.json").read_text(encoding="utf-8"))["calibration"]
    frozen = evidence("S0PRE_CALIBRATION_SCOPE.json")
    for signature in ("R", "S", "P"):
        scope = ss.declare_scope(bank, signatures=[signature])
        assert scope["n"] == 150 and scope["one_row_per_parent_per_signature"] is True
        if signature in q0:  # canary: the committed Q0 calibration digests
            assert scope["ids_sha256"] == q0[signature]["ids_sha256"]
        wanted = set(scope["ids"])
        rows = [row for row in bank if row["id"] in wanted]
        responses = oracle_responses(rows, oracles)
        with pytest.raises(ValueError, match="needs its frozen section 1 scope receipt"):
            ss.score_scope(bank, scope, responses, META)
        receipt = ss.score_scope(bank, scope, responses, META, scope_receipt=frozen)
        assert receipt["role"] == "calibration" and receipt["counts"]["overall"]["strict_correct"] == 150
        assert receipt["scope_receipt_sha256"] == digest(frozen)
        texts = {item["id"]: item["response"] for item in responses}
        assert all(r["strict"] == reward(row, texts[row["id"]])
                   for r, row in zip(receipt["rows"], sorted(rows, key=lambda x: x["id"])))
    with pytest.raises(ValueError, match="empty scope"):
        ss.declare_scope(bank, signatures=["RS"])  # CALIBRATION has no RS
    # F2: one call binds the multi-signature R+S+P scope to the frozen CALIBRATION receipt.
    every = ss.declare_scope(bank, signatures=["R", "S", "P"])
    rows = [row for row in bank if row["id"] in set(every["ids"])]
    receipt = ss.score_scope(bank, every, oracle_responses(rows, oracles), META, scope_receipt=frozen)
    assert receipt["counts"]["overall"]["strict_correct"] == 450
    assert {name: e["binding"] for name, e in receipt["scope_binding"]["signatures"].items()} == dict.fromkeys(
        ("P", "R", "S"), "full")
    # F6: CALIBRATION's 68 boundary S rows are refused without the diagnostic flag.
    with pytest.raises(ValueError, match="never read"):
        ss.declare_scope(bank, signatures=["S"], tag="boundary")
    assert ss.declare_scope(bank, signatures=["S"], tag="boundary", diagnostic=True)["n"] == 68


def test_guard_dev_legacy_bank(comp_rs_data):
    bank = ss.load_role(comp_rs_data, "guard_dev")
    q0 = json.loads((PACKAGE / "evidence/Q0_RECEIPT.json").read_text(encoding="utf-8"))["guard_dev"]
    assert bank.bank_sha256 == q0["bank_sha256"] and len(bank) == q0["rows"] == 1200
    with pytest.raises(ValueError, match="empty scope"):
        ss.declare_scope(bank)  # guard rows are tagged 'preservation'
    scope = ss.declare_scope(bank, tag="preservation")
    oracles = {o["id"]: o for o in verified_jsonl(comp_rs_data, "oracles/guard_dev.jsonl")}
    responses = oracle_responses(bank, oracles)
    receipt = ss.score_scope(bank, scope, responses, META)
    assert {r["id"]: r["strict"] for r in receipt["rows"]} == behaviour.score_responses(list(bank), four(responses))
    assert receipt["counts"]["overall"]["strict_correct"] == 1200
    assert len(receipt["counts"]["by_signature"]) == 8
    assert receipt["scope_binding"]["kind"] == "guard_bank_sha256" and receipt["scope_binding"]["bank_sha256"] == q0[
        "bank_sha256"]
    assert ss.declare_scope(bank, tag="preservation", families=["countdown"])["n"] == 150
