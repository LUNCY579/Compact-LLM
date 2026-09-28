"""Amendment 1 section 8.5: the scorer canary (clarification C8 for its scope receipts).

Synthetic banks run everywhere. Data-backed tests read only the manifest and the DEV, CALIBRATION and
guard_dev canonical/oracle files (the canary checks each against manifest 86ac8f0d) and skip when the
dataset is absent. Every sealed-role refusal and every CLI refusal runs on an empty tmp directory,
never on the dataset. The lenient-verifier test runs a scratch copy of the code in a subprocess on a
synthetic bank passed through a tmp file; that subprocess opens no data file.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.maxrl_comp002_pretraining_v1 import canary
from experiments.maxrl_comp002_pretraining_v1 import scoped_scorer as ss
from experiments.maxrl_comp002_pretraining_v1.preflight import MANIFEST, digest, sha

PACKAGE = Path(__file__).resolve().parents[1]
LETTERS = ("a_references", "b_wrappers", "b_composition_signatures", "c_derangement", "d_lookup_blind",
           "e_legacy_families")
ROWS_DIFFER = "scorer rows differ from the declared scope"

R_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "lookup": {"requested": ["x"], "links": {"x": 1}}}
R2_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "lookup": {"requested": ["x", "y"], "links": {"x": 2, "y": 3}}}
S_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "selection": {"k": 2, "target": 11}}
S3_SPEC = {"initial": [1, 2, 3, 4, 5, 6], "selection": {"k": 3, "target": 6}}
S_INFEASIBLE = {"initial": [1, 2, 3, 4, 5, 6], "selection": {"k": 2, "target": 100}}
# Value triples {0,1,2}=5 and {3,4,5}=7; resolved {0,1,3,4}; pass = one resolved 5 and one resolved 7.
RS_SPEC = {"initial": [5, 5, 5, 7, 7, 7],
           "lookup": {"requested": ["a", "b", "c", "d"],
                      "links": {"a": 0, "b": 1, "c": 3, "d": 4, "e": 2, "f": 5}},
           "selection": {"k": 2, "target": 12}}
RS_ALL_RESOLVED = {"initial": [5, 5, 5, 7, 7, 7],
                   "lookup": {"requested": ["a", "b", "c", "d", "e", "f"],
                              "links": {"a": 0, "b": 1, "c": 3, "d": 4, "e": 2, "f": 5}},
                   "selection": {"k": 2, "target": 12}}

COMPOSITION = [  # id, spec, signature, parent, tag, reference answers
    ("r1", R_SPEC, "R", "w1", "core", [[1]]),
    ("r2", R2_SPEC, "R", "w2", "core", [[2, 3], [3, 2]]),
    ("s1", S_SPEC, "S", "w1", "core", [[4, 5], [5, 4]]),
    ("s2", S_SPEC, "S", "w2", "core", [[4, 5]]),          # same spec as s1: its reference passes s1
    ("s3", S3_SPEC, "S", "w3", "core", [[0, 1, 2], [2, 1, 0], [1, 0, 2]]),
    ("s4", S_INFEASIBLE, "S", "w4", "core", ["UNKNOWN"]),  # the one row where bare UNKNOWN applies
    ("rs1", RS_SPEC, "RS", "w1", "core", [[0, 3], [1, 4]]),
    ("rs2", RS_SPEC, "RS", "w2", "core", [[1, 3]]),
    ("sb", S_SPEC, "S", "w3", "boundary", [[4, 5]]),      # boundary: never scoped
]
LEGACY = [  # id, family, problem, parent, reference answers
    ("g1", "subset_sum", {"values": [1, 2, 3, 4, 5], "k": 2, "target": 9}, "p1", [[3, 4], [4, 3]]),
    ("g2", "subset_sum", {"values": [2, 4, 6, 8, 10], "k": 2, "target": 18}, "p2", [[3, 4]]),
    ("g3", "countdown", {"numbers": [3, 4, 5], "target": 27}, "p3", ["(4+5)*3", "3*(4+5)"]),
]
REFERENCE_DEPTHS = {"R": [(0, 2), (1, 1)], "S": [(0, 4), (1, 2), (2, 1)], "RS": [(0, 2), (1, 1)]}


def ref(answer) -> str:
    return json.dumps({"answer": answer}, separators=(",", ":"))


def composition_section(entries=COMPOSITION, role="dev"):
    rows, oracles = [], {}
    for ident, spec, sig, parent, tag, answers in entries:
        rows.append({"id": ident, "parent_id": parent, "role": role, "kind": "composition", "tag": tag,
                     "skills": list(sig), "data_source": "rineya_maxrl_comp_v2",
                     "prompt_hash": digest(["prompt", ident]),
                     "verifier": {"version": "2.0.0", "kind": "composition", "spec": spec}})
        oracles[ident] = {"id": ident, "parent_id": parent, "references": [ref(a) for a in answers],
                          "negative": ref("UNKNOWN")}
    return ss.Bank(rows, role=role, bank_sha256=digest(rows)), oracles, "0" * 64


def legacy_section(entries=LEGACY):
    rows, oracles = [], {}
    for ident, family, problem, parent, answers in entries:
        rows.append({"id": ident, "parent_id": parent, "role": "guard_dev", "kind": "legacy",
                     "tag": "preservation", "skills": ["legacy:" + family],
                     "data_source": "rineya_maxrl_comp_v2/legacy/" + family,
                     "prompt_hash": digest(["prompt", ident]),
                     "verifier": {"version": "2.0.0", "kind": "legacy",
                                  "spec": {"family": family, "problem": problem}}})
        oracles[ident] = {"id": ident, "parent_id": parent, "references": [ref(a) for a in answers],
                          "negative": ref([0])}
    return ss.Bank(rows, role="guard_dev", bank_sha256=digest(rows)), oracles, "1" * 64


def frozen_receipt(bank, signatures=None):
    """A section 1 style scope receipt (C8: bank sha key included) for the core scopes of `signatures`."""
    if signatures is None:
        signatures = sorted({ss.signature(row) for row in bank if row["tag"] == "core"})
    entries = {}
    for sig in signatures:
        scope = ss.declare_scope(bank, signatures=[sig])
        entries[sig] = {"ids_sha256": scope["ids_sha256"], "parents_sha256": scope["parents_sha256"],
                        "rows": scope["n"], "parents": scope["parents"]}
    return {"dataset_manifest_sha256": MANIFEST, "role": bank.role,
            "canonical_" + bank.role + "_sha256": bank.bank_sha256, "signatures": entries}


def run_dev(entries=COMPOSITION, **options):
    """The canary on one synthetic DEV section, bound to its scope receipt, legacy exemption recorded."""
    section = composition_section(entries)
    options.setdefault("scope_receipts", {"dev": frozen_receipt(section[0])})
    options.setdefault("legacy_exemption", True)
    return canary.run_canary_on([section], **options)


def bundle() -> str:
    return digest(ss.code_sha256())


def records(receipt, check, signature=None, wrapper=None):
    return [r for r in receipt["checks"] if r["check"] == check
            and (signature is None or r["signature"] == signature) and (wrapper is None or r["wrapper"] == wrapper)]


def statuses(receipt):
    return {letter: receipt["summary"][letter]["status"] for letter in LETTERS}


def failures_of(receipt):
    return [(r["check"], r["signature"], r["wrapper"], r["failures"]) for r in receipt["checks"]
            if r["status"] == "FAIL"]


def consistent_flip(monkeypatch, flips: dict) -> None:
    """Patch the scorer so (row id, response) in flips gets that strict value, content following it."""
    original_reward, original_row = ss.reward, ss.score_row

    def reward(row, response):
        return flips.get((row["id"], response), original_reward(row, response))

    def score_row(row, response):
        result = dict(original_row(row, response))
        if (row["id"], response) in flips:
            result["strict_pass"] = result["content_pass"] = bool(flips[(row["id"], response)])
        return result

    monkeypatch.setattr(ss, "reward", reward)
    monkeypatch.setattr(ss, "score_row", score_row)


# ---------------------------------------------------------------- synthetic: helpers

def test_wrappers_and_applicability():
    reference = ref([4, 5])
    assert canary.wrap("code_fence", reference, [4, 5]) == '```json\n{"answer":[4,5]}\n```'
    assert canary.wrap("trailing_period", reference, [4, 5]) == '{"answer":[4,5]}.'
    assert canary.wrap("bare_list", reference, [4, 5]) == "[4,5]"
    assert canary.wrap("extra_field", reference, [4, 5]) == '{"answer":[4,5],"reason":"ok"}'
    assert canary.wrap("bare_unknown", ref("UNKNOWN"), "UNKNOWN") == "UNKNOWN"
    assert canary.applicable("bare_list", [4, 5], legacy=False) and canary.applicable("bare_list", [1], legacy=True)
    assert not canary.applicable("bare_list", "(4+5)*3", legacy=True)   # countdown has no bare form
    assert not canary.applicable("bare_list", "UNKNOWN", legacy=False)
    assert canary.applicable("bare_unknown", "UNKNOWN", legacy=False)
    assert not canary.applicable("bare_unknown", [4, 5], legacy=False)
    assert not canary.applicable("bare_unknown", "UNKNOWN", legacy=True)  # no legacy bare UNKNOWN
    for wrapper in ("code_fence", "trailing_period", "extra_field"):
        assert canary.applicable(wrapper, "(4+5)*3", legacy=True) and canary.applicable(wrapper, [1], legacy=False)
    for bad in ("reference", "prose", None):
        with pytest.raises(ValueError, match="unknown wrapper"):
            canary.applicable(bad, [1], legacy=False)
    with pytest.raises(ValueError, match="unknown wrapper"):
        canary.wrap("prose", reference, [4, 5])
    with pytest.raises(ValueError, match="malformed oracle reference"):
        canary.reference_answer('{"answer":[1],"alt":[2]}')
    assert canary.reference_rs_class("UNKNOWN") == "UNKNOWN_ANSWER" and canary.reference_rs_class([0, 3]) == "RS_PASS"
    assert canary.REQUIRED_COMPOSITION_WRAPPERS == ("code_fence", "trailing_period", "bare_list", "extra_field")
    assert set(canary.COMPOSITION_WRAPPERS) - set(canary.REQUIRED_COMPOSITION_WRAPPERS) == {"bare_unknown"}


def test_blind_certificate_is_the_first_lex_subset_outside_the_resolved_set():
    assert canary.blind_certificate(RS_SPEC) == [0, 5]  # (0,3), (0,4) are inside; 5 is unresolved
    assert canary.blind_certificate(RS_ALL_RESOLVED) is None
    unresolvable = dict(RS_SPEC, lookup={"requested": ["a"], "links": {"a": "b", "b": "a"}})
    assert canary.blind_certificate(unresolvable) == [0, 3]  # nothing resolves, so nothing is inside


def test_direct_strict_uses_the_verifiers():
    bank, _, _ = composition_section()
    guard, _, _ = legacy_section()
    assert canary.direct_strict(bank[0], ref([1])) == 1 and canary.direct_strict(bank[0], "[1]") == 0
    assert canary.direct_strict(guard[2], ref("(4+5)*3")) == 1
    odd = copy.deepcopy(bank[0])
    odd["verifier"]["kind"] = "other"
    with pytest.raises(ValueError, match="unknown verifier kind"):
        canary.direct_strict(odd, ref([1]))


@pytest.mark.parametrize("role,include_legacy,at_freeze,expected", [
    ("dev", True, False, ["dev", "guard_dev"]),
    ("calibration", True, False, ["calibration", "guard_dev"]),
    ("guard_dev", True, False, ["guard_dev"]),
    ("dev", False, False, ["dev"]),
    ("test", True, True, ["test", "guard_test"]),
    ("guard_test", True, True, ["guard_test"]),
])
def test_canary_roles(role, include_legacy, at_freeze, expected):
    assert canary.canary_roles(role, include_legacy, at_freeze) == expected


@pytest.mark.parametrize("role,include_legacy,at_freeze,message", [
    ("guard_dev", False, False, "legacy-only"),
    ("dev", "yes", False, "must be a bool"),
    ("test", True, False, "at_freeze"),
    ("guard_test", True, False, "at_freeze"),
    ("test", True, 1, "at_freeze"),          # only the literal True unseals
    ("challenge", True, True, "not admitted"),
    ("guard_challenge", True, True, "not admitted"),
    ("train", True, False, "not admitted"),
])
def test_canary_roles_refusals(role, include_legacy, at_freeze, message):
    with pytest.raises(ValueError, match=message):
        canary.canary_roles(role, include_legacy, at_freeze)


# ---------------------------------------------------------------- synthetic: the canary

def test_synthetic_composition_canary_passes():
    section = composition_section()
    frozen = frozen_receipt(section[0])
    receipt = canary.run_canary_on([section], scope_receipts={"dev": frozen}, legacy_exemption=True)
    assert receipt["overall"] == "PASS" and receipt["overall_failures"] == [], failures_of(receipt)
    assert statuses(receipt) == {"a_references": "PASS", "b_wrappers": "PASS", "b_composition_signatures": "PASS",
                                 "c_derangement": "PASS", "d_lookup_blind": "PASS", "e_legacy_families": "NOT_RUN"}
    described = receipt["sections"][0]
    assert described["kind"] == "composition" and described["absent_signatures"] == []
    assert described["scopes"]["S"]["n"] == 4  # the boundary row is not scoped
    assert described["scopes"]["S"]["ids_sha256"] == digest(["s1", "s2", "s3", "s4"])
    assert described["scopes"]["RS"]["parents_sha256"] == digest(["w1", "w2"])
    assert described["scope_receipt_sha256"] == digest(frozen)
    # (a) EVERY reference index, unwrapped, through the scorer: strict n/n; RS rows classify RS_PASS
    for sig, depths in REFERENCE_DEPTHS.items():
        passes = records(receipt, "references", sig)
        assert [(r["reference_index"], r["scope"]["n"]) for r in passes] == depths
        for record in passes:
            n = record["scope"]["n"]
            assert record["observed"]["strict"] == record["precomputed"]["strict"] == n
            assert record["observed"]["content"] == n and record["observed"]["subtype"] == {}
            assert record["observed"]["normalization"] == {"answer_object": n}
        # every reference is scored bare and in the code fence (which applies to every reference)
        assert sum(r["scope"]["n"] for r in passes) == sum(
            r["scope"]["n"] for r in records(receipt, "wrappers", sig, "code_fence"))
    assert [r["observed"]["rs_class"] for r in records(receipt, "references", "RS")] == [{"RS_PASS": 2},
                                                                                           {"RS_PASS": 1}]
    assert receipt["summary"]["a_references"]["rows_scored"] == 3 + 7 + 3
    # (b) every reference in each applicable wrapper; bare UNKNOWN only on the UNKNOWN reference
    fence = records(receipt, "wrappers", "S", "code_fence")
    assert [(r["reference_index"], r["scope"]["n"]) for r in fence] == [(0, 4), (1, 2), (2, 1)]
    assert sum(r["scope"]["n"] for r in records(receipt, "wrappers", "S", "bare_list")) == 6
    (unknown,) = records(receipt, "wrappers", "S", "bare_unknown")
    assert unknown["status"] == "PASS" and unknown["observed"]["normalization"] == {"bare_unknown": 1}
    for sig in ("R", "RS"):
        (skipped,) = records(receipt, "wrappers", sig, "bare_unknown")
        assert skipped["status"] == "NOT_APPLICABLE"
    for record in records(receipt, "wrappers"):
        if record["status"] == "PASS":
            n = record["scope"]["n"]
            assert record["observed"]["strict"] == record["precomputed"]["strict"] == 0
            assert record["observed"]["content"] == n
            assert record["observed"]["normalization"] == {canary.CATEGORY[record["wrapper"]]: n}
            assert record["precomputed"]["reference_strict"] == n
    assert {r["observed"]["rs_class"]["RS_PASS"] for r in records(receipt, "wrappers", "RS")
            if r["status"] == "PASS"} == {1, 2}
    # (b) per signature: bare UNKNOWN is the only wrapper a core signature may lack (F7)
    table = receipt["summary"]["b_composition_signatures"]
    assert table["signatures_incomplete"] == [] and table["wrappers_required"] == list(
        canary.REQUIRED_COMPOSITION_WRAPPERS)
    for sig in ("R", "S", "RS"):
        entry = table["by_section"]["dev"][sig]
        assert entry["complete"] and entry["references"] == "PASS" and entry["derangement"] == "PASS"
        assert entry["wrappers"] == dict.fromkeys(canary.COMPOSITION_WRAPPERS, "PASS") | (
            {} if sig == "S" else {"bare_unknown": "NOT_APPLICABLE"})
    # (c) derangement counts come from the direct verifiers; content equals strict
    derangement = {r["signature"]: r["observed"]["strict"] for r in records(receipt, "derangement")}
    assert derangement == {"R": 0, "S": 1, "RS": 2}
    for record in records(receipt, "derangement"):
        assert record["expected"]["strict"] == record["precomputed"]["strict"] == record["observed"]["content"]
        assert "reference" not in record  # no reference receipt was given
    # (d) lookup-blind certificates
    (blind,) = records(receipt, "lookup_blind")
    assert blind["observed"]["strict"] == 0 and blind["observed"]["rs_class"] == {"R_BLIND_S_VALID": 2}
    assert blind["observed"]["content"] == 0 and blind["observed"]["subtype"] == {}
    assert described["lookup_blind"] == {"rows": 2, "with_certificate": 2, "without_certificate": 0}
    # binding and flags: every pass was scored with the frozen scope receipt (C8)
    scored = [r for r in receipt["checks"] if "scope" in r]
    assert all(r["scope_binding"] == "scope_receipt" and r["scope_receipt_sha256"] == digest(frozen) for r in scored)
    assert set(receipt["code_sha256"]) == set(ss.CODE_FILES)
    assert receipt["code_bundle_sha256"] == digest(receipt["code_sha256"]) == bundle()
    assert receipt["code_check"]["status"] == "PASS" and receipt["code_check"]["source"] == "score_scope receipts"
    assert all(r["code_bundle_sha256"] == receipt["code_bundle_sha256"] for r in scored)
    assert receipt["canary_code_sha256"] == sha(PACKAGE / "canary.py")
    assert receipt["model_output_involved"] is False and receipt["model_outputs_read"] is False
    assert receipt["test_challenge_read"] is False and receipt["sealed_roles_read"] == []
    assert receipt["challenge_read"] is False and receipt["role"] == "dev" and receipt["legacy_included"] is False
    assert receipt["legacy_exemption"] is True and receipt["reference_receipt"] is None
    assert receipt["summary"]["e_legacy_families"]["exempt"] is True
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    assert canary.run_canary_on([composition_section()], scope_receipts={"dev": frozen},
                                legacy_exemption=True) == receipt  # deterministic


def test_synthetic_legacy_section_requires_all_eight_families():
    receipt = canary.run_canary_on([legacy_section()])
    assert statuses(receipt) == {"a_references": "PASS", "b_wrappers": "PASS", "b_composition_signatures": "NOT_RUN",
                                 "c_derangement": "PASS", "d_lookup_blind": "NOT_RUN", "e_legacy_families": "FAIL"}
    assert receipt["overall"] == "FAIL"
    assert receipt["overall_failures"] == ["e_legacy_families is FAIL, required PASS"]
    e = receipt["summary"]["e_legacy_families"]
    assert len(e["families_missing"]) == 6 and e["families_incomplete"] == e["families_missing"]
    table = e["by_section"]["guard_dev"]
    assert table["countdown"]["complete"] and table["subset_sum"]["complete"]
    assert table["countdown"]["wrappers"]["bare_list"] == "NOT_APPLICABLE"
    assert table["subset_sum"]["wrappers"] == dict.fromkeys(canary.LEGACY_WRAPPERS, "PASS")
    (bare,) = records(receipt, "wrappers", "legacy:countdown", "bare_list")
    assert bare["status"] == "NOT_APPLICABLE"
    (single,) = records(receipt, "derangement", "legacy:countdown")
    assert single["status"] == "NOT_APPLICABLE"  # one row has no derangement
    (subset,) = records(receipt, "derangement", "legacy:subset_sum")
    assert subset["status"] == "PASS"
    assert subset["expected"] == {"strict": 2, "content": 2, "normalization": {"answer_object": 2}, "subtype": {}}
    # (a) every reference index of every family: g1 and g3 hold two references each
    assert [(r["reference_index"], r["scope"]["n"]) for r in records(receipt, "references", "legacy:subset_sum")] == [
        (0, 2), (1, 1)]
    assert [(r["reference_index"], r["scope"]["n"]) for r in records(receipt, "references", "legacy:countdown")] == [
        (0, 1), (1, 1)]
    described = receipt["sections"][0]
    assert described["kind"] == "legacy" and described["rows_covered"] == 3 and receipt["legacy_included"] is True
    assert described["scopes"]["legacy:subset_sum"]["filter"]["tag"] == "preservation"
    # a guard bank is bound by its manifest-checked bank sha (C8): no scope receipt
    assert all(r["scope_binding"] == "guard_bank_sha256" and r["scope_receipt_sha256"] is None
               for r in receipt["checks"] if "scope" in r)


def test_composition_and_legacy_sections_together():
    section = composition_section()
    receipt = canary.run_canary_on([section, legacy_section()], scope_receipts={"dev": frozen_receipt(section[0])})
    assert [s["role"] for s in receipt["sections"]] == ["dev", "guard_dev"]
    assert receipt["summary"]["a_references"]["rows_scored"] == 13 + 5  # every reference of both banks
    assert receipt["summary"]["e_legacy_families"]["status"] == "FAIL"  # 2 of 8 families
    assert receipt["summary"]["b_composition_signatures"]["status"] == "PASS"
    assert receipt["overall"] == "FAIL" and receipt["legacy_exemption"] is False
    assert receipt["overall_failures"] == ["e_legacy_families is FAIL, required PASS"]


def test_a_guard_bank_must_hold_only_legacy_preservation_rows():
    bank, oracles, sha_ = legacy_section()
    moved = copy.deepcopy(list(bank))
    moved[0]["tag"] = "core"
    with pytest.raises(ValueError, match="only legacy preservation rows"):
        canary.run_canary_on([(ss.Bank(moved, role="guard_dev", bank_sha256=digest(moved)), oracles, sha_)])


def test_a_required_check_that_cannot_run_fails_the_canary():
    # One R row: no derangement exists, so check (c) is NOT_APPLICABLE and the canary cannot PASS.
    single = [entry for entry in COMPOSITION if entry[0] == "r1"]
    receipt = run_dev(single, signatures=["R"])
    assert statuses(receipt)["c_derangement"] == "NOT_APPLICABLE" and statuses(receipt)["a_references"] == "PASS"
    assert receipt["overall"] == "FAIL"
    assert receipt["overall_failures"] == ["c_derangement is NOT_APPLICABLE, required PASS"]
    # Core RS rows with no lookup-blind certificate: check (d) is required and cannot run.
    blindless = [("rs1", RS_ALL_RESOLVED, "RS", "w1", "core", [[0, 3]]),
                 ("rs2", RS_ALL_RESOLVED, "RS", "w2", "core", [[1, 4]])]
    receipt = run_dev(blindless, signatures=["RS"])
    assert receipt["sections"][0]["lookup_blind"] == {"rows": 2, "with_certificate": 0, "without_certificate": 2}
    assert receipt["overall_failures"] == ["d_lookup_blind is NOT_APPLICABLE, required PASS"]
    assert receipt["overall"] == "FAIL"


def test_every_required_wrapper_must_pass_on_every_composition_signature(monkeypatch):
    """F7: a wrapper that never ran on a composition signature is not a PASS of check (b)."""
    real = canary.applicable
    skipped = ("trailing_period", "bare_list", "extra_field")
    monkeypatch.setattr(canary, "applicable",
                        lambda wrapper, answer, legacy: False if not legacy and wrapper in skipped
                        else real(wrapper, answer, legacy))
    receipt = run_dev()
    assert statuses(receipt)["b_wrappers"] == "PASS"  # every wrapper pass that ran, passed
    table = receipt["summary"]["b_composition_signatures"]
    assert table["status"] == "FAIL" and table["signatures_incomplete"] == ["dev:R", "dev:RS", "dev:S"]
    assert table["by_section"]["dev"]["S"]["wrappers"] == {
        "code_fence": "PASS", "trailing_period": "NOT_APPLICABLE", "bare_list": "NOT_APPLICABLE",
        "extra_field": "NOT_APPLICABLE", "bare_unknown": "PASS"}
    assert receipt["overall"] == "FAIL"
    assert receipt["overall_failures"] == ["b_composition_signatures is FAIL, required PASS"]
    # One wrapper missing on one signature is enough: bare list never runs on the RS references.
    rs_answers = ([0, 3], [1, 4], [1, 3])
    monkeypatch.setattr(canary, "applicable",
                        lambda wrapper, answer, legacy: False if wrapper == "bare_list" and answer in rs_answers
                        else real(wrapper, answer, legacy))
    receipt = run_dev()
    table = receipt["summary"]["b_composition_signatures"]
    assert table["signatures_incomplete"] == ["dev:RS"]
    assert table["by_section"]["dev"]["RS"]["wrappers"]["bare_list"] == "NOT_APPLICABLE"
    assert receipt["overall_failures"] == ["b_composition_signatures is FAIL, required PASS"]


def test_every_reference_index_goes_through_the_scorer(monkeypatch):
    """F8: a scorer that rejects only s1's second reference, unwrapped, fails check (a)."""
    consistent_flip(monkeypatch, {("s1", ref([5, 4])): 0})
    receipt = run_dev()
    failed = [r for r in receipt["checks"] if r["status"] == "FAIL"]
    assert [(r["check"], r["signature"], r["reference_index"]) for r in failed] == [("references", "S", 1)]
    assert "strict: expected 2, observed 1" in failed[0]["failures"]
    assert failed[0]["differing_ids"] == ["s1"]
    assert receipt["overall"] == "FAIL"
    assert receipt["overall_failures"] == ["a_references is FAIL, required PASS",
                                           "b_composition_signatures is FAIL, required PASS"]
    assert receipt["summary"]["b_composition_signatures"]["signatures_incomplete"] == ["dev:S"]


def test_the_legacy_section_is_required_unless_an_exemption_is_recorded():
    """F8: a canary without a legacy section cannot PASS unless the exemption is passed, and it is recorded."""
    section = composition_section()
    receipts = {"dev": frozen_receipt(section[0])}
    plain = canary.run_canary_on([section], scope_receipts=receipts)
    assert plain["overall"] == "FAIL" and plain["legacy_exemption"] is False
    assert plain["overall_failures"] == ["e_legacy_families is NOT_RUN, required PASS"]
    assert plain["summary"]["e_legacy_families"] == {"status": "NOT_RUN", "exempt": False,
                                                     "reason": "no legacy section"}
    exempt = canary.run_canary_on([section], scope_receipts=receipts, legacy_exemption=True)
    assert exempt["overall"] == "PASS" and exempt["legacy_exemption"] is True
    assert exempt["summary"]["e_legacy_families"] == {"status": "NOT_RUN", "exempt": True,
                                                      "reason": "legacy exemption recorded: no legacy section"}
    with pytest.raises(ValueError, match="only for a canary without a legacy section"):
        canary.run_canary_on([section, legacy_section()], scope_receipts=receipts, legacy_exemption=True)
    with pytest.raises(ValueError, match="only for a canary without a legacy section"):
        canary.run_canary_on([legacy_section()], legacy_exemption=True)
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match="legacy_exemption must be a bool"):
            canary.run_canary_on([section], scope_receipts=receipts, legacy_exemption=bad)


def _consistent(original, **changes):
    return lambda row, response: dict(original(row, response), **changes)


SIGNATURE_TABLE = "b_composition_signatures"
SABOTAGE = {  # name -> (patches, summary letters expected to FAIL)
    "reward_zero": (lambda o: {"reward": lambda row, response: 0},
                    {"a_references", SIGNATURE_TABLE, "c_derangement"}),
    "strict_zero_consistently": (lambda o: {"reward": lambda row, response: 0,
                                            "score_row": _consistent(o["score_row"], strict_pass=False)},
                                 {"a_references", SIGNATURE_TABLE, "c_derangement"}),
    "strict_one_consistently": (lambda o: {"reward": lambda row, response: 1,
                                           "score_row": _consistent(o["score_row"], strict_pass=True,
                                                                    content_pass=True)},
                                {"b_wrappers", SIGNATURE_TABLE, "c_derangement", "d_lookup_blind"}),
    "normalization_mislabelled": (lambda o: {"score_row": _consistent(o["score_row"],
                                                                      normalization="answer_object")},
                                  {"b_wrappers", SIGNATURE_TABLE}),
    "content_always_passes": (lambda o: {"score_row": _consistent(o["score_row"], content_pass=True)},
                              {SIGNATURE_TABLE, "c_derangement", "d_lookup_blind"}),
    "classifier_broken": (lambda o: {"classify_rs": lambda spec, response: "OTHER_WRONG"},
                          {"a_references", "b_wrappers", SIGNATURE_TABLE, "d_lookup_blind"}),
    "subtyper_labels_everything": (lambda o: {"subtype_strict_failure": lambda row, response, stop: "OTHER"},
                                   {"a_references", "b_wrappers", SIGNATURE_TABLE, "c_derangement",
                                    "d_lookup_blind"}),
}


@pytest.mark.parametrize("name", sorted(SABOTAGE))
def test_sabotaged_scorer_fails_the_canary(monkeypatch, name):
    clean = run_dev()
    patches, failing = SABOTAGE[name]
    for attribute, value in patches({"score_row": ss.score_row}).items():
        monkeypatch.setattr(ss, attribute, value)
    receipt = run_dev()
    assert receipt["overall"] == "FAIL"
    assert {letter for letter, state in statuses(receipt).items() if state == "FAIL"} == failing
    # The independent path never goes through the scorer: its precomputation is unchanged.
    assert [r.get("precomputed") for r in receipt["checks"]] == [r.get("precomputed") for r in clean["checks"]]
    assert [r.get("expected") for r in receipt["checks"]] == [r.get("expected") for r in clean["checks"]]


def test_reward_sabotage_is_reported_as_a_scorer_error(monkeypatch):
    monkeypatch.setattr(ss, "reward", lambda row, response: 0)
    receipt = run_dev()
    failed = records(receipt, "references")
    assert len(failed) == 7  # every reference index of R (2), S (3) and RS (2)
    assert all(r["status"] == "FAIL" and "disagrees with corpus.reward" in r["error"] for r in failed)
    assert all(r["observed"] is None for r in failed)
    assert "a_references is FAIL, required PASS" in receipt["overall_failures"]


def test_derangement_compares_row_ids_not_only_counts(monkeypatch):
    # A scorer that moves the one passing deranged S row from s1 to s2 keeps the count (1) but not the ids.
    consistent_flip(monkeypatch, {("s1", ref([4, 5])): 0, ("s2", ref([0, 1, 2])): 1})
    receipt = run_dev()
    (record,) = records(receipt, "derangement", "S")
    assert record["observed"]["strict"] == record["expected"]["strict"] == 1
    assert record["status"] == "FAIL" and record["differing_ids"] == ["s1", "s2"]
    assert record["failures"] == ["strict differs from the direct verifiers on 2 rows"]
    assert receipt["overall"] == "FAIL"


@pytest.mark.parametrize("mode", ["reversed", "missing"])
def test_scorer_rows_out_of_order_or_missing_fail_the_canary(monkeypatch, mode):
    """TA-11: the scorer's rows must be exactly the declared scope, in order."""
    real = ss.score_scope

    def tampered(*args, **kwargs):
        receipt = real(*args, **kwargs)
        rows = receipt["rows"][::-1] if mode == "reversed" else receipt["rows"][:-1]
        return dict(receipt, rows=rows)

    monkeypatch.setattr(ss, "score_scope", tampered)
    receipt = run_dev()
    scored = [r for r in receipt["checks"] if "scope" in r]
    assert any(r["scope"]["n"] > 1 for r in scored) and any(r["scope"]["n"] == 1 for r in scored)
    for record in scored:
        if mode == "missing":
            assert ROWS_DIFFER in record["failures"] and record["status"] == "FAIL"
        elif record["scope"]["n"] > 1:  # the same rows in another order: only the order check sees it
            assert record["failures"] == [ROWS_DIFFER]
        else:
            assert record["status"] == "PASS"
    assert receipt["overall"] == "FAIL"


def test_one_flipped_legacy_derangement_row_makes_its_family_incomplete(monkeypatch):
    """TA-11: a legacy derangement FAIL leaves that family incomplete in check (e)."""
    # g2's own reference is [4,3], so the deranged text it receives from g1 ([3,4]) is scored only by (c).
    entries = [entry if entry[0] != "g2" else entry[:4] + ([[4, 3]],) for entry in LEGACY]
    clean = canary.run_canary_on([legacy_section(entries)])
    assert clean["summary"]["e_legacy_families"]["by_section"]["guard_dev"]["subset_sum"]["complete"]
    assert records(clean, "derangement", "legacy:subset_sum")[0]["observed"]["strict"] == 2
    consistent_flip(monkeypatch, {("g2", ref([3, 4])): 0})
    receipt = canary.run_canary_on([legacy_section(entries)])
    (deranged,) = records(receipt, "derangement", "legacy:subset_sum")
    assert deranged["status"] == "FAIL" and deranged["differing_ids"] == ["g2"]
    assert [r["check"] for r in receipt["checks"] if r["status"] == "FAIL"] == ["derangement"]
    e = receipt["summary"]["e_legacy_families"]
    entry = e["by_section"]["guard_dev"]["subset_sum"]
    assert entry["references"] == "PASS" and entry["wrappers"] == dict.fromkeys(canary.LEGACY_WRAPPERS, "PASS")
    assert entry["derangement"] == "FAIL" and entry["complete"] is False
    assert "subset_sum" in e["families_incomplete"] and "subset_sum" not in e["families_missing"]
    assert e["by_section"]["guard_dev"]["countdown"]["complete"]


def test_a_wrong_oracle_reference_fails_the_canary():
    bank, oracles, sha_ = composition_section()
    oracles["r2"]["references"] = [ref([2])]  # r2 needs both resolved indices [2, 3]
    receipt = canary.run_canary_on([(bank, oracles, sha_)], scope_receipts={"dev": frozen_receipt(bank)},
                                   legacy_exemption=True)
    (record,) = records(receipt, "references", "R")
    assert "precomputed strict 1 != registered 2" in record["failures"] and record["status"] == "FAIL"
    assert any("reference_strict" in f for r in records(receipt, "wrappers", "R") for f in r["failures"])
    assert receipt["overall"] == "FAIL"


def test_code_bundle_is_checked():
    assert run_dev(expected_code_bundle=bundle())["overall"] == "PASS"
    receipt = run_dev(expected_code_bundle="0" * 64)
    assert receipt["code_check"]["status"] == "FAIL" and receipt["overall"] == "FAIL"
    assert receipt["code_check"]["observed"] == bundle() and receipt["overall_failures"] == ["code check FAIL"]
    for bad in ("0" * 63, "A" * 64, "g" * 64, 5, ["0" * 64]):
        with pytest.raises(ValueError, match="lowercase sha256 hex digest"):
            run_dev(expected_code_bundle=bad)


def test_canary_source_changed_since_import_fails_the_code_check(monkeypatch):
    monkeypatch.setattr(canary, "_IMPORT_CANARY_SHA256", "0" * 64)
    receipt = run_dev()
    assert receipt["code_check"]["failures"] == ["canary source changed since import"]
    assert receipt["overall"] == "FAIL" and receipt["overall_failures"] == ["code check FAIL"]


def test_at_freeze_canary_needs_the_expected_code_bundle(monkeypatch):
    """TA-03: an at-freeze canary is refused without the S1/S2 bundle, before anything is scored."""
    calls = []
    real = ss.score_scope
    monkeypatch.setattr(ss, "score_scope", lambda *a, **k: calls.append(1) or real(*a, **k))
    bank, oracles, sha_ = composition_section()
    sealed = ss.Bank([dict(row, role="test") for row in bank], role="test", bank_sha256="2" * 64)
    receipts = {"test": frozen_receipt(sealed)}
    with pytest.raises(ValueError, match="at-freeze canary needs expected_code_bundle"):
        canary.run_canary_on([(sealed, oracles, sha_)], at_freeze=True, scope_receipts=receipts,
                             legacy_exemption=True)
    assert calls == []
    receipt = canary.run_canary_on([(sealed, oracles, sha_)], at_freeze=True, scope_receipts=receipts,
                                   legacy_exemption=True, expected_code_bundle=bundle())
    assert receipt["overall"] == "PASS" and receipt["at_freeze"] is True and receipt["sealed_roles_read"] == ["test"]
    assert receipt["code_check"]["expected"] == bundle()


def test_scorer_code_change_during_the_canary_fails_it(monkeypatch):
    real = ss.score_scope
    calls = []

    def drifting(*args, **kwargs):
        receipt = real(*args, **kwargs)
        calls.append(1)
        if len(calls) > 1:  # a later pass reports a different code map
            receipt = dict(receipt, code_sha256={k: "f" * 64 for k in receipt["code_sha256"]},
                           code_bundle_sha256="e" * 64)
        return receipt

    monkeypatch.setattr(ss, "score_scope", drifting)
    receipt = run_dev()
    assert receipt["code_check"]["failures"] == ["scorer code changed during the canary"]
    assert receipt["overall"] == "FAIL" and len(receipt["code_check"]["bundles_seen"]) == 2


def test_absent_signatures_are_recorded_not_failed():
    receipt = run_dev(signatures=["R", "P"])
    assert receipt["overall"] == "PASS" and receipt["sections"][0]["absent_signatures"] == ["P"]
    assert receipt["summary"]["d_lookup_blind"]["status"] == "NOT_RUN"  # no RS scoped: not required
    assert list(receipt["summary"]["b_composition_signatures"]["by_section"]["dev"]) == ["R"]
    with pytest.raises(ValueError, match="no requested signature"):
        run_dev(signatures=["P", "SP"])


@pytest.mark.parametrize("signatures", ["RS", [], ["RX"], ["R", "R"], [""], [1]])
def test_signature_argument_refusals(signatures):
    with pytest.raises(ValueError, match="signature"):
        run_dev(signatures=signatures)


def test_section_refusals():
    bank, oracles, sha_ = composition_section()
    receipts = {"dev": frozen_receipt(bank)}
    with pytest.raises(ValueError, match="no sections"):
        canary.run_canary_on([])
    with pytest.raises(ValueError, match="Bank"):
        canary.run_canary_on([(list(bank), oracles, sha_)])
    with pytest.raises(ValueError, match="role differs"):
        canary.run_canary_on([(bank, oracles, sha_)], role="calibration", scope_receipts=receipts)
    with pytest.raises(ValueError, match="duplicate section"):
        canary.run_canary_on([(bank, oracles, sha_), (bank, oracles, sha_)], scope_receipts=receipts)
    sealed = ss.Bank([dict(row, role="test") for row in bank], role="test", bank_sha256="2" * 64)
    with pytest.raises(ValueError, match="at_freeze"):
        canary.run_canary_on([(sealed, oracles, sha_)])
    missing = dict(oracles)
    del missing["sb"]
    with pytest.raises(ValueError, match="oracle ids differ"):
        canary.run_canary_on([(bank, missing, sha_)], scope_receipts=receipts)
    moved = copy.deepcopy(oracles)
    moved["r1"]["parent_id"] = "w9"
    with pytest.raises(ValueError, match="parent mismatch"):
        canary.run_canary_on([(bank, moved, sha_)], scope_receipts=receipts)
    empty = copy.deepcopy(oracles)
    empty["r1"]["references"] = []
    with pytest.raises(ValueError, match="references malformed"):
        canary.run_canary_on([(bank, empty, sha_)], scope_receipts=receipts)
    malformed = copy.deepcopy(oracles)
    malformed["r1"]["references"] = ['{"answer":[1],"why":"x"}']
    with pytest.raises(ValueError, match="malformed oracle reference"):
        canary.run_canary_on([(bank, malformed, sha_)], scope_receipts=receipts)


def test_composition_sections_need_their_scope_receipt(monkeypatch):
    """C8: a composition section without its frozen scope receipt is refused before anything is scored."""
    calls = []
    real = ss.score_scope
    monkeypatch.setattr(ss, "score_scope", lambda *a, **k: calls.append(1) or real(*a, **k))
    section = composition_section()
    for sections in ([section], [section, legacy_section()]):
        with pytest.raises(ValueError, match=r"needs its frozen section 1 scope receipt \(C8\): dev"):
            canary.run_canary_on(sections, legacy_exemption=len(sections) == 1)
    calibration = composition_section(role="calibration")
    with pytest.raises(ValueError, match="scope receipt \\(C8\\): calibration"):
        canary.run_canary_on([section, calibration], scope_receipts={"dev": frozen_receipt(section[0])},
                             legacy_exemption=True)
    assert calls == []
    # every section is bound before any is scored: a bad second receipt stops the first section too
    wrong = frozen_receipt(calibration[0])
    wrong["signatures"]["R"]["rows"] = 3
    with pytest.raises(ValueError, match="differs from the frozen scope receipt: R"):
        canary.run_canary_on([section, calibration], legacy_exemption=True,
                             scope_receipts={"dev": frozen_receipt(section[0]), "calibration": wrong})
    assert calls == []
    both = canary.run_canary_on([section, calibration], legacy_exemption=True,
                                scope_receipts={"dev": frozen_receipt(section[0]),
                                                "calibration": frozen_receipt(calibration[0])})
    assert both["overall"] == "PASS" and [s["role"] for s in both["sections"]] == ["dev", "calibration"]


def test_scope_receipts_bind_the_canary_scopes(monkeypatch):
    section = composition_section()
    frozen = frozen_receipt(section[0], ("R", "S", "RS"))
    receipt = canary.run_canary_on([section], scope_receipts={"dev": frozen}, legacy_exemption=True)
    assert receipt["overall"] == "PASS" and receipt["sections"][0]["scope_receipt_sha256"] == digest(frozen)
    # partial passes (a reference index or a wrapper some rows lack) bind as a subset of the receipt
    partial = [r for r in receipt["checks"] if "scope" in r and r["scope"]["filter"]["ids_declared"]]
    assert partial and all(r["scope_receipt_sha256"] == digest(frozen) for r in partial)
    calls = []
    real = ss.score_scope
    monkeypatch.setattr(ss, "score_scope", lambda *a, **k: calls.append(1) or real(*a, **k))
    tampered = copy.deepcopy(frozen)
    tampered["signatures"]["RS"]["ids_sha256"] = "3" * 64
    lacking = frozen_receipt(section[0], ("R", "S"))
    keyless = {k: v for k, v in frozen.items() if k != "canonical_dev_sha256"}
    other_bank = dict(frozen, canonical_dev_sha256="4" * 64)
    for bad, message in ((tampered, "differs from the frozen scope receipt: RS"), (lacking, "lacks signature RS"),
                         (keyless, "lacks the bank sha key canonical_dev_sha256"), (other_bank, "bank mismatch")):
        with pytest.raises(ValueError, match=message):
            canary.run_canary_on([section], scope_receipts={"dev": bad}, legacy_exemption=True)
    assert calls == []  # a scope receipt mismatch refuses before anything is scored
    with pytest.raises(ValueError, match="composition sections"):
        canary.run_canary_on([section, legacy_section()], scope_receipts={"dev": frozen, "guard_dev": frozen})
    with pytest.raises(ValueError, match="composition sections"):
        canary.run_canary_on([section], scope_receipts={"dev": frozen, "calibration": frozen})
    with pytest.raises(ValueError, match="maps a role"):
        canary.run_canary_on([section], scope_receipts={"dev": "receipt"})


# ---------------------------------------------------------------- synthetic: reference receipts (TA-03)

def test_reference_receipt_binds_the_derangement_counts(monkeypatch):
    section = composition_section()
    options = {"scope_receipts": {"dev": frozen_receipt(section[0])}, "legacy_exemption": True}
    clean = canary.run_canary_on([section], **options)
    bound = canary.run_canary_on([section], reference_receipt=clean, **options)
    assert bound["overall"] == "PASS", failures_of(bound)
    assert bound["reference_receipt"] == {"sha256": digest(clean), "code_bundle_sha256": clean["code_bundle_sha256"],
                                          "derangements_compared": 3}
    for record in records(bound, "derangement"):
        assert record["reference"]["strict"] == record["precomputed"]["strict"]
        assert record["reference"]["ids_sha256"] == record["scope"]["ids_sha256"]
    # A reference whose S count or ids differ fails (c): the same-run precomputation is not the reference.
    for key, value, message in (("strict", 2, "precomputed strict 1 != reference receipt 2"),
                                ("strict_ids_sha256", "5" * 64, "precomputed strict_ids_sha256 ")):
        moved = copy.deepcopy(clean)
        (entry,) = records(moved, "derangement", "S")
        entry["precomputed"][key] = value
        receipt = canary.run_canary_on([section], reference_receipt=moved, **options)
        (record,) = records(receipt, "derangement", "S")
        assert record["status"] == "FAIL" and any(f.startswith(message) for f in record["failures"])
        assert record["observed"]["strict"] == record["expected"]["strict"] == 1  # scorer and verifier agree
        assert receipt["overall_failures"] == ["b_composition_signatures is FAIL, required PASS",
                                               "c_derangement is FAIL, required PASS"]
    lacking = copy.deepcopy(clean)
    lacking["checks"] = [r for r in lacking["checks"] if not (r["check"] == "derangement" and r["signature"] == "RS")]
    receipt = canary.run_canary_on([section], reference_receipt=lacking, **options)
    (record,) = records(receipt, "derangement", "RS")
    assert record["failures"] == ["reference receipt has no scored derangement for this scope"]
    rescoped = copy.deepcopy(clean)
    records(rescoped, "derangement", "R")[0]["scope"]["ids_sha256"] = "6" * 64
    (record,) = records(canary.run_canary_on([section], reference_receipt=rescoped, **options), "derangement", "R")
    assert record["failures"] == ["reference receipt derangement scope differs"]
    # A reference that does not describe these banks, did not PASS or is not a canary receipt is refused
    # before anything is scored.
    other_bank = copy.deepcopy(clean)
    other_bank["sections"][0]["bank_sha256"] = "7" * 64
    other_oracles = copy.deepcopy(clean)
    other_oracles["sections"][0]["oracle_sha256"] = "8" * 64
    guard_only = canary.run_canary_on([legacy_section()])
    calls = []
    real = ss.score_scope
    monkeypatch.setattr(ss, "score_scope", lambda *a, **k: calls.append(1) or real(*a, **k))
    for bad, message in ((dict(clean, overall="FAIL"), "did not PASS"),
                         (dict(clean, kind="scoped_scoring_receipt"), "not a scorer canary receipt"),
                         ("receipt", "not a scorer canary receipt"),
                         (other_bank, "section differs from this bank: dev"),
                         (other_oracles, "section differs from this bank: dev"),
                         (dict(guard_only, overall="PASS"), "lacks section dev"),
                         ({k: v for k, v in clean.items() if k != "checks"}, "reference receipt schema")):
        with pytest.raises(ValueError, match=message):
            canary.run_canary_on([section], reference_receipt=bad, **options)
    assert calls == []


def test_reference_receipt_binds_each_legacy_family(monkeypatch):
    section = legacy_section()
    reference = dict(canary.run_canary_on([section]), overall="PASS")  # 2 of 8 families: marked PASS here
    receipt = canary.run_canary_on([section], reference_receipt=reference)
    assert receipt["reference_receipt"]["derangements_compared"] == 1  # countdown has one row: no derangement
    assert statuses(receipt)["c_derangement"] == "PASS"
    (entry,) = records(reference, "derangement", "legacy:subset_sum")
    entry["precomputed"]["strict"] = 1
    receipt = canary.run_canary_on([section], reference_receipt=reference)
    (record,) = records(receipt, "derangement", "legacy:subset_sum")
    assert record["failures"] == ["precomputed strict 2 != reference receipt 1"]
    assert "subset_sum" in receipt["summary"]["e_legacy_families"]["families_incomplete"]


# The review's lenient verifier: selection accepts any length-k subset inside the candidates, whatever it sums to.
VERIFIER = "experiments/maxrl_composable_skills_v2/verifier.py"
CANARY_FILE = "experiments/maxrl_comp002_pretraining_v1/canary.py"
STRICT_SELECTION = b"if len(answer)!=q['k'] or sum(values[i] for i in answer)!=q['target']:return 0"
LENIENT_SELECTION = b"if len(answer)!=q['k']:return 0"
SCRATCH_RUNNER = '''
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from experiments.maxrl_comp002_pretraining_v1 import canary, scoped_scorer as ss
from experiments.maxrl_composable_skills_v2 import verifier
imported = [pathlib.Path(m.__file__).resolve() for m in (canary, ss, verifier)]
assert all(path.is_relative_to(root) for path in imported), imported
payload = json.loads(pathlib.Path(sys.argv[2]).read_bytes().decode("utf-8"))
bank = ss.Bank(payload["rows"], role="dev", bank_sha256=payload["bank_sha256"])
sections = [(bank, payload["oracles"], payload["oracle_sha256"])]
out = {"imported": [str(path) for path in imported]}
for label, options in payload["runs"].items():
    receipt = canary.run_canary_on(sections, scope_receipts={"dev": payload["scope_receipt"]},
                                   legacy_exemption=True, **options)
    deranged = [r for r in receipt["checks"] if r["check"] == "derangement"]
    out[label] = {"overall": receipt["overall"], "overall_failures": receipt["overall_failures"],
                  "derangement": {r["signature"]: r["observed"]["strict"] for r in deranged},
                  "derangement_failures": {r["signature"]: r["failures"] for r in deranged},
                  "code_bundle_sha256": receipt["code_bundle_sha256"],
                  "code_check": receipt["code_check"]["failures"]}
sys.stdout.write(json.dumps(out))
'''


def _scratch_canary(tmp_path, name, lenient, runs, payload):
    """Run the canary from a scratch copy of the code (the scorer's CODE_FILES, the lock, canary.py)."""
    root = tmp_path / name
    for relative in ss.CODE_FILES + (ss.SOURCE_LOCK_FILE, CANARY_FILE):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ss.REPO / relative).read_bytes())
    if lenient:
        path = root / VERIFIER
        before = path.read_bytes()
        assert before.count(STRICT_SELECTION) == 1
        path.write_bytes(before.replace(STRICT_SELECTION, LENIENT_SELECTION))
        # Re-pin the scratch lock too, so the scorer's source-lock check passes: only the reference
        # receipt (or an expected code bundle) is left to see the change.
        lock_path = root / ss.SOURCE_LOCK_FILE
        lock = json.loads(lock_path.read_bytes().decode("utf-8"))
        lock["source_inventories"]["maxrl_composable_skills_v2"]["verifier.py"] = hashlib.sha256(
            path.read_bytes()).hexdigest()
        lock_path.write_bytes(json.dumps(lock, indent=2).encode("utf-8"))
    runner = tmp_path / (name + "_runner.py")
    runner.write_bytes(SCRATCH_RUNNER.encode("utf-8"))
    request = tmp_path / (name + "_payload.json")
    request.write_bytes(json.dumps(dict(payload, runs=runs)).encode("utf-8"))
    done = subprocess.run([sys.executable, "-I", str(runner), str(root), str(request)],
                          capture_output=True, timeout=600)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    out = json.loads(done.stdout.decode("utf-8"))
    assert all(Path(path).is_relative_to(root.resolve()) for path in out["imported"])
    return out


def test_a_lenient_verifier_scratch_copy_is_caught_by_the_reference_receipt(tmp_path):
    """TA-03: a verifier regression moves the scorer and the same-run precomputation together; a reference
    receipt made by the committed code (or an expected code bundle) catches it."""
    bank, oracles, oracle_sha = section = composition_section()
    frozen = frozen_receipt(bank)
    clean = canary.run_canary_on([section], scope_receipts={"dev": frozen}, legacy_exemption=True)
    assert clean["overall"] == "PASS"
    payload = {"rows": list(bank), "bank_sha256": bank.bank_sha256, "oracles": oracles,
               "oracle_sha256": oracle_sha, "scope_receipt": frozen}
    runs = {"no_reference": {}, "with_reference": {"reference_receipt": clean},
            "with_bundle": {"expected_code_bundle": clean["code_bundle_sha256"]}}
    control = _scratch_canary(tmp_path, "clean_copy", False, runs, payload)  # the copy itself changes nothing
    for label in runs:
        assert control[label]["overall"] == "PASS", control[label]
        assert control[label]["derangement"] == {"R": 0, "S": 1, "RS": 2}
        assert control[label]["code_bundle_sha256"] == clean["code_bundle_sha256"]
    lenient = _scratch_canary(tmp_path, "lenient_copy", True, runs, payload)
    # Without a reference or a bundle the canary PASSes: S's deranged count moved from 1 to 2 on both paths.
    assert lenient["no_reference"]["overall"] == "PASS" and lenient["no_reference"]["derangement"]["S"] == 2
    assert lenient["no_reference"]["code_bundle_sha256"] != clean["code_bundle_sha256"]
    caught = lenient["with_reference"]
    assert caught["overall"] == "FAIL"
    assert caught["overall_failures"] == ["b_composition_signatures is FAIL, required PASS",
                                          "c_derangement is FAIL, required PASS"]
    assert "precomputed strict 2 != reference receipt 1" in caught["derangement_failures"]["S"]
    assert caught["derangement_failures"]["R"] == caught["derangement_failures"]["RS"] == []
    assert lenient["with_bundle"]["overall"] == "FAIL"
    assert lenient["with_bundle"]["code_check"] == ["scorer code bundle differs from the expected one"]


# ---------------------------------------------------------------- synthetic: score_with_canary

def _responses(bank, scope, oracles):
    by_id = {row["id"]: row for row in bank}
    return [{"id": i, "prompt_hash": by_id[i]["prompt_hash"], "response": oracles[i]["references"][0],
             "generated_tokens": 5, "stop_reason": "eos"} for i in scope["ids"]]


META = {"model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64, "decoding_sha256": "c" * 64}


def _bound(section, **options):
    options.setdefault("scope_receipts", {"dev": frozen_receipt(section[0])})
    options.setdefault("legacy_exemption", True)
    options.setdefault("expected_code_bundle", bundle())
    return options


def test_score_with_canary_scores_the_batch_in_the_same_pass():
    bank, oracles, sha_ = section = composition_section()
    scope = ss.declare_scope(bank, signatures=["RS"])
    options = _bound(section)
    out = canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META, **options)
    assert out["canary"]["overall"] == "PASS" and out["kind"] == "canary_bound_scoring"
    assert out["scoring"]["code_bundle_sha256"] == out["canary"]["code_bundle_sha256"] == out["code_bundle_sha256"]
    assert out["scoring"]["counts"]["overall"]["strict_correct"] == 2
    assert out["canary_receipt_sha256"] == digest(out["canary"])
    assert out["scoring_receipt_sha256"] == digest(out["scoring"])
    # C8: the batch is bound to the same frozen scope receipt the canary used
    frozen = options["scope_receipts"]["dev"]
    assert out["scoring"]["scope_receipt_sha256"] == out["canary"]["sections"][0]["scope_receipt_sha256"] == digest(
        frozen)
    assert out["scoring"]["scope_binding"]["kind"] == "scope_receipt"
    with pytest.raises(ValueError, match="one of the canary's sections"):
        other = ss.Bank(list(bank), role="dev", bank_sha256=bank.bank_sha256)
        canary.score_with_canary([section], other, scope, _responses(bank, scope, oracles), META, **options)
    guard = legacy_section()
    guard_scope = ss.declare_scope(guard[0], tag="preservation", families=["countdown"])
    with pytest.raises(canary.CanaryFailed):  # 2 of 8 families: (e) FAILs, so the guard batch is not scored
        canary.score_with_canary([guard], guard[0], guard_scope, _responses(guard[0], guard_scope, guard[1]), META,
                                 expected_code_bundle=bundle())


def test_score_with_canary_requires_the_expected_code_bundle(monkeypatch):
    """TA-03: score_with_canary refuses without the S1/S2 bundle, before the canary or the batch runs."""
    bank, oracles, sha_ = section = composition_section()
    scope = ss.declare_scope(bank, signatures=["S"])
    calls = []
    real_canary, real_score = canary.run_canary_on, ss.score_scope
    monkeypatch.setattr(canary, "run_canary_on", lambda *a, **k: calls.append("canary") or real_canary(*a, **k))
    monkeypatch.setattr(ss, "score_scope", lambda *a, **k: calls.append("score") or real_score(*a, **k))
    options = _bound(section)
    del options["expected_code_bundle"]
    with pytest.raises(ValueError, match="score_with_canary needs expected_code_bundle"):
        canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META, **options)
    with pytest.raises(ValueError, match="lowercase sha256 hex digest"):
        canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META,
                                 expected_code_bundle="x" * 64, **options)
    assert calls == []
    out = canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META,
                                   expected_code_bundle=bundle(), **options)
    assert out["canary"]["code_check"]["expected"] == bundle() and calls[0] == "canary" and calls[-1] == "score"


def test_score_with_canary_binds_the_batch_to_the_canary_code_bundle(monkeypatch):
    bank, oracles, sha_ = section = composition_section()
    scope = ss.declare_scope(bank, signatures=["S"])
    real = canary.run_canary_on
    monkeypatch.setattr(canary, "run_canary_on",
                        lambda *a, **k: dict(real(*a, **k), code_bundle_sha256="d" * 64))
    with pytest.raises(ValueError, match="code bundle differs"):
        canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META, **_bound(section))


def test_score_with_canary_refuses_a_batch_the_canary_did_not_cover():
    bank, oracles, sha_ = section = composition_section()
    options = _bound(section)
    scope = ss.declare_scope(bank, signatures=["S"])
    with pytest.raises(ValueError, match="did not cover this batch: S"):
        canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META, signatures=["R"],
                                 **options)
    boundary = ss.declare_scope(bank, tag="boundary", diagnostic=True)  # C8: boundary rows need the flag
    with pytest.raises(ValueError, match="did not cover this batch: tag 'boundary'"):
        canary.score_with_canary([section], bank, boundary, _responses(bank, boundary, oracles), META, **options)
    every_tag = ss.declare_scope(bank, tag=None, signatures=["S"], diagnostic=True)
    with pytest.raises(ValueError, match="did not cover this batch: tag None"):
        canary.score_with_canary([section], bank, every_tag, _responses(bank, every_tag, oracles), META, **options)
    both = ss.declare_scope(bank, signatures=["R", "S"])
    out = canary.score_with_canary([section], bank, both, _responses(bank, both, oracles), META, **options)
    assert out["scoring"]["scope"]["rows_by_signature"] == {"R": 2, "S": 4}
    assert {k: v["binding"] for k, v in out["scoring"]["scope_binding"]["signatures"].items()} == {"R": "full",
                                                                                                    "S": "full"}
    legacy = legacy_section()  # 2 of 8 families, so its canary FAILs e; coverage is checked directly
    guard_receipt = canary.run_canary_on([legacy])
    for scope_of_guard in (ss.declare_scope(legacy[0], tag="preservation", families=["countdown"]),
                           ss.declare_scope(legacy[0], tag=None)):
        assert canary.uncovered(guard_receipt, legacy[0], scope_of_guard) == []
    composition_receipt = canary.run_canary_on([section], signatures=["RS"], **{
        k: options[k] for k in ("scope_receipts", "legacy_exemption")})
    assert canary.uncovered(composition_receipt, bank, ss.declare_scope(bank, signatures=["RS"])) == []
    assert canary.uncovered(composition_receipt, bank, both) == ["R", "S"]


def test_score_with_canary_refuses_to_score_after_a_failed_canary(monkeypatch):
    bank, oracles, sha_ = section = composition_section()
    scope = ss.declare_scope(bank, signatures=["S"])
    monkeypatch.setattr(ss, "reward", lambda row, response: 0)
    with pytest.raises(canary.CanaryFailed, match="batch was not scored") as caught:
        canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META, **_bound(section))
    assert caught.value.receipt["overall"] == "FAIL"
    assert "a_references is FAIL, required PASS" in caught.value.receipt["overall_failures"]
    # without the exemption, a canary with no legacy section FAILs, so nothing is scored
    monkeypatch.undo()
    with pytest.raises(canary.CanaryFailed, match="e_legacy_families is NOT_RUN"):
        canary.score_with_canary([section], bank, scope, _responses(bank, scope, oracles), META,
                                 **_bound(section, legacy_exemption=False))


# ---------------------------------------------------------------- refusals before any file

@pytest.mark.parametrize("role,kwargs,message", [
    ("test", {}, "at_freeze"),
    ("guard_test", {}, "at_freeze"),
    ("challenge", {"at_freeze": True}, "not admitted"),
    ("guard_challenge", {"at_freeze": True}, "not admitted"),
])
def test_run_canary_refuses_sealed_roles_before_any_file(tmp_path, role, kwargs, message):
    with pytest.raises(ValueError, match=message):
        canary.run_canary(tmp_path, role, **kwargs)
    with pytest.raises(ValueError, match=message):
        canary.load_oracles(tmp_path, role, **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("role,kwargs,message", [
    ("dev", {}, r"scope receipt \(C8\): dev"),
    ("calibration", {}, r"scope receipt \(C8\): calibration"),
    ("dev", {"scope_receipt": "receipt"}, r"scope receipt \(C8\): dev"),
    ("test", {"at_freeze": True, "scope_receipt": {}}, "at-freeze canary needs expected_code_bundle"),
    ("guard_test", {"at_freeze": True}, "at-freeze canary needs expected_code_bundle"),
    ("dev", {"scope_receipt": {}, "expected_code_bundle": "x"}, "lowercase sha256 hex digest"),
    ("dev", {"scope_receipt": {}, "legacy_exemption": True}, "only for a canary without a legacy section"),
    ("guard_dev", {"legacy_exemption": True}, "only for a canary without a legacy section"),
    ("dev", {"scope_receipt": {}, "include_legacy": False, "legacy_exemption": "yes"}, "must be a bool"),
    ("guard_dev", {"scope_receipt": {"signatures": {}}}, "composition sections"),
])
def test_run_canary_refusals_before_any_file(tmp_path, role, kwargs, message):
    with pytest.raises(ValueError, match=message):
        canary.run_canary(tmp_path, role, **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("role", ["test", "guard_test"])
def test_cli_refuses_sealed_roles_without_at_freeze(tmp_path, role):
    with pytest.raises(SystemExit, match="requires --at-freeze"):
        canary.main(["--data", str(tmp_path), "--role", role, "--out", str(tmp_path / "canary.json")])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(SystemExit):
        canary.main(["--data", str(tmp_path), "--role", "challenge", "--out", str(tmp_path / "canary.json")])


@pytest.mark.parametrize("role", ["test", "guard_test"])
def test_cli_at_freeze_requires_the_expected_code_bundle(tmp_path, role):
    """TA-03: an --at-freeze run is refused without --expected-code-bundle, before any data read."""
    receipt = tmp_path / "scope.json"
    with pytest.raises(SystemExit, match="--at-freeze requires --expected-code-bundle"):
        canary.main(["--data", str(tmp_path), "--role", role, "--at-freeze", "--scope-receipt", str(receipt),
                     "--out", str(tmp_path / "canary.json")])
    assert list(tmp_path.iterdir()) == []


def test_cli_refusals_before_any_data_read(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "out" / "canary.json"
    scope = ["--scope-receipt", str(tmp_path / "scope.json")]
    with pytest.raises(SystemExit, match=r"role dev requires --scope-receipt \(C8\)"):
        canary.main(["--data", str(data), "--role", "dev", "--out", str(out)])
    with pytest.raises(SystemExit, match=r"role calibration requires --scope-receipt \(C8\)"):
        canary.main(["--data", str(data), "--role", "calibration", "--out", str(out)])
    with pytest.raises(SystemExit, match="pins a --scope-receipt"):
        canary.main(["--data", str(data), "--role", "guard_dev", "--scope-receipt-sha256", "0" * 64, "--out", str(out)])
    with pytest.raises(SystemExit, match="--exempt-legacy is only for a composition role run with --no-legacy"):
        canary.main(["--data", str(data), "--role", "dev", *scope, "--exempt-legacy", "--out", str(out)])
    with pytest.raises(SystemExit, match="--exempt-legacy is only for a composition role run with --no-legacy"):
        canary.main(["--data", str(data), "--role", "guard_dev", "--exempt-legacy", "--out", str(out)])
    with pytest.raises(SystemExit, match="data root not found"):
        canary.main(["--data", str(tmp_path / "absent"), "--role", "dev", *scope, "--out", str(out)])
    with pytest.raises(SystemExit, match="protected output path"):
        canary.main(["--data", str(data), "--role", "dev", *scope, "--out", str(data / "canary.json")])
    with pytest.raises(SystemExit, match="scope receipt is never read from the data root"):
        canary.main(["--data", str(data), "--role", "dev", "--scope-receipt", str(data / "scope.json"),
                     "--out", str(out)])
    with pytest.raises(SystemExit, match="reference receipt is never read from the data root"):
        canary.main(["--data", str(data), "--role", "guard_dev", "--reference-receipt", str(data / "ref.json"),
                     "--out", str(out)])
    with pytest.raises(SystemExit, match="legacy-only"):
        canary.main(["--data", str(data), "--role", "guard_dev", "--no-legacy", "--out", str(out)])
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="create-only"):
        canary.main(["--data", str(data), "--role", "dev", *scope, "--out", str(existing)])
    assert list(data.iterdir()) == [] and not out.exists()


def test_cli_pins_and_strictly_parses_the_scope_receipt_before_any_data_read(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    receipt = tmp_path / "scope.json"
    receipt.write_bytes(b'{"signatures": {}, "signatures": {}}')
    with pytest.raises(SystemExit, match="scope receipt file sha differs"):
        canary.main(["--data", str(data), "--role", "dev", "--scope-receipt", str(receipt),
                     "--scope-receipt-sha256", "0" * 64, "--out", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit, match="duplicate"):
        canary.main(["--data", str(data), "--role", "dev", "--scope-receipt", str(receipt),
                     "--out", str(tmp_path / "out.json")])
    assert list(data.iterdir()) == []


# ---------------------------------------------------------------- data-backed (DEV, CALIBRATION, guard_dev)

def _q(name):
    return json.loads((PACKAGE / "evidence" / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_receipt(comp_rs_data):
    # The committed DEV canary receipt is the reference: check (c) must reproduce its derangement counts.
    return canary.run_canary(comp_rs_data, "dev", scope_receipt=_q("Q1_DEV_SCOPE.json"),
                             reference_receipt=_q("S0PRE_CANARY_DEV.json"))


def test_dev_canary_passes(dev_receipt):
    receipt = dev_receipt
    assert receipt["overall"] == "PASS", failures_of(receipt)
    assert statuses(receipt) == dict.fromkeys(LETTERS, "PASS") and receipt["overall_failures"] == []
    dev, guard = receipt["sections"]
    assert (dev["role"], guard["role"]) == ("dev", "guard_dev") and dev["absent_signatures"] == []
    q1 = _q("Q1_DEV_SCOPE.json")
    assert dev["bank_sha256"] == q1["canonical_dev_sha256"] and dev["scope_receipt_sha256"] == digest(q1)
    for sig in ("R", "S", "RS"):  # the canary's scopes are the frozen Q1 DEV scopes
        assert dev["scopes"][sig]["ids_sha256"] == q1["signatures"][sig]["ids_sha256"]
        assert dev["scopes"][sig]["parents_sha256"] == q1["signatures"][sig]["parents_sha256"]
        assert dev["scopes"][sig]["n"] == 250 and dev["scopes"][sig]["filter"]["tag"] == "core"
    assert guard["bank_sha256"] == _q("Q0_RECEIPT.json")["guard_dev"]["bank_sha256"]
    for record in receipt["checks"]:  # C8: DEV passes bound to Q1, guard passes to the bank sha
        if "scope" in record:
            expected = ("scope_receipt", digest(q1)) if record["section"] == "dev" else ("guard_bank_sha256", None)
            assert (record["scope_binding"], record["scope_receipt_sha256"]) == expected
    e = receipt["summary"]["e_legacy_families"]
    assert e["families_missing"] == [] and e["families_incomplete"] == []
    assert all(entry["complete"] for entry in e["by_section"]["guard_dev"].values())
    table = receipt["summary"]["b_composition_signatures"]
    assert table["signatures_incomplete"] == [] and list(table["by_section"]["dev"]) == ["R", "S", "RS"]
    # (a) and (b): every reference of every core row, bare and in each applicable wrapper
    # (the counts are those of the committed S0PRE_CANARY_DEV.json code-fence passes: 750 references per
    # core signature, 2,419 guard_dev references; (a) used to score only references[0], 250 and 1,200)
    scored = {sig: sum(r["scope"]["n"] for r in records(receipt, "wrappers", sig, "code_fence")) for sig in ("R", "S", "RS")}
    assert scored == {"R": 750, "S": 750, "RS": 750}
    for sig in ("R", "S", "RS"):
        assert sum(r["scope"]["n"] for r in records(receipt, "references", sig)) == scored[sig]
        for wrapper in ("trailing_period", "bare_list", "extra_field"):
            assert sum(r["scope"]["n"] for r in records(receipt, "wrappers", sig, wrapper)) == scored[sig]
    guard_refs = sum(r["scope"]["n"] for r in records(receipt, "references") if r["section"] == "guard_dev")
    assert guard_refs == 2419
    assert receipt["summary"]["a_references"]["rows_scored"] == 2250 + 2419
    # (c) counts measured by this canary on 2026-09-26; the direct precomputation and the committed
    # reference receipt S0PRE_CANARY_DEV.json both agree.
    derangement = {r["signature"]: r["observed"]["strict"] for r in records(receipt, "derangement")}
    assert {k: derangement[k] for k in ("R", "S", "RS")} == {"R": 0, "S": 14, "RS": 4}
    assert all(r["observed"]["strict"] == r["precomputed"]["strict"] == r["reference"]["strict"]
               for r in records(receipt, "derangement"))
    assert receipt["reference_receipt"]["derangements_compared"] == 3 + 8
    assert receipt["reference_receipt"]["sha256"] == digest(_q("S0PRE_CANARY_DEV.json"))
    # (d)
    (blind,) = records(receipt, "lookup_blind")
    assert blind["observed"]["rs_class"] == {"R_BLIND_S_VALID": 250} and blind["observed"]["strict"] == 0
    assert dev["lookup_blind"] == {"rows": 250, "with_certificate": 250, "without_certificate": 0}
    skipped = {(r["signature"], r["wrapper"]) for r in receipt["checks"] if r["status"] == "NOT_APPLICABLE"}
    assert skipped == {("R", "bare_unknown"), ("S", "bare_unknown"), ("RS", "bare_unknown"),
                       ("legacy:countdown", "bare_list")}
    assert receipt["code_bundle_sha256"] == bundle()
    assert receipt["model_output_involved"] is False and receipt["test_challenge_read"] is False
    assert receipt["sealed_roles_read"] == [] and receipt["legacy_exemption"] is False


def test_calibration_canary_passes(comp_rs_data):
    frozen = _q("S0PRE_CALIBRATION_SCOPE.json")
    receipt = canary.run_canary(comp_rs_data, "calibration", signatures=["R", "S", "P", "RS"], scope_receipt=frozen,
                                reference_receipt=_q("S0PRE_CANARY_CALIBRATION.json"))
    assert receipt["overall"] == "PASS", failures_of(receipt)
    calibration, guard = receipt["sections"]
    assert calibration["absent_signatures"] == ["RS"] and guard["role"] == "guard_dev"
    assert calibration["scope_receipt_sha256"] == digest(frozen)
    assert receipt["summary"]["d_lookup_blind"]["status"] == "NOT_RUN"  # CALIBRATION has no RS
    q0 = _q("Q0_RECEIPT.json")["calibration"]
    for sig in ("R", "S"):
        assert calibration["scopes"][sig]["ids_sha256"] == q0[sig]["ids_sha256"]
    for sig in ("R", "S", "P"):
        assert calibration["scopes"][sig]["n"] == 150
    derangement = {r["signature"]: r["observed"]["strict"] for r in records(receipt, "derangement")
                   if r["section"] == "calibration"}
    assert derangement == {"R": 0, "S": 18, "P": 0}
    assert receipt["reference_receipt"]["derangements_compared"] == 3 + 8
    with pytest.raises(ValueError, match="lacks the bank sha key canonical_calibration_sha256"):
        canary.run_canary(comp_rs_data, "calibration", signatures=["R"], include_legacy=False,
                          legacy_exemption=True, scope_receipt=_q("Q1_DEV_SCOPE.json"))


def test_guard_dev_canary_passes(comp_rs_data):
    receipt = canary.run_canary(comp_rs_data, "guard_dev", reference_receipt=_q("S0PRE_CANARY_GUARD_DEV.json"))
    assert receipt["overall"] == "PASS" and len(receipt["sections"]) == 1, failures_of(receipt)
    (guard,) = receipt["sections"]
    assert guard["families_present"] == list(canary.LEGACY_FAMILIES) and guard["rows_covered"] == 1200
    assert all(scope["n"] == 150 for scope in guard["scopes"].values()) and len(guard["scopes"]) == 8
    assert receipt["summary"]["e_legacy_families"]["status"] == "PASS"
    assert receipt["summary"]["b_composition_signatures"]["status"] == "NOT_RUN"
    assert receipt["reference_receipt"]["derangements_compared"] == 8


def test_canary_fails_when_scorer_reward_is_sabotaged(comp_rs_data, dev_receipt, monkeypatch):
    monkeypatch.setattr(ss, "reward", lambda row, response: 0)  # corpus.reward inside the scorer
    receipt = canary.run_canary(comp_rs_data, "dev", scope_receipt=_q("Q1_DEV_SCOPE.json"))
    assert receipt["overall"] == "FAIL"
    assert receipt["summary"]["a_references"]["status"] == "FAIL"
    references = receipt["summary"]["a_references"]
    passes = records(receipt, "references")
    assert len(passes) > 3 + 8  # every reference index of every R/S/RS scope and family
    assert (references["failed"], references["passes"]) == (len(passes), 0)
    assert all("disagrees with corpus.reward" in r["error"] for r in passes)
    assert receipt["summary"]["e_legacy_families"]["families_incomplete"] == sorted(canary.LEGACY_FAMILIES)
    assert [r.get("precomputed") for r in receipt["checks"]] == [r.get("precomputed") for r in dev_receipt["checks"]]


def test_guard_dev_canary_fails_when_one_family_is_mislabelled(comp_rs_data, monkeypatch):
    original = ss.score_row

    def score_row(row, response):
        result = dict(original(row, response))
        if row["verifier"]["spec"].get("family") == "countdown" and result["normalization"] == "code_fence+answer_object":
            result["normalization"] = "answer_object"
        return result

    monkeypatch.setattr(ss, "score_row", score_row)
    receipt = canary.run_canary(comp_rs_data, "guard_dev")
    assert receipt["overall"] == "FAIL"
    assert statuses(receipt)["b_wrappers"] == "FAIL" and statuses(receipt)["a_references"] == "PASS"
    e = receipt["summary"]["e_legacy_families"]
    assert e["status"] == "FAIL" and e["families_incomplete"] == ["countdown"]
    assert e["by_section"]["guard_dev"]["countdown"]["wrappers"]["code_fence"] == "FAIL"


def test_load_oracles_verifies_the_manifest_and_the_file(comp_rs_data, tmp_path):
    oracles, file_sha = canary.load_oracles(comp_rs_data, "dev")
    manifest = json.loads((comp_rs_data / "manifest.json").read_bytes())
    assert file_sha == manifest["files"]["oracles/dev.jsonl"]["sha256"] and len(oracles) == 4150
    (tmp_path / "oracles").mkdir()
    (tmp_path / "oracles" / "dev.jsonl").write_bytes(b'{"id":"x"}\n')
    (tmp_path / "manifest.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="manifest drift"):
        canary.load_oracles(tmp_path, "dev")
    (tmp_path / "manifest.json").write_bytes((comp_rs_data / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="data file drift: oracles/dev.jsonl"):
        canary.load_oracles(tmp_path, "dev")


def test_cli_writes_a_create_only_receipt(comp_rs_data, tmp_path, capsys):
    out = tmp_path / "canary_guard_dev.json"
    reference = PACKAGE / "evidence" / "S0PRE_CANARY_GUARD_DEV.json"
    assert canary.main(["--data", str(comp_rs_data), "--role", "guard_dev", "--reference-receipt", str(reference),
                        "--expected-code-bundle", bundle(), "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    raw = out.read_bytes()
    assert b"\r" not in raw and printed["overall"] == "PASS"
    receipt = json.loads(raw)
    assert receipt["overall"] == "PASS" and receipt["role"] == "guard_dev"
    assert receipt["reference_receipt"]["derangements_compared"] == 8
    assert receipt["code_check"]["expected"] == bundle()
    assert printed["receipt_sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(SystemExit, match="create-only"):
        canary.main(["--data", str(comp_rs_data), "--role", "guard_dev", "--out", str(out)])
    assert out.read_bytes() == raw
