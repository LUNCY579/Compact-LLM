"""Scorer canary (COMP-RS-001 Amendment 1 section 8.5; clarification C8).

Every check scores constructed responses THROUGH scoped_scorer.score_scope, the single scoring
entrypoint. An independent path precomputes each row's strict outcome by calling the unchanged
verifiers directly (maxrl_composable_skills_v2.verifier.score, maxrl_verifiable_data_v1.verifier.score),
never corpus.reward, content_scoring or the scorer. Checks:
(a) EVERY oracle reference (each reference index in turn, unwrapped) scores strict n/n through the
    scorer on every scoped core row and on every legacy family;
(b) each reference in each APPLICABLE wrapper (code fence, trailing period, bare list for list
    answers, the non-candidate extra field {"answer":...,"reason":"ok"}; bare UNKNOWN only for
    composition UNKNOWN references, which core rows lack) scores strict 0/n and content n/n under the
    registered normalization category;
(b, per composition signature) code fence, trailing period, bare list and extra field must each PASS
    on every scoped composition signature; only bare UNKNOWN may be NOT_APPLICABLE there;
(c) within-signature derangement (rows sorted by id; row i gets row i+1's references[0], cyclically)
    reproduces the precomputed strict count and ids. The same-run precomputation calls the same
    verifier as the scorer, so it cannot see a verifier regression: pass reference_receipt (an earlier
    PASS canary receipt of the same banks) and each derangement's strict count and ids must also equal
    that receipt's;
(d) on core RS rows, a constructed lookup-blind certificate (the lexicographically first length-k
    distinct-index subset over all indices that hits the target and is not inside the resolved set)
    scores strict 0 and classifies R_BLIND_S_VALID;
(e) the legacy section covers all 8 families: references, and every applicable wrapper, per family.
overall is PASS only if every required check is PASS (a, b, c; the per-signature table when a
composition section is present; d when core RS rows are scoped; e unless an explicit legacy exemption
is passed, which is recorded), no check FAILs, one scorer code bundle served every pass, and canary.py
is unchanged since import.

Scope receipts (C8). Every composition section (dev, calibration, test) must come with its frozen
section 1 scope receipt; every scope is bound to it before anything is scored, and every scoring pass
hands it to score_scope. Guard sections are bound by their manifest-checked bank sha (the scorer's rule).

score_with_canary runs the canary and then one scored batch on the same loaded banks, bound to the
canary's code bundle (section 8.5: "in the same pass as every scored batch"). It needs
expected_code_bundle (the S1/S2 scorer bundle), and so does every at-freeze canary.
No model output is involved: every response is built from oracle references or from the spec.

python -m experiments.maxrl_comp002_pretraining_v1.canary --data DATA --role ROLE --out NEW.json [options]
  --role                  calibration, dev or guard_dev; test and guard_test only with --at-freeze;
                          challenge banks are never admitted
  --scope-receipt FILE    frozen section 1 scope receipt of --role: required for calibration, dev and test
                          (C8), refused for guard roles; never read from the data root
  --scope-receipt-sha256  pin the sha256 of the scope receipt file's bytes
  --signatures S ...      composition signatures to scope (default R S RS; pass P explicitly)
  --no-legacy             skip the paired guard bank; overall then cannot PASS without --exempt-legacy
  --exempt-legacy         record an explicit exemption from check (e); only with --no-legacy
  --reference-receipt FILE  a PASS canary receipt of the same banks; check (c) must reproduce its counts
  --at-freeze             required for test and guard_test; requires --expected-code-bundle
  --expected-code-bundle  the scorer code bundle every pass must report (the S1/S2 bundle at TEST)
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

from experiments.maxrl_comp002_pretraining_v1 import scoped_scorer
from experiments.maxrl_comp002_pretraining_v1.content_scoring import EXPRESSION_FAMILY
from experiments.maxrl_comp002_pretraining_v1.preflight import MANIFEST, digest, sha
from experiments.maxrl_comp002_pretraining_v1.scoped_scorer import (
    FREEZE_ROLES, MAX_LINE, ROLES, Bank, admit_role, declare_scope, load_role, signature, write_receipt)
from experiments.maxrl_composable_skills_v2.contracts import strict_json
from experiments.maxrl_composable_skills_v2.verifier import concrete
from experiments.maxrl_composable_skills_v2.verifier import score as composition_score
from experiments.maxrl_verifiable_data_v1.specs import FAMILIES as LEGACY_FAMILIES
from experiments.maxrl_verifiable_data_v1.verifier import score as legacy_score

SCHEMA_VERSION = "1.1.0"
SIGNATURES = ("R", "S", "RS")
GUARD_ROLES = ("guard_dev", "guard_test")
PAIRED_GUARD = {"dev": "guard_dev", "calibration": "guard_dev", "test": "guard_test"}
LEGACY_TAG = "preservation"  # every guard row carries this tag; composition endpoints use 'core'
CHECKS = {"a_references": "references", "b_wrappers": "wrappers", "c_derangement": "derangement",
          "d_lookup_blind": "lookup_blind"}
COMPOSITION_WRAPPERS = ("code_fence", "trailing_period", "bare_list", "extra_field", "bare_unknown")
# Section 8.5 on core rows: each of these must PASS on every scoped composition signature (F7).
REQUIRED_COMPOSITION_WRAPPERS = ("code_fence", "trailing_period", "bare_list", "extra_field")
LEGACY_WRAPPERS = ("code_fence", "trailing_period", "bare_list", "extra_field")  # no legacy bare UNKNOWN
CATEGORY = {"reference": "answer_object", "code_fence": "code_fence+answer_object",
            "trailing_period": "trailing_period+answer_object", "bare_list": "bare_list",
            "extra_field": "extra_object_fields+answer_object", "bare_unknown": "bare_unknown"}
BLIND_CLASS = "R_BLIND_S_VALID"
CANARY_TOKENS = 1  # the scorer's response schema needs a token count; nothing was decoded
CANARY_STOP = "eos"
CANARY_META = {key: hashlib.sha256(("COMP-RS-001 scorer canary: no " + what).encode("ascii")).hexdigest()
               for key, what in (("model_sha256", "model"), ("tokenizer_sha256", "tokenizer"),
                                 ("decoding_sha256", "decoding"))}
EXAMPLES = 20
MAX_REFERENCE_RECEIPT = 1 << 24
HEX = frozenset("0123456789abcdef")
_IMPORT_CANARY_SHA256 = sha(Path(__file__).resolve())  # the canary code that runs, hashed at import


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def reference_answer(reference: str):
    """The answer inside one oracle reference ({"answer": ...} and nothing else)."""
    obj = strict_json(reference, 4096)
    if type(obj) is not dict or set(obj) != {"answer"}:
        raise ValueError("malformed oracle reference")
    return obj["answer"]


def applicable(wrapper: str, answer, legacy: bool) -> bool:
    """Section 8.5 applicability: bare list for list answers, bare UNKNOWN for composition UNKNOWN."""
    if wrapper not in CATEGORY or wrapper == "reference":
        raise ValueError("unknown wrapper: " + repr(wrapper))
    if wrapper == "bare_list":
        return type(answer) is list
    if wrapper == "bare_unknown":
        return not legacy and type(answer) is str and answer == "UNKNOWN"
    return True


def wrap(wrapper: str, reference: str, answer) -> str:
    """One reference in one content wrapper; strict must reject it and content must accept it."""
    if wrapper == "code_fence":
        return "```json\n" + reference + "\n```"
    if wrapper == "trailing_period":
        return reference + "."
    if wrapper == "bare_list":
        return _compact(answer)
    if wrapper == "extra_field":
        return _compact({"answer": answer, "reason": "ok"})
    if wrapper == "bare_unknown":
        return "UNKNOWN"
    raise ValueError("unknown wrapper: " + repr(wrapper))


def direct_strict(row: dict, response: str) -> int:
    """Strict outcome from the verifiers themselves (not corpus.reward, not the scorer)."""
    verifier = row["verifier"]
    if verifier["kind"] == "composition":
        return int(composition_score(verifier["spec"], response))
    if verifier["kind"] == "legacy":
        spec = verifier["spec"]
        return int(legacy_score(spec["family"], spec["problem"], response))
    raise ValueError("unknown verifier kind")


def reference_rs_class(answer) -> str:
    """The RS class an accepted reference must receive (UNKNOWN_ANSWER precedes RS_PASS)."""
    return "UNKNOWN_ANSWER" if type(answer) is str and answer == "UNKNOWN" else "RS_PASS"


def blind_certificate(spec: dict) -> list[int] | None:
    """First (lex) length-k distinct-index subset hitting the target and not inside the resolved set."""
    resolved, values = concrete(spec)
    resolved = resolved or set()  # an unresolvable lookup resolves nothing
    k, target = spec["selection"]["k"], spec["selection"]["target"]
    for chosen in combinations(range(len(values)), k):
        if sum(values[i] for i in chosen) == target and not set(chosen) <= resolved:
            return list(chosen)
    return None


def load_oracles(data, role: str, *, at_freeze: bool = False) -> tuple[dict, str]:
    """Oracle rows of one role keyed by id, and the file sha (manifest and per-file sha checked)."""
    admit_role(role, at_freeze)
    root = Path(data)
    manifest_raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST:
        raise ValueError("dataset manifest drift")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    relative = "oracles/" + role + ".jsonl"
    raw = (root / relative).read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    if relative not in manifest["files"] or file_sha != manifest["files"][relative]["sha256"]:
        raise ValueError("data file drift: " + relative)
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    oracles = {}
    for line in lines:
        oracle = strict_json(line.decode("utf-8"), MAX_LINE)
        if type(oracle) is not dict or type(oracle.get("id")) is not str:
            raise ValueError("malformed oracle row: " + role)
        if oracle["id"] in oracles:
            raise ValueError("duplicate oracle id: " + role)
        oracles[oracle["id"]] = oracle
    if len(oracles) != manifest["counts"][role]:
        raise ValueError("oracle count mismatch: " + role)
    return oracles, file_sha


def load_reference_receipt(path) -> dict:
    """Read an earlier canary receipt file for reference_receipt (duplicate keys and NaN refused)."""
    value = strict_json(Path(path).read_bytes().decode("utf-8"), MAX_REFERENCE_RECEIPT)
    if type(value) is not dict:
        raise ValueError("reference receipt schema")
    return value


def bind_oracles(bank: Bank, oracles: dict) -> None:
    """Every bank row has one oracle on the same parent with well-formed {"answer": ...} references."""
    if type(oracles) is not dict or set(oracles) != {row["id"] for row in bank}:
        raise ValueError("oracle ids differ from the bank")
    for row in bank:
        oracle = oracles[row["id"]]
        if oracle.get("parent_id") != row["parent_id"]:
            raise ValueError("oracle parent mismatch: " + row["id"])
        references = oracle.get("references")
        if type(references) is not list or not references or any(type(r) is not str for r in references):
            raise ValueError("oracle references malformed: " + row["id"])
        for reference in references:
            reference_answer(reference)


def canary_roles(role: str, include_legacy: bool = True, at_freeze: bool = False) -> list[str]:
    """The banks one canary reads, in order; every role is admitted before any file is touched."""
    admit_role(role, at_freeze)
    if role in GUARD_ROLES:
        if include_legacy is not True:
            raise ValueError("a guard bank is legacy-only: include_legacy must be True")
        return [role]
    if include_legacy is True:
        return [role, admit_role(PAIRED_GUARD[role], at_freeze)]
    if include_legacy is False:
        return [role]
    raise ValueError("include_legacy must be a bool")


def _signatures(signatures) -> list[str]:
    if type(signatures) not in (list, tuple) or not signatures:
        raise ValueError("signatures must be a non-empty list")
    if any(type(s) is not str or not s or set(s) - set("RUSP") for s in signatures):
        raise ValueError("signatures must be composition signatures")
    if len(set(signatures)) != len(signatures):
        raise ValueError("duplicate signatures")
    return list(signatures)


def _expected_bundle(value, at_freeze) -> str | None:
    """TA-03: an at-freeze canary must be told the scorer bundle it has to see (the S1/S2 one)."""
    if value is None:
        if at_freeze is True:
            raise ValueError("an at-freeze canary needs expected_code_bundle (the S1/S2 scorer bundle, section 8.5)")
        return None
    if type(value) is not str or len(value) != 64 or set(value) - HEX:
        raise ValueError("expected_code_bundle must be a lowercase sha256 hex digest")
    return value


def _scope_digest(scope: dict) -> dict:
    return {key: scope[key] for key in ("ids_sha256", "parents_sha256", "n", "parents",
                                        "one_row_per_parent_per_signature", "rows_by_signature", "filter")}


def _counts(rows: list, key: str) -> dict:
    return dict(sorted(Counter(r[key] for r in rows if r[key] is not None).items()))


def _reference_table(reference: dict, sections: list) -> dict:
    """(section role, signature) -> the reference receipt's scored derangement, bound to these banks.

    The reference must be a PASS canary receipt holding every section of this canary on the same bank,
    manifest and oracle file; anything else is refused before anything is scored.
    """
    if type(reference) is not dict or reference.get("kind") != "scorer_canary_receipt":
        raise ValueError("reference receipt is not a scorer canary receipt")
    if reference.get("overall") != "PASS":
        raise ValueError("reference receipt did not PASS")
    try:
        described = {entry["role"]: entry for entry in reference["sections"]}
        for bank, _, oracle_sha in sections:
            entry = described.get(bank.role)
            if entry is None:
                raise ValueError("reference receipt lacks section " + bank.role)
            if (entry["bank_sha256"], entry["manifest_sha256"], entry["oracle_sha256"]) != (
                    bank.bank_sha256, bank.manifest_sha256, oracle_sha):
                raise ValueError("reference receipt section differs from this bank: " + bank.role)
        table = {}
        for record in reference["checks"]:
            if record["check"] != "derangement" or record["status"] != "PASS":
                continue
            key = (record["section"], record["signature"])
            if key in table:
                raise ValueError("duplicate reference derangement: " + ":".join(key))
            table[key] = {"n": record["scope"]["n"], "ids_sha256": record["scope"]["ids_sha256"],
                          "strict": record["precomputed"]["strict"],
                          "strict_ids_sha256": record["precomputed"]["strict_ids_sha256"]}
    except (KeyError, TypeError):
        raise ValueError("reference receipt schema") from None
    return table


class _Run:
    """Accumulates per-pass records and the scorer code maps they report."""

    def __init__(self, scope_receipts: dict, reference: dict | None):
        self.records: list[dict] = []
        self.bundles: dict[str, dict] = {}
        self.scope_receipts = scope_receipts  # role -> frozen scope receipt (composition sections only)
        self.reference = reference            # _reference_table, or None without a reference receipt

    def skip(self, check: str, section: Bank, name: str, reason: str, wrapper: str | None = None) -> None:
        self.records.append({"check": check, "section": section.role, "signature": name, "wrapper": wrapper,
                             "reference_index": None, "status": "NOT_APPLICABLE", "reason": reason})

    def score(self, check: str, section: Bank, scope: dict, texts: dict, *, name: str, expected: dict,
              registered: dict, wrapper: str | None = None, reference_index: int | None = None,
              inner: dict | None = None, referenced: bool = False) -> None:
        """Score texts through score_scope; compare with the expected values and the direct verifiers.

        referenced: compare the direct precomputation with the reference receipt's (check (c)).
        """
        by_id = {row["id"]: row for row in section}
        direct = {i: direct_strict(by_id[i], texts[i]) for i in scope["ids"]}
        direct_ids = sorted(i for i, value in direct.items() if value)
        precomputed = {"strict": len(direct_ids), "strict_ids_sha256": digest(direct_ids)}
        if inner is not None:
            precomputed["reference_strict"] = sum(direct_strict(by_id[i], inner[i]) for i in scope["ids"])
        responses = [{"id": i, "prompt_hash": by_id[i]["prompt_hash"], "response": texts[i],
                      "generated_tokens": CANARY_TOKENS, "stop_reason": CANARY_STOP} for i in scope["ids"]]
        record = {"check": check, "section": section.role, "signature": name, "wrapper": wrapper,
                  "reference_index": reference_index, "scope": _scope_digest(scope),
                  "responses_sha256": digest(responses), "expected": expected, "precomputed": precomputed,
                  "observed": None, "error": None, "scoring_receipt_sha256": None, "code_bundle_sha256": None,
                  "scope_binding": None, "scope_receipt_sha256": None}
        failures = [f"precomputed {key} {precomputed[key]} != registered {value}"
                    for key, value in registered.items() if precomputed[key] != value]
        if referenced and self.reference is not None:
            frozen = self.reference.get((section.role, name))
            record["reference"] = frozen
            if frozen is None:
                failures.append("reference receipt has no scored derangement for this scope")
            else:
                if (frozen["n"], frozen["ids_sha256"]) != (scope["n"], scope["ids_sha256"]):
                    failures.append("reference receipt derangement scope differs")
                failures += [f"precomputed {key} {precomputed[key]} != reference receipt {frozen[key]}"
                             for key in ("strict", "strict_ids_sha256") if precomputed[key] != frozen[key]]
        try:
            receipt = scoped_scorer.score_scope(section, scope, responses, CANARY_META,
                                                scope_receipt=self.scope_receipts.get(section.role))
        except Exception as exc:  # a broken scorer is a recorded canary FAIL, never swallowed silently
            record["error"] = type(exc).__name__ + ": " + str(exc)
            failures.append("scorer error: " + record["error"])
        else:
            rows = receipt["rows"]
            observed_ids = sorted(r["id"] for r in rows if r["strict"])
            record["observed"] = {"strict": len(observed_ids), "content": sum(bool(r["content_pass"]) for r in rows),
                                  "normalization": _counts(rows, "normalization"),
                                  "rs_class": _counts(rows, "rs_class"), "subtype": _counts(rows, "subtype"),
                                  "strict_ids_sha256": digest(observed_ids)}
            record["scoring_receipt_sha256"] = digest(receipt)
            record["code_bundle_sha256"] = receipt["code_bundle_sha256"]
            record["scope_binding"] = receipt["scope_binding"]["kind"]
            record["scope_receipt_sha256"] = receipt["scope_receipt_sha256"]
            self.bundles[receipt["code_bundle_sha256"]] = receipt["code_sha256"]
            if [r["id"] for r in rows] != scope["ids"]:
                failures.append("scorer rows differ from the declared scope")
            for key, value in expected.items():
                if record["observed"][key] != value:
                    failures.append(f"{key}: expected {value!r}, observed {record['observed'][key]!r}")
            differing = sorted(set(observed_ids) ^ set(direct_ids))
            if differing:
                failures.append("strict differs from the direct verifiers on %d rows" % len(differing))
                record["differing_ids"] = differing[:EXAMPLES]
        record["status"] = "FAIL" if failures else "PASS"
        record["failures"] = failures
        self.records.append(record)


def _rs_classes(references) -> dict:
    return dict(sorted(Counter(reference_rs_class(reference_answer(r)) for r in references).items()))


def _part(section: Bank, scope: dict, ids) -> dict:
    """The scope itself, or the declared part of it holding exactly `ids` (same filter)."""
    if len(ids) == scope["n"]:
        return scope
    return declare_scope(section, tag=scope["filter"]["tag"], signatures=scope["filter"]["signatures"],
                         families=scope["filter"]["families"], ids=sorted(ids))


def _reference_passes(run: _Run, section: Bank, oracles: dict, scope: dict, name: str,
                      wrappers: tuple, legacy: bool, rs: bool) -> None:
    """Checks (a) and (b) for one signature or family scope: every reference index, bare and wrapped."""
    ids = scope["ids"]
    depth = max(len(oracles[i]["references"]) for i in ids)
    for index in range(depth):  # (a): index 0 holds every row (bind_oracles: no row lacks a reference)
        chosen = {i: oracles[i]["references"][index] for i in ids if index < len(oracles[i]["references"])}
        n = len(chosen)
        expected = {"strict": n, "content": n, "normalization": {CATEGORY["reference"]: n}, "subtype": {}}
        if rs:
            expected["rs_class"] = _rs_classes(chosen.values())
        run.score("references", section, _part(section, scope, chosen), chosen, name=name, expected=expected,
                  registered={"strict": n}, reference_index=index)
    for wrapper in wrappers:
        seen = 0
        for index in range(depth):
            chosen = {}
            for i in ids:
                references = oracles[i]["references"]
                if index < len(references) and applicable(wrapper, reference_answer(references[index]), legacy):
                    chosen[i] = references[index]
            if not chosen:
                continue
            seen += len(chosen)
            texts = {i: wrap(wrapper, ref, reference_answer(ref)) for i, ref in chosen.items()}
            n = len(chosen)
            expected = {"strict": 0, "content": n, "normalization": {CATEGORY[wrapper]: n}, "subtype": {}}
            if rs:
                expected["rs_class"] = _rs_classes(chosen.values())
            run.score("wrappers", section, _part(section, scope, chosen), texts, name=name, expected=expected,
                      registered={"strict": 0, "reference_strict": n}, wrapper=wrapper,
                      reference_index=index, inner=chosen)
        if not seen:
            run.skip("wrappers", section, name, "no reference admits this wrapper", wrapper)


def _derangement_pass(run: _Run, section: Bank, oracles: dict, scope: dict, name: str) -> None:
    """Check (c): row i (ids sorted) gets row i+1's references[0], cyclically.

    The strict count is precomputed on the direct verifiers and, with a reference receipt, must also
    equal that receipt's count and ids. A deranged reference is still a bare answer object, so its
    content verdict equals its strict verdict and it gets no subtype.
    """
    ids = scope["ids"]
    if len(ids) < 2:
        run.skip("derangement", section, name, "fewer than 2 rows: no derangement exists")
        return
    texts = {ids[at]: oracles[ids[(at + 1) % len(ids)]]["references"][0] for at in range(len(ids))}
    by_id = {row["id"]: row for row in section}
    count = sum(direct_strict(by_id[i], texts[i]) for i in ids)
    expected = {"strict": count, "content": count, "normalization": {CATEGORY["reference"]: len(ids)},
                "subtype": {}}
    run.score("derangement", section, scope, texts, name=name, expected=expected, registered={},
              referenced=True)


def _blind_pass(run: _Run, section: Bank, scope: dict, name: str) -> dict:
    """Check (d): constructed lookup-blind certificates on core RS rows."""
    by_id = {row["id"]: row for row in section}
    certificates = {}
    for i in scope["ids"]:
        chosen = blind_certificate(by_id[i]["verifier"]["spec"])
        if chosen is not None:
            certificates[i] = _compact({"answer": chosen})
    if not certificates:
        run.skip("lookup_blind", section, name, "no lookup-blind certificate exists on any scoped RS row")
    else:
        n = len(certificates)
        expected = {"strict": 0, "content": 0, "normalization": {CATEGORY["reference"]: n}, "subtype": {},
                    "rs_class": {BLIND_CLASS: n}}
        run.score("lookup_blind", section, _part(section, scope, certificates), certificates, name=name,
                  expected=expected, registered={"strict": 0})
    return {"rows": len(scope["ids"]), "with_certificate": len(certificates),
            "without_certificate": len(scope["ids"]) - len(certificates)}


def _composition_scopes(bank: Bank, signatures: list, scope_receipt: dict) -> tuple[dict, list]:
    """The core scope of each requested signature present in the bank, each bound to the scope receipt."""
    present = {signature(row) for row in bank if row["kind"] == "composition" and row["tag"] == "core"}
    absent = [s for s in signatures if s not in present]
    if len(absent) == len(signatures):
        raise ValueError("no requested signature has core rows in " + bank.role)
    scopes = {}
    for name in signatures:
        if name in absent:
            continue
        scope = declare_scope(bank, signatures=[name])
        scoped_scorer.assert_scope_receipt(scope, scope_receipt, name)
        scopes[name] = scope
    return scopes, absent


def _composition_section(run: _Run, bank: Bank, oracles: dict, signatures: list, plan: tuple,
                         scope_receipt: dict) -> dict:
    scopes, absent = plan
    blind = None
    for name, scope in scopes.items():
        _reference_passes(run, bank, oracles, scope, name, COMPOSITION_WRAPPERS, legacy=False, rs=name == "RS")
        _derangement_pass(run, bank, oracles, scope, name)
        if name == "RS":
            blind = _blind_pass(run, bank, scope, name)
    return {"kind": "composition", "signatures_requested": signatures, "absent_signatures": absent,
            "scopes": {name: _scope_digest(scope) for name, scope in scopes.items()},
            "scope_receipt_sha256": digest(scope_receipt), "lookup_blind": blind}


def _legacy_scopes(bank: Bank) -> dict:
    """One preservation scope per legacy family present; together they must cover the guard bank."""
    if any(row["kind"] != "legacy" or row["tag"] != LEGACY_TAG for row in bank):
        raise ValueError("a guard bank holds only legacy preservation rows: " + bank.role)
    present = {row["verifier"]["spec"]["family"] for row in bank}
    scopes = {family: declare_scope(bank, tag=LEGACY_TAG, families=[family])
              for family in LEGACY_FAMILIES if family in present}
    if sum(scope["n"] for scope in scopes.values()) != len(bank):
        raise ValueError("legacy family scopes do not cover the guard bank: " + bank.role)
    return scopes


def _legacy_section(run: _Run, bank: Bank, oracles: dict, scopes: dict) -> dict:
    for family, scope in scopes.items():
        name = "legacy:" + family
        _reference_passes(run, bank, oracles, scope, name, LEGACY_WRAPPERS, legacy=True, rs=False)
        _derangement_pass(run, bank, oracles, scope, name)
    return {"kind": "legacy", "families_present": list(scopes),
            "families_missing": [f for f in LEGACY_FAMILIES if f not in scopes],
            "scopes": {"legacy:" + family: _scope_digest(scope) for family, scope in scopes.items()},
            "rows_covered": sum(scope["n"] for scope in scopes.values())}


def _status(records: list) -> str:
    states = {r["status"] for r in records}
    if "FAIL" in states:
        return "FAIL"
    if "PASS" in states:
        return "PASS"
    return "NOT_APPLICABLE" if states else "NOT_RUN"


def _family_table(records: list, role: str) -> dict:
    """Per family of one legacy section: reference, wrapper and derangement states, and completeness.

    Complete means references PASS and every wrapper PASS, except bare_list on the countdown family
    (its answer is a string), which must be NOT_APPLICABLE.
    """
    table = {}
    for family in LEGACY_FAMILIES:
        mine = [r for r in records if r["section"] == role and r["signature"] == "legacy:" + family]
        wrappers = {w: _status([r for r in mine if r["check"] == "wrappers" and r["wrapper"] == w])
                    for w in LEGACY_WRAPPERS}
        entry = {"references": _status([r for r in mine if r["check"] == "references"]), "wrappers": wrappers,
                 "derangement": _status([r for r in mine if r["check"] == "derangement"])}
        needed = {w: ("NOT_APPLICABLE" if w == "bare_list" and family == EXPRESSION_FAMILY else "PASS")
                  for w in LEGACY_WRAPPERS}
        entry["complete"] = (entry["references"] == "PASS" and wrappers == needed
                             and entry["derangement"] != "FAIL")
        table[family] = entry
    return table


def _signature_table(records: list, role: str, names: list) -> dict:
    """Per scoped signature of one composition section (F7): reference, wrapper and derangement states.

    Complete means references PASS; code_fence, trailing_period, bare_list and extra_field each PASS;
    bare_unknown PASS or NOT_APPLICABLE (the only wrapper core rows may lack); derangement not FAIL.
    """
    table = {}
    for name in names:
        mine = [r for r in records if r["section"] == role and r["signature"] == name]
        wrappers = {w: _status([r for r in mine if r["check"] == "wrappers" and r["wrapper"] == w])
                    for w in COMPOSITION_WRAPPERS}
        entry = {"references": _status([r for r in mine if r["check"] == "references"]), "wrappers": wrappers,
                 "derangement": _status([r for r in mine if r["check"] == "derangement"])}
        entry["complete"] = (entry["references"] == "PASS"
                             and all(wrappers[w] == "PASS" for w in REQUIRED_COMPOSITION_WRAPPERS)
                             and wrappers["bare_unknown"] in ("PASS", "NOT_APPLICABLE")
                             and entry["derangement"] != "FAIL")
        table[name] = entry
    return table


def _summary(records: list, described: list, legacy_exemption: bool) -> dict:
    summary = {}
    for letter, check in CHECKS.items():
        chosen = [r for r in records if r["check"] == check]
        summary[letter] = {"status": _status(chosen), "passes": sum(r["status"] == "PASS" for r in chosen),
                           "failed": sum(r["status"] == "FAIL" for r in chosen),
                           "not_applicable": sum(r["status"] == "NOT_APPLICABLE" for r in chosen),
                           "rows_scored": sum(r["scope"]["n"] for r in chosen if "scope" in r)}
        if summary[letter]["status"] == "NOT_RUN":
            summary[letter]["reason"] = "no scoped rows for this check"
    composition = [entry for entry in described if entry["kind"] == "composition"]
    if composition:
        by_section = {entry["role"]: _signature_table(records, entry["role"], list(entry["scopes"]))
                      for entry in composition}
        incomplete = sorted(role + ":" + name for role, table in by_section.items()
                            for name, entry in table.items() if not entry["complete"])
        summary["b_composition_signatures"] = {"status": "FAIL" if incomplete else "PASS",
                                               "wrappers_required": list(REQUIRED_COMPOSITION_WRAPPERS),
                                               "signatures_incomplete": incomplete, "by_section": by_section}
    else:
        summary["b_composition_signatures"] = {"status": "NOT_RUN", "reason": "no composition section"}
    legacy = [entry for entry in described if entry["kind"] == "legacy"]
    if not legacy:
        summary["e_legacy_families"] = {"status": "NOT_RUN", "exempt": legacy_exemption,
                                        "reason": "legacy exemption recorded: no legacy section" if legacy_exemption
                                        else "no legacy section"}
        return summary
    by_section = {entry["role"]: _family_table(records, entry["role"]) for entry in legacy}
    incomplete = sorted({family for table in by_section.values() for family, entry in table.items()
                         if not entry["complete"]})
    missing = sorted({f for entry in legacy for f in entry["families_missing"]})
    summary["e_legacy_families"] = {"status": "FAIL" if incomplete or missing else "PASS", "exempt": False,
                                    "families_required": list(LEGACY_FAMILIES), "families_missing": missing,
                                    "families_incomplete": incomplete, "by_section": by_section}
    return summary


def _overall(summary: dict, code_check: dict, described: list, legacy_exemption: bool) -> tuple[str, list[str]]:
    required = ["a_references", "b_wrappers"]
    if any(entry["kind"] == "composition" for entry in described):
        required.append("b_composition_signatures")  # F7: every required wrapper on every signature
    required.append("c_derangement")
    if any(entry.get("lookup_blind") is not None for entry in described):
        required.append("d_lookup_blind")  # core RS rows were scoped
    if not legacy_exemption:
        required.append("e_legacy_families")  # F8: no legacy section is a FAIL unless exempted
    reasons = [letter + " is " + summary[letter]["status"] + ", required PASS"
               for letter in required if summary[letter]["status"] != "PASS"]
    reasons += [letter + " FAIL" for letter, value in summary.items()
                if letter not in required and value["status"] == "FAIL"]
    if code_check["status"] != "PASS":
        reasons.append("code check FAIL")
    return ("FAIL" if reasons else "PASS"), reasons


def run_canary_on(sections: list, *, role: str | None = None, signatures=SIGNATURES, at_freeze: bool = False,
                  expected_code_bundle: str | None = None, scope_receipts: dict | None = None,
                  legacy_exemption: bool = False, reference_receipt: dict | None = None) -> dict:
    """The canary over already verified banks: sections = [(Bank from load_role, oracles by id, oracle sha)].

    A guard bank (guard_dev, guard_test) gets the legacy checks; any other bank gets the composition
    checks on its core rows of the requested signatures. role, if given, must be the first section's.
    scope_receipts maps EVERY composition section's role to its frozen section 1 scope receipt (C8);
    every scope is asserted against it (scoped_scorer.assert_scope_receipt) and every pass is scored
    with it. legacy_exemption=True records an explicit exemption from check (e) and is refused when a
    legacy section is present. reference_receipt: an earlier PASS canary receipt of the same banks whose
    derangement counts and ids check (c) must reproduce. at_freeze needs expected_code_bundle.
    Every argument, receipt and scope is checked before anything is scored.
    """
    signatures = _signatures(signatures)
    if type(sections) is not list or not sections:
        raise ValueError("no sections")
    if any(type(item) is not tuple or len(item) != 3 or not isinstance(item[0], Bank) for item in sections):
        raise ValueError("sections need (Bank from load_role, oracles by id, oracle sha)")
    for bank, _, _ in sections:
        admit_role(bank.role, at_freeze)
    if role is not None and sections[0][0].role != role:
        raise ValueError("role differs from the first section")
    if len({bank.role for bank, _, _ in sections}) != len(sections):
        raise ValueError("duplicate section role")
    expected_code_bundle = _expected_bundle(expected_code_bundle, at_freeze)
    if type(legacy_exemption) is not bool:
        raise ValueError("legacy_exemption must be a bool")
    guards = [bank.role for bank, _, _ in sections if bank.role in GUARD_ROLES]
    if legacy_exemption and guards:
        raise ValueError("a legacy exemption is only for a canary without a legacy section")
    composition = [bank.role for bank, _, _ in sections if bank.role not in GUARD_ROLES]
    scope_receipts = {} if scope_receipts is None else scope_receipts
    if type(scope_receipts) is not dict or any(type(v) is not dict for v in scope_receipts.values()):
        raise ValueError("scope_receipts maps a role to a scope receipt")
    if set(scope_receipts) - set(composition):
        raise ValueError("scope receipts bind composition sections of this canary only")
    lacking = [name for name in composition if name not in scope_receipts]
    if lacking:
        raise ValueError("composition section needs its frozen section 1 scope receipt (C8): " + ", ".join(lacking))
    reference = None if reference_receipt is None else _reference_table(reference_receipt, sections)
    plans = []
    for bank, oracles, oracle_sha in sections:  # bind everything before anything is scored
        bind_oracles(bank, oracles)
        if bank.role in GUARD_ROLES:
            plans.append(_legacy_scopes(bank))
        else:
            plans.append(_composition_scopes(bank, signatures, scope_receipts[bank.role]))
    run, described = _Run(dict(scope_receipts), reference), []
    for (bank, oracles, oracle_sha), plan in zip(sections, plans):
        if bank.role in GUARD_ROLES:
            body = _legacy_section(run, bank, oracles, plan)
        else:
            body = _composition_section(run, bank, oracles, signatures, plan, scope_receipts[bank.role])
        described.append({"role": bank.role, "bank_sha256": bank.bank_sha256,
                          "manifest_sha256": bank.manifest_sha256, "oracle_sha256": oracle_sha,
                          "rows": len(bank), **body})
    summary = _summary(run.records, described, legacy_exemption)
    if len(run.bundles) == 1:
        (bundle, code), source = next(iter(run.bundles.items())), "score_scope receipts"
    else:
        code = scoped_scorer.code_sha256()
        bundle, source = digest(code), "scoped_scorer.code_sha256()"
    failures = []
    if len(run.bundles) > 1:
        failures.append("scorer code changed during the canary")
    if not run.bundles:
        failures.append("no scoring pass completed")
    if expected_code_bundle is not None and expected_code_bundle != bundle:
        failures.append("scorer code bundle differs from the expected one")
    if sha(Path(__file__).resolve()) != _IMPORT_CANARY_SHA256:
        failures.append("canary source changed since import")
    code_check = {"status": "FAIL" if failures else "PASS", "failures": failures, "expected": expected_code_bundle,
                  "observed": bundle, "bundles_seen": sorted(run.bundles), "source": source}
    overall, reasons = _overall(summary, code_check, described, legacy_exemption)
    sealed = sorted(entry["role"] for entry in described if entry["role"] in FREEZE_ROLES)
    compared = sum(1 for r in run.records if r.get("reference") is not None)
    return {"schema_version": SCHEMA_VERSION, "kind": "scorer_canary_receipt", "experiment": "COMP-RS-001",
            "amendment": "COMP_RS_001_AMENDMENT_1.md section 8.5; clarification C8",
            "role": described[0]["role"] if role is None else role, "at_freeze": at_freeze is True,
            "signatures_requested": signatures, "legacy_included": bool(guards),
            "legacy_exemption": legacy_exemption, "manifest_sha256": MANIFEST, "sections": described,
            "reference_receipt": None if reference_receipt is None else {
                "sha256": digest(reference_receipt), "code_bundle_sha256": reference_receipt.get("code_bundle_sha256"),
                "derangements_compared": compared},
            "entrypoint": "experiments.maxrl_comp002_pretraining_v1.scoped_scorer.score_scope",
            "independent_path": "maxrl_composable_skills_v2.verifier.score and maxrl_verifiable_data_v1.verifier.score, "
                                "called directly per row",
            "run_meta": dict(CANARY_META),
            "run_meta_rule": "sha256 of fixed labels: no model, tokenizer or decoding was involved",
            "response_placeholders": {"generated_tokens": CANARY_TOKENS, "stop_reason": CANARY_STOP},
            "categories": dict(CATEGORY), "checks": run.records, "summary": summary,
            "code_sha256": code, "code_bundle_sha256": bundle, "code_check": code_check,
            "canary_code_sha256": _IMPORT_CANARY_SHA256,
            "model_output_involved": False, "model_outputs_read": False,
            "sealed_roles_read": sealed, "test_challenge_read": bool(sealed), "challenge_read": False,
            "overall": overall, "overall_failures": reasons}


def run_canary(data, role: str, *, signatures=SIGNATURES, include_legacy: bool = True, at_freeze: bool = False,
               expected_code_bundle: str | None = None, scope_receipt: dict | None = None,
               legacy_exemption: bool = False, reference_receipt: dict | None = None) -> dict:
    """Load and verify the role's bank, plus its paired guard bank when include_legacy (dev and
    calibration pair with guard_dev, test with guard_test; a guard role is its own legacy section),
    then run every canary check. scope_receipt is the frozen scope receipt of the (composition) role
    itself: required for calibration, dev and test (C8), refused for guard roles. Every role, the
    receipt, the exemption and the at-freeze bundle are checked before any file is opened."""
    roles = canary_roles(role, include_legacy, at_freeze)
    signatures = _signatures(signatures)
    _expected_bundle(expected_code_bundle, at_freeze)
    if role in GUARD_ROLES and scope_receipt is not None:
        raise ValueError("scope receipts bind composition sections of this canary only")
    if role not in GUARD_ROLES and type(scope_receipt) is not dict:
        raise ValueError("composition section needs its frozen section 1 scope receipt (C8): " + role)
    if type(legacy_exemption) is not bool:
        raise ValueError("legacy_exemption must be a bool")
    if legacy_exemption and (len(roles) != 1 or role in GUARD_ROLES):
        raise ValueError("a legacy exemption is only for a canary without a legacy section")
    sections = []
    for name in roles:
        bank = load_role(data, name, at_freeze=at_freeze)
        oracles, oracle_sha = load_oracles(data, name, at_freeze=at_freeze)
        sections.append((bank, oracles, oracle_sha))
    return run_canary_on(sections, role=role, signatures=signatures, at_freeze=at_freeze,
                         expected_code_bundle=expected_code_bundle,
                         scope_receipts=None if scope_receipt is None else {role: scope_receipt},
                         legacy_exemption=legacy_exemption, reference_receipt=reference_receipt)


class CanaryFailed(ValueError):
    """The canary did not PASS, so the batch was not scored; .receipt holds the canary receipt."""

    def __init__(self, receipt: dict):
        super().__init__("scorer canary FAIL: the batch was not scored (" + "; ".join(receipt["overall_failures"]) + ")")
        self.receipt = receipt


def uncovered(canary: dict, bank: Bank, scope: dict) -> list[str]:
    """What a scored scope holds that the canary never scored on this bank (empty when covered).

    A legacy section covers every row of its guard bank. A composition section covers only the core
    rows of the signatures it scoped, so a non-core scope or another signature is uncovered.
    """
    entry = next(s for s in canary["sections"] if s["role"] == bank.role)
    if entry["kind"] == "legacy":
        return []
    if scope["filter"]["tag"] != "core":
        return ["tag " + repr(scope["filter"]["tag"])]
    return sorted(set(scope["rows_by_signature"]) - set(entry["scopes"]))


def score_with_canary(sections: list, bank: Bank, scope: dict, responses: list, run_meta: dict, *,
                      signatures=SIGNATURES, at_freeze: bool = False, expected_code_bundle: str | None = None,
                      scope_receipts: dict | None = None, legacy_exemption: bool = False,
                      reference_receipt: dict | None = None) -> dict:
    """Section 8.5 'in the same pass as every scored batch': run the canary on the loaded banks, then
    score one batch through score_scope, bound to the canary's code bundle and, for a composition bank,
    to the same frozen scope receipt the canary used (scope_receipts[bank.role], C8). bank must be one
    of the canary's section Banks (the same object). expected_code_bundle (the S1/S2 scorer bundle) is
    required and checked before anything runs (TA-03). A canary that does not PASS raises CanaryFailed;
    a batch holding rows the canary did not cover (another signature, or a non-core tag) is refused."""
    if expected_code_bundle is None:
        raise ValueError("score_with_canary needs expected_code_bundle (the S1/S2 scorer bundle, section 8.5)")
    _expected_bundle(expected_code_bundle, at_freeze)
    if type(sections) is not list or not any(type(item) is tuple and item and item[0] is bank for item in sections):
        raise ValueError("the scored bank must be one of the canary's sections")
    canary = run_canary_on(sections, signatures=signatures, at_freeze=at_freeze,
                           expected_code_bundle=expected_code_bundle, scope_receipts=scope_receipts,
                           legacy_exemption=legacy_exemption, reference_receipt=reference_receipt)
    if canary["overall"] != "PASS":
        raise CanaryFailed(canary)
    missing = uncovered(canary, bank, scope)
    if missing:
        raise ValueError("the canary did not cover this batch: " + ", ".join(missing))
    frozen = None if bank.role in GUARD_ROLES else scope_receipts[bank.role]
    scoring = scoped_scorer.score_scope(bank, scope, responses, run_meta, scope_receipt=frozen,
                                        expected_code_bundle=canary["code_bundle_sha256"])
    return {"kind": "canary_bound_scoring", "canary": canary, "canary_receipt_sha256": digest(canary),
            "scoring": scoring, "scoring_receipt_sha256": digest(scoring),
            "code_bundle_sha256": scoring["code_bundle_sha256"]}


def _outside(path: Path, data: Path, what: str) -> Path:
    if path.resolve().is_relative_to(data):
        raise SystemExit("a " + what + " is never read from the data root")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--role", choices=ROLES, required=True)
    ap.add_argument("--signatures", nargs="+", default=list(SIGNATURES),
                    help="composition signatures to scope (default R S RS; pass P explicitly)")
    ap.add_argument("--no-legacy", action="store_true", help="skip the paired guard bank (not for guard roles)")
    ap.add_argument("--exempt-legacy", action="store_true",
                    help="record an explicit exemption from check (e); only with --no-legacy")
    ap.add_argument("--at-freeze", action="store_true",
                    help="required for test and guard_test; requires --expected-code-bundle")
    ap.add_argument("--scope-receipt", type=Path, default=None,
                    help="frozen section 1 scope receipt of --role (required for calibration, dev, test; C8)")
    ap.add_argument("--scope-receipt-sha256", default=None, help="pin the sha256 of the scope receipt file")
    ap.add_argument("--reference-receipt", type=Path, default=None,
                    help="an earlier PASS canary receipt of the same banks; check (c) must reproduce its counts")
    ap.add_argument("--expected-code-bundle", default=None,
                    help="scorer code bundle every pass must report (the S1/S2 bundle at TEST)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.role in FREEZE_ROLES and not args.at_freeze:
        raise SystemExit("role " + args.role + " requires --at-freeze")
    if args.at_freeze and args.expected_code_bundle is None:
        raise SystemExit("--at-freeze requires --expected-code-bundle")
    composition = args.role not in GUARD_ROLES
    if composition and args.scope_receipt is None:
        raise SystemExit("role " + args.role + " requires --scope-receipt (C8)")
    if args.scope_receipt_sha256 is not None and args.scope_receipt is None:
        raise SystemExit("--scope-receipt-sha256 pins a --scope-receipt")
    if args.exempt_legacy and (not args.no_legacy or not composition):
        raise SystemExit("--exempt-legacy is only for a composition role run with --no-legacy")
    if args.out.exists():
        raise SystemExit("canary receipt exists (create-only)")
    if not args.data.is_dir():
        raise SystemExit("data root not found")
    data = args.data.resolve(strict=True)
    if args.out.resolve().is_relative_to(data):
        raise SystemExit("protected output path")
    scope_receipt = reference = None
    try:
        if args.scope_receipt is not None:
            scope_receipt = scoped_scorer.load_scope_receipt(_outside(args.scope_receipt, data, "scope receipt"),
                                                             expected_file_sha256=args.scope_receipt_sha256)
        if args.reference_receipt is not None:
            reference = load_reference_receipt(_outside(args.reference_receipt, data, "reference receipt"))
        receipt = run_canary(data, args.role, signatures=args.signatures, include_legacy=not args.no_legacy,
                             at_freeze=args.at_freeze, expected_code_bundle=args.expected_code_bundle,
                             scope_receipt=scope_receipt, legacy_exemption=args.exempt_legacy,
                             reference_receipt=reference)
    except ValueError as exc:
        raise SystemExit(str(exc))
    written = write_receipt(args.out, receipt)
    print(json.dumps({"role": args.role, "overall": receipt["overall"], "receipt_sha256": written,
                      "summary": {key: value["status"] for key, value in receipt["summary"].items()}}))
    return 0 if receipt["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
