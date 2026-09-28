"""Hash-bound scoped scorer (COMP-RS-001 Amendment 1 section 8.0; clarification C8).

The shipped ``behaviour`` path refuses role calibration and partial response sets. This scorer:
- admits roles {calibration, dev, guard_dev, test, guard_test}; challenge banks are always refused,
  and test/guard_test only with ``at_freeze=True``, checked before any file is touched;
- scores a declared row subset whose sorted-ID digest (preflight.digest) is bound in the receipt;
- takes the strict endpoint from the unchanged ``maxrl_composable_skills_v2.corpus.reward``;
- adds bounded content (content_scoring.score_row), the strict-failure subtype and, on core RS
  rows, the RS response class (failure_taxonomy). These are diagnostic and never change strict.

Scope receipts (C8). Every CALIBRATION, DEV or TEST scope is bound to its frozen section 1 scope
receipt, or refused before any response is scored. The receipt must carry the bank sha key
(canonical_<role>_sha256) and match the bank; EVERY signature of the scope is checked against its
entry, recomputed on the bank's whole core population of that signature, so a multi-signature scope
or a declared subset binds in one call. The caller may pin the receipt's digest. Guard banks
(guard_dev, guard_test) are bound by their manifest-checked bank sha: guard scope receipts are future
work, and that is recorded in the receipt.

Boundary rows (section 1, C8). tag 'boundary' and tag None are refused on a composition bank unless
the scope is declared ``diagnostic=True``. The flag is recorded in the scope and the receipt, and no
endpoint may use a diagnostic call: enrollment refuses a diagnostic receipt unless asked explicitly.
An endpoint (non-diagnostic) scope must also hold one row per parent per signature (section 9).

A scope is tamper-evident: score_scope re-declares it from its recorded filter and refuses any
difference. The receipt binds the bank, manifest, scope, scope receipt, responses and run hashes, and
the code of the whole scoring path: CODE_FILES is every repo file loaded when this module is imported
or score_scope runs, hashed at import and re-checked when scoring. Every source-locked file among them
must equal source_lock.json ("the unchanged corpus.reward"). The scorer never generates, loads a model
or reads model output files.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.maxrl_composable_skills_v2.behaviour import validate_meta
from experiments.maxrl_composable_skills_v2.contracts import strict_json
from experiments.maxrl_composable_skills_v2.corpus import reward
from experiments.maxrl_comp002_pretraining_v1.content_scoring import row_family, score_row
from experiments.maxrl_comp002_pretraining_v1.failure_taxonomy import classify_rs, subtype_strict_failure
from experiments.maxrl_comp002_pretraining_v1.preflight import MANIFEST, digest, read_json, sha

ROLES = ("calibration", "dev", "guard_dev", "test", "guard_test")
FREEZE_ROLES = ("test", "guard_test")
RECEIPT_ROLES = ("calibration", "dev", "test")  # C8: a frozen scope receipt is mandatory
GUARD_ROLES = ("guard_dev", "guard_test")       # C8: bound by the manifest-checked bank sha
TAGS = ("core", "boundary", "preservation")
NON_ENDPOINT_TAGS = (None, "boundary")          # refused on composition banks unless diagnostic
RESPONSE_FIELDS = frozenset({"id", "prompt_hash", "response", "generated_tokens", "stop_reason"})
RECEIPT_ENTRY_FIELDS = {"rows": int, "parents": int, "ids_sha256": str, "parents_sha256": str}
STOP_REASONS = ("eos", "length")
MAX_NEW_TOKENS = 128  # section 8.1: every scored read decodes with max_new_tokens=128
MAX_LINE = 131072
MAX_SCOPE_RECEIPT = 1 << 20
SCHEMA_VERSION = "1.0.0"
REPO = Path(__file__).resolve().parents[2]
# The whole strict/content scoring path: every repo file loaded when this module is imported (module-
# level imports, package __init__ files included) or executed by score_scope (the lazy imports in
# corpus.py and verifier.py included). tests/test_s0pre_scoped_scorer.py derives both sets and checks them.
CODE_FILES = ("experiments/maxrl_comp002_pretraining_v1/scoped_scorer.py",
              "experiments/maxrl_comp002_pretraining_v1/content_scoring.py",
              "experiments/maxrl_comp002_pretraining_v1/failure_taxonomy.py",
              "experiments/maxrl_comp002_pretraining_v1/preflight.py",
              "experiments/maxrl_composable_skills_v2/__init__.py",
              "experiments/maxrl_composable_skills_v2/behaviour.py",
              "experiments/maxrl_composable_skills_v2/contracts.py",
              "experiments/maxrl_composable_skills_v2/corpus.py",
              "experiments/maxrl_composable_skills_v2/generator.py",
              "experiments/maxrl_composable_skills_v2/oracle.py",
              "experiments/maxrl_composable_skills_v2/verifier.py",
              "experiments/maxrl_verifiable_data_v1/__init__.py",
              "experiments/maxrl_verifiable_data_v1/specs.py",
              "experiments/maxrl_verifiable_data_v1/verifier.py")
SOURCE_LOCK_FILE = "experiments/maxrl_comp002_pretraining_v1/source_lock.json"
RESPONSES_DIGEST_RULE = "preflight.digest over the response objects sorted by id"
GUARD_BINDING_NOTE = ("C8: guard scope receipts do not exist yet (future work); a guard scope is bound by "
                      "its manifest-checked bank sha")


class Bank(list):
    """Canonical rows of one role, carrying the digests they were verified against."""

    def __init__(self, rows, *, role: str, bank_sha256: str, manifest_sha256: str = MANIFEST):
        super().__init__(rows)
        self.role = role
        self.bank_sha256 = bank_sha256
        self.manifest_sha256 = manifest_sha256


def code_sha256(root: Path | None = None) -> dict:
    """sha256 of each scoring file, keyed by its repo-relative path."""
    base = REPO if root is None else Path(root)
    return {name: sha(base / name) for name in CODE_FILES}


def source_lock(root: Path | None = None) -> tuple[str, dict]:
    """(sha256 of source_lock.json, {repo-relative path: locked sha256}) for the source-locked packages."""
    path = (REPO if root is None else Path(root)) / SOURCE_LOCK_FILE
    lock = read_json(path)
    locked = {"experiments/" + package + "/" + name: value
              for package, inventory in lock["source_inventories"].items() for name, value in inventory.items()}
    return sha(path), locked


_IMPORT_CODE = code_sha256()
SOURCE_LOCK_SHA256, SOURCE_LOCKED = source_lock()


def admit_role(role: str, at_freeze: bool = False) -> str:
    """Refuse challenge banks always and test/guard_test unless at freeze; touches no file."""
    if type(role) is not str or role not in ROLES:
        raise ValueError("role not admitted: " + repr(role))
    if role in FREEZE_ROLES and at_freeze is not True:
        raise ValueError("role " + role + " requires at_freeze=True")
    return role


def load_role(data, role: str, *, at_freeze: bool = False) -> Bank:
    """Verified canonical rows of one role (manifest sha, per-file sha, row count, unique ids)."""
    admit_role(role, at_freeze)
    root = Path(data)
    manifest_raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST:
        raise ValueError("dataset manifest drift")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    relative = "canonical/" + role + ".jsonl"
    raw = (root / relative).read_bytes()
    bank_sha = hashlib.sha256(raw).hexdigest()
    if relative not in manifest["files"] or bank_sha != manifest["files"][relative]["sha256"]:
        raise ValueError("data file drift: " + relative)
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    rows = [strict_json(line.decode("utf-8"), MAX_LINE) for line in lines]
    if len(rows) != manifest["counts"][role]:
        raise ValueError("row count mismatch: " + role)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate id in bank: " + role)
    if any(row["role"] != role for row in rows):
        raise ValueError("row role mismatch: " + role)
    return Bank(rows, role=role, bank_sha256=bank_sha, manifest_sha256=MANIFEST)


def load_scope_receipt(path, *, expected_file_sha256: str | None = None) -> dict:
    """Read a frozen scope receipt file (duplicate keys and non-finite numbers refused).

    expected_file_sha256, if given, pins the sha256 of the file's bytes; the bytes checked are the
    bytes parsed. score_scope pins the parsed object separately (expected_scope_receipt_sha256).
    """
    raw = Path(path).read_bytes()
    if expected_file_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError("scope receipt file sha differs from the pinned one")
    value = strict_json(raw.decode("utf-8"), MAX_SCOPE_RECEIPT)
    if type(value) is not dict:
        raise ValueError("scope receipt schema")
    return value


def signature(row: dict) -> str:
    return "".join(row["skills"])


def is_core_rs(row: dict) -> bool:
    return row["kind"] == "composition" and row["tag"] == "core" and signature(row) == "RS"


def _names(value, label: str) -> list[str] | None:
    if value is None:
        return None
    if type(value) not in (list, tuple, set, frozenset) or not value or any(type(v) is not str for v in value):
        raise ValueError(label + " must be a non-empty collection of strings")
    if len(set(value)) != len(value):
        raise ValueError("duplicate " + label)
    return sorted(value)


def declare_scope(rows, *, signatures=None, families=None, ids=None, tag: str | None = "core",
                  diagnostic: bool = False) -> dict:
    """Declare a row subset; ids are sorted and bound by ids_sha256 (preflight.digest).

    Section 1 endpoints use tag 'core'; boundary rows are never read. On a bank holding composition
    rows, tag 'boundary' and tag None (every tag) are refused unless diagnostic=True, which is recorded
    in the scope's filter and so in every receipt. A non-diagnostic scope must hold one row per parent
    per signature (the section 9 exact tests rely on it). families filters legacy rows by verifier
    family; a composition row never matches it.
    """
    if type(rows) is not Bank and type(rows) is not list:
        raise ValueError("rows must be a list")
    if not rows or any(type(row) is not dict for row in rows):
        raise ValueError("empty or malformed bank")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate id in bank")
    roles = {row["role"] for row in rows}
    if len(roles) != 1:
        raise ValueError("bank mixes roles")
    (role,) = roles
    admit_role(role, at_freeze=True)
    if isinstance(rows, Bank) and rows.role != role:
        raise ValueError("bank role mismatch")
    if type(diagnostic) is not bool:
        raise ValueError("diagnostic must be a bool")
    if tag is not None and tag not in TAGS:
        raise ValueError("unknown tag: " + repr(tag))
    if tag in NON_ENDPOINT_TAGS and not diagnostic and any(row["kind"] == "composition" for row in rows):
        raise ValueError("boundary rows are never read (section 1): tag " + repr(tag)
                         + " on a composition bank needs diagnostic=True")
    signatures = _names(signatures, "signatures")
    families = _names(families, "families")
    declared = _names(ids, "ids")
    wanted = set(declared) if declared is not None else None
    selected = [row for row in rows
                if (tag is None or row["tag"] == tag)
                and (signatures is None or signature(row) in signatures)
                and (families is None or row_family(row) in families)
                and (wanted is None or row["id"] in wanted)]
    if wanted is not None and len(selected) != len(wanted):
        raise ValueError("declared ids outside the filtered bank: " + str(len(wanted) - len(selected)))
    if not selected:
        raise ValueError("empty scope")
    scoped_ids = sorted(row["id"] for row in selected)
    parents = sorted({row["parent_id"] for row in selected})
    per_signature = Counter(signature(row) for row in selected)
    per_parent = Counter((signature(row), row["parent_id"]) for row in selected)
    one_per_parent = max(per_parent.values()) == 1
    if not one_per_parent and not diagnostic:
        raise ValueError("an endpoint scope holds one row per parent per signature (section 1)")
    return {"role": role, "bank_sha256": getattr(rows, "bank_sha256", None),
            "filter": {"tag": tag, "signatures": signatures, "families": families,
                       "ids_declared": declared is not None, "diagnostic": diagnostic},
            "ids": scoped_ids, "ids_sha256": digest(scoped_ids), "n": len(scoped_ids),
            "parents_sha256": digest(parents), "parents": len(parents),
            "rows_by_signature": dict(sorted(per_signature.items())),
            "one_row_per_parent_per_signature": one_per_parent}


def check_scope(rows: Bank, scope: dict) -> None:
    """Refuse a scope that is not exactly what its recorded filter declares on this bank."""
    if not isinstance(rows, Bank):
        raise ValueError("rows must be a Bank from load_role")
    if type(scope) is not dict or type(scope.get("filter")) is not dict or type(scope.get("ids")) is not list:
        raise ValueError("scope schema")
    if scope.get("bank_sha256") is None or scope["bank_sha256"] != rows.bank_sha256:
        raise ValueError("scope was not declared on this bank")
    flt = scope["filter"]
    if (set(flt) != {"tag", "signatures", "families", "ids_declared", "diagnostic"}
            or type(flt["ids_declared"]) is not bool or type(flt["diagnostic"]) is not bool):
        raise ValueError("scope filter schema")
    again = declare_scope(rows, tag=flt["tag"], signatures=flt["signatures"], families=flt["families"],
                          ids=scope["ids"] if flt["ids_declared"] else None, diagnostic=flt["diagnostic"])
    if again != scope:
        raise ValueError("scope does not match its declaration")


def _receipt_header(scope_receipt, role: str, bank_sha256, manifest_sha256,
                    expected_sha256: str | None) -> str:
    """Check what every scope receipt must carry (C8); return its bank sha key."""
    if type(scope_receipt) is not dict or type(scope_receipt.get("signatures")) is not dict:
        raise ValueError("scope receipt schema")
    if expected_sha256 is not None:
        if type(expected_sha256) is not str:
            raise ValueError("pinned scope receipt digest must be a string")
        if digest(scope_receipt) != expected_sha256:
            raise ValueError("scope receipt digest differs from the pinned one")
    if scope_receipt.get("dataset_manifest_sha256") != MANIFEST or manifest_sha256 != MANIFEST:
        raise ValueError("scope receipt manifest mismatch")
    if "role" in scope_receipt and scope_receipt["role"] != role:
        raise ValueError("scope receipt role mismatch")
    key = "canonical_" + role + "_sha256"
    if key not in scope_receipt:
        raise ValueError("scope receipt lacks the bank sha key " + key)
    if scope_receipt[key] != bank_sha256:
        raise ValueError("scope receipt bank mismatch")
    return key


def _receipt_entry(scope_receipt: dict, name: str) -> dict:
    entry = scope_receipt["signatures"].get(name)
    if type(entry) is not dict:
        raise ValueError("scope receipt lacks signature " + name)
    if any(key not in entry or type(entry[key]) is not kind for key, kind in RECEIPT_ENTRY_FIELDS.items()):
        raise ValueError("scope receipt entry schema: " + name)
    return entry


def _entry_key(entry: dict) -> tuple:
    return entry["ids_sha256"], entry["parents_sha256"], entry["rows"], entry["parents"]


def assert_scope_receipt(scope: dict, scope_receipt: dict, signature_name: str, *,
                         expected_sha256: str | None = None) -> None:
    """Bind one whole single-signature core scope to a frozen section 1 scope receipt (C8).

    The receipt must carry the role's bank sha key and match the scope's bank. The scope must be the
    signature's whole core population; bind_scope_receipt, which has the bank, also binds
    multi-signature scopes and declared subsets.
    """
    if scope["filter"]["tag"] != "core" or list(scope["rows_by_signature"]) != [signature_name]:
        raise ValueError("scope is not the core " + signature_name + " scope")
    _receipt_header(scope_receipt, scope["role"], scope["bank_sha256"], MANIFEST, expected_sha256)
    entry = _receipt_entry(scope_receipt, signature_name)
    if _entry_key(entry) != (scope["ids_sha256"], scope["parents_sha256"], scope["n"], scope["parents"]):
        raise ValueError("scope differs from the frozen scope receipt: " + signature_name)


def bind_scope_receipt(rows: Bank, scope: dict, scope_receipt: dict | None, *,
                       expected_sha256: str | None = None) -> dict:
    """C8 binding of a checked scope; returns the binding record or raises before anything is scored.

    CALIBRATION, DEV and TEST: the receipt is mandatory. Each signature of the scope must have an entry
    that equals the bank's whole core population of that signature (ids_sha256, parents_sha256, rows,
    parents); the scope's rows of that signature are then that population ('full') or a declared part
    of it ('subset'). Only a diagnostic scope may hold rows outside the receipt ('outside_receipt':
    boundary rows, or a signature the receipt does not name). Guard banks: bound by their
    manifest-checked bank sha; passing a scope receipt is refused.
    """
    role = rows.role
    wanted = set(scope["ids"])
    scoped = [row for row in rows if row["id"] in wanted]
    if role in GUARD_ROLES:
        if scope_receipt is not None or expected_sha256 is not None:
            raise ValueError("guard scope receipts do not exist yet (C8): a guard bank is bound by its "
                             "manifest-checked bank sha")
        if any(row["kind"] != "legacy" for row in scoped):
            raise ValueError("a guard bank holds only legacy rows: " + role)
        if rows.manifest_sha256 != MANIFEST:
            raise ValueError("guard bank manifest mismatch")
        return {"kind": "guard_bank_sha256", "bank_sha256": rows.bank_sha256,
                "manifest_sha256": rows.manifest_sha256, "scope_receipt_sha256": None, "pinned": False,
                "note": GUARD_BINDING_NOTE}
    if role not in RECEIPT_ROLES:
        raise ValueError("role not admitted: " + repr(role))
    if any(row["kind"] != "composition" for row in scoped):
        raise ValueError("a " + role + " scope holds composition rows only")
    if scope_receipt is None:
        raise ValueError("a " + role + " scope needs its frozen section 1 scope receipt (C8)")
    key = _receipt_header(scope_receipt, role, rows.bank_sha256, rows.manifest_sha256, expected_sha256)
    diagnostic = scope["filter"]["diagnostic"]
    population: dict[str, list] = {}
    for row in rows:
        if row["kind"] == "composition" and row["tag"] == "core":
            population.setdefault(signature(row), []).append(row)
    bound = {}
    for name in scope["rows_by_signature"]:
        mine = [row for row in scoped if signature(row) == name]
        if name not in scope_receipt["signatures"]:
            if not diagnostic:
                raise ValueError("scope receipt lacks signature " + name)
            bound[name] = {"binding": "outside_receipt", "rows": len(mine), "rows_in_receipt": 0,
                           "receipt_rows": None}
            continue
        entry = _receipt_entry(scope_receipt, name)
        frozen = population.get(name, [])
        frozen_ids = sorted(row["id"] for row in frozen)
        if _entry_key(entry) != (digest(frozen_ids), digest(sorted({row["parent_id"] for row in frozen})),
                                 len(frozen), len({row["parent_id"] for row in frozen})):
            raise ValueError("scope differs from the frozen scope receipt: " + name)
        inside = set(frozen_ids)
        in_receipt = sum(row["id"] in inside for row in mine)
        if in_receipt != len(mine) and not diagnostic:
            raise ValueError("scope rows outside the frozen scope receipt: " + name)
        if in_receipt != len(mine):
            binding = "outside_receipt"
        else:
            binding = "full" if in_receipt == len(frozen) else "subset"
        bound[name] = {"binding": binding, "rows": len(mine), "rows_in_receipt": in_receipt,
                       "receipt_rows": len(frozen)}
    return {"kind": "scope_receipt", "bank_key": key, "bank_sha256": rows.bank_sha256,
            "scope_receipt_sha256": digest(scope_receipt), "pinned": expected_sha256 is not None,
            "stage": scope_receipt.get("stage"), "signatures": bound}


def _index_responses(responses, scope: dict) -> dict:
    if type(responses) is not list:
        raise ValueError("responses must be a list")
    indexed = {}
    for item in responses:
        if type(item) is not dict or set(item) != RESPONSE_FIELDS:
            raise ValueError("response schema")
        if type(item["id"]) is not str or item["id"] in indexed:
            raise ValueError("duplicate or invalid response id")
        if type(item["prompt_hash"]) is not str or type(item["response"]) is not str:
            raise ValueError("response/prompt_hash type")
        tokens = item["generated_tokens"]
        if type(tokens) is not int or not 0 < tokens <= MAX_NEW_TOKENS:
            raise ValueError("generated_tokens outside 1.." + str(MAX_NEW_TOKENS))
        if item["stop_reason"] not in STOP_REASONS:
            raise ValueError("unbound stop_reason")
        if item["stop_reason"] == "length" and tokens != MAX_NEW_TOKENS:
            raise ValueError("length stop below the token cap")
        indexed[item["id"]] = item
    expected = set(scope["ids"])
    if set(indexed) != expected:
        raise ValueError("response id set differs from the scope: missing %d, extra %d"
                         % (len(expected - set(indexed)), len(set(indexed) - expected)))
    return indexed


def _check_code() -> dict:
    """The scoring path's code now: unchanged since import, and every source-locked file as locked."""
    code = code_sha256()
    if code != _IMPORT_CODE:
        raise ValueError("scorer source changed since import")
    drift = sorted(name for name, value in code.items() if name in SOURCE_LOCKED and value != SOURCE_LOCKED[name])
    if drift:
        raise ValueError("source-locked scoring file differs from source_lock.json: " + ", ".join(drift))
    return code


def score_one(row: dict, item: dict) -> dict:
    """One per-row record: strict from corpus.reward, plus diagnostic content/subtype/RS class."""
    if item["prompt_hash"] != row["prompt_hash"]:
        raise ValueError("wrong rendered prompt: " + row["id"])
    response = item["response"]
    strict = int(reward(row, response))
    content = score_row(row, response)
    if content["strict_pass"] != bool(strict):
        raise AssertionError("content_scoring strict disagrees with corpus.reward: " + row["id"])
    return {"id": row["id"], "parent_id": row["parent_id"], "signature": signature(row), "tag": row["tag"],
            "strict": strict, "content_pass": content["content_pass"],
            "normalization": content["normalization"], "format_only_failure": content["format_only_failure"],
            "subtype": subtype_strict_failure(row, response, item["stop_reason"]),
            "rs_class": classify_rs(row["verifier"]["spec"], response) if is_core_rs(row) else None,
            "stop_reason": item["stop_reason"], "generated_tokens": item["generated_tokens"]}


def _tally(records: list[dict]) -> dict:
    return {"rows": len(records),
            "strict_correct": sum(r["strict"] for r in records),
            "content_correct": sum(r["content_pass"] for r in records),
            "format_only_failures": sum(r["format_only_failure"] for r in records),
            "strict_failures_without_subtype": sum(1 for r in records if not r["strict"] and r["subtype"] is None),
            "normalization": dict(sorted(Counter(r["normalization"] for r in records).items())),
            "subtype": dict(sorted(Counter(r["subtype"] for r in records if r["subtype"]).items())),
            "rs_class": dict(sorted(Counter(r["rs_class"] for r in records if r["rs_class"]).items())),
            "stop_reason": dict(sorted(Counter(r["stop_reason"] for r in records).items())),
            "generated_tokens": sum(r["generated_tokens"] for r in records)}


def score_scope(rows: Bank, scope: dict, responses: list, run_meta: dict, *,
                scope_receipt: dict | None = None, expected_code_bundle: str | None = None,
                expected_scope_receipt_sha256: str | None = None) -> dict:
    """Score exactly the declared scope and return the hash-bound receipt.

    scope_receipt is mandatory for CALIBRATION, DEV and TEST (C8, bind_scope_receipt) and refused for
    guard banks; expected_scope_receipt_sha256 pins its preflight.digest. Every binding is checked
    before any response is scored.
    """
    validate_meta(run_meta)
    check_scope(rows, scope)
    binding = bind_scope_receipt(rows, scope, scope_receipt, expected_sha256=expected_scope_receipt_sha256)
    indexed = _index_responses(responses, scope)
    code = _check_code()
    bundle = digest(code)
    if expected_code_bundle is not None and expected_code_bundle != bundle:
        raise ValueError("scorer code bundle differs from the expected one")
    by_id = {row["id"]: row for row in rows}
    records = [score_one(by_id[i], indexed[i]) for i in scope["ids"]]
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(record["signature"], []).append(record)
    return {"schema_version": SCHEMA_VERSION, "kind": "scoped_scoring_receipt", "experiment": "COMP-RS-001",
            "amendment": "COMP_RS_001_AMENDMENT_1.md section 8.0; clarification C8",
            "role": rows.role, "bank_sha256": rows.bank_sha256, "manifest_sha256": rows.manifest_sha256,
            "scope": {k: v for k, v in scope.items() if k != "ids"},
            "diagnostic": scope["filter"]["diagnostic"],
            "scope_binding": binding, "scope_receipt_sha256": binding["scope_receipt_sha256"],
            "responses_sha256": digest(sorted(responses, key=lambda item: item["id"])),
            "responses_digest_rule": RESPONSES_DIGEST_RULE, "run": dict(run_meta),
            "decoding_checks": {"max_new_tokens": MAX_NEW_TOKENS, "stop_reasons": list(STOP_REASONS)},
            "code_sha256": code, "code_bundle_sha256": bundle,
            "source_lock": {"path": SOURCE_LOCK_FILE, "sha256": SOURCE_LOCK_SHA256,
                            "files_checked": sorted(name for name in code if name in SOURCE_LOCKED)},
            "strict_endpoint": "experiments.maxrl_composable_skills_v2.corpus.reward (unchanged)",
            "diagnostics": "content_scoring.score_row; failure_taxonomy.subtype_strict_failure; "
                           "failure_taxonomy.classify_rs on core RS rows only",
            "counts": {"overall": _tally(records),
                       "by_signature": {sig: _tally(group) for sig, group in sorted(groups.items())}},
            "strict_correct_ids": [r["id"] for r in records if r["strict"]],
            "rows": records}


def enrollment_from_receipt(receipt: dict, *, diagnostic: bool = False) -> dict:
    """The scoped analogue of behaviour.enroll, derived from one scoring receipt.

    A diagnostic receipt is never an endpoint (C8): it is refused unless diagnostic=True, and the
    enrollment records the flag.
    """
    if type(receipt) is not dict or receipt.get("kind") != "scoped_scoring_receipt":
        raise ValueError("not a scoped scoring receipt")
    flagged = receipt.get("diagnostic", False)
    if flagged is not False and diagnostic is not True:
        raise ValueError("a diagnostic scoring receipt is not an endpoint: enrolling it needs diagnostic=True")
    scores = {r["id"]: r["strict"] for r in receipt["rows"]}
    correct = sorted(i for i, value in scores.items() if value)
    return {"schema_version": SCHEMA_VERSION, "kind": "scoped_source_correct_enrollment",
            "role": receipt["role"], "bank_sha256": receipt["bank_sha256"],
            "scope_ids_sha256": receipt["scope"]["ids_sha256"],
            "scope_parents_sha256": receipt["scope"]["parents_sha256"],
            "diagnostic": flagged is not False, "scope_receipt_sha256": receipt.get("scope_receipt_sha256"),
            "run": receipt["run"], "responses_sha256": receipt["responses_sha256"],
            "code_sha256": receipt["code_sha256"], "code_bundle_sha256": receipt["code_bundle_sha256"],
            "scoring_receipt_sha256": digest(receipt),
            "row_scores": scores, "source_correct_ids": correct, "source_correct_count": len(correct),
            "scoped_rows": len(scores)}


def enroll_scope(rows: Bank, scope: dict, responses: list, run_meta: dict, *, diagnostic: bool = False,
                 **options) -> dict:
    """Score the scope, then return its correct-id list and counts."""
    return enrollment_from_receipt(score_scope(rows, scope, responses, run_meta, **options),
                                   diagnostic=diagnostic)


def write_receipt(path, receipt: dict) -> str:
    """Write a receipt create-only with LF line endings; return the sha256 of the bytes written."""
    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
