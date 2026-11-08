"""DMRF-PCD-001: outcome-preserving, fail-closed per-pair disjointness audit.

No model calls, scoring, network access, training, or historical-file updates.
Run this file with --manifest PATH --out NEW_PATH. See PAIRWISE_DISJOINTNESS.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

SCHEMA = "DMRF-PCD-001"
BASE_COMMIT = "e9f7e9f9fbb8a6577c96ccc357ffe7048318a122"
METHODS = {"rrfr2_row_v1", "rrfr2_stem_v1", "srb_row_v1", "text_v1", "ngram20_v1"}
TEXT_METHODS = {"text_v1", "ngram20_v1"}
RRFR = re.compile(r"(RRFR2-[0-9a-f]{24})(-step[0-9]+-valid)?")
SRB = re.compile(r"(rineya_srb_v1_(?:safety_boundary|clarification|knowledge_boundary|uncertainty))_[0-9]+")
MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 20000
MAX_KEYS = 2000000


class Refusal(ValueError):
    def __init__(self, status: str, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.status, self.reason, self.detail = status, reason, detail or {}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _need(condition: bool, reason: str) -> None:
    # Deliberately not assert: these checks must survive python -O.
    if not condition:
        raise Refusal("INVALID", reason)


def _object(value: Any, keys: set[str], label: str) -> None:
    _need(isinstance(value, dict) and set(value) == keys, f"invalid {label} fields")


def _string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _no_duplicate_keys(pairs: list) -> dict:
    out = {}
    for key, value in pairs:
        _need(key not in out, "duplicate JSON object key")
        out[key] = value
    return out


def _json(text: str) -> Any:
    def invalid_constant(_: str) -> None:
        raise Refusal("INVALID", "non-finite JSON number")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=invalid_constant)
    except json.JSONDecodeError as exc:
        raise Refusal("INVALID", "malformed JSON") from exc


def validate_plan(plan: dict) -> None:
    _object(plan, {"schema", "source_commit", "sources", "checks"}, "manifest")
    _need(plan["schema"] == SCHEMA, "unsupported schema")
    _need(plan["source_commit"] == BASE_COMMIT, "wrong historical source commit")
    _need(isinstance(plan["sources"], dict) and bool(plan["sources"]), "empty sources")
    for name, source in plan["sources"].items():
        _need(_string(name), "invalid source name")
        _object(source, {"path", "sha256", "rows", "row_id_field"}, "source")
        # Null paths/digests in the template remain explicit UNVERIFIED source bindings.
        _need(source["path"] is None or _string(source["path"]), "invalid source path")
        _need(source["sha256"] is None or (isinstance(source["sha256"], str) and
              re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None), "invalid source digest")
        _need(type(source["rows"]) is int and 0 < source["rows"] <= MAX_ROWS, "invalid row count")
        _need(_string(source["row_id_field"]), "invalid row id field")
    _need(isinstance(plan["checks"], list) and bool(plan["checks"]), "empty checks")
    ids, used, pairs, signatures = set(), set(), {}, set()
    for check in plan["checks"]:
        _object(check, {"id", "pair", "method", "left", "right"}, "check")
        _need(_string(check["id"]) and check["id"] not in ids, "invalid or duplicate check id")
        ids.add(check["id"])
        _need(_string(check["pair"]) and isinstance(check["method"], str)
              and check["method"] in METHODS, "invalid pair or method")
        for side in ("left", "right"):
            spec = check[side]
            _object(spec, {"source", "field", "prefix"}, "side")
            _need(isinstance(spec["source"], str) and spec["source"] in plan["sources"], "unknown source")
            _need(_string(spec["field"]) and isinstance(spec["prefix"], str), "invalid selector")
            _need(check["method"] not in TEXT_METHODS or spec["prefix"] == "", "text cannot strip id prefixes")
            used.add(spec["source"])
        identity = (check["left"]["source"], check["right"]["source"])
        _need(identity[0] != identity[1], "self-comparison cannot be a disjointness claim")
        _need(check["pair"] not in pairs or pairs[check["pair"]] == identity, "pair id reused for different sources")
        pairs[check["pair"]] = identity
        signature = digest({k: check[k] for k in ("method", "left", "right")})
        _need(signature not in signatures, "duplicate comparison")
        signatures.add(signature)
    _need(used == set(plan["sources"]), "unused sources would imply unmeasured coverage")


def _read_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_BYTES + 1)
    except OSError as exc:
        # A failed local read is not evidence that a remote/original artifact does not exist.
        raise Refusal("UNOBSERVABLE", "cannot read the explicitly bound local artifact") from exc
    _need(len(data) <= MAX_BYTES, "source exceeds bounded reader limit")
    return data


def _value(row: dict, field: str) -> str:
    if field == "messages.user":
        messages = row.get("messages")
        _need(isinstance(messages, list) and bool(messages), "missing messages")
        _need(all(isinstance(m, dict) and isinstance(m.get("role"), str)
                  and isinstance(m.get("content"), str) for m in messages), "invalid message")
        users = [m for m in messages if m["role"] == "user"]
        _need(len(users) == 1, "messages.user requires exactly one user turn; no concatenation")
        value = users[0]["content"]
    else:
        value = row.get(field)
    _need(_string(value), f"missing or empty selected field: {field}")
    return value


def load_source(spec: dict, root: Path) -> tuple[list[dict], dict]:
    if spec["path"] is None or spec["sha256"] is None:
        raise Refusal("UNVERIFIED", "source path and immutable SHA-256 must both be bound")
    path = (root / spec["path"]).resolve()
    data = _read_bytes(path)
    observed = sha(data)
    _need(observed == spec["sha256"], "source SHA-256 mismatch")
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Refusal("INVALID", "source is not UTF-8") from exc
    _need(len(lines) == spec["rows"], "source row count mismatch (no dropped/blank rows)")
    rows, ids = [], set()
    for line in lines:
        row = _json(line)
        _need(isinstance(row, dict), "JSONL row is not an object")
        row_id = _value(row, spec["row_id_field"])
        _need(row_id not in ids, "duplicate raw row id")
        ids.add(row_id)
        rows.append(row)
    return rows, {"medium": "local_file_snapshot", "path": str(path), "sha256": observed,
                  "bytes": len(data), "rows": len(rows), "row_id_field": spec["row_id_field"]}


def _payload(row: dict, spec: dict, method: str) -> str:
    value = _value(row, spec["field"])
    if method not in TEXT_METHODS:
        prefix = spec["prefix"]
        if not value.startswith(prefix):
            raise Refusal("UNOBSERVABLE", "declared id prefix does not match actual rows")
        value = value[len(prefix):]
    return value


def _extract(row: dict, spec: dict, method: str) -> tuple[frozenset, str]:
    value = _payload(row, spec, method)
    if method in TEXT_METHODS:
        normal = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        if method == "text_v1":
            return frozenset([normal]) if normal else frozenset(), "normalized_user_text_v1"
        words = normal.split()
        return frozenset(tuple(words[i:i + 20]) for i in range(len(words) - 19)), "word_20gram_v1"
    if method.startswith("rrfr2_"):
        match = RRFR.fullmatch(value)
        if not match or (method == "rrfr2_row_v1" and not match[2]):
            return frozenset(), "unrecognized"
        key = match[1] if method == "rrfr2_stem_v1" else match[0]
        return frozenset([key]), "RRFR2"
    match = SRB.fullmatch(value)
    return (frozenset([value]), match[1]) if match else (frozenset(), "unrecognized")


def _index(rows: list[dict], spec: dict, method: str) -> tuple[list[frozenset], dict]:
    keys, namespaces, count = [], set(), 0
    for row in rows:
        values, namespace = _extract(row, spec, method)
        keys.append(values)
        if values:
            namespaces.add(namespace)
        count += len(values)
        _need(count <= MAX_KEYS, "feature count exceeds bounded audit limit")
    covered = sum(bool(k) for k in keys)
    return keys, {"rows": len(rows), "covered_rows": covered,
                  "uncovered_rows": len(rows) - covered,
                  "unique_keys": len(set().union(*keys)), "namespaces": sorted(namespaces)}


def _compare(left: list[frozenset], right: list[frozenset]) -> dict:
    common = set().union(*left) & set().union(*right)
    return {"keys": common, "left_rows": {i for i, k in enumerate(left) if k & common},
            "right_rows": {i for i, k in enumerate(right) if k & common}}


def _measure(left: list[dict], right: list[dict], ls: dict, rs: dict, method: str) -> dict:
    lk, lc = _index(left, ls, method)
    rk, rc = _index(right, rs, method)
    coverage = {"left": lc, "right": rc}
    if not left or not right or lc["uncovered_rows"] or rc["uncovered_rows"]:
        raise Refusal("UNOBSERVABLE", "full-pair extraction coverage not established", coverage)
    if lc["namespaces"] != rc["namespaces"]:
        raise Refusal("UNOBSERVABLE", "observed namespace sets differ; no whole-pair id claim", coverage)
    return {"left": lk, "right": rk, "coverage": coverage, "match": _compare(lk, rk)}


def _injected(receiver: dict, donor: dict, rs: dict, ds: dict, method: str) -> dict:
    """Transfer raw payload before extraction; preserve receiver schema and id wrapper.

    Do NOT inject already-normalized keys: that would bypass the very extractor at issue.
    """
    row = copy.deepcopy(receiver)
    value = _payload(donor, ds, method)
    if method not in TEXT_METHODS:
        value = rs["prefix"] + value
    if rs["field"] == "messages.user":
        _value(row, rs["field"])
        for message in row["messages"]:
            if message["role"] == "user":
                message["content"] = value
    else:
        row[rs["field"]] = value
    return row


def _controls(left: list[dict], right: list[dict], ls: dict, rs: dict, method: str) -> dict:
    # Always execute afresh for this ordered source pair, selector pair, and method.
    result = {"self": [], "cross_injections": []}
    for rows, spec in ((left, ls), (right, rs)):
        measured = _measure(rows, rows, spec, spec, method)
        actual = measured["match"]
        expected = measured["coverage"]["left"]["unique_keys"]
        ok = len(actual["keys"]) == expected and len(actual["left_rows"]) == len(rows) \
            and len(actual["right_rows"]) == len(rows)
        result["self"].append({"rows": len(rows), "expected_unique_keys": expected,
                               "observed_unique_keys": len(actual["keys"]), "passed": ok})
    for label, donors, receivers, ds, rspec in (
        ("left_to_right", left, right, ls, rs), ("right_to_left", right, left, rs, ls)
    ):
        # One live raw-row canary per observed namespace, in BOTH directions.
        representatives = {}
        receiver_indices = {}
        for i, row in enumerate(donors):
            representatives.setdefault(_extract(row, ds, method)[1], i)
        for i, row in enumerate(receivers):
            receiver_indices.setdefault(_extract(row, rspec, method)[1], i)
        for namespace, di in sorted(representatives.items()):
            ri = receiver_indices[namespace]
            changed = list(receivers)
            changed[ri] = _injected(receivers[ri], donors[di], rspec, ds, method)
            # Keep the original ordered left/right pipeline even for reverse injection.
            if label == "left_to_right":
                m = _measure(donors, changed, ls, rs, method)
                li, rj = di, ri
            else:
                m = _measure(changed, donors, ls, rs, method)
                li, rj = ri, di
            target = m["left"][li] & m["right"][rj]
            actual = m["match"]
            ok = bool(target) and target <= actual["keys"] and li in actual["left_rows"] \
                and rj in actual["right_rows"]
            result["cross_injections"].append({"direction": label, "namespace": namespace,
                "donor_row": di, "receiver_row": ri, "payload_sha256": sha(_payload(donors[di], ds, method).encode()),
                "target_shared_keys": len(target), "passed": ok})
    result["passed"] = all(c["passed"] for c in result["self"] + result["cross_injections"])
    return result


def audit_check(check: dict, sources: dict, code_sha: str) -> dict:
    report = {"check_id": check["id"], "pair": check["pair"], "method": check["method"],
              "selectors": {s: check[s] for s in ("left", "right")},
              "status": "UNVERIFIED", "zero_admissible": False, "overlap": None, "controls": None}
    try:
        loaded = [sources[check[s]["source"]] for s in ("left", "right")]
        for source in loaded:
            if isinstance(source, Refusal):
                raise source
        (left, lp), (right, rp) = loaded
        report["sources"] = {"left": lp, "right": rp}
        report["binding_sha256"] = digest({"check": check, "sources": report["sources"], "code_sha256": code_sha,
            "python": sys.version, "unicode_data_version": unicodedata.unidata_version})
        before = digest([left, right])
        measured = _measure(left, right, check["left"], check["right"], check["method"])
        report["coverage"] = measured["coverage"]
        controls = _controls(left, right, check["left"], check["right"], check["method"])
        report["controls"] = controls
        _need(digest([left, right]) == before, "positive control mutated original in-memory records")
        if not controls["passed"]:
            raise Refusal("UNOBSERVABLE", "live per-pair collision control failed")
        match = measured["match"]
        report["overlap"] = {"unique_keys": len(match["keys"]), "left_rows": len(match["left_rows"]),
                             "right_rows": len(match["right_rows"])}
        report["zero_admissible"] = not match["keys"]
        report["status"] = "SUPPORTED_ZERO" if report["zero_admissible"] else "CONTRADICTED"
    except Refusal as exc:
        report.update(status=exc.status, reason=exc.reason, detail=exc.detail)
    return report


def audit_manifest(plan: dict, root: Path) -> dict:
    validate_plan(plan)
    code_sha = sha(Path(__file__).read_bytes())
    sources = {}
    for name, spec in plan["sources"].items():
        try:
            sources[name] = load_source(spec, root)
        except Refusal as exc:
            sources[name] = exc
    checks = [audit_check(check, sources, code_sha) for check in plan["checks"]]
    statuses = {c["status"] for c in checks}
    status = "SUPPORTED_ZERO" if statuses == {"SUPPORTED_ZERO"} else (
        "CONTRADICTED" if statuses <= {"SUPPORTED_ZERO", "CONTRADICTED"} else "REFUSED")
    return {"schema": SCHEMA, "status": status, "manifest_sha256": digest(plan),
            "source_commit": plan["source_commit"], "implementation_sha256": code_sha,
            "runtime": {"python": sys.version, "unicode_data_version": unicodedata.unidata_version},
            "normalization": "NFKC+casefold+whitespace; digits/punctuation retained; 20-gram=20 whitespace words",
            "checks": checks, "scope": "Only enumerated pairs, selectors, and exact methods; not semantic leakage or generalization proof.",
            "authority": {"model_calls": 0, "training_steps": 0, "rescoring": False,
                          "historical_outcome_updates": False, "new_execution_authorization": False}}


def require_supported(report: dict) -> None:
    """Fail closed at the new command boundary; do not feed legacy outcome writers."""
    if report["status"] != "SUPPORTED_ZERO":
        raise Refusal(report["status"], "one or more registered comparisons did not establish admissible zero")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New receipt only; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        plan_bytes = _read_bytes(args.manifest)
        try:
            plan = _json(plan_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise Refusal("INVALID", "manifest is not UTF-8") from exc
        report = audit_manifest(plan, args.manifest.resolve().parent)
        report["manifest_file_sha256"] = sha(plan_bytes)
        # Exclusive creation also protects source files, the manifest, and historical receipts.
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        require_supported(report)
        return 0
    except Refusal as exc:
        print(json.dumps({"status": exc.status, "reason": exc.reason}), file=sys.stderr)
        return 1 if exc.status == "CONTRADICTED" else 2
    except OSError:
        print(json.dumps({"status": "INVALID", "reason": "cannot create new receipt; nothing overwritten"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
