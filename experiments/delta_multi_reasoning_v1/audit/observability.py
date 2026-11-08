"""OBS-001: read-only, source-aware claim admission. No model or scoring imports.

A consistent review is NOT verification of every claim. DIRECT observations are
collector attestations, not signatures; this module cannot authenticate a remote
service or determine whether arbitrary prose is true. See SOURCE_OBSERVABILITY_AUDIT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = 1
# Only these methods can measure these properties. Callers cannot add capabilities.
METHODS = {
    "local_inventory": ({"local_fs"}, "presence"),
    "remote_git_inventory": ({"remote_git"}, "presence"),
    "restored_byte_comparison": ({"local_fs", "remote_git"}, "byte_identity"),
    "tensor_manifest_comparison": ({"tensor_manifest"}, "tensor_identity"),
    "rendered_prose_scan": ({"rendered_bank"}, "mechanical_defect"),
    "literal_search": ({"local_fs", "remote_git"}, "literal_match"),
    "qualified_rule_sweep": ({"rule_bank"}, "band_swing"),
    "prediction_inventory": ({"prediction_bank"}, "prediction_present"),
    "dataset_overlap": ({"dataset_pair"}, "overlap"),
}
# The expected boolean is a statement about the property, not a generic PASS.
PREDICATES = {
    "present": ("presence", True), "absent": ("presence", False),
    "byte_identical": ("byte_identity", True),
    "byte_different": ("byte_identity", False),
    "tensor_identical": ("tensor_identity", True),
    "tensor_different": ("tensor_identity", False),
    "no_detected_mechanical_defect": ("mechanical_defect", False),
    "no_literal_match": ("literal_match", False),
    "no_detected_rule_swing": ("band_swing", False),
    "rule_has_band_swing": ("band_swing", True),
    "predictions_complete": ("prediction_present", True),
    "declared_datasets_disjoint": ("overlap", False),
}
NEGATIVE_SEARCHES = {"absent", "no_detected_mechanical_defect", "no_literal_match",
                     "no_detected_rule_swing", "declared_datasets_disjoint"}
STATES = {"COMPLETE", "PARTIAL", "UNAVAILABLE", "ERROR", "NOT_RUN", "INAPPLICABLE"}
BASES = {"DIRECT", "REPORTED", "SYNTHETIC"}
SENTENCE = re.compile(r"(?<=[.!?])\s+")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _ids(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(x, str) and x.strip() and not any(c in x for c in "*?[]")
                    for x in value) and len(set(value)) == len(value))


def _source(value: Any) -> bool:
    return (isinstance(value, dict) and set(value) == {"kind", "locator", "revision"}
            and isinstance(value["kind"], str) and bool(value["kind"])
            and isinstance(value["locator"], str) and bool(value["locator"].strip())
            and isinstance(value["revision"], str)
            and bool(REVISION.fullmatch(value["revision"])))


def _hash(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def assess(claim: Mapping, observation: Mapping) -> dict:
    """Admit only an exact scoped predicate, never a broader narrative inference.

    A partial inventory can prove presence, but cannot prove absence. Error/denied
    access is not a negative finding. A finite rule sweep never proves 'no confound',
    a p-value never proves equivalence, and file identity never proves determinism.
    """
    def result(status: str, reason: str, subjects: Sequence[str] = ()) -> dict:
        return {"schema_version": VERSION, "claim_id": claim.get("id"),
                "status": status, "reason": reason, "subjects": list(subjects),
                "admissible_as_stated": status == "SUPPORTED"}

    if not isinstance(claim.get("id"), str) or not claim["id"].strip():
        return result("INVALID", "MISSING_CLAIM_ID")
    if not _ids(claim.get("subjects")):
        return result("INVALID", "EMPTY_DUPLICATE_OR_WILDCARD_SCOPE")
    if not _source(claim.get("source")) or not _source(observation.get("source")):
        return result("UNOBSERVABLE", "SOURCE_IDENTITY_UNBOUND")
    if claim["source"] != observation["source"]:
        return result("UNOBSERVABLE", "SOURCE_SCOPE_OR_REVISION_MISMATCH")
    predicate = claim.get("predicate")
    if not isinstance(predicate, str) or predicate not in PREDICATES:
        return result("UNOBSERVABLE", "UNSUPPORTED_PREDICATE")
    method = observation.get("method")
    if not isinstance(method, str) or method not in METHODS:
        return result("UNOBSERVABLE", "UNQUALIFIED_METHOD")
    kinds, property_name = METHODS[method]
    required, expected = PREDICATES[predicate]
    if observation["source"]["kind"] not in kinds or property_name != required:
        return result("UNOBSERVABLE", "METHOD_CANNOT_MEASURE_PREDICATE")
    if not isinstance(claim.get("measurement"), dict) or not claim["measurement"] or not isinstance(observation.get("measurement"), dict):
        return result("UNOBSERVABLE", "MEASUREMENT_CONTRACT_UNBOUND")
    try:
        measurement_sha = digest(observation["measurement"])
        if digest(claim["measurement"]) != measurement_sha:
            return result("UNOBSERVABLE", "MEASUREMENT_CONTRACT_MISMATCH")
    except (ValueError, TypeError):
        return result("INVALID", "INVALID_MEASUREMENT_CONTRACT")
    if (not isinstance(observation.get("state"), str) or observation["state"] not in STATES
            or not isinstance(observation.get("basis"), str) or observation["basis"] not in BASES):
        return result("INVALID", "INVALID_OBSERVATION_STATE_OR_BASIS")
    if observation["state"] in {"UNAVAILABLE", "ERROR", "NOT_RUN", "INAPPLICABLE"}:
        return result("UNOBSERVABLE", "OBSERVATION_" + observation["state"])
    if observation["basis"] != "DIRECT":
        return result("UNVERIFIED", "EVIDENCE_IS_" + observation["basis"])
    if not _hash(observation.get("evidence_sha256")) or not _hash(observation.get("method_sha256")):
        return result("UNVERIFIED", "EVIDENCE_OR_METHOD_HASH_MISSING")
    values = observation.get("results")
    if not isinstance(values, dict) or any(not isinstance(k, str) or
                                         (v is not None and type(v) is not bool)
                                         for k, v in values.items()):
        return result("INVALID", "INVALID_RESULT_VALUES")
    subjects = claim["subjects"]
    seen = {k: values.get(k) for k in subjects}
    if observation["state"] == "PARTIAL":
        # A returned object is a positive witness even in a paginated listing.
        # No other conclusion is licensed by a partially completed check.
        seen = {k: (v if required == "presence" and v is True else None)
                for k, v in seen.items()}
    counterexamples = [k for k, v in seen.items() if v is not None and v != expected]
    if counterexamples:
        return result("CONTRADICTED", "DIRECT_SCOPED_COUNTEREXAMPLE", counterexamples)
    unknown = [k for k, v in seen.items() if v is None]
    if unknown:
        return result("INCONCLUSIVE", "INCOMPLETE_SUBJECT_COVERAGE", unknown)
    if predicate in NEGATIVE_SEARCHES:
        pc = observation.get("positive_control", {})
        if not (isinstance(pc, dict) and pc.get("passed") is True
                and pc.get("source") == observation["source"]
                and pc.get("method_sha256") == observation["method_sha256"]
                and pc.get("measurement_sha256") == measurement_sha
                and _hash(pc.get("evidence_sha256"))):
            return result("UNOBSERVABLE", "MISSING_OR_UNBOUND_POSITIVE_CONTROL")
    return result("SUPPORTED", "DIRECT_SCOPED_WITNESSES" if observation["state"] == "PARTIAL"
                  else "DIRECT_COMPLETE_SCOPED_OBSERVATION", subjects)


def row_coverage(rows: Sequence[Mapping], fields: Sequence[str] = ("context",)) -> dict:
    """Preflight BEFORE legacy text checks; no empty string fallback is evidence."""
    problems, ids = [], []
    if not rows:
        problems.append("EMPTY_ROWS")
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"{i}:NOT_A_ROW")
            continue
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip():
            problems.append(f"{i}:INVALID_ID")
        elif rid in ids:
            problems.append(f"{rid}:DUPLICATE_ID")
        else:
            ids.append(rid)
        for field in fields:
            if not isinstance(row.get(field), str) or not row[field].strip():
                problems.append(f"{i}:MISSING_OR_EMPTY_{field}")
    return {"state": "COMPLETE" if not problems else "INAPPLICABLE",
            "rows_seen": len(rows), "ids": ids, "problems": problems}


def locator_coverage(rows: Sequence[Mapping], candidates: Sequence[str] | None) -> dict:
    """Restrict candidate-floor claims to one common, explicitly scored population.

    Returns IDs, not a replacement score. Any skipped row forbids an ALL-bank claim.
    A caller may register a narrower claim on scored_ids instead.
    """
    base = row_coverage(rows, ("context", "reference"))
    if base["state"] != "COMPLETE":
        return base
    if not candidates or any(not isinstance(c, str) or not c.strip() for c in candidates):
        return {"state": "INAPPLICABLE", "problems": ["CANDIDATE_UNIVERSE_MISSING"]}
    scored, skipped, floors = [], [], {}
    for row in rows:
        ss = [s.strip() for s in SENTENCE.split(row["context"]) if s.strip()]
        targets = [i for i, s in enumerate(ss) if row["reference"] in s]
        bearing = [i for i, s in enumerate(ss) if any(c in s for c in candidates)]
        if len(ss) < 2 or len(targets) != 1 or row["reference"] not in candidates or not bearing:
            skipped.append(row["id"])
        else:
            scored.append(row["id"])
            floors[row["id"]] = 1.0 / len(bearing)
    return {"state": "COMPLETE" if scored and not skipped else
            "PARTIAL" if scored else "INAPPLICABLE", "scored_ids": scored,
            "skipped_ids": skipped, "candidate_floor_by_scored_id": floors,
            "whole_bank_claim_admissible": bool(scored) and not skipped}


def sweep_coverage(result: Mapping, *, pool_observed: bool,
                   expected_rules: Sequence[str], min_swing: float = 0.02) -> dict:
    """Preserve not-applicable and zero sensitivity; never collapse either to PASS.

    pool_observed must come from a separate row/pool check, not from sweep scores.
    expected_rules is the prospectively declared rule universe, NOT the returned list.
    Values describe only that finite rule set; no causal-exclusion claim is emitted.
    """
    def blocked(reason):
        return {"state": "INAPPLICABLE", "problems": [reason], "eligible_rules": None}
    if type(min_swing) not in (int, float) or not math.isfinite(min_swing) or not 0 < min_swing <= 1:
        return blocked("INVALID_SWING_THRESHOLD")
    if result.get("applicable") is not True:
        return blocked("LEGACY_SWEEP_NOT_APPLICABLE")
    if pool_observed is not True:
        return blocked("CANDIDATE_POOL_NOT_OBSERVED")
    levels, ns = result.get("levels"), result.get("n_per_level")
    if (not isinstance(levels, list) or len(levels) != 2 or not _ids(levels)
            or not isinstance(ns, dict) or set(ns) != set(levels)
            or any(type(n) is not int or n <= 0 for n in ns.values())):
        return blocked("EMPTY_OR_INVALID_LEVELS")
    if not _ids(list(expected_rules)):
        return blocked("RULE_UNIVERSE_UNBOUND")
    rules = result.get("rules")
    if not isinstance(rules, list) or not rules:
        return blocked("EMPTY_RULES")
    names, eligible = [], []
    for row in rules:
        if not isinstance(row, Mapping) or not isinstance(row.get("rule"), str):
            return blocked("INVALID_RULE")
        names.append(row["rule"])
        vals = [row.get("swing"), row.get("abs_swing"),
                row.get("rate_" + levels[0]), row.get("rate_" + levels[1])]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in vals):
            return blocked("NONFINITE_OR_MISSING_MEASUREMENT")
        swing, absolute, a, b = vals
        if not (-1 <= swing <= 1 and 0 <= absolute <= 1 and 0 <= a <= 1 and 0 <= b <= 1):
            return blocked("OUT_OF_RANGE_MEASUREMENT")
        if abs(abs(swing) - absolute) > 0.00011 or abs((a - b) - swing) > 0.00011:
            return blocked("INCONSISTENT_SWING")
        if absolute >= min_swing:
            eligible.append(row["rule"])
    if len(names) != len(set(names)) or set(names) != set(expected_rules):
        return blocked("RULE_COVERAGE_MISMATCH")
    return {"state": "COMPLETE", "eligible_rules": eligible,
            "can_test_cue_differential": bool(eligible), "min_swing": min_swing,
            "meaning": "observed sensitivity of declared rules only; not absence of all confounds"}


def checked_rule_sweep(result: Mapping, rows: Sequence[Mapping], pool: Sequence[str],
                       split_field: str, expected_rules: Sequence[str],
                       min_swing: float = 0.02) -> dict:
    """Use this entry point when the rendered rows are available. Read-only.

    It checks the actual inputs rather than inferring pool coverage from zero
    scores. The comparison remains restricted to the declared rule universe.
    """
    input_sha = digest({"rows": rows, "pool": list(pool), "split_field": split_field})
    if result.get("input_sha256") != input_sha:
        return {"state": "INAPPLICABLE", "eligible_rules": None,
                "problems": ["MEASUREMENT_INPUT_HASH_MISSING_OR_MISMATCHED"]}
    coverage = row_coverage(rows, ("context", "reference", split_field))
    levels = result.get("levels")
    if coverage["state"] != "COMPLETE" or not _ids(list(pool)):
        return {"state": "INAPPLICABLE", "eligible_rules": None,
                "problems": ["ROW_OR_POOL_COVERAGE_INVALID"]}
    if not isinstance(levels, list) or len(levels) != 2 or not _ids(levels):
        return {"state": "INAPPLICABLE", "eligible_rules": None,
                "problems": ["INVALID_LEVELS"]}
    counts = {level: 0 for level in levels}
    for row in rows:
        if (row[split_field] not in levels or row["reference"] not in pool
                or row["reference"] not in row["context"]
                or not any(c in row["context"] for c in pool)):
            return {"state": "INAPPLICABLE", "eligible_rules": None,
                    "problems": ["UNOBSERVED_CANDIDATE_OR_EXCLUDED_ROW"]}
        counts[row[split_field]] += 1
    if counts != result.get("n_per_level"):
        return {"state": "INAPPLICABLE", "eligible_rules": None,
                "problems": ["SWEEP_POPULATION_MISMATCH"]}
    report = sweep_coverage(result, pool_observed=True, expected_rules=expected_rules,
                            min_swing=min_swing)
    report["input_sha256"] = input_sha
    return report


def prediction_coverage(expected_ids: Sequence[str], records: Sequence[Mapping], *,
                        allow_extra: bool = False) -> dict:
    """Distinguish an emitted empty answer from a missing prediction field; reject duplicates."""
    if not _ids(list(expected_ids)):
        return {"state": "INAPPLICABLE", "problems": ["INVALID_EXPECTED_IDS"]}
    seen, problems = set(), []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
            problems.append("INVALID_PREDICTION_RECORD")
            continue
        rid = record["id"]
        if rid in seen:
            problems.append("DUPLICATE_PREDICTION:" + rid)
        seen.add(rid)
        if not isinstance(record.get("prediction"), str):
            problems.append("MISSING_OR_INVALID_PREDICTION:" + rid)
    missing, extra = sorted(set(expected_ids) - seen), sorted(seen - set(expected_ids))
    if missing:
        problems.append("MISSING_EXPECTED_ROWS")
    if extra and not allow_extra:
        problems.append("UNREGISTERED_EXTRA_ROWS")
    return {"state": "COMPLETE" if not problems else "INAPPLICABLE",
            "missing": missing, "extra": extra, "problems": problems}


def pair_coverage(rows: Sequence[Mapping], expected_cells: Sequence[str],
                  bands: Sequence[str] = ("early", "late")) -> dict:
    base = row_coverage(rows, ())
    problems, groups = list(base["problems"]), {}
    if not _ids(list(expected_cells)) or not _ids(list(bands)):
        problems.append("INVALID_PAIR_CONTRACT")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pid, cell, band = row.get("pair_id"), row.get("cell"), row.get("band")
        if (not isinstance(pid, str) or not pid or not isinstance(cell, str)
                or not isinstance(band, str) or cell not in expected_cells or band not in bands):
            problems.append("UNBOUND_PAIR_ROW")
            continue
        group = groups.setdefault(pid, {})
        if band in group:
            problems.append("DUPLICATE_PAIR_BAND:" + pid)
        group[band] = cell
    for pid, group in groups.items():
        if set(group) != set(bands) or len(set(group.values())) != 1:
            problems.append("INCOMPLETE_OR_CROSS_CELL_PAIR:" + pid)
    if set(expected_cells) != {r.get("cell") for r in rows if isinstance(r, Mapping)
                               and isinstance(r.get("cell"), str)}:
        problems.append("MISSING_OR_EXTRA_CELL")
    return {"state": "COMPLETE" if not problems else "INAPPLICABLE", "problems": problems}


def source_bindings(root: Path, bindings: Mapping[str, str]) -> dict:
    """Read-only Git blob identity check of inspected files, not a repository-wide certificate."""
    if not bindings:
        return {"state": "INAPPLICABLE", "problems": ["EMPTY_BINDINGS"]}
    problems = []
    root = root.resolve()
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            problems.append(relative + ":OUTSIDE_ROOT")
            continue
        try:
            data = path.read_bytes()
        except OSError:
            problems.append(relative + ":UNAVAILABLE")
            continue
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != expected:
            problems.append(relative + ":SOURCE_CHANGED_REAUDIT_REQUIRED")
    return {"state": "COMPLETE" if not problems else "INAPPLICABLE", "problems": problems}


def review_registry(document: Mapping) -> dict:
    entries = document.get("entries")
    if document.get("schema_version") != VERSION or not isinstance(entries, list) or not entries:
        return {"review_consistent": False, "error": "EMPTY_OR_INVALID_REGISTRY", "claims": []}
    reports, ids = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("claim"), dict) or not isinstance(entry.get("observation"), dict):
            return {"review_consistent": False, "error": "INVALID_ENTRY", "claims": reports}
        report = assess(entry["claim"], entry["observation"])
        report["declared_status"] = entry.get("declared_status")
        report["declaration_consistent"] = (report["status"] != "INVALID"
                                            and report["status"] == entry.get("declared_status")
                                            and report["claim_id"] not in ids)
        if isinstance(report["claim_id"], str):
            ids.add(report["claim_id"])
        reports.append(report)
    return {"review_consistent": all(r["declaration_consistent"] for r in reports),
            "supported_claim_count": sum(r["status"] == "SUPPORTED" for r in reports),
            "note": "review consistency is not verification; unresolved claims remain unresolved",
            "claims": reports}


def _strict_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key: " + k)
        result[k] = v
    return result


def load_json(path: Path) -> Any:
    def bad_constant(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object,
                      parse_constant=bad_constant)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--out", type=Path, help="create-only sidecar; never overwrite a receipt")
    args = parser.parse_args(argv)
    try:
        if bool(args.bindings) != bool(args.repo_root):
            raise ValueError("--bindings and --repo-root must be supplied together")
        report = review_registry(load_json(args.registry))
        if args.bindings:
            report["source_bindings"] = source_bindings(args.repo_root, load_json(args.bindings)["source_bindings"])
            report["review_consistent"] &= report["source_bindings"]["state"] == "COMPLETE"
        text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(text, end="")
        return 0 if report["review_consistent"] else 2
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({"review_consistent": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
