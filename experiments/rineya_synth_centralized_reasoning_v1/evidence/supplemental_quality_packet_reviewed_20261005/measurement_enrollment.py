"""Offline exact enrollment arithmetic; never qualifies a grader or native data.

Caller-provided correctness flags only define a computational frozen cohort.
SYNTH's owner must separately admit genuine model/scorer/audit evidence before
using any such membership as native study authority.
"""
from __future__ import annotations

import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _validated(cohort, baseline):
    require(type(cohort) is dict and type(cohort.get("rows")) is list and cohort["rows"], "complete cohort required")
    require(type(baseline) is dict, "typed baseline required")
    keys, parents, roles = set(), {}, {}
    for row in cohort["rows"]:
        require(type(row) is dict and all(type(row.get(k)) is str and row[k].strip()
            for k in ("key", "parent_id", "source", "role", "stratum", "rendering", "prompt_hash")),
            "exact public cohort metadata required")
        require(row["role"] in {"R", "S", "RS", "retention"}, "unregistered cohort role")
        require(len(row["prompt_hash"]) == 64 and all(c in "0123456789abcdef" for c in row["prompt_hash"]),
                "cohort prompt hash")
        require(row["key"] not in keys, "duplicate cohort key")
        keys.add(row["key"])
        family = (row["role"], row["source"])
        require(row["parent_id"] not in parents or parents[row["parent_id"]] == family,
                "parent crosses skills or retention sources")
        parents[row["parent_id"]] = family
        roles[row["key"]] = row
    require(set(baseline) == keys, "baseline must cover the exact complete cohort; no omitted rows")
    for key, result in baseline.items():
        require(type(result) is dict and all(type(result.get(k)) is bool
            for k in ("correct", "format_correct", "abstained", "capped")), "typed unchanged-source outcomes")
    return roles


def freeze_enrollment(cohort, baseline, *, baseline_id, required_sources):
    """Freeze every guard row and M0-correct member; never filter the cohort."""
    require(type(baseline_id) is str and baseline_id.strip(), "explicit unchanged baseline ID required")
    require(type(required_sources) is list and required_sources
            and all(type(s) is str and s.strip() for s in required_sources)
            and len(set(required_sources)) == len(required_sources), "explicit unique retention sources")
    rows = _validated(cohort, baseline)
    guards = {}
    for source in required_sources:
        selected = [r for r in rows.values() if r["role"] == "retention" and r["source"] == source]
        enrolled = sorted(r["key"] for r in selected)
        prior_correct = sorted(r["key"] for r in selected if baseline[r["key"]]["correct"])
        guards[source] = {"all_guard_row_ids": enrolled, "prior_correct_row_ids": prior_correct,
            "parent_ids": sorted({r["parent_id"] for r in selected}),
            "status": "COMPUTATIONAL_SUPPORT_PRESENT" if prior_correct else "UNDEFINED_ZERO_SOURCE_CORRECT_SUPPORT"}
    present = {r["source"] for r in rows.values() if r["role"] == "retention"}
    require(present <= set(required_sources), "undeclared retention source cannot be silently excluded")
    payload = {"schema": "synth-computational-enrollment-v1", "baseline_id": baseline_id,
        "cohort_sha256": digest(cohort), "baseline_sha256": digest(baseline),
        "complete_row_ids": sorted(rows), "guards": guards,
        "population_not_filtered_by_source_success": True,
        "native_membership_admitted": False, "caller_correctness_flags_independently_qualified": False,
        "native_scoring_authorized": False, "training_authorized": False, "confirmation_authorized": False,
        "promotion_authorized": False,
        "note": "Computational enrollment only. Genuine SYNTH source/scorer/baseline admission remains separate."}
    return {"payload": payload, "sha256": digest(payload)}


def check_enrollment(enrollment, cohort, baseline, *, baseline_id, required_sources):
    require(type(enrollment) is dict and set(enrollment) == {"payload", "sha256"}, "enrollment envelope")
    require(enrollment["sha256"] == digest(enrollment["payload"]), "enrollment checksum drift")
    expected = freeze_enrollment(cohort, baseline, baseline_id=baseline_id, required_sources=required_sources)
    require(enrollment == expected, "frozen source/cohort/membership drift; no retrospective substitution")
    return enrollment["payload"]
