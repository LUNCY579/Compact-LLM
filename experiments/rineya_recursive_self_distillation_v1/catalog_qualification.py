"""Offline, hash-bound catalog review prerequisites; no capability certification.

Human review records and their evidence files establish the declared provenance
chain. Hashes detect drift; this validator cannot establish that attestations are
true, certify intermediate reasoning, or perform semantic decontamination.
"""
from __future__ import annotations

import re
from pathlib import Path

from .contracts import file_hash, read_json, require
from .data import load_catalog, require_training_roles


REVIEW_STATUS = "REVIEWED_CATALOG_NOT_CAPABILITY_EVIDENCE"
REVIEW_FIELDS = {
    "schema", "experiment", "status", "evidence_kind", "catalog", "exclusions",
    "reviewed_by", "sources", "global_parent_grouping_attested",
    "historical_exclusion_authority_evidence", "reference_review_complete",
    "parent_training_overlap_reviewed",
}
SOURCE_FIELDS = {
    "source", "revision", "ownership_license_evidence",
    "independent_reference_review_evidence",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _bound_file(spec, base: Path, label: str):
    require(type(spec) is dict and set(spec) == {"path", "sha256"},
            "catalog review file binding fields: " + label)
    require(type(spec["path"]) is str and bool(spec["path"].strip()),
            "catalog review missing evidence path: " + label)
    require(type(spec["sha256"]) is str and SHA256.fullmatch(spec["sha256"]) is not None,
            "catalog review invalid SHA256: " + label)
    path = Path(spec["path"])
    path = (path if path.is_absolute() else base / path).resolve()
    require(path.is_file() and path.stat().st_size > 0,
            "catalog review missing/empty evidence file: " + label)
    require(file_hash(path) == spec["sha256"], "catalog review bound file drift: " + label)
    return path, {"path": str(path), "sha256": spec["sha256"]}


def validate_catalog_review(catalog_path, exclusions_path, review_path) -> dict:
    """Validate one explicit RSD catalog review without model imports or outputs.

    Schema 1 uses exact REVIEW_FIELDS and SOURCE_FIELDS. All file bindings are
    ``{path, sha256}``; relative paths resolve beside the review. ``real_review``
    denotes the operator's claim about evidence, never independent verification
    by this function. Synthetic tests exercise this contract only.
    """
    catalog_path, exclusions_path, review_path = (
        Path(path).resolve() for path in (catalog_path, exclusions_path, review_path)
    )
    review = read_json(review_path)
    require(type(review) is dict and set(review) == REVIEW_FIELDS,
            "catalog review schema fields")
    require(type(review["schema"]) is int and review["schema"] == 1
            and review["experiment"] == "RSD-001" and review["status"] == REVIEW_STATUS,
            "catalog review experiment/schema/status")
    require(review["evidence_kind"] == "real_review",
            "catalog review requires real reviewed provenance; fixtures cannot qualify")
    require(type(review["reviewed_by"]) is str and bool(review["reviewed_by"].strip()),
            "catalog review named reviewer required")
    for field in ("global_parent_grouping_attested", "reference_review_complete",
                  "parent_training_overlap_reviewed"):
        require(review[field] is True, "catalog review missing attestation: " + field)
    catalog, catalog_binding = _bound_file(review["catalog"], review_path.parent, "catalog")
    exclusions, exclusions_binding = _bound_file(review["exclusions"], review_path.parent, "exclusions")
    require(catalog == catalog_path and exclusions == exclusions_path,
            "catalog review input path mismatch")
    tasks, audit = load_catalog(catalog, exclusions)
    require_training_roles(tasks)
    require(not any("fixture" in task.provenance.casefold() for task in tasks),
            "fixture cannot enroll natively through a catalog review")
    expected = {(task.source, task.revision) for task in tasks}
    require(type(review["sources"]) is list and bool(review["sources"]),
            "catalog review source evidence required")
    records, seen = [], set()
    input_paths = {catalog, exclusions, review_path}
    for source in review["sources"]:
        require(type(source) is dict and set(source) == SOURCE_FIELDS,
                "catalog review source fields")
        require(all(type(source[field]) is str and bool(source[field].strip())
                    for field in ("source", "revision")), "catalog review source identity")
        identity = source["source"], source["revision"]
        require(identity not in seen, "catalog review duplicate source revision")
        seen.add(identity)
        record = {field: source[field] for field in ("source", "revision")}
        for field in ("ownership_license_evidence", "independent_reference_review_evidence"):
            evidence_path, record[field] = _bound_file(source[field], review_path.parent, field)
            require(evidence_path not in input_paths,
                    "catalog review source evidence must be a separate evidence file")
        records.append(record)
    require(seen == expected, "catalog review source revision coverage mismatch")
    authority_path, authority = _bound_file(review["historical_exclusion_authority_evidence"],
                                           review_path.parent, "historical exclusion authority")
    require(authority_path not in input_paths,
            "catalog review historical exclusion authority needs separate evidence")
    return {
        "status": REVIEW_STATUS,
        "review": {"path": str(review_path), "sha256": file_hash(review_path)},
        "catalog": catalog_binding, "exclusions": exclusions_binding,
        "reviewed_by": review["reviewed_by"],
        "sources": sorted(records, key=lambda record: (record["source"], record["revision"])),
        "historical_exclusion_authority_evidence": authority,
        "audit": audit, "semantic_decontamination": False,
        "reference_validation": "endpoint-only; intermediate reasoning not certified by this validator",
        "evidence_limit": "hash-bound human review attestations; their truth is not established by this validator",
        "native_efficacy_claim": False,
    }
