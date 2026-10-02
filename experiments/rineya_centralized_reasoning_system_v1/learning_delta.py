"""Same-panel learning arithmetic over bound endpoint bytes, not run attestation.

No dataset/model/host access, launch, qualification or promotion. Endpoint item
scores are declared evaluator evidence; their actual native truth is an external
review obligation. Original atlas/scorers remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from . import atlas_v2

VERSION = "CRS-SAME-PANEL-LEARNING-DELTA-v1"
MAX_RECEIPT_BYTES = 262144
MAX_CASES = 4096
SCALAR_TOLERANCE = 1e-12
META = ("panel_id", "panel_sha256", "scorer_sha256", "scoring_version", "endpoint", "scope")
FIELDS = {"schema_version", "status", *META, "checkpoint_sha256", "membership", "items", "proportion", "error"}
ITEM_FIELDS = {"semantic_exact", "strict_exact", "format_valid", "operational_error", "abstains"}
ENDPOINTS = {"semantic_accuracy": "semantic_exact", "strict_accuracy": "strict_exact"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_value(value):
    return type(value) is str and 0 < len(value) <= 256 and value == value.strip()


def proportion(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def finite_json(value, depth=0):
    require(depth <= 16, "endpoint nesting ceiling")
    if type(value) is dict:
        require(all(type(k) is str for k in value), "JSON keys must be strings")
        for child in value.values():
            finite_json(child, depth + 1)
    elif type(value) is list:
        for child in value:
            finite_json(child, depth + 1)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite endpoint value")
    else:
        require(value is None or type(value) in (str, int, bool), "JSON types required")


def canonical(value):
    finite_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def valid_hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_membership(membership):
    require(type(membership) is dict and 1 <= len(membership) <= MAX_CASES, "bounded nonempty case membership required")
    variants, parents = set(), {}
    for cid, row in membership.items():
        require(text_value(cid) and type(row) is dict and set(row) == {"parent_id", "variant_id", "family"}, "case identity schema")
        require(all(text_value(v) for v in row.values()), "explicit parent/variant/family identities")
        identity = (row["parent_id"], row["variant_id"])
        require(identity not in variants, "duplicate parent variant")
        variants.add(identity)
        previous = parents.setdefault(row["parent_id"], row["family"])
        require(previous == row["family"], "parent family drift")
    return membership


def validate_endpoint(endpoint):
    require(type(endpoint) is dict and set(endpoint) == FIELDS, "complete endpoint schema required")
    finite_json(endpoint)
    require(type(endpoint["schema_version"]) is int and endpoint["schema_version"] == 1, "endpoint version1 required")
    require(type(endpoint["status"]) is str and endpoint["status"] in {"AVAILABLE", "UNAVAILABLE", "OPERATIONAL_FAILURE"}, "endpoint availability status")
    require(all(text_value(endpoint[k]) for k in ("panel_id", "scoring_version", "scope")), "explicit panel/version/scope required")
    require(type(endpoint["endpoint"]) is str and endpoint["endpoint"] in ENDPOINTS, "only registered higher-is-better accuracy endpoints supported")
    require(all(valid_hash(endpoint[k]) for k in ("panel_sha256", "scorer_sha256", "checkpoint_sha256")), "exact panel/scorer/checkpoint hashes required")
    membership = validate_membership(endpoint["membership"])
    items = endpoint["items"]
    require(type(items) is dict and set(items) <= set(membership), "unregistered scored item")
    for cid, row in items.items():
        require(type(row) is dict and set(row) == ITEM_FIELDS and all(type(v) is bool for v in row.values()), "strict scoring booleans required")
        require(row["strict_exact"] == (row["semantic_exact"] and row["format_valid"]), "strict/semantic score inconsistency")
        if row["operational_error"]:
            require(not any(row[k] for k in ("semantic_exact", "strict_exact", "format_valid", "abstains")), "operational error cannot succeed or abstain")
    if endpoint["status"] != "AVAILABLE":
        require(endpoint["proportion"] is None and text_value(endpoint["error"]), "unavailable/failed endpoint has no numeric estimate")
        return endpoint
    require(endpoint["error"] is None and set(items) == set(membership), "every scheduled available case must be scored")
    require(proportion(endpoint["proportion"]), "baseline/candidate must be proportions in[0,1]")
    score_field = ENDPOINTS[endpoint["endpoint"]]
    computed = sum(row[score_field] for row in items.values()) / len(membership)
    require(math.isclose(endpoint["proportion"], computed, rel_tol=0, abs_tol=SCALAR_TOLERANCE), "proportion differs from complete scored-item denominator")
    return endpoint


def read_endpoint(receipt_bytes, expected_sha256):
    if receipt_bytes is None:
        require(expected_sha256 is None, "null endpoint cannot claim receipt hash binding")
        return None
    require(type(receipt_bytes) is bytes and 0 < len(receipt_bytes) <= MAX_RECEIPT_BYTES, "actual bounded endpoint receipt bytes required")
    require(valid_hash(expected_sha256) and sha256(receipt_bytes) == expected_sha256, "endpoint receipt byte hash mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate endpoint JSON field")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite endpoint constant: " + value)
    value = json.loads(receipt_bytes, object_pairs_hook=unique, parse_constant=bad)
    require(type(value) is dict and set(value) == {"schema_version", "kind", "payload"}
            and type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["kind"] == "CRS_LEARNING_ENDPOINT", "endpoint receipt envelope")
    return validate_endpoint(value["payload"])


def _pair(baseline, candidate, expected_membership):
    if expected_membership is not None:
        validate_membership(expected_membership)
    for endpoint in (baseline, candidate):
        if endpoint is not None and expected_membership is not None:
            require(canonical(endpoint["membership"]) == canonical(expected_membership), "endpoint differs from frozen expected case bank")
    if baseline is not None and candidate is not None:
        require(all(baseline[k] == candidate[k] for k in META), "baseline/candidate panel,scorer,version,endpoint or scope mismatch")
        require(canonical(baseline["membership"]) == canonical(candidate["membership"]), "baseline/candidate parent/item membership mismatch")
    available = baseline is not None and candidate is not None and baseline["status"] == candidate["status"] == "AVAILABLE"
    return available


def derive_delta(baseline_receipt, candidate_receipt, *, baseline_sha256, candidate_sha256,
                 declared_gain=None, expected_membership=None):
    baseline = read_endpoint(baseline_receipt, baseline_sha256)
    candidate = read_endpoint(candidate_receipt, candidate_sha256)
    available = _pair(baseline, candidate, expected_membership)
    gain = candidate["proportion"] - baseline["proportion"] if available else None
    if declared_gain is not None:
        require(type(declared_gain) in (int, float) and math.isfinite(declared_gain) and -1 <= declared_gain <= 1, "finite declared proportion difference required")
        require(gain is not None and math.isclose(declared_gain, gain, rel_tol=0, abs_tol=SCALAR_TOLERANCE), "declared gain is not bound same-panel candidate-minus-baseline")
    template = candidate if candidate is not None else baseline
    failed = any(e is not None and e["status"] == "OPERATIONAL_FAILURE" for e in (baseline, candidate))
    counts = None
    if available:
        score_field = ENDPOINTS[baseline["endpoint"]]
        ids = baseline["membership"]
        counts = {"scheduled_cases": len(ids),
                  "baseline_operational_errors": sum(baseline["items"][cid]["operational_error"] for cid in ids),
                  "candidate_operational_errors": sum(candidate["items"][cid]["operational_error"] for cid in ids),
                  "repairs": sum(not baseline["items"][cid][score_field] and candidate["items"][cid][score_field] for cid in ids),
                  "damages": sum(baseline["items"][cid][score_field] and not candidate["items"][cid][score_field] for cid in ids)}
    return {"schema_version": 1, "version": VERSION, "status": "AVAILABLE" if available else "OPERATIONAL_FAILURE" if failed else "UNAVAILABLE",
            "gain": gain, "baseline": None if baseline is None else baseline["proportion"],
            "candidate": None if candidate is None else candidate["proportion"],
            "source_checkpoint_sha256": None if baseline is None else baseline["checkpoint_sha256"],
            "candidate_checkpoint_sha256": None if candidate is None else candidate["checkpoint_sha256"],
            "metadata": None if template is None else {k: template[k] for k in META},
            "membership_sha256": None if template is None else sha256(canonical(template["membership"])),
            "baseline_receipt_sha256": baseline_sha256, "candidate_receipt_sha256": candidate_sha256,
            "paired_counts": counts,
            "same_panel_membership_verified": baseline is not None and candidate is not None,
            "expected_bank_bound": expected_membership is not None,
            "receipt_byte_association_only": True, "native_execution_attested": False,
            "qualification_granted": False, "promotion_allowed": False}


def efficiency_from_endpoints(row, baseline_receipt, candidate_receipt, *, baseline_sha256,
                              candidate_sha256, metric="acquisition", declared_gain=None,
                              expected_membership=None, measurement_receipt=None):
    """Use unchanged atlas_v2 only after endpoint/row/gain association.

    Native measurement receipt gain must equal the exact returned difference;
    rounded display values are separate. Qualification and dose comparison truth
    remain obligations of the external preregistered runner/reviewer.
    """
    atlas_v2.validate_row(row)
    require(type(metric) is str and metric in {"acquisition", "transfer", "retention"}, "registered accuracy atlas metric required")
    delta = derive_delta(baseline_receipt, candidate_receipt, baseline_sha256=baseline_sha256,
                         candidate_sha256=candidate_sha256, declared_gain=declared_gain,
                         expected_membership=expected_membership)
    if delta["source_checkpoint_sha256"] is not None:
        require(row["source_checkpoint_sha256"] == delta["source_checkpoint_sha256"], "atlas source checkpoint binding mismatch")
    if delta["candidate_checkpoint_sha256"] is not None:
        require(row["checkpoint_sha256"] == delta["candidate_checkpoint_sha256"], "atlas candidate checkpoint binding mismatch")
    if delta["metadata"] is not None:
        require(row["scorer_sha256"] == delta["metadata"]["scorer_sha256"], "atlas scorer binding mismatch")
    if delta["gain"] is not None:
        measure = row[metric]
        require(type(measure) is dict and measure["scope"] == delta["metadata"]["scope"]
                and math.isclose(measure["estimate"], delta["candidate"], rel_tol=0, abs_tol=SCALAR_TOLERANCE), "atlas metric is not this candidate endpoint/scope")
    ratios = atlas_v2.efficiency(delta["gain"], row, measurement_receipt=measurement_receipt)
    return {"delta": delta, "efficiency": ratios, "native_execution_attested": False,
            "qualification_granted": False, "promotion_allowed": False}
