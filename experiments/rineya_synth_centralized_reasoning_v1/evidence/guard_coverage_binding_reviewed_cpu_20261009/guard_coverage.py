"""Closed descriptor-only guard coverage planning; no scoring or payload I/O.

Policy references, candidate counts, qualification references and enrollment
references remain caller declarations. No returned field certifies native work,
source/parent disjointness, enrolled source-correct support or model quality.
The existing quality decision/scorer stays external and unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "synth_guard_coverage_plan_v1"
SHORT_DOMAINS = ("arithmetic", "retrieval", "abstention", "dialogue")
DOMAINS = SHORT_DOMAINS + ("long_context",)
POLICY_IDS = {
    "engineering": "SYNTH-QUALITY-3PP-ZERO-DAMAGE-20261005",
    "task": "SYNTH-TASK-GUARD-001",
    "statistics": "SYNTH-DEV-STATISTICS-001",
}
TASK_MAPPING = {"retrieval": "retrieval", "abstention": "answerability_abstention"}
DISJOINT_RELATIONS = (
    "training_vs_short_guards", "dev_vs_short_guards",
    "training_vs_long_context", "dev_vs_long_context", "short_guards_vs_long_context",
)
CLAIM_FIELDS = (
    "native_admissible", "scoring_authorized", "retention_qualified",
    "model_quality_gain", "coverage_complete", "retention_eligible",
)
# Convenience template only; mutable caller aliases are never authority inputs.
CLOSED_CLAIMS = dict.fromkeys(CLAIM_FIELDS, False)


def closed_claims():
    """Fresh permanently false claims, independent of any caller-owned template."""
    return dict.fromkeys(CLAIM_FIELDS, False)


class ValidationError(ValueError):
    """A malformed, stale, conflicting or authority-escalating plan was refused."""


def _require(condition, label):
    if not condition:
        raise ValidationError(label)


def _json(value):
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "JSON keys must be strings")
        for child in value.values():
            _json(child)
    elif type(value) is list:
        for child in value:
            _json(child)
    else:
        _require(value is None or type(value) in (str, bool, int, float),
                 "only exact JSON types are allowed")


def canonical(value):
    try:
        _json(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as error:
        raise ValidationError("finite closed JSON required") from error


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _same(actual, expected, label):
    _require(canonical(actual) == canonical(expected), label)


def _object(value, fields, label):
    _require(type(value) is dict and set(value) == set(fields), label + ": exact fields required")


def _text(value, label):
    _require(type(value) is str and bool(value.strip()) and value == value.strip()
             and all(ord(c) >= 32 for c in value), label + ": nonempty text required")


def _hash(value, label):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             label + ": lowercase SHA256 required")


def _count(value, label):
    _require(value is None or type(value) is int and value >= 0,
             label + ": missing or nonnegative integer required; bool is refused")


def _descriptor(value, label, references):
    _object(value, {"path", "sha256"}, label)
    _text(value["path"], label + ".path")
    path = value["path"].replace("\\", "/")
    _require(re.match(r"^[A-Za-z]:/", path) is not None, label + ": absolute stable Windows path required")
    parts = path[3:].split("/")
    _require(parts and all(p and p not in (".", "..") and not p.endswith((".", " "))
                           and not any(c in p for c in ':*?"<>|') for p in parts),
             label + ": aliased/device path refused")
    devices = {"con", "prn", "aux", "nul", "conin$", "conout$"}
    devices |= {prefix + suffix for prefix in ("com", "lpt") for suffix in "123456789¹²³"}
    _require(all(p.split(".", 1)[0].casefold() not in devices for p in parts),
             label + ": Windows device path refused")
    _hash(value["sha256"], label + ".sha256")
    normalized = path.casefold()
    if normalized in references:
        _same(value["sha256"], references[normalized],
              label + ": conflicting hashes for the same normalized descriptor path")
    references[normalized] = value["sha256"]
    return normalized


def _policy_descriptors(policies, references):
    _object(policies, POLICY_IDS, "frozen_policies")
    paths = set()
    for role, policy in policies.items():
        _object(policy, {"policy_id", "path", "sha256"}, "frozen_policies." + role)
        _same(policy["policy_id"], POLICY_IDS[role], "frozen policy role/identity substitution refused")
        path = _descriptor({k: policy[k] for k in ("path", "sha256")}, "frozen_policies." + role, references)
        _require(path not in paths, "conflicting selected policy paths")
        paths.add(path)


def _slot(value, label, references):
    _object(value, {"status", "descriptor"}, label)
    if value["status"] == "PENDING":
        _same(value["descriptor"], None, label + ": pending binding must remain None")
        return None
    _same(value["status"], "DECLARED_UNVERIFIED",
          label + ": receipt metadata cannot claim qualification/enrollment/readiness")
    return _descriptor(value["descriptor"], label + ".descriptor", references)


def _disjointness(value, references):
    _object(value, DISJOINT_RELATIONS, "disjointness")
    for relation, pair in value.items():
        _object(pair, {"source", "parent"}, "disjointness." + relation)
        for unit, declaration in pair.items():
            label = "disjointness." + relation + "." + unit
            _object(declaration, {"status", "declared_disjoint", "descriptor"}, label)
            if declaration["status"] == "PENDING":
                _same(declaration["declared_disjoint"], None, label + ": missing assertion remains None")
                _same(declaration["descriptor"], None, label + ": missing declaration reference remains None")
            else:
                _same(declaration["status"], "DECLARED_UNVERIFIED", label + ": assertions never certify disjointness")
                _same(declaration["declared_disjoint"], True, label + ": known overlap/conflicting assertion refused")
                if declaration["descriptor"] is not None:
                    _descriptor(declaration["descriptor"], label + ".descriptor", references)


def validate_coverage(plan, frozen_policies, expected_plan_sha256):
    """Join one frozen CLOSED plan and return structural preparation only.

    frozen_policies is an independently selected descriptor registry; this
    function does not read it from disk or verify policy/receipt file contents.
    Positive declared counts with receipt refs remain unverified counts. All
    qualified-source-correct counts and scientific coverage remain undefined.
    """
    references = {}
    _policy_descriptors(frozen_policies, references)
    _hash(expected_plan_sha256, "expected_plan_sha256")
    _require(digest(plan) == expected_plan_sha256, "plan changed from externally frozen SHA256")
    _object(plan, {"schema", "experiment", "status", "selected_policies", "required_short_domains",
                   "long_context_domain", "domains", "disjointness", "claims",
                   "scientific_coverage_status"}, "plan")
    for key, value in {
        "schema": SCHEMA, "experiment": "SYNTH-CRS-001", "status": "CLOSED_PREPARATION",
        "selected_policies": frozen_policies, "required_short_domains": list(SHORT_DOMAINS),
        "long_context_domain": "long_context", "claims": closed_claims(),
        "scientific_coverage_status": "UNDEFINED",
    }.items():
        _same(plan[key], value, "plan." + key + ": closed policy/source/authority binding mismatch")
    domains = plan["domains"]
    _require(type(domains) is list and len(domains) == len(DOMAINS),
             "all four short domains plus separate long_context are required")
    names, population_scopes, missing, domain_results = set(), set(), [], []
    for domain in domains:
        _object(domain, {"domain", "context", "mechanism_mapping", "population_scope_id",
                         "population", "scorer", "qualification", "enrollment",
                         "candidate_rows", "declared_source_correct_rows"}, "domain")
        name = domain["domain"]
        _require(type(name) is str and name in DOMAINS, "unknown/substituted domain name")
        _require(name not in names, "duplicate domain/scope")
        names.add(name)
        label = "domains." + name
        _same(domain["context"], "long" if name == "long_context" else "short",
              label + ": long_context cannot substitute for a short domain")
        mapping = domain["mechanism_mapping"]
        if mapping is None:
            missing.append(label + ".mechanism_mapping")
        else:
            _object(mapping, {"policy", "mechanism", "policy_sha256"}, label + ".mechanism_mapping")
            _require(name in TASK_MAPPING, label + ": current task mechanisms cannot substitute arithmetic/dialogue/long_context")
            _same(mapping, {"policy": "task", "mechanism": TASK_MAPPING[name],
                            "policy_sha256": frozen_policies["task"]["sha256"]},
                  label + ": stale policy hash or mechanism/domain substitution")
        descriptor_paths = {}
        for role in ("population", "scorer", "qualification", "enrollment"):
            path = _slot(domain[role], label + "." + role, references)
            if path is None:
                missing.append(label + "." + role + ".descriptor")
            else:
                _require(path not in descriptor_paths.values(),
                         label + ": candidate/scorer/qualification/enrollment descriptor-role substitution")
                descriptor_paths[role] = path
        scope = domain["population_scope_id"]
        if domain["population"]["descriptor"] is None:
            _same(scope, None, label + ": unbound population scope must remain None")
        else:
            _require(type(scope) is str and re.fullmatch(r"[a-z][a-z0-9_.:-]{0,95}", scope) is not None,
                     label + ": exact population scope ID required")
            _require(scope not in population_scopes, "conflicting/duplicate population scope")
            population_scopes.add(scope)
        candidates, declared = domain["candidate_rows"], domain["declared_source_correct_rows"]
        _count(candidates, label + ".candidate_rows")
        _count(declared, label + ".declared_source_correct_rows")
        if domain["enrollment"]["descriptor"] is None:
            _same(declared, None, label + ": candidates/booleans cannot supply enrolled source-correct counts")
        elif declared is not None and candidates is not None:
            _require(declared <= candidates, label + ": declared source-correct rows exceed candidate population")
        if declared is None:
            missing.append(label + ".declared_source_correct_rows")
        domain_results.append({
            "domain": name, "context": domain["context"],
            "declared_candidate_rows": candidates, "declared_source_correct_rows": declared,
            "qualified_source_correct_rows": None, "actual_support": None,
            "scientific_coverage_status": "UNDEFINED",
        })
    _same(sorted(names), sorted(DOMAINS), "omitted required domain")
    _disjointness(plan["disjointness"], references)
    for relation, pair in plan["disjointness"].items():
        for unit, declaration in pair.items():
            if declaration["declared_disjoint"] is None:
                missing.append("disjointness." + relation + "." + unit + ".declared_disjoint")
            if declaration["descriptor"] is None:
                missing.append("disjointness." + relation + "." + unit + ".descriptor")
    return {
        "schema": "synth_guard_coverage_validation_v1", "status": "CLOSED_PLAN_STRUCTURALLY_VALID",
        "plan_sha256": expected_plan_sha256, "structural_only": True,
        **closed_claims(), "scientific_coverage_status": "UNDEFINED",
        "actual_support": None,
        "domains": domain_results, "missing_bindings": missing,
        "preserved_plan": json.loads(canonical(plan)),
        "policy_contents_verified": False, "receipt_contents_verified": False,
        "source_disjointness_verified": False, "parent_disjointness_verified": False,
        "actual_enrollment_verified": False, "payloads_read": False,
        "external_requirement": "Exact independently qualified population/scorer/qualification and actual pre-candidate source-correct enrollment remain owner verification; no coverage or quality decision is supplied.",
    }
