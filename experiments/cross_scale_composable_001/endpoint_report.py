"""Typed, development-only endpoint comparison; no model or bank discovery.

The caller supplies a frozen registry and its independently retained digest.
Registry integrity is checked, not authenticated: it must come from qualified
run/output custody rather than being generated from arbitrary prediction files.
Training roots are honest per-subject ancestry, never forced into evaluation keys.
registry.source_commit identifies the current clean report implementation.
Each subject.source_commit identifies its frozen actual probe implementation;
training-root source commits identify the corresponding historical training code.
Historical probe/training commits must resolve in Git, but need not equal report HEAD.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from . import receipts as r


EVALUATION_HASHES = (
    "views_sha256", "evaluation_bank_sha256", "tokenizer_sha256", "runtime_binding_sha256",
    "model_binding_sha256", "base_checkpoint_sha256", "implementation_digest",
    "base_probe_weight_sha256",
    "scoring_contract_sha256", "decoding_contract_sha256", "endpoint_population_sha256",
)
ROOT_HASHES = (
    "base_checkpoint_sha256", "delta_sha256", "dataset_sha256", "training_rows_sha256",
    "parent_partition_sha256", "training_recipe_sha256",
)
RELATIONS = {"independent_training", "same_checkpoint_transform", "base_anchor",
             "fixed_component_composition", "paired_compositions"}
KINDS = {"base", "trained", "transform", "composition"}
SELECT_VIEWS_SHA256 = "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3"
MATCHABLE_ROOT_FIELDS = {"dataset_sha256", "training_rows_sha256", "parent_partition_sha256",
                         "optimizer_steps", "rank_budget", "parameter_budget", "target_modules"}


def _commit(value):
    r.require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value), "invalid source commit")


def _root(record, expected_base):
    r.require(isinstance(record, dict), "invalid training root")
    for key in ROOT_HASHES:
        r.valid_hash(record.get(key), "training root " + key)
    _commit(record.get("source_commit"))
    r.require(record["base_checkpoint_sha256"] == expected_base, "training root has incompatible base")
    r.require(isinstance(record.get("run_id"), str) and record["run_id"], "missing root run ID")
    for key in ("optimizer_steps", "rank_budget", "parameter_budget"):
        r.require(type(record.get(key)) is int and record[key] > 0, "invalid root " + key)
    r.validate_modules(record.get("target_modules"))
    r.require(record.get("completed") is True, "training root incomplete")


def _term(term, roots):
    r.require(isinstance(term, dict) and set(term) == {
        "root_sha256", "delta_sha256", "transform_receipt_sha256", "coefficient"}, "invalid composition term")
    root = term.get("root_sha256")
    r.require(root in roots, "unregistered component training root")
    r.valid_hash(term.get("delta_sha256"), "component delta")
    transform = term.get("transform_receipt_sha256")
    if transform is None:
        r.require(term["delta_sha256"] == roots[root]["delta_sha256"], "component differs from learned root")
    else:
        r.valid_hash(transform, "component transform receipt")
    coefficient = term.get("coefficient")
    r.require(type(coefficient) in (int, float) and math.isfinite(coefficient) and coefficient > 0,
              "component coefficient must be positive finite")
    return {**term, "coefficient": float(coefficient)}


def subject_term(subject):
    """Identity of the exact standalone delta, with coefficient one in a sum."""
    r.require(subject.get("kind") in ("trained", "transform") and len(subject.get("training_roots", [])) == 1,
              "only a single learned delta can be a fixed constituent")
    return {"root_sha256": subject["training_roots"][0], "delta_sha256": subject["delta_sha256"],
            "transform_receipt_sha256": subject["transform_receipt_sha256"], "coefficient": 1.0}


def _subject(subject, roots, evaluation):
    r.require(isinstance(subject, dict) and subject.get("kind") in KINDS, "invalid subject kind")
    _commit(subject.get("source_commit"))
    r.require(subject.get("evaluation") == evaluation, "subject evaluation/runtime/base contract differs")
    for key in ("probe_weight_sha256", "subject_receipt_sha256", "endpoint_sha256"):
        r.valid_hash(subject.get(key), "subject " + key)
    r.require(subject.get("completed") is True, "subject probe incomplete")
    ancestry = subject.get("training_roots")
    r.require(isinstance(ancestry, list) and ancestry == sorted(set(ancestry))
              and all(root in roots for root in ancestry), "invalid subject training ancestry")
    kind = subject["kind"]
    if kind == "base":
        r.require(not ancestry and subject.get("delta_sha256") is None
                  and subject.get("transform_receipt_sha256") is None and subject.get("terms") == [],
                  "base must have no fabricated training/delta ancestry")
        r.require(not any(key in subject for key in ("training_run_id", "training_rows_sha256", "optimizer_steps")),
                  "base contains fabricated training metadata")
        r.require(subject["probe_weight_sha256"] == evaluation["base_probe_weight_sha256"],
                  "base probe weights differ from the qualified base realization")
        return
    r.valid_hash(subject.get("delta_sha256"), "subject delta")
    if kind in ("trained", "transform"):
        r.require(len(ancestry) == 1 and subject.get("terms") == [], "single-draw subject has composite ancestry")
        if kind == "trained":
            r.require(subject["delta_sha256"] == roots[ancestry[0]]["delta_sha256"]
                      and subject.get("transform_receipt_sha256") is None, "trained subject differs from root")
        else:
            r.valid_hash(subject.get("transform_receipt_sha256"), "transform receipt")
    else:
        r.require(subject.get("transform_receipt_sha256") is None, "composition needs component transform receipts")
        terms = subject.get("terms")
        r.require(isinstance(terms, list) and len(terms) == 2, "initial composition requires two fixed components")
        normalized = [_term(term, roots) for term in terms]
        r.require(len({term["root_sha256"] for term in normalized}) == 2, "composition repeats one learned root")
        r.require(sorted(term["root_sha256"] for term in normalized) == ancestry, "composition ancestry omits a component")
        r.require(not any(key in subject for key in ("training_run_id", "training_rows_sha256", "optimizer_steps")),
                  "composition cannot invent a single training run")


def _relation(left, right, declaration):
    relation = declaration.get("relation")
    r.require(relation in RELATIONS, "unregistered contrast relation")
    a, b = left["training_roots"], right["training_roots"]
    singles = left["kind"] in ("trained", "transform") and right["kind"] in ("trained", "transform")
    if relation == "independent_training":
        r.require(singles and len(a) == len(b) == 1 and a != b,
                  "independent-training relation needs disjoint singleton roots")
    elif relation == "same_checkpoint_transform":
        r.require(singles and a == b and len(a) == 1
                  and "transform" in (left["kind"], right["kind"]),
                  "same-checkpoint transform relation has different ancestry")
    elif relation == "base_anchor":
        r.require((left["kind"] == "base") != (right["kind"] == "base"),
                  "base-anchor relation requires exactly one honest base")
    elif relation == "fixed_component_composition":
        r.require((left["kind"] == "composition") != (right["kind"] == "composition"),
                  "fixed-composition relation requires one composition and its constituent")
        single, composition = (right, left) if left["kind"] == "composition" else (left, right)
        wanted = subject_term(single)
        r.valid_hash(declaration.get("fixed_component_sha256"), "registered fixed component")
        r.require(r.digest(wanted) == declaration["fixed_component_sha256"], "fixed constituent declaration differs")
        normalized = [{**term, "coefficient": float(term["coefficient"])} for term in composition["terms"]]
        r.require(sum(term == wanted for term in normalized) == 1,
                  "composition does not retain the exact declared constituent at coefficient one")
    else:
        r.require(left["kind"] == right["kind"] == "composition" and a == b and len(a) == 2,
                  "paired compositions require the same two training roots")
        fixed = declaration.get("fixed_component_sha256")
        r.valid_hash(fixed, "registered fixed composition component")
        shared = []
        for subject in (left, right):
            normalized = [{**term, "coefficient": float(term["coefficient"])} for term in subject["terms"]]
            matches = [term for term in normalized if term["coefficient"] == 1.0 and r.digest(term) == fixed]
            r.require(len(matches) == 1, "paired compositions lack the exact fixed coefficient-one component")
            shared.append(matches[0])
        r.require(shared[0] == shared[1], "paired compositions changed the protected constituent")
    return relation == "independent_training"


def validate_registry(registry, expected_registry_sha256):
    r.valid_hash(expected_registry_sha256, "registry digest")
    r.require(isinstance(registry, dict) and r.digest(registry) == expected_registry_sha256, "registry digest differs")
    r.require(type(registry.get("schema")) is int and registry["schema"] == 1, "unsupported endpoint registry")
    _commit(registry.get("source_commit"))
    evaluation = registry.get("evaluation")
    r.require(isinstance(evaluation, dict), "missing evaluation contract")
    for key in EVALUATION_HASHES:
        r.valid_hash(evaluation.get(key), "evaluation " + key)
    r.require(evaluation.get("use") == "DEV" and evaluation.get("instrument_role") in
              ("ADMITTED_FOR_SELECTION", "REPORT_ONLY"), "confirmation/training use is closed")
    r.require(isinstance(evaluation.get("endpoint_name"), str) and evaluation["endpoint_name"], "missing endpoint name")
    r.require(isinstance(evaluation.get("instrument_id"), str) and evaluation["instrument_id"], "missing instrument ID")
    if evaluation.get("endpoint_domain") == "acquisition":
        r.require(evaluation["instrument_id"] == "SELECT-VIEWS-001"
                  and evaluation["instrument_role"] == "ADMITTED_FOR_SELECTION"
                  and evaluation["views_sha256"] == SELECT_VIEWS_SHA256,
                  "initial acquisition report requires SELECT development instrument")
    else:
        r.require(evaluation.get("endpoint_domain") in ("retention", "long_context", "abstention"), "unsupported endpoint domain")
        if evaluation["endpoint_domain"] in ("long_context", "abstention"):
            r.require(evaluation["instrument_role"] == "REPORT_ONLY", "spent endpoint cannot acquire selection role")
    ids = registry.get("endpoint_row_ids")
    r.require(isinstance(ids, list) and bool(ids) and all(isinstance(row, str) and row for row in ids)
              and len(ids) == len(set(ids)), "invalid exact endpoint population")
    r.require(r.digest(ids) == evaluation["endpoint_population_sha256"], "registered population digest differs")
    roots = registry.get("training_roots")
    r.require(isinstance(roots, dict), "missing root registry")
    run_ids = set()
    for key, record in roots.items():
        r.valid_hash(key, "root registry key")
        r.require(r.digest(record) == key, "training root receipt digest differs")
        _root(record, evaluation["base_checkpoint_sha256"])
        r.require(record["run_id"] not in run_ids, "one training run relabelled as independent roots")
        run_ids.add(record["run_id"])
    subjects = registry.get("subjects")
    r.require(isinstance(subjects, dict) and set(subjects) == {"left", "right"}, "exact two compared subjects required")
    for subject in subjects.values():
        _subject(subject, roots, evaluation)
    contrast = registry.get("contrast")
    r.require(isinstance(contrast, dict), "missing registered contrast")
    independent = _relation(subjects["left"], subjects["right"], contrast)
    matched = contrast.get("matched_root_fields", [])
    r.require(isinstance(matched, list) and all(isinstance(key, str) for key in matched)
              and len(matched) == len(set(matched)) and set(matched) <= MATCHABLE_ROOT_FIELDS,
              "invalid matched-training contract")
    if matched:
        r.require(independent, "matched training fields require singleton independent-draw contrast")
        left_root, right_root = (roots[subjects[side]["training_roots"][0]] for side in ("left", "right"))
        for key in matched:
            r.require(left_root[key] == right_root[key], "registered matched training field differs: " + key)
    return independent


def compare_endpoint_files(left, right, registry, expected_registry_sha256, *, source_root):
    """left-minus-right exact paired DEV report after all metadata/custody gates.

    Artifacts are {manifest: actual_metadata, path: endpoint_json}. Endpoint JSON
    contains ordered row-ID -> Boolean outcomes, with no population intersections.
    Native collector/scorer custody must already have frozen the expected registry.
    """
    independent = validate_registry(registry, expected_registry_sha256)
    for side, artifact in (("left", left), ("right", right)):
        r.require(artifact.get("manifest") == registry["subjects"][side], "observed subject/weights differ from registry")
    r.verify_source(source_root, registry["source_commit"], clean=True)
    historical_commits = {root["source_commit"] for root in registry["training_roots"].values()}
    historical_commits.update(subject["source_commit"] for subject in registry["subjects"].values())
    for commit in historical_commits:
        checked = subprocess.run(["git", "-C", str(source_root), "rev-parse", "--verify", commit + "^{commit}"],
                                 capture_output=True, text=True, timeout=30)
        r.require(checked.returncode == 0 and checked.stdout.strip() == commit,
                  "training/probe source commit does not resolve to an actual exact Git commit")
    for side, artifact in (("left", left), ("right", right)):
        r.verify_file(artifact["path"], registry["subjects"][side]["endpoint_sha256"])
    outputs = []
    for artifact in (left, right):
        value = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"), object_pairs_hook=r._unique_json_object)
        r.require(isinstance(value, dict) and list(value) == registry["endpoint_row_ids"],
                  "endpoint rows/order differ from frozen population")
        outputs.append(value)
    # The 20-row instrument floor applies only to acquisition across training draws;
    # other endpoints keep their registered paired interpretation and own thresholds.
    floor_applies = independent and registry["evaluation"]["endpoint_domain"] == "acquisition"
    paired = r.paired_report(*outputs, cross_training_draw=floor_applies)
    return {"schema": 1, "evidence_kind": "DEVELOPMENT_ONLY", "relation": registry["contrast"]["relation"],
            "independent_training_roots": independent, "acquisition_20_row_floor_applies": floor_applies,
            "net": paired["net"], "nd": paired["nd"], "exact_p": paired["exact_p"], "p": paired["p"],
            "paired": paired, "registry_sha256": expected_registry_sha256,
            "instrument_id": registry["evaluation"]["instrument_id"],
            "endpoint_name": registry["evaluation"]["endpoint_name"],
            "improvement_established": False, "confirmation_claim": False,
            "interpretation": "Paired diagnostics only. Floor passage and nominal p are not a significance-family, retention, replication, or improvement verdict."}
