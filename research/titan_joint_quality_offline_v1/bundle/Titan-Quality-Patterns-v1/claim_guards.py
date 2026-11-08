"""Additive fixture claim guards; reviewed v2 source remains immutable.

No model or native result import exists. Legacy diagnostics remain intact. The
eligibility and deep-gain guards are explicit additional prospective rules.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
import uuid


V2_ROOT = Path("C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2")
V2_RUNS = Path("C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2-runs")
REVIEW_PATH = Path("C:/Users/clclc/BusinessProject/Titan-Quality-Patterns-v1-runs/v2-instrument-independent-001.json")
REVIEW_SHA256 = "2f47cb3a172f629d06c89dd1103efa1f1fdef777849cbe2a9c2206984a3420ba"
BOUND_SOURCES = {
    "titan_joint_quality_v1/__init__.py": "91a483aa9c8cb0b4e281600fe91f93ddc2c916b8975db669f0400621a6715357",
    "titan_joint_quality_v1/common.py": "686371e14eaebdacac1902bf50d40ca35b87584a990b6d9630d95e909d53fd32",
    "titan_joint_quality_v1/data.py": "dd4a39762f805b4d2a99d9c7416cbf454c2d99244ca2f42e4e73faa17d4dba1b",
    "titan_joint_quality_v1/analysis.py": "58da8ae3f794b20bffcad4dfa86e9ad7b65bfc179db837bae439a834f02b603e",
}
NON_REASONING_DOMAINS = ("transition", "arithmetic", "grounding", "long_context", "format")
MODELS = ("supa", "sparse", "transition", "control", "joint")


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _fixture_input_bindings(prompts, references, predictions, freeze, pair_proof, floor):
    """Bind actual caller values; source review does not attest these fixtures."""
    values = {"prompts": prompts, "references": references, "predictions": predictions,
              "baseline_freeze": freeze, "pair_proof": pair_proof,
              "reasoning_shortcut_floor": floor}
    def canonical_sha(value):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        return _sha_bytes(encoded)
    return {"encoding": "UTF8_SORTED_KEYS_COMPACT_JSON_NO_NAN",
            "sha256": {key: canonical_sha(value) for key, value in values.items()},
            "combined_inputs_sha256": canonical_sha(values),
            "reviewed_open_artifact_membership_attested": False,
            "native_inputs_attested": False,
            "scope": "Actual validated caller fixture values; source/review pins attest code, not caller data or model evidence."}


def _verified_review():
    review_bytes = REVIEW_PATH.read_bytes()
    if _sha_bytes(review_bytes) != REVIEW_SHA256:
        raise RuntimeError("independent review checksum differs; no source import")
    review = json.loads(review_bytes.decode("utf-8"))
    if review.get("status") != "PASS_V2_INSTRUMENT_AND_PRUNING_MECHANICS_ONLY":
        raise RuntimeError("independent review scope/status differs")
    buffers = {}
    for relative, expected in BOUND_SOURCES.items():
        source_bytes = (V2_ROOT / relative).read_bytes()
        if review.get("reviewed_source_sha256", {}).get(relative) != expected or _sha_bytes(source_bytes) != expected:
            raise RuntimeError("reviewed v2 source checksum differs: " + relative)
        buffers[relative] = source_bytes
    return review, buffers


@contextmanager
def _bound_v2():
    """Compile verified bytes in a fresh namespace, without writing pycache."""
    _, buffers = _verified_review()
    prefix = "_titan_claim_guard_v2_" + uuid.uuid4().hex
    created = []
    try:
        package = ModuleType(prefix)
        package.__package__ = prefix
        package.__path__ = [str(V2_ROOT / "titan_joint_quality_v1")]
        package.__file__ = str(V2_ROOT / "titan_joint_quality_v1/__init__.py")
        sys.modules[prefix] = package
        created.append(prefix)
        exec(compile(buffers["titan_joint_quality_v1/__init__.py"], package.__file__, "exec"), package.__dict__)
        modules = {}
        for name in ("common", "data", "analysis"):
            path = V2_ROOT / "titan_joint_quality_v1" / (name + ".py")
            module = ModuleType(prefix + "." + name)
            module.__package__, module.__file__ = prefix, str(path)
            sys.modules[module.__name__] = module
            created.append(module.__name__)
            exec(compile(buffers["titan_joint_quality_v1/" + name + ".py"], str(path), "exec"), module.__dict__)
            modules[name] = module
        yield modules
    finally:
        for name in reversed(created):
            sys.modules.pop(name, None)


def load_open_pilot_fixture():
    """Read only the independently bound, existing open v2 PILOT artifacts."""
    review, _ = _verified_review()
    names = ("pilot.prompts.jsonl", "pilot.references.jsonl")
    values = []
    recorded = {Path(path).resolve(): sha for path, sha in review["reviewed_artifact_sha256"].items()}
    for name in names:
        path = (V2_RUNS / "data-001" / name).resolve()
        artifact_bytes = path.read_bytes()
        if recorded.get(path) != _sha_bytes(artifact_bytes):
            raise RuntimeError("reviewed open PILOT artifact differs: " + name)
        values.append([json.loads(line) for line in artifact_bytes.decode("utf-8").splitlines()])
    return tuple(values)


def _dependency_depth(task):
    if task.get("kind") != "dependencies":
        raise ValueError("reasoning PILOT must use the registered dependency graph")
    leaves, expressions, query = task.get("leaves"), task.get("expressions"), task.get("query")
    if not isinstance(leaves, dict) or not isinstance(expressions, dict) or not expressions:
        raise ValueError("dependency leaves/expressions are required")
    if len(leaves) + len(expressions) > 64 or set(leaves) & set(expressions):
        raise ValueError("dependency namespace or fixture graph bound differs")
    if any(not isinstance(name, str) or not name for name in (*leaves, *expressions)):
        raise ValueError("dependency names must be nonempty strings")
    if any(type(value) is not int for value in leaves.values()):
        raise ValueError("dependency leaves must be exact integers")
    if query not in expressions:
        raise ValueError("query must name a public expression")
    state, depths = {}, {}
    def visit(name):
        if name in leaves: return 0
        if name not in expressions: raise ValueError("missing dependency node: " + str(name))
        if state.get(name) == "visiting": raise ValueError("dependency cycle detected")
        if name in depths: return depths[name]
        expression = expressions[name]
        if (not isinstance(expression, list) or len(expression) != 3 or
                expression[1] not in ("+", "-", "*") or
                not isinstance(expression[0], str) or not isinstance(expression[2], str)):
            raise ValueError("unsupported public expression")
        state[name] = "visiting"
        depths[name] = 1 + max(visit(expression[0]), visit(expression[2]))
        state[name] = "complete"
        return depths[name]
    # Reject a cycle even in an unqueried public expression.
    for name in expressions: visit(name)
    return depths[query]


def _bind_inputs(prompts, references, modules):
    analysis, data = modules["analysis"], modules["data"]
    refs = analysis.reference_map(references)
    if not isinstance(prompts, list) or len(prompts) != len(refs):
        raise ValueError("complete matching PILOT prompt IDs required")
    mapped = {}
    for row in prompts:
        if not isinstance(row, dict) or set(row) != {"case_id", "messages"}:
            raise ValueError("prompt projection must contain case_id/messages only")
        if row["case_id"] in mapped or row["case_id"] not in refs:
            raise ValueError("extra or duplicate PILOT prompt ID")
        mapped[row["case_id"]] = row["messages"]
    if set(mapped) != set(refs):
        raise ValueError("mismatched PILOT prompt/reference IDs")
    depths = {}
    for key, reference in refs.items():
        if reference["domain"] == "reasoning":
            depths[key] = _dependency_depth(data.parse_prompt(mapped[key]))
    merged = [{**reference, "messages": mapped[key]} for key, reference in refs.items()]
    data.validate_bank(merged, "pilot")
    return refs, depths


def _panel(ids, refs):
    ids = sorted(ids)
    parents = sorted({refs[key]["parent_id"] for key in ids})
    return {"case_count": len(ids), "parent_count": len(parents), "case_ids": ids, "parent_ids": parents}


def _accuracy(ids, scored):
    return sum(bool(scored[key]["semantic_correct"]) for key in ids) / len(ids) if ids else None


def _source_metric(ids, refs, scored):
    if not ids: return None
    parents = defaultdict(list)
    for key in ids: parents[refs[key]["parent_id"]].append(key)
    correct = [key for key in ids if scored[key]["semantic_correct"]]
    lost = [key for key in ids if not scored[key]["semantic_correct"]]
    complete = [parent for parent, keys in parents.items() if all(scored[key]["semantic_correct"] for key in keys)]
    return {"eligible_cases": len(ids), "eligible_parents": len(parents), "correct_cases": len(correct),
            "incorrect_case_ids": sorted(lost), "case_accuracy": len(correct) / len(ids),
            "all_eligible_cases_correct_parents": len(complete),
            "parent_scope": "all eligible source-correct cases within a parent; not unobserved tasks"}


def _deep_report(refs, depths, scores):
    strata = {}
    for depth in sorted(set(depths.values())):
        ids = [key for key, value in depths.items() if value == depth]
        strata[str(depth)] = {**_panel(ids, refs), "case_accuracy": {model: _accuracy(ids, scores[model]) for model in MODELS}}
    deep = [key for key, depth in depths.items() if depth >= 2]
    parents = defaultdict(list)
    for key in deep: parents[refs[key]["parent_id"]].append(key)
    parent_scores = {model: {parent: Fraction(sum(bool(scores[model][key]["semantic_correct"]) for key in ids), len(ids))
                            for parent, ids in parents.items()} for model in MODELS}
    means = {model: sum(values.values(), Fraction(0)) / len(parents) if parents else None
             for model, values in parent_scores.items()}
    comparisons = {}
    for baseline in ("sparse", "control"):
        changes = {parent: parent_scores["joint"][parent] - parent_scores[baseline][parent] for parent in parents}
        comparisons[baseline] = {"equal_weight_parent_delta": float(means["joint"] - means[baseline]) if parents else None,
                                 "gained_parent_ids": sorted(parent for parent, change in changes.items() if change > 0),
                                 "lost_parent_ids": sorted(parent for parent, change in changes.items() if change < 0),
                                 "unchanged_parent_ids": sorted(parent for parent, change in changes.items() if change == 0)}
    return {"definition": "actual queried dependency graph depth >=2; leaves depth0",
            "queried_depth_strata": strata, "deep_case_count": len(deep), "deep_parent_count": len(parents),
            "reasoning_parent_count": len({refs[key]["parent_id"] for key in depths}),
            "reasoning_family_count": len({refs[key]["family_id"] for key in depths}),
            "equal_weight_deep_parent_accuracy": {model: float(value) if value is not None else None for model, value in means.items()},
            "comparisons": comparisons,
            "positive_deep_parent_gain_vs_sparse": bool(parents) and means["joint"] > means["sparse"],
            "claim_scope": "overall reasoning-domain gain is not automatically a deep-reasoning or family-transfer gain"}


def inspect_fixture_claim(prompts, references, predictions, freeze, pair_proof, *, reasoning_shortcut_floor,
                          evidence_kind="fixture"):
    """Preserve exact legacy diagnostics and add explicit fixture-only guards.

    Required source panels are Sparse- and Supa-correct in each NON-R domain.
    Individual zero-deficit domains are NULL/NO_OBSERVED_DEFICIT; restoration also
    needs a nonempty overall NON-R Supa-correct/Sparse-wrong residual. Minimum
    nonemptiness proves no statistical adequacy. Deep positivity is a NEW claim
    rule; the legacy5pp/2pp gates are neither renamed nor altered.
    """
    if evidence_kind != "fixture":
        raise PermissionError("native/protected claim import is closed; fixture mechanics only")
    with _bound_v2() as modules:
        analysis = modules["analysis"]
        refs, depths = _bind_inputs(prompts, references, modules)
        legacy = analysis.analyze_fixture(references, predictions, freeze, pair_proof,
                                          reasoning_shortcut_floor=reasoning_shortcut_floor)
        input_bindings = _fixture_input_bindings(prompts, references, predictions, freeze,
                                                pair_proof, reasoning_shortcut_floor)
        scores = {model: analysis.score_predictions(references, predictions[model]) for model in MODELS}
        eligibility, retention, recovery = {}, {}, {}
        missing_panels, non_r_recovery = [], []
        for domain in NON_REASONING_DOMAINS:
            ids = [key for key, reference in refs.items() if reference["domain"] == domain]
            sparse = [key for key in ids if scores["sparse"][key]["semantic_correct"]]
            supa = [key for key in ids if scores["supa"][key]["semantic_correct"]]
            residual = sorted(set(supa) - set(sparse))
            non_r_recovery.extend(residual)
            eligibility[domain] = {"sparse_correct": _panel(sparse, refs), "supa_correct": _panel(supa, refs),
                                   "supa_correct_sparse_wrong": _panel(residual, refs)}
            for name, panel in (("sparse_correct", sparse), ("supa_correct", supa)):
                if not panel: missing_panels.append(domain + ":" + name)
            retention[domain] = {model: _source_metric(sparse, refs, scores[model]) for model in ("control", "joint")}
            recovery[domain] = {"status": "MEASURABLE_OBSERVED_DEFICIT" if residual else "NO_OBSERVED_DEFICIT",
                                "metrics": {model: _source_metric(residual, refs, scores[model]) for model in ("control", "joint")} if residual else None}
        source_panels_present = not missing_panels
        sparse_panels_present = not any(name.endswith(":sparse_correct") for name in missing_panels)
        supa_panels_present = not any(name.endswith(":supa_correct") for name in missing_panels)
        retention_measurable = sparse_panels_present
        recovery_measurable = supa_panels_present and bool(non_r_recovery)
        if not retention_measurable:
            retention = {domain: None for domain in NON_REASONING_DOMAINS}
        if not recovery_measurable and not supa_panels_present:
            recovery = {domain: {"status": "UNMEASURABLE_REQUIRED_SOURCE_PANEL_EMPTY", "metrics": None} for domain in NON_REASONING_DOMAINS}
        deep = _deep_report(refs, depths, scores)
        extra = {"required_non_R_source_panels_nonempty": source_panels_present,
                 "actual_non_R_recovery_measurable": recovery_measurable,
                 "NEW_positive_deep_parent_gain_vs_sparse": deep["positive_deep_parent_gain_vs_sparse"]}
        guarded_pass = all(legacy["gates"].values()) and all(extra.values())
        restored_all_non_r = recovery_measurable and all(scores["joint"][key]["semantic_correct"] for key in non_r_recovery)
        fixture_recovery_flag = bool(guarded_pass and legacy["release_full_recovery_satisfied_fixture_only"] and restored_all_non_r)
        return {"schema": "TITAN_ADDITIVE_FIXTURE_CLAIM_GUARDS_V1",
                "status": "GUARDED_FIXTURE_MECHANICS_PASS" if guarded_pass else "GUARDED_FIXTURE_CLAIMS_BLOCKED",
                "evidence_kind": "fixture", "legacy_diagnostics": legacy, "legacy_gates_unchanged": True,
                "additional_claim_rules": {"scope": "PROPOSED_ADDITIONAL_RULES_PENDING_OWNER_REGISTRATION_NOT_HISTORICAL_GATES",
                                           "owner_must_freeze_before_model_outcomes": True, "gates": extra,
                                           "no_new_deep_control_noninferiority_margin": True},
                "eligibility_by_non_R_domain": eligibility, "missing_required_source_panels": missing_panels,
                "retention_status": "MEASURABLE_MINIMUM_NON_VACUITY_ONLY" if retention_measurable else "UNMEASURABLE_REQUIRED_SOURCE_PANEL_EMPTY",
                "retention_metrics": retention if retention_measurable else None,
                "recovery_status": "MEASURABLE_OBSERVED_NON_R_DEFICIT" if recovery_measurable else "UNMEASURABLE_REQUIRED_SUPA_PANEL_EMPTY" if not supa_panels_present else "UNMEASURABLE_NO_OBSERVED_NON_R_DEFICIT",
                "recovery_metrics": recovery if recovery_measurable else None,
                "recovery_domain_status": {domain: value["status"] for domain, value in recovery.items()},
                "non_R_recovery_panel": _panel(non_r_recovery, refs), "deep_reasoning": deep,
                "guarded_fixture_quality_gate_pass": bool(guarded_pass),
                "full_recovery_claim_satisfied_fixture_only": fixture_recovery_flag,
                "release_permitted": False, "model_quality_established": False, "native_winner_declared": False,
                "native_authorized": False, "native_model_calls": 0, "training_authorized": False,
                "confirmation_authorized": False, "independent_review_sha256": REVIEW_SHA256,
                "bound_source_sha256": dict(BOUND_SOURCES),
                "caller_fixture_input_bindings": input_bindings,
                "limits": ["Fixture arithmetic/scoring only, never native model evidence.",
                           "Minimum nonempty panels establish no adequacy, power or population noninferiority.",
                           "Owner must prospectively review denominators, native adapter and claim scope.",
                           "One reasoning family cannot establish broad family transfer."]}
