"""Strict open-panel scoring and paired gates. Fixtures cannot become capability evidence."""
from __future__ import annotations
from collections import defaultdict
import json
import re
from .common import CONTRACT, DOMAINS, PARENT_SHA, SUPA_SHA, digest

MODELS = ("supa", "sparse", "transition", "control", "joint")
TRANSITION_SHA = "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"

def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out

def parse_answer(text):
    """Only complete clean wrappers; never keyword/number search inside prose."""
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    s = text.strip()
    if re.fullmatch(r"[A-Za-z0-9_+|,:/\-]+\.?", s):
        return s.removesuffix("."), "plain"
    if s.startswith("{"):
        try:
            obj = json.loads(s, object_pairs_hook=_unique_object)
        except ValueError:
            return None, None
        if isinstance(obj, dict) and set(obj) == {"answer"}:
            v = obj["answer"]
            if type(v) is int or (isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_+|,:/\-]+", v)):
                return str(v), "json"
    match = re.fullmatch(r"<answer>\s*([A-Za-z0-9_+|,:/\-]+)\s*</answer>", s)
    if match:
        return match.group(1), "tag"
    return None, None

def score_response(response, reference):
    answer, wrapper = parse_answer(response)
    gold = reference["canonical_answer"]
    if not isinstance(gold, str) or not re.fullmatch(r"[A-Za-z0-9_+|,:/\-]+", gold):
        raise ValueError("unsupported canonical answer contract")
    abstained = answer == "UNKNOWN"
    malformed = answer is None
    semantic_correct = answer == gold
    # Numeric truth is explicit for the arithmetic/reasoning domains; symbolic
    # state, retrieval and formatting answers remain exact strings.
    if reference["domain"] in ("reasoning", "arithmetic") and re.fullmatch(r"[+-]?\d+", gold):
        semantic_correct = answer is not None and re.fullmatch(r"[+-]?\d+", answer) is not None and int(answer) == int(gold)
    format_valid = wrapper == reference["requested_format"] and not malformed
    if wrapper == "plain" and response.strip().endswith("."):
        format_valid = False
    if wrapper == "json" and not isinstance(json.loads(response, object_pairs_hook=_unique_object)["answer"], str):
        format_valid = False
    return {"semantic_correct": semantic_correct, "strict_correct": response.strip() == reference["target"],
        "format_valid": format_valid,
        "literal_UNKNOWN": response.strip() == "UNKNOWN", "semantic_abstention": abstained,
        "false_abstention": gold != "UNKNOWN" and abstained,
        "unsupported_answer": gold == "UNKNOWN" and answer not in (None, "UNKNOWN"),
        "ambiguous_response": malformed, "requires_blind_review": malformed}

def reference_map(references):
    if not isinstance(references, list) or len(references) != 384:
        raise ValueError("the complete 384-row open pilot is required")
    mapped = {}
    parents = defaultdict(list)
    for r in references:
        key = r["case_id"]
        if key in mapped or r.get("split") not in (None, "pilot"):
            raise ValueError("duplicate/nonpilot reference")
        if r["domain"] not in DOMAINS:
            raise ValueError("unregistered domain")
        mapped[key] = r
        parents[r["parent_id"]].append(r)
    if any(sum(r["domain"] == d for r in references) != 64 for d in DOMAINS):
        raise ValueError("domain counts differ")
    if len(parents) != 192 or any(len(v) != 2 or {r["variant_id"] for r in v} != {0, 1} for v in parents.values()):
        raise ValueError("paired parent geometry differs")
    return mapped

def score_predictions(references, predictions):
    refs = reference_map(references)
    if not isinstance(predictions, list):
        raise ValueError("complete prediction rows required")
    scores = {}
    for p in predictions:
        key = p.get("case_id")
        if key not in refs or key in scores:
            raise ValueError("extra or duplicate prediction row")
        if p.get("completed") is not True or p.get("truncated") is not False:
            raise ValueError("failed/truncated generation; no survivor comparison")
        scores[key] = score_response(p["response"], refs[key])
    if set(scores) != set(refs):
        raise ValueError("missing prediction rows; no survivor comparison")
    return scores

def freeze_baselines(references, predictions, source_bindings, *, evidence_kind):
    if evidence_kind != "fixture":
        raise PermissionError("native baseline freezing requires the existing owner's reviewed native adapter")
    refs = reference_map(references)
    sources = {"supa": SUPA_SHA, "sparse": PARENT_SHA, "transition": TRANSITION_SHA}
    if source_bindings != sources or set(predictions) != set(sources):
        raise ValueError("baseline source identity/coverage differs")
    scored = {m: score_predictions(references, predictions[m]) for m in sources}
    sparse_correct = sorted(k for k in refs if scored["sparse"][k]["semantic_correct"])
    supa_correct = sorted(k for k in refs if scored["supa"][k]["semantic_correct"])
    recovery = sorted(set(supa_correct) - set(sparse_correct))
    value = {"evidence_kind": "fixture", "native_qualified": False, "reference_digest": digest(references),
        "source_bindings": sources, "sparse_correct_cases": sparse_correct,
        "supa_correct_cases": supa_correct, "supa_recovery_cases": recovery,
        "baseline_prediction_digests": {m: digest(v) for m, v in predictions.items()},
        "frozen_before_candidate_predictions": True, "model_quality_evidence": False}
    return {**value, "freeze_digest": digest(value)}

def validate_pair_proof(proof):
    errors = []
    if proof.get("evidence_kind") != "fixture":
        raise PermissionError("native result import is closed until owner integration/result review")
    c, j = proof.get("control", {}), proof.get("joint", {})
    fields = ("parent_sha256", "selected_case_ids", "sequence_digest", "reasoning_labels_digest",
              "groups", "planned_denominators", "processed_tokens", "updates", "plan_sha256")
    for f in fields:
        if c.get(f) is None or j.get(f) is None or c[f] != j[f]:
            errors.append("matched pair field missing/different:" + f)
    for name, arm in (("control", c), ("joint", j)):
        if arm.get("parent_sha256") != PARENT_SHA or arm.get("status") != "COMPLETE":
            errors.append("arm parent/completeness differs:" + name)
        if type(arm.get("processed_tokens")) is not int or not 0 < arm.get("processed_tokens", 0) <= 54172:
            errors.append("arm processed budget differs:" + name)
        ids, groups, denoms = arm.get("selected_case_ids"), arm.get("groups"), arm.get("planned_denominators")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            errors.append("arm selected rows absent/duplicate:" + name)
        if not isinstance(groups, list) or not groups or type(arm.get("updates")) is not int or arm.get("updates") != len(groups):
            errors.append("arm update geometry differs:" + name)
        if not isinstance(denoms, list) or not isinstance(groups, list) or len(denoms) != len(groups) or any(type(v) is not int or v <= 0 for v in denoms):
            errors.append("arm common denominators invalid:" + name)
        if arm.get("frozen_parameters_unchanged") is not True or arm.get("gradients_cleared") is not True:
            errors.append("arm frozen/gradient proof absent:" + name)
        if not isinstance(arm.get("final_checkpoint_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", arm.get("final_checkpoint_sha256", "")):
            errors.append("one final checkpoint binding absent:" + name)
    return errors

def analyze_fixture(references, predictions, freeze, pair_proof, *, reasoning_shortcut_floor):
    """All five complete artifacts required, even when illustrating negative gates."""
    refs = reference_map(references)
    errors = validate_pair_proof(pair_proof)
    if set(predictions) != set(MODELS):
        raise ValueError("all three baselines and two arms required")
    scores = {m: score_predictions(references, predictions[m]) for m in MODELS}
    original = {k: v for k, v in freeze.items() if k != "freeze_digest"}
    if freeze.get("evidence_kind") != "fixture" or freeze.get("freeze_digest") != digest(original):
        errors.append("baseline freeze digest/kind differs")
    if freeze.get("reference_digest") != digest(references):
        errors.append("baseline reference set differs")
    if freeze.get("source_bindings") != {"supa": SUPA_SHA, "sparse": PARENT_SHA, "transition": TRANSITION_SHA}:
        errors.append("frozen source identity differs")
    if freeze.get("frozen_before_candidate_predictions") is not True:
        errors.append("baseline chronology proof absent")
    for model in ("supa", "sparse", "transition"):
        if freeze.get("baseline_prediction_digests", {}).get(model) != digest(predictions[model]):
            errors.append("baseline predictions changed after freeze:" + model)
    if type(reasoning_shortcut_floor) not in (int, float) or not 0 <= reasoning_shortcut_floor <= 1:
        raise ValueError("frozen shortcut floor required")
    reasoning = [k for k, r in refs.items() if r["domain"] == "reasoning"]
    accuracy = {m: sum(scores[m][k]["semantic_correct"] for k in reasoning) / len(reasoning) for m in MODELS}
    preservation = freeze.get("sparse_correct_cases", [])
    recovery = freeze.get("supa_recovery_cases", [])
    if any(k not in refs for k in preservation + recovery):
        raise ValueError("unknown frozen source-correct case")
    actual_preserved = sorted(k for k in refs if scores["sparse"][k]["semantic_correct"])
    actual_supa_correct = sorted(k for k in refs if scores["supa"][k]["semantic_correct"])
    actual_recovery = sorted(k for k in refs if scores["supa"][k]["semantic_correct"] and not scores["sparse"][k]["semantic_correct"])
    if preservation != actual_preserved or recovery != actual_recovery or freeze.get("supa_correct_cases") != actual_supa_correct:
        errors.append("frozen source-correct/recovery membership differs")
    lost = {m: [k for k in preservation if not scores[m][k]["semantic_correct"]] for m in ("control", "joint")}
    restored = {m: [k for k in recovery if scores[m][k]["semantic_correct"]] for m in ("control", "joint")}
    source_union = sorted(set(preservation + freeze.get("supa_correct_cases", [])))
    better_parents = sorted({refs[k]["parent_id"] for k in source_union
        if scores["joint"][k]["semantic_correct"] and not scores["control"][k]["semantic_correct"]})
    source_harms_vs_control = [k for k in source_union if scores["control"][k]["semantic_correct"] and not scores["joint"][k]["semantic_correct"]]
    harms = {}
    for field in ("unsupported_answer", "false_abstention"):
        harms[field] = [k for k in refs if scores["joint"][k][field] and not scores["sparse"][k][field]]
    harms["format_failure"] = [k for k in refs if scores["sparse"][k]["format_valid"] and not scores["joint"][k]["format_valid"]]
    blind_review = [k for k in refs if any(scores[m][k]["requires_blind_review"] for m in MODELS)]
    gates = {"complete_matched_pair": not errors,
        "reasoning_gain_at_least_5pp": accuracy["joint"] - accuracy["sparse"] >= .05 - 1e-12,
        "reasoning_above_frozen_floor": accuracy["joint"] > reasoning_shortcut_floor,
        "control_deficit_at_most_2pp": accuracy["joint"] >= accuracy["control"] - .02 - 1e-12,
        "zero_new_sparse_correct_losses": len(lost["joint"]) == 0,
        "better_source_parent_than_control": bool(better_parents) and not source_harms_vs_control,
        "no_new_unsupported_or_false_abstention_or_format_failures": not any(harms.values()),
        "grounding_blind_review_complete": not blind_review}
    return {"status": "FIXTURE_GATES_PASS_NOT_MODEL_EVIDENCE" if all(gates.values()) else "FIXTURE_GATES_FAIL_NOT_MODEL_EVIDENCE",
        "evidence_kind": "fixture", "gates": gates, "errors": errors, "reasoning_accuracy": accuracy,
        "reasoning_shortcut_floor": reasoning_shortcut_floor, "correct_case_losses": lost,
        "recovered_supa_correct_cases": restored, "better_source_parents_than_control": better_parents,
        "source_harms_vs_control": source_harms_vs_control, "paired_behavior_harms": harms,
        "blind_review_required_cases": blind_review, "parent_units": 192, "case_count": 384,
        "supa_recovery_endpoint_measurable": bool(recovery),
        "release_full_recovery_satisfied_fixture_only": bool(recovery) and len(restored["joint"]) == len(recovery),
        "advance_to_replication": False, "release_permitted": False, "model_quality_established": False,
        "limits": ["Fixture gate behavior only; zero native model calls.",
                   "No population noninferiority from zero observed losses.",
                   "Ambiguous grounding outputs require blind semantic review."]}
