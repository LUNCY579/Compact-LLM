"""Complete, identity-bound paired DEV accounting; no capability from fixtures."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from statistics import mean

from . import EXPERIMENT, SOURCE_CORE, SOURCE_WEIGHT_SHA256
from .contract import DEV_COUNTS
from .evaluation import score_prediction

IDENTITY_FIELDS = ("source_weight_sha256", "source_core", "compiled_plan_sha256", "code_sha256",
                   "runtime", "initial_frozen_state_sha256", "generation_sha256", "scorer_sha256",
                   "initial_parameters_sha256", "experiment", "checkout", "review_sha256")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _identity(identity):
    if not isinstance(identity, dict) or any(k not in identity for k in IDENTITY_FIELDS):
        raise ValueError("Complete source/compiled-plan/code/runtime/initial-state/generation/scorer identity required")
    if (identity["experiment"] != EXPERIMENT or identity["source_core"] != SOURCE_CORE
            or identity["source_weight_sha256"] != SOURCE_WEIGHT_SHA256):
        raise ValueError("Exact RQSFT ungated Supa source identity required")
    for key in IDENTITY_FIELDS:
        if key.endswith("sha256") and not SHA.fullmatch(str(identity[key])):
            raise ValueError("Invalid identity digest: " + key)
    if not isinstance(identity["runtime"], dict) or not identity["runtime"] or not identity["checkout"]:
        raise ValueError("Explicit runtime and checkout identity required")
    return {key: identity[key] for key in IDENTITY_FIELDS}


def _rows(rows, require_complete):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Explicit DEV rows required")
    keyed = {}
    for row in rows:
        if (row.get("split") != "dev" or row.get("category") not in DEV_COUNTS
                or not isinstance(row.get("id"), str) or not row["id"]
                or not isinstance(row.get("family"), str) or not row["family"]
                or not isinstance(row.get("parent_id"), str) or not row["parent_id"]):
            raise ValueError("Frozen DEV cohort/family/parent identity required")
        if row["id"] in keyed:
            raise ValueError("Duplicate DEV row")
        keyed[row["id"]] = row
    counts = Counter(row["category"] for row in rows)
    if require_complete and (len(rows) != 512 or dict(counts) != DEV_COUNTS):
        raise ValueError("Exactly 512 fixed DEV cases with all four 128-case cohorts required")
    return keyed


def _costs(costs):
    required = ("forward_calls", "processed_input_tokens", "generated_tokens", "wall_seconds")
    if not isinstance(costs, dict) or any(k not in costs for k in required):
        return False
    for key in required:
        value = costs[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            return False
        if key != "wall_seconds" and (type(value) is not int):
            return False
    return True


def _metrics(scored, rows, group_key="category"):
    grouped = defaultdict(list)
    for identifier, score in scored.items():
        row = rows[identifier]
        grouped[row["category"] if group_key == "category" else row["evaluation"]["dimension"]].append((row, score))
    result = {}
    for category, values in sorted(grouped.items()):
        known = [(row, score) for row, score in values if type(score["correct"]) is bool]
        family = defaultdict(list)
        for row, score in known:
            family[row["family"]].append(score["correct"])
        resolved = len(known) == len(values)
        result[category] = {"n": len(values), "correctness_resolved": len(known),
            "accuracy": mean(score["correct"] for _, score in known) if known and resolved else None,
            "macro_family_accuracy": mean(mean(v) for v in family.values()) if family and resolved else None,
            "family_accuracy": {name: mean(v) for name, v in sorted(family.items())} if resolved else None,
            "strict_format_failures": sum(score["strict_format"] is False for _, score in values),
            "literal_unknown_count": sum(score["literal_unknown"] is True for _, score in values),
            "semantic_abstention_count": (sum(score["semantic_abstention"] for _, score in values)
                if all(type(score["semantic_abstention"]) is bool for _, score in values) else None)}
    return result


def _eos_ids(eos_token_ids, eos_token_id):
    if eos_token_ids is None:
        eos_token_ids = [eos_token_id] if type(eos_token_id) is int else []
    if (not isinstance(eos_token_ids, (list, tuple)) or not eos_token_ids
            or any(type(token) is not int or token < 0 for token in eos_token_ids)
            or len(set(eos_token_ids)) != len(eos_token_ids)
            or eos_token_id is not None and eos_token_id not in eos_token_ids):
        raise ValueError("Exact full official generation EOS list required")
    return tuple(eos_token_ids)


def summarize_source(rows, records, *, identity, adjudications=None, costs=None, eos_token_ids=None, eos_token_id=None,
                     require_complete=True, fixture_only=False, input_hashes=None):
    """Describe all attempted cases; missing capture/reviews block advancement."""
    identity = _identity(identity)
    eos_ids = _eos_ids(eos_token_ids, eos_token_id)
    keyed = _rows(rows, require_complete)
    if require_complete and (not isinstance(input_hashes, dict) or set(input_hashes) != set(keyed)):
        raise ValueError("All512 compiled original-prefix hashes are required")
    adjudications = adjudications or {}
    captured, failures = {}, []
    for record in records:
        identifier = record.get("id")
        if identifier not in keyed or identifier in captured:
            raise ValueError("Unexpected/duplicate model case; fixed coverage cannot change")
        captured[identifier] = record
    missing = sorted(set(keyed) - set(captured))
    failures.extend({"id": i, "reason": "missing_capture"} for i in missing)
    scored = {}
    for identifier, record in captured.items():
        if not SHA.fullmatch(str(record.get("input_sha256", ""))):
            raise ValueError("Original native inference input hash required")
        if input_hashes is not None and record["input_sha256"] != input_hashes.get(identifier):
            raise ValueError("Inference input differs from the compiled original prompt")
        generated = record.get("generated_ids")
        valid_tokens = (isinstance(generated, list) and bool(generated) and len(generated) <= 64
                        and all(type(t) is int and t >= 0 for t in generated))
        if (type(record.get("context_tokens")) is not int or not 1 <= record["context_tokens"] <= 1984
                or type(record.get("generated_tokens")) is not int
                or not valid_tokens or record["generated_tokens"] != len(generated)):
            failures.append({"id": identifier, "reason": "invalid_complete_context_or_generation_accounting"})
        stop = record.get("stop_reason")
        if (not valid_tokens or stop not in {"eos", "length_limit"}
                or stop == "eos" and generated[-1] not in eos_ids
                or stop == "length_limit" and (len(generated) != 64 or any(token in eos_ids for token in generated))
                or stop == "eos" and any(token in eos_ids for token in generated[:-1])):
            failures.append({"id": identifier, "reason": "incomplete_or_inconsistent_termination"})
        score = score_prediction(keyed[identifier], record.get("prediction"), adjudications.get(identifier))
        scored[identifier] = score
        if score["status"] != "COMPLETE":
            failures.append({"id": identifier, "reason": "unresolved_adjudication_or_correctness",
                             "unresolved": score["unresolved"]})
        if record.get("error") is not None:
            failures.append({"id": identifier, "reason": "generation_failure", "error": record["error"]})
    if not _costs(costs):
        failures.append({"id": None, "reason": "missing_or_invalid_actual_costs"})
    elif (costs["generated_tokens"] != sum(len(r.get("generated_ids", [])) for r in captured.values()
                                           if isinstance(r.get("generated_ids"), list))
          or costs["forward_calls"] < len(captured)
          or costs["processed_input_tokens"] < len(captured)):
        failures.append({"id": None, "reason": "actual_costs_do_not_cover_captured_work"})
    return {"status": "INCOMPLETE" if failures else "COMPLETE", "identity": identity,
            "expected_rows": len(keyed), "captured_rows": len(captured), "missing_ids": missing,
            "scored": scored, "cohorts": _metrics(scored, keyed), "dimensions": _metrics(scored, keyed, "dimension"),
            "error_map_counts": dict(Counter(category for score in scored.values() for category in score["error_categories"])),
            "unassigned_diagnostics": sum(score.get("diagnostic_status") == "UNASSIGNED_NO_MECHANISM_CLAIM" and score["correct"] is not True for score in scored.values()),
            "failures": failures,
            "costs": costs, "eos_token_ids": list(eos_ids), "fixture_only": bool(fixture_only), "promotion_allowed": False,
            "capability_evidence": not fixture_only and not failures,
            "measurement_scope": "Fixed DEV only; no independent confirmation or release evidence"}


def _updates(envelope, source):
    updates = envelope.get("updates")
    if source:
        return type(updates) is int and updates == 0 or type(updates) is list and updates == []
    if not isinstance(updates, list) or not 1 <= len(updates) <= 32:
        return False
    for number, update in enumerate(updates, 1):
        if not isinstance(update, dict) or type(update.get("step")) is not int or update["step"] != number:
            return False
        for key in ("loss", "preclip_gradient_norm", "update_l2"):
            value = update.get(key)
            if type(value) not in {int, float} or not math.isfinite(value) or key != "loss" and value < 0:
                return False
    return True


def compare_quality(rows, baseline, candidate, *, expected_identity=None, baseline_adjudications=None,
                    candidate_adjudications=None, eos_token_ids=None, eos_token_id=None, input_hashes=None, require_complete=True):
    """Pair every source/final case, retaining damage and repair separately."""
    if baseline.get("role") != "source" or candidate.get("role") not in {"whole_1e-5", "response_1e-5"}:
        raise ValueError("Explicit source and fixed final SFT arm roles required")
    before_identity, after_identity = _identity(baseline.get("identity")), _identity(candidate.get("identity"))
    if before_identity != after_identity or expected_identity is not None and before_identity != _identity(expected_identity):
        raise ValueError("Source/final compiled-plan/code/runtime/initial-state binding differs")
    if type(baseline.get("fixture_only")) is not bool or type(candidate.get("fixture_only")) is not bool:
        raise ValueError("Explicit fixture/native evidence attribution required")
    if baseline["fixture_only"] != candidate["fixture_only"]:
        raise ValueError("Fixture and native results cannot be pooled")
    keyed = _rows(rows, require_complete)
    base_records = {r["id"]: r for r in baseline["records"]}
    final_records = {r["id"]: r for r in candidate["records"]}
    for identifier in set(base_records) & set(final_records):
        if base_records[identifier].get("input_sha256") != final_records[identifier].get("input_sha256"):
            raise ValueError("Paired source/final raw input hashes differ")
    common = dict(eos_token_ids=eos_token_ids, eos_token_id=eos_token_id, input_hashes=input_hashes, require_complete=require_complete)
    before = summarize_source(rows, baseline["records"], identity=before_identity,
        adjudications=baseline_adjudications, costs=baseline.get("costs"), fixture_only=baseline["fixture_only"], **common)
    after = summarize_source(rows, candidate["records"], identity=after_identity,
        adjudications=candidate_adjudications, costs=candidate.get("costs"), fixture_only=candidate["fixture_only"], **common)
    complete = before["status"] == after["status"] == "COMPLETE" and _updates(baseline, True) and _updates(candidate, False)
    transitions, gains = {}, {}
    for category in DEV_COUNTS:
        identifiers = [i for i, row in keyed.items() if row["category"] == category]
        pairs = [(before["scored"].get(i, {}).get("correct"), after["scored"].get(i, {}).get("correct")) for i in identifiers]
        resolved = bool(pairs) and all(type(a) is bool and type(b) is bool for a, b in pairs)
        if not resolved:
            transitions[category] = {"n": len(pairs), "source_correct": None, "retained": None,
                                     "damages": None, "repairs": None, "source_retained_fraction": None}
            gains[category] = None
            continue
        source_correct = sum(a for a, _ in pairs)
        retained = sum(a and b for a, b in pairs)
        transitions[category] = {"n": len(pairs), "source_correct": source_correct, "retained": retained,
            "damages": sum(a and not b for a, b in pairs), "repairs": sum(not a and b for a, b in pairs),
            "source_retained_fraction": retained / source_correct if source_correct else None}
        gains[category] = 100 * (after["cohorts"][category]["macro_family_accuracy"] - before["cohorts"][category]["macro_family_accuracy"])
    communication = [i for i, row in keyed.items() if row["evaluation"]["dimension"] == "communication"]
    rating_deltas = []
    for identifier in communication:
        a = before["scored"].get(identifier, {}).get("communication_rating")
        b = after["scored"].get(identifier, {}).get("communication_rating")
        if a is None or b is None:
            complete = False
        else:
            rating_deltas.append(b - a)
    communication_delta = mean(rating_deltas) if communication and len(rating_deltas) == len(communication) else None
    communication_families = defaultdict(list)
    for identifier, delta in zip(communication, rating_deltas):
        communication_families[keyed[identifier]["family"]].append(delta)
    if communication_delta is not None:
        communication_delta = mean(mean(values) for values in communication_families.values())
    critical_correctness_losses = [i for i, row in keyed.items() if row["evaluation"]["critical"]
        and row["evaluation"]["dimension"] == "grounding_answerability"
        and before["scored"].get(i, {}).get("correct") is True
        and after["scored"].get(i, {}).get("correct") is False]
    critical_safe_abstention_losses = [i for i, row in keyed.items() if row["evaluation"]["critical"]
        and row["evaluation"]["dimension"] == "grounding_answerability"
        and row["evaluation"]["answerable"] is False
        and before["scored"].get(i, {}).get("semantic_abstention") is True
        and after["scored"].get(i, {}).get("semantic_abstention") is False]
    critical_losses = sorted(set(critical_correctness_losses) | set(critical_safe_abstention_losses))
    preservation = transitions["preservation_dialogue"]
    dimensions = {row["evaluation"]["dimension"] for row in rows}
    critical_grounding = [row for row in rows if row["evaluation"]["critical"] and row["evaluation"]["dimension"] == "grounding_answerability"]
    evidence_adequate = (bool(communication) and bool(critical_grounding)
                        and len(dimensions) == 5 and preservation["source_retained_fraction"] is not None)
    gates = {"acquisition_at_least_5pp": gains["acquisition"] is not None and gains["acquisition"] >= 5.,
        "harder_transfer_positive": gains["harder_transfer"] is not None and gains["harder_transfer"] > 0.,
        "zero_new_critical_grounding_losses": not critical_losses,
        "source_success_retention_at_least_98pct": preservation["source_retained_fraction"] is not None and preservation["source_retained_fraction"] >= .98,
        "macro_preservation_at_least_minus2pp": gains["preservation_dialogue"] is not None and gains["preservation_dialogue"] >= -2.,
        "communication_ratings_nondecline": communication_delta is not None and communication_delta >= 0.,
        "complete_coverage_adjudication_costs_finite_updates": complete}
    status = "INCOMPLETE" if not complete else "INCONCLUSIVE" if not evidence_adequate else "DEV_CONDITIONS_MET" if all(gates.values()) else "DEV_CONDITIONS_NOT_MET"
    return {"status": status, "arm": candidate["role"], "gates": gates, "macro_family_gain_pp": gains,
        "transitions": transitions, "new_critical_grounding_loss_ids": critical_losses,
        "new_critical_answer_correctness_loss_ids": sorted(critical_correctness_losses),
        "new_critical_safe_abstention_loss_ids": sorted(critical_safe_abstention_losses),
        "overall_source_correct": (sum(t["source_correct"] for t in transitions.values())
            if all(t["source_correct"] is not None for t in transitions.values()) else None),
        "overall_damages": (sum(t["damages"] for t in transitions.values())
            if all(t["damages"] is not None for t in transitions.values()) else None),
        "overall_repairs": (sum(t["repairs"] for t in transitions.values())
            if all(t["repairs"] is not None for t in transitions.values()) else None),
        "communication_delta": communication_delta, "source_summary": before, "candidate_summary": after,
        "fixture_only": baseline["fixture_only"], "advancement_conditions_met": status == "DEV_CONDITIONS_MET",
        "native_advancement_conditions_met": status == "DEV_CONDITIONS_MET" and not baseline["fixture_only"],
        "promotion_allowed": False, "causal_retention_claim": False,
        "cumulative_update_l2": sum(u["update_l2"] for u in candidate["updates"]) if _updates(candidate, False) else None,
        "interpretation": "Acquisition, transfer, critical damages, preservation, communication and costs all remain separate. A smaller update alone is not success; fixed DEV is not confirmation."}
