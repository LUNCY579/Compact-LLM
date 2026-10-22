"""Bounded abstract mechanism witnesses, never a bank generator/admission gate.

Only modular affine mathematics is computed here. No file, model, tokenizer,
source bank, native permit or Rineya result is read or created. External metadata
references are declarations, not authenticated evidence. Existing bank quotas,
scientific gates and dataset-owner authority are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re

MAX_OPERATIONS = 16
MAX_METADATA_WITNESSES = 32
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _inputs(initial, operations, modulus):
    if type(initial) is not int or type(modulus) is not int or modulus < 2:
        raise ValueError("Exact integer initial state and modulus>=2 required; bool is not integer evidence")
    if not isinstance(operations, (list, tuple)) or not 1 <= len(operations) <= MAX_OPERATIONS:
        raise ValueError("One through sixteen explicit affine operations required")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2
           or any(type(value) is not int for value in pair) for pair in operations):
        raise ValueError("Every operation must be an exact integer (multiplier,offset) pair")
    return tuple(tuple(pair) for pair in operations)


def _sequential(initial, operations, modulus):
    state = initial % modulus
    for multiplier, offset in operations:
        state = (multiplier * state + offset) % modulus
    return state


def _composed(initial, operations, modulus):
    # Independent algorithm: compose coefficients before applying the input.
    multiplier, offset = 1, 0
    for event_multiplier, event_offset in operations:
        multiplier = event_multiplier * multiplier % modulus
        offset = (event_multiplier * offset + event_offset) % modulus
    return (multiplier * initial + offset) % modulus


def affine_final(initial, operations, modulus):
    """Compute the final queried residue; compare two independent algorithms."""
    operations = _inputs(initial, operations, modulus)
    sequential = _sequential(initial, operations, modulus)
    composed = _composed(initial, operations, modulus)
    if sequential != composed:
        raise ArithmeticError("Independent sequential/composed oracle disagreement")
    return sequential


def _commutator(first, second, modulus):
    af, bf = first
    ag, bg = second
    # Original g(f(s)) minus swapped f(g(s)), independent of pre-swap state.
    return ((ag - 1) * bf - (af - 1) * bg) % modulus


def affine_order_witness(initial, operations, modulus, queryfinal):
    """Test bounded valid adjacent swaps against independently supplied gold.

    A local nonzero commutator does not suffice: the complete downstream suffix
    may erase it. Only different FULL queried endpoints witness order sensitivity.
    No witness in the tested adjacent set does not prove all permutations commute.
    Pairwise-zero commutators, when observed for every pair, provide the separate
    algebraic proof that every permutation of these affine maps is invariant.
    Conditional/state-dependent rules are outside this affine-only certificate.
    """
    operations = _inputs(initial, operations, modulus)
    if type(queryfinal) is not int or not 0 <= queryfinal < modulus:
        raise ValueError("Explicit independent gold must be an exact canonical final residue")
    baseline = affine_final(initial, operations, modulus)
    noncommuting = sum(_commutator(operations[i], operations[j], modulus) != 0
                      for i in range(len(operations)) for j in range(i + 1, len(operations)))
    swaps = []
    for index in range(len(operations) - 1):
        if operations[index] == operations[index + 1]:
            continue  # Exchanging identical events is not a nontrivial mutation.
        variant = list(operations)
        variant[index], variant[index + 1] = variant[index + 1], variant[index]
        sequential = _sequential(initial, variant, modulus)
        composed = _composed(initial, variant, modulus)
        if sequential != composed:
            raise ArithmeticError("Independent variant oracle disagreement")
        local_difference = _commutator(operations[index], operations[index + 1], modulus)
        suffix_multiplier = 1
        for multiplier, _ in operations[index + 2:]:
            suffix_multiplier = suffix_multiplier * multiplier % modulus
        predicted_difference = local_difference * suffix_multiplier % modulus
        actual_difference = (baseline - sequential) % modulus
        if actual_difference != predicted_difference:
            raise ArithmeticError("Full endpoint difference disagrees with suffix commutator proof")
        swaps.append({"adjacent_indices": [index, index + 1], "valid_original": True,
            "valid_variant": True, "nontrivial_permutation": True,
            "original_query_final": baseline, "variant_query_final": sequential,
            "sequential_variant_final": sequential, "composed_variant_final": composed,
            "local_commutator_modulus": local_difference, "suffix_multiplier_modulus": suffix_multiplier,
            "predicted_full_query_difference_modulus": predicted_difference,
            "actual_full_query_difference_modulus": actual_difference,
            "changed_query": baseline != sequential,
            "suffix_erased_local_difference": local_difference != 0 and actual_difference == 0})
    changed = sum(swap["changed_query"] for swap in swaps)
    gold_agrees = baseline == queryfinal
    if not gold_agrees:
        status = "INVALID_INDEPENDENT_GOLD"
    elif len(operations) < 2:
        status = "UNVERIFIED_INSUFFICIENT_EVENTS"
    elif changed:
        status = "ORDER_SENSITIVE_WITNESS_FOUND"
    elif noncommuting == 0:
        status = "ORDER_INSENSITIVE_PAIRWISE_COMMUTATIVE"
    else:
        status = "UNVERIFIED_NO_TESTED_ADJACENT_QUERY_CHANGE"
    mathematical_input = {"initial": initial, "operations": [list(pair) for pair in operations], "modulus": modulus}
    return {"schema": 1, "evidence_scope": "MATHEMATICAL_FIXTURE", "rineya_evidence": "NO_RINEYA_GAIN",
        "status": status, "input": mathematical_input, "input_sha256": _digest(mathematical_input),
        "independent_query_gold": queryfinal, "original_query_final": baseline,
        "sequential_final": baseline, "composed_final": _composed(initial, operations, modulus),
        "independent_gold_agrees": gold_agrees, "independent_algorithms_agree": True,
        "proof_type": "SEQUENTIAL_AND_COMPOSED_FULL_ENDPOINTS_WITH_SUFFIX_COMMUTATOR_CROSSCHECK",
        "operation_count": len(operations), "adjacent_position_count": max(0, len(operations) - 1),
        "valid_nontrivial_adjacent_permutations": len(swaps), "changed_query_count": changed,
        "unchanged_query_count": len(swaps) - changed, "noncommuting_pair_count": noncommuting,
        "all_pairs_commute": noncommuting == 0,
        "mathematical_order_witness": gold_agrees and changed > 0, "swaps": swaps,
        "admission_granted": False, "native_permission": False, "model_calls": 0,
        "limitations": ["Abstract mathematics only; no real bank or Rineya capability measured.",
            "Tested mutations are original adjacent swaps; absence of their query changes is not general order invariance unless all pair commutators vanish.",
            "Conditional and non-affine event semantics are not implemented.",
            "Rendered prompt/query correspondence and actual owner evidence custody require independent review."]}


def _reference(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 2048


def _sha(value):
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def _provenance(metadata):
    reasons = []
    if not _sha(metadata.get("subject_sha256")):
        reasons.append("missing_bound_subject_sha256")
    rules = metadata.get("public_rules", {})
    if (not isinstance(rules, dict) or not _reference(rules.get("reference"))
            or not _sha(rules.get("sha256"))):
        reasons.append("missing_public_rules_reference_and_hash")
    state = metadata.get("state_evidence", {})
    if (not isinstance(state, dict) or not _reference(state.get("reference"))
            or not _sha(state.get("sha256")) or state.get("actual_reference_state") is not True):
        reasons.append("missing_actual_reference_state_evidence")
    solvers = metadata.get("independent_solvers", [])
    if (not isinstance(solvers, list) or not 2 <= len(solvers) <= 8
            or any(not isinstance(solver, dict) or not _reference(solver.get("id"))
                   or not _reference(solver.get("public_reference")) or not _sha(solver.get("implementation_sha256"))
                   or solver.get("independent") is not True for solver in solvers)):
        reasons.append("missing_two_independent_public_solver_provenance_records")
    elif (len({solver["id"] for solver in solvers}) != len(solvers)
          or len({solver["implementation_sha256"] for solver in solvers}) != len(solvers)):
        reasons.append("repeated_solver_identity_or_implementation")
    return reasons


def validate_mechanism_attestation(metadata):
    """Advisory descriptor check; complete declarations still need owner receipts.

    Metadata IDs/hashes cannot prove correspondence to an actual bank. This
    function never authenticates references, dereferences a state, reads case
    bodies, grants admission, changes quotas, or changes native qualification.
    An inline abstract affine certificate can expose a mathematical contradiction
    or witness, but cannot establish the actual owner's empirical assay.
    """
    report = {"status": "UNVERIFIED", "advisory_only": True, "admission_granted": False,
        "native_permission": False, "actual_owner_bank_verified": False, "owner_receipt_required": True,
        "evidence_scope": "MATHEMATICAL_FIXTURE", "rineya_evidence": "NO_RINEYA_GAIN", "model_calls": 0,
        "quotas_or_native_gates_changed": False, "descriptor_complete": False,
        "mathematical_claim_supported": None, "reasons": []}
    if not isinstance(metadata, dict):
        report["reasons"] = ["missing_typed_mechanism_descriptor"]
        return report
    claim = metadata.get("claim")
    if claim not in {"ordered_reasoning", "entity_binding"}:
        report["reasons"] = ["missing_or_unsupported_declared_mechanism_claim"]
        return report
    report["claim"] = claim
    reasons = _provenance(metadata)
    witnesses = metadata.get("witnesses", [])
    if not isinstance(witnesses, list) or not 1 <= len(witnesses) <= MAX_METADATA_WITNESSES:
        reasons.append("missing_bounded_actual_oracle_witness_records")
        witnesses = []
    for witness in witnesses:
        if (not isinstance(witness, dict) or not _reference(witness.get("public_evidence_reference"))
                or not _sha(witness.get("evidence_sha256")) or witness.get("independent_oracle_recomputed") is not True
                or witness.get("valid_original") is not True or witness.get("valid_variant") is not True
                or not _sha(witness.get("original_query_gold_sha256"))
                or not _sha(witness.get("variant_query_gold_sha256"))):
            reasons.append("missing_typed_bound_independent_oracle_witness")
            continue
        if claim == "ordered_reasoning":
            if (witness.get("kind") != "valid_nontrivial_order_permutation"
                    or witness.get("nontrivial_permutation") is not True or witness.get("changed_query") is not True
                    or witness["original_query_gold_sha256"] == witness["variant_query_gold_sha256"]):
                reasons.append("ordered_claim_lacks_changed_valid_nontrivial_queried_gold")
        elif (witness.get("kind") not in {"bijective_entity_renaming", "relevant_binding_counterfactual",
                                         "irrelevant_entity_invariance", "necessary_binding_evidence_removal"}
              or not _reference(witness.get("original_state_reference"))
              or not _reference(witness.get("variant_state_reference"))
              or not _sha(witness.get("original_state_sha256")) or not _sha(witness.get("variant_state_sha256"))):
            reasons.append("entity_binding_requires_actual_state_counterfactual_evidence_not_ids")
        elif (witness["kind"] == "bijective_entity_renaming" and witness.get("gold_equivariant_after_inverse_rename") is not True
              or witness["kind"] == "relevant_binding_counterfactual" and
              (witness.get("changed_query") is not True or witness["original_query_gold_sha256"] == witness["variant_query_gold_sha256"])
              or witness["kind"] == "irrelevant_entity_invariance" and
              (witness.get("gold_unchanged") is not True or witness["original_query_gold_sha256"] != witness["variant_query_gold_sha256"])
              or witness["kind"] == "necessary_binding_evidence_removal" and witness.get("answerability_changed") is not True):
            reasons.append("entity_binding_gold_relation_not_declared_or_contradicted")
    if claim == "entity_binding" and not {"bijective_entity_renaming", "relevant_binding_counterfactual"}.issubset(
            {w.get("kind") for w in witnesses if isinstance(w, dict)}):
        reasons.append("both_renaming_and_relevant_binding_counterfactual_are_required")
    certificate = metadata.get("mathematical_affine_witness")
    if certificate is not None:
        try:
            if not isinstance(certificate, dict) or certificate.get("semantics") != "modular_affine_final_query":
                raise ValueError("Only explicitly declared affine final-query certificates are implemented")
            mathematical = affine_order_witness(certificate["initial"], certificate["operations"],
                                                certificate["modulus"], certificate["queryfinal"])
            report["mathematical_witness"] = mathematical
            if claim == "ordered_reasoning":
                report["mathematical_claim_supported"] = mathematical["mathematical_order_witness"]
            else:
                reasons.append("affine_order_witness_does_not_certify_entity_binding")
            if not mathematical["independent_gold_agrees"]:
                reasons.append("mathematical_independent_query_gold_mismatch")
            elif claim == "ordered_reasoning" and mathematical["all_pairs_commute"]:
                reasons.append("ordered_claim_contradicted_by_pairwise_commutative_maps")
            elif claim == "ordered_reasoning" and not mathematical["mathematical_order_witness"]:
                reasons.append("no_full_queried_endpoint_change_in_tested_permutations")
            if metadata.get("mathematical_input_sha256") != mathematical["input_sha256"]:
                reasons.append("mathematical_witness_input_binding_missing_or_changed")
        except (KeyError, TypeError, ValueError) as error:
            reasons.append("invalid_or_unsupported_mathematical_certificate:" + str(error))
    report["descriptor_complete"] = not reasons
    report["reasons"] = sorted(set(reasons)) + ["actual_independent_owner_receipts_and_public_evidence_replay_still_required"]
    return report
