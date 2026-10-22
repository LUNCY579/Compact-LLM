"""Tiny abstract mathematics only; no actual bank, Rineya or admission evidence."""
import copy

import pytest

from experiments.rineya_quality_sft_v1.mechanism_witness import (
    affine_final, affine_order_witness, validate_mechanism_attestation,
)


def test_simple_additions_commute_and_cannot_license_ordered_reasoning():
    operations = [(1, 3), (1, -2), (1, 5)]
    report = affine_order_witness(4, operations, 17, 10)
    assert affine_final(4, operations, 17) == 10
    assert report["status"] == "ORDER_INSENSITIVE_PAIRWISE_COMMUTATIVE"
    assert report["valid_nontrivial_adjacent_permutations"] == 2
    assert report["changed_query_count"] == 0 and report["unchanged_query_count"] == 2
    assert not report["mathematical_order_witness"]


def test_noncommuting_affine_events_change_the_complete_query_endpoint():
    report = affine_order_witness(1, [(1, 1), (2, 0)], 17, 4)
    assert report["status"] == "ORDER_SENSITIVE_WITNESS_FOUND"
    assert report["sequential_final"] == report["composed_final"] == 4
    assert report["swaps"][0]["variant_query_final"] == 3
    assert report["swaps"][0]["predicted_full_query_difference_modulus"] == 1
    assert report["changed_query_count"] == 1
    assert not report["admission_granted"] and not report["native_permission"]
    assert report["rineya_evidence"] == "NO_RINEYA_GAIN"


def test_multiplier_events_can_commute_too():
    report = affine_order_witness(4, [(2, 1), (3, 2)], 17, 12)
    assert report["status"] == "ORDER_INSENSITIVE_PAIRWISE_COMMUTATIVE"
    assert report["swaps"][0]["local_commutator_modulus"] == 0


@pytest.mark.parametrize("operations,modulus,initial,gold", [
    ([(1, 1), (2, 0), (0, 7)], 17, 1, 7),
    ([(1, 4), (2, 0), (2, 0)], 8, 1, 4),
])
def test_downstream_noninvertible_suffix_can_erase_a_local_difference(operations, modulus, initial, gold):
    report = affine_order_witness(initial, operations, modulus, gold)
    first = report["swaps"][0]
    assert first["local_commutator_modulus"] != 0
    assert first["suffix_erased_local_difference"] and not first["changed_query"]
    assert first["original_query_final"] == first["variant_query_final"] == gold


def test_no_original_adjacent_witness_does_not_prove_global_commutativity():
    operations = [(1, 1), (1, 0), (2, 0)]
    report = affine_order_witness(1, operations, 17, 4)
    assert report["changed_query_count"] == 0
    assert report["noncommuting_pair_count"] == 1
    assert report["status"] == "UNVERIFIED_NO_TESTED_ADJACENT_QUERY_CHANGE"
    assert affine_final(1, list(reversed(operations)), 17) == 3


def test_gold_and_exact_integer_bounds_are_not_inferred_or_coerced():
    mismatch = affine_order_witness(1, [(1, 1), (2, 0)], 17, 3)
    assert mismatch["status"] == "INVALID_INDEPENDENT_GOLD"
    assert not mismatch["mathematical_order_witness"]
    for initial, operations, modulus in [(True, [(1, 1)], 17), (1, [(True, 1)], 17),
        (1, [(1, 1)], True), (1, [(1, 1)], 1), (1, [], 17), (1, [(1, 1)] * 17, 17)]:
        with pytest.raises(ValueError): affine_final(initial, operations, modulus)
    with pytest.raises(ValueError): affine_order_witness(1, [(1, 1)], 17, True)


def descriptor(operations=((1, 1), (2, 0)), initial=1, modulus=17, gold=4):
    math = affine_order_witness(initial, operations, modulus, gold)
    return {"claim": "ordered_reasoning", "subject_sha256": "a" * 64,
        "public_rules": {"reference": "fixture://public-affine-rule", "sha256": "b" * 64},
        "state_evidence": {"reference": "fixture://abstract-state", "sha256": math["input_sha256"], "actual_reference_state": True},
        "independent_solvers": [{"id": "sequential", "public_reference": "fixture://sequential", "implementation_sha256": "c" * 64, "independent": True},
                                {"id": "composed", "public_reference": "fixture://composed", "implementation_sha256": "d" * 64, "independent": True}],
        "witnesses": [{"kind": "valid_nontrivial_order_permutation", "public_evidence_reference": "fixture://permutation-proof",
            "evidence_sha256": "e" * 64, "independent_oracle_recomputed": True,
            "valid_original": True, "valid_variant": True, "nontrivial_permutation": True,
            "original_query_gold_sha256": "f" * 64, "variant_query_gold_sha256": "0" * 64, "changed_query": True}],
        "mathematical_affine_witness": {"semantics": "modular_affine_final_query", "initial": initial,
            "operations": [list(pair) for pair in operations], "modulus": modulus, "queryfinal": gold},
        "mathematical_input_sha256": math["input_sha256"]}


def test_absent_or_id_only_descriptors_are_unverified_not_admitted():
    for value in (None, {}, {"claim": "entity_binding", "id": "entity-marker"}):
        report = validate_mechanism_attestation(value)
        assert report["status"] == "UNVERIFIED" and not report["descriptor_complete"]
        assert not report["admission_granted"] and not report["actual_owner_bank_verified"]
        assert not report["quotas_or_native_gates_changed"]


def test_perfect_fixture_declarations_still_require_actual_independent_owner_receipts():
    report = validate_mechanism_attestation(descriptor())
    assert report["descriptor_complete"] and report["mathematical_claim_supported"]
    assert report["status"] == "UNVERIFIED" and report["owner_receipt_required"]
    assert not report["admission_granted"] and report["model_calls"] == 0
    assert report["evidence_scope"] == "MATHEMATICAL_FIXTURE"


def test_declared_order_claim_fails_when_actual_mathematical_operations_commute():
    metadata = descriptor(((1, 1), (1, 2)), gold=4)
    report = validate_mechanism_attestation(metadata)
    assert not report["descriptor_complete"] and report["mathematical_claim_supported"] is False
    assert "ordered_claim_contradicted_by_pairwise_commutative_maps" in report["reasons"]
    assert report["status"] == "UNVERIFIED"


def test_changed_ids_do_not_replace_reference_state_or_independent_solvers():
    metadata = descriptor()
    metadata["state_evidence"] = {"id": "new-state-id", "actual_reference_state": True}
    metadata["independent_solvers"][1] = copy.deepcopy(metadata["independent_solvers"][0])
    report = validate_mechanism_attestation(metadata)
    assert "missing_actual_reference_state_evidence" in report["reasons"]
    assert "repeated_solver_identity_or_implementation" in report["reasons"]
    assert not report["descriptor_complete"] and not report["native_permission"]


def test_entity_binding_requires_both_state_renaming_and_relevant_counterfactual_evidence():
    metadata = descriptor()
    metadata["claim"] = "entity_binding"
    metadata.pop("mathematical_affine_witness")
    metadata.pop("mathematical_input_sha256")
    shared = dict(metadata["witnesses"][0], original_state_reference="fixture://before-state",
        variant_state_reference="fixture://after-state", original_state_sha256="1" * 64,
        variant_state_sha256="2" * 64)
    metadata["witnesses"] = [dict(shared, kind="bijective_entity_renaming", gold_equivariant_after_inverse_rename=True),
                             dict(shared, kind="relevant_binding_counterfactual", changed_query=True)]
    report = validate_mechanism_attestation(metadata)
    assert report["descriptor_complete"] and report["status"] == "UNVERIFIED"
    assert report["mathematical_claim_supported"] is None
    metadata["witnesses"][1].pop("variant_state_reference")
    assert not validate_mechanism_attestation(metadata)["descriptor_complete"]


def test_affine_order_proof_cannot_be_relabeled_as_entity_binding_proof():
    metadata = descriptor()
    metadata["claim"] = "entity_binding"
    report = validate_mechanism_attestation(metadata)
    assert report["mathematical_claim_supported"] is None
    assert "affine_order_witness_does_not_certify_entity_binding" in report["reasons"]
    assert not report["admission_granted"] and report["status"] == "UNVERIFIED"
