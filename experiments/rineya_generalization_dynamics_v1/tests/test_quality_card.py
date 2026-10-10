"""Synthetic public-summary metadata tests; no real native execution or bank IO."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import Blocked, digest, signed, verify
from experiments.rineya_generalization_dynamics_v1.quality_card import (
    DOMAINS, SCHEMA, build_quality_card, validate_quality_card,
)


def unsigned(value):
    return {k: copy.deepcopy(v) for k, v in value.items() if k != "sha256"}


def build_fixture_card(*, evidence_level="mechanical", measured=True, intervention_type=None):
    """Return only explicit TOY fixture declarations; create/read no files.

    Nonmechanical levels are forged caller-metadata branches for negative and
    contract tests. They are never real qualification or capability evidence.
    """
    source = {"model_id": "TOY_FIXTURE_SOURCE_NOT_RINEYA", "parameters": 596_049_920,
              "model_sha256": digest("explicit toy source payload"), "tokenizer_id": "TOY_TOKENIZER",
              "tokenizer_sha256": digest("explicit toy tokenizer payload")}
    runtime = signed({"identity": "TOY_RUNTIME", "settings": {"backend": "fixture", "threads": 1}})
    policy = signed({"identity": "TOY_POLICY", "prompt": {"template": "toy-public-only"},
                     "decoding": {"mode": "toy-greedy", "max_new_tokens": 16}, "readout": {"bias": 0}})
    common_checkpoint = {"model_id": source["model_id"], "source_model_sha256": source["model_sha256"],
                         "tokenizer_sha256": source["tokenizer_sha256"], "policy_sha256": policy["sha256"],
                         "runtime_sha256": runtime["sha256"]}
    baseline = {**common_checkpoint, "checkpoint_id": "TOY_BASELINE", "content_sha256": digest("toy baseline")}
    candidate = {**common_checkpoint, "checkpoint_id": "TOY_CANDIDATE", "content_sha256": digest("toy candidate")}
    data = {"data_id": "TOY_PUBLIC_SUMMARY_DATA", "data_sha256": digest("toy-data"),
            "split": "confirmation" if evidence_level == "confirmed" else "dev",
            "split_sha256": digest("toy-confirmation" if evidence_level == "confirmed" else "toy-dev"),
            "receipt_id": "data-receipt"}
    observed = {"wall_seconds": 1.0, "forward_calls": 3 if evidence_level != "mechanical" else 0,
                "model_loads": 1 if evidence_level != "mechanical" else 0,
                "optimizer_updates": 1 if evidence_level in {"exploratory", "confirmed"} else 0}
    budget = signed({"budget_id": "TOY_STUDY_BUDGET", "study_id": "TOY_STUDY", "owner": "toy-owner",
                     "scope": "study_specific", "limits": {"wall_seconds": 10.0, "forward_calls": 10,
                     "model_loads": 2, "optimizer_updates": 2}, "observed": observed, "receipt_id": "resources-receipt"})
    card = {"schema": SCHEMA, "study_id": "TOY_STUDY", "owner": "toy-owner", "source": source,
            "baseline": baseline, "candidate": candidate, "runtime": runtime,
            "baseline_policy": policy, "candidate_policy": copy.deepcopy(policy),
            "changed_policy_dimensions": [], "data": data, "resource_budget": budget,
            "evidence_level": evidence_level,
            "intervention_type": intervention_type or ("mechanical" if evidence_level == "mechanical" else "learned"),
            "provenance": {"declaration": "caller_supplied_public_receipts", "custody_reviewed_by": "toy-custody-reviewer",
                           "receipts": []}, "domains": {}, "confirmation": None}
    add_receipt(card, "data-receipt", "data", {"data_context": {k: v for k, v in data.items() if k != "receipt_id"}})
    add_receipt(card, "resources-receipt", "resources", {"resource_context": {k: v for k, v in unsigned(budget).items()
                                                                             if k != "receipt_id"}},
                native=evidence_level != "mechanical")
    if measured:
        add_endpoint(card, "arithmetic_state_tracking", "toy-semantic")
    if evidence_level != "mechanical":
        add_receipt(card, "native-receipt", "native_qualification", {"status": "PASS_NATIVE_MECHANICS_ONLY"}, native=True)
    if evidence_level in {"exploratory", "confirmed"}:
        add_receipt(card, "scoring-receipt", "scoring_qualification", {"status": "PASS_SCORING_ONLY"}, native=True)
    if evidence_level == "confirmed":
        add_receipt(card, "selection-receipt", "selection", {"selection": {"status": "LOCKED_DEV_ONLY",
                    "selected_checkpoint_id": candidate["checkpoint_id"], "dev_split_sha256": digest("toy-dev"),
                    "locked_before_confirmation": True}})
        add_receipt(card, "authorization-receipt", "confirmation_authorization", {"authorization": {
                    "scope": "confirmation_only", "authorized_by": "toy-independent-authorizer",
                    "independent_authorization": True, "confirmation_split_sha256": data["split_sha256"],
                    "selection_receipt_id": "selection-receipt"}})
        add_receipt(card, "confirmation-receipt", "confirmation", {"confirmation": {"status": "COMPLETED",
                    "independent_confirmation": True, "selection_receipt_id": "selection-receipt",
                    "authorization_receipt_id": "authorization-receipt", "split_sha256": data["split_sha256"],
                    "confirmation_attempt_id": "TOY_SINGLE_ATTEMPT", "single_use": True}}, native=True)
        card["confirmation"] = {"native_qualification": "native-receipt", "scoring_qualification": "scoring-receipt",
                                "selection": "selection-receipt", "authorization": "authorization-receipt",
                                "confirmation": "confirmation-receipt"}
        for receipt in card["provenance"]["receipts"]:
            if receipt["receipt_id"] in card["confirmation"].values():
                receipt["summary"] = signed({**unsigned(receipt["summary"]), "independent_review": {
                    "reviewer": "toy-independent-reviewer", "independent_of": card["owner"], "attested": True}})
    return signed(card)


def add_receipt(card, identifier, role, payload, *, native=False):
    reference_hash = digest({"explicit_fixture_receipt_bytes": identifier})
    common = {"scope": "public_aggregate_metadata", "role": role, "study_id": card["study_id"],
              "owner": card["owner"], "evidence_level": card["evidence_level"], "native_executed": native,
              "source": copy.deepcopy(card["source"]), "baseline_checkpoint_id": card["baseline"]["checkpoint_id"],
              "candidate_checkpoint_id": card["candidate"]["checkpoint_id"], "runtime_sha256": card["runtime"]["sha256"],
              "baseline": copy.deepcopy(card["baseline"]), "candidate": copy.deepcopy(card["candidate"]),
              "baseline_policy_sha256": card["baseline_policy"]["sha256"],
              "candidate_policy_sha256": card["candidate_policy"]["sha256"], "reference_sha256": reference_hash}
    # A retained locator is supplied explicitly but never created or opened.
    locator = Path(__file__).resolve().parents[4] / "quality-card-toy-public-receipts" / f"{identifier}.json"
    card["provenance"]["receipts"].append({"receipt_id": identifier,
        "reference": {"path": str(locator), "sha256": reference_hash}, "file_hash_verified_by": "toy-custody-reviewer",
        "summary": signed({**common, **payload})})


def add_endpoint(card, domain, identifier, construct="semantic_correctness", *, source_support=4):
    endpoint = {"endpoint_id": identifier, "construct": construct, "cohort": "all-public-toy-rows",
                "denominator": 10, "source_successes": source_support,
                "candidate_successes": 5 if source_support else 2, "gained": 2,
                "lost": 1 if source_support else 0, "retained": 3 if source_support else 0,
                "threshold": {"original_gate_id": "TOY_ORIGINAL_GATE", "minimum_accuracy": .5,
                              "maximum_source_success_failures": 1},
                "retention": {"source_success_membership_sha256": digest(["toy-a", "toy-b", "toy-c", "toy-d"] if source_support else []),
                              "candidate_evaluated_membership_sha256": digest(["toy-a", "toy-b", "toy-c", "toy-d"] if source_support else []),
                              "source_correct_support": source_support, "candidate_evaluated_support": source_support,
                              "retained_successes": 3 if source_support else 0, "failures": 1 if source_support else 0},
                "receipt_id": identifier + "-receipt"}
    card["domains"].setdefault(domain, {"status": "MEASURED", "endpoints": []})["endpoints"].append(endpoint)
    add_receipt(card, endpoint["receipt_id"], "endpoint", {"endpoint": {k: v for k, v in endpoint.items() if k != "receipt_id"},
                "domain": domain, "data_context": {k: v for k, v in card["data"].items() if k != "receipt_id"}},
                native=card["evidence_level"] != "mechanical")


def receipt(card, identifier):
    return next(r for r in card["provenance"]["receipts"] if r["receipt_id"] == identifier)


class QualityCardTests(unittest.TestCase):
    def card(self, **kwargs):
        return unsigned(build_fixture_card(**kwargs))

    def endpoint(self, card):
        return card["domains"]["arithmetic_state_tracking"]["endpoints"][0]

    def blocked(self, card, message="."):
        with self.assertRaisesRegex(Blocked, message):
            validate_quality_card(card)

    def test_fixture_card_keeps_missing_domains_null_and_no_authority(self):
        result = verify(validate_quality_card(build_fixture_card()))
        self.assertEqual(set(result["domains"]), set(DOMAINS))
        self.assertEqual(len(result["unmeasured_domains"]), 5)
        for domain in result["unmeasured_domains"]:
            self.assertEqual(result["domains"][domain], {"status": "UNMEASURED", "endpoints": None})
        for field in ("balanced_quality_established", "authorizes_execution", "authorizes_training", "authorizes_confirmation",
                      "permits_model_promotion", "native_execution_verified_by_this_module", "external_file_hashes_verified_by_this_module"):
            self.assertIs(result[field], False)
        self.assertEqual(result["evidence_scope"], "aggregate_public_receipt_metadata_only")

    def test_build_requires_supplied_identities_and_normalizes_domains(self):
        fixture = self.card(measured=False)
        fixture.pop("schema")
        fixture.pop("domains")
        built = build_quality_card(**fixture)
        self.assertEqual(len(built["domains"]), 6)
        self.assertTrue(all(v["endpoints"] is None for v in built["domains"].values()))
        fixture["source"].pop("model_sha256")
        with self.assertRaises(Blocked):
            build_quality_card(**fixture)

    def test_gains_do_not_hide_fixed_source_correct_losses(self):
        result = validate_quality_card(self.card())
        endpoint = result["domains"]["arithmetic_state_tracking"]["endpoints"][0]
        self.assertEqual(endpoint["independent_source_incorrect_gains"], 2)
        self.assertEqual(endpoint["source_correct_losses"], 1)
        self.assertEqual(endpoint["source_correct_retained"], 3)
        self.assertEqual(endpoint["retention_fraction"], .75)

    def test_zero_source_success_support_is_undefined_not_perfect(self):
        card = self.card(measured=False)
        add_endpoint(card, "grounded_retrieval", "toy-zero-support", source_support=0)
        endpoint = validate_quality_card(card)["domains"]["grounded_retrieval"]["endpoints"][0]
        self.assertIsNone(endpoint["retention_fraction"])

    def test_endpoint_constructs_are_separate_without_pooled_score(self):
        card = self.card()
        for construct in ("formatting", "abstention", "termination", "groundedness", "joint_correctness"):
            add_endpoint(card, "arithmetic_state_tracking", "toy-" + construct, construct)
        result = validate_quality_card(card)
        self.assertEqual(len(result["domains"]["arithmetic_state_tracking"]["endpoints"]), 6)
        self.assertNotIn("aggregate_accuracy", result)

    def test_count_coercion_and_nonfinite_values_fail_closed(self):
        for field in ("denominator", "source_successes", "candidate_successes", "gained", "lost", "retained"):
            for value in (True, 1.0, float("nan"), float("inf"), -1, "1"):
                card = self.card()
                self.endpoint(card)[field] = value
                self.blocked(card)
        for value in (True, float("nan"), float("inf"), -1):
            card = self.card()
            budget = unsigned(card["resource_budget"])
            budget["observed"]["wall_seconds"] = value
            if type(value) is float and not value < float("inf"):
                card["resource_budget"]["observed"]["wall_seconds"] = value
            else:
                card["resource_budget"] = signed(budget)
            self.blocked(card)

    def test_inconsistent_pairs_or_denominators_block(self):
        for field, value in (("source_successes", 5), ("candidate_successes", 7), ("lost", 0),
                             ("gained", 9), ("retained", 4), ("denominator", 5)):
            card = self.card()
            self.endpoint(card)[field] = value
            self.blocked(card, "Paired endpoint counts")

    def test_retention_membership_cannot_move_or_shrink(self):
        for field, value in (("candidate_evaluated_membership_sha256", digest(["toy-a", "toy-b", "toy-c"])),
                             ("candidate_evaluated_support", 3), ("source_correct_support", 5),
                             ("retained_successes", 4), ("failures", 0), ("failures", True)):
            card = self.card()
            self.endpoint(card)["retention"][field] = value
            self.blocked(card, "retention|integer")

    def test_original_threshold_or_endpoint_mutation_blocks(self):
        card = self.card()
        self.endpoint(card)["threshold"]["minimum_accuracy"] = .1
        self.blocked(card, "original threshold changed")
        card = self.card()
        self.endpoint(card)["cohort"] = "new-easier-cohort"
        self.blocked(card, "original threshold changed")

    def test_signed_card_runtime_policy_and_receipt_mutation_blocks(self):
        mutations = (lambda c: c.update(owner="other"), lambda c: c["runtime"]["settings"].update(threads=2),
                     lambda c: c["baseline_policy"]["readout"].update(bias=3.75),
                     lambda c: c["provenance"]["receipts"][0]["summary"].update(owner="other"))
        for mutate in mutations:
            card = build_fixture_card()
            mutate(card)
            self.blocked(card, "digest mismatch")

    def test_same_model_tokenizer_and_checkpoint_policy_identity_required(self):
        for field, value in (("model_id", "OTHER_MODEL"), ("source_model_sha256", digest("other")),
                             ("tokenizer_sha256", digest("other tokenizer")),
                             ("policy_sha256", digest("other policy")), ("runtime_sha256", digest("other runtime"))):
            card = self.card()
            card["candidate"][field] = value
            self.blocked(card, "exact source|policy/runtime")

    def test_every_public_receipt_binds_both_exact_checkpoint_contents(self):
        for role in ("baseline", "candidate"):
            card = self.card()
            card[role]["content_sha256"] = digest("different model weights with same checkpoint name")
            self.blocked(signed(card), "Receipt source/study/runtime/policy/evidence identity differs")

    def test_endpoint_domain_and_data_split_cannot_be_reassigned(self):
        card = self.card()
        card["domains"]["language_long_context_preservation"] = card["domains"].pop("arithmetic_state_tracking")
        self.blocked(card, "Endpoint domain/data/split differs")
        card = self.card()
        card["data"]["split_sha256"] = digest("different public DEV rows")
        item = receipt(card, "data-receipt")
        item["summary"] = signed({**unsigned(item["summary"]), "data_context": {
            k: v for k, v in card["data"].items() if k != "receipt_id"}})
        self.blocked(card, "Endpoint domain/data/split differs")

    def test_learned_policy_change_is_rejected_even_if_all_hashes_rebound(self):
        card = self.card(evidence_level="exploratory")
        policy = unsigned(card["candidate_policy"])
        policy["readout"]["bias"] = 3.75
        card["candidate_policy"] = signed(policy)
        card["candidate"]["policy_sha256"] = card["candidate_policy"]["sha256"]
        card["changed_policy_dimensions"] = ["readout"]
        self.blocked(card, "Learned comparison must keep")

    def test_policy_changes_must_match_declared_dimensions(self):
        card = self.card()
        card["changed_policy_dimensions"] = ["readout"]
        self.blocked(card, "Declared policy contrast")
        for values in (["readout", "readout"], ["new-policy-type"], [True]):
            card = self.card()
            card["changed_policy_dimensions"] = values
            self.blocked(card, "Declare exact changed")

    def test_serving_cannot_compare_different_model_bytes(self):
        card = self.card(evidence_level="exploratory", intervention_type="serving")
        policy = unsigned(card["candidate_policy"])
        policy["readout"]["bias"] = 3.75
        card["candidate_policy"] = signed(policy)
        card["candidate"]["policy_sha256"] = card["candidate_policy"]["sha256"]
        card["changed_policy_dimensions"] = ["readout"]
        self.blocked(card, "Serving comparison must retain the same model bytes")

    def test_same_checkpoint_serving_policy_contrast_validates_as_metadata_only(self):
        card = self.card(evidence_level="exploratory", intervention_type="serving")
        policy = unsigned(card["candidate_policy"])
        policy["readout"]["bias"] = 3.75
        card["candidate_policy"] = signed(policy)
        card["candidate"]["policy_sha256"] = card["candidate_policy"]["sha256"]
        card["candidate"]["content_sha256"] = card["baseline"]["content_sha256"]
        card["changed_policy_dimensions"] = ["readout"]
        budget = unsigned(card["resource_budget"])
        budget["observed"]["optimizer_updates"] = 0
        card["resource_budget"] = signed(budget)
        for item in card["provenance"]["receipts"]:
            body = unsigned(item["summary"])
            body["candidate"] = copy.deepcopy(card["candidate"])
            body["candidate_policy_sha256"] = card["candidate_policy"]["sha256"]
            if body["role"] == "resources":
                body["resource_context"] = {k: v for k, v in budget.items() if k != "receipt_id"}
            item["summary"] = signed(body)
        result = validate_quality_card(card)
        self.assertEqual(result["card"]["intervention_type"], "serving")
        self.assertIs(result["balanced_quality_established"], False)
        self.assertIs(result["authorizes_execution"], False)

    def test_existing_learned_candidate_can_be_assessed_without_new_study_updates(self):
        card = self.card(evidence_level="exploratory", intervention_type="learned")
        budget = unsigned(card["resource_budget"])
        budget["observed"]["optimizer_updates"] = 0
        card["resource_budget"] = signed(budget)
        item = receipt(card, "resources-receipt")
        item["summary"] = signed({**unsigned(item["summary"]), "resource_context": {
            k: v for k, v in budget.items() if k != "receipt_id"}})
        result = validate_quality_card(card)
        self.assertTrue(any("historical training" in warning for warning in result["warnings"]))
        self.assertIs(result["authorizes_training"], False)

    def test_resource_receipt_cannot_deny_declared_native_execution(self):
        card = self.card(evidence_level="exploratory")
        item = receipt(card, "resources-receipt")
        item["summary"] = signed({**unsigned(item["summary"]), "native_executed": False})
        self.blocked(card, "Resource execution provenance disagrees")

    def test_budget_is_per_study_and_cannot_import_gd_permit(self):
        for field, value in (("study_id", "GD-001-OTHER-STUDY"), ("owner", "another-owner"),
                             ("scope", "inherited_gd_budget")):
            card = self.card()
            budget = unsigned(card["resource_budget"])
            budget[field] = value
            card["resource_budget"] = signed(budget)
            self.blocked(card, "another study")
        card = self.card()
        budget = unsigned(card["resource_budget"])
        budget["observed"]["forward_calls"] = 100
        card["resource_budget"] = signed(budget)
        self.blocked(card, "exceed this study")

    def test_resource_counters_are_strict_and_original_receipt_bound(self):
        for field in ("forward_calls", "model_loads", "optimizer_updates"):
            card = self.card()
            budget = unsigned(card["resource_budget"])
            budget["observed"][field] = True
            card["resource_budget"] = signed(budget)
            self.blocked(card, "exact integer")
        card = self.card()
        budget = unsigned(card["resource_budget"])
        budget["limits"]["wall_seconds"] = 100.0
        card["resource_budget"] = signed(budget)
        self.blocked(card, "Resource receipt binding differs")

    def test_unmeasured_is_null_unknown_domains_and_missing_identities_block(self):
        for value in (0, [], {}):
            card = self.card()
            card["domains"]["instruction_following"] = {"status": "UNMEASURED", "endpoints": value}
            self.blocked(card, "null, never zero")
        card = self.card()
        card["domains"]["made_up_quality"] = {"status": "UNMEASURED", "endpoints": None}
        self.blocked(card, "Unknown quality domain")
        for field in ("runtime", "source", "provenance", "resource_budget", "baseline_policy", "candidate"):
            card = self.card()
            card.pop(field)
            self.blocked(card, "Incomplete/unknown quality card")

    def test_full_mechanical_coverage_still_cannot_establish_balanced_quality(self):
        card = self.card(measured=False)
        for domain in DOMAINS:
            add_endpoint(card, domain, "toy-" + domain)
        result = validate_quality_card(card)
        self.assertEqual(result["unmeasured_domains"], [])
        self.assertIs(result["balanced_quality_established"], False)
        self.assertIs(result["permits_model_promotion"], False)

    def test_mechanical_receipts_cannot_be_relabelled_as_native(self):
        card = self.card()
        card["evidence_level"] = "exploratory"
        self.blocked(card, "Mechanical intervention")
        card["intervention_type"] = "learned"
        self.blocked(card, "evidence identity differs")
        card = self.card()
        item = receipt(card, "toy-semantic-receipt")
        item["summary"] = signed({**unsigned(item["summary"]), "native_executed": True})
        self.blocked(card, "Mechanical receipts cannot claim native")

    def test_native_qualification_is_not_efficacy_or_execution_authority(self):
        result = validate_quality_card(self.card(evidence_level="native_qualification", intervention_type="mechanical"))
        self.assertIs(result["balanced_quality_established"], False)
        self.assertIs(result["authorizes_execution"], False)
        self.assertTrue(any("cannot establish model-quality efficacy" in w for w in result["warnings"]))

    def test_exploratory_needs_qualified_native_and_scoring_metadata(self):
        for identifier in ("native-receipt", "scoring-receipt"):
            card = self.card(evidence_level="exploratory")
            card["provenance"]["receipts"] = [r for r in card["provenance"]["receipts"] if r["receipt_id"] != identifier]
            self.blocked(card, "Missing qualified")
        card = self.card(evidence_level="exploratory")
        item = receipt(card, "native-receipt")
        item["summary"] = signed({**unsigned(item["summary"]), "status": "PASS_FIXTURE_ONLY"})
        self.blocked(card, "Missing qualified")

    def test_confirmed_chain_is_supplied_metadata_not_verified_execution(self):
        result = validate_quality_card(self.card(evidence_level="confirmed"))
        self.assertIs(result["confirmed_chain_metadata_complete"], True)
        self.assertIs(result["native_execution_verified_by_this_module"], False)
        self.assertIs(result["balanced_quality_established"], False)
        self.assertIs(result["authorizes_confirmation"], False)

    def test_confirmed_needs_every_independent_authorization_and_selection_role(self):
        for key in ("native_qualification", "scoring_qualification", "selection", "authorization", "confirmation"):
            card = self.card(evidence_level="confirmed")
            card["confirmation"].pop(key)
            self.blocked(card, "confirmation chain")
        card = self.card(evidence_level="confirmed")
        item = receipt(card, "selection-receipt")
        body = unsigned(item["summary"])
        body["selection"]["locked_before_confirmation"] = False
        item["summary"] = signed(body)
        self.blocked(card, "locked DEV selection")
        card = self.card(evidence_level="confirmed")
        item = receipt(card, "authorization-receipt")
        body = unsigned(item["summary"])
        body["authorization"]["independent_authorization"] = False
        item["summary"] = signed(body)
        self.blocked(card, "separate source-bound confirmation authorization")

    def test_confirmed_independence_and_moving_confirmation_split_block(self):
        card = self.card(evidence_level="confirmed")
        item = receipt(card, "confirmation-receipt")
        body = unsigned(item["summary"])
        body["independent_review"]["reviewer"] = card["owner"]
        item["summary"] = signed(body)
        self.blocked(card, "independent supplied review")
        card = self.card(evidence_level="confirmed")
        item = receipt(card, "confirmation-receipt")
        body = unsigned(item["summary"])
        body["confirmation"]["split_sha256"] = digest("moved bank")
        item["summary"] = signed(body)
        self.blocked(card, "completed independent confirmation")

    def test_public_reference_summary_hashes_and_caller_custody_are_explicit(self):
        for mutate in (lambda r: r["reference"].update(sha256=digest("other file")),
                       lambda r: r.update(file_hash_verified_by=""),
                       lambda r: r["reference"].update(path="relative-receipt.json")):
            card = self.card()
            mutate(card["provenance"]["receipts"][0])
            self.blocked(card)

    def test_validator_never_opens_models_receipts_or_bank_payloads(self):
        card = self.card(evidence_level="confirmed")
        with patch.object(Path, "open", side_effect=AssertionError("No bank or receipt IO")), \
             patch.object(Path, "read_text", side_effect=AssertionError("No payload access")):
            result = validate_quality_card(card)
        self.assertIs(result["confirmed_chain_metadata_complete"], True)

    def test_source_parameter_count_is_exact_and_within_campaign(self):
        for count in (True, 596_049_920.0, 344_000_000, 700_000_001):
            card = self.card()
            card["source"]["parameters"] = count
            self.blocked(card, "500M-700M")


if __name__ == "__main__":
    unittest.main()
