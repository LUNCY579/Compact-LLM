"""Finite illustrative lookup properties only; never native or dataset evidence."""
from __future__ import annotations

import copy
import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import Blocked, digest, signed, verify
from experiments.rineya_generalization_dynamics_v1.query_dependence_witness import (
    SCHEMA, audit_witness, build_witness,
)


def unsigned(value):
    return {k: copy.deepcopy(v) for k, v in value.items() if k != "sha256"}


def refresh(context, *, answers=False):
    # Test-only mechanical fixture rebinding; no actual bank or source is involved.
    context["context_text"] = "\n".join(f'Record(entity={r["entity"]},field={r["field"]},value={r["value"]}).'
                                        for r in context["records"])
    context["context_sha256"] = digest(context["context_text"])
    for query in context["queries"]:
        query["public_input"]["context_text"] = context["context_text"]
        if answers:
            matches = [r["value"] for r in context["records"]
                       if (r["entity"], r["field"]) == (query["entity"], query["field"])]
            query["oracle_answer"] = {"v": matches[0] if matches else None}


class QueryDependenceWitnessTests(unittest.TestCase):
    def fixture(self, **kwargs):
        return unsigned(build_witness(**kwargs))

    def blocked(self, fixture, pattern="."):
        with self.assertRaisesRegex(Blocked, pattern):
            audit_witness(signed(fixture))

    def test_minimal_exact_four_context_witness_has_all_conditional_half_bounds(self):
        witness = build_witness()
        certificate = verify(audit_witness(witness))
        self.assertEqual(certificate["contexts"], 4)
        self.assertEqual(certificate["queries"], 64)
        self.assertEqual(len(certificate["conditional_chance_certificates"]), 5)
        for bound in certificate["conditional_chance_certificates"].values():
            self.assertEqual(bound["exact_fraction"], "1/2")
            self.assertEqual(bound["maximum_label_accuracy"], .5)
            self.assertEqual(bound["maximum_typed_accuracy"], .5)
        context_bound = certificate["conditional_chance_certificates"]["context_only"]
        self.assertEqual(context_bound["maximum_typed_correct"], 32)
        self.assertEqual(context_bound["conditional_group_sizes"], {16: 4})
        self.assertEqual(certificate["matched_nuisance_profile"]["record_count"], 12)

    def test_all_regular_enumerates_90_contexts_and_1440_equal_weight_queries(self):
        certificate = audit_witness(build_witness(family="all_regular"))
        self.assertEqual(certificate["contexts"], 90)
        self.assertEqual(certificate["queries"], 1440)
        query_only = certificate["conditional_chance_certificates"]["query_only_without_context"]
        self.assertEqual(query_only["groups"], 16)
        self.assertEqual(query_only["conditional_group_sizes"], {90: 16})
        self.assertEqual(query_only["maximum_label_correct"], 720)
        self.assertEqual(query_only["maximum_typed_correct"], 720)

    def test_all_outputs_explicitly_exclude_learning_banks_admission_and_authority(self):
        witness = build_witness()
        certificate = audit_witness(witness)
        self.assertIs(certificate["no_model_quality_claim"], True)
        for key in ("actual_train_admission", "actual_dev_admission", "native_bank", "actual_learning_established",
                    "retention_established", "model_quality_claim", "confirmation_authorized", "authorizes_execution"):
            self.assertIs(witness["generator_policy"][key], False)
            self.assertIs(certificate[key], False)
        self.assertIs(certificate["token_length_invariance_established"], False)
        self.assertIn("oracle properties only", certificate["operator_invariances"]["scope"])

    def test_opaque_symbols_unique_values_and_nuisance_profiles_match_every_context(self):
        witness = build_witness(seed=9)
        inventory = None
        for context in witness["contexts"]:
            values = sorted(r["value"] for r in context["records"])
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual({len(v) for v in values}, {4})
            if inventory is None:
                inventory = values
            self.assertEqual(values, inventory)
            answers = [q["oracle_answer"]["v"] for q in context["queries"]]
            self.assertEqual(answers.count(None), 8)
            self.assertEqual(max(answers.count(v) for v in set(answers)), 8)
            public = [q["public_input"] for q in context["queries"]]
            self.assertTrue(all(p["context_text"] == context["context_text"] for p in public))

    def test_each_joint_query_balanced_across_contexts_not_just_global_marginals(self):
        witness = build_witness()
        for e, f in itertools.product(witness["entities"], witness["fields"]):
            answers = [q["oracle_answer"]["v"] for c in witness["contexts"] for q in c["queries"]
                       if (q["entity"], q["field"]) == (e, f)]
            self.assertEqual(len(answers), 4)
            self.assertEqual(sum(v is None for v in answers), 2)

    def test_fixed_query_fact_switch_changes_answer_without_nuisance_changes(self):
        witness = build_witness()
        e, f = witness["entities"][0], witness["fields"][0]
        a, b = witness["contexts"][0], witness["contexts"][2]
        answer = lambda context: next(q["oracle_answer"]["v"] for q in context["queries"]
                                      if (q["entity"], q["field"]) == (e, f))
        self.assertIsNotNone(answer(a))
        self.assertIsNone(answer(b))
        self.assertEqual(sorted(r["value"] for r in a["records"]), sorted(r["value"] for r in b["records"]))

    def test_oracle_permutation_irrelevance_selected_value_and_relation_checks_execute(self):
        result = audit_witness(build_witness())
        counts = result["operator_invariances"]
        self.assertEqual(counts["record_permutation_query_checks"], 4 * 32)
        self.assertEqual(counts["irrelevant_value_query_checks"], 4 * 16)
        self.assertEqual(counts["selected_value_query_checks"], 4 * 4 * 16)
        self.assertEqual(counts["relation_switch_query_checks"], 4 * 16)

    def test_public_context_permutation_preserves_audit_with_rebound_fixture_bytes(self):
        witness = self.fixture()
        for context in witness["contexts"]:
            context["records"].reverse()
            refresh(context)
        self.assertEqual(audit_witness(signed(witness))["queries"], 64)

    def test_duplicate_omitted_or_out_of_domain_queries_fail(self):
        for mutation in (lambda q: q.pop(), lambda q: q.__setitem__(1, copy.deepcopy(q[0])),
                         lambda q: q[0].update(entity="BAD0"), lambda q: q[0].update(query_id=q[1]["query_id"])):
            witness = self.fixture()
            mutation(witness["contexts"][0]["queries"])
            self.blocked(witness)

    def test_oracle_mismatch_and_boolean_numeric_coercion_fail(self):
        for value in (False, True, 0, 1.0, "BAD0"):
            witness = self.fixture()
            witness["contexts"][0]["queries"][0]["oracle_answer"] = {"v": value}
            self.blocked(witness, "Oracle|oracle")
        witness = self.fixture()
        query = next(q for q in witness["contexts"][0]["queries"] if q["oracle_answer"]["v"] is not None)
        query["oracle_answer"] = {"v": None}
        self.blocked(witness, "oracle")

    def test_context_cannot_change_per_query_even_if_queries_globally_balance(self):
        witness = self.fixture()
        witness["contexts"][0]["queries"][0]["public_input"]["context_text"] += "\nextra factual cue"
        self.blocked(witness, "Changing context per query")

    def test_duplicate_facts_and_changed_value_inventory_fail(self):
        witness = self.fixture()
        context = witness["contexts"][0]
        context["records"][1] = copy.deepcopy(context["records"][0])
        refresh(context)
        self.blocked(witness, "Duplicate entity-field")
        witness = self.fixture()
        context = witness["contexts"][0]
        context["records"][0]["value"] = context["records"][1]["value"]
        refresh(context, answers=True)
        self.blocked(witness, "value inventory")

    def test_global_only_balance_is_rejected_by_conditional_entity_degree_check(self):
        witness = self.fixture()
        context = witness["contexts"][0]
        matrix = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
        # Eight present cells and field degree2 globally, but entity degrees4/0/2/2.
        self.assertEqual(sum(sum(r) for r in matrix), 8)
        self.assertEqual([sum(r[j] for r in matrix) for j in range(4)], [2] * 4)
        values = iter(witness["values"][:8])
        core = [{"entity": witness["entities"][i], "field": witness["fields"][j], "value": next(values)}
                for i, row in enumerate(matrix) for j, present in enumerate(row) if present]
        nuisance = [r for r in context["records"] if r["entity"] in witness["nuisance_entities"]]
        context["records"] = core + nuisance
        refresh(context, answers=True)
        self.blocked(witness, "Per-context conditional")

    def test_nuisance_facts_and_full_frequency_profiles_must_match_across_graphs(self):
        witness = self.fixture()
        context = witness["contexts"][1]
        nuisance = [r for r in context["records"] if r["entity"] in witness["nuisance_entities"]]
        nuisance[0]["value"], nuisance[1]["value"] = nuisance[1]["value"], nuisance[0]["value"]
        refresh(context, answers=True)
        self.blocked(witness, "irrelevant facts changed")
        witness = self.fixture()
        for context in witness["contexts"]:
            for record in context["records"]:
                if record["entity"] == witness["nuisance_entities"][1]:
                    record["entity"] = witness["nuisance_entities"][0]
            refresh(context)
        self.blocked(witness, "exact entity and field frequency")

    def test_duplicate_graphs_and_missing_complements_rejected_not_fake_replication(self):
        witness = self.fixture()
        witness["contexts"][1] = copy.deepcopy(witness["contexts"][0])
        witness["contexts"][1]["context_id"] = "replacement_context"
        self.blocked(witness, "Exact regular graph family")
        witness = self.fixture()
        witness["family"] = "all_regular"
        self.blocked(witness, "family incomplete")

    def test_role_id_label_metadata_never_enters_public_projection(self):
        for mutation in (lambda q: q["public_input"].update(context_id="label-revealing-id"),
                         lambda q: q["public_input"].update(role="present"),
                         lambda q: q["public_input"].update(request=q["public_input"]["request"] + " PRESENT"),
                         lambda q: q.update(label="present")):
            witness = self.fixture()
            mutation(witness["contexts"][0]["queries"][0])
            self.blocked(witness, "metadata|role/ID/label")
        certificate = audit_witness(build_witness())
        self.assertTrue(any("IDs" in warning for warning in certificate["warnings"]))
        self.assertTrue(any("side channels" in warning for warning in certificate["warnings"]))

    def test_digest_tampering_and_permission_relabeling_fail_closed(self):
        witness = build_witness()
        witness["contexts"][0]["queries"][0]["oracle_answer"] = {"v": "BAD0"}
        with self.assertRaisesRegex(Blocked, "digest mismatch"):
            audit_witness(witness)
        for value in (True, 0, "APPROVED"):
            witness = self.fixture()
            witness["generator_policy"]["actual_train_admission"] = value
            self.blocked(witness, "cannot authorize")

    def test_seed_dtype_and_symbol_metadata_coercions_fail_closed(self):
        for seed in (True, 1.0, -1, 2**63):
            with self.assertRaises(Blocked):
                build_witness(seed=seed)
            witness = self.fixture()
            witness["fixture_seed"] = seed
            self.blocked(witness, "fixture seed")
        witness = self.fixture()
        witness["entities"][0] = False
        self.blocked(witness, "opaque equal-width")
        witness = self.fixture()
        witness["contexts"][0]["records"][0]["value"] = 1
        self.blocked(witness, "symbols/types")

    def test_reproducibility_and_input_immutability(self):
        a, b = build_witness(seed=11), build_witness(seed=11)
        self.assertEqual(a, b)
        self.assertNotEqual(a["sha256"], build_witness(seed=12)["sha256"])
        before = copy.deepcopy(a)
        audit_witness(a)
        self.assertEqual(a, before)

    def test_build_and_audit_perform_no_files_models_or_bank_io(self):
        with patch("builtins.open", side_effect=AssertionError("No file IO")), \
             patch.object(Path, "open", side_effect=AssertionError("No bank IO")), \
             patch.object(Path, "read_text", side_effect=AssertionError("No payload reads")):
            witness = build_witness()
            certificate = audit_witness(witness)
        self.assertEqual(certificate["queries"], 64)
        self.assertEqual(certificate["schema"], SCHEMA)


if __name__ == "__main__":
    unittest.main()
