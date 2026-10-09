"""In-memory fictional descriptors only; no real populations or receipts."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import guard_coverage as registry


def reference(label):
    return {"path": "C:/Users/clclc/BusinessProject/INERT_COVERAGE_METADATA/" + label + ".json",
            "sha256": registry.digest({"fictional_reference": label})}


def pending():
    return {"status": "PENDING", "descriptor": None}


def declared(label):
    return {"status": "DECLARED_UNVERIFIED", "descriptor": reference(label)}


def make_fixture():
    """Supply the minimal closed five-domain plan without creating payloads."""
    policies = {role: dict(reference("policy_" + role), policy_id=identifier)
                for role, identifier in registry.POLICY_IDS.items()}
    domains = []
    for name in registry.DOMAINS:
        mapping = ({"policy": "task", "mechanism": registry.TASK_MAPPING[name],
                    "policy_sha256": policies["task"]["sha256"]}
                   if name in registry.TASK_MAPPING else None)
        domains.append({
            "domain": name, "context": "long" if name == "long_context" else "short",
            "mechanism_mapping": mapping, "population_scope_id": None,
            "population": pending(), "scorer": pending(), "qualification": pending(),
            "enrollment": pending(), "candidate_rows": None, "declared_source_correct_rows": None,
        })
    declarations = {relation: {
        unit: {"status": "PENDING", "declared_disjoint": None, "descriptor": None}
        for unit in ("source", "parent")} for relation in registry.DISJOINT_RELATIONS}
    plan = {
        "schema": registry.SCHEMA, "experiment": "SYNTH-CRS-001", "status": "CLOSED_PREPARATION",
        "selected_policies": deepcopy(policies), "required_short_domains": list(registry.SHORT_DOMAINS),
        "long_context_domain": "long_context", "domains": domains, "disjointness": declarations,
        "claims": dict(registry.CLOSED_CLAIMS), "scientific_coverage_status": "UNDEFINED",
    }
    return plan, policies, registry.digest(plan)


def declare_domain(domain, source_correct_rows=0, candidate_rows=10):
    """Unverified declarations do not become native scorer/enrollment evidence."""
    name = domain["domain"]
    for role in ("population", "scorer", "qualification", "enrollment"):
        domain[role] = declared(name + "_" + role)
    domain["population_scope_id"] = "declared." + name
    domain["candidate_rows"] = candidate_rows
    domain["declared_source_correct_rows"] = source_correct_rows


class GuardCoverageControls(unittest.TestCase):
    def setUp(self):
        self.plan, self.policies, self.pin = make_fixture()

    def domain(self, name="retrieval"):
        return next(d for d in self.plan["domains"] if d["domain"] == name)

    def validate(self, refreeze=False):
        if refreeze:
            self.pin = registry.digest(self.plan)
        return registry.validate_coverage(self.plan, self.policies, self.pin)

    def refused(self, mutation, message="."):
        mutation()
        with self.assertRaisesRegex(registry.ValidationError, message):
            self.validate(refreeze=True)

    def test_minimal_closed_plan_preserves_exact_missing_bindings(self):
        result = self.validate()
        self.assertTrue(result["structural_only"])
        self.assertEqual(result["scientific_coverage_status"], "UNDEFINED")
        self.assertIsNone(result["actual_support"])
        self.assertEqual(result["preserved_plan"], self.plan)
        self.assertIn("domains.arithmetic.mechanism_mapping", result["missing_bindings"])
        self.assertIn("domains.dialogue.qualification.descriptor", result["missing_bindings"])
        self.assertEqual({d["domain"] for d in result["domains"]}, set(registry.DOMAINS))
        for domain in result["domains"]:
            self.assertIsNone(domain["qualified_source_correct_rows"])
            self.assertIsNone(domain["actual_support"])
        for name, value in registry.CLOSED_CLAIMS.items():
            self.assertIs(result[name], value)
        for name in ("policy_contents_verified", "receipt_contents_verified",
                     "source_disjointness_verified", "parent_disjointness_verified",
                     "actual_enrollment_verified", "payloads_read"):
            self.assertIs(result[name], False)

    def test_no_descriptor_policy_or_population_is_opened(self):
        with patch("builtins.open", side_effect=AssertionError("payload I/O forbidden")):
            self.validate()

    def test_registry_imports_only_stdlib_and_no_scorer(self):
        source = Path(registry.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertEqual(imports, {"__future__", "hashlib", "json", "re"})

    def test_valid_current_task_mappings_are_explicit(self):
        self.assertEqual(self.domain("retrieval")["mechanism_mapping"]["mechanism"], "retrieval")
        self.assertEqual(self.domain("abstention")["mechanism_mapping"]["mechanism"], "answerability_abstention")
        self.assertIsNone(self.domain("arithmetic")["mechanism_mapping"])
        self.assertIsNone(self.domain("dialogue")["mechanism_mapping"])
        self.assertIsNone(self.domain("long_context")["mechanism_mapping"])

    def test_instruction_following_cannot_substitute_arithmetic(self):
        self.refused(lambda: self.domain("arithmetic").update(
            mechanism_mapping={"policy": "task", "mechanism": "instruction_following",
                               "policy_sha256": self.policies["task"]["sha256"]}),
                     "cannot substitute")

    def test_retrieval_source_cannot_substitute_arithmetic_or_dialogue(self):
        for name in ("arithmetic", "dialogue"):
            with self.subTest(name=name):
                self.setUp()
                self.refused(lambda: self.domain(name).update(
                    mechanism_mapping=deepcopy(self.domain("retrieval")["mechanism_mapping"])),
                             "cannot substitute")

    def test_instruction_or_answerability_cannot_substitute_retrieval(self):
        for mechanism in ("instruction_following", "answerability_abstention"):
            with self.subTest(mechanism=mechanism):
                self.setUp()
                self.refused(lambda: self.domain()["mechanism_mapping"].update(mechanism=mechanism),
                             "substitution")

    def test_stale_mechanism_policy_hash_refused(self):
        self.refused(lambda: self.domain()["mechanism_mapping"].update(policy_sha256="0" * 64), "stale")

    def test_selected_policy_descriptor_hash_cannot_change(self):
        self.refused(lambda: self.plan["selected_policies"]["task"].update(sha256="0" * 64), "binding mismatch")

    def test_policy_roles_and_duplicate_policy_paths_refused(self):
        self.policies["task"]["policy_id"] = self.policies["engineering"]["policy_id"]
        with self.assertRaisesRegex(registry.ValidationError, "role/identity substitution"):
            self.validate()
        self.setUp()
        self.policies["task"]["path"] = self.policies["engineering"]["path"]
        self.policies["task"]["sha256"] = self.policies["engineering"]["sha256"]
        with self.assertRaisesRegex(registry.ValidationError, "conflicting selected policy paths"):
            self.validate()

    def test_unfrozen_plan_mutation_refused(self):
        self.domain()["candidate_rows"] = 10
        with self.assertRaisesRegex(registry.ValidationError, "externally frozen SHA256"):
            self.validate()

    def test_domain_omission_or_duplication_refused(self):
        self.refused(lambda: self.plan["domains"].pop(), "all four short")
        self.setUp()
        self.refused(lambda: self.plan["domains"].__setitem__(3, deepcopy(self.domain())), "duplicate domain")

    def test_unknown_domain_cannot_expand_or_replace_registered_scope(self):
        self.refused(lambda: self.domain("dialogue").update(domain="language"), "unknown/substituted")

    def test_long_guard_cannot_replace_a_short_guard(self):
        self.refused(lambda: self.domain("arithmetic").update(context="long"), "cannot substitute")

    def test_short_task_source_cannot_replace_long_context(self):
        self.refused(lambda: self.domain("long_context").update(
            mechanism_mapping=deepcopy(self.domain()["mechanism_mapping"])), "cannot substitute")

    def test_population_scope_is_bound_and_globally_unique(self):
        declare_domain(self.domain("retrieval"))
        declare_domain(self.domain("abstention"))
        self.refused(lambda: self.domain("abstention").update(
            population_scope_id=self.domain("retrieval")["population_scope_id"]), "duplicate population scope")
        self.setUp()
        self.refused(lambda: self.domain().update(population_scope_id="unbound.retrieval"), "unbound population")

    def test_candidate_counts_are_not_inferred_enrollment(self):
        self.domain()["candidate_rows"] = 768
        result = self.validate(refreeze=True)
        row = next(d for d in result["domains"] if d["domain"] == "retrieval")
        self.assertEqual(row["declared_candidate_rows"], 768)
        self.assertIsNone(row["declared_source_correct_rows"])
        self.assertIsNone(row["qualified_source_correct_rows"])
        self.assertEqual(result["scientific_coverage_status"], "UNDEFINED")

    def test_candidate_to_enrolled_count_claim_without_reference_refused(self):
        self.domain()["candidate_rows"] = 768
        self.refused(lambda: self.domain().update(declared_source_correct_rows=768), "cannot supply enrolled")

    def test_authored_dialogue_candidates_do_not_qualify_dialogue(self):
        declare_domain(self.domain("dialogue"), source_correct_rows=20, candidate_rows=128)
        result = self.validate(refreeze=True)
        row = next(d for d in result["domains"] if d["domain"] == "dialogue")
        self.assertEqual(row["declared_source_correct_rows"], 20)
        self.assertIsNone(row["qualified_source_correct_rows"])
        self.assertEqual(row["scientific_coverage_status"], "UNDEFINED")
        self.assertIs(result["retention_qualified"], False)
        self.assertIn("domains.dialogue.mechanism_mapping", result["missing_bindings"])

    def test_qualification_receipt_reference_never_proves_native_readiness(self):
        for domain in self.plan["domains"]:
            declare_domain(domain, source_correct_rows=10, candidate_rows=10)
        result = self.validate(refreeze=True)
        self.assertFalse(result["native_admissible"])
        self.assertFalse(result["scoring_authorized"])
        self.assertFalse(result["retention_qualified"])
        self.assertFalse(result["model_quality_gain"])
        self.assertFalse(result["actual_enrollment_verified"])
        self.assertFalse(result["receipt_contents_verified"])
        self.assertTrue(all(d["qualified_source_correct_rows"] is None for d in result["domains"]))

    def test_zero_declared_source_correct_support_is_undefined_not_perfect(self):
        for domain in self.plan["domains"]:
            declare_domain(domain, source_correct_rows=0, candidate_rows=10)
        result = self.validate(refreeze=True)
        self.assertEqual(result["scientific_coverage_status"], "UNDEFINED")
        self.assertIs(result["retention_eligible"], False)
        self.assertIs(result["coverage_complete"], False)

    def test_zero_or_missing_support_cannot_claim_eligible_complete_retention(self):
        for claim in ("retention_eligible", "coverage_complete", "retention_qualified"):
            with self.subTest(claim=claim):
                self.setUp()
                for domain in self.plan["domains"]:
                    declare_domain(domain, source_correct_rows=0)
                self.refused(lambda: self.plan["claims"].update({claim: True}), "authority binding mismatch")

    def test_native_qualified_boolean_cannot_supply_a_binding(self):
        self.refused(lambda: self.domain()["qualification"].update(native_qualified=True), "exact fields")

    def test_integer_counts_refuse_bools_floats_negative_values(self):
        for value in (True, False, 1.0, -1):
            with self.subTest(value=value):
                self.setUp()
                self.refused(lambda: self.domain().update(candidate_rows=value), "integer required")
        self.setUp()
        declare_domain(self.domain())
        self.refused(lambda: self.domain().update(declared_source_correct_rows=True), "integer required")

    def test_declared_source_correct_rows_cannot_exceed_candidate_rows(self):
        declare_domain(self.domain(), source_correct_rows=11, candidate_rows=10)
        with self.assertRaisesRegex(registry.ValidationError, "exceed candidate"):
            self.validate(refreeze=True)

    def test_descriptor_role_substitution_refused(self):
        declare_domain(self.domain())
        self.refused(lambda: self.domain()["enrollment"].update(
            descriptor=deepcopy(self.domain()["population"]["descriptor"])), "descriptor-role substitution")

    def test_same_population_path_with_conflicting_domain_hashes_refused(self):
        declare_domain(self.domain("retrieval"))
        declare_domain(self.domain("abstention"))
        self.refused(lambda: self.domain("abstention")["population"]["descriptor"].update(
            path=self.domain("retrieval")["population"]["descriptor"]["path"].upper().replace("/", "\\")),
                     "conflicting hashes")

    def test_policy_and_domain_reference_path_hash_conflict_refused(self):
        self.domain()["scorer"] = declared("different_scorer")
        self.refused(lambda: self.domain()["scorer"]["descriptor"].update(
            path=self.policies["task"]["path"]), "conflicting hashes")

    def test_domain_and_disjointness_reference_hash_conflict_refused(self):
        self.domain()["scorer"] = declared("scorer")
        declaration = self.plan["disjointness"]["training_vs_short_guards"]["source"]
        declaration.update(status="DECLARED_UNVERIFIED", declared_disjoint=True,
                           descriptor=deepcopy(self.domain()["scorer"]["descriptor"]))
        self.refused(lambda: declaration["descriptor"].update(sha256="0" * 64), "conflicting hashes")

    def test_consistent_shared_scorer_reference_is_allowed_but_unverified(self):
        self.domain("retrieval")["scorer"] = declared("shared_scorer")
        self.domain("abstention")["scorer"] = deepcopy(self.domain("retrieval")["scorer"])
        result = self.validate(refreeze=True)
        self.assertFalse(result["receipt_contents_verified"])
        self.assertFalse(result["scoring_authorized"])
        self.assertIsNone(result["actual_support"])

    def test_pending_status_preserves_none_and_cannot_hide_binding(self):
        self.refused(lambda: self.domain()["scorer"].update(descriptor=reference("hidden")), "must remain None")

    def test_declared_status_requires_reference_but_never_allows_qualified_status(self):
        self.refused(lambda: self.domain()["scorer"].update(status="DECLARED_UNVERIFIED"), "exact fields")
        self.setUp()
        self.refused(lambda: self.domain()["qualification"].update(status="QUALIFIED"), "cannot claim qualification")

    def test_declared_source_parent_disjointness_stays_unverified(self):
        for pair in self.plan["disjointness"].values():
            for unit, declaration in pair.items():
                declaration.update(status="DECLARED_UNVERIFIED", declared_disjoint=True,
                                   descriptor=reference("disjointness_" + unit))
        result = self.validate(refreeze=True)
        self.assertFalse(result["source_disjointness_verified"])
        self.assertFalse(result["parent_disjointness_verified"])
        self.assertEqual(result["preserved_plan"]["disjointness"], self.plan["disjointness"])

    def test_known_source_or_parent_overlap_conflicts_refused(self):
        for unit in ("source", "parent"):
            with self.subTest(unit=unit):
                self.setUp()
                declaration = self.plan["disjointness"]["short_guards_vs_long_context"][unit]
                self.refused(lambda: declaration.update(status="DECLARED_UNVERIFIED", declared_disjoint=False),
                             "known overlap")

    def test_disjointness_cannot_claim_verified_or_bool_like_integer(self):
        declaration = self.plan["disjointness"]["training_vs_short_guards"]["source"]
        self.refused(lambda: declaration.update(status="VERIFIED", declared_disjoint=True), "never certify")
        self.setUp()
        declaration = self.plan["disjointness"]["training_vs_short_guards"]["parent"]
        self.refused(lambda: declaration.update(status="DECLARED_UNVERIFIED", declared_disjoint=1), "known overlap")

    def test_disjointness_scopes_cannot_be_omitted_or_invented(self):
        self.refused(lambda: self.plan["disjointness"].pop("short_guards_vs_long_context"), "exact fields")
        self.setUp()
        self.refused(lambda: self.plan["disjointness"].update(all_sources_disjoint=True), "exact fields")

    def test_all_authority_flags_remain_closed(self):
        for claim in registry.CLOSED_CLAIMS:
            with self.subTest(claim=claim):
                self.setUp()
                self.refused(lambda: self.plan["claims"].update({claim: True}), "authority binding mismatch")

    def test_aliased_input_claim_template_cannot_change_validation_or_output(self):
        original = dict(registry.CLOSED_CLAIMS)
        try:
            self.plan["claims"] = registry.CLOSED_CLAIMS
            self.plan["claims"]["native_admissible"] = True
            with self.assertRaisesRegex(registry.ValidationError, "authority binding mismatch"):
                self.validate(refreeze=True)
            self.plan["claims"] = registry.closed_claims()
            result = self.validate(refreeze=True)
            for claim in registry.CLAIM_FIELDS:
                self.assertIs(result[claim], False)
        finally:
            registry.CLOSED_CLAIMS.clear()
            registry.CLOSED_CLAIMS.update(original)

    def test_nonzero_counts_cannot_claim_complete_or_eligible_scientific_coverage(self):
        for domain in self.plan["domains"]:
            declare_domain(domain, source_correct_rows=10, candidate_rows=10)
        self.refused(lambda: self.plan.update(scientific_coverage_status="COMPLETE"), "authority binding mismatch")

    def test_unknown_fields_and_scorer_payloads_refused(self):
        self.refused(lambda: self.domain()["scorer"].update(labels=[True]), "exact fields")
        self.setUp()
        self.refused(lambda: self.plan.update(model_outputs=[]), "exact fields")

    def test_relative_aliased_reserved_paths_refused(self):
        for path in ("relative.json", "C:/owned/../scorer.json", "C:/owned/NUL.json", "C:/owned/COM¹.json"):
            with self.subTest(path=path):
                self.setUp()
                self.domain()["scorer"] = declared("scorer")
                self.refused(lambda: self.domain()["scorer"]["descriptor"].update(path=path))

    def test_malformed_reference_hash_and_nonjson_tuple_refused(self):
        self.domain()["scorer"] = declared("scorer")
        self.refused(lambda: self.domain()["scorer"]["descriptor"].update(sha256="pending"), "SHA256")
        self.setUp()
        self.refused(lambda: self.plan.update(required_short_domains=tuple(registry.SHORT_DOMAINS)), "finite closed JSON")

    def test_missing_bindings_are_not_rewritten_and_inputs_not_mutated(self):
        before = (registry.canonical(self.plan), registry.canonical(self.policies))
        result = self.validate()
        result["preserved_plan"]["domains"][0]["population"]["descriptor"] = reference("new")
        self.assertEqual(before, (registry.canonical(self.plan), registry.canonical(self.policies)))
        self.assertIsNone(self.validate()["preserved_plan"]["domains"][0]["population"]["descriptor"])


if __name__ == "__main__":
    unittest.main()
