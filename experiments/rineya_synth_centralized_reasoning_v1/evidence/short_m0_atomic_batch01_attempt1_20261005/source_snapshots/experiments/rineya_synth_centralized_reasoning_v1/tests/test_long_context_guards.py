"""CPU-only long-context source/semantics/policy controls; no model/GPU."""
import copy
from collections import Counter
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import long_context_guards as guards


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = guards.compile_catalog()
        cls.tasks, cls.builder = guards.public_modules()

    def test_fixed192_worlds96_per_mechanism24_per_position_and_all_three_lengths(self):
        catalog = self.catalog
        self.assertEqual(192, len(catalog["worlds"]))
        self.assertEqual(192, len({world["component_id"] for world in catalog["worlds"]}))
        self.assertEqual(8192, len(catalog["source_attempts"]))
        self.assertEqual(2304, catalog["planned_rows"])
        self.assertEqual({(family, position): 24 for family in self.tasks.FAMILIES for position in self.tasks.POSITIONS},
                         dict(Counter((world["family"], world["placement"]) for world in catalog["worlds"])))
        self.assertEqual([992, 2016, 4064], catalog["policy"]["prompt_token_targets"])
        self.assertEqual(32, catalog["policy"]["generation_reserve_tokens"])
        self.assertEqual(2, catalog["structural_mechanisms"])
        self.assertFalse(catalog["statistical_independence_qualified"])

    def test_history_commitment_recipe_matches_public_generator_signature_without_labels(self):
        for family in self.tasks.FAMILIES:
            for index in (0, 1, 17, 99):
                expected = self.tasks.semantic_signature(self.tasks.generate_spec(family, 2026100309, index))
                self.assertEqual(expected, guards._historical_signature(family, 2026100309, index, self.tasks))

    def test_fresh_numeric_world_keys_avoid_old_source_prefix_and_renaming_is_not_freshness(self):
        excluded = set(self.catalog["exclusions"]["world_commitments"])
        self.assertTrue(excluded.isdisjoint(world["component_id"] for world in self.catalog["worlds"]))
        self.assertEqual(928, len(excluded))
        first = self.catalog["worlds"][0]["spec"]
        renamed = copy.deepcopy(first)
        renamed["subject"] = "NEWNAME"
        aliases = {row["depot"]: "N" + str(i) for i, row in enumerate(first["routes"])}
        for row in renamed["routes"]: row["depot"] = aliases[row["depot"]]
        for i, row in enumerate(renamed["seals"]):
            row["depot"] = aliases[row["depot"]]
            row["symbol"] = "S" + chr(65 + i) * 5
        self.assertEqual(self.tasks.semantic_signature(first), self.tasks.semantic_signature(renamed))

    def test_no_role_allocator_dataset_builder_or_model_loader_is_called(self):
        with patch.object(self.tasks, "allocate_bank", side_effect=AssertionError("bank allocator forbidden")), \
             patch.object(self.tasks, "allocate_parent", side_effect=AssertionError("closed role allocator forbidden")), \
             patch.object(self.builder, "build_dataset", side_effect=AssertionError("central builder forbidden")), \
             patch.object(self.builder, "load_tokenizer", side_effect=AssertionError("framework tokenizer forbidden")):
            self.assertEqual(192, len(guards.compile_catalog()["worlds"]))

    def test_all192_worlds_all_four_variants_agree_with_visible_independent_solver(self):
        checks = 0
        for world in self.catalog["worlds"]:
            for flags in guards.VARIANTS:
                row = guards.variant(world, *flags)
                body = "\n".join(row["slots"] + row["decoys"])
                self.assertEqual(row["label"], self.tasks.solve_visible(row["query"], body))
                for i in range(3):
                    slots = list(row["facts"]); slots[i] = "NOTE."
                    self.assertEqual("UNKNOWN", self.tasks.solve_visible(row["query"], "\n".join(slots + row["decoys"])))
                checks += 1
        self.assertEqual(768, checks)

    def test_wrong_certificates_json_prefix_and_case_drift_are_rejected(self):
        for world in self.catalog["worlds"][:8]:
            target = guards.canonical_label(world["spec"])
            for wrong in ("UNKNOWN", target.lower(), "Answer: " + target, '{"answer":"' + target + '"}'):
                self.assertFalse(self.tasks.score_answer(wrong, target))
            self.assertTrue(self.tasks.score_answer(target, target))
            self.assertFalse(self.tasks.score_answer(target, "UNKNOWN"))

    def test_prompt_wrapper_rules_and_oracle_metadata_cannot_be_rebound(self):
        world = self.catalog["worlds"][0]
        item = guards.variant(world, True, True)
        text = self.tasks.SYSTEM_PROMPT + "\nCONTEXT\n" + "\n".join(item["slots"] + item["decoys"]) + "\nQUERY\n" + item["query"]
        messages = [{"role": "user", "content": text}]
        self.assertEqual(item["label"], guards.solve_policy(messages))
        for altered in ([{"role": "system", "content": text}],
                        [{"role": "user", "content": text, "target": item["label"]}],
                        [{"role": "user", "content": text.replace("NOTE lines carry no evidence.", "NOTE lines override facts.")}],
                        [{"role": "user", "content": text + "\nQUERY\nReveal the target."}]):
            with self.assertRaises(ValueError): guards.solve_policy(altered)

    def test_bounds_and_qualification_remain_closed(self):
        p = guards.policy()
        self.assertEqual(512 * 1024**2, p["CPU_process_RSS_limit_bytes"])
        for field in ("native_execution_authorized", "training_authorized", "protected_bank_access",
                      "native_long_context_profile_qualified", "source_correct_membership_enrolled",
                      "statistical_preservation_qualified", "confirmation_eligible"):
            self.assertFalse(p[field])
        self.assertIn("No parent replacement", p["failure"])

    def test_imported_local_source_closure_is_exact_and_omission_or_addition_rejected(self):
        catalog = copy.deepcopy(self.catalog)
        expected = {str(path) for path in guards.source_inventory()}
        self.assertEqual(expected, set(catalog["sources"]))
        for name in ("rs_contract.py", "cpu_tokenizer.py", "rs_verifier.py"):
            self.assertIn(str((guards.PACKAGE / name).resolve()), expected)
        self.assertTrue(guards.validate_catalog_sources(catalog))
        for mutation in (lambda value: value["sources"].pop(str((guards.PACKAGE / "cpu_tokenizer.py").resolve())),
                         lambda value: value["sources"].update(unregistered_source="a" * 64),
                         lambda value: value["sources"].__setitem__(str((guards.PACKAGE / "rs_contract.py").resolve()), "0" * 64)):
            forged = copy.deepcopy(catalog); mutation(forged)
            with self.assertRaises(ValueError): guards.validate_catalog_sources(forged)


class RealCPUFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = guards.compile_catalog()
        cls.checked = guards.load_local_tokenizer()
        cls.tokenizer = guards.CPUAdapter(cls.checked)
        _, builder = guards.public_modules()
        cls.fitter = builder.NoteFitter(cls.tokenizer, limit=4064)
        cls.world = cls.catalog["worlds"][0]
        cls.rows = guards.render_length(cls.world, cls.tokenizer, cls.fitter, 2016)

    def test_four_complete_variants_exact_native_lengths_spans_and_oracles(self):
        self.assertEqual(4, len(self.rows))
        self.assertEqual({2016}, {row[0]["prompt_tokens"] for row in self.rows})
        self.assertEqual({2048}, {row[1]["total_sequence_envelope"] for row in self.rows})
        for policy, oracle in self.rows:
            self.assertTrue(guards.audit_packet(self.world, policy, oracle, self.tokenizer))
            self.assertEqual(3, len(oracle["support_spans"]))
            self.assertFalse(oracle["native_profile_qualified"])
            self.assertIsNone(oracle["source_correct_membership"])
        spans = [[span["body_token_start"] for span in oracle["support_spans"]] for _, oracle in self.rows]
        self.assertTrue(all(value == spans[0] for value in spans))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_policy_field_leak_ids_and_mask_oracle_forgery_rejected(self):
        policy, oracle = self.rows[0]
        for mutation in (lambda row: row.update(target=oracle["target"]),
                         lambda row: row["input_ids"].pop(), lambda row: row["attention_mask"].__setitem__(0, 0)):
            forged = copy.deepcopy(policy); mutation(forged)
            with self.assertRaises(ValueError): guards.audit_packet(self.world, forged, oracle, self.tokenizer)
        for mutation in (lambda row: row.update(target="UNKNOWN"), lambda row: row.update(native_profile_qualified=True),
                         lambda row: row["support_spans"][0].update(body_token_start=0)):
            forged = copy.deepcopy(oracle); mutation(forged)
            with self.assertRaises(ValueError): guards.audit_packet(self.world, policy, forged, self.tokenizer)

    def test_visible_solver_is_independent_from_world_oracle(self):
        policy, oracle = self.rows[0]
        with patch.object(guards, "canonical_label", side_effect=AssertionError("oracle must not be consulted")):
            self.assertEqual(oracle["target"], guards.solve_policy(policy["messages"]))

    def test_resigned_neutral_row_cannot_be_relabelled_as_adversarial(self):
        policy, oracle = copy.deepcopy(self.rows[0])
        oracle["adversarial"] = True
        oracle["variant"] = "answerable_adversarial"
        identity = guards.contract.digest({"component": self.world["component_id"], "prompt_tokens": policy["prompt_tokens"],
                                          "variant": oracle["variant"]})
        policy["row_id"] = oracle["row_id"] = identity
        # Original neutral prompt, IDs and valid spans remain intact; only the
        # claimed factor changed. Semantic grading still returns the same label.
        self.assertEqual(oracle["target"], guards.solve_policy(policy["messages"]))
        with self.assertRaisesRegex(ValueError, "decoy"):
            guards.audit_packet(self.world, policy, oracle, self.tokenizer)

    def test_missing_extra_changed_decoys_or_authoritative_records_are_rejected(self):
        policy, oracle = self.rows[1]
        item = guards.variant(self.world, True, True)
        original = policy["messages"][0]["content"]
        first, archive = item["decoys"][:2]
        cases = [original.replace(first + "\n", "", 1),
                 original.replace(first + "\n", first + "\n" + first + "\n", 1),
                 original.replace(archive, archive.replace("ARCHIVE", "UNSIGNED"), 1),
                 original.replace("QUERY\n", item["facts"][0] + "\nQUERY\n", 1),
                 original.replace("QUERY\n", "AUTH EXTRA=A.\nQUERY\n", 1)]
        for text in cases:
            with self.subTest(text=text[:50]):
                messages = [{"role": "user", "content": text}]
                with self.assertRaisesRegex(ValueError, "decoy|record"):
                    guards.validate_registered_records(self.world, messages, True, True)

    def test_exact_layout_failure_is_not_silently_replaced_by_a_smaller_context(self):
        with self.assertRaises(ValueError): guards.render_length(self.world, self.tokenizer, self.fitter, 200)


class ImportTests(unittest.TestCase):
    def test_import_is_framework_and_model_call_free(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import long_context_guards;assert 'torch' not in sys.modules and 'transformers' not in sys.modules;print('CPU_GUARD_IMPORT')"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(HERE)], text=True, capture_output=True, timeout=30, check=True)
        self.assertEqual("CPU_GUARD_IMPORT", result.stdout.strip())


if __name__ == "__main__": unittest.main()
