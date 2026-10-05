"""Synthetic adversarial controls; these tests establish no native authority."""
from copy import copy, deepcopy
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import candidate_import_contract as contract
import fixture_support as fixtures


class CandidateImportControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="candidate_fixture_", dir=fixtures.HERE)
        self.root = Path(self.directory.name).resolve()
        self.scenario = fixtures.Scenario(contract, self.root)

    def tearDown(self):
        # Only this test's explicitly created staging child is removed.
        self.assertEqual(self.root.parent, fixtures.HERE.resolve())
        self.directory.cleanup()

    def altered_import(self, *, ordinal=1, inputs=None, outputs=None, restore=None, factors=None):
        """Keep audit/config bytes consistent so semantic controls face the mutation."""
        scenario = self.scenario
        original = scenario.batch_documents[ordinal - 1]
        inputs = deepcopy(original["inputs"] if inputs is None else inputs)
        outputs = deepcopy(original["outputs"] if outputs is None else outputs)
        restore = deepcopy(scenario.restore if restore is None else restore)
        factors = deepcopy(scenario.factors if factors is None else factors)
        identity = contract.value(scenario.selection, "selection")["identity"]["sha256"]
        refs = fixtures.batch_refs(self.root / "replacement", scenario.arm, ordinal,
                                   inputs, outputs, identity, restore, factors)
        changed = contract.fixture_batch(scenario.selection, ordinal, inputs, outputs,
                                         restore, factors, refs, root=self.root)
        batches = list(scenario.batches)
        batches[ordinal - 1] = changed
        post = contract.fixture_import_selection(scenario.selection, batches, root=self.root)
        return scenario.prepare(batches=batches, import_selection=post)

    def validate_changed_chain(self, chain, latest=None):
        return contract.validate_study_chain(self.scenario.registration, "M1", chain,
            chain[-1]["terminal_costs"] if latest is None else latest, root=self.root)

    def test_complete384_fixture_preserves_closed_permissions_and_both_finishes(self):
        packet = self.scenario.prepare()
        self.assertEqual(packet["kind"], "CPU_SYNTHETIC_CANDIDATE_IMPORT_FIXTURE")
        self.assertEqual(len(packet["rows"]), 384)
        self.assertEqual(len({row["row_id"] for row in packet["rows"]}), 384)
        self.assertEqual(len({row["component_id"] for row in packet["rows"]}), 96)
        self.assertEqual({row["raw"]["finish_reason"] for row in packet["rows"]}, {"eos", "generation_cap"})
        for name in ("native_launch_authorized", "scoring_authorized", "training_authorized",
                     "retention_certified", "confirmation_eligible", "protected_bank_access",
                     "paid_services", "quality_gain_established", "native_candidate_import_admitted", "study_completed"):
            self.assertIs(packet[name], False)
        self.assertEqual(packet["sha256"], fixtures.digest({k: v for k, v in packet.items() if k != "sha256"}))
        self.assertEqual(len(self.scenario.chain[0]), 11)
        self.assertEqual(len(self.scenario.chain[0]["envelope"]), 12)
        self.assertEqual(len(self.scenario.chain[0]["envelope"]["backend_descriptor"]["metadata"]), 17)
        self.assertEqual(len(self.scenario.restore), 23)
        self.assertEqual(len(self.scenario.batch_documents[0]["outputs"][0]), 30)

    def test_separate_M2_fixture_has_its_own_candidate_identity(self):
        second = fixtures.Scenario(contract, self.root / "second_arm", arm="M2")
        packet = second.prepare()
        self.assertEqual(packet["arm"], "M2")
        self.assertNotEqual(packet["candidate_identity"]["sha256"],
                            contract.value(self.scenario.selection, "selection")["identity"]["sha256"])

    def test_missing_duplicate_or_reordered_batches_cannot_import(self):
        scenario = self.scenario
        for batches in (scenario.batches[:-1],
                        [scenario.batches[0]] * 6,
                        [scenario.batches[1], scenario.batches[0], *scenario.batches[2:]]):
            with self.subTest(length=len(batches)), self.assertRaises(ValueError):
                scenario.prepare(batches=batches)

    def test_missing_or_extra_row_cannot_change_denominator(self):
        original = self.scenario.batch_documents[0]["outputs"]
        for outputs in (original[:-1], original + [deepcopy(original[-1])]):
            with self.subTest(rows=len(outputs)), self.assertRaises(ValueError):
                self.altered_import(outputs=outputs)

    def test_prompt_replacement_or_oracle_field_is_rejected(self):
        original = self.scenario.batch_documents[0]
        inputs = deepcopy(original["inputs"])
        inputs[0]["prompt_ids"][0] += 1
        with self.assertRaises(ValueError):
            self.altered_import(inputs=inputs, outputs=fixtures.raw_outputs(inputs))
        inputs = deepcopy(original["inputs"])
        inputs[0]["oracle_answer"] = "synthetic forbidden fixture field"
        with self.assertRaises(ValueError):
            self.altered_import(inputs=inputs, outputs=fixtures.raw_outputs(inputs))

    def test_trace_prefix_uses_zero_based_pre_generation_position(self):
        outputs = deepcopy(self.scenario.batch_documents[0]["outputs"])
        outputs[0]["trace"][0]["prefix_length"] += 1
        with self.assertRaises(ValueError):
            self.altered_import(outputs=outputs)

    def test_boolean_token_positive_logprob_or_trace_array_divergence_is_rejected(self):
        for mutation in (
            lambda raw: raw["raw_generated_ids"].__setitem__(0, True),
            lambda raw: raw["selected_logprobs"].__setitem__(0, 0.25),
            lambda raw: raw["trace"][0].update(token=42),
        ):
            outputs = deepcopy(self.scenario.batch_documents[0]["outputs"])
            mutation(outputs[0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.altered_import(outputs=outputs)

    def test_boolean_trace_logprob_cannot_alias_numeric_zero(self):
        outputs = deepcopy(self.scenario.batch_documents[0]["outputs"])
        outputs[0]["selected_logprobs"][0] = 0.0
        outputs[0]["trace"][0]["logprob"] = False
        with self.assertRaises(ValueError):
            self.altered_import(outputs=outputs)

    def test_cumulative_capture_costs_cannot_rollback_or_omit_positions(self):
        for key in ("attempted_forwards", "input_positions"):
            outputs = deepcopy(self.scenario.batch_documents[0]["outputs"])
            if key == "attempted_forwards":
                outputs[1]["costs"][key] -= 1
            else:
                outputs[1]["costs"][key]["returned"] -= 1
            with self.subTest(cost=key), self.assertRaises(ValueError):
                self.altered_import(outputs=outputs)

    def test_finish_laundering_or_early_EOS_is_rejected(self):
        for mutation in (
            lambda outputs: outputs[0].update(truncated=True),
            lambda outputs: outputs[1]["raw_generated_ids"].__setitem__(0, 151645),
            lambda outputs: outputs[1]["raw_generated_ids"].pop(),
        ):
            outputs = deepcopy(self.scenario.batch_documents[0]["outputs"])
            mutation(outputs)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.altered_import(outputs=outputs)

    def test_failed_restore_or_missing_factor_copy_cannot_be_admitted(self):
        for mutation in (
            lambda result: result.update(status="FAILED_RESTORE_MODEL_INVALID_DO_NOT_COLLECT", inference_allowed=False),
            lambda result: result["operation_costs"].update(final_adapter_parameter_copy=fixtures.count(111)),
            lambda result: result.update(optimizer_restored=True),
        ):
            restore = deepcopy(self.scenario.restore)
            mutation(restore)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.altered_import(restore=restore)

    def test_factor_byte_drift_or_FP32_shape_substitution_is_rejected(self):
        first = next(iter(fixtures.factor_shapes()))
        for mutation in (
            lambda item: item.update(sha256=fixtures.digest("different synthetic factor bytes")),
            lambda item: item.update(bytes=item["bytes"] // 2),
            lambda item: item.update(shape=[1, item["shape"][0] * item["shape"][1]]),
        ):
            factors = deepcopy(self.scenario.factors)
            observation = factors["resident_after_capture"]
            mutation(observation["parameters"][first])
            observation.update(fixtures.seal(observation))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.altered_import(factors=factors)

    def test_all_resealed_factor_maps_cannot_replace_selected_archive_observation(self):
        factors = deepcopy(self.scenario.factors)
        first = next(iter(fixtures.factor_shapes()))
        for observation in factors.values():
            observation["parameters"][first]["sha256"] = fixtures.digest("replacement synthetic archive factor")
            observation.update(fixtures.seal(observation))
        with self.assertRaises(ValueError):
            self.altered_import(factors=factors)

    def test_full27_recipe_cannot_be_laundered_into_study_chain(self):
        chain = deepcopy(self.scenario.chain)
        for resume in chain:
            envelope = resume["envelope"]
            envelope["recipe"]["purpose"] = "full27_three_fresh_process_nine_update_qualification_only"
            envelope["recipe"].update(fixtures.seal(envelope["recipe"], study=True))
            descriptor = envelope["backend_descriptor"]
            descriptor["metadata"]["recipe_sha256"] = envelope["recipe"]["sha256"]
            fixtures.refresh_checkpoint(descriptor)
            fixtures.refresh_resume(resume)
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_source_recipe_pin_drift_is_rejected_even_when_resealed(self):
        chain = deepcopy(self.scenario.chain)
        for resume in chain:
            envelope = resume["envelope"]
            envelope["recipe"]["chunk_source_sha256"] = fixtures.digest("different synthetic source")
            envelope["recipe"].update(fixtures.seal(envelope["recipe"], study=True))
            descriptor = envelope["backend_descriptor"]
            descriptor["metadata"]["recipe_sha256"] = envelope["recipe"]["sha256"]
            fixtures.refresh_checkpoint(descriptor)
            fixtures.refresh_resume(resume)
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_archive_parameter_order_is_exact_not_only112_unique_names(self):
        chain = deepcopy(self.scenario.chain)
        descriptor = chain[-1]["envelope"]["backend_descriptor"]
        descriptor["metadata"]["adapter_parameter_order"].reverse()
        fixtures.refresh_checkpoint(descriptor)
        fixtures.refresh_resume(chain[-1])
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_resealed_archive_inventory_digest_must_match_source_inventory(self):
        chain = deepcopy(self.scenario.chain)
        descriptor = chain[-1]["envelope"]["backend_descriptor"]
        descriptor["metadata"]["adapter_inventory_sha256"] = fixtures.digest("fabricated source inventory", study=True)
        fixtures.refresh_checkpoint(descriptor)
        fixtures.refresh_resume(chain[-1])
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_resealed_fresh_start_attestation_cannot_change_between_chunks(self):
        chain = deepcopy(self.scenario.chain)
        descriptor = chain[-1]["envelope"]["backend_descriptor"]
        descriptor["metadata"]["fresh_start_attestation_sha256"] = fixtures.digest("different synthetic fresh start")
        fixtures.refresh_checkpoint(descriptor)
        fixtures.refresh_resume(chain[-1])
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_inert_checkpoint_embedded_metadata_cannot_disagree_with_descriptor(self):
        chain = deepcopy(self.scenario.chain)
        descriptor = chain[-1]["envelope"]["backend_descriptor"]
        path = Path(descriptor["path"])
        embedded = json.loads(path.read_text(encoding="utf-8"))
        embedded["metadata"]["arm"] = "M2"
        fixtures.write_json(path, embedded)
        changed_anchor = fixtures.raw_file_anchor(path)
        descriptor["bytes"] = changed_anchor["bytes"]
        descriptor["sha256"] = changed_anchor["file_sha256"]
        fixtures.refresh_resume(chain[-1])
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_terminal_cost_rollback_and_erased_prior_primitive_are_rejected(self):
        with self.assertRaises(ValueError):
            self.validate_changed_chain(deepcopy(self.scenario.chain), self.scenario.chain[1]["terminal_costs"])
        chain = deepcopy(self.scenario.chain)
        del chain[1]["envelope"]["posted_save_costs"]["primitive_operations"]["fixture_source_metadata_read"]
        fixtures.refresh_resume(chain[1], recompute_costs=True)
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_changed_final_cursor_cannot_skip_update27(self):
        chain = deepcopy(self.scenario.chain)
        chain[-1]["envelope"]["binding"]["next_update_ordinal"] = 27
        fixtures.refresh_resume(chain[-1])
        with self.assertRaises(ValueError):
            self.validate_changed_chain(chain)

    def test_postcapture_config_or_audit_replacement_is_rejected(self):
        for role in ("config", "audit"):
            # Each role is checked before any output, using the original selected file hash.
            path = Path(self.scenario.batch_documents[0]["refs"][role]["path"])
            original = path.read_bytes()
            changed = json.loads(original)
            changed["ordinal"] = 2
            fixtures.write_json(path, fixtures.seal(changed))
            try:
                with self.subTest(role=role), self.assertRaises(ValueError):
                    self.scenario.prepare()
            finally:
                path.write_bytes(original)

    def test_postcapture_manifest_cannot_be_rebased_to_replacement_batch(self):
        scenario = self.scenario
        document = scenario.batch_documents[0]
        identity = contract.value(scenario.selection, "selection")["identity"]["sha256"]
        refs = fixtures.batch_refs(self.root / "alternative_selection", "M1", 1,
            document["inputs"], document["outputs"], identity, scenario.restore, scenario.factors)
        replacement = scenario.batch(1, refs=refs)
        with self.assertRaises(ValueError):
            scenario.prepare(batches=[replacement, *scenario.batches[1:]])

    def test_extra_owned_directory_file_is_not_silently_ignored(self):
        directory = Path(self.scenario.batch_documents[0]["refs"]["config"]["path"]).parent
        fixtures.write_json(directory / "unexpected_fixture.json", {"fixture": True})
        with self.assertRaises(ValueError):
            self.scenario.prepare()

    def test_dict_copy_serialized_or_attribute_clone_is_not_factory_authority(self):
        scenario = self.scenario
        with self.assertRaises(ValueError):
            scenario.prepare(population=contract.value(scenario.population, "population"))
        for operation in (copy, deepcopy, pickle.dumps, json.dumps):
            with self.subTest(operation=operation), self.assertRaises((TypeError, ValueError)):
                operation(scenario.population)
        clone = object.__new__(type(scenario.population))
        for slot in ("_value", "_hash", "_readset", "_kind"):
            setattr(clone, slot, getattr(scenario.population, slot))
        with self.assertRaises(ValueError):
            scenario.prepare(population=clone)

    def test_modified_handle_is_detected_before_import(self):
        self.scenario.population._value["scope"] = "fabricated native scope"
        with self.assertRaises(ValueError):
            self.scenario.prepare()

    def test_modified_handle_cannot_reseal_its_mutable_private_hash(self):
        self.scenario.population._value["scoring_authorized"] = True
        self.scenario.population._hash = fixtures.digest(self.scenario.population._value)
        with self.assertRaises(ValueError):
            self.scenario.prepare()

    def test_strict_parser_rejects_duplicate_keys_constants_and_overflow(self):
        path = self.root / "parser" / "fixture.json"
        path.parent.mkdir(parents=True)
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
                    b'{"ignored":{"value":1e999}}'):
            path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contract.read_json(path)

    def test_JSONL_requires_complete_bounded_lines_and_exact_line_cap(self):
        path = self.root / "parser" / "fixture.jsonl"
        path.parent.mkdir(parents=True)
        for raw, limit in ((b'{"a":1}', 2), (b'{"a":1}\n{"a":2}\n', 1),
                           (b'{"a":"' + b'x' * 65536 + b'"}\n', 2)):
            path.write_bytes(raw)
            with self.subTest(limit=limit, bytes=len(raw)), self.assertRaises(ValueError):
                contract.read_jsonl(path, max_lines=limit)

    def test_distinct_ASCII_and_UTF8_role_hash_conventions(self):
        value = {"fixture_unicode": "r\u00e9sum\u00e9", "sha256": fixtures.digest("unsealed field")}
        path = fixtures.write_json(self.root / "parser" / "unicode.json", value, study=True)
        self.assertNotEqual(fixtures.digest(value), fixtures.digest(value, study=True))
        for convention in ("ASCII_UNSEALED_FULL_DOCUMENT", "UTF8_STUDY_UNSEALED_FULL_DOCUMENT"):
            self.assertEqual(contract.json_anchor(path, convention), fixtures.json_anchor(path, convention))

    def test_native_entry_points_close_before_file_reads_and_no_framework_import_occurs(self):
        forbidden = {"torch", "transformers", "jax", "tensorflow", "flax"}
        self.assertFalse(forbidden.intersection(name.split(".", 1)[0] for name in sys.modules))
        with patch.object(contract, "file_bytes", side_effect=AssertionError("native path was read")):
            for function in (contract.load_native_selection, contract.load_native_candidate_batch):
                with self.subTest(function=function), self.assertRaises(ValueError):
                    function(self.root / "missing_native_receipt.json")


if __name__ == "__main__":
    unittest.main()
