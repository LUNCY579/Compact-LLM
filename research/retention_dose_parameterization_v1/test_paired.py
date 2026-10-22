"""Offline checks: exact paired arithmetic and evidence gates; no model imports."""

import copy
import itertools
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

if __package__:
    from . import paired
else:
    import paired


class ExactPairedTests(unittest.TestCase):
    def test_matches_enumerated_random_sign_null(self):
        # Enumerate the null sample space, independently of binomial coefficients.
        for nd in range(11):
            signs = list(itertools.product((-1, 1), repeat=nd))
            for wins in range(nd + 1):
                observed_abs_net = abs(2 * wins - nd)
                tail = sum(abs(sum(signs_i)) >= observed_abs_net for signs_i in signs)
                self.assertEqual(paired.exact_mcnemar_fraction(wins, nd - wins),
                                 Fraction(tail, len(signs)))

    def test_identical_rows_and_smallest_significant_pair(self):
        self.assertEqual(paired.exact_mcnemar(0, 0), 1.0)
        self.assertEqual(paired.exact_mcnemar(5, 0), 0.0625)
        self.assertEqual(paired.exact_mcnemar(6, 0), 0.03125)
        self.assertIsNone(paired.minimum_detectable_net(0))
        self.assertIsNone(paired.minimum_detectable_net(5))
        self.assertEqual(paired.minimum_detectable_net(6), 6)

    def test_known_control_retraining_contrasts(self):
        self.assertAlmostEqual(paired.exact_mcnemar(36, 31), 0.6254068289416487)
        self.assertAlmostEqual(paired.exact_mcnemar(39, 33), 0.5559977177609708)
        self.assertEqual(paired.exact_mcnemar(43, 42), 1.0)
        for nd, expected in ((67, 17), (72, 18), (85, 21), (384, 40)):
            with self.subTest(nd=nd):
                actual = paired.minimum_detectable_net(nd)
                self.assertEqual(actual, expected)
                self.assertEqual(actual % 2, nd % 2)
                self.assertLessEqual(paired.exact_mcnemar((nd + actual) // 2,
                                                        (nd - actual) // 2), 0.05)
                self.assertGreater(paired.exact_mcnemar((nd + actual - 2) // 2,
                                                       (nd - actual + 2) // 2), 0.05)

    def test_six_row_endpoint_floor_is_not_nd47_threshold(self):
        self.assertEqual(paired.minimum_detectable_net(47), 15)
        self.assertEqual(paired.minimum_detectable_net(6), 6)

    def test_rows_are_keyed_and_direction_is_candidate_minus_reference(self):
        ref = {"a": True, "b": False, "c": False, "d": True}
        cand = {"d": False, "c": True, "b": True, "a": True}
        result = paired.paired_boolean_summary(ref, cand, relationship="same_training_draw")
        self.assertEqual((result["wins"], result["losses"], result["net"], result["nd"]),
                         (2, 1, 1, 3))
        self.assertEqual((result["reference_correct"], result["candidate_correct"]), (2, 3))

    def test_refuses_invalid_statistics_inputs(self):
        for wins, losses in ((True, 1), (-1, 3), (2, 1.5)):
            with self.subTest(wins=wins, losses=losses), self.assertRaises(ValueError):
                paired.exact_mcnemar(wins, losses)
        for alpha in (0, 1, -0.1, True, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                paired.minimum_detectable_net(10, alpha)
        for left, right in (({}, {}), ({"a": True}, {"b": True}),
                            ({"a": 1}, {"a": True}), ({1: True}, {1: False})):
            with self.subTest(left=left), self.assertRaises(ValueError):
                paired.paired_boolean_summary(left, right, relationship="same_training_draw")


class AcquisitionBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.views = self.root / "views.jsonl"
        self.views.write_text("".join(json.dumps({"id": str(i)}) + "\n" for i in range(6)),
                              encoding="utf-8")
        self.left = self.make_arm("reference", [False] * 6)
        self.right = self.make_arm("candidate", [True] * 6)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_arm(self, name, values):
        arm_dir = self.root / name
        arm_dir.mkdir()
        weight = arm_dir / "model.safetensors"
        weight.write_bytes(name.encode("ascii"))
        weight_sha = paired.sha256_file(weight)
        provenance = {"source_weight_sha256": "c" * 64, "checkpoint_manifest_sha256": "d" * 64}
        self.write_json(arm_dir / "ARM.json", {"arm_weight_sha256": weight_sha, **provenance})
        self.write_json(arm_dir / "MATERIALIZATION.json", {"out_weight_sha256": weight_sha, **provenance})
        for sidecar in paired.SIDE_CARS:
            (arm_dir / sidecar).write_bytes(b"fixture sidecar")
        artifact = arm_dir / "PROCESS.json"
        raw = {"record": "PROCESS_TRANSITION_PROBE", "model_dir": str(arm_dir),
               "instrument": "fixture", "implementation_digest": "a" * 64,
               "views_sha256": paired.sha256_file(self.views), "views": len(values),
               "max_new_tokens": 96, "numerics": {"tf32": False}, "runtime": {"python": "fixture"},
               "aggregate": {"canonical_correct": 999999999},  # Never trust aggregate totals.
               "predictions": [{"id": str(i), "canonical": value} for i, value in enumerate(values)]}
        self.write_json(artifact, raw)
        binding = {"schema": 1, "status": "COMPLETE", "complete": True, "returncode": 0, "kind": "process",
                   "arm_dir": str(arm_dir), "weight_path": str(weight),
                   "weight_sha256": weight_sha, "weight_sha256_before": weight_sha,
                   "weight_sha256_after": weight_sha, "implementation_digest": "a" * 64,
                   "additional_input_sha256_before": {}, "additional_input_sha256_after": {}}
        for stem, path in (("raw_result", artifact), ("arm_receipt", arm_dir / "ARM.json"),
                           ("materialization", arm_dir / "MATERIALIZATION.json"), ("views", self.views)):
            binding[stem + "_path"] = str(path)
            binding[stem + "_sha256"] = paired.sha256_file(path)
        for stem in ("script", "wrapper"):
            path = arm_dir / (stem + ".txt")
            path.write_text(stem, encoding="utf-8")
            binding[stem + "_path"] = str(path)
            binding[stem + "_sha256"] = paired.sha256_file(path)
        supervisor_path = arm_dir / "supervisor.json"
        self.write_json(supervisor_path, {"status": "EXITED", "returncode": 0})
        terminal_path = arm_dir / "probe_terminal.json"
        self.write_json(terminal_path, {"status": "COMPLETED", "complete": True,
                         "system_exit_code": 0, "failure": None, "script_unchanged": True,
                         "script_sha256_after": binding["script_sha256"],
                         "script_hash_error": None, "result": str(artifact)})
        for stem, path in (("supervisor", supervisor_path), ("terminal", terminal_path)):
            binding[stem + "_path"] = str(path)
            binding[stem + "_sha256"] = paired.sha256_file(path)
        snapshot = {name: paired.sha256_file(arm_dir / name) for name in paired.SIDE_CARS}
        binding["sidecar_sha256_before"] = snapshot
        binding["sidecar_sha256_after"] = copy.deepcopy(snapshot)
        binding_path = arm_dir / "PROBE_WEIGHT_BINDING.json"
        self.write_json(binding_path, binding)
        return artifact, binding_path

    def compare(self, relationship="same_training_draw"):
        return paired.compare_acquisition(self.left[0], self.right[0], relationship=relationship)

    def mutate_binding(self, mutate):
        path = self.right[1]
        data = json.loads(path.read_text())
        mutate(data)
        self.write_json(path, data)

    def mutate_raw(self, mutate):
        path = self.right[0]
        data = json.loads(path.read_text())
        mutate(data)
        self.write_json(path, data)
        self.mutate_binding(lambda value: value.update(raw_result_sha256=paired.sha256_file(path)))

    def mutate_bound_file(self, stem, mutate):
        binding = json.loads(self.right[1].read_text())
        path = Path(binding[stem + "_path"])
        value = json.loads(path.read_text())
        mutate(value)
        self.write_json(path, value)
        self.mutate_binding(lambda data: data.update({stem + "_sha256": paired.sha256_file(path)}))

    def mutate_provenance(self, key, value):
        for stem in ("arm_receipt", "materialization"):
            self.mutate_bound_file(stem, lambda data: data.update({key: value}))

    def convert_candidate_kind(self, kind):
        raw = json.loads(self.right[0].read_text())
        raw["evidence_kind"] = {"retention": "exploratory_typed_retention_probe_only",
                                "long_context": "exploratory_long_context_retention_probe_only"}[kind]
        extra = {}
        arm_dir = self.right[0].parent
        if kind == "retention":
            raw.update(target="hf:" + str(arm_dir), reference_digest=raw["implementation_digest"],
                       probe_sha256=raw["views_sha256"], probe_rows=6)
            for name in ("reference.json", "label_audit.json"):
                path = arm_dir / name
                self.write_json(path, {"fixture": name})
                extra[str(path)] = paired.sha256_file(path)
            predictions_path = arm_dir / "retention_predictions.json"
            self.write_json(predictions_path, {"scope": "UNCHANGED_LEGACY_GENERATOR_RETURN",
                            "scorer_modified": False, "rows": 6,
                            "predictions": [{"id": str(i), "correct": True} for i in range(6)]})
            self.mutate_binding(lambda data: data.update(retention_predictions_path=str(predictions_path),
                                retention_predictions_sha256=paired.sha256_file(predictions_path)))
        else:
            raw.update(model_directory=str(arm_dir), checkpoint_applied=None, rows_total=6,
                       predictions={str(i): {"id": str(i), "prediction": "fixture"} for i in range(6)},
                       per_row_correct={str(i): True for i in range(6)})
            answer_key = arm_dir / "answer_key.jsonl"
            answer_key.write_bytes(self.views.read_bytes())
            extra[str(answer_key)] = paired.sha256_file(answer_key)
        self.mutate_raw(lambda data: data.update(raw))
        self.mutate_binding(lambda data: data.update(kind=kind, additional_input_sha256_before=extra,
                                                     additional_input_sha256_after=copy.deepcopy(extra)))

    def validate_candidate(self, kind):
        return paired.validate_probe_binding(*self.right, kind)

    def assert_blocked(self, result=None):
        if result is None:
            result = self.compare()
        self.assertEqual(result["status"], paired.BLOCKED, result)
        self.assertNotIn("paired", result)
        for forbidden in ("999999999", "reference_correct", "candidate_correct", "exact_p", '"net"'):
            self.assertNotIn(forbidden, json.dumps(result))

    def test_valid_raw_pair_recomputed_from_rows(self):
        result = self.compare()
        self.assertEqual(result["status"], "COMPARABLE", result)
        self.assertEqual((result["paired"]["net"], result["paired"]["nd"], result["paired"]["exact_p"]),
                         (6, 6, 0.03125))
        self.assertTrue(result["paired"]["nominal_structural_acquisition_claim_eligible"])
        self.assertIsNone(result["paired"]["cross_training_draw_abs_net_floor"])
        self.assertFalse(result["selection_permitted"])

    def test_cross_training_floor_is_separate_from_p_value(self):
        result = self.compare("cross_training_draw")
        self.assertEqual(result["status"], "COMPARABLE", result)
        self.assertTrue(result["paired"]["nominally_significant"])
        self.assertEqual(result["paired"]["cross_training_draw_abs_net_floor"], 20)
        self.assertFalse(result["paired"]["cross_training_draw_floor_cleared"])
        self.assertFalse(result["paired"]["nominal_structural_acquisition_claim_eligible"])

    def test_cross_training_floor_is_inclusive_absolute_twenty(self):
        self.views.write_text("".join(json.dumps({"id": str(i)}) + "\n" for i in range(20)),
                              encoding="utf-8")
        self.left = self.make_arm("reference20", [False] * 20)
        self.right = self.make_arm("candidate19", [True] * 19 + [False])
        result = self.compare("cross_training_draw")["paired"]
        self.assertTrue(result["nominally_significant"])
        self.assertFalse(result["cross_training_draw_floor_cleared"])
        self.right = self.make_arm("candidate20", [True] * 20)
        result = self.compare("cross_training_draw")["paired"]
        self.assertEqual(result["net"], 20)
        self.assertTrue(result["nominal_structural_acquisition_claim_eligible"])
        self.left, self.right = self.right, self.left
        result = self.compare("cross_training_draw")["paired"]
        self.assertEqual(result["net"], -20)
        self.assertTrue(result["nominal_structural_acquisition_claim_eligible"])

    def test_historical_artifact_without_execution_binding_blocks(self):
        self.right[1].unlink()
        self.assert_blocked()

    def test_completed_label_cannot_hide_failed_leg(self):
        for field, value in (("status", "RUNNING"), ("complete", 1), ("returncode", 1), ("returncode", None)):
            original = self.right[1].read_bytes()
            try:
                self.mutate_binding(lambda data: data.update({field: value}))
                with self.subTest(field=field, value=value):
                    self.assert_blocked()
            finally:
                self.right[1].write_bytes(original)

    def test_rehashed_supervisor_must_establish_success(self):
        for field, value in (("status", "TIMEOUT_OR_INTERRUPTED"), ("returncode", 1),
                             ("returncode", False), ("returncode", None)):
            path = self.right[0].parent / "supervisor.json"
            original, binding = path.read_bytes(), self.right[1].read_bytes()
            try:
                self.mutate_bound_file("supervisor", lambda data: data.update({field: value}))
                with self.subTest(field=field, value=value):
                    self.assert_blocked()
            finally:
                path.write_bytes(original)
                self.right[1].write_bytes(binding)

    def test_rehashed_terminal_must_establish_success_and_script_identity(self):
        for field, value in (("status", "FAILED"), ("complete", 1), ("system_exit_code", 1),
                             ("system_exit_code", False), ("failure", {"type": "fixture failure"}),
                             ("script_unchanged", False), ("script_sha256_after", "0" * 64),
                             ("script_hash_error", "unreadable"), ("result", str(self.left[0]))):
            path = self.right[0].parent / "probe_terminal.json"
            original, binding = path.read_bytes(), self.right[1].read_bytes()
            try:
                self.mutate_bound_file("terminal", lambda data: data.update({field: value}))
                with self.subTest(field=field, value=value):
                    self.assert_blocked()
            finally:
                path.write_bytes(original)
                self.right[1].write_bytes(binding)

    def test_copied_result_is_bound_to_terminal_original_bytes(self):
        original = self.right[0]
        copied = original.parent / "copied.json"
        copied.write_bytes(original.read_bytes())
        self.mutate_binding(lambda data: data.update(raw_result_path=str(copied),
                                                    original_raw_result_path=str(original),
                                                    original_raw_result_sha256=paired.sha256_file(original)))
        self.right = copied, self.right[1]
        self.assertEqual(self.compare()["status"], "COMPARABLE")
        original.write_bytes(original.read_bytes() + b" ")
        self.assert_blocked()

    def test_actual_raw_bytes_must_match_receipt(self):
        with self.right[0].open("a") as stream:
            stream.write(" ")
        self.assert_blocked()

    def test_raw_hash_binding_cannot_point_to_other_artifact(self):
        self.mutate_binding(lambda data: data.update(raw_result_path=str(self.left[0]),
                                                    raw_result_sha256=paired.sha256_file(self.left[0])))
        self.assert_blocked()

    def test_stale_views_hash_even_if_both_raws_agree_blocks(self):
        self.views.write_text(self.views.read_text() + " ", encoding="utf-8")
        self.assert_blocked()

    def test_cross_bank_comparison_blocks_even_with_same_ids(self):
        other = self.root / "other_views.jsonl"
        other.write_bytes(self.views.read_bytes().replace(b'"id":', b'"id" :'))
        digest = paired.sha256_file(other)
        self.mutate_raw(lambda data: data.update(views_sha256=digest))
        self.mutate_binding(lambda data: data.update(views_path=str(other), views_sha256=digest))
        self.assert_blocked()

    def test_implementation_mismatch_blocks_even_when_locally_bound(self):
        self.mutate_raw(lambda data: data.update(implementation_digest="b" * 64))
        self.mutate_binding(lambda data: data.update(implementation_digest="b" * 64))
        self.assert_blocked()

    def test_same_implementation_digest_does_not_hide_different_probe_code(self):
        for stem in ("script", "wrapper"):
            path = self.right[0].parent / (stem + ".txt")
            original, binding = path.read_bytes(), self.right[1].read_bytes()
            terminal = self.right[0].parent / "probe_terminal.json"
            terminal_before = terminal.read_bytes()
            try:
                path.write_bytes(b"changed model execution code")
                digest = paired.sha256_file(path)
                self.mutate_binding(lambda data: data.update({stem + "_sha256": digest}))
                if stem == "script":
                    self.mutate_bound_file("terminal", lambda data: data.update(script_sha256_after=digest))
                result = self.compare()
                with self.subTest(stem=stem):
                    self.assert_blocked(result)
                    self.assertIn("different probe code", result["reasons"][0])
            finally:
                path.write_bytes(original)
                terminal.write_bytes(terminal_before)
                self.right[1].write_bytes(binding)

    def test_same_draw_requires_matching_checkpoint_and_source_provenance(self):
        for key in ("source_weight_sha256", "checkpoint_manifest_sha256"):
            paths = (self.right[0].parent / "ARM.json", self.right[0].parent / "MATERIALIZATION.json",
                     self.right[1])
            originals = [p.read_bytes() for p in paths]
            try:
                self.mutate_provenance(key, "e" * 64)
                result = self.compare()
                with self.subTest(key=key):
                    self.assert_blocked(result)
                    self.assertIn("same_training_draw provenance differs", result["reasons"][0])
                    self.assertEqual(self.compare("cross_training_draw")["status"], "COMPARABLE")
            finally:
                for path, original in zip(paths, originals):
                    path.write_bytes(original)

    def test_training_provenance_absence_or_receipt_disagreement_blocks(self):
        for key in ("source_weight_sha256", "checkpoint_manifest_sha256"):
            path = self.right[0].parent / "ARM.json"
            original, binding = path.read_bytes(), self.right[1].read_bytes()
            try:
                self.mutate_bound_file("arm_receipt", lambda data: data.pop(key))
                with self.subTest(key=key, failure="missing"):
                    self.assert_blocked()
                self.mutate_bound_file("arm_receipt", lambda data: data.update({key: "e" * 64}))
                with self.subTest(key=key, failure="disagree"):
                    self.assert_blocked()
            finally:
                path.write_bytes(original)
                self.right[1].write_bytes(binding)

    def test_generation_configuration_mismatch_blocks(self):
        self.mutate_raw(lambda data: data.update(max_new_tokens=95))
        self.assert_blocked()

    def test_changed_weight_before_after_or_live_bytes_blocks(self):
        self.mutate_binding(lambda data: data.update(weight_sha256_after="0" * 64))
        self.assert_blocked()

    def test_changed_live_weight_blocks(self):
        (self.right[0].parent / "model.safetensors").write_bytes(b"different weights")
        self.assert_blocked()

    def test_reclaimed_weights_allow_contemporaneous_receipt(self):
        (self.right[0].parent / "model.safetensors").unlink()
        self.assertEqual(self.compare()["status"], "COMPARABLE")

    def test_weight_receipt_disagreement_blocks_after_receipt_rehash(self):
        self.mutate_bound_file("arm_receipt", lambda data: data.update(arm_weight_sha256="0" * 64))
        self.assert_blocked()

    def test_sidecar_changed_during_run_blocks(self):
        self.mutate_binding(lambda data: data["sidecar_sha256_after"].update({"config.json": "0" * 64}))
        self.assert_blocked()

    def test_live_sidecar_hash_mismatch_blocks(self):
        (self.right[0].parent / "config.json").write_bytes(b"different config")
        self.assert_blocked()

    def test_model_directory_mismatch_blocks(self):
        self.mutate_raw(lambda data: data.update(model_dir=str(self.root / "unbound_arm")))
        self.assert_blocked()

    def test_code_or_terminal_artifact_changed_blocks(self):
        (self.right[0].parent / "probe_terminal.json").write_text("altered", encoding="utf-8")
        self.assert_blocked()

    def test_wrong_prediction_id_or_type_blocks(self):
        for mutate in (lambda data: data["predictions"][0].update(id="other"),
                       lambda data: data["predictions"][0].update(id="1"),
                       lambda data: data["predictions"][0].update(canonical=1),
                       lambda data: data["predictions"].pop()):
            original_raw = self.right[0].read_bytes()
            original_binding = self.right[1].read_bytes()
            try:
                self.mutate_raw(mutate)
                self.assert_blocked()
            finally:
                self.right[0].write_bytes(original_raw)
                self.right[1].write_bytes(original_binding)

    def test_missing_relationship_blocks(self):
        self.assert_blocked(self.compare("unknown"))

    def test_public_validator_returns_receipt_verified_training_identity(self):
        raw, binding, ids = self.validate_candidate("process")
        self.assertEqual(ids, {str(i) for i in range(6)})
        self.assertEqual(binding["training_identity"],
                         {"source_weight_sha256": "c" * 64, "checkpoint_manifest_sha256": "d" * 64})
        self.assertEqual(raw["record"], "PROCESS_TRANSITION_PROBE")

    def test_public_validator_wraps_missing_files_and_wrong_kind(self):
        with self.assertRaises(paired.ComparabilityError):
            self.validate_candidate("retention")
        self.right[1].unlink()
        with self.assertRaises(paired.ComparabilityError):
            self.validate_candidate("process")

    def test_public_process_validator_rejects_partial_prediction_rows(self):
        self.mutate_raw(lambda data: data["predictions"].pop())
        with self.assertRaisesRegex(paired.ComparabilityError, "prediction IDs differ"):
            self.validate_candidate("process")

    def test_retention_generic_validator_accepts_hf_target_and_full_capture(self):
        self.convert_candidate_kind("retention")
        raw, binding, ids = self.validate_candidate("retention")
        self.assertTrue(raw["target"].startswith("hf:"))
        self.assertEqual(len(ids), 6)
        self.assertIn("training_identity", binding)

    def test_retention_partial_rehashed_capture_is_rejected(self):
        self.convert_candidate_kind("retention")
        self.mutate_bound_file("retention_predictions", lambda data: data["predictions"].pop())
        with self.assertRaisesRegex(paired.ComparabilityError, "capture is incomplete"):
            self.validate_candidate("retention")

    def test_retention_changed_additional_input_is_rejected(self):
        self.convert_candidate_kind("retention")
        (self.right[0].parent / "reference.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(paired.ComparabilityError, "additional probe input hash mismatch"):
            self.validate_candidate("retention")

    def test_retention_changed_during_probe_additional_input_is_rejected(self):
        self.convert_candidate_kind("retention")
        self.mutate_binding(lambda data: data["additional_input_sha256_after"].clear())
        with self.assertRaisesRegex(paired.ComparabilityError, "inputs missing or changed"):
            self.validate_candidate("retention")

    def test_long_context_generic_validator_accepts_complete_bound_key(self):
        self.convert_candidate_kind("long_context")
        raw, _, ids = self.validate_candidate("long_context")
        self.assertEqual(set(raw["per_row_correct"]), ids)

    def test_long_context_rehashed_wrong_key_ids_rejected(self):
        self.convert_candidate_kind("long_context")
        path = self.right[0].parent / "answer_key.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"0"', b'"different"'))
        digest = paired.sha256_file(path)
        self.mutate_binding(lambda data: data.update(additional_input_sha256_before={str(path): digest},
                                                     additional_input_sha256_after={str(path): digest}))
        with self.assertRaisesRegex(paired.ComparabilityError, "answer key IDs differ"):
            self.validate_candidate("long_context")

    def test_long_context_partial_scores_rejected(self):
        self.convert_candidate_kind("long_context")
        self.mutate_raw(lambda data: data["per_row_correct"].pop("0"))
        with self.assertRaisesRegex(paired.ComparabilityError, "prediction/score IDs differ"):
            self.validate_candidate("long_context")


if __name__ == "__main__":
    unittest.main()
