"""In-memory invented trace fixtures; no Rineya execution or capability evidence."""
import copy
import math
import unittest
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import Blocked, digest, signed, verify
from experiments.rineya_generalization_dynamics_v1.export_parity import (
    EXPECTED_SCHEMA, POLICY_SCHEMA, TRACE_SCHEMA, compare_export_parity,
)


def resign(value):
    value.pop("sha256", None)
    value.update(signed(value))
    return value


def refresh_trace(trace):
    for checkpoint in trace["checkpoints"]:
        resign(checkpoint["load_receipt"])
        scoring = checkpoint["scoring_receipt"]
        scoring["load_receipt_sha256"] = checkpoint["load_receipt"]["sha256"]
        scoring["probe_payload_sha256"] = digest(checkpoint["probes"])
        resign(scoring)
    return resign(trace)


def fixture(evidence_level="mechanical_fixture"):
    """All values, including 600M declarations, are invented fixture metadata."""
    policy = signed({"schema": POLICY_SCHEMA, "registered_before_outcomes": True,
                     "abs_tolerance": 1e-6, "rel_tolerance": 1e-6,
                     "max_prompt_tokens": 16, "max_candidate_tokens": 4,
                     "max_generated_tokens": 4, "max_probes": 4,
                     "max_logit_probe_tokens": 4, "logit_probe_token_ids": [1, 2, 3]})
    runtime = {"versions": {"torch": "fixture-only", "transformers": "fixture-only"},
               "python": "fixture-only", "device": "cpu", "dtype": "float32", "cpu_threads": 2,
               "attention": "eager", "deterministic_algorithms": True, "local_files_only": True,
               "trust_remote_code": False, "quantized": False, "one_checkpoint_at_a_time": True}
    decoding = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                "use_cache": True, "enable_thinking": False, "truncation": False,
                "candidate_encoding": "space_prefix_no_special_tokens", "max_new_tokens": 4,
                "eos_token_id": 99, "pad_token_id": 99}
    source, base, tokenizer = [digest(label) for label in ("fixture-source", "fixture-base", "fixture-tokenizer")]
    schedule = [{"id": "base", "step": 0, "kind": "base", "checkpoint_content_sha256": base,
                 "snapshot_receipt_sha256": None, "origin": "base"}]
    schedule += [{"id": f"step_{step}", "step": step, "kind": "peft",
                  "checkpoint_content_sha256": digest(f"fixture-save-{step}"),
                  "snapshot_receipt_sha256": digest(f"fixture-receipt-{step}"),
                  "origin": "original_training_snapshot"} for step in (10, 20)]
    prompt = "Fixture-only arithmetic: 12 plus 3 equals?"
    candidate_text = [("answer", "15", [4, 5]), ("distractor", "14", [4, 6])]
    golden = {"id": "dev-fixture-multitoken", "prompt_sha256": digest(prompt),
              "rendered_prompt_sha256": digest(prompt), "prompt_token_sha256": digest([7, 8, 9]),
              "prompt_tokens": 3, "candidates": [{"label": label, "text_sha256": digest(text),
                 "token_sha256": digest(ids), "tokens": len(ids)} for label, text, ids in candidate_text]}
    expected = signed({"schema": EXPECTED_SCHEMA, "registered_before_outcomes": True,
                       "parity_policy_sha256": policy["sha256"], "trajectory_id": "fixture-only",
                       "source_sha256": source, "base_content_sha256": base,
                       "tokenizer_content_sha256": tokenizer, "expected_base_parameters": 600_000_000,
                       "dev_suite_sha256": digest([golden]), "split": "dev", "runtime": runtime,
                       "prompt_format": "plain", "decoding_policy": decoding, "evidence_level": evidence_level,
                       "base_checkpoint_id": "base", "final_checkpoint_id": "step_20",
                       "checkpoints": schedule, "probes": [golden]})
    probes = [{"id": golden["id"], "split": "dev", "prompt": prompt, "rendered_prompt": prompt,
               "truncation_applied": False, "prompt_token_ids": [7, 8, 9],
               "prompt_token_sha256": digest([7, 8, 9]), "generated_token_ids": [4, 5, 99],
               "eos_token_id": 99, "stop_reason": "eos", "generated_text": "15",
               "candidates": [{"label": label, "text": text, "token_ids": ids, "token_sha256": digest(ids),
                   "teacher_forced": True, "includes_prompt_loss": False, "logit_shift": 1,
                   "candidate_encoding": "space_prefix_no_special_tokens", "answer_logp": -3.0,
                   "token_scores": [{"position": index, "token_id": token,
                                     "probe_logits": [0.5 + index, -0.2, 0.25], "target_logp": -1.0 - index}
                                    for index, token in enumerate(ids)]}
                              for label, text, ids in candidate_text]}]

    def trace(role):
        result = {"schema": TRACE_SCHEMA, "role": role, "split": "dev", "expected_sha256": expected["sha256"],
                  "parity_policy_sha256": policy["sha256"], "trajectory_id": expected["trajectory_id"],
                  "source_sha256": source, "base_content_sha256": base, "tokenizer_content_sha256": tokenizer,
                  "expected_base_parameters": 600_000_000, "dev_suite_sha256": expected["dev_suite_sha256"],
                  "runtime": copy.deepcopy(runtime), "prompt_format": "plain", "decoding_policy": copy.deepcopy(decoding),
                  "evidence_level": evidence_level, "caller_declared_model_execution": evidence_level == "native_pilot",
                  "checkpoints": []}
        for saved in schedule:
            adapter = True if saved["kind"] == "peft" else None
            load = signed({"status": "STRICT_LOAD_REPORTED", "evidence_level": evidence_level,
                           "checkpoint_id": saved["id"], "source_sha256": source, "base_content_sha256": base,
                           "tokenizer_content_sha256": tokenizer, "checkpoint_content_sha256": saved["checkpoint_content_sha256"],
                           "runtime_sha256": digest(runtime), "actual_base_parameters": 600_000_000,
                           "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": [],
                           "eval_mode": True, "requires_grad": False, "adapter_tensor_keys_equal": adapter,
                           "adapter_tensor_shapes_equal": adapter, "adapter_tensors_equal": adapter})
            scoring = signed({"status": "DEV_SCORING_REPORTED", "split": "dev", "evidence_level": evidence_level,
                              "checkpoint_id": saved["id"], "load_receipt_sha256": load["sha256"],
                              "dev_suite_sha256": expected["dev_suite_sha256"], "probe_payload_sha256": digest(probes),
                              "full_candidate_scoring": True})
            result["checkpoints"].append({**saved, "load_receipt": load, "scoring_receipt": scoring,
                                          "probes": copy.deepcopy(probes)})
        return signed(result)
    return trace("producer"), trace("gd"), expected, policy


class ExportParityTests(unittest.TestCase):
    def setUp(self):
        self.producer, self.gd, self.expected, self.policy = fixture()

    def compare(self):
        return compare_export_parity(self.producer, self.gd, self.expected, self.policy)

    def candidate(self, trace=None):
        return (trace or self.gd)["checkpoints"][1]["probes"][0]["candidates"][0]

    def rebind_expected(self):
        resign(self.expected)
        for trace in (self.producer, self.gd):
            trace["expected_sha256"] = self.expected["sha256"]
            refresh_trace(trace)

    def test_complete_fixture_matches_without_becoming_native_evidence(self):
        result = verify(self.compare())
        self.assertEqual(result["status"], "TRACE_PARITY_MATCHED")
        self.assertEqual(result["compared_candidate_tokens"], 12)
        self.assertEqual(result["evidence_level"], "mechanical_fixture")
        self.assertEqual(result["loaded_scored_qualification"], "NOT_ESTABLISHED")
        self.assertFalse(result["native_execution_independently_verified"])
        self.assertFalse(result["permits_confirmation"])
        self.assertFalse(result["permits_model_promotion"])

    def test_native_declared_traces_still_do_not_attest_native_execution(self):
        result = compare_export_parity(*fixture("native_pilot"))
        self.assertEqual(result["evidence_level"], "native_pilot")
        self.assertFalse(result["native_execution_independently_verified"])
        self.assertTrue(result["requires_separate_strict_load_and_dev_scoring_receipts"])

    def test_runtime_must_also_be_supported_by_gd_loader(self):
        for device, dtype in (("cpu", "bfloat16"), ("cpu", "float16"), ("cuda", "float16")):
            producer, gd, expected, policy = fixture()
            expected["runtime"].update(device=device, dtype=dtype)
            resign(expected)
            with self.subTest(device=device, dtype=dtype), self.assertRaises(Blocked):
                compare_export_parity(producer, gd, expected, policy)

    def test_comparator_imports_no_model_runtime(self):
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], {"torch", "transformers", "peft", "safetensors"})
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            self.compare()

    def test_inputs_are_not_mutated(self):
        original = copy.deepcopy((self.producer, self.gd, self.expected, self.policy))
        self.compare()
        self.assertEqual((self.producer, self.gd, self.expected, self.policy), original)

    def test_finite_numeric_deviation_within_registered_tolerance_matches(self):
        candidate = self.candidate()
        candidate["token_scores"][1]["target_logp"] += 1e-7
        candidate["token_scores"][1]["probe_logits"][0] += 1e-7
        candidate["answer_logp"] += 1e-7
        refresh_trace(self.gd)
        self.assertEqual(self.compare()["status"], "TRACE_PARITY_MATCHED")

    def test_out_of_tolerance_logits_rejected(self):
        self.candidate()["token_scores"][0]["probe_logits"][0] += 0.1
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "teacher-forced logit"):
            self.compare()

    def test_per_token_logp_difference_rejected_even_when_total_matches(self):
        scores = self.candidate()["token_scores"]
        scores[0]["target_logp"] -= 0.1
        scores[1]["target_logp"] += 0.1
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "target-token"):
            self.compare()

    def test_full_scoring_total_cannot_use_first_token_only(self):
        self.candidate()["answer_logp"] = -1.0
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "full answer log probability sum"):
            self.compare()

    def test_omitted_tail_token_vector_rejected(self):
        self.candidate()["token_scores"].pop()
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Every candidate token"):
            self.compare()

    def test_complete_token_ids_cannot_be_shortened(self):
        candidate = self.candidate()
        candidate["token_ids"].pop()
        candidate["token_sha256"] = digest(candidate["token_ids"])
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Complete candidate tokenization"):
            self.compare()

    def test_one_token_golden_screen_cannot_qualify_full_answer_scoring(self):
        for candidate in self.expected["probes"][0]["candidates"]:
            candidate["tokens"] = 1
        self.rebind_expected()
        with self.assertRaisesRegex(Blocked, "one-token screen"):
            self.compare()

    def test_logits_must_cover_all_predeclared_vocabulary_positions(self):
        self.candidate()["token_scores"][0]["probe_logits"].pop()
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Incomplete bounded logit vector"):
            self.compare()

    def test_prompt_loss_or_double_logit_shift_rejected(self):
        for field, value in (("includes_prompt_loss", True), ("logit_shift", 2), ("teacher_forced", False)):
            with self.subTest(field=field):
                self.setUp()
                self.candidate()[field] = value
                refresh_trace(self.gd)
                with self.assertRaisesRegex(Blocked, "answer-only teacher-forced"):
                    self.compare()

    def test_nonfinite_numeric_payload_rejected_before_comparison(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                self.setUp()
                self.candidate()["token_scores"][0]["probe_logits"][0] = value
                with self.assertRaises(Blocked):
                    self.compare()

    def test_positive_log_probability_rejected(self):
        self.candidate()["token_scores"][0]["target_logp"] = 1.0
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Positive target-token"):
            self.compare()

    def test_boolean_values_do_not_count_as_tokens_or_numbers(self):
        self.candidate()["token_scores"][0]["target_logp"] = True
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Non-finite/invalid"):
            self.compare()

    def test_modified_unsigned_trace_fails_integrity(self):
        self.gd["source_sha256"] = digest("another source")
        with self.assertRaisesRegex(Blocked, "digest"):
            self.compare()

    def test_nested_load_receipt_signature_is_verified(self):
        self.gd["checkpoints"][1]["load_receipt"]["actual_base_parameters"] += 1
        resign(self.gd)
        with self.assertRaisesRegex(Blocked, "digest"):
            self.compare()

    def test_loosened_tolerance_cannot_replace_frozen_policy(self):
        self.policy["abs_tolerance"] = 0.5
        resign(self.policy)
        with self.assertRaisesRegex(Blocked, "changed or loosened"):
            self.compare()

    def test_policy_tolerance_must_be_finite_nonnegative(self):
        self.policy["rel_tolerance"] = -1.0
        resign(self.policy)
        self.expected["parity_policy_sha256"] = self.policy["sha256"]
        self.rebind_expected()
        with self.assertRaisesRegex(Blocked, "Negative relative tolerance"):
            self.compare()

    def test_finite_tolerance_cannot_overflow_to_an_infinite_comparison_bound(self):
        self.policy["rel_tolerance"] = 1e308
        resign(self.policy)
        self.expected["parity_policy_sha256"] = self.policy["sha256"]
        resign(self.expected)
        for trace in (self.producer, self.gd):
            trace["expected_sha256"] = self.expected["sha256"]
            trace["parity_policy_sha256"] = self.policy["sha256"]
            refresh_trace(trace)
        with self.assertRaisesRegex(Blocked, "non-finite comparison bound"):
            self.compare()

    def test_registration_after_outcomes_rejected(self):
        self.expected["registered_before_outcomes"] = False
        self.rebind_expected()
        with self.assertRaisesRegex(Blocked, "before outcomes"):
            self.compare()

    def test_source_base_tokenizer_runtime_and_policy_mismatches_rejected(self):
        mutations = {"source_sha256": digest("wrong source"), "base_content_sha256": digest("wrong base"),
                     "tokenizer_content_sha256": digest("wrong tokenizer"), "prompt_format": "chat",
                     "runtime": {"incomplete": True}, "decoding_policy": {"do_sample": True}}
        for field, value in mutations.items():
            with self.subTest(field=field):
                self.setUp()
                self.gd[field] = value
                refresh_trace(self.gd)
                with self.assertRaisesRegex(Blocked, field):
                    self.compare()

    def test_truncated_actual_prompt_rejected(self):
        self.gd["checkpoints"][1]["probes"][0]["truncation_applied"] = True
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Truncated prompts"):
            self.compare()

    def test_changed_prompt_tokens_cannot_hide_behind_same_prompt_text(self):
        row = self.gd["checkpoints"][1]["probes"][0]
        row["prompt_token_ids"] = [7, 8]
        row["prompt_token_sha256"] = digest(row["prompt_token_ids"])
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Actual prompt tokenization"):
            self.compare()

    def test_actual_prompt_payloads_must_be_preserved(self):
        del self.gd["checkpoints"][1]["probes"][0]["rendered_prompt"]
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Incomplete/invalid"):
            self.compare()

    def test_generated_tokens_must_match_exactly(self):
        self.gd["checkpoints"][1]["probes"][0]["generated_token_ids"] = [4, 6, 99]
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "generated_token_ids"):
            self.compare()

    def test_invalid_eos_and_length_stop_receipts_rejected(self):
        for tokens, stop in (([4, 5], "eos"), ([99, 4, 99], "eos"), ([4, 5], "length"), ([4, 5, 99], "length")):
            with self.subTest(tokens=tokens, stop=stop):
                self.setUp()
                row = self.gd["checkpoints"][1]["probes"][0]
                row["generated_token_ids"], row["stop_reason"] = tokens, stop
                refresh_trace(self.gd)
                with self.assertRaisesRegex(Blocked, "stop receipt"):
                    self.compare()

    def test_missing_snapshot_and_wrong_schedule_order_rejected(self):
        for mode in ("missing", "reordered", "receipt"):
            with self.subTest(mode=mode):
                self.setUp()
                if mode == "missing":
                    self.gd["checkpoints"].pop()
                elif mode == "reordered":
                    self.gd["checkpoints"].reverse()
                else:
                    self.gd["checkpoints"][1]["snapshot_receipt_sha256"] = digest("wrong receipt")
                refresh_trace(self.gd)
                with self.assertRaises(Blocked):
                    self.compare()

    def test_schedule_step_types_are_exact_even_when_boolean_equals_zero(self):
        self.gd["checkpoints"][0]["step"] = False
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "schedule/receipt"):
            self.compare()

    def test_runtime_policy_number_types_are_exact(self):
        self.gd["decoding_policy"]["repetition_penalty"] = True
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "decoding_policy"):
            self.compare()

    def test_qualification_updates_cannot_substitute_original_training_saves(self):
        self.expected["checkpoints"][1]["origin"] = "qualification_update"
        self.rebind_expected()
        with self.assertRaisesRegex(Blocked, "qualification updates"):
            self.compare()

    def test_base_plus_two_saved_snapshots_is_minimum(self):
        self.expected["checkpoints"].pop()
        self.rebind_expected()
        with self.assertRaisesRegex(Blocked, "base plus 2-19"):
            self.compare()

    def test_actual_base_count_must_match_exact_500M_700M_source(self):
        self.gd["checkpoints"][1]["load_receipt"]["actual_base_parameters"] = 344_000_000
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Actual loaded base parameter count"):
            self.compare()

    def test_nonempty_strict_load_errors_and_inexact_adapter_rejected(self):
        for field, value in (("missing_keys", ["layer.weight"]), ("unexpected_keys", ["readout"]),
                             ("mismatched_keys", ["shape"]), ("error_msgs", ["failed"]),
                             ("adapter_tensors_equal", False), ("adapter_tensor_shapes_equal", False)):
            with self.subTest(field=field):
                self.setUp()
                self.gd["checkpoints"][1]["load_receipt"][field] = value
                refresh_trace(self.gd)
                with self.assertRaises(Blocked):
                    self.compare()

    def test_missing_full_dev_scoring_receipt_rejected(self):
        del self.gd["checkpoints"][1]["scoring_receipt"]
        resign(self.gd)
        with self.assertRaisesRegex(Blocked, "Incomplete/invalid"):
            self.compare()

    def test_scoring_receipt_must_bind_actual_probe_payload(self):
        self.gd["checkpoints"][1]["probes"][0]["generated_text"] = "14"
        resign(self.gd)
        with self.assertRaisesRegex(Blocked, "DEV scoring receipt"):
            self.compare()

    def test_missing_probe_or_distractor_rejected(self):
        self.gd["checkpoints"][1]["probes"][0]["candidates"].pop()
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Both complete candidates"):
            self.compare()

    def test_mechanical_cannot_satisfy_native_expected_specification(self):
        self.expected["evidence_level"] = "native_pilot"
        self.rebind_expected()
        with self.assertRaisesRegex(Blocked, "evidence_level"):
            self.compare()

    def test_false_native_execution_claim_rejected(self):
        self.producer, self.gd, self.expected, self.policy = fixture("native_pilot")
        self.gd["caller_declared_model_execution"] = False
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "Mechanical/native"):
            self.compare()

    def test_confirmation_trace_or_specification_rejected(self):
        self.gd["split"] = "confirmation"
        refresh_trace(self.gd)
        with self.assertRaisesRegex(Blocked, "DEV split"):
            self.compare()
        self.setUp()
        self.expected["split"] = "confirmation"
        self.rebind_expected()
        with self.assertRaisesRegex(Blocked, "Only DEV"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
