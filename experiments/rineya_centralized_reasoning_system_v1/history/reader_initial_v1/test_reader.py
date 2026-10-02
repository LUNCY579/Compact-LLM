"""Lightweight stdlib-only reader rejection/geometry/termination controls."""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import reader
from experiments.rineya_centralized_reasoning_system_v1 import reader_execution as execution


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = reader.read_json(reader.CONTRACT)

    def fake_forward(self, tokens):
        iterator = iter(tokens)
        def call(ids, keep, stage):
            self.assertEqual(keep, 1)
            return {"argmax_id": next(iterator), "lowest_id_argmax_verified": True,
                    "input_ids_sha256": reader.digest(ids), "input_length": len(ids),
                    "selected_model_logprob": -.1, "raw_logits_sha256": "a" * 64,
                    "raw_hidden_sha256": "b" * 64}
        return call

    def test_public_token_boundary_rejects_evaluator_fields_and_unsupported_shapes(self):
        base = {"input_ids": [16], "attention_mask": [1], "max_new_tokens": 64}
        for request in ({**base, "gold": "12"}, {**base, "max_new_tokens": 65},
                        {**base, "input_ids": [True]}, {**base, "attention_mask": [0]},
                        {**base, "input_ids": [16] * 513, "attention_mask": [1] * 513}):
            with self.assertRaises(reader.ReaderBlocked): reader.validate_request(request)

    def test_each_eos_id_and_eos_at_cap_take_precedence(self):
        for eos in reader.EOS_IDS:
            request = {"input_ids": [eos], "attention_mask": [1], "max_new_tokens": 2}
            out = reader.manual_greedy(request, self.fake_forward([17, eos]))
            self.assertEqual(out["generation_count"], 2)
            self.assertEqual(out["stop_reason"], "eos")
            self.assertEqual(out["visible_ids"], [17])
            self.assertEqual(out["behavior_logprobs"], [0.0, 0.0])

    def test_prompt_eos_is_not_output_termination_and_cap_never_emits_extra(self):
        request = {"input_ids": [151645] * 512, "attention_mask": [1] * 512, "max_new_tokens": 64}
        out = reader.manual_greedy(request, self.fake_forward([17] * 64))
        self.assertEqual(len(out["generated_ids"]), 64)
        self.assertEqual(out["stop_reason"], "cap")
        self.assertEqual(out["steps"][-1]["input_length"], 575)

    def test_invalid_argmax_nonfinite_and_foreign_prefix_rejected(self):
        request = {"input_ids": [16], "attention_mask": [1], "max_new_tokens": 1}
        for change in ({"lowest_id_argmax_verified": False}, {"argmax_id": True},
                       {"selected_model_logprob": float("nan")}, {"input_ids_sha256": "foreign"}):
            original = self.fake_forward([17])
            def bad(ids, keep, stage): return {**original(ids, keep, stage), **change}
            with self.assertRaises(reader.ReaderBlocked): reader.manual_greedy(request, bad)

    def test_exact_278_forward_upper_schedule_and_nonemitting576_geometry(self):
        plan = reader.qualifier_plan(self.cfg)
        self.assertEqual(sum(b["max_forwards"] for b in plan["blocks"]), 278)
        self.assertEqual(sum(b["max_input_positions"] for b in plan["blocks"]), 146316)
        self.assertEqual(plan["max_logical_calls"], 590)
        lengths = [len(ids) for ids in plan["forced_mechanical_prefixes"]]
        self.assertEqual(lengths, list(range(512, 577)))
        self.assertFalse(plan["mechanical_is_model_generation"])
        self.assertEqual(sum(lengths), 35360)

    def test_original_likelihood_bounds_are_not_replaced_by_replay_identity(self):
        observed = reader.likelihood_measurement([-.5, -.41], [-.46, -.37])
        self.assertFalse(observed["passed"])
        self.assertEqual(observed["tolerances"], {"max_abs": .03125, "mean_abs": .01})
        self.assertEqual(self.cfg["historical_bulk_capture"]["status"], "FAILED_UNRESOLVED")
        with self.assertRaises(reader.ReaderBlocked): reader.likelihood_measurement([float("nan")], [0])

    def test_same_prefix_raw_hash_not_merely_equal_argmax(self):
        first = [{"input_ids_sha256": "a", "raw_logits_sha256": "b", "raw_hidden_sha256": "c", "selected_model_logprob": -.1}]
        second = copy.deepcopy(first); second[0]["raw_hidden_sha256"] = "different"
        with self.assertRaises(reader.ReaderBlocked): reader.require_same_prefix(first, second)

    def test_caller_pass_or_d2_receipt_cannot_open_reader(self):
        for evidence in (None, {"status": "PASS"}, {"scope": "D2", "native_qualified": True}):
            with self.assertRaises(reader.ReaderBlocked): reader.verify_qualification(evidence, {})

    def test_builder_review_and_wrong_binding_fail_without_model_import(self):
        expected = {"source_commit": "a" * 40}
        for review in ({"status": "APPROVED_FOR_READER_QUALIFICATION", "binding": expected, "reviewer": "/root/runtime_audit"},
                       {"status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC", "binding": expected, "reviewer": "independent"}):
            with self.assertRaises(reader.ReaderBlocked): execution.review_matches(review, expected)

    def test_stock_policy_is_no_cache_greedy_without_processors(self):
        policy = self.cfg["stock_generation_config"]
        self.assertFalse(policy["do_sample"]); self.assertFalse(policy["use_cache"])
        self.assertEqual(policy["num_beams"], 1); self.assertEqual(policy["repetition_penalty"], 1)
        self.assertIsNone(policy["forced_eos_token_id"])
        self.assertEqual(policy["min_new_tokens"], 0)
        self.assertEqual(policy["eos_token_id"], [151645, 151643])

    def test_no_torch_transformers_or_tokenizer_loaded_by_this_module(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("tokenizers", sys.modules)

    def test_source_contract_drift_rejected_before_framework_import(self):
        with self.assertRaises(reader.ReaderBlocked): execution.load_contract("0" * 64)


if __name__ == "__main__":
    unittest.main()
