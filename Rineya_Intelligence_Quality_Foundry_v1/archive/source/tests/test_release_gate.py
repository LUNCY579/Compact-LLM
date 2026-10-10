"""Fault injection for incomplete, stale and self-reviewed release evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import strict_json, write_rows
from pipeline import validate_reviews, mutation_gate, validate_profile, validate_assignments

class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.row = {"row_id": "r", "row_sha256": "h", "author_id": "generator"}
        self.assignment = {"rubric_version": "rubric", "rows": 1, "assignments": [
            {"role": "semantic", "reviewer_id": "s", "packet_sha256": "ps", "row_ids": ["r"]},
            {"role": "adversarial", "reviewer_id": "a", "packet_sha256": "pa", "row_ids": ["r"]}]}
        self.receipts = [{"row_id": "r", "row_sha256": "h", "role": role,
            "reviewer_id": actor, "packet_sha256": packet, "rubric_version": "rubric",
            "reviewer_model_identity": "inherited model; deployment not attested",
            "blind_answer": '{"answer":3}', "disposition": "accept", "findings": [],
            "evidence": ["actual source witness"], "method": "public parser and solver",
            "limitations": "no human calibration"}
            for role, actor, packet in [("semantic", "s", "ps"), ("adversarial", "a", "pa")]]

    def check_receipts(self, receipts, assignment=None):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for i, group in enumerate(receipts):
                path = Path(directory) / (str(i) + ".jsonl")
                write_rows(path, group)
                files.append(path)
            with patch("verifier.verify_answer", side_effect=lambda row, text: text == '{"answer":3}'):
                return validate_reviews([self.row], assignment or self.assignment, files)

    def test_complete_independent_reviews(self):
        self.assertEqual(self.check_receipts([[r] for r in self.receipts])["reviewed_rows_per_role"], 1)

    def test_stale_missing_wrong_or_unassigned_reviews_block(self):
        for field, value in [("row_sha256", "stale"), ("packet_sha256", "other"),
                             ("rubric_version", "old"), ("reviewer_id", "generator"),
                             ("blind_answer", '{"answer":4}'), ("evidence", []),
                             ("disposition", "repair"), ("findings", ["ambiguous"])]:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.receipts)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    self.check_receipts([[r] for r in rows])
        with self.assertRaises(ValueError):
            self.check_receipts([[self.receipts[0]]])
        with self.assertRaises(ValueError):
            self.check_receipts([[self.receipts[0], self.receipts[0]], [self.receipts[1]]])

    def test_same_actor_in_both_roles_blocks(self):
        assignment = copy.deepcopy(self.assignment)
        assignment["assignments"][1]["reviewer_id"] = "s"
        rows = copy.deepcopy(self.receipts)
        rows[1]["reviewer_id"] = "s"
        with self.assertRaises(ValueError):
            self.check_receipts([[r] for r in rows], assignment)

    def test_actual_generator_actor_alias_blocks(self):
        assignment = copy.deepcopy(self.assignment)
        assignment["assignments"][0]["reviewer_id"] = "/root/existing_corpus_audit"
        rows = copy.deepcopy(self.receipts)
        rows[0]["reviewer_id"] = "/root/existing_corpus_audit"
        with self.assertRaises(ValueError):
            self.check_receipts([[r] for r in rows], assignment)

    def test_role_paths_are_rejected_before_use(self):
        assignment = copy.deepcopy(self.assignment)
        assignment["assignments"][0]["role"] = "../../reserved_payload"
        with self.assertRaises(ValueError):
            validate_assignments([self.row], assignment)

    def test_integer_reviewer_identity_and_boolean_row_count_block(self):
        for key, value in [("reviewer_id", 17), ("role", 17)]:
            assignment = copy.deepcopy(self.assignment)
            assignment["assignments"][0][key] = value
            with self.assertRaises(ValueError):
                validate_assignments([self.row], assignment)
        assignment = copy.deepcopy(self.assignment)
        assignment["rows"] = True
        with self.assertRaises(ValueError):
            validate_assignments([self.row], assignment)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"answer":1,"answer":2}')

    def test_forged_mutation_summary_does_not_hide_failed_witness(self):
        checks = [{"name": str(i), "applicable": True, "passed": True} for i in range(6)]
        result = {"source_valid": True, "checks": checks, "applicable_checks": 6,
                  "passed_checks": 6, "failed_checks": 0}
        self.assertTrue(mutation_gate(result))
        checks[1]["passed"] = False
        self.assertFalse(mutation_gate(result))
        checks[1]["passed"] = True
        checks[1]["applicable"] = False
        checks[1]["passed"] = None
        self.assertFalse(mutation_gate(result))

class ProfileGateTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.policy = strict_json((root / "config/quality_policy.json").read_text())
        self.row = {"row_id": "r", "row_sha256": "h", "family": "family", "split": "TRAIN", "profile": "short"}
        self.record = dict(self.row, prompt_tokens=10, completion_tokens_including_eos=3,
            total_tokens=13, padding_tokens=0, reserved_response_budget=128,
            ce512_length_fit_only=True, profile_bucket=512, truncated=False,
            packing=False, native_training_eligible=False)
        for key in ["prompt_ids_sha256", "completion_ids_sha256", "input_ids_sha256", "labels_sha256", "rendered_prompt_sha256"]:
            self.record[key] = "a" * 64
        self.summary = {"tokenizer_files": {"tokenizer.json": self.policy["tokenizer_sha256"],
                           "tokenizer_config.json": self.policy["tokenizer_config_sha256"]},
            "chat_template_sha256": self.policy["chat_template_sha256"],
            "runtime": {k: self.policy["tokenizer_runtime"][k] for k in ["transformers", "tokenizers"]},
            "python": self.policy["tokenizer_runtime"]["python"],
            "template_args": {"enable_thinking": False, "add_generation_prompt": True},
            "model_framework_imported": False, "model_weights_read": False,
            "native_long_context_qualified": False, "training_ready": False,
            "rows": 1, "ce512_length_fit_rows": 1, "minimum_total_tokens": 13,
            "maximum_total_tokens": 13, "input_tokens": 13, "supervised_tokens": 3,
            "length_bucket_counts": {"512": 1}}

    def test_consistent_metadata_passes(self):
        validate_profile([self.row], self.summary, [self.record], self.policy)

    def test_altered_token_counts_flags_and_budgets_block(self):
        for key, value in [("prompt_tokens", -42), ("completion_tokens_including_eos", False),
                           ("total_tokens", -999), ("padding_tokens", False),
                           ("native_training_eligible", True), ("ce512_length_fit_only", False),
                           ("reserved_response_budget", 127), ("profile_bucket", 8192)]:
            with self.subTest(key=key):
                record = dict(self.record, **{key: value})
                with self.assertRaises(ValueError):
                    validate_profile([self.row], self.summary, [record], self.policy)

    def test_forged_tokenizer_summary_and_native_claims_block(self):
        for key, value in [("chat_template_sha256", "wrong"), ("tokenizer_files", {}),
                           ("model_framework_imported", True), ("model_weights_read", True),
                           ("training_ready", True), ("input_tokens", 999),
                           ("template_args", {"enable_thinking": 0, "add_generation_prompt": 1}),
                           ("runtime", {"tokenizers": "wrong"})]:
            with self.subTest(key=key):
                summary = dict(self.summary, **{key: value})
                with self.assertRaises(ValueError):
                    validate_profile([self.row], summary, [self.record], self.policy)

if __name__ == "__main__":
    unittest.main()
