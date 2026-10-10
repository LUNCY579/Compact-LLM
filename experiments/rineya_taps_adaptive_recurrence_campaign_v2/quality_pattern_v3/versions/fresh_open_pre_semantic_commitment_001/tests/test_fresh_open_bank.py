"""Fresh source CPU fixtures only; never production nonce/model/tokenizer."""
import ast
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_open_bank as bank
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_open_verifier as verifier


class FreshOpenBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = bank.build_fixture_bank()
        cls.audit = bank.audit_rows(cls.rows)

    def test_full_fixture_contract_and_all_twelve_positive_controls(self):
        self.assertEqual(self.audit["errors"], [])
        self.assertEqual((self.audit["cases"], self.audit["parents"]), (432, 216))
        self.assertEqual(self.audit["family_counts"], dict.fromkeys(bank.FAMILIES, 72))
        self.assertEqual(len(self.audit["mechanism_counts"]), 12)
        self.assertEqual(set(self.audit["mechanism_counts"].values()), {36})
        self.assertTrue(all(x["constructor_forward_text_independent_text_agree"] for x in self.audit["positive_controls"]))
        self.assertEqual(self.audit["difficulty_counts"], {"easy": 216, "hard": 216})

    def test_fixture_determinism_without_using_scientific_nonce(self):
        self.assertEqual(bank.build_fixture_bank(), self.rows)
        self.assertTrue(all(x["nonce"] == bank.FIXTURE_NONCE and x["exposure"] == "CPU_FIXTURE_ONLY" for x in self.rows))
        self.assertNotEqual(bank.FIXTURE_NONCE, bank.NONCE)
        with self.assertRaises(PermissionError): bank.build_fixture_bank(bank.NONCE)

    def test_actual_corpus_release_guard_precedes_construction(self):
        with patch.object(bank, "_build", side_effect=AssertionError("scientific construction reached")):
            with self.assertRaises(FileNotFoundError):
                bank.generate_registered_bank(bank.STUDY / "reviews/not-a-source-review.json",
                                               bank.STUDY / "not-a-policy-freeze.json")

    def test_balanced_query_roles_truth_status_and_numeric_sign(self):
        for f, counts in self.audit["query_role_counts_by_family"].items():
            self.assertEqual(counts, dict.fromkeys(range(4), 18), f)
        logic = [r["gold"] for r in self.rows if r["family"] == "deduction_composition"]
        self.assertEqual(Counter(logic), {"YES": 36, "NO": 36})
        self.assertEqual(self.audit["source_status_counts_by_family"]["document_retrieval_reconciliation"],
                         {"ANSWER": 36, "MISSING": 18, "CONFLICT": 18})
        self.assertEqual(self.audit["source_status_counts_by_family"]["answerability_source_conflict"],
                         {"ANSWER": 36, "MISSING": 18, "CONFLICT": 18})
        for counts in self.audit["gold_sign_counts_by_numeric_family"].values():
            self.assertEqual(counts, {"negative": 24, "zero": 24, "positive": 24})

    def test_parent_pairs_and_no_variant_answerability_or_style_leak(self):
        evidence = [x for x in self.rows if x["pair_kind"] == "EVIDENCE_TRANSFER"]
        for variant in (0, 1):
            self.assertEqual(Counter(x["gold"] == "UNKNOWN" for x in evidence if x["variant_id"] == variant),
                             {False: 18, True: 18})
        for mechanism in (0, 1):
            for style in (0, 1):
                subset = [r for r in evidence if r["mechanism_index"] == mechanism and r["wording_style"] == style]
                self.assertEqual(sum(r["gold"] == "UNKNOWN" for r in subset) * 2, len(subset))
        self.assertEqual(len({r["semantic_world_sha256"] for r in self.rows}), 216)
        self.assertEqual(len({r["prompt"] for r in self.rows}), 432)

    def test_full_response_scorer_parity_and_no_last_number_repair(self):
        # Only the frozen pure scorer is imported here; no old constructor used.
        from experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_bank import score_response as old_score
        responses = ["1", " 1 ", "+1", "01", "-0", "-7", "YES", "yes", "UNKNOWN", "unknown", "1\n2", "answer 1", "1.0", "", "1\u0662"]
        for kind, gold in (("integer", "1"), ("choice", "YES"), ("integer_or_unknown", "UNKNOWN")):
            case = {"answer_kind": kind, "gold": gold}
            for response in responses:
                actual, expected = bank.score_response(response, case), old_score(response, case)
                for key in ("correct", "format_valid", "canonical_output", "false_UNKNOWN", "unsupported_canonical_answer"):
                    self.assertEqual(actual[key], expected[key], (kind, response, key))
        self.assertFalse(bank.score_response("The result is 1", {"answer_kind": "integer", "gold": "1"})["correct"])
        self.assertTrue(all(r["max_new_tokens"] == 8 and r["tokenizer_shape_status"] == "NOT_BOUND_ROOT_GATE_REQUIRED" for r in self.rows))

    def test_frozen_documents_grammar_recognition_and_unrecognized_holdouts(self):
        from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.opposed_gate_policy import document_certificate
        recognized = [r for r in self.rows if r["grammar_role"] == "SHARED_FROZEN_DOCUMENTS_CONTRACT_NEW_WORLD"]
        self.assertEqual(len(recognized), 72)
        for row in self.rows:
            value = document_certificate(row["prompt"])
            if row in recognized:
                self.assertEqual(value, {"ANSWER": "CONSISTENT", "MISSING": "ABSENT", "CONFLICT": "CONFLICT"}[row["source_status"]])
            else:
                self.assertEqual(value, "UNRECOGNIZED")

    def test_copy_queryblind_header_and_skeleton_floors_complete(self):
        floors = self.audit["floor_correct_counts"]
        self.assertLessEqual(max(floors["overall"].values()), 216)
        self.assertTrue(all(max(floors[f].values()) < 72 for f in bank.FAMILIES))
        self.assertEqual(floors["overall"]["constant_UNKNOWN"], 108)
        self.assertIn("query_blind_first_entity_solver", floors["overall"])
        self.assertIn("any_CLEAR_or_REVOKE_UNKNOWN_else_last_number", floors["overall"])
        self.assertIn("rule_RHS_presence", floors["overall"])
        self.assertEqual(floors["overall"]["copy_full_prompt"], 0)

    def test_horn_backward_proof_and_path_nonempty_semantics(self):
        prompt = bank.HEADERS["horn"] + "\nFacts AAA,BBB.\nRules AAA&BBB>CCC;CCC>DDD;EEE>FFF.\nQuery: DDD. Reply only YES or NO."
        self.assertEqual(verifier.verify_prompt(prompt)["answer"], "YES")
        self.assertEqual(verifier.verify_prompt(prompt.replace("DDD. Reply", "FFF. Reply"))["answer"], "NO")
        path = bank.HEADERS["links"] + "\nAAA>BBB;BBB>CCC.\nQuery: AAA to AAA. Reply only YES or NO."
        self.assertEqual(verifier.verify_prompt(path)["answer"], "NO")
        self.assertEqual(verifier.verify_prompt(path.replace("BBB>CCC", "BBB>AAA"))["answer"], "YES")

    def test_register_expression_snapshot_and_stack_order_underflow(self):
        prompt = bank.HEADERS["register"] + "\nInitial AAA=2,BBB=5.\nSteps 2 BBB=AAA+3;1 AAA=BBB+1;3 IF BBB>8 THEN AAA=BBB+4 ELSE AAA=BBB-2.\nQuery: AAA. Reply one integer."
        self.assertEqual(verifier.verify_prompt(prompt)["answer"], "13")
        stack = bank.HEADERS["stack"] + "\nInitial AAA=[2,3].\nSteps 3 AAA MUL;2 AAA P 4;1 AAA ADD.\nQuery: AAA. Reply one integer."
        self.assertEqual(verifier.verify_prompt(stack)["answer"], "20")
        with self.assertRaises(ValueError): verifier.verify_prompt(stack.replace("[2,3]", "[2]"))
        with self.assertRaises(ValueError): verifier.verify_prompt(prompt.replace("2 BBB=", "1 BBB="))

    def test_owner_state_backward_ack_witnesses(self):
        p = bank.HEADERS["messages"] + "\n1 AAA OFFER 4;2 AAA ACK 4;3 AAA REVOKE;4 AAA ACK 4.\nQuery: AAA. Reply one integer or UNKNOWN."
        self.assertEqual(verifier.verify_prompt(p)["answer"], "UNKNOWN")
        restored = p.replace("2 AAA ACK 4", "2 AAA QUOTE 9")
        self.assertEqual(verifier.verify_prompt(restored)["answer"], "4")  # revoke leaves pending intact
        changed = restored.replace("2 AAA QUOTE 9", "2 AAA OFFER 5")
        self.assertEqual(verifier.verify_prompt(changed)["answer"], "UNKNOWN")

    def test_authority_dates_current_pointer_and_revocation_are_query_specific(self):
        dated = bank.DOCUMENT_HEADER + bank.LATEST + bank.DOCUMENT_RULE + "\nAAA A d1 9;AAA A d2 -4;AAA A d2 -4;BBB A d2 2;BBB A d2 3.\nQuery: AAA. Reply one integer or UNKNOWN."
        self.assertEqual(verifier.verify_prompt(dated)["answer"], "-4")
        self.assertEqual(verifier.verify_prompt(dated.replace("Query: AAA", "Query: BBB"))["source_status"], "CONFLICT")
        snap = bank.HEADERS["snapshots"] + "\nCurrent AAA=v2.\nRows AAA/v2 SIGNED 5;AAA/v9 SIGNED 8;AAA/v2 UNSIGNED 7.\nQuery: AAA. Reply one integer or UNKNOWN."
        self.assertEqual(verifier.verify_prompt(snap)["answer"], "5")  # greatest numeric revision is not current
        att = bank.HEADERS["attestations"] + "\nAuthorized BBB.\nRevoked R2.\nReceipts R1 AAA BBB 5;R2 AAA BBB 6;R3 AAA CCC 7.\nQuery: AAA. Reply one integer or UNKNOWN."
        self.assertEqual(verifier.verify_prompt(att)["answer"], "5")
        self.assertEqual(verifier.verify_prompt(att.replace("Revoked R2", "Revoked R1,R2"))["source_status"], "MISSING")

    def test_verifier_requires_visible_text_full_contract_not_annotations(self):
        row = self.rows[0]
        with self.assertRaises(TypeError): verifier.verify_prompt(row)
        with self.assertRaises(ValueError): verifier.verify_prompt(row["prompt"].replace("Reply one integer.", "Reply only YES or NO."))
        with self.assertRaises(ValueError): verifier.verify_prompt(row["prompt"] + "\ngold=0")
        altered = copy.deepcopy(self.rows); altered[0]["gold"] = "9999"
        self.assertTrue(any("positive control mismatch" in x for x in bank.audit_rows(altered)["errors"]))

    def test_no_old_constructor_import_model_or_tokenizer_dependencies(self):
        for module in (bank, verifier):
            source = Path(module.__file__).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            self.assertFalse(any("bulk_bank" in name or "torch" in name or "transformers" in name for name in imports))
            self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("make_world", "from_pretrained") for n in ast.walk(tree)))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertFalse((bank.STUDY / "data/fresh_open_replication_001").exists())


if __name__ == "__main__":
    unittest.main()
