"""CPU-only checks of membership, oracle semantics, scoring and paired damage."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as b


class BenchmarkTests(unittest.TestCase):
    def test_screen_allocation_and_frozen_gold(self):
        rows = b.build_screen()
        self.assertEqual(len(rows), 8)
        self.assertEqual([row.task_kind for row in rows].count("transition_parent"), 2)
        self.assertEqual([row.task_kind for row in rows].count("answerability_parent"), 2)
        self.assertEqual([row.task_kind for row in rows].count("third_skill_transfer"), 3)
        self.assertEqual([row.task_kind for row in rows].count("known_state_control"), 1)
        self.assertEqual(sum(row.metadata.get("transfer_endpoint", False) for row in rows), 3)
        self.assertEqual([row.expected for row in rows], [
            '{"A":-39,"B":30,"C":46,"D":-24}', '{"X":55,"Y":23,"Z":44}',
            "9", "UNKNOWN", "93", "25", "UNKNOWN", "UNKNOWN",
        ])
        self.assertEqual(rows, b.build_screen())
        self.assertTrue(all(row.metadata["training"] is False for row in rows))

    def test_original_transition_system_contracts_verbatim(self):
        # Parse only source constants. No import of legacy generation, no bank reads.
        source = Path(__file__).resolve().parents[1] / "fresh_dual_skill_instrument_v1" / "generate.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        systems = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == "SYSTEM" for target in node.targets))
        self.assertEqual(b.LEDGER_SYSTEM, systems["ledger_replay"])
        self.assertEqual(b.REGISTER_SYSTEM, systems["register_machine"])

    def test_local_membership_without_global_claim(self):
        membership = b.validate_local_membership()
        self.assertTrue(membership["local_disjoint"])
        self.assertFalse(membership["legacy_exclusion_verified"])
        self.assertFalse(membership["protected_confirmation"])
        rows = (*b.build_development(), *b.build_screen(), *b.build_conjunction_controls())
        self.assertEqual(len({row.row_id for row in rows}), len(rows))
        self.assertTrue(all(len(row.world_signature) == 64 for row in rows))

    def test_reset_erasure_known_chain_and_surviving_ambiguity(self):
        rows = b.build_screen()[4:]
        self.assertEqual([row.metadata["outcome_count"] for row in rows], [1, 1, 97, 2])
        self.assertEqual(b.affine_outcomes(None, ((2, -1), (0, 7), (-1, 3))), (93,))
        self.assertEqual(b.affine_outcomes((2, 94), ((2, 1), (-1, 4), (1, -2))), (7, 94))
        self.assertEqual(len(b.affine_outcomes(None, ((2, 3), (-1, 5)))), 97)
        self.assertEqual(rows[1].task_kind, "known_state_control")
        self.assertFalse(rows[1].metadata["transfer_endpoint"])
        self.assertFalse(rows[1].metadata["new_task_structure"])
        self.assertFalse(rows[1].metadata["reset_present"])
        self.assertFalse(rows[1].metadata["initial_conflict"])
        self.assertEqual(b.affine_outcomes((95,), ((-1, 4), (3, -6), (2, 1))), (25,))
        self.assertEqual(b.affine_outcomes((94,), ((-1, 4), (3, -6), (2, 1))), (31,))
        self.assertTrue(b.build_development()[5].metadata["initial_conflict"])
        self.assertTrue(b.build_development()[5].metadata["reset_present"])
        # Modulo-97 scale 97 is a reset even without a literal zero coefficient.
        self.assertEqual(b.affine_outcomes(None, ((97, 12),)), (12,))
        with self.assertRaises(ValueError):
            b.affine_outcomes([], ((1, 0),))
        with self.assertRaises(ValueError):
            b.affine_outcomes([97], ((1, 0),))

    def test_parent_oracle_guard_simultaneity_and_modulus(self):
        ledger = {"before": {"A": 9, "B": -2},
                  "event": {"kind": "transfer", "account": "A", "other": "B", "amount": 4,
                            "status": "pending"}}
        self.assertEqual(b.transition_oracle("ledger_replay", ledger), {"A": 9, "B": -2})
        self.assertEqual(ledger["before"], {"A": 9, "B": -2})
        swap = {"before": {"X": 4, "Y": 92, "Z": 3},
                "event": {"kind": "swap", "register": "X", "other": "Y"}}
        self.assertEqual(b.transition_oracle("register_machine", swap), {"X": 92, "Y": 4, "Z": 3})
        self.assertEqual(swap["before"]["X"], 4)
        conditional = {"before": {"X": 3, "Y": 4},
                       "event": {"kind": "conditional", "register": "X", "other": "Y",
                                 "threshold": 4, "value": 9}}
        self.assertEqual(b.transition_oracle("register_machine", conditional), {"X": 91, "Y": 4})

    def test_strict_vs_semantic_wrapper_diagnostics(self):
        row = b.build_screen()[4]
        self.assertTrue(b.score_response(row, " 93\n")["strict_correct"])
        for wrapped in ('Answer: 93', 'Final answer: 93', '```text\n93\n```', '{"answer":93}'):
            score = b.score_response(row, wrapped)
            self.assertFalse(score["strict_correct"], wrapped)
            self.assertTrue(score["semantic_correct"], wrapped)
        for invalid in ('93 and 86', 'Answer: 86\nFinal answer: 93', '{"answer":true}',
                        '{"answer":86,"answer":93}', '93.0', '0.93e2', '9\u0663',
                        '86\nFinal answer: 93', 'The final value is 86.\nFinal answer: 93'):
            self.assertFalse(b.score_response(row, invalid)["semantic_correct"], invalid)
            self.assertFalse(b.score_response(row, invalid)["strict_correct"], invalid)
        unknown = b.build_screen()[-1]
        self.assertFalse(b.score_response(unknown, "unknown")["strict_correct"])
        self.assertTrue(b.score_response(unknown, "unknown")["semantic_correct"])
        self.assertFalse(b.score_response(unknown, "The value could be 7 or 94.")["semantic_correct"])

    def test_json_schema_strict_types_full_keys_and_wrappers(self):
        row = b.build_screen()[1]
        self.assertTrue(b.score_response(row, '{"Z":44, "Y":23, "X":55}')["strict_correct"])
        wrapped = b.score_response(row, '```json\n{"X":55,"Y":23,"Z":44}\n```')
        self.assertFalse(wrapped["strict_correct"])
        self.assertTrue(wrapped["semantic_correct"])
        for invalid in ('{"X":55,"Y":23}', '{"X":55.0,"Y":23,"Z":44}',
                        '{"X":55,"Y":23,"Z":44,"other":0}',
                        '{"X":0,"X":55,"Y":23,"Z":44}',
                        '{"X":NaN,"Y":23,"Z":44}'):
            self.assertFalse(b.score_response(row, invalid)["strict_correct"], invalid)
            self.assertFalse(b.score_response(row, invalid)["strict_valid"], invalid)
        self.assertFalse(b.score_response(row, '{}')["strict_valid"])
        wrong_value = b.score_response(row, '{"X":0,"Y":23,"Z":44}')
        self.assertTrue(wrong_value["strict_valid"])
        self.assertFalse(wrong_value["strict_correct"])

    def test_conjunction_is_separate_from_transfer(self):
        rows = b.build_conjunction_controls()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.split == "development" for row in rows))
        self.assertTrue(all(row.task_kind == "parent_task_conjunction" for row in rows))
        self.assertTrue(all(row.metadata["new_task_structure"] is False for row in rows))
        self.assertEqual(json.loads(rows[0].expected), {"state": 28, "fact": "UNKNOWN"})
        self.assertTrue(all(b.score_response(row, row.expected)["strict_correct"] for row in rows))

    def test_paired_gain_damage_missing_output_and_uncertainty(self):
        rows = b.build_screen()[4:]
        source = {row.row_id: row.expected for row in rows}
        candidate = {row.row_id: row.expected for row in rows}
        source[rows[0].row_id] = "UNKNOWN"
        candidate[rows[1].row_id] = ""
        result = b.compare_outputs(rows, source, candidate)
        self.assertEqual(result["overall"]["gains"], 1)
        self.assertEqual(result["overall"]["source_correct_damage"], 1)
        self.assertEqual(result["overall"]["net_gain"], 0)
        self.assertEqual(result["overall"]["candidate_strict_valid_n"], 3)
        self.assertEqual(result["overall"]["source_correct_format_damage"], 1)
        self.assertEqual(result["overall"]["source_correct_damage_rate"], 1 / 3)
        self.assertFalse(result["rows"][1]["candidate_missing"])
        self.assertEqual(result["by_answerability"]["answerable"]["n"], 2)
        self.assertEqual(result["by_answerability"]["unanswerable"]["n"], 2)
        self.assertEqual(result["overall"]["exact_paired_mcnemar_p_two_sided"], 1.0)
        self.assertEqual(b.compare_outputs([], {}, {})["overall"]["n"], 0)
        with self.assertRaises(ValueError):
            b.compare_outputs((rows[0], rows[0]), {}, {})
        interval = b.wilson_interval(4, 4)
        self.assertLess(interval[0], 0.6)
        self.assertAlmostEqual(interval[1], 1.0)
        self.assertIsNone(b.wilson_interval(0, 0))
        with self.assertRaises(ValueError):
            b.wilson_interval(2, 1)

    def test_family_retention_and_false_abstention_remain_separate(self):
        rows = b.build_screen()
        source = {row.row_id: row.expected for row in rows}
        candidate = dict(source)
        candidate[rows[2].row_id] = "UNKNOWN"
        candidate[rows[3].row_id] = "42"
        result = b.compare_outputs(rows, source, candidate)
        self.assertEqual(result["by_transition_family"]["ledger_replay"]["n"], 1)
        self.assertEqual(result["by_transition_family"]["register_machine"]["n"], 1)
        self.assertEqual(result["by_family"]["answerability_abstention"]["source_correct_damage"], 2)
        self.assertEqual(result["overall"]["candidate_false_abstentions"], 1)
        self.assertEqual(result["overall"]["candidate_unsupported_answers"], 1)

    def test_absent_calls_are_unmeasured_and_cannot_manufacture_gains(self):
        row = b.build_screen()[4]
        result = b.compare_outputs((row,), {}, {row.row_id: row.expected})
        self.assertEqual(result["overall"]["gains"], 0)
        self.assertEqual(result["overall"]["paired_measured_n"], 0)
        self.assertIsNone(result["rows"][0]["source_correct"])
        self.assertIsNone(result["overall"]["exact_paired_mcnemar_p_two_sided"])
        result = b.compare_outputs((row,), {row.row_id: row.expected}, {})
        self.assertEqual(result["overall"]["source_correct_damage"], 0)
        self.assertIsNone(result["overall"]["candidate_accuracy_interval_95"])
        self.assertIsNone(result["overall"]["source_correct_damage_rate"])


if __name__ == "__main__":
    unittest.main()
