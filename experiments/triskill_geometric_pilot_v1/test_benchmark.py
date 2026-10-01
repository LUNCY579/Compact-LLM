"""CPU-only checks of membership, oracle semantics, scoring and paired damage."""
from __future__ import annotations

import ast
import json
import math
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

    def test_format_variants_preserve_semantic_cases_and_stay_development_only(self):
        screen_before = b.build_screen()
        sources = {row.row_id: row for row in b.build_development()
                   if row.output_kind == "integer_or_unknown"}
        variants = b.build_format_variants()
        self.assertEqual(len(variants), 24)
        self.assertEqual(len({row.row_id for row in variants}), 24)
        self.assertEqual({row.metadata["semantic_case_id"] for row in variants}, set(sources))
        for case_id, source in sources.items():
            linked = [row for row in variants if row.metadata["semantic_case_id"] == case_id]
            self.assertEqual({row.output_kind for row in linked}, set(b.ATOM_OUTPUT_KINDS))
            for row in linked:
                self.assertEqual(row.expected, source.expected)
                self.assertEqual(row.user, source.user)
                self.assertEqual(row.world_signature, source.world_signature)
                self.assertEqual(row.state_signature, source.state_signature)
                self.assertEqual(row.rule_signature, source.rule_signature)
                self.assertEqual(row.task_kind, source.task_kind)
                self.assertEqual(row.split, "development")
                self.assertEqual(row.metadata["format_source_row_id"], source.row_id)
                self.assertEqual(row.metadata["format_panel_scoring_unit"], "semantic_case_cluster")
                self.assertEqual(row.metadata["native_format_panel"], "unmeasured_not_authorized")
                self.assertFalse(row.metadata["operator_selection_allowed"])
                self.assertFalse(row.metadata["training"])
                self.assertIn(row.metadata["strict_format_contract"], row.system)
                if row.task_kind == "answerability_parent":
                    self.assertIn("if the requested value cannot be determined", row.system)
                else:
                    self.assertIn("if all admissible final values agree", row.system)
        self.assertEqual(b.build_screen(), screen_before)

    def test_explicit_format_contracts_separate_semantics_from_strict_format(self):
        for row in b.build_format_variants():
            atom = row.expected
            answer = atom if atom == b.UNKNOWN else int(atom)
            responses = {
                "integer_or_unknown": atom,
                "answer_json": json.dumps({"answer": answer}),
                "answer_tag": f"<answer>{atom}</answer>",
                "fenced_atom": f"```text\n{atom}\n```",
            }
            for contract, response in responses.items():
                scored = b.score_response(row, response)
                self.assertTrue(scored["semantic_correct"], (row.row_id, response))
                self.assertEqual(scored["strict_valid"], contract == row.output_kind)
                self.assertEqual(scored["strict_correct"], contract == row.output_kind)
            for response in (
                f"The answer is 86.\n<answer>{atom}</answer>",
                f"<answer>{atom}</answer><answer>86</answer>",
                f"```text\n{atom}\n```\n86",
                f'{{"answer": "{atom}", "other": 86}}',
                f'{{"answer":86,"answer":"{atom}"}}',
            ):
                scored = b.score_response(row, response)
                self.assertFalse(scored["semantic_correct"], response)
                self.assertFalse(scored["strict_valid"], response)
        number_row = next(row for row in b.build_format_variants()
                          if row.output_kind == "answer_json" and row.expected != b.UNKNOWN)
        numeric_string = b.score_response(number_row, json.dumps({"answer": number_row.expected}))
        self.assertTrue(numeric_string["semantic_correct"])
        self.assertFalse(numeric_string["strict_valid"])
        for response in ('{"answer":true}', '{"answer":6.0}', '{"answer":NaN}'):
            self.assertFalse(b.score_response(number_row, response)["strict_valid"])

    def test_semantic_gains_and_format_repairs_are_distinct(self):
        rows = b.build_screen()[4:]
        source = {row.row_id: row.expected for row in rows}
        candidate = dict(source)
        source[rows[0].row_id] = f"Answer: {rows[0].expected}"
        source[rows[1].row_id] = "UNKNOWN"
        candidate[rows[2].row_id] = "86"
        result = b.compare_outputs(rows, source, candidate)["overall"]
        self.assertEqual(result["gains"], 2)
        self.assertEqual(result["source_correct_damage"], 1)
        self.assertEqual(result["semantic_gains"], 1)
        self.assertEqual(result["source_semantic_correct_damage"], 1)
        self.assertEqual(result["semantic_net_gain"], 0)
        self.assertEqual(result["format_repair_only"], 1)
        self.assertEqual(result["strict_paired"]["n01"], 2)
        self.assertEqual(result["semantic_paired"]["n01"], 1)
        self.assertEqual(result["semantic_paired"]["n10"], 1)
        self.assertEqual(result["semantic_paired"]["n11"], 2)
        self.assertEqual(result["semantic_paired"]["n00"], 0)
        self.assertEqual(result["paired_net_rate"], 0.25)
        self.assertEqual(result["semantic_paired_net_rate"], 0)

    def test_paired_tables_and_three_wins_do_not_confirm(self):
        rows = [row for row in b.build_screen() if row.metadata.get("transfer_endpoint")]
        self.assertEqual(len(rows), 3)
        source = {row.row_id: "" for row in rows}
        candidate = {row.row_id: row.expected for row in rows}
        result = b.compare_outputs(rows, source, candidate)["overall"]
        for endpoint in ("strict_paired", "semantic_paired"):
            summary = result[endpoint]
            self.assertEqual({key: summary[key] for key in ("n00", "n01", "n10", "n11")},
                             {"n00": 0, "n01": 3, "n10": 0, "n11": 0})
            self.assertEqual(sum(summary[key] for key in ("n00", "n01", "n10", "n11")), 3)
            self.assertEqual(summary["paired_net_rate"], 1)
            self.assertAlmostEqual(summary["exact_mcnemar_p_two_sided"], 0.25)
            self.assertLess(summary["paired_net_rate_interval_95"][0], 0)
            self.assertEqual(summary["paired_net_rate_interval_95"][1], 1)
        self.assertAlmostEqual(b.exact_paired_mcnemar(3, 0), 0.25)
        self.assertAlmostEqual(b.exact_paired_mcnemar(0, 3), 0.25)
        self.assertEqual(b.exact_paired_mcnemar(0, 0), 1)
        self.assertEqual(b.exact_paired_mcnemar(2, 2), 1)
        # Counts large enough to overflow a float conversion of 2**n remain finite.
        large_p = b.exact_paired_mcnemar(2100, 1900)
        self.assertTrue(math.isfinite(large_p))
        self.assertGreater(large_p, 0)
        self.assertLess(large_p, 0.01)
        for gains, losses in ((-1, 0), (0, -1), (1.0, 0), (True, 0)):
            with self.assertRaises(ValueError):
                b.exact_paired_mcnemar(gains, losses)

    def test_paired_uncertainty_is_conservative_and_empty_is_unmeasured(self):
        diagnostic = b.paired_net_rate_interval(3, 0, 3)
        # Bonferroni uses alpha/4=.0125 in each marginal tail.
        lower_gain = 0.0125 ** (1 / 3)
        self.assertAlmostEqual(diagnostic["gain_probability_interval"][0], lower_gain)
        self.assertAlmostEqual(diagnostic["loss_probability_interval"][1], 1 - lower_gain)
        self.assertAlmostEqual(diagnostic["interval"][0], 2 * lower_gain - 1)
        self.assertEqual(diagnostic["interval"][1], 1)
        self.assertIn("iid", diagnostic["interpretation"])
        empty = b.paired_net_rate_interval(0, 0, 0)
        self.assertIsNone(empty["net_rate"])
        self.assertIsNone(empty["interval"])
        self.assertIsNone(b.clopper_pearson_interval(0, 0))
        self.assertEqual(b.clopper_pearson_interval(0, 3)[0], 0)
        self.assertEqual(b.clopper_pearson_interval(3, 3)[1], 1)
        for counts in ((1, 1, 1), (-1, 0, 1), (0, 0, -1), (0.0, 0, 0)):
            with self.assertRaises(ValueError):
                b.paired_net_rate_interval(*counts)
        for alpha in (-0.1, 0, 1, 1.1):
            with self.assertRaises(ValueError):
                b.clopper_pearson_interval(0, 3, alpha)
        variants = b.build_format_variants()
        no_calls = b.compare_outputs(variants, {}, {})["overall"]
        self.assertEqual(no_calls["paired_measured_n"], 0)
        self.assertEqual(no_calls["semantic_gains"], 0)
        self.assertIsNone(no_calls["semantic_paired_net_rate"])
        for endpoint in ("strict_paired", "semantic_paired"):
            self.assertIsNone(no_calls[endpoint]["paired_net_rate_interval_95"])
            self.assertIsNone(no_calls[endpoint]["exact_mcnemar_p_two_sided"])
            self.assertEqual(sum(no_calls[endpoint][key] for key in ("n00", "n01", "n10", "n11")), 0)

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
