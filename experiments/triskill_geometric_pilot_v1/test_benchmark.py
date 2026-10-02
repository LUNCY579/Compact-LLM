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
        rows = b.build_legacy_screen()
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
        self.assertEqual(rows, b.build_legacy_screen())
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
        rows = (*b.build_legacy_development(), *b.build_legacy_screen(), *b.build_conjunction_controls())
        self.assertEqual(len({row.row_id for row in rows}), len(rows))
        self.assertTrue(all(len(row.world_signature) == 64 for row in rows))

    def test_reset_erasure_known_chain_and_surviving_ambiguity(self):
        rows = b.build_legacy_screen()[4:]
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
        self.assertTrue(b.build_legacy_development()[5].metadata["initial_conflict"])
        self.assertTrue(b.build_legacy_development()[5].metadata["reset_present"])
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
        row = b.build_legacy_screen()[4]
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
        unknown = b.build_legacy_screen()[-1]
        self.assertFalse(b.score_response(unknown, "unknown")["strict_correct"])
        self.assertTrue(b.score_response(unknown, "unknown")["semantic_correct"])
        self.assertFalse(b.score_response(unknown, "The value could be 7 or 94.")["semantic_correct"])

    def test_json_schema_strict_types_full_keys_and_wrappers(self):
        row = b.build_legacy_screen()[1]
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
        screen_before = b.build_legacy_screen()
        sources = {row.row_id: row for row in b.build_legacy_development()
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
        self.assertEqual(b.build_legacy_screen(), screen_before)

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
        rows = b.build_legacy_screen()[4:]
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
        rows = [row for row in b.build_legacy_screen() if row.metadata.get("transfer_endpoint")]
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
        rows = b.build_legacy_screen()[4:]
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
        rows = b.build_legacy_screen()
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
        row = b.build_legacy_screen()[4]
        result = b.compare_outputs((row,), {}, {row.row_id: row.expected})
        self.assertEqual(result["overall"]["gains"], 0)
        self.assertEqual(result["overall"]["paired_measured_n"], 0)
        self.assertIsNone(result["rows"][0]["source_correct"])
        self.assertIsNone(result["overall"]["exact_paired_mcnemar_p_two_sided"])
        result = b.compare_outputs((row,), {row.row_id: row.expected}, {})
        self.assertEqual(result["overall"]["source_correct_damage"], 0)
        self.assertIsNone(result["overall"]["candidate_accuracy_interval_95"])
        self.assertIsNone(result["overall"]["source_correct_damage_rate"])


RECORDS = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
OK = {"padding_parity_ok": True, "liveness_ok": True}


def gold_reply(row):
    return row.expected if row.output_kind == "json_object" else row.metadata["strict_reply"]


def wrong_reply(row):
    if row.output_kind == "json_object":
        return "{}"
    return "0" if row.expected == b.UNKNOWN else b.ABSTAIN_PHRASE


class SectionHBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.rows = b.build_screen()
        self.groups = b.row_groups(self.rows)
        self.by_id = {row.row_id: row for row in self.rows}

    # ---------- rows ----------
    def test_screen_composition_balance_and_golds(self):
        self.assertEqual(len(self.rows), 18)
        self.assertEqual({group: len(rows) for group, rows in self.groups.items()}, {"T": 4, "H": 4, "J": 8, "K": 2})
        self.assertEqual([row.expected for row in self.groups["J"]],
                         ["18", "76", "UNKNOWN", "UNKNOWN", "57", "62", "UNKNOWN", "UNKNOWN"])
        self.assertEqual([row.expected for row in self.groups["K"]], ["3", "64"])
        answerable = [row for row in self.groups["J"] if row.expected != b.UNKNOWN]
        unanswerable = [row for row in self.groups["J"] if row.expected == b.UNKNOWN]
        meta = lambda rows, key: sum(bool(row.metadata[key]) for row in rows)
        self.assertEqual((meta(answerable, "reset_visible"), meta(answerable, "reset_disguised"),
                          meta(answerable, "collision_answerable")), (1, 1, 2))
        self.assertEqual(meta(unanswerable, "reset_present"), 0)
        self.assertEqual(meta(unanswerable, "decoy_multiplier"), 2)
        # Conditional rules appear on both sides, so their presence does not decide answerability.
        self.assertEqual((meta(answerable, "conditional_present"), meta(unanswerable, "conditional_present")), (2, 2))
        self.assertEqual(meta(self.groups["J"], "initial_missing"), 4)
        for row in (*self.groups["J"], *self.groups["K"]):
            outcomes = b.chain_outcomes(row.metadata["initial"], [tuple(rule) for rule in row.metadata["rules"]])
            self.assertEqual(row.expected, str(outcomes[0]) if len(outcomes) == 1 else b.UNKNOWN)
        golds = {row.expected for row in answerable}
        constants = {23} | {rule[1] % 97 for row in self.groups["J"] for rule in row.metadata["rules"]}
        self.assertFalse(golds & {str(value) for value in constants})
        self.assertEqual(self.rows, b.build_screen())
        self.assertTrue(all(row.metadata["training"] is False for row in self.rows))
        self.assertEqual([row.metadata["h_row_kind"] for row in self.groups["H"]],
                         ["answerable", "answerable", "distractor", "cue"])
        self.assertTrue(all(not row.user.startswith("Complete") for row in self.groups["H"]))

    def test_conditional_chain_rule_semantics_and_collisions(self):
        rule = ("if_above", 50, -20)
        self.assertEqual((b.chain_step(70, rule), b.chain_step(30, rule), b.chain_step(50, rule)), (50, 50, 70))
        self.assertEqual(b.chain_outcomes((70, 30), [rule, (3, 4)]), (57,))
        self.assertEqual(b.chain_outcomes((40, 75), [("if_above", 50, 10), (2, 3)]), (63, 76))
        self.assertEqual(b.chain_outcomes(None, [(291, 30)]), (30,))
        self.assertEqual(len(b.chain_outcomes(None, [(195, 6)])), 97)
        for bad in ([("if_below", 1, 2)], [(1.0, 2)], [("if_above", 1)]):
            with self.assertRaises(ValueError):
                b.chain_outcomes((1,), bad)
        sorrel = self.by_id["h-screen-C-sorrel"]
        self.assertIn("if x is greater than 50; otherwise (x - (-20)) modulo 97.", sorrel.user)

    def test_t_rows_inside_trained_support_and_check_can_fail(self):
        for row in (*b.build_screen(), *b.build_development()):
            if row.task_kind == "transition_parent":
                self.assertEqual(b.t_support_violations(row), [], row.row_id)
        legacy = [row for row in b.build_legacy_screen() if row.task_kind == "transition_parent"]
        self.assertIn("amount", b.t_support_violations(legacy[0]))
        register = b.t_support_violations(legacy[1])
        self.assertIn("value", register)
        self.assertTrue(any(problem.startswith("event keys") for problem in register))
        wrapping = b._transition_h("screen", 9, "register_machine", {
            "before": {"X": 40, "Y": 88, "Z": 13},
            "event": {"kind": "add", "other": "Z", "register": "Y", "threshold": 61, "tick": 34, "value": 11}})
        self.assertEqual(b.t_support_violations(wrapping), ["modulo wrap"])

    def test_registered_contracts_match_their_sources(self):
        rule = RECORDS / "hskill-views-002" / "boundary_rule.py"
        views = RECORDS / "hskill-views-002" / "HSKILL_VIEWS_V2.jsonl"
        if not rule.exists() or not views.exists():
            self.skipTest("records checkout absent")
        import hashlib
        self.assertEqual(hashlib.sha256(rule.read_bytes()).hexdigest(), b.BOUNDARY_RULE_SHA256)
        namespace = {}
        exec(compile(rule.read_text(encoding="utf-8"), str(rule), "exec"), namespace)
        self.assertEqual(tuple(namespace["ABSTAIN_PATTERNS"]), b.ABSTAIN_PATTERNS)
        for text in ("I don't know based on the available information.", "Which city?", "No, 42", "42", ""):
            self.assertEqual(namespace["abstains"](text), any(p.search(text.strip().lower()) for p in b._ABSTAIN), text)
        systems = set()
        for line in views.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("row_type") == "answerable":
                systems.update(m["content"] for m in record["prompt"] if m["role"] == "system")
        self.assertIn(b.ANSWERABILITY_SYSTEM_H, systems)

    # ---------- scoring ----------
    def test_credited_strict_and_semantic_scoring(self):
        onyx, quartz, ledger = self.by_id["h-screen-C-onyx"], self.by_id["h-screen-C-quartz"], self.groups["T"][0]
        cases = [
            (onyx, "18", (True, True)), (onyx, " 18\n", (True, True)), (onyx, "18.", (True, False)),
            (onyx, "Answer: 18", (False, False)), (onyx, "18?", (False, False)),
            (onyx, b.ABSTAIN_PHRASE, (False, False)), (onyx, "UNKNOWN", (False, False)),
            (quartz, b.ABSTAIN_PHRASE, (True, True)), (quartz, "UNKNOWN", (True, False)),
            (quartz, "unknown.", (True, False)), (quartz, "The available information is insufficient.", (True, False)),
            (quartz, "It cannot be determined.", (True, False)), (quartz, "Cannot determine.", (True, False)),
            (quartz, "The final value is unknown.", (True, False)), (quartz, "Final values differ.", (True, False)),
            (quartz, "Indeterminate.", (True, False)),
            (quartz, "I don\u2019t know based on the available information.", (True, False)),
            # A hedged number is still an answer on an abstention row.
            (quartz, "No, the final value is 5", (False, False)), (quartz, "5?", (False, False)),
            (quartz, "It might be 5", (False, False)), (quartz, "The value 74 is unknown", (False, False)),
            (quartz, "42", (False, False)), (quartz, "", (False, False)),
            (ledger, ledger.expected, (True, True)), (ledger, ledger.expected + ".", (True, False)),
            (ledger, json.dumps(json.loads(ledger.expected)), (True, True)),
            (ledger, "```json\n" + ledger.expected + "\n```", (False, False)),
        ]
        for row, response, (credited, strict) in cases:
            scored = b.score_response(row, response)
            self.assertEqual((scored["credited"], scored["strict_correct"]), (credited, strict), (row.row_id, response))
        self.assertTrue(b.score_response(onyx, "Answer: 18")["semantic_correct"])

    def test_floors_leave_headroom_only_for_the_dependent_computation(self):
        floors = b.screen_floors(self.rows)
        self.assertEqual(floors["oracle"], {"J": 8, "K": 2})
        self.assertEqual(floors["always_abstain"]["J"], 4)
        self.assertEqual(floors["copy_marker_23"]["J"], 0)
        self.assertLessEqual(floors["best_constant_integer"]["J"], 1)
        for name in ("visible_zero_cue", "literal_0_or_97_cue", "magnitude_ge_97_cue"):
            self.assertEqual(floors[name]["J"], 5, name)
        # Even recognizing every multiple of 97 misses both conditional collisions.
        self.assertEqual(floors["affine_reset_oracle"]["J"], 6)
        shortcut = max(value["J"] for name, value in floors.items() if name != "oracle")
        self.assertEqual(shortcut, 6)

    # ---------- parent gate ----------
    def _scores(self, outputs):
        return {arm: {row_id: b.score_response(self.by_id[row_id], text) for row_id, text in replies.items()}
                for arm, replies in outputs.items()}

    def _outputs(self):
        gold = {row.row_id: gold_reply(row) for row in self.rows}
        base = dict(gold)
        for row in (*self.groups["T"], *self.groups["J"]):
            base[row.row_id] = wrong_reply(row)
        for row in self.groups["H"]:
            if row.expected == b.UNKNOWN:
                base[row.row_id] = "27"
        th = dict(gold)
        for row in self.groups["J"]:
            if row.expected != b.UNKNOWN:
                th[row.row_id] = "0"
        outputs = {"BASE": base, "T": dict(gold), "H": dict(gold)}
        for row in self.groups["J"]:
            outputs["T"][row.row_id] = outputs["H"][row.row_id] = th[row.row_id]
        for arm in ("TH", "IDENTITY", "ROTATION", "REVERSE", "RANDOM"):
            outputs[arm] = dict(th)
        chains = {row.row_id for row in (*self.groups["J"], *self.groups["K"])}
        outputs["PROMPT"] = {row_id: text for row_id, text in th.items() if row_id in chains}
        return outputs, gold

    def _gate(self, outputs):
        return b.parent_gate(self.rows, self._scores(outputs))

    def test_parent_gate_thresholds_are_pinned(self):
        outputs, gold = self._outputs()
        self.assertTrue(self._gate(outputs)["passed"])
        t_rows = [row.row_id for row in self.groups["T"]]
        two = {**outputs, "T": {**outputs["T"], **{row_id: "{}" for row_id in t_rows[2:]}}}
        self.assertTrue(self._gate(two)["T"]["passed"])
        one = {**outputs, "T": {**outputs["T"], **{row_id: "{}" for row_id in t_rows[1:]}}}
        self.assertEqual(self._gate(one)["stop"], "STOP_SCREEN_NO_PARENT_SIGNAL:T")
        base_two = {**two, "BASE": {**outputs["BASE"], **{row_id: gold[row_id] for row_id in t_rows[:2]}}}
        self.assertFalse(self._gate(base_two)["T"]["passed"])
        # H above BASE but without a single missing-row abstention must fail.
        h_missing = [row.row_id for row in self.groups["H"] if row.expected == b.UNKNOWN]
        h_answerable = [row.row_id for row in self.groups["H"] if row.expected != b.UNKNOWN]
        no_abstain = {**outputs, "H": {**outputs["H"], **{row_id: "5" for row_id in h_missing}},
                      "BASE": {**outputs["BASE"], **{row_id: "0" for row_id in h_answerable}}}
        gate = self._gate(no_abstain)
        self.assertGreater(gate["H"]["credited"], gate["H"]["base_credited"])
        self.assertEqual(gate["stop"], "STOP_SCREEN_NO_PARENT_SIGNAL:H")
        one_abstain = {**no_abstain, "H": {**no_abstain["H"], h_missing[0]: b.ABSTAIN_PHRASE}}
        self.assertTrue(self._gate(one_abstain)["H"]["passed"])

    def test_base_ceiling_is_a_separate_stop(self):
        outputs, gold = self._outputs()
        ceiling = {**outputs, "BASE": {**outputs["BASE"], **{row.row_id: gold[row.row_id] for row in self.groups["H"]}}}
        self.assertEqual(self._gate(ceiling)["stop"], "STOP_SCREEN_PARENT_CEILING:H")

    # ---------- verdict ladder ----------
    def test_not_reached_is_distinct_from_failed(self):
        outputs, _ = self._outputs()
        decide = lambda out, mech: b.decide_screen(self.rows, out, mech, stopped_by="X")["status"]
        self.assertEqual(decide(outputs, {}), "STOPPED_BEFORE:padding")
        self.assertEqual(decide(outputs, {"padding_parity_ok": False}), "STOP_NATIVE_MECHANICS_PADDING")
        self.assertEqual(decide({"BASE": outputs["BASE"]}, {"padding_parity_ok": True}), "STOPPED_BEFORE:parent_gate")
        self.assertEqual(decide(outputs, {"padding_parity_ok": True}), "STOPPED_BEFORE:liveness")
        self.assertEqual(decide(outputs, {"padding_parity_ok": True, "liveness_ok": False}), "STOP_MECHANICS_HOOK_LIVENESS")
        verdict = b.decide_screen(self.rows, outputs, {}, stopped_by="STOP_HOST_ACTIVE_RESERVE")
        self.assertEqual(verdict["stopped_by"], "STOP_HOST_ACTIVE_RESERVE")
        self.assertIn("exact_mcnemar_p_two_sided", verdict["rotation_vs_TH_on_J"])

    def test_identity_canary_fires_even_when_later_arms_are_missing(self):
        outputs, _ = self._outputs()
        stopped = {arm: replies for arm, replies in outputs.items() if arm not in ("ROTATION", "REVERSE", "RANDOM", "PROMPT")}
        stopped["IDENTITY"] = {**stopped["IDENTITY"], self.groups["K"][0].row_id: "99"}
        self.assertEqual(b.decide_screen(self.rows, stopped, OK)["status"], "STOP_IDENTITY_CANARY")
        del stopped["IDENTITY"]
        self.assertEqual(b.decide_screen(self.rows, stopped, OK)["status"], "INCOMPLETE_COVERAGE")

    def test_undetectable_is_defined_on_dependent_rows(self):
        outputs, _ = self._outputs()
        self.assertEqual(b.decide_screen(self.rows, outputs, OK)["status"], "UNDETECTABLE_ON_J")
        t_row = self.groups["T"][0]
        moved = {**outputs, "ROTATION": {**outputs["ROTATION"], t_row.row_id: json.dumps(json.loads(t_row.expected))}}
        verdict = b.decide_screen(self.rows, moved, OK)
        self.assertEqual(verdict["status"], "UNDETECTABLE_ON_J")
        self.assertEqual(verdict["texts_differ_from_TH"]["ROTATION"], {"J": 0, "all": 1})

    def _signal(self):
        outputs, gold = self._outputs()
        return {**outputs, "ROTATION": {**outputs["ROTATION"], **{row.row_id: gold[row.row_id] for row in self.groups["J"]}}}, gold

    def test_signal_is_reachable_and_each_condition_can_fail(self):
        signal, gold = self._signal()
        verdict = b.decide_screen(self.rows, signal, OK)
        self.assertEqual(verdict["status"], "SCREEN_SIGNAL_EXPLORATORY")
        self.assertEqual(len(verdict["vs_TH"]["signal_gains"]), 4)
        self.assertFalse(verdict["confirmation"])
        reasons = lambda out: b.decide_screen(self.rows, out, OK)["reasons"]
        for comparator in b.J_COMPARATORS:  # every comparator individually blocks S1
            tie = {**signal, comparator: {**signal[comparator], **{row_id: signal["ROTATION"][row_id]
                                                                   for row_id in signal[comparator]
                                                                   if row_id in {r.row_id for r in self.groups["J"]}}}}
            if comparator in ("IDENTITY", "TH"):
                # IDENTITY must equal TH, so moving either fires the identity canary first;
                # a ROTATION tie with TH itself is excluded by S2 (no gains).
                self.assertEqual(b.decide_screen(self.rows, tie, OK)["status"], "STOP_IDENTITY_CANARY")
                continue
            self.assertIn("S1_above_every_comparator", reasons(tie), comparator)
        for group in ("T", "H", "K"):  # S4 covers every retention group
            row = self.groups[group][0]
            lost = {**signal, "ROTATION": {**signal["ROTATION"], row.row_id: wrong_reply(row)}}
            self.assertIn("S4_no_parent_or_known_control_loss_vs_TH", reasons(lost), group)

    def test_gains_equal_to_losses_fail_s2(self):
        outputs, gold = self._outputs()
        answerable = [row for row in self.groups["J"] if row.expected != b.UNKNOWN]
        unanswerable = [row for row in self.groups["J"] if row.expected == b.UNKNOWN]
        rot = dict(outputs["ROTATION"])
        rot[answerable[0].row_id] = gold[answerable[0].row_id]      # one gain
        rot[unanswerable[0].row_id] = "5"                          # one loss
        verdict = b.decide_screen(self.rows, {**outputs, "ROTATION": rot}, OK)
        self.assertIn("S2_net_signal_gain_vs_TH", verdict["reasons"])
        self.assertEqual(verdict["status"], "NO_SIGNAL")

    def test_pure_abstention_or_hedged_wording_cannot_pass(self):
        outputs, gold = self._outputs()
        th = {**outputs["TH"], **{row.row_id: "5" for row in self.groups["J"]}}
        for arm in ("TH", "IDENTITY", "REVERSE", "RANDOM", "T", "H"):
            outputs[arm] = {**outputs[arm], **{row.row_id: th[row.row_id] for row in self.groups["J"]}}
        outputs["PROMPT"] = {**outputs["PROMPT"], **{row.row_id: th[row.row_id] for row in self.groups["J"]}}
        outputs["ROTATION"] = {**outputs["TH"], **{row.row_id: b.ABSTAIN_PHRASE for row in self.groups["J"]}}
        verdict = b.decide_screen(self.rows, outputs, OK)
        self.assertEqual(verdict["status"], "NO_SIGNAL")
        self.assertIn("S5_gains_a_computed_answer", verdict["reasons"])
        self.assertIn("S6_above_always_abstain_floor", verdict["reasons"])
        hedged = {**outputs, "ROTATION": {**outputs["TH"], self.groups["J"][2].row_id: "No, 5"}}
        self.assertEqual(b.decide_screen(self.rows, hedged, OK)["status"], "NO_SIGNAL")

    def test_blanket_abstention_shift_cannot_pass(self):
        outputs, gold = self._outputs()
        answerable = [row for row in self.groups["J"] if row.expected != b.UNKNOWN]
        th = dict(outputs["TH"])
        for row in self.groups["J"]:
            th[row.row_id] = "5" if row.expected == b.UNKNOWN else gold[row.row_id]
        th[answerable[0].row_id] = "0"
        for arm in ("TH", "IDENTITY", "REVERSE", "RANDOM"):
            outputs[arm] = dict(th)
        outputs["PROMPT"] = {row_id: th[row_id] for row_id in outputs["PROMPT"]}
        for arm in ("T", "H"):
            outputs[arm].update({row.row_id: th[row.row_id] for row in self.groups["J"]})
        outputs["ROTATION"] = {**th, **{row.row_id: b.ABSTAIN_PHRASE for row in self.groups["J"]}}
        verdict = b.decide_screen(self.rows, outputs, OK)
        self.assertEqual(verdict["status"], "NO_SIGNAL")
        self.assertIn("S3_no_answerable_chain_loss_vs_TH", verdict["reasons"])

    def test_format_and_parent_repairs_are_not_signal(self):
        outputs, gold = self._outputs()
        answerable = [row.row_id for row in self.groups["J"] if self.by_id[row.row_id].expected != b.UNKNOWN]
        formatted = {**outputs}
        for arm in ("TH", "IDENTITY", "REVERSE", "RANDOM", "T", "H"):
            formatted[arm] = {**outputs[arm], **{row_id: f"Answer: {gold[row_id]}" for row_id in answerable}}
        formatted["PROMPT"] = {**outputs["PROMPT"], **{row_id: f"Answer: {gold[row_id]}" for row_id in answerable}}
        formatted["ROTATION"] = {**formatted["TH"], **{row_id: gold[row_id] for row_id in answerable}}
        verdict = b.decide_screen(self.rows, formatted, OK)
        self.assertEqual(verdict["status"], "NO_SIGNAL")
        self.assertEqual(sorted(verdict["vs_TH"]["format_repairs"]), sorted(answerable))
        signal, _ = self._signal()
        repaired = {**signal, "T": {**signal["T"], **{row_id: gold[row_id] for row_id in answerable}}}
        verdict = b.decide_screen(self.rows, repaired, OK)
        self.assertEqual(sorted(verdict["vs_TH"]["parent_repairs"]), sorted(answerable))
        self.assertIn("S5_gains_a_computed_answer", verdict["reasons"])

    def test_unknown_row_ids_are_ignored_not_fatal(self):
        signal, _ = self._signal()
        signal = {**signal, "ROTATION": {**signal["ROTATION"], "not-a-row": "1"}}
        verdict = b.decide_screen(self.rows, signal, OK)
        self.assertEqual(verdict["ignored_unknown_row_ids"], ["not-a-row"])
        self.assertEqual(verdict["status"], "SCREEN_SIGNAL_EXPLORATORY")

    def test_membership_is_disjoint_from_development_and_legacy_rows(self):
        membership = b.validate_local_membership()
        self.assertTrue(membership["local_disjoint"])
        self.assertTrue(membership["legacy_rows_disjoint"])
        self.assertEqual((membership["screen_n"], membership["development_n"]), (18, 8))

    def test_t_rows_do_not_collide_with_exposed_banks_or_training_and_canary_does(self):
        import freeze_sectionh as fs
        if not fs.DELTA_T_TRAINING[0].exists():
            self.skipTest("records checkout absent")
        rows = [row for row in (*b.build_screen(), *b.build_development()) if row.task_kind == "transition_parent"]
        result = fs.t_collisions(rows)
        self.assertTrue(result["collision_free"], result["hits"])
        self.assertEqual(result["scanned_payloads"], 1408)
        bank = json.loads(fs.EXPOSED_T_BANKS[0].read_text(encoding="utf-8"))["rows"]
        source = next(row for row in bank if row["family"] == "register_machine")
        content = source["prompt"][-1]["content"] if isinstance(source["prompt"], list) else source["prompt"]
        canary = b._transition_h("development", 9, "register_machine", json.loads(content))
        hits = {kind for _, kind in fs.t_collisions([canary])["hits"][canary.row_id]}
        self.assertEqual(hits, {"payload", "operative", "before_answer"})

    def test_credited_paired_table_is_reported(self):
        rows = self.groups["J"]
        source = {row.row_id: wrong_reply(row) for row in rows}
        candidate = {row.row_id: gold_reply(row) for row in rows}
        candidate[rows[0].row_id] = "18."
        result = b.compare_outputs(rows, source, candidate)["overall"]
        self.assertEqual(result["credited_paired"]["n01"], 8)
        self.assertEqual(result["strict_paired"]["n01"], 7)


if __name__ == "__main__":
    unittest.main()
