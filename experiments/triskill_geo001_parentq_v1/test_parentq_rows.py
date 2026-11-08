"""CPU tests of the PARENTQ qualification rows (no model)."""
import json
import re
import unittest
from collections import Counter
from pathlib import Path

import geo_frozen
import parentq_decide
import parentq_rows
import parentq_worker

HERE = Path(__file__).resolve().parent
benchmark = geo_frozen.load("benchmark")


class RowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = parentq_rows.build_bank(benchmark)

    def test_deterministic_and_equal_to_frozen_membership(self):
        again = parentq_rows.build_bank(benchmark)
        rows = [row.to_dict() for row in parentq_rows.all_rows(self.bank)]
        self.assertEqual(rows, [row.to_dict() for row in parentq_rows.all_rows(again)])
        self.assertEqual(rows, json.loads((HERE / parentq_worker.BANK_JSON).read_text(encoding="utf-8")))

    def test_composition(self):
        self.assertEqual(Counter(parentq_rows.subtype(r) for r in self.bank["T"]),
                         {name: parentq_rows.T_PER_SUBTYPE for name in parentq_rows.T_SUBTYPES})
        self.assertEqual(Counter(parentq_rows.subtype(r) for r in self.bank["H_ON"]), parentq_rows.H_KINDS)
        self.assertEqual(Counter((r.metadata["h_row_kind"], r.metadata["contract"]) for r in self.bank["H_OFF"]),
                         {(kind, f"h_training_prompt_{label}"): count // 2
                          for kind, count in parentq_rows.H_KINDS.items() for label in ("native_a", "native_b")})

    def test_t_rows_are_screen_constructed_in_support_and_correct(self):
        for row in self.bank["T"]:
            self.assertEqual(benchmark.t_support_violations(row), [], row.row_id)
            payload = json.loads(row.user)
            family = row.metadata["transition_family"]
            self.assertEqual(json.loads(row.expected), benchmark.transition_oracle(family, payload))
            self.assertEqual(row.system, benchmark.LEDGER_SYSTEM if family == "ledger_replay" else benchmark.REGISTER_SYSTEM)
            self.assertEqual(row.output_kind, "json_object")
            if family == "ledger_replay":
                self.assertEqual(payload["event"]["status"], "posted")
                self.assertTrue(all(-30 <= value <= 40 for value in payload["before"].values()))

    def test_h_rows_use_the_screen_contract_and_off_rows_only_change_the_system_prompt(self):
        on = {row.row_id: row for row in self.bank["H_ON"]}
        for row in self.bank["H_ON"]:
            self.assertEqual(row.system, benchmark.ANSWERABILITY_SYSTEM_H)
            self.assertEqual(row.expected == benchmark.UNKNOWN, row.metadata["h_row_kind"] != "answerable")
        for row in self.bank["H_OFF"]:
            source = on[row.metadata["on_row_id"]]
            self.assertIn(row.system, parentq_rows.H_NATIVE_PROMPTS.values())
            self.assertNotIn("reply exactly", row.system)
            self.assertEqual((row.user, row.expected, row.output_kind), (source.user, source.expected, source.output_kind))
        self.assertEqual(len({row.row_id for row in parentq_rows.all_rows(self.bank)}), 88)

    def test_worlds_are_fresh(self):
        geo_rows = (*benchmark.build_screen(), *benchmark.build_development(), *benchmark.build_legacy_screen(),
                    *benchmark.build_legacy_development(), *benchmark.build_conjunction_controls())
        words = set(re.findall(r"[a-z]+", " ".join(f"{r.system} {r.user} {r.row_id}" for r in geo_rows).lower()))
        self.assertEqual(len(set(parentq_rows.WORLDS)), len(parentq_rows.WORLDS))
        self.assertEqual([w for w in parentq_rows.WORLDS if w.lower() in words], [])

    def test_screen_composition_matches_the_decision_rule(self):
        groups = benchmark.row_groups(benchmark.build_screen())
        self.assertEqual(tuple(parentq_decide.subtype(r) for r in groups["T"]), parentq_rows.T_SUBTYPES)
        self.assertEqual([parentq_decide.subtype(r) for r in groups["H"]], ["answerable", "answerable", "distractor", "cue"])
        self.assertEqual(parentq_decide.T_SUBTYPES, parentq_rows.T_SUBTYPES)
        self.assertEqual(set(parentq_decide.H_KINDS), set(parentq_rows.H_KINDS))
        for row in (*self.bank["T"], *self.bank["H_ON"]):
            self.assertEqual(parentq_decide.subtype(row), parentq_rows.subtype(row))

    def test_canary_rows_discriminate_the_three_arms(self):
        canary = json.loads((HERE / parentq_worker.CANARY_JSON).read_text(encoding="utf-8"))
        self.assertEqual(Counter(r["row_type"] for r in canary["rows"]), {"answerable": 4, "boundary": 4})
        for row in canary["rows"]:
            self.assertEqual(len(set(row["stored"].values())), 3)
            self.assertLessEqual(max(row["stored_generated_tokens"].values()), 31)
            self.assertTrue(row["row_id"].startswith("canary:"))


if __name__ == "__main__":
    unittest.main()
