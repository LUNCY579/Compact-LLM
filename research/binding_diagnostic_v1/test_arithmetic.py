"""Fresh arithmetic semantic fixtures; frozen scorer code only, no old prompts."""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_diagnostic_v1 import arithmetic as a


def prompt(facts, kind, response="unit_explicit"):
    return [{"role": "system", "content": a.SYSTEM},
            {"role": "user", "content": facts + "\n\n" + a.INSTRUCTIONS[kind][response]}]


COUNT = "At Testcove, an inventory has 84 resin discs. Exactly 25 percent are marked violet. How many resin discs are marked violet?"
MONEY = "At Testcove, a price is 38 dollars. A rebate is exactly 15 percent of that price. Determine the rebate amount in dollars."
RATIO = "At Testcove, a coating rule uses 7 grams of powder for every 3 panels. The same rule is used for 8 panels. How many grams of powder are needed?"
CLOCK = "At Testcove, a courier departs at 23:50 on a 24-hour clock. The trip lasts 35 minutes. Determine the arrival clock time. If midnight is crossed, use the time on the following day."
DURATION = "At Testcove, a drying interval begins at 23:55 and ends at 00:25 on the next day. The clock readings use a 24-hour format. Determine the elapsed time in minutes."


class IndependentOracleTests(unittest.TestCase):
    def test_exact_percentage_and_currency_values(self):
        count = a.parse_policy_input(prompt(COUNT, "count"))
        money = a.parse_policy_input(prompt(MONEY, "money"))
        self.assertEqual(Fraction(count["exact_value"]), 21)
        self.assertEqual(Fraction(money["exact_value"]), Fraction(57, 10))
        self.assertEqual(a.reference_text(money), "$5.7")

    def test_nonterminating_ratio_is_an_exact_fraction(self):
        parsed = a.parse_policy_input(prompt(RATIO, "quantity"))
        self.assertEqual(parsed["exact_value"], "56/3")
        self.assertEqual(a.typed_targets(parsed)["strict"], {"kind": "quantity", "value": "56/3", "bare_allowed": False, "unit": "gram"})

    def test_clock_wraparound_is_minute_of_day_not_elapsed_time(self):
        parsed = a.parse_policy_input(prompt(CLOCK, "clock"))
        self.assertEqual(a.reference_text(parsed), "00:25")
        self.assertEqual(a.typed_targets(parsed)["strict"], {"kind": "clock", "minute_of_day": 25})
        numeric = a.parse_policy_input(prompt(CLOCK, "clock", "numeric_value_only"))
        self.assertEqual(a.reference_text(numeric), "25")
        self.assertEqual(a.typed_targets(numeric)["strict"], {"kind": "number", "value": "25", "bare_allowed": True})

    def test_overnight_duration_uses_exact_seconds_in_typed_target(self):
        parsed = a.parse_policy_input(prompt(DURATION, "duration"))
        self.assertEqual(parsed["exact_value"], "30")
        self.assertEqual(a.reference_text(parsed), "30 minutes")
        self.assertEqual(a.typed_targets(parsed)["strict"], {"kind": "duration", "value": "1800", "bare_allowed": False})
        with self.assertRaisesRegex(ValueError, "positive"):
            a.parse_policy_input(prompt(DURATION.replace("next day", "same day"), "duration"))

    def test_wrong_answer_form_instruction_cannot_mislabel_gold(self):
        with self.assertRaisesRegex(ValueError, "response instruction"):
            a.parse_policy_input(prompt(COUNT, "money"))

    def test_invalid_clock_and_fractional_entity_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "24-hour"):
            a.parse_policy_input(prompt(CLOCK.replace("23:50", "23:75"), "clock"))
        with self.assertRaisesRegex(ValueError, "whole"):
            a.parse_policy_input(prompt(COUNT.replace("84 resin", "85 resin"), "count"))


class FrozenGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read CODE only; the pure typed grammar has no filesystem/model imports.
        path = Path(a.FROZEN_SCORER["path"])
        cls.scorer_bytes = path.read_bytes()
        if hashlib.sha256(cls.scorer_bytes).hexdigest() != a.FROZEN_SCORER["sha256"]:
            raise AssertionError("frozen typed grammar changed")
        spec = importlib.util.spec_from_file_location("fresh_arithmetic_test_frozen_grammar", path)
        cls.scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scorer)

    def scores(self, facts, kind, text, response="unit_explicit", eos=True):
        parsed = a.parse_policy_input(prompt(facts, kind, response))
        return {name: self.scorer.score_prediction(text, eos, target)["correct"]
                for name, target in a.typed_targets(parsed).items()}

    def test_numeric_only_requests_do_not_penalize_omitted_units(self):
        for facts, kind, value in ((COUNT, "count", "21"), (MONEY, "money", "5.70"),
                                   (RATIO, "quantity", "56/3"), (CLOCK, "clock", "25"), (DURATION, "duration", "30")):
            with self.subTest(kind=kind):
                self.assertEqual(self.scores(facts, kind, value, "numeric_value_only"), {"permissive": True, "strict": True})
        self.assertEqual(self.scores(MONEY, "money", "$5.70", "numeric_value_only"), {"permissive": False, "strict": False})

    def test_explicit_units_preserve_the_frozen_bare_value_policy(self):
        for facts, kind, bare, annotated in ((COUNT, "count", "21", "21 resin discs"),
                                             (MONEY, "money", "5.7", "$5.70"),
                                             (RATIO, "quantity", "56/3", "56/3 grams")):
            with self.subTest(kind=kind):
                self.assertEqual(self.scores(facts, kind, bare), {"permissive": True, "strict": False})
                self.assertEqual(self.scores(facts, kind, annotated), {"permissive": True, "strict": True})

    def test_clock_duration_confusions_and_conflicting_numbers_fail(self):
        for facts, kind, text in ((CLOCK, "clock", "25 minutes"), (DURATION, "duration", "00:30"),
                                  (DURATION, "duration", "30 hours"), (MONEY, "money", "$5.70 or $6.70"),
                                  (RATIO, "quantity", "56/3 liters"), (COUNT, "count", "21 ceramic pins")):
            with self.subTest(kind=kind, text=text):
                self.assertEqual(self.scores(facts, kind, text), {"permissive": False, "strict": False})

    def test_duration_equivalence_and_clock_meridiem_rules_remain_unchanged(self):
        self.assertEqual(self.scores(DURATION, "duration", "1800 seconds"), {"permissive": True, "strict": True})
        self.assertEqual(self.scores(CLOCK, "clock", "12:25 AM"), {"permissive": True, "strict": True})
        self.assertEqual(self.scores(CLOCK, "clock", "12:25 PM"), {"permissive": False, "strict": False})
        self.assertEqual(self.scores(DURATION, "duration", "30"), {"permissive": False, "strict": False})

    def test_eos_remains_required(self):
        self.assertEqual(self.scores(COUNT, "count", "21 resin discs", eos=False), {"permissive": False, "strict": False})

    def test_every_generated_reference_passes_both_unchanged_policies(self):
        bank = a.build_rows()
        for row in bank["gold"]:
            for policy, target in row["targets"].items():
                with self.subTest(row=row["id"], policy=policy):
                    self.assertTrue(self.scorer.score_prediction(row["reference"], True, target)["correct"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("tokenizers", sys.modules)


class BankContractTests(unittest.TestCase):
    def test_geometry_reproducibility_and_no_policy_factor_fields(self):
        bank = a.build_rows(31)
        self.assertEqual(bank, a.build_rows(31))
        self.assertNotEqual(bank, a.build_rows(32))
        self.assertEqual(a.validate_rows(bank)["worlds_by_subtype"], {name: 16 for name in a.SUBTYPES})
        self.assertEqual(len([v for v in bank["gold"] if v["numeric_variant"] == 0 and v["response_type"] == "unit_explicit"]), 96)
        for row in bank["policy_inputs"]:
            self.assertEqual(set(row), {"id", "parent_id", "prompt"})
            self.assertNotIn("unit_explicit", row["id"])
            self.assertNotIn("numeric_value_only", row["id"])
        json.dumps(bank, allow_nan=False)

    def test_text_oracle_catches_corrupted_gold_and_operands(self):
        bank = a.build_rows()
        altered = copy.deepcopy(bank)
        altered["gold"][0]["oracle"]["exact_value"] = "999999"
        with self.assertRaisesRegex(ValueError, "oracle"):
            a.validate_rows(altered)
        altered = copy.deepcopy(bank)
        text = altered["policy_inputs"][0]["prompt"][1]["content"]
        parsed = a.parse_policy_input(altered["policy_inputs"][0]["prompt"])
        total = parsed["parameters"]["total"]
        altered["policy_inputs"][0]["prompt"][1]["content"] = text.replace(f"has {total} ", f"has {total + 20} ")
        with self.assertRaisesRegex(ValueError, "oracle"):
            a.validate_rows(altered)

    def test_partial_or_mislabeled_world_is_not_384_independent_trials(self):
        bank = a.build_rows()
        altered = copy.deepcopy(bank); altered["gold"][0]["numeric_variant"] = 1
        with self.assertRaisesRegex(ValueError, "two-by-two"):
            a.validate_rows(altered)
        altered = copy.deepcopy(bank); altered["policy_inputs"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 384"):
            a.validate_rows(altered)


if __name__ == "__main__":
    unittest.main()
