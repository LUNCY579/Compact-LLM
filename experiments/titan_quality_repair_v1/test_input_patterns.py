"""Counterexamples for input-dose interpretations; synthetic fixtures only."""
import unittest
from . import generator, registry
from .input_patterns import summarize_input_rows


class InputPatternTests(unittest.TestCase):
    def test_regular_and_counterfactual_unanswerable_fraction_measured(self):
        recipes = registry.build_registry()["recipes"]
        for difficulty in ("regular", "counterfactual"):
            recipe = next(r for r in recipes if r["prompt_coverage"] == "canonical"
                          and r["difficulty"] == difficulty and not r["replay_fraction"])
            rows = generator.training_rows(recipe, 17, [])
            audits = [{"row_id": r["row_id"], "loss_tokens": 2 if not r["answerable"] else 20,
                       "prompt_tokens": 100} for r in rows]
            summary = summarize_input_rows(rows, audits)
            self.assertEqual(summary["unanswerable_examples"], 500)
            self.assertEqual(summary["unanswerable_example_fraction"], .25)
            self.assertFalse(summary["gradient_magnitude_inferred"])
            self.assertFalse(summary["model_gain_verified"])

    def test_token_mass_cannot_be_called_equal_example_mass(self):
        rows = generator.training_rows(registry.build_registry()["recipes"][0], 17, [])
        audits = [{"row_id": r["row_id"], "loss_tokens": 1 if not r["answerable"] else 30,
                   "prompt_tokens": 100} for r in rows]
        result = summarize_input_rows(rows, audits)
        missing = [g for g in result["groups"] if not g["answerable"]]
        self.assertAlmostEqual(sum(g["example_fraction"] for g in missing), .25)
        self.assertLess(sum(g["supervised_token_fraction"] for g in missing), .02)
        self.assertIn("equal example coefficients", result["loss_reduction_interpretation"])

    def test_reordered_token_audit_cannot_be_used_for_input_claim(self):
        rows = generator.training_rows(registry.build_registry()["recipes"][0], 17, [])
        audits = [{"row_id": r["row_id"], "loss_tokens": 1, "prompt_tokens": 100} for r in reversed(rows)]
        with self.assertRaisesRegex(ValueError, "ROW_AUDIT_BINDING"):
            summarize_input_rows(rows, audits)
