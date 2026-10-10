"""In-memory CPU fixtures only; never load source weights or a bank."""
import unittest

from .schema import ABSTENTION, SCHEMA_VERSION
from .scoring import aggregate_rows, score_response


def row(expected=7, *, kind="integer", answerable=True, row_id="r0", world_id="w0",
        variant="canonical", family="answerability", split="fixture"):
    return {"schema_version": SCHEMA_VERSION, "row_id": row_id, "world_id": world_id,
            "pair_id": None, "split": split, "family": family, "variant_id": variant,
            "system": "Fixture only", "user": "Fixture query", "answerable": answerable,
            "expected": expected if answerable else None,
            "contract": {"kind": kind, "abstention": ABSTENTION if kind == "text" else "UNKNOWN"},
            "payload": {}, "operative_key": "fixture:" + world_id,
            "support_expansion": False, "strata": {}}


class ScoringTests(unittest.TestCase):
    def test_integer_semantics_and_format_are_independent(self):
        self.assertTrue(score_response(row(), "7")["strict_correct"])
        result = score_response(row(), "Answer: 7")
        self.assertTrue(result["semantic_correct"])
        self.assertFalse(result["strict_correct"])
        for invalid in ("7.0", "+7", "07", "7 and also 8", "7\nI don't know", "True"):
            self.assertFalse(score_response(row(), invalid)["semantic_correct"], invalid)

    def test_json_full_typed_schema_rejects_duplicates_and_nonfinite(self):
        fixture = row({"A": 7, "B": -2}, kind="json_object")
        self.assertTrue(score_response(fixture, '{"B":-2,"A":7}')["strict_correct"])
        for invalid in ('{"A":7}', '{"A":7,"B":-2,"C":0}', '{"A":7,"B":-2,"A":7}',
                        '{"A":7.0,"B":-2}', '{"A":true,"B":-2}', '{"A":NaN,"B":-2}',
                        '{"A":7,"B":-2} explanation'):
            self.assertFalse(score_response(fixture, invalid)["semantic_correct"], invalid)
        wrapped = score_response(fixture, '```json\n{"A":7,"B":-2}\n```')
        self.assertTrue(wrapped["semantic_correct"])
        self.assertFalse(wrapped["strict_correct"])

    def test_natural_abstention_does_not_require_unknown(self):
        fixture = row(None, answerable=False, kind="text")
        self.assertTrue(score_response(fixture, ABSTENTION)["strict_correct"])
        for text in ("Not provided.", "Not recorded.", "Insufficient information."):
            result = score_response(fixture, text)
            self.assertTrue(result["semantic_correct"])
            self.assertTrue(result["abstained"])
        for text in ("7\nI don't know", "Unknown, but the answer is 7.", "Perhaps it is 7", "Probably unknown"):
            result = score_response(fixture, text)
            self.assertFalse(result["semantic_correct"])
            self.assertTrue(result["unsupported_answer"])

    def test_unnecessary_abstention_and_wrong_supported_answer(self):
        abstained = score_response(row(), "UNKNOWN")
        self.assertTrue(abstained["unnecessary_abstention"])
        self.assertFalse(abstained["unsupported_answer"])
        wrong = score_response(row(), "8")
        self.assertFalse(wrong["semantic_correct"])
        self.assertFalse(wrong["unnecessary_abstention"])

    def test_text_integer_oracle_and_text_answer_format(self):
        self.assertTrue(score_response(row(7, kind="text"), "7")["semantic_correct"])
        response = score_response(row("Alex.", kind="text"), "alex")
        self.assertTrue(response["semantic_correct"])
        self.assertFalse(response["strict_correct"])
        with self.assertRaises(ValueError):
            score_response(row(True), "1")

    def test_complete_coverage_and_duplicate_rejections(self):
        fixtures = [row(), row(row_id="r1", world_id="w1")]
        for outputs in ({"r0": "7"}, {"r0": "7", "r1": "7", "extra": "7"},
                        {"r0": "7", "r1": None},
                        [{"row_id": "r0", "response": "7"}, {"row_id": "r0", "response": "7"}]):
            with self.assertRaises(ValueError):
                aggregate_rows(fixtures, outputs)
        with self.assertRaises(ValueError):
            aggregate_rows([fixtures[0], fixtures[0]], {"r0": "7"})
        with self.assertRaises(ValueError):
            aggregate_rows([], {})

    def test_measured_blank_generation_is_failure_with_full_coverage(self):
        fixtures = [row(), row(None, answerable=False, row_id="r1", world_id="w1")]
        for blank in ("", "  \n"):
            result = aggregate_rows(fixtures, {"r0": "7", "r1": blank})
            self.assertEqual(result["overall"]["n"], 2)
            self.assertEqual(result["overall"]["semantic_accuracy"], .5)
            self.assertEqual(result["overall"]["no_answer"], 1)
            self.assertEqual(result["overall"]["coverage"], .5)
            item = result["primary"]["answerability"][1]
            self.assertFalse(item["semantic_correct"])
            self.assertFalse(item["strict_correct"])
            self.assertFalse(item["unsupported_answer"])
            self.assertTrue(item["no_answer"])

    def test_variants_do_not_inflate_n_and_track_world_disagreement(self):
        fixtures = [row(), row(row_id="r1", variant="natural")]
        result = aggregate_rows(fixtures, {"r0": "7", "r1": "8"})
        self.assertEqual(result["overall"]["n"], 1)
        self.assertEqual(result["variants"]["independent_n"], 0)
        self.assertEqual(result["variants"]["canonical_disagreements"], 1)
        self.assertFalse(result["native_capability_claim"])
        with self.assertRaises(ValueError):
            aggregate_rows([fixtures[1]], {"r1": "7"})

    def test_support_expansion_report_separate(self):
        fixtures = [row(), row(row_id="r1", world_id="w1")]
        fixtures[1]["support_expansion"] = True
        result = aggregate_rows(fixtures, {"r0": "7", "r1": "8"})
        self.assertEqual(result["support_expansion"]["semantic_accuracy"], 0)
        self.assertEqual(result["original_support"]["semantic_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
