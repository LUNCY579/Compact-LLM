import json
import unittest
from experiments.rineya_centralized_reasoning_system_v1.runtime_worker import request
from experiments.rineya_centralized_reasoning_system_v1.runner import irrelevant_snapshot
from experiments.rineya_centralized_reasoning_system_v1.core import supplied_snapshot
from experiments.rineya_centralized_reasoning_system_v1.scoring import score_one


class PublicBoundaryTests(unittest.TestCase):
    def test_aggregate_integer_and_failure_abstention_scoring(self):
        self.assertTrue(score_one({"status": "SUPPORTED", "value": 2}, {"status": "SUPPORTED", "value": 2})["semantic_exact"])
        self.assertFalse(score_one({"status": "SUPPORTED", "value": True}, {"status": "SUPPORTED", "value": 2})["format_valid"])
        self.assertFalse(score_one({"error": "budget exhausted"}, {"status": "UNKNOWN", "value": None})["semantic_exact"])

    def test_gold_unknown_duplicate_and_nonfinite_json_rejected(self):
        for raw in (b'{"gold":1}', b'{"case":{},"case":{},"arm":"B","snapshot":null}',
                    b'{"case":NaN,"arm":"B","snapshot":null}'):
            with self.assertRaises(ValueError):
                request(raw)

    def test_irrelevant_snapshot_same_size_distinct_applicability(self):
        supplied = supplied_snapshot()
        irrelevant = irrelevant_snapshot(supplied)
        self.assertEqual(len(supplied._bytes), len(irrelevant._bytes))
        self.assertNotEqual(supplied.sha256, irrelevant.sha256)
        self.assertNotEqual(supplied.entries()[0]["applicability"], irrelevant.entries()[0]["applicability"])


if __name__ == "__main__":
    unittest.main()
