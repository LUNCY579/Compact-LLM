import unittest
from experiments.rineya_centralized_reasoning_system_v1.atlas import REQUIRED_ROW_FIELDS, efficiency, validate_row, paired_parent_interval, skill_matrices, schedule


class AtlasTests(unittest.TestCase):
    def test_missing_bindings_nonfinite_metrics_cannot_be_native_evidence(self):
        row = {k: None for k in REQUIRED_ROW_FIELDS}
        row.update(evidence_class="NATIVE_CONFIRMATION", unique_parents=1, presentations=1,
                   processed_tokens=1, supervised_tokens=1, updates=1, failures=0,
                   acquisition={"estimate": float("nan"), "interval95": [0, 1], "scope": "f"})
        with self.assertRaises(ValueError):
            validate_row(row)
        row.update(evidence_class="UNMEASURED", acquisition=None)
        self.assertTrue(all(v is None for v in efficiency(None, row).values()))
        with self.assertRaises(ValueError):
            efficiency(.1, row)

    def test_clustered_variants_not_independent_samples(self):
        rows = [{"parent_id": "p1", "family": "f", "B": 0, "C": 1},
                {"parent_id": "p1", "family": "f", "B": 1, "C": 1},
                {"parent_id": "p2", "family": "f", "B": 1, "C": 0}]
        r = paired_parent_interval(rows, resamples=100)
        self.assertEqual(r["parent_count"], 2)
        self.assertEqual(r["delta"], -.25)
        self.assertEqual(r, paired_parent_interval(rows, resamples=100))

    def test_retention_zero_denominator_and_worst_skill(self):
        source = {"s1": {"a": True, "b": False}, "s2": {"c": False}}
        checkpoints = {"c1": {"s1": {"a": False, "b": True}, "s2": {"c": False}}}
        r = skill_matrices(checkpoints, source)
        self.assertEqual(r["forgetting"]["c1"]["s1"]["source_losses"], 1)
        self.assertIsNone(r["forgetting"]["c1"]["s2"]["source_correct_retention"])
        self.assertFalse(r["forgetting"]["c1"]["s1"]["noninferiority_established"])
        with self.assertRaises(ValueError):
            skill_matrices({"bad": {"s1": {"b": True}, "s2": {"c": False}}}, source)

    def test_scheduler_probability_cost_risk_and_bad_nan(self):
        candidates = [{"id": "risky", "gain": .9, "uncertainty": .1, "forgetting_risk": .9, "cost": 10},
                      {"id": "safer", "gain": .2, "uncertainty": .2, "forgetting_risk": .1, "cost": 1}]
        r = schedule(candidates, seed=5)
        self.assertAlmostEqual(sum(r["selection_probabilities"]), 1)
        self.assertTrue(all(p > 0 for p in r["selection_probabilities"]))
        candidates[0]["cost"] = float("nan")
        with self.assertRaises(ValueError):
            schedule(candidates, seed=5)


if __name__ == "__main__":
    unittest.main()
