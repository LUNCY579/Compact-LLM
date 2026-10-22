"""Synthetic paired-record controls, never Rineya capability evidence."""
import copy
import unittest

from experiments.rineya_quality_queue_v1.attenuation_audit import compare_records, lost_source_ids


def retention(lost):
    return {"probe_sha256": "same-panel", "reference_digest": "same-reference", "reference_receipt": "same-source-reference",
            "by_bare_policy": {"permissive": {"source_correct_total": 5, "lost_total": len(lost),
                "arm_correct_total": 5-len(lost), "arm_correct_outside_source": 0,
                "per_class": {"source": {"lost_ids": lost}}}}}


def process(counts):
    rows = [{"id": family+str(i), "family": family, "canonical": i < count, "reference": "same-reference"}
            for family, count in zip(("a", "b", "c"), counts) for i in range(10)]
    return {"views_sha256": "same-bank", "implementation_digest": "same-scorer",
            "max_new_tokens": 96, "runtime": {"fixture": True}, "numerics": {"fixture": True},
            "predictions": rows, "per_family": {family: {} for family in ("a", "b", "c")}}


class AttenuationAuditTests(unittest.TestCase):
    def test_equal_aggregate_cannot_hide_family_acquisition_loss(self):
        result = compare_records(process((8,5,5)), process((6,7,5)), retention(["x"]), retention(["x"]))
        self.assertEqual(result["acquisition"]["net"], 0)
        self.assertTrue(result["point_overall_gap_within_03"])
        self.assertFalse(result["acquisition_by_family"]["a"]["point_gap_within_05"])
        self.assertEqual(result["acquisition_by_family"]["a"]["difference"], -2)
        self.assertFalse(result["confirmation_or_magnitude_specific_superiority_established"])

    def test_source_loss_sets_preserve_paired_repairs_and_harms(self):
        result = compare_records(process((8,5,5)), process((8,5,5)), retention(["x", "y"]), retention(["y", "z"]))
        retained = result["source_retention"]
        self.assertEqual((retained["repairs"], retained["damages"], retained["difference"]), (1,1,0))
        self.assertEqual(retained["exact_mcnemar_two_sided"], 1)
        self.assertFalse(retained["unseen_case_independence_or_no_future_harm_established"])

    def test_incomplete_or_duplicate_source_loss_ids_are_rejected(self):
        data = retention(["x"])
        data["by_bare_policy"]["permissive"]["lost_total"] = 2
        with self.assertRaises(ValueError):
            lost_source_ids(data)
        with self.assertRaises(ValueError):
            lost_source_ids(retention(["x", "x"]))

    def test_runtime_budget_or_source_cohort_mismatch_blocks_pairing(self):
        left = process((8,5,5))
        for field, value in (("max_new_tokens", 128), ("runtime", {"different": True}), ("numerics", {"different": True})):
            with self.assertRaises(ValueError):
                compare_records(left, dict(left, **{field: value}), retention([]), retention([]))
        right_ret = retention([])
        right_ret["probe_sha256"] = "different-bank"
        with self.assertRaises(ValueError):
            compare_records(left, left, retention([]), right_ret)

    def test_paired_family_identity_cannot_change(self):
        left = process((8,5,5))
        right = copy.deepcopy(left)
        right["predictions"][0]["family"] = "b"
        with self.assertRaises(ValueError):
            compare_records(left, right, retention([]), retention([]))

    def test_same_ids_do_not_allow_different_reference_targets(self):
        left = process((8,5,5))
        right = copy.deepcopy(left)
        right["predictions"][0]["reference"] = "changed-target"
        with self.assertRaisesRegex(ValueError, "reference targets"):
            compare_records(left, right, retention([]), retention([]))
        right_ret = retention([])
        right_ret["reference_receipt"] = "changed-source-reference"
        with self.assertRaisesRegex(ValueError, "receipt identity"):
            compare_records(left, left, retention([]), right_ret)
