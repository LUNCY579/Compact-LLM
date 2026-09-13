"""Small-artifact validation tests: no model, tokenizer, GPU, or torch."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.rab001_exploration import frontier_report as report


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel = [json.loads(line) for line in (report.BASE / "data/monitor_probe.jsonl").read_text().splitlines()]
        cls.context = report.make_context(
            panel,
            report.read_json(report.BASE / f"outputs/retention-{report.DIGEST[:8]}/B28/retention.json"),
            report.read_json(report.EVIDENCE / "HEADROOM_AUDIT.json"),
            report.read_json(report.EVIDENCE / "PROBE_LABEL_AUDIT.json"),
            report.read_json(report.BASE / "protocol.json")["G1_RETENTION"],
        )
        cls.original = report.read_json(report.EVIDENCE / "ARM_DOSE_B28last8_lr1e-05_b5200_seed17.json")

    def validate(self, body=None):
        return report.validate_arm(body or self.original, "B28last8", self.context)

    def test_saved_control_and_endpoint_recompute_without_torch(self):
        result = self.validate()
        self.assertEqual((result["endpoint"]["correct_total"], result["endpoint"]["retained_total"]), (47, 46))
        self.assertEqual(result["endpoint"]["gained_by_stratum"], {"wrong_answer": 1, "format_only": 0, "runaway": 0})
        self.assertEqual(result["endpoint"]["loss_strata_status"], "COMPLETE")
        self.assertEqual(sum(result["endpoint"]["lost_by_stratum"].values()), result["endpoint"]["lost_total"])
        self.assertNotIn("torch", sys.modules)

    def test_mismatched_identity_is_rejected_instead_of_intersected(self):
        body = copy.deepcopy(self.original)
        rowmap = body["checkpoints"][-1]["retention"]["per_row"]
        rowmap["unexpected"] = rowmap.pop(next(iter(rowmap)))
        with self.assertRaisesRegex(report.InvalidArtifact, "150 boolean"):
            self.validate(body)

    def test_corrupted_output_is_not_trusted_because_aggregate_matches(self):
        body = copy.deepcopy(self.original)
        point = body["checkpoints"][-1]
        rid = next(rid for rid, ok in point["retention"]["per_row"].items() if ok)
        point["outputs"][rid]["terminated"] = False
        with self.assertRaisesRegex(report.InvalidArtifact, "re-score"):
            self.validate(body)

    def test_termination_integrity_change_rejected_with_identical_text(self):
        body = copy.deepcopy(self.original)
        initial = body["checkpoints"][0]["outputs"]
        rid = next(iter(initial))
        initial[rid]["terminated"] = not initial[rid]["terminated"]
        with self.assertRaisesRegex(report.InvalidArtifact, "integrity control"):
            self.validate(body)

    def test_metadata_controls_and_dose_rejected(self):
        for key, bad in (("completed", False), ("reference_digest", "wrong"), ("seed", 23),
                         ("probe_device", "cuda"), ("beta2", .95), ("supervised_target_tokens", 5234)):
            with self.subTest(key=key):
                body = copy.deepcopy(self.original)
                body[key] = bad
                with self.assertRaises(report.InvalidArtifact):
                    self.validate(body)
        body = copy.deepcopy(self.original)
        body["numerics"]["cudnn_allow_tf32"] = True
        with self.assertRaisesRegex(report.InvalidArtifact, "TF32"):
            self.validate(body)

    def test_headline_mismatch_is_rejected(self):
        body = copy.deepcopy(self.original)
        body["headline"]["correct_end"] += 1
        with self.assertRaisesRegex(report.InvalidArtifact, "headline correct_end"):
            self.validate(body)

    def test_consumed_id_hash_is_verified(self):
        body = copy.deepcopy(self.original)
        ids = [f"row-{i}" for i in range(body["optimizer_steps"])]
        body["train_row_ids_consumed"] = ids
        body["train_row_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        self.assertIn("train_row_ids_sha256", self.validate(body)["recorded_training_bindings"])
        body["train_row_ids_consumed"][0] = "different"
        with self.assertRaisesRegex(report.InvalidArtifact, "row-ID hash"):
            self.validate(body)

    def test_pair_rejects_order_or_binding_disagreement(self):
        left = self.validate()
        right = copy.deepcopy(left)
        right["token_sequence"][:2] = reversed(right["token_sequence"][:2])
        with self.assertRaisesRegex(report.InvalidArtifact, "sequence"):
            report.compare_pair(left, right)
        right = copy.deepcopy(left)
        left["recorded_training_bindings"] = {"data_sha256": {"train": "A"}}
        right["recorded_training_bindings"] = {"data_sha256": {"train": "B"}}
        with self.assertRaisesRegex(report.InvalidArtifact, "binding"):
            report.compare_pair(left, right)

    def test_recorded_data_hash_must_match_current_frozen_input(self):
        body = copy.deepcopy(self.original)
        body["data_sha256"] = {"train": "claimed"}
        context = copy.deepcopy(self.context)
        context["current_data_sha256"] = {"train": "actual"}
        with self.assertRaisesRegex(report.InvalidArtifact, "train hash"):
            report.validate_arm(body, "B28last8", context)

    def test_output_path_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("preserved")
            with self.assertRaisesRegex(report.InvalidArtifact, "already exists"):
                report.main(["--out", str(path)])
            self.assertEqual(path.read_text(), "preserved")

    def test_historical_complete_model_flag_is_annotated_not_rewritten(self):
        body = copy.deepcopy(self.original)
        body["checkpoint"]["complete_model"] = True
        result = self.validate(body)
        self.assertIn("partial", result["checkpoint_metadata"]["warning"])
        self.assertTrue(body["checkpoint"]["complete_model"])

    def test_exact_p_and_holm_known_values(self):
        self.assertEqual(report.mcnemar_exact(0, 0), 1)
        self.assertEqual(report.mcnemar_exact(6, 0), Fraction(1, 32))
        self.assertEqual(report.mcnemar_exact(3, 1), Fraction(5, 8))
        corrected = report.holm({"a": Fraction(1, 100), "b": Fraction(4, 100), "c": Fraction(6, 100)})
        self.assertEqual(corrected, {"a": Fraction(3, 100), "b": Fraction(8, 100), "c": Fraction(8, 100)})

    def test_missing_pairs_keep_ten_test_correction(self):
        left = self.validate()
        right = copy.deepcopy(left)
        for rid in list(right["_per_row"])[:6]:
            left["_per_row"][rid], right["_per_row"][rid] = True, False
        arms = {"B28last8": left, "B28last12": right}
        pairs = report.fixed_pair_report(arms)
        self.assertEqual(len(pairs), 10)
        observed = pairs["B28last8~B28last12"]
        self.assertEqual(observed["p_raw"], 1 / 32)
        self.assertEqual(observed["p_holm"], 10 / 32)
        self.assertFalse(observed["significant"])
        self.assertEqual(observed["significance_status"], "WITHHELD_INCOMPLETE_FAMILY")
        self.assertEqual(sum(p["p_is_missing_placeholder"] for p in pairs.values()), 9)

    def test_no_results_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            result = report.build_report(Path(directory), self.context)
        self.assertEqual(result["family_status"], "INCOMPLETE")
        self.assertEqual(len(result["pairs"]), 10)
        self.assertTrue(all(pair["status"] == "UNAVAILABLE" for pair in result["pairs"].values()))

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"row": true, "row": false}')
            with self.assertRaisesRegex(report.InvalidArtifact, "duplicate JSON"):
                report.read_json(path)


if __name__ == "__main__":
    unittest.main()
