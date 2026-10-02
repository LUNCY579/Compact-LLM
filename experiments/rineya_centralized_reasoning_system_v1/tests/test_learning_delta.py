"""Small independent CPU arithmetic/schema fixtures; no datasets or native runs."""
import copy
import hashlib
import json
import unittest

from experiments.rineya_centralized_reasoning_system_v1 import learning_delta as delta, atlas_v2


def endpoint(correct, *, count=4, checkpoint="a" * 64, name="semantic_accuracy"):
    membership = {f"c{i}": {"parent_id": f"p{i // 2}", "variant_id": f"v{i % 2}", "family": "fixture"} for i in range(count)}
    items = {f"c{i}": {"semantic_exact": i < correct, "strict_exact": i < correct,
                        "format_valid": True, "operational_error": False, "abstains": False} for i in range(count)}
    return {"schema_version": 1, "status": "AVAILABLE", "panel_id": "test-only-panel",
            "panel_sha256": "d" * 64, "scorer_sha256": "c" * 64, "scoring_version": "test-only-scorer-v2",
            "endpoint": name, "scope": "test-only-shared-panel", "checkpoint_sha256": checkpoint,
            "membership": membership, "items": items, "proportion": correct / count, "error": None}


def receipt(value):
    raw = json.dumps({"schema_version": 1, "kind": "CRS_LEARNING_ENDPOINT", "payload": value},
                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def derive(baseline, candidate, **kwargs):
    a, ah = receipt(baseline); b, bh = receipt(candidate)
    return delta.derive_delta(a, b, baseline_sha256=ah, candidate_sha256=bh, **kwargs)


def atlas_row(candidate_proportion=.5):
    """Invented metadata is never recorded as actual native evidence."""
    row = {key: None for key in atlas_v2.REQUIRED_ROW_FIELDS}
    row.update(schema_version=2, study="test-only", recipe="test-only", evidence_class="NATIVE_EXPLORATORY",
               family="fixture", composition_depth=1, train_draw_seed=1, order_seed=2, eval_family_seed=3,
               initialization_seed=4, unique_parents=2, presentations=2, processed_tokens=100, supervised_tokens=20,
               updates=1, worker_seconds=10, peak_ram_bytes=100, peak_vram_bytes=100, optimizer={"name": "test-only"},
               update_scope="test-only", code_commit="1" * 40, exposure="open-test-only", failures=0,
               stop_reason="test-only", checkpoint_sha256="b" * 64, source_checkpoint_sha256="a" * 64,
               data_sha256="d" * 64, scorer_sha256="c" * 64, qualification_sha256="e" * 64,
               measurement_receipt_sha256="f" * 64,
               acquisition={"estimate": candidate_proportion, "interval95": [0, 1], "scope": "test-only-shared-panel"})
    return row


def measurement(row, gain):
    raw = json.dumps({"schema_version": 2, "kind": "CRS_LA_MEASUREMENT", "payload": {
        "row": {k: v for k, v in row.items() if k != "measurement_receipt_sha256"}, "gain": gain}},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    row["measurement_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    return raw


class LearningDeltaTests(unittest.TestCase):
    def test_same_panel_gain_is_computed_from_bound_proportions(self):
        result = derive(endpoint(1), endpoint(2, checkpoint="b" * 64), declared_gain=.25)
        self.assertEqual(result["baseline"], .25)
        self.assertEqual(result["candidate"], .5)
        self.assertEqual(result["gain"], .25)
        self.assertEqual(result["paired_counts"]["repairs"], 1)
        self.assertFalse(result["native_execution_attested"])
        self.assertFalse(result["qualification_granted"])

    def test_impossible_point9_gain_for_point1_candidate_rejected(self):
        with self.assertRaisesRegex(ValueError, "declared gain"):
            derive(endpoint(0, count=10), endpoint(1, count=10, checkpoint="b" * 64), declared_gain=.9)
        impossible = endpoint(0, count=10); impossible["proportion"] = -.8
        with self.assertRaises(ValueError):
            derive(impossible, endpoint(1, count=10, checkpoint="b" * 64))

    def test_panel_scorer_version_endpoint_scope_mismatch_rejected(self):
        for key, value in (("panel_id", "other"), ("panel_sha256", "2" * 64),
                           ("scorer_sha256", "3" * 64), ("scoring_version", "other-version"),
                           ("endpoint", "strict_accuracy"), ("scope", "other-scope")):
            candidate = endpoint(2, checkpoint="b" * 64); candidate[key] = value
            with self.assertRaises(ValueError, msg=key):
                derive(endpoint(1), candidate)

    def test_parent_and_item_membership_cannot_change_before_difference(self):
        candidate = endpoint(2, checkpoint="b" * 64)
        candidate["membership"]["c0"]["parent_id"] = "different-parent"
        with self.assertRaises(ValueError):
            derive(endpoint(1), candidate)
        expected = endpoint(1)["membership"]
        expected = copy.deepcopy(expected); expected["c0"]["family"] = "different"
        with self.assertRaises(ValueError):
            derive(endpoint(1), endpoint(2, checkpoint="b" * 64), expected_membership=expected)
        missing = endpoint(2, checkpoint="b" * 64); del missing["items"]["c3"]
        with self.assertRaises(ValueError):
            derive(endpoint(1), missing)

    def test_duplicate_parent_variant_rejected(self):
        candidate = endpoint(2, checkpoint="b" * 64)
        candidate["membership"]["c1"]["variant_id"] = "v0"
        with self.assertRaises(ValueError):
            delta.validate_endpoint(candidate)

    def test_operational_item_cannot_be_success_or_abstention(self):
        for key in ("semantic_exact", "abstains"):
            candidate = endpoint(0, checkpoint="b" * 64)
            bad = {"semantic_exact": False, "strict_exact": False, "format_valid": False,
                   "operational_error": True, "abstains": False}
            bad[key] = True
            if key == "semantic_exact": bad["format_valid"] = True; bad["strict_exact"] = True
            candidate["items"]["c0"] = bad
            with self.assertRaises(ValueError):
                delta.validate_endpoint(candidate)
        zero = endpoint(0, checkpoint="b" * 64)
        zero["items"]["c0"] = {"semantic_exact": False, "strict_exact": False, "format_valid": False,
                                "operational_error": True, "abstains": False}
        result = derive(endpoint(1), zero)
        self.assertEqual(result["gain"], -.25)
        self.assertEqual(result["paired_counts"]["candidate_operational_errors"], 1)

    def test_failed_unavailable_or_null_endpoint_has_undefined_gain(self):
        for status in ("UNAVAILABLE", "OPERATIONAL_FAILURE"):
            candidate = endpoint(0, checkpoint="b" * 64)
            candidate.update(status=status, proportion=None, error="test-only unavailable", items={})
            self.assertIsNone(derive(endpoint(1), candidate)["gain"])
            with self.assertRaises(ValueError):
                derive(endpoint(1), candidate, declared_gain=0)
        result = delta.derive_delta(None, None, baseline_sha256=None, candidate_sha256=None)
        self.assertIsNone(result["gain"])
        self.assertFalse(result["same_panel_membership_verified"])

    def test_type_finite_range_and_scored_denominator(self):
        for value in (True, float("nan"), float("inf"), -1, 1.1):
            candidate = endpoint(2, checkpoint="b" * 64); candidate["proportion"] = value
            with self.assertRaises(ValueError):
                delta.validate_endpoint(candidate)
        candidate = endpoint(2, checkpoint="b" * 64); candidate["proportion"] = .9
        with self.assertRaisesRegex(ValueError, "denominator"):
            delta.validate_endpoint(candidate)

    def test_actual_receipt_bytes_required_and_duplicate_json_rejected(self):
        raw, actual_sha = receipt(endpoint(1))
        with self.assertRaises(ValueError):
            delta.read_endpoint(raw + b" ", actual_sha)
        with self.assertRaises(ValueError):
            delta.read_endpoint({"sha256": actual_sha, "truth": True}, actual_sha)
        duplicate = raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1)
        with self.assertRaises(ValueError):
            delta.read_endpoint(duplicate, hashlib.sha256(duplicate).hexdigest())

    def test_semantic_and_strict_endpoint_are_not_silently_interchanged(self):
        candidate = endpoint(2, checkpoint="b" * 64)
        candidate["items"]["c1"].update(format_valid=False, strict_exact=False)
        result = derive(endpoint(1), candidate)
        self.assertEqual(result["gain"], .25)
        strict_base = endpoint(1, name="strict_accuracy")
        strict_candidate = copy.deepcopy(candidate); strict_candidate.update(endpoint="strict_accuracy", proportion=.25)
        self.assertEqual(derive(strict_base, strict_candidate)["gain"], 0)

    def test_checkpoint_metric_scope_and_measurement_binding_before_efficiency(self):
        a, ah = receipt(endpoint(1)); b, bh = receipt(endpoint(2, checkpoint="b" * 64))
        row = atlas_row(); raw = measurement(row, .25)
        result = delta.efficiency_from_endpoints(row, a, b, baseline_sha256=ah, candidate_sha256=bh,
                    declared_gain=.25, expected_membership=endpoint(1)["membership"], measurement_receipt=raw)
        self.assertEqual(result["efficiency"]["ratios"]["updates"], .25)
        self.assertFalse(result["native_execution_attested"])
        for key, value in (("checkpoint_sha256", "4" * 64), ("source_checkpoint_sha256", "5" * 64),
                           ("scorer_sha256", "6" * 64)):
            bad = copy.deepcopy(row); bad[key] = value
            with self.assertRaises(ValueError):
                delta.efficiency_from_endpoints(bad, a, b, baseline_sha256=ah, candidate_sha256=bh, measurement_receipt=raw)
        bad = copy.deepcopy(row); bad["acquisition"]["estimate"] = .1
        with self.assertRaises(ValueError):
            delta.efficiency_from_endpoints(bad, a, b, baseline_sha256=ah, candidate_sha256=bh, measurement_receipt=raw)

    def test_unavailable_endpoint_returns_undefined_ratios_without_fake_native_gain(self):
        a, ah = receipt(endpoint(1)); failure = endpoint(0, checkpoint="b" * 64)
        failure.update(status="OPERATIONAL_FAILURE", proportion=None, error="test-only OOM", items={})
        b, bh = receipt(failure)
        result = delta.efficiency_from_endpoints(atlas_row(), a, b, baseline_sha256=ah, candidate_sha256=bh)
        self.assertIsNone(result["delta"]["gain"])
        self.assertTrue(all(v is None for v in result["efficiency"]["ratios"].values()))
        self.assertFalse(result["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
