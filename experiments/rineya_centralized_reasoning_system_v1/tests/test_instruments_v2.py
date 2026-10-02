"""Independent counterexamples for v2 CPU instruments; no native result creation."""
import copy
import hashlib
import json
import unittest

from experiments.rineya_centralized_reasoning_system_v1 import scoring_v2 as scorer, atlas_v2 as atlas


def native_shaped_fixture():
    """Invented metadata for rejection/schema tests, never saved as model evidence."""
    row = {k: None for k in atlas.REQUIRED_ROW_FIELDS}
    row.update(schema_version=2, study="synthetic-test-only", recipe="ordinary-fixture", family="assignment",
               evidence_class="NATIVE_EXPLORATORY", composition_depth=2, train_draw_seed=101,
               order_seed=211, eval_family_seed=307, initialization_seed=17, unique_parents=32,
               presentations=32, processed_tokens=512, supervised_tokens=128, updates=8,
               worker_seconds=1, peak_ram_bytes=100, peak_vram_bytes=100, optimizer={"name": "fixture"},
               update_scope="qv", code_commit="b" * 40, exposure="open-fixture", failures=0,
               stop_reason="fixture", acquisition={"estimate": .5, "interval95": [.2, .8], "scope": "fixture"})
    for k in atlas.HASHES:
        row[k] = "a" * 64
    return row


def bound_fixture(row, gain):
    payload = {"schema_version": 2, "kind": "CRS_LA_MEASUREMENT",
               "payload": {"row": {k: v for k, v in row.items() if k != "measurement_receipt_sha256"}, "gain": gain}}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    row["measurement_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    return raw


def pair(cid, variant, parent="p", family="f", before=0, after=1):
    return {"case_id": cid, "variant_id": variant, "parent_id": parent, "family": family, "B": before, "C": after}


class SemanticFormatControls(unittest.TestCase):
    def test_correct_semantic_fields_with_extra_are_not_strict(self):
        gold = {"status": "SUPPORTED", "value": "Alice"}
        row = scorer.score_one(dict(gold, explanation="ignored extra field"), gold)
        self.assertTrue(row["semantic_exact"])
        self.assertTrue(row["status_correct"])
        self.assertTrue(row["value_correct"])
        self.assertFalse(row["strict_exact"])
        self.assertFalse(row["format_valid"])
        report = scorer.score({"c": dict(gold, extra=3)}, {"c": gold})
        self.assertEqual(report["semantic_accuracy"], 1)
        self.assertEqual(report["strict_accuracy"], 0)

    def test_operation_failure_never_becomes_abstention_or_success(self):
        gold = {"status": "UNKNOWN", "value": None}
        row = scorer.score_one({"error": "OOM", **gold}, gold)
        self.assertTrue(row["operational_error"])
        self.assertFalse(row["semantic_exact"])
        self.assertFalse(row["abstains"])
        self.assertFalse(row["format_valid"])

    def test_missing_failures_extra_ids_keep_scheduled_denominator(self):
        gold = {"a": {"status": "SUPPORTED", "value": "A"}, "b": {"status": "UNKNOWN", "value": None}}
        report = scorer.score({"a": {"error": "unavailable"}, "extra": gold["b"]}, gold)
        self.assertEqual(report["scheduled_cases"], 2)
        self.assertEqual(report["counts"]["missing"], 1)
        self.assertEqual(report["counts"]["operational_error"], 1)
        self.assertEqual(report["semantic_accuracy"], 0)
        self.assertFalse(report["membership_valid"])

    def test_integer_boolean_float_and_nonfinite_do_not_conflate(self):
        gold = {"status": "SUPPORTED", "value": 1}
        self.assertTrue(scorer.score_one(gold, gold)["semantic_exact"])
        for value in (True, 1.0, float("nan"), float("inf")):
            self.assertFalse(scorer.score_one({"status": "SUPPORTED", "value": value}, gold)["semantic_exact"])
        with self.assertRaises(ValueError):
            scorer.score_one(gold, {"status": "SUPPORTED", "value": True})

    def test_unique_json_no_prose_gold_repair(self):
        gold = {"status": "SUPPORTED", "value": 1}
        duplicate = '{"status":"SUPPORTED","value":1,"value":1}'
        self.assertFalse(scorer.score_one(duplicate, gold)["semantic_exact"])
        self.assertFalse(scorer.score_one("The answer is 1.", gold)["semantic_exact"])
        self.assertTrue(scorer.score_one(json.dumps(gold), gold)["strict_exact"])

    def test_refuted_missing_and_contradictory_statuses_are_distinct(self):
        gold = {s: {"status": s, "value": None} for s in ("REFUTED", "UNKNOWN", "CONTRADICTORY")}
        self.assertEqual(scorer.score(gold, gold)["semantic_accuracy"], 1)
        refuted = scorer.score({"r": gold["REFUTED"]}, {"r": gold["REFUTED"]})
        self.assertEqual(refuted["determinate_accuracy"], 1)
        self.assertIsNone(refuted["answerable_accuracy"])
        self.assertFalse(scorer.score_one(gold["UNKNOWN"], gold["REFUTED"])["semantic_exact"])

    def test_optional_provenance_format_never_attests_evidence_truth(self):
        gold = {"status": "SUPPORTED", "value": "A"}
        duplicate_refs = {**gold, "evidence_ids": ["e0", "e0"]}
        self.assertTrue(scorer.score_one(duplicate_refs, gold)["semantic_exact"])
        self.assertFalse(scorer.score_one(duplicate_refs, gold)["format_valid"])
        result = scorer.score({"c": {**gold, "evidence_ids": ["unverified"]}}, {"c": gold})
        self.assertFalse(result["provenance_truth_graded"])
        self.assertIsNone(result["calibration"])

    def test_bounds_and_nonfinite_extras_fail_even_projection(self):
        gold = {"status": "SUPPORTED", "value": "A"}
        self.assertFalse(scorer.score_one({**gold, "extra": float("nan")}, gold)["semantic_exact"])
        self.assertFalse(scorer.score_one({**gold, "extra": "x" * 5000}, gold)["semantic_exact"])


class AtlasSchemaBindingControls(unittest.TestCase):
    def test_32_unique_parents_eight_presentations_rejected(self):
        row = native_shaped_fixture(); row["presentations"] = 8
        with self.assertRaisesRegex(ValueError, "unique parents"):
            atlas.validate_row(row)

    def test_explicit_seeds_depth_and_hashes_required(self):
        for key, bad in (("train_draw_seed", None), ("order_seed", True), ("eval_family_seed", -1),
                         ("initialization_seed", None), ("composition_depth", True), ("family", None),
                         ("scorer_sha256", "missing"), ("code_commit", "stale")):
            row = native_shaped_fixture(); row[key] = bad
            with self.assertRaises(ValueError, msg=key):
                atlas.validate_row(row)

    def test_supervised_processed_updates_and_nonfinite_recipe(self):
        for updates in ({"supervised_tokens": 513}, {"updates": 33},
                        {"processed_tokens": 0, "supervised_tokens": 0}, {"optimizer": {"lr": float("nan")}}):
            row = native_shaped_fixture(); row.update(updates)
            with self.assertRaises(ValueError):
                atlas.validate_row(row)

    def test_caller_native_fields_cannot_compute_unbound_efficiency(self):
        row = native_shaped_fixture()
        self.assertEqual(atlas.validate_row(row), row)  # Schema validity, not truth.
        with self.assertRaises(ValueError):
            atlas.efficiency(.1, row)

    def test_measurement_bytes_bind_dose_and_gain_without_qualifying_native_truth(self):
        row = native_shaped_fixture(); raw = bound_fixture(row, .1)
        result = atlas.efficiency(.1, row, measurement_receipt=raw)
        self.assertAlmostEqual(result["ratios"]["unique_parents"], .1 / 32)
        self.assertFalse(result["native_truth_attested"])
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["measurement_binding"]["qualification_granted"])
        changed = copy.deepcopy(row); changed["worker_seconds"] = 2
        with self.assertRaises(ValueError):
            atlas.efficiency(.1, changed, measurement_receipt=raw)
        with self.assertRaises(ValueError):
            atlas.efficiency(.2, row, measurement_receipt=raw)
        with self.assertRaises(ValueError):
            atlas.efficiency(.1, row, measurement_receipt=raw + b" ")

    def test_receipt_numeric_type_change_does_not_match_integer_dose(self):
        row = native_shaped_fixture(); raw = bound_fixture(row, .1)
        altered = json.loads(raw); altered["payload"]["row"]["presentations"] = 32.0
        changed = json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()
        row["measurement_receipt_sha256"] = hashlib.sha256(changed).hexdigest()
        with self.assertRaises(ValueError):
            atlas.efficiency(.1, row, measurement_receipt=changed)

    def test_unmeasured_is_null_and_zero_dose_ratios_undefined(self):
        row = native_shaped_fixture(); row.update(evidence_class="UNMEASURED", acquisition=None)
        self.assertTrue(all(v is None for v in atlas.efficiency(None, row)["ratios"].values()))
        with self.assertRaises(ValueError):
            atlas.efficiency(.1, row)
        native_row = native_shaped_fixture()
        for k in ("unique_parents", "presentations", "processed_tokens", "supervised_tokens", "updates", "worker_seconds"):
            native_row[k] = 0
        raw = bound_fixture(native_row, 0)
        self.assertTrue(all(v is None for v in atlas.efficiency(0, native_row, measurement_receipt=raw)["ratios"].values()))


class PairAndRetentionControls(unittest.TestCase):
    def test_exact_duplicate_that_previously_biased_mean_rejected(self):
        records = [pair("one", "v0"), pair("two", "v1", before=1, after=0)]
        self.assertEqual(atlas.paired_parent_interval(records, resamples=100)["delta"], 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            atlas.paired_parent_interval(records + [records[0]], resamples=100)

    def test_relabeling_variant_or_case_cannot_evade_duplicate_detection(self):
        for second in (pair("one", "different"), pair("different", "v0")):
            with self.assertRaises(ValueError):
                atlas.paired_parent_interval([pair("one", "v0"), second], resamples=100)

    def test_bank_binding_rejects_missing_case_and_family_drift(self):
        records = [pair("one", "v0"), pair("two", "v1", before=1, after=0)]
        expected = {r["case_id"]: {k: r[k] for k in ("parent_id", "variant_id", "family")} for r in records}
        self.assertTrue(atlas.paired_parent_interval(records, expected_cases=expected, resamples=100)["membership_bound"])
        with self.assertRaises(ValueError):
            atlas.paired_parent_interval(records[:1], expected_cases=expected, resamples=100)
        changed = copy.deepcopy(records); changed[1]["family"] = "changed"
        with self.assertRaises(ValueError):
            atlas.paired_parent_interval(changed, resamples=100)

    def test_missing_incumbent_unavailable_and_source_losses_preserved(self):
        source = {"skill": {"a": True, "b": False}}
        candidate = {"skill": {"a": False, "b": True}}
        result = atlas.skill_matrices({"c": candidate}, source)
        cell = result["forgetting"]["c"]["skill"]
        self.assertEqual(cell["source_losses"], 1)
        self.assertEqual(cell["source_repairs"], 1)
        self.assertEqual(cell["incumbent_status"], "UNAVAILABLE")
        self.assertIsNone(cell["incumbent_losses"])
        self.assertIsNone(cell["incumbent_correct_denominator"])
        self.assertFalse(result["incumbent_available"])
        self.assertFalse(cell["noninferiority_established"])

    def test_explicit_incumbent_and_zero_correct_denominators(self):
        source = {"skill": {"a": False, "b": False}}
        incumbent = {"skill": {"a": True, "b": False}}
        candidate = {"skill": {"a": False, "b": True}}
        result = atlas.skill_matrices({"c": candidate}, source, incumbent)
        cell = result["forgetting"]["c"]["skill"]
        self.assertIsNone(cell["source_correct_retention"])
        self.assertEqual(cell["incumbent_losses"], 1)
        self.assertEqual(cell["incumbent_correct_denominator"], 1)
        self.assertEqual(cell["incumbent_correct_retention"], 0)
        with self.assertRaises(ValueError):
            atlas.skill_matrices({"c": candidate}, source, {"skill": {"a": True}})


if __name__ == "__main__":
    unittest.main()
