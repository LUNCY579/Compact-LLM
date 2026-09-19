"""OBS-001 CPU-only adversarial fixtures. No pretrained, bank or historical receipt reads."""
from copy import deepcopy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "audit" / "observability.py"
SPEC = importlib.util.spec_from_file_location("obs001", MODULE_PATH)
obs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(obs)


def case(kind="remote_git", method="remote_git_inventory", predicate="absent", value=False):
    source = {"kind": kind, "locator": "fixture://explicit-source", "revision": "a" * 40}
    claim = {"id": "fixture", "source": source, "predicate": predicate, "subjects": ["target"],
             "measurement": {"scope": "exact_manifest_subjects"}}
    observation = {"source": deepcopy(source), "method": method, "state": "COMPLETE",
                   "basis": "DIRECT", "measurement": deepcopy(claim["measurement"]),
                   "method_sha256": "b" * 64,
                   "evidence_sha256": "c" * 64, "results": {"target": value},
                   "positive_control": {"source": deepcopy(source), "passed": True,
                                        "method_sha256": "b" * 64, "evidence_sha256": "d" * 64,
                                        "measurement_sha256": obs.digest(claim["measurement"])}}
    return claim, observation


def sweep():
    return {"applicable": True, "levels": ["early", "late"],
            "n_per_level": {"early": 2, "late": 2},
            "rules": [{"rule": "identity", "rate_early": 0.5, "rate_late": 0.5,
                       "swing": 0.0, "abs_swing": 0.0}]}


def bound_sweep(rows, pool):
    result = sweep()
    result["input_sha256"] = obs.digest({"rows": rows, "pool": pool, "split_field": "band"})
    return result


class ClaimAdmission(unittest.TestCase):
    def status(self, c, o):
        return obs.assess(c, o)["status"]

    def test_remote_vault_not_visible_to_local_find(self):
        c, o = case()
        o["source"]["kind"] = "local_fs"
        o["method"] = "local_inventory"
        self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_local_method_cannot_be_relabelled_remote(self):
        c, o = case()
        o["method"] = "local_inventory"
        self.assertEqual(obs.assess(c, o)["reason"], "METHOD_CANNOT_MEASURE_PREDICATE")

    def test_wrong_repository_locator_and_revision(self):
        for field, value in [("locator", "fixture://other"), ("revision", "e" * 40)]:
            with self.subTest(field=field):
                c, o = case(); o["source"][field] = value
                self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_same_source_different_measurement_is_unobservable(self):
        c, o = case()
        o["measurement"] = {"scope": "different_query_or_population"}
        self.assertEqual(obs.assess(c, o)["reason"], "MEASUREMENT_CONTRACT_MISMATCH")
        o.pop("measurement")
        self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_control_for_different_measurement_does_not_qualify(self):
        c, o = case()
        o["positive_control"]["measurement_sha256"] = "e" * 64
        self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_unbound_remote_is_unknown_not_absent(self):
        c, o = case(); c["source"]["revision"] = None
        self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_unavailable_error_inapplicable_and_unrun_not_negative(self):
        for state in ("UNAVAILABLE", "ERROR", "INAPPLICABLE", "NOT_RUN"):
            with self.subTest(state=state):
                c, o = case(); o["state"] = state
                self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_pagination_cannot_prove_absence(self):
        c, o = case(); o["state"] = "PARTIAL"
        self.assertEqual(self.status(c, o), "INCONCLUSIVE")

    def test_positive_witness_survives_partial_inventory(self):
        c, o = case(predicate="present", value=True); o["state"] = "PARTIAL"
        self.assertEqual(self.status(c, o), "SUPPORTED")

    def test_present_witness_refutes_absence_even_with_incomplete_listing(self):
        c, o = case(value=True); o["state"] = "PARTIAL"
        self.assertEqual(self.status(c, o), "CONTRADICTED")

    def test_complete_bound_absence_with_control(self):
        c, o = case()
        self.assertEqual(self.status(c, o), "SUPPORTED")

    def test_missing_failed_and_wrong_source_controls(self):
        for change in ("missing", "failed", "source", "method", "hash"):
            with self.subTest(change=change):
                c, o = case()
                if change == "missing": o.pop("positive_control")
                if change == "failed": o["positive_control"]["passed"] = False
                if change == "source": o["positive_control"]["source"]["revision"] = "e" * 40
                if change == "method": o["positive_control"]["method_sha256"] = "e" * 64
                if change == "hash": o["positive_control"]["evidence_sha256"] = None
                self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_reported_and_synthetic_receipts_do_not_verify(self):
        for basis in ("REPORTED", "SYNTHETIC"):
            c, o = case(); o["basis"] = basis
            self.assertEqual(self.status(c, o), "UNVERIFIED")

    def test_missing_evidence_or_method_digest(self):
        for field in ("evidence_sha256", "method_sha256"):
            c, o = case(); o.pop(field)
            self.assertEqual(self.status(c, o), "UNVERIFIED")

    def test_three_hashes_cannot_verify_twenty_two(self):
        c, o = case(method="restored_byte_comparison", predicate="byte_identical", value=True)
        c["subjects"] = [f"file-{i}" for i in range(22)]
        o["results"] = {f"file-{i}": True for i in range(3)}
        self.assertEqual(self.status(c, o), "INCONCLUSIVE")
        c["subjects"] = c["subjects"][:3]
        self.assertEqual(self.status(c, o), "SUPPORTED")

    def test_listing_is_not_byte_verification(self):
        c, o = case(predicate="byte_identical", value=True)
        self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_file_mismatch_is_not_tensor_or_behaviour_difference(self):
        c, o = case(method="restored_byte_comparison", predicate="byte_different", value=False)
        self.assertEqual(self.status(c, o), "SUPPORTED")
        for predicate in ("tensor_different", "behaviour_changed", "build_deterministic"):
            c["predicate"] = predicate
            self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_text_search_cannot_prove_implementation_absent(self):
        c, o = case(method="literal_search", predicate="no_literal_match")
        self.assertEqual(self.status(c, o), "SUPPORTED")
        c["predicate"] = "implementation_absent"
        self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_dataset_disjointness_is_not_training_execution_history(self):
        c, o = case(kind="dataset_pair", method="dataset_overlap", predicate="declared_datasets_disjoint")
        self.assertEqual(self.status(c, o), "SUPPORTED")
        c["predicate"] = "nobody_trained_on_selection"
        self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_zero_swing_and_nonsignificance_not_causal_exclusion(self):
        c, o = case(kind="rule_bank", method="qualified_rule_sweep", predicate="no_detected_rule_swing")
        self.assertEqual(self.status(c, o), "SUPPORTED")
        for predicate in ("confound_excluded", "no_effect", "equivalent", "binding_competence"):
            c["predicate"] = predicate
            self.assertEqual(self.status(c, o), "UNOBSERVABLE")

    def test_empty_duplicate_and_wildcard_scope(self):
        for subjects in ([], ["a", "a"], ["*"], [""]):
            c, o = case(); c["subjects"] = subjects
            self.assertEqual(self.status(c, o), "INVALID")

    def test_missing_subject_is_not_false(self):
        c, o = case(); o["results"] = {}
        self.assertEqual(self.status(c, o), "INCONCLUSIVE")

    def test_nonboolean_results_and_malformed_states(self):
        for value in (0, 1, "false", [], {}):
            c, o = case(); o["results"]["target"] = value
            self.assertEqual(self.status(c, o), "INVALID")
        c, o = case(); o["state"] = []
        self.assertEqual(self.status(c, o), "INVALID")

    def test_empty_and_duplicate_registry_fail(self):
        self.assertFalse(obs.review_registry({"schema_version": 1, "entries": []})["review_consistent"])
        c, o = case(); e = {"claim": c, "observation": o, "declared_status": "SUPPORTED"}
        self.assertFalse(obs.review_registry({"schema_version": 1, "entries": [e, e]})["review_consistent"])

    def test_false_publication_is_rejected(self):
        c, o = case(); o["state"] = "NOT_RUN"
        doc = {"schema_version": 1, "entries": [{"claim": c, "observation": o, "declared_status": "SUPPORTED"}]}
        self.assertFalse(obs.review_registry(doc)["review_consistent"])
        doc["entries"][0]["declared_status"] = "UNOBSERVABLE"
        reviewed = obs.review_registry(doc)
        self.assertTrue(reviewed["review_consistent"])
        self.assertEqual(reviewed["supported_claim_count"], 0)


class CoverageAndSafety(unittest.TestCase):
    def test_empty_missing_and_blank_rendered_text_rejected(self):
        for rows in ([], [{"id": "a"}], [{"id": "a", "context": "  "}], [{"id": "a", "context": None}]):
            self.assertEqual(obs.row_coverage(rows)["state"], "INAPPLICABLE")

    def test_valid_rows_and_duplicate_ids(self):
        rows = [{"id": "a", "context": "Actual text."}]
        self.assertEqual(obs.row_coverage(rows)["state"], "COMPLETE")
        self.assertEqual(obs.row_coverage(rows + rows)["state"], "INAPPLICABLE")

    def test_locator_missing_candidates_rejected(self):
        rows = [{"id": "a", "context": "Ann arrives. Work ends.", "reference": "Ann"}]
        self.assertEqual(obs.locator_coverage(rows, None)["state"], "INAPPLICABLE")

    def test_locator_common_population_and_skips(self):
        rows = [{"id": "a", "context": "Ann arrives. Bob leaves.", "reference": "Ann"},
                {"id": "b", "context": "Ann arrives. Ann leaves.", "reference": "Ann"}]
        d = obs.locator_coverage(rows, ["Ann", "Bob"])
        self.assertEqual(d["state"], "PARTIAL")
        self.assertEqual(d["scored_ids"], ["a"])
        self.assertEqual(d["skipped_ids"], ["b"])
        self.assertEqual(d["candidate_floor_by_scored_id"], {"a": 0.5})
        self.assertFalse(d["whole_bank_claim_admissible"])

    def test_locator_complete_and_no_scorable_rows(self):
        r = {"id": "a", "context": "Ann arrives. Bob leaves.", "reference": "Ann"}
        self.assertEqual(obs.locator_coverage([r], ["Ann", "Bob"])["state"], "COMPLETE")
        r["reference"] = "Missing"
        self.assertEqual(obs.locator_coverage([r], ["Ann", "Bob"])["state"], "INAPPLICABLE")

    def test_inapplicable_sweep_never_becomes_zero(self):
        d = obs.sweep_coverage({"applicable": False}, pool_observed=True, expected_rules=["identity"])
        self.assertEqual(d["state"], "INAPPLICABLE")
        self.assertIsNone(d["eligible_rules"])

    def test_zero_swing_is_observed_but_insensitive(self):
        d = obs.sweep_coverage(sweep(), pool_observed=True, expected_rules=["identity"])
        self.assertEqual(d["state"], "COMPLETE")
        self.assertFalse(d["can_test_cue_differential"])
        self.assertEqual(d["eligible_rules"], [])

    def test_swing_positive_control_detected(self):
        s = sweep(); s["rules"][0].update(rate_early=1.0, swing=0.5, abs_swing=0.5)
        d = obs.sweep_coverage(s, pool_observed=True, expected_rules=["identity"])
        self.assertTrue(d["can_test_cue_differential"])

    def test_unseen_pool_empty_levels_and_missing_rules(self):
        self.assertEqual(obs.sweep_coverage(sweep(), pool_observed=False, expected_rules=["identity"])["state"], "INAPPLICABLE")
        s = sweep(); s["n_per_level"]["early"] = 0
        self.assertEqual(obs.sweep_coverage(s, pool_observed=True, expected_rules=["identity"])["state"], "INAPPLICABLE")
        self.assertEqual(obs.sweep_coverage(sweep(), pool_observed=True, expected_rules=["identity", "sentence_0"])["state"], "INAPPLICABLE")

    def test_nonfinite_and_inconsistent_swing_refused(self):
        for value in (float("nan"), float("inf"), 0.7, "0"):
            s = sweep(); s["rules"][0]["abs_swing"] = value
            self.assertEqual(obs.sweep_coverage(s, pool_observed=True, expected_rules=["identity"])["state"], "INAPPLICABLE")
        for threshold in (0, -1, float("nan"), True):
            self.assertEqual(obs.sweep_coverage(sweep(), pool_observed=True, expected_rules=["identity"], min_swing=threshold)["state"], "INAPPLICABLE")

    def test_checked_sweep_requires_actual_nonempty_pool(self):
        rows = [{"id": str(i), "context": "Ann arrives. Bob leaves.",
                 "reference": "Ann", "band": "early" if i < 2 else "late"} for i in range(4)]
        self.assertEqual(obs.checked_rule_sweep(bound_sweep(rows, []), rows, [], "band", ["identity"])["state"], "INAPPLICABLE")
        self.assertEqual(obs.checked_rule_sweep(bound_sweep(rows, ["Ann", "Bob"]), rows, ["Ann", "Bob"], "band", ["identity"])["state"], "COMPLETE")

    def test_checked_sweep_refuses_unseen_source_text(self):
        rows = [{"id": str(i), "context": "Nobody is named.",
                 "reference": "Ann", "band": "early" if i < 2 else "late"} for i in range(4)]
        self.assertEqual(obs.checked_rule_sweep(bound_sweep(rows, ["Ann", "Bob"]), rows, ["Ann", "Bob"], "band", ["identity"])["state"], "INAPPLICABLE")

    def test_checked_sweep_population_mismatch(self):
        rows = [{"id": str(i), "context": "Ann arrives. Bob leaves.",
                 "reference": "Ann", "band": "early" if i < 2 else "late"} for i in range(3)]
        self.assertEqual(obs.checked_rule_sweep(bound_sweep(rows, ["Ann", "Bob"]), rows, ["Ann", "Bob"], "band", ["identity"])["state"], "INAPPLICABLE")

    def test_checked_sweep_inputs_are_not_mutated(self):
        rows = [{"id": str(i), "context": "Ann arrives. Bob leaves.",
                 "reference": "Ann", "band": "early" if i < 2 else "late"} for i in range(4)]
        before = deepcopy(rows)
        obs.checked_rule_sweep(bound_sweep(rows, ["Ann", "Bob"]), rows, ["Ann", "Bob"], "band", ["identity"])
        self.assertEqual(rows, before)

    def test_checked_sweep_requires_prior_measurement_binding(self):
        rows = [{"id": str(i), "context": "Ann arrives. Bob leaves.",
                 "reference": "Ann", "band": "early" if i < 2 else "late"} for i in range(4)]
        report = obs.checked_rule_sweep(sweep(), rows, ["Ann", "Bob"], "band", ["identity"])
        self.assertEqual(report["problems"], ["MEASUREMENT_INPUT_HASH_MISSING_OR_MISMATCHED"])

    def test_checked_sweep_rejects_stale_equal_sized_bank(self):
        rows = [{"id": str(i), "context": "Ann arrives. Bob leaves.",
                 "reference": "Ann", "band": "early" if i < 2 else "late"} for i in range(4)]
        result = bound_sweep(rows, ["Ann", "Bob"])
        rows[0]["context"] = "Bob arrives. Ann leaves."
        self.assertEqual(obs.checked_rule_sweep(result, rows, ["Ann", "Bob"], "band", ["identity"])["state"], "INAPPLICABLE")

    def test_missing_prediction_distinct_from_emitted_empty(self):
        self.assertEqual(obs.prediction_coverage(["a"], [{"id": "a"}])["state"], "INAPPLICABLE")
        self.assertEqual(obs.prediction_coverage(["a"], [{"id": "a", "prediction": ""}])["state"], "COMPLETE")

    def test_duplicate_missing_and_extra_predictions(self):
        r = {"id": "a", "prediction": "Ann"}
        for records in ([r, r], [], [r, {"id": "b", "prediction": "Bob"}]):
            self.assertEqual(obs.prediction_coverage(["a"], records)["state"], "INAPPLICABLE")
        self.assertEqual(obs.prediction_coverage(["a"], [r, {"id": "b", "prediction": "Bob"}], allow_extra=True)["state"], "COMPLETE")

    def test_pair_validation_stops_silent_overwrite(self):
        rows = [{"id": "e", "pair_id": "p", "cell": "k1d1", "band": "early"},
                {"id": "l", "pair_id": "p", "cell": "k1d1", "band": "late"}]
        self.assertEqual(obs.pair_coverage(rows, ["k1d1"])["state"], "COMPLETE")
        self.assertEqual(obs.pair_coverage(rows + [dict(rows[0], id="other")], ["k1d1"])["state"], "INAPPLICABLE")
        rows[1]["cell"] = "k3d1"
        self.assertEqual(obs.pair_coverage(rows, ["k1d1", "k3d1"])["state"], "INAPPLICABLE")

    def test_missing_cell_and_empty_pairs_refused(self):
        self.assertEqual(obs.pair_coverage([], ["k1d1"])["state"], "INAPPLICABLE")
        row = {"id": "e", "pair_id": "p", "cell": "k1d1", "band": "early"}
        self.assertEqual(obs.pair_coverage([row], ["k1d1", "k3d1"])["state"], "INAPPLICABLE")

    def test_strict_json_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}'):
                p.write_text(text)
                with self.assertRaises(ValueError): obs.load_json(p)

    def test_bound_source_bytes_and_missing_or_changed_source(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "code.py"; p.write_bytes(b"original\n")
            expected = hashlib.sha1(b"blob 9\0original\n").hexdigest()
            self.assertEqual(obs.source_bindings(Path(td), {"code.py": expected})["state"], "COMPLETE")
            p.write_bytes(b"changed\n")
            self.assertEqual(obs.source_bindings(Path(td), {"code.py": expected})["state"], "INAPPLICABLE")
            self.assertEqual(obs.source_bindings(Path(td), {"missing.py": expected})["state"], "INAPPLICABLE")
            self.assertEqual(obs.source_bindings(Path(td), {})["state"], "INAPPLICABLE")
            self.assertEqual(obs.source_bindings(Path(td), {"../escape": expected})["state"], "INAPPLICABLE")

    def test_create_only_receipt_and_inputs_unchanged(self):
        c, o = case()
        doc = {"schema_version": 1, "entries": [{"claim": c, "observation": o, "declared_status": "SUPPORTED"}]}
        with tempfile.TemporaryDirectory() as td:
            registry, receipt = Path(td) / "registry.json", Path(td) / "receipt.json"
            registry.write_text(json.dumps(doc)); before = registry.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(obs.main(["--registry", str(registry), "--out", str(receipt)]), 0)
                receipt_before = receipt.read_bytes()
                self.assertEqual(obs.main(["--registry", str(registry), "--out", str(receipt)]), 2)
            self.assertEqual(registry.read_bytes(), before)
            self.assertEqual(receipt.read_bytes(), receipt_before)

    def test_cli_missing_source_read_does_not_certify(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(obs.main(["--registry", str(Path(td) / "missing")]), 2)


if __name__ == "__main__":
    unittest.main()
