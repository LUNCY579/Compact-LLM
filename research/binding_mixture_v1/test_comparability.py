"""Model-free regression tests for decision, retry and cross-clone boundaries."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_mixture_v1 import report


class CriteriaTests(unittest.TestCase):
    def setUp(self):
        self.criteria = report.read(report.ROOT / "CRITERIA.json")
        self.candidate = {"dose": {"correct": 284}, "select": {"correct": 284},
                          "retention": {"policies": {"permissive": {"references_retained": 40, "panel_correct": 62}}},
                          "long_context": {"correct": 0, "literal_unknown": 0}}
        self.binding = {"endpoints": {"mixed": {"totals": {"false_unknown_on_positive": 10}, "categories": {
            cat: {"canonical_complete_exact": count} for cat, count in
            (("person_missing", 72), ("person_vacant", 80), ("person_answerable", 64), ("role_answerable", 72))}}},
            "world_contrasts_mixed_minus_control": {"canonical_complete_exact": {
                "person_missing": {"net": 7, "holm_adjusted_p": .03125},
                "person_vacant": {"net": 0, "holm_adjusted_p": 1.0}}}}
        self.acquisition = {"dose": {"net": -20}, "select": {"net": 0}}

    def verdict(self):
        return report.prediction_verdicts(self.candidate, self.binding, self.acquisition, self.criteria)

    def test_long_context_cannot_rescue_or_fail_primary(self):
        self.assertEqual(self.verdict()["primary_engineering_conjunction"], "PASS")
        self.candidate["dose"]["correct"] = 283
        self.candidate["long_context"]["correct"] = 256
        self.assertEqual(self.verdict()["primary_engineering_conjunction"], "FAILED_PREDICTION")

    def test_all_binding_bands_required(self):
        for category in report.bind_score.CATEGORIES:
            with self.subTest(category=category):
                changed = copy.deepcopy(self.binding)
                changed["endpoints"]["mixed"]["categories"][category]["canonical_complete_exact"] = 0
                value = report.prediction_verdicts(self.candidate, changed, self.acquisition, self.criteria)
                self.assertEqual(value["primary_engineering_conjunction"], "FAILED_PREDICTION")

    def test_absolute_bands_are_not_efficacy(self):
        self.binding["world_contrasts_mixed_minus_control"]["canonical_complete_exact"]["person_missing"]["holm_adjusted_p"] = .06
        self.assertEqual(self.verdict()["binding_efficacy"], "NOT_ESTABLISHED")
        self.assertEqual(self.verdict()["primary_engineering_conjunction"], "PASS")

    def test_significant_wrong_direction_is_not_gain(self):
        self.binding["world_contrasts_mixed_minus_control"]["canonical_complete_exact"]["person_missing"]["net"] = -7
        self.assertEqual(self.verdict()["binding_efficacy"], "NOT_ESTABLISHED")

    def test_positive_controls_and_false_unknown_gate_efficacy(self):
        self.binding["endpoints"]["mixed"]["totals"]["false_unknown_on_positive"] = 11
        self.assertEqual(self.verdict()["binding_efficacy"], "NOT_ESTABLISHED")

    def test_cost_allowance_does_not_establish_preservation(self):
        cost = self.verdict()["acquisition_cost_engineering"]
        self.assertEqual(cost["dose"]["status"], "PASS")
        self.assertIs(cost["dose"]["establishes_preservation"], False)
        self.acquisition["dose"]["net"] = -21
        self.assertEqual(self.verdict()["acquisition_cost_engineering"]["dose"]["status"], "FAILED_PREDICTION")

    def test_criteria_json_is_authoritative(self):
        self.criteria["primary_engineering"]["each_acquisition_at_least"] = 285
        self.assertEqual(self.verdict()["primary_engineering_conjunction"], "FAILED_PREDICTION")


class ComparabilityTests(unittest.TestCase):
    def fixture(self):
        result = {}
        for leg in report.prior.LEGS:
            raw = {"device": "cuda", "runtime": {}, "numerics": {}, "max_new_tokens": 128,
                   "transformers": "fixed", "max_context": 2048, "registered_bare_policy": "permissive",
                   "frozen_reference_sizes": {}, "probe_rows": 150}
            binding = {"views_sha256": "v", "implementation_digest": "i", "script_sha256": "s", "wrapper_sha256": "w",
                       "additional_input_sha256_before": {}, "training_snapshot_before": "own arm data"}
            result[leg] = (raw, binding, {"one"})
        return result

    def test_different_registered_training_inputs_are_permitted(self):
        candidate, control = self.fixture(), self.fixture()
        candidate["dose"][1]["training_snapshot_before"] = "new mixture"
        control["dose"][1]["training_snapshot_before"] = "frozen control"
        archive = type("Archive", (), {"normalize_additional_inputs": staticmethod(lambda v: v)})()
        report.check_standard_pair(candidate, control, archive)

    def test_changed_eval_views_still_block(self):
        candidate, control = self.fixture(), self.fixture()
        candidate["dose"][1]["views_sha256"] = "tampered"
        archive = type("Archive", (), {"normalize_additional_inputs": staticmethod(lambda v: v)})()
        with self.assertRaisesRegex(ValueError, "evaluation identity"):
            report.check_standard_pair(candidate, control, archive)

    def test_changed_generation_settings_block(self):
        candidate, control = self.fixture(), self.fixture()
        candidate["dose"][0]["max_new_tokens"] = 129
        archive = type("Archive", (), {"normalize_additional_inputs": staticmethod(lambda v: v)})()
        with self.assertRaisesRegex(ValueError, "execution setting"):
            report.check_standard_pair(candidate, control, archive)

    def test_two_local_retries_cannot_become_two_global_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "ATTEMPT_MAP.json").write_text(json.dumps({"schema": 1, "replacements": {
                "mixed-binding": "eval/mixed-binding-retry1", "mixed-dose": "eval/mixed-dose-retry1"}}))
            with self.assertRaisesRegex(ValueError, "ONE total"):
                report.attempt_map(path)

    def test_blocked_report_has_no_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            result = report.build_report(Path(temp))
            self.assertEqual(result["status"], "BLOCKED_ON_COMPARABILITY")
            self.assertNotIn("endpoints", result)
            self.assertNotIn("predictions", result)
            self.assertIn("BLOCKED", report.markdown(result))

    def test_holm_uses_only_the_two_registered_categories(self):
        rows = {"missing": {"exact_p": .03}, "vacant": {"exact_p": .04}}
        report.control_report.holm(rows)
        self.assertEqual(rows["missing"]["holm_adjusted_p"], .06)
        self.assertEqual(rows["vacant"]["holm_adjusted_p"], .06)

    def test_world_level_discordance_threshold(self):
        reference = {str(k): False for k in range(32)}
        candidate = {str(k): k < 7 for k in range(32)}
        contrast = report.paired.paired_boolean_summary(reference, candidate, relationship="cross_training_draw")
        self.assertEqual((contrast["n"], contrast["nd"], contrast["net"]), (32, 7, 7))
        self.assertEqual(contrast["exact_p"], .015625)
        self.assertEqual(contrast["minimum_significant_abs_net_at_observed_nd"], 7)

    def test_report_accepts_receipt_string_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text('{"status":"PASS"}', encoding="utf-8")
            self.assertEqual(report.read(str(path)), {"status": "PASS"})
            self.assertEqual(report.sha(str(path)), report.sha(path))

    def test_frozen_numeric_reference_exists_and_matches_baselines(self):
        plan = report.read(report.BASELINE / "BASELINE_PLAN.json")
        for filename, expected in plan["code_sha256"].items():
            self.assertEqual(report.sha(report.bc.baseline.HERE / filename), expected)
        for role in ("source", "lora"):
            generations = report.read(report.BASELINE / role / "GENERATIONS.json")
            self.assertEqual(report.bc.baseline.NUMERICS, generations["numerics"])
        self.assertNotIn("torch", sys.modules)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / "posttrain").mkdir()
        self.envelope = report.read(report.ROOT / "SESSION_ENVELOPE.json")
        for leg, (rows, maximum) in report.RESERVATIONS.items():
            key = "mixed-" + leg
            directory = self.root / "eval" / key
            directory.mkdir(parents=True)
            supervisor_path = directory / ("supervisor.json" if leg == "binding" else "worker/supervisor.json")
            supervisor_path.parent.mkdir(exist_ok=True)
            supervisor_path.write_text(json.dumps({"status": "EXITED", "returncode": 0, "wall_seconds": 10.0}))
            (self.root / "posttrain" / (key + ".json")).write_text(json.dumps({"leg": key, "out": str(directory),
                "returncode": 0, "reserved_rows": rows, "reserved_generation_tokens": rows * maximum, "worker_seconds_charged": 10.0}))

    def tearDown(self):
        self.temp.cleanup()

    def test_all_five_legs_reserve_exact_global_rows_and_tokens(self):
        value = report.probe_cost_accounting(self.root, {}, self.envelope)
        self.assertEqual((value["attempt_count"], value["probe_rows_reserved"], value["probe_generation_tokens_reserved"]), (5, 1558, 117264))

    def test_local_subcap_cannot_override_global_token_limit(self):
        self.envelope["probe_generation_tokens_reserved_all_attempts"] = 117263
        with self.assertRaisesRegex(ValueError, "global five-leg"):
            report.probe_cost_accounting(self.root, {}, self.envelope)

    def test_missing_observed_supervisor_costs_full_reservation(self):
        path = self.root / "eval/mixed-binding/supervisor.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "unobserved reservation"):
            report.probe_cost_accounting(self.root, {}, self.envelope)
        receipt_path = self.root / "posttrain/mixed-binding.json"
        receipt = report.read(receipt_path); receipt["worker_seconds_charged"] = 900.0
        receipt_path.write_text(json.dumps(receipt))
        self.assertEqual(report.probe_cost_accounting(self.root, {}, self.envelope)["probe_worker_seconds"], 940.0)

    def test_recovery_cannot_change_preserved_attempt_cost(self):
        original = report.read(self.root / "posttrain/mixed-binding.json")
        altered = {**original, "worker_seconds_charged": 1.0}
        (self.root / "posttrain/mixed-binding.json").write_text(json.dumps(altered))
        terminal_path = self.root / "posttrain/TERMINAL.json"
        terminal_path.write_text(json.dumps({"status": "FAILED_OR_BLOCKED", "attempts": [original]}))
        map_path = self.root / "ATTEMPT_MAP.json"
        replacements = {"mixed-binding": "eval/mixed-binding-retry1"}
        map_path.write_text(json.dumps({"schema": 1, "replacements": replacements}))
        (self.root / "posttrain/RECOVERY_COMPLETE.json").write_text(json.dumps({"status": "COMPLETE_FIVE_PROBES",
            "prior_terminal_sha256": report.sha(terminal_path), "attempt_map_sha256": report.sha(map_path)}))
        with self.assertRaisesRegex(ValueError, "original attempt receipt changed"):
            report.completion_receipt(self.root, replacements, self.envelope)


if __name__ == "__main__":
    unittest.main()
