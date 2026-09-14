"""Explicit retry aliases must preserve, rather than conceal, the failed canonical run."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments.rab001_exploration import frontier_report as report
from experiments.rab001_exploration import alpha_report


class RunRegistryTests(unittest.TestCase):
    canonical = "B28_lr1e-05_b5200_seed17"
    retry = canonical + "_retry2"

    def registry(self):
        return report.read_json(report.RUN_REGISTRY)

    def test_active_retry_has_hash_bound_original_failure(self):
        registry = report.load_run_registry()
        selection = report.resolve_run(self.canonical, registry=registry)
        self.assertEqual(selection["active_tag"], self.retry)
        first, second = selection["preserved_failures"]
        self.assertTrue(first["no_endpoint_observed"])
        self.assertEqual(first["observed_supervised_target_tokens"], 186)
        self.assertEqual(first["observed_optimizer_steps"], 1)
        self.assertFalse(second["no_endpoint_observed"])
        self.assertTrue(second["completed_training"])
        self.assertEqual(second["observed_supervised_target_tokens"], 5233)
        self.assertTrue(second["known_score_is_not_retry_criterion"])
        self.assertEqual(registry["registry_sha256"], report.sha256(report.RUN_REGISTRY))

    def test_absent_alias_does_not_select_an_existing_retry_filename(self):
        empty = {"schema": 1, "active_tags": {}, "retries": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"ARM_DOSE_{self.retry}.json").write_text("{}")
            selected = report.resolve_run(self.canonical, root, empty)
            self.assertEqual(Path(selected["artifact_path"]).name, f"ARM_DOSE_{self.canonical}.json")
            self.assertFalse(Path(selected["artifact_path"]).exists())
        with self.assertRaisesRegex(report.InvalidArtifact, "unregistered"):
            report.validate_run_identity({"canonical_tag": self.canonical, "run_label": "retry2"}, selected)

    def test_unsupported_retry_alias_is_rejected(self):
        registry = self.registry()
        second = self.canonical + "_retry3"
        registry["active_tags"][self.canonical] = second
        registry["retries"][second] = registry["retries"].pop(self.retry)
        with self.assertRaisesRegex(report.InvalidArtifact, "unsupported registry alias"):
            report.validate_run_registry(registry)

    def test_missing_or_mismatched_metadata_is_rejected(self):
        registry = self.registry()
        registry["retries"] = {}
        with self.assertRaisesRegex(report.InvalidArtifact, "disagree"):
            report.validate_run_registry(registry)
        registry = self.registry()
        registry["retries"][self.retry]["canonical_tag"] = "different"
        with self.assertRaisesRegex(report.InvalidArtifact, "canonical_tag"):
            report.validate_run_registry(registry)

    def test_failure_hash_or_no_endpoint_claim_cannot_be_changed(self):
        for key, value in (("failure_artifact_sha256", "0" * 64), ("no_endpoint_observed", True),
                           ("observed_supervised_target_tokens", 5200)):
            with self.subTest(key=key):
                registry = self.registry()
                registry["retries"][self.retry][key] = value
                with self.assertRaises(report.InvalidArtifact):
                    report.validate_run_registry(registry)

    def test_retry_body_must_bind_both_canonical_tag_and_label(self):
        selection = report.resolve_run(self.canonical, registry=report.load_run_registry())
        report.validate_run_identity({"canonical_tag": self.canonical, "run_label": "retry2"}, selection)
        for body in ({"run_label": "retry2"}, {"canonical_tag": self.canonical, "run_label": "retry1"},
                     {"canonical_tag": "different", "run_label": "retry2"}):
            with self.subTest(body=body), self.assertRaisesRegex(report.InvalidArtifact, "canonical_tag/run_label"):
                report.validate_run_identity(body, selection)

    def test_alpha_rejects_alias_without_registry_selection(self):
        # Tag checks happen before any outputs are read; no fake model result is needed.
        body = {"completed": True, "tag": self.retry}
        with self.assertRaisesRegex(report.InvalidArtifact, "unregistered"):
            alpha_report.validate_sweep(body, "B28", {}, {"canonical_tag": self.canonical, "run_label": "retry2"})

    def test_missing_retry_report_keeps_original_failure_visible(self):
        registry = report.load_run_registry()
        with tempfile.TemporaryDirectory() as directory:
            output = report.build_report(Path(directory), {}, registry)
        result = output["arms"]["B28"]
        self.assertEqual(result["status"], "MISSING")
        self.assertEqual(result["run_selection"]["active_tag"], self.retry)
        self.assertEqual(len(result["run_selection"]["preserved_failures"]), 2)
        self.assertIn("free physical memory", result["run_selection"]["preserved_failure"]["failure_reason"])
        self.assertEqual(output["run_registry"]["registry_sha256"], registry["registry_sha256"])
        self.assertEqual(output["family_status"], "INCOMPLETE")
        self.assertEqual(len(output["pairs"]), 10)

    def test_previous_retry1_registry_remains_valid_and_byte_bound(self):
        previous = report.REPO / "experiments/rab001_exploration/outputs/RUN_REGISTRY_BEFORE_RETRY2_20260913.json"
        registry = report.load_run_registry(previous)
        self.assertEqual(registry["active_tags"][self.canonical], self.canonical + "_retry1")

    def test_retry2_does_not_hide_that_a_final_score_was_already_observed(self):
        for key, value in (("known_score_is_not_retry_criterion", False), ("completed_training", False),
                           ("retry_reason", "improve_score")):
            with self.subTest(key=key):
                registry = self.registry()
                registry["retries"][self.retry][key] = value
                with self.assertRaises(report.InvalidArtifact):
                    report.validate_run_registry(registry)

    def test_failed_save_snapshot_has_observations_but_no_success_artifact(self):
        path = report.REPO / "experiments/rab001_exploration/outputs/R4_RETRY1_SAVE_FAILURE_20260913.json"
        snapshot = report.read_json(path)
        self.assertTrue(snapshot["completed_training"])
        self.assertFalse(snapshot["checkpoint_saved"])
        self.assertFalse(snapshot["overall_completed"])
        rows = snapshot["source_records"]["final_probe"]["retention"]["per_row"]
        self.assertEqual(len(rows), 150)
        self.assertEqual(sum(rows.values()), snapshot["final_summary"]["correct_total"])
        self.assertEqual(snapshot["final_summary"]["gained_by_stratum"], {"wrong_answer": 21, "format_only": 11, "runaway": 0})
        self.assertFalse((report.REPO / snapshot["successful_ARM_DOSE_artifact"]).exists())

    def test_seed23_extra_alias_is_a_soup_ingredient_and_keeps_depth_family_fixed(self):
        registry = report.load_run_registry()
        canonical = "B28_lr1e-05_b5200_seed23"
        selected = report.resolve_run(canonical, registry=registry)
        self.assertEqual(selected["active_tag"], canonical + "_retry1")
        failure = selected["preserved_failure"]
        self.assertEqual(failure["comparison_role"], "soup_ingredient_only")
        self.assertTrue(failure["no_endpoint_observed"])
        self.assertEqual(failure["observed_supervised_target_tokens"], 100)
        self.assertEqual(failure["observed_optimizer_steps"], 1)
        self.assertEqual(report.resolve_run(self.canonical, registry=registry)["active_tag"], self.retry)
        with tempfile.TemporaryDirectory() as directory:
            output = report.build_report(Path(directory), {}, registry)
        self.assertEqual(tuple(output["arms"]), ("B28last8", "B28last12", "B28last16", "B28last20", "B28"))
        self.assertEqual(len(output["pairs"]), 10)
        self.assertEqual(alpha_report.SPECS["B28"][0], self.canonical)

    def test_extra_alias_rejects_unregistered_recipe_or_retry_number(self):
        canonical = "B28_lr1e-05_b5200_seed23"
        tag = canonical + "_retry1"
        for bad_canonical, bad_tag in (("B28_lr1e-05_b5200_seed99", "B28_lr1e-05_b5200_seed99_retry1"),
                                       (canonical, canonical + "_retry2")):
            with self.subTest(alias=bad_tag):
                registry = self.registry()
                metadata = registry["retries"].pop(tag)
                registry["active_tags"].pop(canonical)
                registry["active_tags"][bad_canonical] = bad_tag
                registry["retries"][bad_tag] = {**metadata, "canonical_tag": bad_canonical}
                with self.assertRaisesRegex(report.InvalidArtifact, "unsupported registry alias"):
                    report.validate_run_registry(registry)

    def test_extra_alias_cannot_borrow_another_failure_filename(self):
        registry = self.registry()
        tag = "B28_lr1e-05_b5200_seed23_retry1"
        registry["retries"][tag]["failure_artifact_path"] = registry["retries"][self.canonical + "_retry1"]["failure_artifact_path"]
        with self.assertRaisesRegex(report.InvalidArtifact, "failure artifact path differs"):
            report.validate_run_registry(registry)

    def test_extra_alias_cannot_change_its_comparison_role(self):
        registry = self.registry()
        registry["retries"]["B28_lr1e-05_b5200_seed23_retry1"]["comparison_role"] = "depth_endpoint"
        with self.assertRaisesRegex(report.InvalidArtifact, "soup ingredient"):
            report.validate_run_registry(registry)

    def test_seed17_retry_chain_is_unchanged_by_soup_alias_addition(self):
        previous_path = report.REPO / "experiments/rab001_exploration/outputs/RUN_REGISTRY_BEFORE_SOUP_SEED23_RETRY1_20260914.json"
        previous = report.read_json(previous_path)
        current = self.registry()
        self.assertEqual(previous["active_tags"][self.canonical], current["active_tags"][self.canonical])
        for tag in (self.canonical + "_retry1", self.canonical + "_retry2"):
            self.assertEqual(previous["retries"][tag], current["retries"][tag])


if __name__ == "__main__":
    unittest.main()
