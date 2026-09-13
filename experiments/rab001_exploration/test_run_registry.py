"""Explicit retry aliases must preserve, rather than conceal, the failed canonical run."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments.rab001_exploration import frontier_report as report
from experiments.rab001_exploration import alpha_report


class RunRegistryTests(unittest.TestCase):
    canonical = "B28_lr1e-05_b5200_seed17"
    retry = canonical + "_retry1"

    def registry(self):
        return report.read_json(report.RUN_REGISTRY)

    def test_active_retry_has_hash_bound_original_failure(self):
        registry = report.load_run_registry()
        selection = report.resolve_run(self.canonical, registry=registry)
        self.assertEqual(selection["active_tag"], self.retry)
        failure = selection["preserved_failure"]
        self.assertTrue(failure["no_endpoint_observed"])
        self.assertEqual(failure["observed_supervised_target_tokens"], 186)
        self.assertEqual(failure["observed_optimizer_steps"], 1)
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
            report.validate_run_identity({"canonical_tag": self.canonical, "run_label": "retry1"}, selected)

    def test_unsupported_retry_alias_is_rejected(self):
        registry = self.registry()
        second = self.canonical + "_retry2"
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
        for key, value in (("failure_artifact_sha256", "0" * 64), ("no_endpoint_observed", False),
                           ("observed_supervised_target_tokens", 5200)):
            with self.subTest(key=key):
                registry = self.registry()
                registry["retries"][self.retry][key] = value
                with self.assertRaises(report.InvalidArtifact):
                    report.validate_run_registry(registry)

    def test_retry_body_must_bind_both_canonical_tag_and_label(self):
        selection = report.resolve_run(self.canonical, registry=report.load_run_registry())
        report.validate_run_identity({"canonical_tag": self.canonical, "run_label": "retry1"}, selection)
        for body in ({"run_label": "retry1"}, {"canonical_tag": self.canonical, "run_label": "retry2"},
                     {"canonical_tag": "different", "run_label": "retry1"}):
            with self.subTest(body=body), self.assertRaisesRegex(report.InvalidArtifact, "canonical_tag/run_label"):
                report.validate_run_identity(body, selection)

    def test_alpha_rejects_alias_without_registry_selection(self):
        # Tag checks happen before any outputs are read; no fake model result is needed.
        body = {"completed": True, "tag": self.retry}
        with self.assertRaisesRegex(report.InvalidArtifact, "unregistered"):
            alpha_report.validate_sweep(body, "B28", {}, {"canonical_tag": self.canonical, "run_label": "retry1"})

    def test_missing_retry_report_keeps_original_failure_visible(self):
        registry = report.load_run_registry()
        with tempfile.TemporaryDirectory() as directory:
            output = report.build_report(Path(directory), {}, registry)
        result = output["arms"]["B28"]
        self.assertEqual(result["status"], "MISSING")
        self.assertEqual(result["run_selection"]["active_tag"], self.retry)
        self.assertIn("free physical memory", result["run_selection"]["preserved_failure"]["failure_reason"])
        self.assertEqual(output["run_registry"]["registry_sha256"], registry["registry_sha256"])
        self.assertEqual(output["family_status"], "INCOMPLETE")
        self.assertEqual(len(output["pairs"]), 10)


if __name__ == "__main__":
    unittest.main()
