"""Synthetic packet mechanics only; these tests create no native qualification.

Every positive packet is invented test evidence. Acceptance means only that the
offline schema/hash checks work, never that Rineya or human review is qualified.
Temporary files live beside the checkout, outside tracked source.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from experiments.rineya_recursive_self_distillation_v1.catalog_qualification import (
    REVIEW_STATUS, validate_catalog_review,
)
from experiments.rineya_recursive_self_distillation_v1.contracts import file_hash
from experiments.rineya_recursive_self_distillation_v1.data import demo_catalog, write_catalog


class CatalogReviewTests(unittest.TestCase):
    def setUp(self):
        # Current checkout's parent is BusinessProject. Never place test outputs
        # under the tracked checkout or .codex.
        output_root = Path(__file__).resolve().parents[3].parent
        self.tmp = tempfile.TemporaryDirectory(prefix="rsd001-catalog-review-test-", dir=output_root)
        self.root = Path(self.tmp.name).resolve()
        if self.root.parent != output_root.resolve():
            raise RuntimeError("fixture output containment failed")
        self.catalog = self.root / "catalog.json"
        self.exclusions = self.root / "exclusions.json"
        self.review = self.root / "review.json"
        self.tasks = [replace(task, provenance="Verified-shaped unit test metadata; no native evidence")
                      for task in demo_catalog(1)]
        write_catalog(self.catalog, self.tasks)
        self.exclusions.write_text(json.dumps({"parents": [], "normalized_prompt_sha256": []}), encoding="utf-8")
        for name in ("ownership.txt", "reference-review.txt", "exclusion-authority.txt"):
            (self.root / name).write_text("Invented UNIT TEST evidence, not actual provenance review.\n", encoding="utf-8")
        self.packet = {
            "schema": 1, "experiment": "RSD-001", "status": REVIEW_STATUS,
            "evidence_kind": "real_review", "catalog": self.bind(self.catalog),
            "exclusions": self.bind(self.exclusions), "reviewed_by": "Synthetic test reviewer",
            "sources": [{"source": self.tasks[0].source, "revision": self.tasks[0].revision,
                         "ownership_license_evidence": self.bind(self.root / "ownership.txt"),
                         "independent_reference_review_evidence": self.bind(self.root / "reference-review.txt")}],
            "global_parent_grouping_attested": True,
            "historical_exclusion_authority_evidence": self.bind(self.root / "exclusion-authority.txt"),
            "reference_review_complete": True, "parent_training_overlap_reviewed": True,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def bind(self, path):
        return {"path": path.name, "sha256": file_hash(path)}

    def validate(self, packet=None):
        self.review.write_text(json.dumps(self.packet if packet is None else packet), encoding="utf-8")
        return validate_catalog_review(self.catalog, self.exclusions, self.review)

    def rewrite_catalog(self, tasks):
        self.catalog.unlink()
        write_catalog(self.catalog, tasks)
        self.packet["catalog"] = self.bind(self.catalog)

    def test_valid_shaped_packet_preserves_limitations(self):
        result = self.validate()
        self.assertEqual(result["status"], REVIEW_STATUS)
        self.assertEqual(result["audit"]["rows"], 5)
        self.assertEqual(result["review"]["sha256"], file_hash(self.review))
        self.assertFalse(result["semantic_decontamination"])
        self.assertFalse(result["native_efficacy_claim"])
        self.assertIn("endpoint-only", result["reference_validation"])
        self.assertIn("truth is not established", result["evidence_limit"])

    def test_legacy_unknown_and_other_experiment_fields_rejected(self):
        for field, value in (("schema", True), ("schema", 2), ("experiment", "DSD-001"),
                             ("status", "QUALIFIED_CAPABILITY"), ("evidence_kind", "fixture")):
            with self.subTest(field=field, value=value):
                packet = dict(self.packet, **{field: value})
                with self.assertRaises(ValueError):
                    self.validate(packet)
        with self.assertRaises(ValueError):
            self.validate(dict(self.packet, unknown=True))

    def test_missing_attestations_and_reviewer_rejected(self):
        for field in ("global_parent_grouping_attested", "reference_review_complete",
                      "parent_training_overlap_reviewed"):
            for value in (False, 1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.validate(dict(self.packet, **{field: value}))
        with self.assertRaises(ValueError):
            self.validate(dict(self.packet, reviewed_by="  "))

    def test_catalog_and_exclusions_hash_drift_rejected(self):
        for field in ("catalog", "exclusions"):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                packet[field]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "bound file drift"):
                    self.validate(packet)

    def test_hash_correct_other_catalog_path_rejected(self):
        other = self.root / "other.json"
        other.write_bytes(self.catalog.read_bytes())
        packet = dict(self.packet, catalog=self.bind(other))
        with self.assertRaisesRegex(ValueError, "input path mismatch"):
            self.validate(packet)

    def test_missing_extra_duplicate_and_wrong_revision_sources_rejected(self):
        cases = [[], self.packet["sources"] * 2]
        wrong = copy.deepcopy(self.packet["sources"])
        wrong[0]["revision"] = "other-revision"
        cases.append(wrong)
        extra = copy.deepcopy(self.packet["sources"])
        extra.append(dict(extra[0], source="unenrolled-source"))
        cases.append(extra)
        for sources in cases:
            with self.subTest(sources=sources):
                with self.assertRaises(ValueError):
                    self.validate(dict(self.packet, sources=sources))

    def test_missing_ownership_or_reference_evidence_rejected(self):
        for field in ("ownership_license_evidence", "independent_reference_review_evidence"):
            packet = copy.deepcopy(self.packet)
            del packet["sources"][0][field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "source fields"):
                    self.validate(packet)

    def test_missing_empty_and_drifted_evidence_rejected(self):
        evidence = self.root / "reference-review.txt"
        evidence.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty evidence file"):
            self.validate()
        evidence.write_text("Changed synthetic unit test evidence", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound file drift"):
            self.validate()
        evidence.unlink()
        with self.assertRaisesRegex(ValueError, "missing/empty evidence file"):
            self.validate()

    def test_evidence_cannot_be_enrollment_or_review_document(self):
        packet = copy.deepcopy(self.packet)
        packet["sources"][0]["ownership_license_evidence"] = self.bind(self.catalog)
        with self.assertRaisesRegex(ValueError, "separate evidence file"):
            self.validate(packet)
        packet = copy.deepcopy(self.packet)
        packet["historical_exclusion_authority_evidence"] = self.bind(self.exclusions)
        with self.assertRaisesRegex(ValueError, "separate evidence"):
            self.validate(packet)

    def test_fixture_provenance_cannot_enroll(self):
        self.rewrite_catalog(demo_catalog(1))
        with self.assertRaisesRegex(ValueError, "fixture cannot enroll"):
            self.validate()

    def test_missing_role_and_retention_source_drift_rejected(self):
        self.rewrite_catalog(self.tasks[:-1])
        with self.assertRaisesRegex(ValueError, "all five"):
            self.validate()
        self.rewrite_catalog([replace(task, source="other-retention-source")
                              if task.split == "retention_train" else task for task in self.tasks])
        with self.assertRaisesRegex(ValueError, "retention training/evaluation sources"):
            self.validate()

    def test_cross_role_parent_collision_rejected(self):
        # Write malformed enrollment directly: write_catalog itself rejects this.
        from dataclasses import asdict
        rows = [asdict(task) for task in self.tasks]
        rows[1]["parent"] = rows[0]["parent"]
        self.catalog.write_text(json.dumps(rows), encoding="utf-8")
        self.packet["catalog"] = self.bind(self.catalog)
        with self.assertRaisesRegex(ValueError, "cross-role collision"):
            self.validate()

    def test_cross_role_normalized_prompt_collision_rejected(self):
        from dataclasses import asdict
        rows = [asdict(task) for task in self.tasks]
        rows[1]["prompt"] = "  " + rows[0]["prompt"].upper() + "  "
        self.catalog.write_text(json.dumps(rows), encoding="utf-8")
        self.packet["catalog"] = self.bind(self.catalog)
        with self.assertRaisesRegex(ValueError, "cross-role collision"):
            self.validate()

    def test_historical_exclusion_collision_rejected(self):
        self.exclusions.write_text(json.dumps({"parents": [self.tasks[0].parent],
                                               "normalized_prompt_sha256": []}), encoding="utf-8")
        self.packet["exclusions"] = self.bind(self.exclusions)
        with self.assertRaisesRegex(ValueError, "exclusion collision"):
            self.validate()

    def test_no_model_imports_in_validator(self):
        import subprocess
        import sys
        command = ("import sys; "
                   "import experiments.rineya_recursive_self_distillation_v1.catalog_qualification; "
                   "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")
        subprocess.run([sys.executable, "-B", "-c", command], check=True,
                       cwd=Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    unittest.main()
