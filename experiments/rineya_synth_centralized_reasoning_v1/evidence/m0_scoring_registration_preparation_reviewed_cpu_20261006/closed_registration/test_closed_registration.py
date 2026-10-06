"""Two pure metadata checks; no scorer, model, capture or DEV input imports."""
import json
from pathlib import Path
import unittest
import build_closed_registration as builder


class ClosedRegistrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = json.loads((builder.HERE / "M0_SCORING_REGISTRATION_CLOSED_TEMPLATE_002.json").read_bytes())

    def test_closed_permissions_and_honest_missing_population(self):
        value = self.value
        for name in ("native_scoring_authorized", "owner_native_launch_authorized", "training_authorized",
                     "protected_bank_access", "confirmation_authorized", "promotion_authorized"):
            self.assertIs(value[name], False)
        self.assertEqual(len(value["batches"]), 6)
        self.assertTrue(all(type(b) is dict for b in value["batches"][:4]))
        self.assertEqual(value["batches"][4:], [None, None])
        self.assertEqual(value["known_audited_raw_rows"], 256)
        self.assertEqual([value["complete_rows_required"], value["complete_joined_parents_required"]], [384, 96])
        for name in ("review_anchor", "approval_anchor", "trusted_audit_anchor_selection", "panel_registration_sha256"):
            self.assertIsNone(value[name])
        self.assertIs(value["requires_treatment_checkpoints"], False)
        self.assertFalse(Path(value["output_path"]).is_relative_to(builder.ROOT))
        self.assertTrue(all(v == 0 for v in value["activity"].values()))

    def test_exact_source_policy_metadata_anchors(self):
        value = self.value
        self.assertEqual(value["sha256"], builder.digest({k: v for k, v in value.items() if k != "sha256"}))
        self.assertEqual(len(value["scorer_sources"]), 8)
        for name, ref in value["scorer_source_descriptors"].items():
            self.assertEqual(builder.metadata(ref["path"]), ref)
        frozen, ref = builder.document(builder.SCORER_STAGE / "SOURCE_FROZEN_003.json")
        self.assertEqual(value["staged_source_freeze"], ref)
        for name, pin in value["implementation_files"].items():
            self.assertEqual(builder.metadata(builder.SCORER_STAGE / name)["sha256"], pin)
            self.assertEqual(frozen["source_files"][name]["sha256"], pin)
        finish, ref = builder.document(builder.PACKAGE / "DEV_FINISH_POLICY_001.json")
        for name in ("correctness", "generation", "timing_disclosure"):
            self.assertEqual(value[name], finish[name])
        self.assertEqual(value["finish_policy_file_sha256"], ref["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
