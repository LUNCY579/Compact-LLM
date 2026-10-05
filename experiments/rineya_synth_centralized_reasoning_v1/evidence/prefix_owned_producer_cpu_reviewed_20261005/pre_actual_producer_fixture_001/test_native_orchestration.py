"""CPU-only native-schema adapter tests; no native approval/probes/binaries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_fixture_support as fixtures


class NativeSchemaOrchestrationControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="native_schema_CPU_", dir=fixtures.HERE)
        self.root = Path(self.directory.name).resolve()
        self.auditor = fixtures.source_module("prefix_input_owned_auditor.py")
        self.frame = fixtures.native_frame(self.root)

    def tearDown(self):
        self.assertEqual(self.root.parent, fixtures.HERE.resolve())
        self.directory.cleanup()

    def audit_adapter(self):
        frame = self.frame
        preservation = json.loads((frame["preserved"] / "PRESERVATION.json").read_text())
        with patch.object(self.auditor, "_direct_package", return_value=True):
            return self.auditor._native_preserved_body(frame["preserved"], original_directory=frame["raw"],
                manifest=frame["manifest"], source_pins=frame["source_pins"],
                expected_preservation_file_sha256=hashlib.sha256((frame["preserved"] / "PRESERVATION.json").read_bytes()).hexdigest(),
                expected_preservation_body_sha256=preservation["sha256"], expected_head="a" * 40, anchors=frame["anchors"],
                _CPU_fixture_capability=self.auditor._mint_CPU_native_schema_fixture())

    def test_actual_native_adapter_positive_remains_CPU_test_without_native_PASS(self):
        self.assertIs(self.frame["explicit_CPU_metadata_fixture"], True)
        result = self.audit_adapter()
        self.assertEqual(result["status"], "CPU_NATIVE_SCHEMA_COMPLETE")
        self.assertIs(result["fixture_only"], True)
        for key in ("native_PASS", "native_authority_issued", "native_fit_established", "M0_membership", "scoring_authorized"):
            self.assertIs(result[key], False)

    def test_actual_native_adapter_rejects_original_copy_drift(self):
        (self.frame["raw"] / "SUPERVISOR_RECEIPT.json").write_bytes(b"{}\n")
        with self.assertRaises(ValueError):
            self.audit_adapter()

    def test_public_native_entry_stays_closed_even_with_CPU_native_schema_fixture(self):
        with patch.object(self.auditor, "Reader", side_effect=AssertionError("native path read")):
            with self.assertRaises(ValueError):
                self.auditor.native_audit(self.frame["preserved"], native_launch_authorized=True)


if __name__ == "__main__":
    unittest.main()
