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
        self.frame = fixtures.actual_native_frame(self.root)

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

    def test_actual_producer_source_and_container_writer_emit_genuine_core_bytes(self):
        produced = fixtures.producer_frame(self.root)
        proof, report = produced["proof"], produced["report"]
        self.assertEqual(proof["kind"], "CPU_NATIVE_CALLER_SOURCE_SIMULATION")
        self.assertIs(proof["fixture_only"], True)
        self.assertFalse(proof["native_PASS"])
        self.assertFalse(proof["native_authority_issued"])
        self.assertEqual(report["status"], "NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT")
        self.assertEqual(report["scope"], self.auditor.NATIVE_SCOPE)
        self.assertEqual(produced["host"].cpu[-1]["available_bytes"], 8 * 1024**3)
        raw = produced["raw"]
        worker = json.loads((raw / "worker_result.json").read_text())
        final = json.loads((raw / "PHASE_ACCOUNTING_FINAL.json").read_text())
        container = json.loads((raw / "OWNED_WORKER_RETURN.json").read_text())
        self.assertEqual(worker["status"], report["status"])
        self.assertEqual(final, report["final_phase_summary"])
        self.assertEqual(container, produced["container"])
        self.assertEqual(container["report"], report)
        self.assertEqual(container["IO"], {"owned_return_write": {"attempted": 1, "returned": 1, "failed": 0, "pending": 0}})
        self.assertEqual(len(container["owned_membership_verified"]), 7)
        phase = [json.loads(line) for line in (raw / "PHASE_OBSERVATIONS.jsonl").read_text().splitlines()]
        observed_owner = next(row["result"] for row in phase if row["operation"] == "worker_ownership_verify" and row["event"] == "returned")
        self.assertEqual(observed_owner, container["owned_membership_verified"])
        self.assertFalse(container["native_PASS"])

    def test_actual_producer_source_preserves_ID_shortcircuit_failure(self):
        produced = fixtures.producer_frame(self.root, api_faults={"ids.shape": (1, 421)})
        self.assertEqual(produced["report"]["status"], "NATIVE_INPUT_CHECKS_FAILED")
        reads = [event for event in produced["api"].recorder.events if event.startswith(("ids.", "mask.")) and not event.endswith(".released")]
        self.assertEqual(reads, ["ids.is_tensor", "ids.shape"])
        self.assertEqual(produced["report"]["error"], {"type": "ValueError", "message": "exact unaltered input tensor"})
        self.assertFalse(produced["proof"]["native_PASS"])
        self.assertFalse(produced["container"]["native_PASS"])

    def test_actual_producer_bytes_feed_independent_native_adapter_as_CPU_only(self):
        base=self.root/'actual_adapter'
        base.mkdir()
        self.frame=fixtures.actual_native_frame(base)
        result=self.audit_adapter()
        self.assertEqual(result['status'],'CPU_NATIVE_SCHEMA_COMPLETE')
        self.assertTrue(result['fixture_only'])
        self.assertFalse(result['native_PASS'])
        for name in ('PHASE_OBSERVATIONS.jsonl','ownership_verified.json','input_check_summary.json',
                     'worker_result.json','PHASE_ACCOUNTING_FINAL.json','OWNED_WORKER_RETURN.json'):
            self.assertEqual((self.frame['raw']/name).read_bytes(),(self.frame['producer']['raw']/name).read_bytes())


if __name__ == "__main__":
    unittest.main()
