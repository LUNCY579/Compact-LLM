"""No framework imports, executable launches, model assets or host DB access."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import owned_runtime_identity_probe as probe


class IdentityProbeTests(unittest.TestCase):
    def tearDown(self):
        sys.setprofile(None); threading.setprofile(None)

    def test_unknown_path_and_known_wrong_hash_stay_distinct_without_registry_growth(self):
        permitted = {"c:/python.exe": "a" * 64}
        unknown = probe.classify_identity("C:\\other.exe", "a" * 64, permitted)
        mismatch = probe.classify_identity("C:\\PYTHON.exe", "b" * 64, permitted)
        self.assertFalse(unknown["path_registered"])
        self.assertTrue(mismatch["path_registered"])
        self.assertFalse(mismatch["hash_matches"])
        self.assertEqual(permitted, {"c:/python.exe": "a" * 64})

    def test_forbidden_python_call_stops_before_side_effect(self):
        with tempfile.TemporaryDirectory() as td:
            namespace = {"__name__": "torch.cuda", "effects": []}
            exec("def is_available():\n effects.append('executed')", namespace)
            profile, violations = probe.import_boundary(Path(td) / "refusals.jsonl")
            with self.assertRaises(probe.native.NativeBlocked): namespace["is_available"]()
            self.assertEqual(namespace["effects"], [])
            self.assertTrue(violations[0]["body_execution_blocked"])
            self.assertEqual(json.loads((Path(td) / "refusals.jsonl").read_text())["name"], "is_available")

    def test_forbidden_api_in_new_python_thread_stops_before_side_effect(self):
        with tempfile.TemporaryDirectory() as td:
            namespace = {"__name__": "torch.cuda", "effects": []}
            exec("def init():\n effects.append('executed')", namespace)
            errors = []
            def invoke():
                try: namespace["init"]()
                except probe.native.NativeBlocked as error: errors.append(str(error))
            profile, violations = probe.import_boundary(Path(td) / "refusals.jsonl")
            thread = threading.Thread(target=invoke); thread.start(); thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(namespace["effects"], [])
            self.assertTrue(errors and violations)

    def test_c_call_profile_rejects_builtin_before_filesystem_effect(self):
        import os
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "must_not_exist"
            with patch.object(probe, "forbidden_call", side_effect=lambda module, name, event: event == "c_call" and name == "mkdir"):
                profile, violations = probe.import_boundary(Path(td) / "refusals.jsonl")
                with self.assertRaises(probe.native.NativeBlocked): os.mkdir(target)
                self.assertFalse(target.exists())
                self.assertEqual(violations[0]["profile_event"], "c_call")

    def test_blank_builder_duplicate_role_review_cannot_admit(self):
        with tempfile.TemporaryDirectory() as td:
            files = [Path(td) / "a.json", Path(td) / "b.json"]
            expected = {"scope": probe.SCOPE}
            for f, role in zip(files, ("operations", "science")):
                f.write_text(json.dumps({"status": "APPROVED_FOR_OWNED_IDENTITY_PROBE", "binding": expected,
                                        "role": role, "reviewer": " "}))
            with self.assertRaises(probe.native.NativeBlocked): probe.checked_reviews(files, expected)
            for f in files:
                f.write_text(json.dumps({"status": "APPROVED_FOR_OWNED_IDENTITY_PROBE", "binding": expected,
                                        "role": "operations", "reviewer": " /root "}))
            with self.assertRaises(probe.native.NativeBlocked): probe.checked_reviews(files, expected)

    def test_log_overflow_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            path.write_bytes(b"preserved")
            with patch.object(probe, "MAX_LOG", 9):
                with self.assertRaises(probe.native.NativeBlocked): probe.append(path, {"large": "x" * 30})
            self.assertEqual(path.read_bytes(), b"preserved")

    def test_contract_drift_rejects_before_framework_import(self):
        with self.assertRaises(probe.native.NativeBlocked): probe.checked_contract("0" * 64)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__": unittest.main()
