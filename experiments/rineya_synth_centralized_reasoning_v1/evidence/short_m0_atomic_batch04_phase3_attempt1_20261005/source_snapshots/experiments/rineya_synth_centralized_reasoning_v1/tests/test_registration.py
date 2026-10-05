"""Positive and negative metadata checks; no model, data, network, or GPU use."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synth_registration", PACKAGE / "validate_manifest.py")
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class RegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))

    def reject(self, section: str, key: str, value: object) -> None:
        changed = copy.deepcopy(self.data)
        changed[section][key] = value
        self.assertTrue(checker.validate(changed))

    def test_valid_founding_registration(self) -> None:
        self.assertEqual([], checker.validate(self.data))

    def test_not_an_object(self) -> None:
        for value in (None, [], True, 1, "manifest"):
            with self.subTest(value=value):
                self.assertTrue(checker.validate(value))

    def test_missing_or_unknown_fields(self) -> None:
        for key in self.data:
            changed = copy.deepcopy(self.data)
            del changed[key]
            with self.subTest(key=key):
                self.assertTrue(checker.validate(changed))
        self.data["implicit_launch_permission"] = True
        self.assertTrue(checker.validate(self.data))

    def test_identity_must_not_be_boolean_version(self) -> None:
        self.data["schema_version"] = True
        self.assertTrue(checker.validate(self.data))

    def test_parent_is_exact(self) -> None:
        self.reject("parent", "commit", "0" * 40)

    def test_non_discard_rules_are_required(self) -> None:
        for key, value in self.data["preservation"].items():
            with self.subTest(key=key):
                self.reject("preservation", key, not value)

    def test_truthy_integers_are_not_policy_booleans(self) -> None:
        self.reject("preservation", "complementary", 1)
        self.reject("preservation", "automatic_sunset", 0)

    def test_every_execution_permission_stays_closed(self) -> None:
        for key in self.data["permissions"]:
            for value in (True, None, "false", 0):
                with self.subTest(key=key, value=value):
                    self.reject("permissions", key, value)

    def test_unknown_permission_rejected(self) -> None:
        self.reject("permissions", "secret_override", True)

    def test_no_fabricated_capability_claim(self) -> None:
        for key in self.data["claims"]:
            with self.subTest(key=key):
                self.reject("claims", key, True)

    def test_student_binding_does_not_imply_qualification(self) -> None:
        self.reject("student", "checkpoint_sha256", "a" * 64)

    def test_separate_scoreboards(self) -> None:
        self.data["scoreboards"] = ["combined_accuracy"]
        self.assertTrue(checker.validate(self.data))

    def test_heldout_composition_not_in_training(self) -> None:
        self.reject("data_design", "train_signatures", ["R", "S", "RS"])

    def test_row_splits_cannot_replace_parent_families(self) -> None:
        self.reject("data_design", "split_unit", "row")

    def test_independent_verification_and_closed_banks(self) -> None:
        self.reject("data_design", "verifier", "teacher_self_grade")
        self.reject("data_design", "protected_banks", "open")

    def test_target_format_not_confounded(self) -> None:
        self.data["arms"][2]["target_format"] = "long_reasoning"
        self.assertTrue(checker.validate(self.data))

    def test_arm_boolean_is_not_integer(self) -> None:
        self.data["arms"][1]["trained"] = 1
        self.assertTrue(checker.validate(self.data))

    def test_strong_primary_control(self) -> None:
        self.data["primary_contrast"] = ["M2", "M0"]
        self.assertTrue(checker.validate(self.data))

    def test_unresolved_launch_requirements_remain_visible(self) -> None:
        self.data["open_run_bindings"].remove("launch_approval")
        self.assertTrue(checker.validate(self.data))

    def test_source_version_pinned(self) -> None:
        self.reject("source", "paper", "https://arxiv.org/abs/2609.37891")

    def test_cli_valid_does_not_authorize_execution(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = checker.main([str(PACKAGE / "manifest.json")])
        self.assertEqual(0, code)
        result = json.loads(output.getvalue())
        self.assertFalse(result["native_execution_authorized"])
        self.assertFalse(result["model_quality_evidence"])

    def test_cli_bad_json_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for text in ('{', '{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(text, encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(2, checker.main([str(path)]))

    def test_cli_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(2, checker.main([str(Path(directory) / "absent.json")]))


if __name__ == "__main__":
    unittest.main()
