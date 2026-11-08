"""Small temporary-bundle tests; never import or execute archived source."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("publication_verifier", Path(__file__).with_name("verify_bundle.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class ArchiveVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "archive"
        self.root.mkdir()
        self.payload = self.root / "bundle/Titan-Tiny/notes.txt"
        self.payload.parent.mkdir(parents=True)
        self.payload.write_bytes(b"portable offline notes\n")
        data = self.payload.read_bytes()
        self.manifest = {"schema": "TITAN_OFFLINE_PUBLICATION_V1", "status": "REVIEWED_OFFLINE_ARCHIVE_UNADMITTED",
            "publication_date": "2026-10-09", "native_model_calls": 0, "native_authorized": False,
            "confirmation_materialized": False, "model_quality_gain_established": False,
            "control_files": ["README.md"], "files": [{
            "path": "bundle/Titan-Tiny/notes.txt", "original_root": "Titan-Tiny",
            "original_relative_path": "notes.txt", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}],
            "exclusions": []}
        (self.root / "README.md").write_text("Fixture archive only.\n", encoding="utf-8")
        self.save_manifest()

    def save_manifest(self):
        (self.root / verifier.MANIFEST).write_text(json.dumps(self.manifest), encoding="utf-8")

    def check(self, expected, phrase="", science=False):
        output = io.StringIO()
        argv = ["--root", str(self.root)] + ([] if science else ["--hash-only"])
        with contextlib.redirect_stdout(output):
            result = verifier.main(argv)
        self.assertEqual(result, expected, output.getvalue())
        self.assertIn(phrase, output.getvalue())

    def test_portable_archive(self):
        self.check(0, "hash/inventory only")

    def test_same_size_tamper(self):
        self.payload.write_bytes(b"Portable offline notes\n")
        self.check(2, "SHA-256 mismatch")

    def test_length_tamper(self):
        self.payload.write_bytes(b"short\n")
        self.check(2, "byte length mismatch")

    def test_path_traversal(self):
        for path in ["../notes.txt", "/outside.txt", "C:/outside.txt", "bundle//Titan-Tiny/notes.txt", "bundle\\Titan-Tiny\\notes.txt"]:
            with self.subTest(path=path):
                self.manifest["files"][0]["path"] = path
                self.save_manifest()
                self.check(2)

    def test_origin_traversal(self):
        self.manifest["files"][0]["original_relative_path"] = "../notes.txt"
        self.save_manifest()
        self.check(2, "unsafe path")

    def test_untracked_member_and_empty_directory(self):
        extra = self.root / "extra.txt"
        extra.write_text("untracked", encoding="utf-8")
        self.check(2, "untracked file")
        extra.unlink()
        (self.root / "extra").mkdir()
        self.check(2, "untracked directory")

    def test_arbitrary_control_bypass(self):
        self.manifest["control_files"].append("payload.py")
        self.save_manifest()
        self.check(2, "unsupported control_files")

    def test_binary_disguised_as_text(self):
        self.payload.write_bytes(b"opaque\x00payload")
        self.check(2, "binary NUL")

    def test_environment_and_cache_members(self):
        for name in [".env", "__pycache__"]:
            path = self.root / name
            if name == ".env":
                path.write_text("SECRET=fixture", encoding="utf-8")
            else:
                path.mkdir()
            self.check(2, "environment/cache")
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()

    def test_out_of_root_symlink(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("external fixture", encoding="utf-8")
        self.payload.unlink()
        try:
            self.payload.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not permit file symlink creation")
        self.check(2, "link/reparse point")

    def test_out_of_root_regular_member(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("external fixture", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes archive root"):
            verifier.check_member(self.root, outside)

    def test_scientific_publication_authority_guard(self):
        self.manifest["native_authorized"] = True
        self.save_manifest()
        self.check(2, "publication claim guards: native_authorized", science=True)

    def test_argument_failure_uses_stdout_summary(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = verifier.main(["--unsupported-argument"])
        self.assertEqual(result, 2)
        self.assertIn("BLOCKED/FAIL", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_top_level_manifest_array_uses_failure_summary(self):
        (self.root / "PUBLICATION_MANIFEST.json").write_text("[]", encoding="utf-8")
        self.check(2, "manifest must be a JSON object")

    def test_science_requires_real_snapshot(self):
        self.check(2, "required frozen snapshot absent", science=True)


if __name__ == "__main__":
    unittest.main()
