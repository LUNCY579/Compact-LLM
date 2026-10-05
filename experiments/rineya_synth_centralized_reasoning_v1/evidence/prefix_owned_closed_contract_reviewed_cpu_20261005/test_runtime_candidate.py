"""Inert CPU controls for bounded metadata/source readers; no native closure."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
HELPER = HERE / "runtime_candidate_002.py"
SPEC = importlib.util.spec_from_file_location("exact_owned_runtime_candidate_002_fixture", HELPER)
runtime = importlib.util.module_from_spec(SPEC)
# Execute the exact spec's explicit UTF8 source, bypassing optional pyc reads.
exec(compile(HELPER.read_text(encoding="utf-8"), str(HELPER), "exec"), runtime.__dict__)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def sealed(value):
    return {**value, "sha256": digest(value)}


class RuntimeCandidateControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="runtime_fixture_", dir=HERE)
        self.root = Path(self.directory.name).resolve()

    def tearDown(self):
        self.assertEqual(self.root.parent, HERE.resolve())
        self.directory.cleanup()

    def write_json(self, name="fixture.json", value=None):
        if value is None:
            value = sealed({"kind": "CPU_RUNTIME_METADATA_FIXTURE", "value": 1,
                            "native_launch_authorized": False})
        path = self.root / name
        raw = (canonical(value) + "\n").encode("ascii")
        path.write_bytes(raw)
        return path, value, hashlib.sha256(raw).hexdigest()

    def test_exact_source_spec_and_strict_duplicate_key_rejection(self):
        self.assertEqual(Path(runtime.__file__).resolve(), HELPER.resolve())
        self.assertEqual(Path(SPEC.origin).resolve(), HELPER.resolve())
        for raw in (b'{"a":1,"a":2}', b'{"outer":{"a":1,"a":2}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runtime.strict_json(raw)

    def test_nonfinite_constants_and_ignored_overflowing_float_reject(self):
        for raw in (b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}',
                    b'{"ignored":{"value":1e999}}', b'{"ignored":-1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runtime.strict_json(raw)
        self.assertEqual(runtime.strict_json(b'{"value":1.25}'), {"value": 1.25})

    def test_external_file_and_body_anchors_preserve_closed_metadata(self):
        path, expected, file_hash = self.write_json()
        value, anchor = runtime.json_source(path, file_hash, expected["sha256"])
        self.assertEqual(value, expected)
        self.assertIs(value["native_launch_authorized"], False)
        self.assertEqual(anchor, {"path": str(path), "bytes": path.stat().st_size,
                                 "file_sha256": file_hash, "body_sha256": expected["sha256"],
                                 "external_anchor_verified": True})
        for file_pin, body_pin in (("0" * 64, expected["sha256"]), (file_hash, "0" * 64)):
            with self.subTest(file_pin=file_pin, body_pin=body_pin), self.assertRaises(ValueError):
                runtime.json_source(path, file_pin, body_pin)

    def test_altered_resealed_document_cannot_refresh_external_anchors(self):
        path, original, original_file = self.write_json()
        replacement = sealed({"kind": "CPU_RUNTIME_METADATA_FIXTURE", "value": 2,
                              "native_launch_authorized": False})
        raw = (canonical(replacement) + "\n").encode("ascii")
        path.write_bytes(raw)
        with self.assertRaises(ValueError):
            runtime.json_source(path, original_file, original["sha256"])
        with self.assertRaises(ValueError):
            runtime.json_source(path, hashlib.sha256(raw).hexdigest(), original["sha256"])

    def test_matching_file_anchor_cannot_hide_wrong_internal_body_seal(self):
        path, original, _ = self.write_json()
        wrong = {**original, "value": 2}
        raw = (canonical(wrong) + "\n").encode("ascii")
        path.write_bytes(raw)
        with self.assertRaises(ValueError):
            runtime.json_source(path, hashlib.sha256(raw).hexdigest(), original["sha256"])

    def test_maximum_and_expected_size_drift_fail_before_any_content_open(self):
        path = self.root / "source.py"
        path.write_bytes(b"abc")
        self.assertEqual(runtime.MAX_JSON, 16 * 1024**2)
        with patch.object(runtime.Path, "open", side_effect=AssertionError("content opened")) as opened:
            for maximum, expected in ((2, None), (16, 4), (16, True)):
                with self.subTest(maximum=maximum, expected=expected), self.assertRaises(ValueError):
                    runtime.bounded_bytes(path, maximum, expected)
            opened.assert_not_called()

    def test_bounded_read_requests_expected_size_plus_one_only(self):
        path = self.root / "source.py"
        raw = b"inert\n"
        path.write_bytes(raw)
        sizes = []
        class Reader(io.BytesIO):
            def read(self, size=-1):
                sizes.append(size)
                return super().read(size)
        with patch.object(runtime.Path, "open", return_value=Reader(raw)) as opened:
            self.assertEqual(runtime.bounded_bytes(path, 16 * 1024**2, len(raw)), raw)
            opened.assert_called_once_with("rb")
        self.assertEqual(sizes, [len(raw) + 1])

    def test_file_size_change_during_read_breaks_stability(self):
        path = self.root / "source.py"
        raw = b"inert\n"
        path.write_bytes(raw)
        actual_open = Path.open
        class MutatingReader(io.BytesIO):
            def read(self, size=-1):
                with actual_open(path, "ab") as changed:
                    changed.write(b"!")
                return super().read(size)
        with patch.object(runtime.Path, "open", return_value=MutatingReader(raw)):
            with self.assertRaises(ValueError):
                runtime.bounded_bytes(path, 16 * 1024**2, len(raw))

    def test_finite_four_suffix_roster_excludes_cache_pyc_and_package_data(self):
        root = self.root / "inert_package"
        root.mkdir()
        for name in ("alpha.py", "beta.pyi", "extension.pyd", "library.dll",
                     "ignored.pyc", "data.json", "payload.dat"):
            (root / name).write_bytes(b"CPU_INERT_PLACEHOLDER")
        cache = root / "__pycache__"
        cache.mkdir()
        for name in ("hidden.py", "hidden.pyi", "hidden.pyd", "hidden.dll", "hidden.pyc"):
            (cache / name).write_bytes(b"CPU_INERT_EXCLUDED")
        with patch.object(runtime.Path, "open", side_effect=AssertionError("roster read file content")) as opened:
            roster = runtime.metadata_roster(root)
            opened.assert_not_called()
        expected = {str(root / name) for name in ("alpha.py", "beta.pyi", "extension.pyd", "library.dll")}
        self.assertEqual(set(roster), expected)
        self.assertEqual(list(roster), sorted(roster))
        self.assertEqual(roster[str(root / "alpha.py")]["role"], "torch_python_source")
        self.assertEqual(roster[str(root / "beta.pyi")]["role"], "torch_python_stub_source")
        for name in ("extension.pyd", "library.dll"):
            self.assertEqual(roster[str(root / name)]["role"], "torch_binary_UNREAD_METADATA_ONLY")
        with patch.object(runtime, "MAX_FILES", 2), self.assertRaises(ValueError):
            runtime.metadata_roster(root)

    def test_binary_hash_requests_refuse_before_metadata_or_content_reads(self):
        with patch.object(runtime, "bounded_bytes", side_effect=AssertionError("binary read attempted")) as read:
            for suffix in (".dll", ".pyd", ".pyc", ".dat"):
                with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                    runtime.hash_source(self.root / ("not_opened" + suffix), 4)
            read.assert_not_called()
        for suffix in (".py", ".pyi"):
            path = self.root / ("inert" + suffix)
            raw = b"# CPU source fixture\n"
            path.write_bytes(raw)
            self.assertEqual(runtime.hash_source(path, len(raw)), hashlib.sha256(raw).hexdigest())

    def test_overlap_pin_conflicts_and_invalid_pin_shapes_reject(self):
        first, second = "a" * 64, "b" * 64
        self.assertEqual(runtime.merge_exact({"z": first}, {"a": second}, {"z": first}),
                         {"a": second, "z": first})
        with self.assertRaises(ValueError):
            runtime.merge_exact({"same": first}, {"same": second})
        for value in ([], {"": first}, {"source": "A" * 64}, {"source": True}, {"source": "a" * 63}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.pin_map(value)

    def test_json_source_suffix_gate_and_framework_absence(self):
        with patch.object(runtime, "bounded_bytes", side_effect=AssertionError("non-JSON read attempted")) as read:
            with self.assertRaises(ValueError):
                runtime.json_source(self.root / "not_opened.dll", "a" * 64, "b" * 64)
            read.assert_not_called()
        forbidden = {"torch", "transformers", "jax", "tensorflow", "flax"}
        self.assertFalse(forbidden.intersection(name.split(".", 1)[0] for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
