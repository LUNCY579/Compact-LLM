"""Tiny stdlib fixtures for narrowly authorized archival source resolution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import paired
else:
    import paired


class SourceRelocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original, self.archive = self.root / "original", self.root / "archive"
        for root in (self.original, self.archive):
            for package in ("model_quality_gpu_v1", "reasoning_posttraining_v1", "architecture_benchmark_001"):
                directory = root / "experiments" / package
                directory.mkdir(parents=True)
                (directory / "core.py").write_bytes(b"frozen = True\n")
            rab = root / "experiments/architecture_benchmark_001"
            (rab / "protocol.json").write_text(json.dumps({"amendments": ["amendments/frozen.md"]}))
            (rab / "amendments").mkdir()
            (rab / "amendments/frozen.md").write_bytes(b"frozen amendment\n")
            for name in ("source_lock.json", "requirements.txt", "requirements-native.txt"):
                (rab / name).write_bytes(b"{}\n")
            for relative in (*paired.SOURCE_CODE_ROLES, *paired.ADDITIONAL_INPUT_PATHS):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((relative + "\n").encode())
        self.python = self.root / "pinned-python.exe"; self.python.write_bytes(b"pinned Python")
        self.manifest_path = self.root / "SOURCE_RELOCATION.json"
        self.manifest = {"schema": 1, "record": "DOSELORA_SOURCE_RELOCATION", "status": "QUALIFIED",
                         "original_root": str(self.original), "archive_root": str(self.archive),
                         "git_head": "a" * 40, "git_branch": "frozen-branch", "python_path": str(self.python),
                         "qualification_files_sha256": paired.source_qualification_files(self.archive), "files": []}
        for relative in sorted(set(paired.SOURCE_CODE_ROLES) | paired.ADDITIONAL_INPUT_PATHS):
            self.manifest["files"].append({"role": paired.SOURCE_CODE_ROLES.get(relative, "additional_input"),
                            "original_path": str(self.original / relative), "archive_path": str(self.archive / relative),
                            "sha256": paired.sha256_file(self.archive / relative)})
        self.write_manifest()
        context = patch.object(paired, "_git_identity", return_value=("a" * 40, "frozen-branch"))
        context.start(); self.addCleanup(context.stop)
        self.script = "experiments/rab001_exploration/process_probe.py"
        self.wrapper = "docs/model-quality/audit-2026-09-17/supervised_probe.py"

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.pin = paired.sha256_file(self.manifest_path)

    def load(self):
        return paired.SourceArchive.load(self.manifest_path, self.pin)

    def test_changed_original_script_uses_exact_archive_bytes_without_rewriting_record(self):
        expected = paired.sha256_file(self.archive / self.script)
        original_path = self.original / self.script
        original_path.write_bytes(b"user edits must survive\n")
        binding = {"script_path": str(original_path), "script_sha256": expected}
        snapshot = dict(binding)
        resolved = paired._bound_file(binding, self.root, "script", self.load())
        self.assertEqual(resolved, self.archive / self.script)
        self.assertEqual(binding, snapshot)
        self.assertEqual(original_path.read_bytes(), b"user edits must survive\n")

    def test_missing_original_code_can_use_archive_but_wrong_expected_sha_cannot(self):
        expected = paired.sha256_file(self.archive / self.script)
        (self.original / self.script).unlink()
        archive = self.load()
        self.assertEqual(archive.resolve_code(self.original / self.script, expected, "script"), self.archive / self.script)
        with self.assertRaisesRegex(paired.ComparabilityError, "not authorized"):
            archive.resolve_code(self.original / self.script, "0" * 64, "script")

    def test_unauthorized_role_and_path_are_rejected(self):
        entry = self.manifest["files"][0]
        entry["role"] = "weights"; self.write_manifest()
        with self.assertRaisesRegex(paired.ComparabilityError, "unauthorized source relocation role"):
            self.load()
        entry["role"] = "script"
        entry["original_path"] = str(self.original / "model.safetensors")
        entry["archive_path"] = str(self.archive / "model.safetensors"); self.write_manifest()
        with self.assertRaisesRegex(paired.ComparabilityError, "unauthorized source relocation path/role"):
            self.load()

    def test_scientific_views_raw_results_and_receipts_cannot_use_code_fallback(self):
        archive = self.load()
        for role in ("views", "raw_result", "arm_receipt", "materialization", "weight"):
            binding = {role + "_path": str(self.original / self.script),
                       role + "_sha256": paired.sha256_file(self.archive / self.script)}
            with self.subTest(role=role), self.assertRaisesRegex(paired.ComparabilityError, "restricted"):
                paired._bound_file(binding, self.root, role, archive)

    def test_unqualified_cwd_head_branch_and_python_are_rejected(self):
        archive = self.load()
        self.assertEqual(archive.canonical_cwd(self.archive, self.python), str(self.original))
        with self.assertRaisesRegex(paired.ComparabilityError, "unqualified probe cwd"):
            archive.canonical_cwd(self.root, self.python)
        with self.assertRaisesRegex(paired.ComparabilityError, "pinned interpreter"):
            archive.canonical_cwd(self.archive, self.root / "another-python.exe")
        with patch.object(paired, "_git_identity", side_effect=[("a" * 40, "frozen-branch"), ("b" * 40, "frozen-branch")]):
            with self.assertRaisesRegex(paired.ComparabilityError, "Git HEAD/branch differs"):
                self.load()

    def test_native_package_change_and_archive_script_change_cannot_be_waived(self):
        source = self.original / "experiments/model_quality_gpu_v1/core.py"
        source.write_bytes(b"changed native package")
        with self.assertRaisesRegex(paired.ComparabilityError, "original package hash changed"):
            self.load()
        source.write_bytes(b"frozen = True\n")
        (self.archive / self.script).write_bytes(b"wrong archived script")
        with self.assertRaisesRegex(paired.ComparabilityError, "archive qualification hash mismatch"):
            self.load()

    def test_only_verified_additional_input_keys_normalize(self):
        archive = self.load()
        relative = sorted(paired.ADDITIONAL_INPUT_PATHS)[0]
        expected = paired.sha256_file(self.archive / relative)
        old = {str(self.original / relative): expected}
        new = {str(self.archive / relative): expected}
        self.assertEqual(archive.normalize_additional_inputs(old), archive.normalize_additional_inputs(new))
        with self.assertRaisesRegex(paired.ComparabilityError, "identity differs"):
            archive.normalize_additional_inputs({str(self.archive / relative): "0" * 64})
        a, b = {"unlisted-original.json": expected}, {"unlisted-archive.json": expected}
        self.assertNotEqual(archive.normalize_additional_inputs(a), archive.normalize_additional_inputs(b))
        (self.original / relative).unlink()
        with self.assertRaisesRegex(paired.ComparabilityError, "identity differs"):
            archive.normalize_additional_inputs(new)

    def test_manifest_is_pinned_and_mapping_cannot_collapse_inputs(self):
        archive = self.load()
        relative = sorted(paired.ADDITIONAL_INPUT_PATHS)[0]
        expected = paired.sha256_file(self.archive / relative)
        with self.assertRaisesRegex(paired.ComparabilityError, "collapse"):
            archive.normalize_additional_inputs({str(self.original / relative): expected, str(self.archive / relative): expected})
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b" ")
        with self.assertRaisesRegex(paired.ComparabilityError, "prospective pin"):
            self.load()

    def make_probe(self, name, values, source_root):
        arm = self.root / name; arm.mkdir()
        weight = arm / "model.safetensors"; weight.write_bytes(name.encode())
        weight_sha = paired.sha256_file(weight)
        identity = {"source_weight_sha256": "c" * 64, "checkpoint_manifest_sha256": "d" * 64}
        def write(path, value): path.write_text(json.dumps(value), encoding="utf-8")
        write(arm / "ARM.json", {"arm_weight_sha256": weight_sha, **identity})
        write(arm / "MATERIALIZATION.json", {"out_weight_sha256": weight_sha, **identity})
        for filename in paired.SIDE_CARS: (arm / filename).write_bytes(b"sidecar")
        views = self.root / "views.jsonl"
        if not views.exists(): views.write_text("".join(json.dumps({"id": str(i)}) + "\n" for i in range(6)))
        raw = arm / "PROCESS.json"
        write(raw, {"record": "PROCESS_TRANSITION_PROBE", "model_dir": str(arm), "instrument": "fixture",
                    "implementation_digest": "b" * 64, "views": 6, "views_sha256": paired.sha256_file(views),
                    "max_new_tokens": 128, "runtime": {}, "numerics": {},
                    "predictions": [{"id": str(i), "canonical": value} for i, value in enumerate(values)]})
        script_path, wrapper_path = source_root / self.script, source_root / self.wrapper
        expected_script = paired.sha256_file(self.archive / self.script)
        write(arm / "supervisor.json", {"status": "EXITED", "returncode": 0})
        write(arm / "terminal.json", {"status": "COMPLETED", "complete": True, "system_exit_code": 0,
                                      "failure": None, "script_unchanged": True, "script_sha256_after": expected_script,
                                      "script_hash_error": None, "result": str(raw)})
        binding = {"schema": 1, "kind": "process", "status": "COMPLETE", "complete": True, "returncode": 0,
                   "arm_dir": str(arm), "weight_path": str(weight), "weight_sha256": weight_sha,
                   "weight_sha256_before": weight_sha, "weight_sha256_after": weight_sha,
                   "implementation_digest": "b" * 64, "additional_input_sha256_before": {}, "additional_input_sha256_after": {}}
        for stem, path in (("raw_result", raw), ("arm_receipt", arm / "ARM.json"),
                           ("materialization", arm / "MATERIALIZATION.json"), ("views", views),
                           ("supervisor", arm / "supervisor.json"), ("terminal", arm / "terminal.json"),
                           ("script", script_path), ("wrapper", wrapper_path)):
            binding[stem + "_path"] = str(path)
            binding[stem + "_sha256"] = expected_script if stem == "script" else paired.sha256_file(path)
        snapshot = {filename: paired.sha256_file(arm / filename) for filename in paired.SIDE_CARS}
        binding.update(sidecar_sha256_before=snapshot, sidecar_sha256_after=snapshot)
        write(arm / "PROBE_WEIGHT_BINDING.json", binding)
        return raw

    def test_compare_acquisition_uses_same_resolver_and_preserves_cross_draw_floor(self):
        left = self.make_probe("left", [False] * 6, self.original)
        right = self.make_probe("right", [True] * 6, self.archive)
        original_binding = (left.parent / "PROBE_WEIGHT_BINDING.json").read_bytes()
        (self.original / self.script).write_bytes(b"user-dirty process code")
        blocked = paired.compare_acquisition(left, right, relationship="cross_training_draw")
        self.assertEqual(blocked["status"], paired.BLOCKED)
        self.assertNotIn("paired", blocked)
        result = paired.compare_acquisition(left, right, relationship="cross_training_draw", source_archive=self.load())
        self.assertEqual(result["status"], "COMPARABLE", result)
        self.assertEqual((result["paired"]["net"], result["paired"]["nd"], result["paired"]["exact_p"]), (6, 6, .03125))
        self.assertFalse(result["paired"]["cross_training_draw_floor_cleared"])
        self.assertEqual(result["reference_resolved_source_files"]["script"]["resolved_path"], str(self.archive / self.script))
        self.assertEqual((left.parent / "PROBE_WEIGHT_BINDING.json").read_bytes(), original_binding)
        left.write_bytes(left.read_bytes() + b" ")
        blocked = paired.compare_acquisition(left, right, relationship="cross_training_draw", source_archive=self.load())
        self.assertEqual(blocked["status"], paired.BLOCKED)
        self.assertNotIn("paired", blocked)


if __name__ == "__main__":
    unittest.main()
