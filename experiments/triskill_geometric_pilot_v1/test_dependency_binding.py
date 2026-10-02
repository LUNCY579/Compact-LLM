"""CPU/stdlib regression tests; all rendering fixtures are harmless surrogates."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import dependency_binding as binding


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.pilot = self.repo / binding.PILOT_RELATIVE
        self.pilot.mkdir(parents=True)
        self.paths = {role: self.pilot / f"{role}.py" for role in ("worker", "geometry", "benchmark")}
        self.paths.update(binding=self.pilot / "dependency_binding.py", renderer=self.repo / binding.RENDERER_RELATIVE,
                          renderer_fixture=self.repo / "dev-only" / "prompts.jsonl")
        for role, path in self.paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if role not in ("renderer", "renderer_fixture"):
                path.write_text(f"ROLE = {role!r}\n", encoding="utf-8")
        # Use the existing renderer's exact implementation with only its fixed
        # DEV-path literal relocated in this private test fixture. Production
        # validation has no root or fixture override option.
        original = Path(binding.__file__).parent.parent / "fresh_dual_skill_instrument_v1" / "score_arms.py"
        text = original.read_text(encoding="utf-8")
        old = 'RECORDED = Path(r"' + binding.RENDERER_FIXTURE.as_posix() + '")'
        new = 'RECORDED = Path(r"' + self.paths["renderer_fixture"].as_posix() + '")'
        self.assertEqual(text.count(old), 1)
        self.renderer_source = text.replace(old, new)
        self.paths["renderer"].write_text(self.renderer_source, encoding="utf-8")
        prompt = ("<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nuser<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.prompt = prompt
        self.paths["renderer_fixture"].write_text("\n".join(json.dumps({"text": prompt}) for _ in range(100)) + "\n", encoding="utf-8")
        self.manifest = self._manifest()

    def _manifest(self):
        return {"schema": binding.SCHEMA, "roles": {
            role: {"kind": binding.ROLE_KINDS[role], "path": binding._manifest_path(role, path, self.repo),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in self.paths.items()
        }}

    def _validate(self, manifest=None):
        return binding._validate_manifest(manifest or self.manifest, self.repo, self.paths)

    @contextlib.contextmanager
    def _cached(self, alias, module):
        prior = sys.modules.get(alias)
        sys.modules[alias] = module
        try:
            yield
        finally:
            if prior is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = prior

    def test_all_required_roles_are_bound_and_kind_separated(self):
        bound = self._validate()
        self.assertEqual(set(bound.roles), set(binding.ROLE_KINDS))
        self.assertEqual(bound.roles["renderer_fixture"]["kind"], "fixture")
        self.assertTrue(all(bound.roles[role]["kind"] == "code" for role in binding.ROLE_KINDS if role != "renderer_fixture"))

    def test_exact_renderer_load_and_original_selftest_on_surrogate(self):
        bound = self._validate()
        renderer = bound.load_renderer()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            renderer.template_selftest()
        self.assertIn("100 recorded prompts reproduced byte-exactly", output.getvalue())
        self.assertEqual(renderer.render("sys", "user"), self.prompt)

    def test_pinned_fixture_snapshot_avoids_unbound_second_read(self):
        bound = self._validate()
        renderer = bound.load_renderer()
        # Verify_current still checks the real file; the self-test's actual
        # reader is the immutable snapshot, so Path.read_text is never called.
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("unbound fixture read")):
            with contextlib.redirect_stdout(io.StringIO()):
                renderer.template_selftest()

    def test_renderer_bytes_changed_before_validation(self):
        self.paths["renderer"].write_text(self.renderer_source + "\n# tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self._validate()

    def test_fixture_bytes_changed_before_validation(self):
        self.paths["renderer_fixture"].write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self._validate()

    def test_fixture_changed_after_binding_aborts_before_selftest(self):
        renderer = self._validate().load_renderer()
        self.paths["renderer_fixture"].write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            renderer.template_selftest()

    def test_fixture_missing_fails_closed(self):
        self.paths["renderer_fixture"].unlink()
        with self.assertRaisesRegex(ValueError, "missing/inaccessible"):
            self._validate()

    def test_local_executed_source_changed_fails_closed(self):
        for role in ("worker", "geometry", "benchmark", "binding"):
            with self.subTest(role=role):
                old = self.paths[role].read_bytes()
                self.paths[role].write_bytes(old + b"# changed\n")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    self._validate()
                self.paths[role].write_bytes(old)

    def test_manifest_cannot_redirect_fixture_or_renderer_or_traverse(self):
        for role, path in (("renderer_fixture", (self.repo / "other.jsonl").as_posix()),
                           ("renderer", "experiments/other/score_arms.py"),
                           ("geometry", "experiments/triskill_geometric_pilot_v1/../triskill_geometric_pilot_v1/geometry.py")):
            with self.subTest(role=role):
                bad = copy.deepcopy(self.manifest)
                bad["roles"][role]["path"] = path
                with self.assertRaisesRegex(ValueError, "role/path mismatch"):
                    self._validate(bad)

    def test_missing_extra_role_and_kind_confusion_fail_closed(self):
        variants = []
        missing = copy.deepcopy(self.manifest)
        missing["roles"].pop("renderer_fixture")
        variants.append(missing)
        extra = copy.deepcopy(self.manifest)
        extra["roles"]["extra_bank"] = extra["roles"]["renderer_fixture"]
        variants.append(extra)
        kind = copy.deepcopy(self.manifest)
        kind["roles"]["renderer_fixture"]["kind"] = "code"
        variants.append(kind)
        for bad in variants:
            with self.subTest(variant=bad):
                with self.assertRaises(ValueError):
                    self._validate(bad)

    def test_cached_geometry_wrong_origin_fails_closed(self):
        fake = types.ModuleType("geometry")
        fake.__file__ = str(self.repo / "substituted_geometry.py")
        with self._cached("geometry", fake):
            with self.assertRaisesRegex(ValueError, "cached local module path mismatch"):
                self._validate().load_local("geometry")

    def test_cached_benchmark_claiming_expected_path_is_unverified(self):
        fake = types.ModuleType("benchmark")
        fake.__file__ = str(self.paths["benchmark"])
        with self._cached("benchmark", fake):
            with self.assertRaisesRegex(ValueError, "unverified cached local module"):
                self._validate().load_local("benchmark")

    def test_cached_renderer_cannot_supply_executable_object(self):
        fake = types.ModuleType("score_arms")
        fake.__file__ = str(self.repo / "other_renderer.py")
        with self._cached("score_arms", fake):
            with self.assertRaisesRegex(ValueError, "cached local module path mismatch"):
                self._validate().load_renderer()

    def test_private_module_cache_is_not_reused(self):
        digest = self.manifest["roles"]["renderer"]["sha256"]
        name = f"_triskill_bound_renderer_{digest}"
        fake = types.ModuleType(name)
        fake.render = lambda *args: "substituted"
        with self._cached(name, fake):
            renderer = self._validate().load_renderer()
            self.assertEqual(renderer.render("sys", "user"), self.prompt)

    def test_executing_worker_requires_verified_byte_marker(self):
        fake = types.ModuleType("worker")
        fake.__file__ = str(self.paths["worker"])
        with self.assertRaisesRegex(ValueError, "unverified executing native worker"):
            self._validate().verify_worker_execution(fake)
        with self._cached("worker", fake):
            loaded = self._validate().load_worker()
        self._validate().verify_worker_execution(loaded)

    def test_same_path_poisoned_worker_is_bootstrap_only(self):
        import worker
        self.paths["worker"].write_text(
            "def _run_bound_screen(*args, **kwargs):\n"
            "    return {'fresh_verified_source': True, 'marker': _triskill_source_binding}\n",
            encoding="utf-8")
        self.manifest = self._manifest()
        freeze = self.repo / "freeze.json"
        freeze.write_text(json.dumps({"dependency_binding": self.manifest}), encoding="utf-8")
        fake = types.ModuleType("worker")
        fake.__file__ = str(self.paths["worker"])
        fake._run_bound_screen = mock.Mock(side_effect=AssertionError("poisoned cached implementation"))
        loader = types.SimpleNamespace(validate_dependencies=lambda manifest: self._validate(manifest))
        with self._cached("worker", fake):
            with mock.patch.object(worker, "_load_binding", return_value=loader):
                result = worker.run_screen(self.repo/"out.json", freeze,
                                           hashlib.sha256(freeze.read_bytes()).hexdigest(), "lease-token")
        self.assertTrue(result["fresh_verified_source"])
        self.assertEqual(result["marker"], (str(self.paths["worker"]), self.manifest["roles"]["worker"]["sha256"]))
        fake._run_bound_screen.assert_not_called()

    def test_unmarked_native_implementation_cannot_be_called_directly(self):
        import worker
        with self.assertRaisesRegex(RuntimeError, "STOP_UNBOUND_NATIVE_WORKER"):
            worker._run_bound_screen(self.repo/"out.json", self.repo/"freeze.json", "0"*64, "lease-token")

    def test_renderer_fixture_constant_override_is_rejected_even_when_repinned(self):
        original = self.paths["renderer_fixture"].as_posix()
        changed = self.renderer_source.replace('Path(r"' + original + '")', 'Path(r"' + original + '.other")')
        self.paths["renderer"].write_text(changed, encoding="utf-8")
        self.manifest = self._manifest()
        with self.assertRaisesRegex(ValueError, "fixture path override"):
            self._validate().load_renderer()

    def test_runtime_renderer_fixture_override_aborts_selftest(self):
        renderer = self._validate().load_renderer()
        renderer._module.RECORDED = self.repo / "other.jsonl"
        with self.assertRaisesRegex(ValueError, "fixture path override"):
            renderer.template_selftest()

    def test_reparse_point_path_substitution_is_rejected(self):
        target = self.paths["renderer_fixture"]
        original_lstat = Path.lstat

        def fake_lstat(path):
            if path == target:
                return types.SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)
            return original_lstat(path)

        with mock.patch.object(Path, "lstat", fake_lstat):
            with self.assertRaisesRegex(ValueError, "path substitution"):
                self._validate()

    def test_symlink_component_is_rejected(self):
        # lstat semantics, including ancestor checks, can be exercised without
        # requesting Windows symlink privileges or changing security settings.
        target = self.paths["renderer_fixture"].parent
        original_lstat = Path.lstat

        def fake_lstat(path):
            if path == target:
                return types.SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
            return original_lstat(path)

        with mock.patch.object(Path, "lstat", fake_lstat):
            with self.assertRaisesRegex(ValueError, "path substitution"):
                self._validate()

    def test_binding_cannot_load_worker_or_fixture_as_code(self):
        bound = self._validate()
        for role in ("worker", "binding", "renderer_fixture", "renderer"):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError, "only geometry/benchmark"):
                    bound.load_local(role)


if __name__ == "__main__":
    unittest.main()
