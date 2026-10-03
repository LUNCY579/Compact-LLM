"""CPU-only output-custody tests; fake resource guards and fake model generators.

Run from repository root:
    python -m unittest discover -s docs/model-quality/audit-2026-09-17 -p test_supervised_probe.py -v

No native model, CUDA qualification, or model-quality claim is exercised here.
"""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


class SupervisedProbeCustodyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("supervised_probe.py").resolve()
        spec = importlib.util.spec_from_file_location("probe_custody_fixture", path)
        self.wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.wrapper)
        self.temporary = tempfile.TemporaryDirectory(prefix="rrp015-probe-custody-")
        self.root = Path(self.temporary.name).resolve()
        # Verify cleanup is restricted to the uniquely created fixture directory.
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.wrapper.ROOT = self.root
        self.resources = ModuleType("experiments.model_quality_gpu_v1.resources")

        @contextmanager
        def fixture_lease(path):
            yield

        self.resources.lease = fixture_lease
        self.resources.guard = lambda **kwargs: {"fixture_only": True}
        self.rows = [{"id": str(i), "text": "7", "terminated": True, "new_tokens": 1}
                     for i in range(150)]
        self.generator = lambda *args, **kwargs: self.rows
        self.package = ModuleType("experiments.architecture_benchmark_001")
        self.package.run = SimpleNamespace(_greedy_probe=self.generator)
        self.saved_modules = {}
        for name, module in ((self.resources.__name__, self.resources),
                             (self.package.__name__, self.package)):
            self.saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

    def tearDown(self):
        for name, previous in self.saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        sys.modules.pop("_rrp015_fixture_sibling", None)
        self.temporary.cleanup()

    def arguments(self, name, kind):
        return [name, "source", "cuda"] if kind == "retention" else ["--label", name]

    def output_script(self, name, kind="process"):
        result = self.wrapper.result_path(kind, self.arguments(name, kind))
        return ("from pathlib import Path\n"
                f"Path({str(result)!r}).write_text('{{\"ok\":true}}', encoding='utf-8')\n")

    def run_fixture(self, name, body, *, kind="process", failure=None, existing=False):
        out = self.root / name
        out.mkdir()
        script = out / "fixture.py"
        script.write_text(body, encoding="utf-8")
        args = SimpleNamespace(kind=kind)
        arguments = self.arguments(name, kind)
        target = self.wrapper.result_path(kind, arguments)
        target.parent.mkdir(parents=True, exist_ok=True)
        if existing:
            target.write_text('{"preserve":true}', encoding="utf-8")
        previous = target.read_bytes() if existing else None
        old_argv, old_path = sys.argv, sys.path[:]
        invocation = lambda: self.wrapper.execute_worker(
            args, arguments, script, self.wrapper.sha(script), out)
        if failure is None:
            self.assertEqual(invocation(), 0)
        else:
            with self.assertRaises(failure):
                invocation()
        terminal = json.loads((out / "probe_terminal.json").read_text(encoding="utf-8"))
        self.assertEqual(terminal["complete"], failure is None)
        self.assertEqual(terminal["status"], "COMPLETED" if failure is None else "FAILED")
        self.assertEqual(terminal["system_exit_code"] == 0, failure is None)
        self.assertIs(sys.argv, old_argv)
        self.assertEqual(sys.path, old_path)
        if existing:
            self.assertEqual(target.read_bytes(), previous)
        return out, target, terminal

    def test_success_and_sibling_import(self):
        sibling = self.root / "success"
        # run_fixture creates this directory, so the script creates its sibling.
        body = ("from pathlib import Path\n"
                "Path(__file__).with_name('_rrp015_fixture_sibling.py').write_text('VALUE = 7')\n"
                "import _rrp015_fixture_sibling\n"
                "assert _rrp015_fixture_sibling.VALUE == 7\n")
        self.run_fixture("success", body + self.output_script("success") + "raise SystemExit(0)\n")

    def test_unexpected_exception_is_failed_not_exit_zero(self):
        _, _, terminal = self.run_fixture("unexpected", "raise RuntimeError('fixture boom')\n",
                                          failure=RuntimeError)
        self.assertEqual(terminal["failure"]["type"], "RuntimeError")

    def test_nonzero_system_exit_is_preserved(self):
        _, _, terminal = self.run_fixture("nonzero", "raise SystemExit(7)\n", failure=SystemExit)
        self.assertEqual(terminal["system_exit_code"], 7)

    def test_zero_exit_without_complete_result_is_failure(self):
        self.run_fixture("missing", "pass\n", failure=RuntimeError)

    def test_guard_failure_has_terminal_and_never_runs_script(self):
        def refuse(**kwargs):
            raise RuntimeError("fixture guard refused")
        self.resources.guard = refuse
        _, _, terminal = self.run_fixture("guard", "raise AssertionError('must not run')\n",
                                          failure=RuntimeError)
        self.assertEqual(terminal["failure"]["phase"], "resource_guard")

    def test_lease_failure_has_terminal_and_never_runs_script(self):
        @contextmanager
        def refuse(path):
            raise FileExistsError("fixture lease already held")
            yield
        self.resources.lease = refuse
        _, _, terminal = self.run_fixture("lease", "raise AssertionError('must not run')\n",
                                          failure=FileExistsError)
        self.assertEqual(terminal["failure"]["phase"], "lease")

    def test_existing_retention_result_is_unchanged(self):
        self.run_fixture("existing", "raise AssertionError('must not run')\n",
                         kind="retention", failure=FileExistsError, existing=True)

    def test_complete_retention_predictions_preserve_generator_return(self):
        body = ("from experiments.architecture_benchmark_001 import run\n"
                "assert len(run._greedy_probe(None)) == 150\n")
        out, _, _ = self.run_fixture("all_rows", body + self.output_script("all_rows", "retention"),
                                      kind="retention")
        evidence = json.loads((out / "retention_predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["predictions"], self.rows)
        self.assertEqual(evidence["rows"], 150)
        self.assertIs(self.package.run._greedy_probe, self.generator)

    def test_atomic_create_only_refuses_write_collision(self):
        collision = self.root / "collision.json"
        collision.write_text("original", encoding="utf-8")
        original_write = Path.write_text
        with self.assertRaises(FileExistsError):
            with self.wrapper.preserve_legacy_outputs("process", collision, self.root):
                collision.write_text("replacement", encoding="utf-8")
        self.assertEqual(collision.read_text(encoding="utf-8"), "original")
        self.assertIs(Path.write_text, original_write)

    def test_traversal_label_is_refused(self):
        with self.assertRaises(ValueError):
            self.wrapper.result_path("retention", ["../unsafe", "source"])

    def test_script_drift_is_failed_after_output(self):
        body = self.output_script("drift") + "Path(__file__).write_text('# changed during fixture')\n"
        _, _, terminal = self.run_fixture("drift", body, failure=RuntimeError)
        self.assertFalse(terminal["script_unchanged"])

    def lease_fixture(self, *, owner=424242, claimed=424242,
                      status="TIMEOUT_OR_INTERRUPTED", child_exited=True):
        out = self.root / "recovery_attempt"
        out.mkdir()
        lock = self.root / "fixture-gpu.lock"
        # Deliberately noncanonical bytes: archival must preserve them exactly.
        raw = ('{ "pid": %d, "token": "fixture-token", "created": 1.0 }\n' % owner).encode()
        lock.write_bytes(raw)
        launch = {"schema": 1, "out": str(out), "supervisor_pid": os.getpid()}
        self.wrapper.write_new(out / "launch.json", launch)
        self.wrapper.write_new(out / "worker_claim.json", {
            "pid": claimed, "supervisor_pid": os.getpid(),
            "launch_sha256": self.wrapper.digest(launch),
        })
        self.wrapper.write_new(out / "supervisor.json", {
            "status": status, "returncode": 1,
            "cleanup": {"child_exited": child_exited},
        })
        return out, lock, raw

    def test_recover_matching_dead_timeout_preserves_exact_lock(self):
        out, lock, raw = self.lease_fixture()
        with mock.patch("psutil.pid_exists", return_value=False) as exists:
            result = self.wrapper.recover_own_timedout_lease(out, lock)
        self.assertTrue(result["removed"])
        self.assertFalse(lock.exists())
        self.assertEqual((out / "timedout_lease.raw.json").read_bytes(), raw)
        self.assertEqual(result["lock_sha256"], self.wrapper.hashlib.sha256(raw).hexdigest())
        self.assertEqual(exists.call_args_list, [mock.call(424242), mock.call(424242)])

    def test_recovery_refuses_live_matching_owner(self):
        out, lock, raw = self.lease_fixture()
        with mock.patch("psutil.pid_exists", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "PID still exists"):
                self.wrapper.recover_own_timedout_lease(out, lock)
        self.assertEqual(lock.read_bytes(), raw)
        self.assertFalse((out / "timedout_lease.raw.json").exists())

    def test_recovery_refuses_unrelated_owner(self):
        out, lock, raw = self.lease_fixture(owner=555555)
        with mock.patch("psutil.pid_exists", return_value=False) as exists:
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                self.wrapper.recover_own_timedout_lease(out, lock)
        exists.assert_not_called()
        self.assertEqual(lock.read_bytes(), raw)

    def test_recovery_refuses_unproven_child_termination(self):
        out, lock, raw = self.lease_fixture(child_exited=False)
        with self.assertRaisesRegex(RuntimeError, "did not establish child termination"):
            self.wrapper.recover_own_timedout_lease(out, lock)
        self.assertEqual(lock.read_bytes(), raw)

    def test_recovery_leaves_normal_exit_lease_untouched(self):
        out, lock, raw = self.lease_fixture(status="EXITED")
        result = self.wrapper.recover_own_timedout_lease(out, lock)
        self.assertEqual(result["status"], "NOT_APPLICABLE")
        self.assertFalse(result["removed"])
        self.assertEqual(lock.read_bytes(), raw)

    def test_recovery_refuses_changed_bytes_after_archive(self):
        out, lock, raw = self.lease_fixture()
        original_write = self.wrapper.write_new
        changed = b'{"pid":555555,"token":"another-owner"}\n'

        def change_after_intent(path, value):
            original_write(path, value)
            if path.name == "lease_recovery_intent.json":
                lock.write_bytes(changed)

        with mock.patch("psutil.pid_exists", return_value=False):
            with mock.patch.object(self.wrapper, "write_new", side_effect=change_after_intent):
                with self.assertRaisesRegex(RuntimeError, "changed after archival"):
                    self.wrapper.recover_own_timedout_lease(out, lock)
        self.assertEqual(lock.read_bytes(), changed)
        self.assertEqual((out / "timedout_lease.raw.json").read_bytes(), raw)

    def test_recovery_refuses_pid_reappearing_after_archive(self):
        out, lock, raw = self.lease_fixture()
        with mock.patch("psutil.pid_exists", side_effect=[False, True]):
            with self.assertRaisesRegex(RuntimeError, "PID appeared"):
                self.wrapper.recover_own_timedout_lease(out, lock)
        self.assertEqual(lock.read_bytes(), raw)
        self.assertEqual((out / "timedout_lease.raw.json").read_bytes(), raw)

    def test_predispatch_live_lease_refuses_before_any_worker(self):
        _, lock, raw = self.lease_fixture()
        script = self.root / self.wrapper.SCRIPTS["process"]
        script.parent.mkdir(parents=True)
        script.write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
        attempt = self.root / "never_launched"
        argv = ["supervised_probe.py", "--kind", "process", "--out", str(attempt),
                "--timeout", "900", "--", "--label", "predispatch"]
        with mock.patch.object(self.wrapper, "gpu_lease_path", return_value=lock):
            with mock.patch.object(self.wrapper, "bounded_child") as child:
                with mock.patch.object(sys, "argv", argv):
                    with self.assertRaisesRegex(FileExistsError, "no worker dispatched"):
                        self.wrapper.main()
        child.assert_not_called()
        self.assertFalse(attempt.exists())
        self.assertEqual(lock.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
