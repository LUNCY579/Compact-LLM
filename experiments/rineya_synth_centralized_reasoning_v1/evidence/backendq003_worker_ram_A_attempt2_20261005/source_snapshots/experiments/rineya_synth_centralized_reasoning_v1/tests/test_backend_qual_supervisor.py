"""No actual worker, model, Torch, GPU query or native process is launched."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
import copy
import hashlib
import inspect
from types import SimpleNamespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

STAGE = Path(__file__).resolve().parents[1]
PACKAGE = STAGE.parents[1] if STAGE.name == "backend_qual_work" else STAGE
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(STAGE))
import native_supervisor as shared
import backend_qual_supervisor as qs


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def snapshot(now, **changes):
    value = {"sample_started_monotonic": now, "sample_completed_monotonic": now,
             "host_available_bytes": 4 * shared.GIB, "disk_free_bytes": 8 * shared.GIB,
             "worker_tree_rss_bytes": 0, "owned_process_identities": [],
             "GPU_index": 0, "gpu_uuid": "GPU-MOCK-0", "gpu_total_bytes": 8 * shared.GIB,
             "gpu_free_bytes": 6 * shared.GIB, "gpu_sample_started_monotonic": now,
             "gpu_sample_completed_monotonic": now, "gpu_observation_id": str(now)}
    value.update(changes)
    return value


class Worker:
    def __init__(self, argv, output):
        self.argv, self.output = argv, output
        self.identity = {"pid": 987654321, "create_time": 1.0, "process_handle": 123, "job_handle": 456,
                         "assigned_before_resume": True, "job_kill_on_close": True, "breakaway_allowed": False}
        self.terminated = self.closed = False
        self.resumed = False
        self.events = ["assigned_suspended"]
        self.on_resume = None
        self.monitors = []
        self.returncode = 0
        self.poll_count = 0
        self.live_descendants = False
        (output / "partial.mock.json").write_text('{"mock":true}', encoding="utf-8")

    def is_alive(self):
        return True

    def poll(self):
        self.poll_count += 1
        if self.closed:
            return 137 if self.terminated else self.returncode
        return None if self.poll_count == 1 else self.returncode

    def process_identities(self):
        return [{"pid": 999, "create_time": 1.0}] if self.live_descendants else []

    def terminate(self):
        self.terminated = True

    def resume(self):
        if not self.monitors or not self.monitors[-1].started:
            raise AssertionError("external monitor must start before resume")
        if self.on_resume:
            self.on_resume()
        self.events.append("resume")
        self.resumed = True

    def close(self):
        self.closed = True


class Monitor:
    def __init__(self, worker, output, deadline, admitted):
        self.worker = worker
        self.deadline = deadline
        self.reason = None
        self.admitted = admitted
        self.started = False
        self.observations = []

    def start(self):
        self.sample()
        self.worker.events.append("monitor_started")
        self.started = True
        return self

    def sample(self):
        self.cpu = qs._cpu_snapshot(self.worker.output, self.worker)
        self.observations.append(self.cpu)

    def check(self):
        if self.reason:
            return self.reason
        if qs.time.monotonic() >= self.deadline:
            return "whole_scope_deadline"
        if not shared._fresh_cpu(self.cpu, qs.time.monotonic()):
            return "stale_external_CPU_watchdog"
        return None

    def pre_resume_check(self):
        self.worker.events.append("monitor_pre_resume_check")
        return self.check() or qs._admission_failure({**self.admitted, **self.cpu}, qs.time.monotonic())

    def _abort(self, reason):
        self.reason = reason
        self.worker.terminate()

    def close(self):
        return [{"mock_external_watchdog": True}, *self.observations]


class InferenceQualificationSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = Path(self.temporary.name) / "package"
        self.package.mkdir()
        self.clock = Clock()
        self.config = {
            "authorization": {"scope": "backend_crossprocess_native_qualification_only", "production_training": False,
                              "protected_bank_access": False, "paid_services": False},
            "resources": dict(qs.BOUNDS), "bindings": {"source": {"fake": "a" * 64}},
            "config_path": str(self.package / "configs" / "scope.json"), "config_sha256": "f" * 64,
            "worker_path": str(self.package / "backend_qual_worker.py"),
        }
        self.workers, self.samples, self.monitors = [], [], []
        self.sample_changes = {}
        self.worker_changes = {}
        self.monitor_reason = None
        self.construction_delay = 0
        self.unjoined_threads = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (
            ("PACKAGE", self.package),
            ("_pid_alive", lambda identity: False), ("_preflight", lambda path: copy.deepcopy(self.config)),
            ("resource_snapshot", self.observe), ("_cpu_snapshot", self.cpu_observe),
            ("_WindowsOwnedWorker", self.launch), ("_Monitor", self.monitor),
            ("_gpu_pending_threads", lambda: []), ("_shutdown_gpu_probe", lambda: None),
        ):
            self.stack.enter_context(mock.patch.object(qs, name, value))
        # Imported helpers keep native_supervisor globals; isolate both namespaces.
        self.stack.enter_context(mock.patch.object(shared, "PACKAGE", self.package))
        self.stack.enter_context(mock.patch.object(shared, "_pid_identity", lambda pid: {"pid": pid, "create_time": 1.0}))
        self.stack.enter_context(mock.patch.object(shared, "_pid_alive", lambda identity: False))
        self.stack.enter_context(mock.patch.object(shared, "_LeaseMutex", lambda: nullcontext()))
        self.stack.enter_context(mock.patch.object(shared.subprocess, "Popen", side_effect=AssertionError("actual child forbidden")))
        self.stack.enter_context(mock.patch.object(shared.ctypes, "WinDLL", side_effect=AssertionError("native DLL forbidden"), create=True))
        self.stack.enter_context(mock.patch.object(qs.time, "monotonic", self.clock.monotonic))
        self.stack.enter_context(mock.patch.object(qs.time, "sleep", self.clock.sleep))

    def observe(self, output, worker=None):
        value = snapshot(self.clock.now, **self.sample_changes)
        self.samples.append(value)
        return value

    def cpu_observe(self, output, worker=None):
        return snapshot(self.clock.now, **self.sample_changes)

    def launch(self, argv, output, *, config_sha256, deadline_monotonic):
        self.assertEqual(2, len(self.samples), "never launch before both independent admissions")
        self.assertEqual(self.config["config_sha256"], config_sha256)
        self.assertLess(self.clock.now, deadline_monotonic)
        self.clock.sleep(self.construction_delay)
        worker = Worker(argv, output)
        worker.monitors = self.monitors
        for key, value in self.worker_changes.items():
            setattr(worker, key, value)
        self.workers.append(worker)
        return worker

    def monitor(self, *args):
        monitor = Monitor(*args)
        monitor.reason = self.monitor_reason
        monitor.unjoined_threads = self.unjoined_threads
        self.monitors.append(monitor)
        return monitor

    def run_scope(self, name="attempt", admission_only=False):
        output = self.package / "artifacts" / name
        result = qs.supervise_backend_qualification(self.config["config_path"], output, admission_only=admission_only)
        self.assertEqual(result, json.loads((output / "SUPERVISOR_RECEIPT.json").read_text(encoding="utf-8")))
        return result

    def test_import_does_not_import_model_frameworks(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_admission_only_two_distinct_observations_no_worker(self):
        result = self.run_scope(admission_only=True)
        self.assertEqual("ADMISSION_PASSED_NO_WORKER", result["status"])
        self.assertEqual(2, len(result["admission_observations"]))
        self.assertEqual(2.0, result["elapsed_seconds"])
        self.assertFalse(result["worker_attempted"])
        self.assertFalse(self.workers)

    def test_exact_scope_only(self):
        for field, value in (("scope", "production_training"), ("production_training", True),
                             ("protected_bank_access", True), ("paid_services", True)):
            with self.subTest(field=field):
                original = copy.deepcopy(self.config)
                self.config["authorization"][field] = value
                result = self.run_scope(field)
                self.assertEqual("ADMISSION_DENIED", result["status"])
                self.assertFalse(result["worker_attempted"])
                self.assertFalse(self.workers)
                self.config = original

    def test_caller_cannot_change_any_resource_bound(self):
        for name, value in qs.BOUNDS.items():
            with self.subTest(bound=name):
                self.config["resources"][name] = value / 2
                result = self.run_scope(name)
                self.assertIn("fixed resource bound changed", result["failure"])
                self.assertFalse(result["worker_attempted"])
                self.config["resources"][name] = value
        self.assertFalse(self.samples)

    def test_bad_source_preflight_records_terminal_before_any_probe_or_worker(self):
        with mock.patch.object(qs, "_preflight", side_effect=ValueError("source not clean committed")):
            result = self.run_scope()
        self.assertIn("source not clean committed", result["failure"])
        self.assertFalse(self.samples)
        self.assertFalse(self.workers)
        self.assertIn("backend_qual_worker.py", result["attempted_worker_argv"][3])

    def test_admission_floors_cannot_be_bypassed(self):
        for key, value in (("host_available_bytes", int(3.3125 * shared.GIB) - 1),
                           ("gpu_free_bytes", 4 * shared.GIB - 1), ("disk_free_bytes", 4 * shared.GIB - 1)):
            with self.subTest(key=key):
                self.sample_changes = {key: value}
                result = self.run_scope(key)
                self.assertIn(key, result["failure"])
                self.assertFalse(result["worker_attempted"])
        self.assertFalse(self.workers)

    def test_stale_and_future_gpu_observations_denied(self):
        for delta in (-2.001, 0.001):
            self.sample_changes = {"gpu_sample_started_monotonic": self.clock.now + delta}
            result = self.run_scope("stale" + str(delta).replace(".", "x"))
            self.assertIn("stale_or_invalid", result["failure"])
        self.assertFalse(self.workers)

    def test_nonfinite_or_untyped_memory_is_not_admission_evidence(self):
        for index, (key, value) in enumerate((("host_available_bytes", float("nan")),
                                             ("gpu_free_bytes", float("inf")),
                                             ("worker_tree_rss_bytes", True),
                                             ("gpu_total_bytes", "unmeasured"))):
            self.sample_changes = {key: value}
            result = self.run_scope("bad_memory_" + str(index))
            self.assertIn("stale_or_invalid", result["failure"])
            if isinstance(value, float):
                self.assertIn("invalid_nonfinite_float", result["admission_observations"][0][key])
        self.assertFalse(self.workers)

    def test_reused_gpu_observation_is_not_two_current_samples(self):
        self.sample_changes = {"gpu_observation_id": "cached"}
        result = self.run_scope()
        self.assertIn("two distinct current observations", result["failure"])
        self.assertFalse(self.workers)

    def test_admission_deadline_no_automatic_poll_retry(self):
        original = self.observe
        def slow(*args):
            self.clock.now += 31
            return original(*args)
        with mock.patch.object(qs, "resource_snapshot", slow):
            result = self.run_scope()
        self.assertIn("admission observation deadline", result["failure"])
        self.assertEqual(1, len(result["admission_observations"]))
        self.assertFalse(self.workers)

    def test_binding_changed_on_revalidation_prevents_admission_and_dispatch(self):
        calls = 0
        def preflight(path):
            nonlocal calls
            calls += 1
            result = copy.deepcopy(self.config)
            if calls == 2:
                result["bindings"]["source"]["fake"] = "b" * 64
            return result
        with mock.patch.object(qs, "_preflight", preflight):
            result = self.run_scope()
        self.assertIn("binding changed", result["failure"])
        self.assertFalse(self.samples)
        self.assertFalse(self.workers)

    def test_success_records_returnstatus_identity_pins_and_partial_outputs(self):
        result = self.run_scope()
        self.assertEqual("SUPERVISED_WORKER_EXITED_ZERO", result["status"])
        self.assertTrue(result["worker_launched"])
        self.assertTrue(result["worker_resumed"])
        self.assertEqual(0, result["returncode"])
        self.assertTrue(result["job_closed_kill_on_close"])
        self.assertTrue(result["owned_handle_alive_verified_after_launch"])
        self.assertEqual(self.config["bindings"], result["preflight"]["bindings"])
        self.assertIn("partial.mock.json", [row["path"] for row in result["partial_outputs"]])
        self.assertFalse(result["mechanical_qualification_established"])
        self.assertTrue((Path(result["lease_path"]) / "worker.json").exists())
        self.assertTrue((Path(result["lease_path"]) / "resume.json").exists())

    def test_slow_suspended_construction_is_followed_by_genuine_fresh_monitor_before_resume(self):
        self.construction_delay = 0.35
        result = self.run_scope()
        self.assertEqual("SUPERVISED_WORKER_EXITED_ZERO", result["status"])
        self.assertTrue(self.workers[0].resumed)
        self.assertTrue(self.monitors[0].started)
        self.assertGreaterEqual(self.monitors[0].observations[0]["sample_started_monotonic"]
                                - result["dispatch_observation"]["sample_started_monotonic"], 0.35 - 1e-10)
        self.assertEqual(700.0, result["deadline_monotonic"])
        self.assertEqual(0.2, result["cadence"]["CPU_freshness_seconds"])

    def test_blocked_slow_resume_keeps_external_observations_current_without_resetting_deadline(self):
        def delayed_resume():
            self.assertTrue(self.monitors[0].started)
            self.clock.sleep(0.35)
            self.monitors[0].sample()  # independent external sampler's current observation
        self.worker_changes = {"on_resume": delayed_resume}
        result = self.run_scope()
        self.assertEqual("SUPERVISED_WORKER_EXITED_ZERO", result["status"])
        self.assertEqual(2, len(self.monitors[0].observations))
        self.assertEqual(700.0, result["deadline_monotonic"])
        self.assertEqual(0.1, result["cadence"]["CPU_watchdog_seconds"])
        self.assertEqual(0.2, result["cadence"]["CPU_freshness_seconds"])

    def test_startup_monitor_failure_never_resumes_suspended_worker(self):
        self.monitor_reason = "startup_external_CPU_watchdog_not_ready"
        result = self.run_scope()
        self.assertEqual("OWNED_WORKER_ABORTED", result["status"])
        self.assertFalse(result["worker_resumed"])
        self.assertFalse(self.workers[0].resumed)
        self.assertTrue(self.workers[0].terminated)

    def test_unjoined_sampler_does_not_claim_terminal_success_or_frozen_inventory(self):
        self.unjoined_threads = [{"name": "blocked_GPU_probe", "ident": 123}]
        result = self.run_scope()
        self.assertEqual("SUPERVISOR_SHUTDOWN_UNVERIFIED", result["status"])
        self.assertFalse(result["sampler_threads_terminal_verified"])
        self.assertFalse(result["partial_outputs_frozen"])
        self.assertTrue(all("sha256" not in row for row in result["partial_outputs"]))
        self.assertTrue(self.workers[0].closed)

    def test_live_supervisor_with_unverified_samplers_cannot_enroll_another_scope(self):
        self.unjoined_threads = [{"name": "blocked_GPU_probe", "ident": 123}]
        self.run_scope("first")
        self.config["config_sha256"] = "e" * 64
        with mock.patch.object(shared, "_pid_alive", side_effect=lambda identity: identity["pid"] == shared.os.getpid()):
            second = self.run_scope("second")
        self.assertIn("live supervisor has unverified sampler shutdown", second["failure"])
        self.assertEqual(1, len(self.workers))

    def test_successful_worker_with_shutdown_beyond_original_deadline_is_not_scope_success(self):
        with mock.patch.object(qs, "_shutdown_gpu_probe", side_effect=lambda: self.clock.sleep(599)):
            result = self.run_scope()
        self.assertEqual("WHOLE_SCOPE_DEADLINE_EXCEEDED", result["status"])
        self.assertEqual("whole_scope_deadline_during_shutdown", result["failure"])
        self.assertEqual(0, result["returncode"])
        self.assertEqual(700.0, result["deadline_monotonic"])

    def test_inventory_beyond_deadline_cannot_claim_scope_success(self):
        inventory = qs._inventory
        def slow_inventory(path):
            self.clock.sleep(599)
            return inventory(path)
        with mock.patch.object(qs, "_inventory", side_effect=slow_inventory):
            result = self.run_scope()
        self.assertEqual("WHOLE_SCOPE_DEADLINE_EXCEEDED", result["status"])
        self.assertEqual("whole_scope_deadline_during_inventory", result["failure"])
        self.assertEqual(0, result["returncode"])
        self.assertGreaterEqual(result["elapsed_seconds"], 600)

    def test_watchdog_failure_terminates_owned_worker_preserves_returncode(self):
        self.monitor_reason = "runtime_system_reserve"
        result = self.run_scope()
        self.assertEqual("OWNED_WORKER_ABORTED", result["status"])
        self.assertEqual("runtime_system_reserve", result["failure"])
        self.assertEqual(137, result["returncode"])
        self.assertTrue(self.workers[0].terminated)
        self.assertTrue(self.workers[0].closed)

    def test_containment_launch_failure_preserves_attempt_pid_and_returnstatus(self):
        failure = qs._LaunchFailure(OSError("mock assign failure"), {
            "worker_pid": 4321, "returncode": 137, "assigned_before_resume": False,
            "owned_process_handle_terminal_verified": True})
        with mock.patch.object(qs, "_WindowsOwnedWorker", side_effect=failure):
            result = self.run_scope()
        self.assertEqual("OWNED_WORKER_LAUNCH_FAILED", result["status"])
        self.assertEqual(137, result["returncode"])
        self.assertEqual(4321, result["launch_failure"]["worker_pid"])
        self.assertTrue(result["worker_attempted"])
        self.assertFalse(result["worker_launched"])

    def test_root_exit_with_owned_descendants_is_abort_not_success(self):
        self.worker_changes = {"live_descendants": True}
        result = self.run_scope()
        self.assertEqual("OWNED_WORKER_ABORTED", result["status"])
        self.assertIn("live_owned_descendants", result["failure"])
        self.assertTrue(self.workers[0].terminated)

    def test_previous_live_worker_is_verified_before_deny_and_never_killed(self):
        prior = self.run_scope()
        with mock.patch.object(shared, "_pid_alive", return_value=True) as alive:
            result = self.run_scope("second")
        self.assertIn("currently alive", result["failure"])
        self.assertEqual(1, len(self.workers))
        self.assertFalse(self.workers[0].terminated)
        self.assertEqual(prior["worker_identity"], alive.call_args.args[0])

    def test_prior_dead_lease_preserved_and_new_lease_unique(self):
        first = self.run_scope("first", admission_only=True)
        second = self.run_scope("second", admission_only=True)
        self.assertNotEqual(first["lease_path"], second["lease_path"])
        self.assertTrue(Path(first["lease_path"]).exists())
        self.assertEqual(1, len(list(Path(first["lease_path"]).glob("death_observation_*.json"))))

    def test_unchanged_failed_worker_cannot_retry(self):
        self.monitor_reason = "whole_scope_deadline"
        self.run_scope("first")
        second = self.run_scope("second")
        self.assertIn("unchanged failed worker", second["failure"])
        self.assertEqual(1, len(self.workers))

    def test_output_write_once_and_direct_package_child(self):
        self.run_scope(admission_only=True)
        with self.assertRaises(FileExistsError):
            self.run_scope(admission_only=True)
        for path in (self.package / "outside", self.package / "artifacts" / "nested" / "leaf",
                     self.package / "artifacts" / "_reserved"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                qs.supervise_backend_qualification(self.config["config_path"], path, admission_only=True)


    def test_wrong_worker_denied_before_lease_probe_or_attempt(self):
        for index, worker in enumerate(("native_worker.py", "study_engine.py", "other/backend_qual_worker.py")):
            with self.subTest(worker=worker):
                self.config["worker_path"] = str(self.package / worker)
                with mock.patch.object(qs, "_acquire_lease", wraps=qs._acquire_lease) as lease:
                    result = self.run_scope("wrong_worker_" + str(index))
                self.assertIn("pinned package/backend_qual_worker.py", result["failure"])
                self.assertFalse(result["worker_attempted"])
                lease.assert_not_called()
        self.assertFalse(self.samples)
        self.assertFalse(self.workers)

    def test_closed_authority_requires_literal_false_and_missing_is_denied(self):
        original = copy.deepcopy(self.config)
        index = 0
        for name in ("production_training", "protected_bank_access", "paid_services"):
            for value in (True, 0, None, "false", "MISSING"):
                with self.subTest(name=name, value=value):
                    self.config = copy.deepcopy(original)
                    if value == "MISSING": self.config["authorization"].pop(name)
                    else: self.config["authorization"][name] = value
                    with mock.patch.object(qs, "_acquire_lease", wraps=qs._acquire_lease) as lease:
                        result = self.run_scope("closed_authority_" + str(index))
                    index += 1
                    self.assertEqual("ADMISSION_DENIED", result["status"])
                    self.assertIn("backend qualification only", result["failure"])
                    self.assertFalse(result["worker_attempted"])
                    lease.assert_not_called()
        self.assertFalse(self.samples)
        self.assertFalse(self.workers)

    def test_fixed_resources_cadence_deadline_and_order_match_qualified_limits(self):
        result = self.run_scope()
        self.assertEqual({"preworker_available_GiB":3.3125, "system_available_reserve_GiB":1.25,
                          "worker_process_tree_RSS_GiB":2.0, "GPU_free_at_admission_GiB":4.0,
                          "GPU_free_during_work_GiB":0.5, "disk_free_GiB":4.0,
                          "external_watchdog_milliseconds":100, "whole_scope_seconds":600}, result["bounds"])
        self.assertEqual({"CPU_watchdog_seconds":.1, "CPU_freshness_seconds":.2,
                          "GPU_query_interval_seconds":.5, "GPU_query_timeout_seconds":2.0,
                          "GPU_freshness_seconds":2.0, "admission_interval_seconds":2.0,
                          "admission_deadline_seconds":30.0, "admission_samples":2}, result["cadence"])
        self.assertEqual(700.0, result["deadline_monotonic"])
        self.assertEqual(["assigned_suspended", "monitor_started", "monitor_pre_resume_check", "resume"], self.workers[0].events)
        self.assertTrue(result["worker_identity"]["assigned_before_resume"])
        self.assertFalse(result["production_training_authorized"])
        self.assertFalse(result["protected_bank_access"])
        self.assertFalse(result["mechanical_qualification_established"])

    def test_cleanup_exception_prevents_terminal_success(self):
        with mock.patch.object(qs, "_shutdown_gpu_probe", side_effect=RuntimeError("mock cleanup failed")):
            result = self.run_scope()
        self.assertEqual("SUPERVISOR_SHUTDOWN_UNVERIFIED", result["status"])
        self.assertFalse(result["gpu_telemetry_cleanup_verified"])
        self.assertTrue(self.workers[0].closed)
        self.assertEqual(0, result["returncode"])
        terminal = json.loads((Path(result["lease_path"]) / "terminal.json").read_text())
        self.assertEqual(result, terminal)


class StaticAdaptationTests(unittest.TestCase):
    def test_function_is_exact_qualified_dispatch_with_only_declared_rebindings(self):
        source_path = PACKAGE / "native_supervisor.py"
        self.assertEqual(qs.ADAPTED_SUPERVISOR_SOURCE_SHA256, hashlib.sha256(source_path.read_bytes()).hexdigest())
        expected = inspect.getsource(shared.supervise)
        for old,new in (("def supervise(","def supervise_backend_qualification("),
                        ("native_worker.py","backend_qual_worker.py"),
                        ("mechanical_native_qualification_only","backend_crossprocess_native_qualification_only"),
                        ("mechanical qualification only","backend qualification only")):
            expected = expected.replace(old,new)
        self.assertEqual(expected, inspect.getsource(qs.supervise_backend_qualification))

    def test_shared_qualified_containment_telemetry_cleanup_helpers_are_reused(self):
        for name in ("BOUNDS", "_WindowsOwnedWorker", "_Monitor", "_acquire_lease", "_bounds_valid",
                     "resource_snapshot", "_cpu_snapshot", "_admission_failure", "_shutdown_gpu_probe",
                     "_gpu_pending_threads", "_inventory", "_new_output"):
            with self.subTest(helper=name): self.assertIs(getattr(shared,name),getattr(qs,name))

    def test_preflight_dispatches_to_backend_contract_only_without_real_contract_calls(self):
        expected = {"scope":"MOCK_ONLY"}
        infer = SimpleNamespace(preflight=mock.Mock(return_value=expected))
        native = SimpleNamespace(preflight=mock.Mock(side_effect=AssertionError("wrong authority")))
        with mock.patch.dict(sys.modules,{"backend_qual_contract":infer,"native_contract":native}):
            self.assertIs(expected,qs._preflight("mock.json"))
        infer.preflight.assert_called_once_with("mock.json")
        native.preflight.assert_not_called()
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("transformers",sys.modules)


if __name__ == "__main__":
    unittest.main()
