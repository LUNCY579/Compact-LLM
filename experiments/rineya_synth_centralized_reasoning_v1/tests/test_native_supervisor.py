"""No actual worker, model, Torch, GPU query or native process is launched."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import native_supervisor as ns


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def snapshot(now, **changes):
    value = {"sample_started_monotonic": now, "sample_completed_monotonic": now,
             "host_available_bytes": 4 * ns.GIB, "disk_free_bytes": 8 * ns.GIB,
             "worker_tree_rss_bytes": 0, "owned_process_identities": [],
             "GPU_index": 0, "gpu_uuid": "GPU-MOCK-0", "gpu_total_bytes": 8 * ns.GIB,
             "gpu_free_bytes": 6 * ns.GIB, "gpu_sample_started_monotonic": now,
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
        self.started = True
        return self

    def sample(self):
        self.cpu = ns._cpu_snapshot(self.worker.output, self.worker)
        self.observations.append(self.cpu)

    def check(self):
        if self.reason:
            return self.reason
        if ns.time.monotonic() >= self.deadline:
            return "whole_scope_deadline"
        if not ns._fresh_cpu(self.cpu, ns.time.monotonic()):
            return "stale_external_CPU_watchdog"
        return None

    def pre_resume_check(self):
        return self.check() or ns._admission_failure({**self.admitted, **self.cpu}, ns.time.monotonic())

    def _abort(self, reason):
        self.reason = reason
        self.worker.terminate()

    def close(self):
        return [{"mock_external_watchdog": True}, *self.observations]


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = Path(self.temporary.name) / "package"
        self.package.mkdir()
        self.clock = Clock()
        self.config = {
            "authorization": {"scope": "mechanical_native_qualification_only", "production_training": False,
                              "protected_bank_access": False, "paid_services": False},
            "resources": dict(ns.BOUNDS), "bindings": {"source": {"fake": "a" * 64}},
            "config_path": str(self.package / "configs" / "scope.json"), "config_sha256": "f" * 64,
            "worker_path": str(self.package / "native_worker.py"),
        }
        self.workers, self.samples, self.monitors = [], [], []
        self.sample_changes = {}
        self.worker_changes = {}
        self.monitor_reason = None
        self.construction_delay = 0
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (
            ("PACKAGE", self.package), ("_pid_identity", lambda pid: {"pid": pid, "create_time": 1.0}),
            ("_pid_alive", lambda identity: False), ("_preflight", lambda path: copy.deepcopy(self.config)),
            ("resource_snapshot", self.observe), ("_cpu_snapshot", self.cpu_observe),
            ("_WindowsOwnedWorker", self.launch), ("_Monitor", self.monitor),
        ):
            self.stack.enter_context(mock.patch.object(ns, name, value))
        self.stack.enter_context(mock.patch.object(ns.time, "monotonic", self.clock.monotonic))
        self.stack.enter_context(mock.patch.object(ns.time, "sleep", self.clock.sleep))

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
        self.monitors.append(monitor)
        return monitor

    def run_scope(self, name="attempt", admission_only=False):
        output = self.package / "artifacts" / name
        result = ns.supervise(self.config["config_path"], output, admission_only=admission_only)
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
        for name, value in ns.BOUNDS.items():
            with self.subTest(bound=name):
                self.config["resources"][name] = value / 2
                result = self.run_scope(name)
                self.assertIn("fixed resource bound changed", result["failure"])
                self.assertFalse(result["worker_attempted"])
                self.config["resources"][name] = value
        self.assertFalse(self.samples)

    def test_bad_source_preflight_records_terminal_before_any_probe_or_worker(self):
        with mock.patch.object(ns, "_preflight", side_effect=ValueError("source not clean committed")):
            result = self.run_scope()
        self.assertIn("source not clean committed", result["failure"])
        self.assertFalse(self.samples)
        self.assertFalse(self.workers)
        self.assertIn("native_worker.py", result["attempted_worker_argv"][3])

    def test_admission_floors_cannot_be_bypassed(self):
        for key, value in (("host_available_bytes", int(3.3125 * ns.GIB) - 1),
                           ("gpu_free_bytes", 4 * ns.GIB - 1), ("disk_free_bytes", 4 * ns.GIB - 1)):
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
        with mock.patch.object(ns, "resource_snapshot", slow):
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
        with mock.patch.object(ns, "_preflight", preflight):
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

    def test_watchdog_failure_terminates_owned_worker_preserves_returncode(self):
        self.monitor_reason = "runtime_system_reserve"
        result = self.run_scope()
        self.assertEqual("OWNED_WORKER_ABORTED", result["status"])
        self.assertEqual("runtime_system_reserve", result["failure"])
        self.assertEqual(137, result["returncode"])
        self.assertTrue(self.workers[0].terminated)
        self.assertTrue(self.workers[0].closed)

    def test_containment_launch_failure_preserves_attempt_pid_and_returnstatus(self):
        failure = ns._LaunchFailure(OSError("mock assign failure"), {
            "worker_pid": 4321, "returncode": 137, "assigned_before_resume": False,
            "owned_process_handle_terminal_verified": True})
        with mock.patch.object(ns, "_WindowsOwnedWorker", side_effect=failure):
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
        with mock.patch.object(ns, "_pid_alive", return_value=True) as alive:
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
                ns.supervise(self.config["config_path"], path, admission_only=True)


class MonitorBoundsTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.worker = mock.Mock()
        self.mon = ns._Monitor(self.worker, Path("unused"), 700, snapshot(100))
        self.mon.gpu_uuid = "GPU-MOCK-0"
        self.patch = mock.patch.object(ns.time, "monotonic", self.clock.monotonic)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_fixed_runtime_floors_and_deadline(self):
        cases = (("cpu", "host_available_bytes", int(1.25 * ns.GIB) - 1, "runtime_system_reserve"),
                 ("cpu", "worker_tree_rss_bytes", 2 * ns.GIB + 1, "runtime_owned_process_tree_RSS"),
                 ("gpu", "gpu_free_bytes", int(0.5 * ns.GIB) - 1, "runtime_GPU_free"),
                 ("gpu", "gpu_uuid", "GPU-OTHER", "runtime_GPU_identity_changed"))
        for target, key, value, expected in cases:
            with self.subTest(key=key):
                getattr(self.mon, target)[key] = value
                self.assertEqual(expected, self.mon.check())
                self.mon.cpu = self.mon.gpu = snapshot(100)
        self.mon.deadline = 100
        self.assertEqual("whole_scope_deadline", self.mon.check())

    def test_stale_gpu_and_cpu_independent_checks(self):
        self.mon.cpu = snapshot(99.799)
        self.assertEqual("stale_external_CPU_watchdog", self.mon.check())
        self.mon.cpu = snapshot(100)
        self.mon.gpu = snapshot(97.999)
        self.assertEqual("stale_external_GPU_observation", self.mon.check())

    def test_abort_terminates_only_owned_job_and_is_idempotent(self):
        self.mon._abort("memory")
        self.mon._abort("second")
        self.worker.terminate.assert_called_once_with()
        self.assertEqual("memory", self.mon.check())

    def test_failed_terminate_closes_owned_job(self):
        self.worker.terminate.side_effect = OSError("mock termination failure")
        self.mon._abort("memory")
        self.worker.job.close.assert_called_once_with()
        self.assertIn("termination_error", self.mon.observations[0])

    def test_cpu_watchdog_memory_failure_uses_no_gpu_query(self):
        def heartbeat(output, pid):
            self.assertEqual(1, self.mon.cpu["host_available_bytes"])
            self.assertTrue(self.mon.ready.is_set(), "publish current CPU before heartbeat IO")
        with mock.patch.object(ns, "_cpu_snapshot", return_value=snapshot(100, host_available_bytes=1)), \
                mock.patch.object(ns, "_gpu_snapshot") as gpu, mock.patch.object(ns, "_heartbeat", side_effect=heartbeat) as heartbeat_mock:
            self.mon._cpu_loop()
        gpu.assert_not_called()
        heartbeat_mock.assert_called_once_with(self.mon.output, ns.os.getpid())
        self.worker.terminate.assert_called_once_with()
        self.assertEqual("runtime_system_reserve", self.mon.reason)

    def test_gpu_timeout_abort_does_not_block_on_cpu_probe(self):
        self.mon.stop = mock.Mock()
        self.mon.stop.wait.return_value = False
        with mock.patch.object(ns, "_gpu_snapshot", side_effect=ns.subprocess.TimeoutExpired("nvidia-smi", 2)), \
                mock.patch.object(ns, "_cpu_snapshot") as cpu:
            self.mon._gpu_loop()
        cpu.assert_not_called()
        self.worker.terminate.assert_called_once_with()
        self.assertIn("GPU_probe_error:TimeoutExpired", self.mon.reason)

    def test_real_monitor_start_replaces_old_cpu_with_current_owned_tree_before_ready(self):
        self.clock.now += 0.35
        calls = []
        monitor = self.mon
        class MockThread:
            def __init__(self, *, target, daemon):
                self.target = target
            def start(self):
                calls.append(self.target.__name__)
                if self.target.__name__ == "_cpu_loop":
                    monitor.ready.set()  # mocked external sampler readiness
            def join(self, timeout):
                pass
        with mock.patch.object(ns, "_cpu_snapshot", return_value=snapshot(self.clock.now)) as cpu, \
                mock.patch.object(ns, "_gpu_snapshot") as gpu, mock.patch.object(ns.threading, "Thread", MockThread):
            self.mon.start()
        self.assertEqual(["_cpu_loop", "_gpu_loop"], calls)
        self.assertEqual(100.35, self.mon.cpu["sample_started_monotonic"])
        cpu.assert_called_once_with(self.mon.output, self.worker)
        gpu.assert_not_called()
        self.assertIsNone(self.mon.pre_resume_check())
        self.assertEqual(700, self.mon.deadline)
        self.assertEqual("worker_suspended_startup", self.mon.observations[0]["phase"])

    def test_real_monitor_readiness_failure_aborts_still_suspended_worker(self):
        self.mon.ready = mock.Mock()
        self.mon.ready.wait.return_value = False
        with mock.patch.object(ns, "_cpu_snapshot", return_value=snapshot(100)), \
                mock.patch.object(ns.threading, "Thread"):
            self.mon.start()
        self.mon.ready.wait.assert_called_once_with(0.2)
        self.worker.terminate.assert_called_once_with()
        self.assertEqual("startup_external_CPU_watchdog_not_ready", self.mon.reason)

    def test_monitor_refreshes_stale_gpu_while_worker_remains_suspended(self):
        self.clock.now += 2.01
        self.mon.ready = mock.Mock()
        self.mon.ready.wait.return_value = True
        with mock.patch.object(ns, "_gpu_snapshot", return_value=snapshot(self.clock.now)) as gpu, \
                mock.patch.object(ns, "_cpu_snapshot", return_value=snapshot(self.clock.now)), \
                mock.patch.object(ns.threading, "Thread"):
            self.mon.start()
        gpu.assert_called_once_with()
        self.assertIsNone(self.mon.pre_resume_check())
        self.assertEqual(700, self.mon.deadline)
        self.assertEqual("GPU", self.mon.observations[0]["kind"])
        self.assertEqual("worker_suspended_startup", self.mon.observations[0]["phase"])


class WindowsContainmentTests(unittest.TestCase):
    def test_job_limits_kill_on_close_and_no_child_breakaway(self):
        api = mock.Mock()
        api.CreateJobObjectW.return_value = 456
        flags = []
        def set_information(handle, kind, pointer, size):
            limits = ns.ctypes.cast(pointer, ns.ctypes.POINTER(ns._ExtendedLimits)).contents
            flags.append(limits.BasicLimitInformation.LimitFlags)
            self.assertEqual(9, kind)
            return True
        api.SetInformationJobObject.side_effect = set_information
        with mock.patch.object(ns.os, "name", "nt"), mock.patch.object(ns.ctypes, "WinDLL", return_value=api, create=True):
            job = ns._WindowsJob()
            job.assign(123)
            job.terminate()
            job.close()
        self.assertEqual([0x2000], flags)
        self.assertEqual(0, flags[0] & (0x800 | 0x1000))
        api.AssignProcessToJobObject.assert_called_once_with(456, 123)
        api.TerminateJobObject.assert_called_once_with(456, 137)
        api.CloseHandle.assert_called_once_with(456)

    def test_popen_suspended_no_window_assign_before_resume(self):
        events = []
        job = mock.Mock(handle=456)
        job.assign.side_effect = lambda handle: events.append("assign")
        process = mock.Mock(pid=43210, _handle=123)
        process.poll.return_value = None
        def resume(pid):
            self.assertTrue((Path(directory) / "supervisor_ownership.json").exists())
            self.assertTrue((Path(directory) / "supervisor_alive.json").exists())
            ownership = json.loads((Path(directory) / "supervisor_ownership.json").read_text(encoding="utf-8"))
            self.assertEqual({"worker_pid": 43210, "supervisor_pid": ns.os.getpid(), "config_sha256": "f" * 64}, ownership)
            events.append("resume")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(ns, "_WindowsJob", return_value=job), \
                mock.patch.object(ns, "_pid_identity", return_value={"pid": 43210, "create_time": 1.0}), \
                mock.patch.object(ns, "_resume_primary_thread", side_effect=resume), \
                mock.patch.object(ns.subprocess, "Popen", return_value=process) as popen:
            worker = ns._WindowsOwnedWorker(["python", "mock_worker"], Path(directory), config_sha256="f" * 64,
                                            deadline_monotonic=ns.time.monotonic() + 600)
            self.assertEqual(["assign"], events)
            self.assertFalse(worker.resumed)
            events.append("monitor_ready")
            worker.resume()
            self.assertEqual(["assign", "monitor_ready", "resume"], events)
            self.assertEqual(0x08000004, popen.call_args.kwargs["creationflags"])
            self.assertFalse(popen.call_args.kwargs["shell"])
            self.assertEqual("1", popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"])
            self.assertTrue(worker.identity["assigned_before_resume"])
            worker.close()
            job.close.assert_called_once_with()

    def test_failed_assignment_never_resumes_and_kills_only_just_created_root(self):
        job = mock.Mock(handle=456)
        job.assign.side_effect = OSError("mock assignment failure")
        process = mock.Mock(pid=43210, _handle=123)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(ns, "_WindowsJob", return_value=job), \
                mock.patch.object(ns, "_pid_identity", return_value={"pid": 43210, "create_time": 1.0}), \
                mock.patch.object(ns, "_resume_primary_thread") as resume, \
                mock.patch.object(ns.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(ns._LaunchFailure, "assignment failure"):
                ns._WindowsOwnedWorker(["python", "mock_worker"], Path(directory), config_sha256="f" * 64,
                                       deadline_monotonic=ns.time.monotonic() + 600)
            resume.assert_not_called()
            process.kill.assert_called_once_with()
            job.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
