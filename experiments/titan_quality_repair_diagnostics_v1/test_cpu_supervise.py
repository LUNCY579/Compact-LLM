"""Small/mocked guardian fixtures; execute only inside the sibling CPU slot.

These fixtures never launch a worker, obtain a native permit, query a GPU,
read weights or touch another owner's handles. Temporary files remain in the
already-owned external worker output directory.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from . import cpu_supervise as guard


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(.01, seconds)


class FakeJob:
    def __init__(self, events, *, close_failure=False, accounting_failure=False):
        self.handle, self.events = 101, events
        self.close_failure, self.accounting_failure = close_failure, accounting_failure

    def terminate(self):
        self.events.append("terminate_owned_job")

    def accounting(self):
        self.events.append("job_accounting")
        if self.accounting_failure:
            raise OSError("fixture accounting unavailable")
        return {"active_processes": 0, "job_memory_limit_bytes": guard.MEMORY_CAP,
                "peak_job_private_commit_bytes": 100, "total_processes": 2,
                "terminated_processes": 0, "job_affinity_mask": 1}

    def close(self):
        self.events.append("job_close")
        if self.close_failure:
            raise OSError("fixture close failure")
        self.handle = None


class FakeHardware:
    def __init__(self, events, *, fail=False):
        self.events, self.handle, self.fail = events, 202, fail

    def close(self):
        self.events.append("hardware_close")
        if self.fail:
            raise OSError("fixture hardware close failure")
        self.handle = None


class FakeChild:
    def __init__(self, events, *, live=False, wait_failure=False):
        self.events, self.returncode, self.wait_failure = events, None if live else 0, wait_failure

    def poll(self):
        return self.returncode

    def kill(self):
        self.events.append("kill_exact_retained_launcher")
        self.returncode = 125

    def wait(self, timeout):
        self.events.append("reap_exact_retained_launcher")
        if self.wait_failure:
            raise TimeoutError("fixture unconfirmed reap")
        return self.returncode


def admission():
    return {"experiment": guard.EXPERIMENT, "stage": "tests", "profile": guard.PROFILE,
            "job_assigned_before_payload": True, "reviewed_proposal_sha256": guard.REVIEWED_SHA,
            "limits": guard.limits(), "output_dir": str(guard.OUTPUT_ROOT / "fixture"),
            "hardware_lock": str(guard.HARDWARE_LOCK), "token": "a" * 32,
            "supervisor_pid": 10, "supervisor_create_time": 100.0,
            "launcher_pid": 11, "launcher_create_time": 101.0,
            "hardware_owner_handle": 21, "job_owner_handle": 22, "cpu": 0,
            "started_monotonic": 90.0, "source_snapshot": {"fixture.py": "b" * 64},
            "source_manifest_sha256": "c" * 64}


def proof(base=None):
    base = admission() if base is None else base
    value = {key: base[key] for key in ("experiment", "stage", "token", "source_manifest_sha256",
                 "supervisor_pid", "supervisor_create_time", "launcher_pid", "launcher_create_time",
                 "hardware_owner_handle", "job_owner_handle")}
    value.update({"worker_pid": 12, "worker_create_time": 102.0,
                  "ancestors": [{"pid": 11, "create_time": 101.0}], "owned_handles_verified": True,
                  "worker_cpu_affinity": [0], "implementation": "cpython",
                  "sys_executable": str(guard.INTERPRETER)})
    return value


class GuardianFixtures(unittest.TestCase):
    def setUp(self):
        self.retained_before = list(guard.RETAINED_HANDLES)

    def tearDown(self):
        # Only our injected fixture objects can be added to this module's list.
        guard.RETAINED_HANDLES[:] = self.retained_before

    def fixture_directory(self):
        root = os.environ.get("DIAGNOSTICS_OUTPUT_DIR")
        if not root:
            raise RuntimeError("STOP_FIXTURES_REQUIRE_ALREADY_OWNED_EXTERNAL_OUTPUT")
        return tempfile.TemporaryDirectory(prefix="tiny-guardian-", dir=Path(root) / "tmp")

    def test_all_five_framework_roots_are_blocked(self):
        blocker = guard.NoModelImports()
        for name in guard.FRAMEWORKS:
            with self.subTest(name=name), self.assertRaisesRegex(ImportError, "MODEL_IMPORT"):
                blocker.find_spec(name + ".submodule")
        self.assertIsNone(blocker.find_spec("unittest"))

    def test_already_loaded_framework_refuses_before_startup_flags(self):
        with patch.dict(guard.sys.modules, {"torch.fixture": object()}):
            with self.assertRaisesRegex(RuntimeError, "ALREADY_LOADED_FRAMEWORK"):
                guard.require_isolated_startup()

    def test_nonisolated_startup_is_rejected(self):
        with patch.object(guard.sys, "flags", types.SimpleNamespace(isolated=0, no_site=1, dont_write_bytecode=1)):
            with self.assertRaisesRegex(RuntimeError, "ISOLATED_I_B_S"):
                guard.require_isolated_startup()

    def test_customization_already_loaded_is_rejected(self):
        with patch.dict(guard.sys.modules, {"sitecustomize": object()}):
            with self.assertRaisesRegex(RuntimeError, "CUSTOMIZATION_ALREADY_LOADED"):
                guard.require_isolated_startup()

    def test_pinned_loader_compiles_snapshot_bytes(self):
        package = "experiments.titan_quality_repair_diagnostics_v1"
        finder = guard.PinnedPackages({package: {"__init__.py": b"", "tiny_probe.py": b"value = 'admitted bytes'\n"}})
        module = types.ModuleType(package + ".tiny_probe")
        finder.exec_module(module)
        self.assertEqual(module.value, "admitted bytes")
        self.assertEqual(module._diagnostics_verified_source_sha256,
                         guard.digest(b"value = 'admitted bytes'\n"))
        with self.assertRaisesRegex(ImportError, "UNPINNED_PACKAGE"):
            finder.find_spec(package + ".missing")
        with self.assertRaisesRegex(ImportError, "OTHER_EXPERIMENT"):
            finder.find_spec("experiments.some_other_lane")

    def test_native_entry_modules_are_unreachable_from_pinned_cpu_loader(self):
        package = "experiments.titan_quality_repair_v1"
        finder = guard.PinnedPackages({package: {"__init__.py": b"", "native_safety.py": b"", "native_proposal.py": b""}})
        with self.assertRaisesRegex(ImportError, "NATIVE_MODULE"):
            finder.find_spec(package + ".native_safety")
        self.assertIsNotNone(finder.find_spec(package + ".native_proposal"))

    def test_both_source_snapshots_include_exact_reviewed_original_40(self):
        bundle = guard.source_bundle()
        self.assertEqual(len(bundle["sources"]["experiments.titan_quality_repair_v1"]), 40)
        self.assertIn("experiments.titan_quality_repair_diagnostics_v1/cpu_supervise.py", bundle["snapshot"])
        self.assertEqual(bundle["snapshot"]["reviewed_proposal"], guard.REVIEWED_SHA)
        with self.assertRaisesRegex(ValueError, "BOTH_PACKAGE_SOURCE_DRIFT"):
            guard.source_bundle({})

    def test_direct_payload_has_no_gate_exists_admission_path(self):
        with patch.object(guard, "require_isolated_startup"), patch.object(guard, "owned_output", return_value=Path("fixture")), \
                patch.object(guard, "append_runtime_metadata") as dependency, patch.object(guard, "run_stage") as stage:
            with self.assertRaisesRegex(RuntimeError, "DIRECT_PAYLOAD_WITHOUT_LIVE_OWNERSHIP"):
                guard.payload("tests", Path("fixture"), None, None)
            dependency.assert_not_called()
            stage.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "DIRECT_PAYLOAD_WITHOUT_LIVE_OWNERSHIP"):
            guard.main(["--payload", "tests", "--output-dir", str(guard.OUTPUT_ROOT / "fixture")])

    def test_admission_rejects_weak_envelope_missing_handles_and_boolean_pid(self):
        good = admission()
        self.assertEqual(guard.validate_admission(good, guard.OUTPUT_ROOT / "fixture", "tests"), good)
        variants = []
        for field, value in (("supervisor_pid", True), ("job_owner_handle", 0),
                             ("supervisor_create_time", float("nan")), ("job_assigned_before_payload", False)):
            bad = copy.deepcopy(good)
            bad[field] = value
            variants.append(bad)
        bad = copy.deepcopy(good)
        bad["limits"]["job_private_commit_bytes"] *= 2
        variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                guard.validate_admission(bad, guard.OUTPUT_ROOT / "fixture", "tests")

    def test_actual_cpython_may_differ_from_owned_launcher(self):
        identity = lambda pid: {12: 102.0}.get(pid)
        result = guard.validate_bootstrap(proof(), admission(), [11, 12], identity)
        self.assertEqual(result, {"worker_pid": 12, "worker_create_time": 102.0})

    def test_stale_identity_foreign_ancestry_handle_substitution_and_no_job_refuse(self):
        variants = []
        for field, value in (("worker_create_time", 999.0), ("ancestors", []),
                             ("job_owner_handle", 999), ("owned_handles_verified", False),
                             ("worker_cpu_affinity", [0, 1])):
            bad = proof()
            bad[field] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                guard.validate_bootstrap(bad, admission(), [11, 12], lambda pid: 102.0)
        with self.assertRaisesRegex(RuntimeError, "IDENTITY_OR_JOB"):
            guard.validate_bootstrap(proof(), admission(), [11], lambda pid: 102.0)

    def test_resource_limits_cover_rss_private_commit_host_output_and_missing_measurements(self):
        value = {"tree_rss_bytes": 1, "tree_private_commit_bytes": 2,
                 "host_available_bytes": guard.HOST_ADMISSION, "new_output_bytes": 3}
        guard.require_resources(value, preload=True)
        for field, bad_value in (("tree_rss_bytes", guard.MEMORY_CAP + 1),
                                 ("tree_private_commit_bytes", guard.MEMORY_CAP + 1),
                                 ("host_available_bytes", guard.HOST_RESERVE - 1),
                                 ("new_output_bytes", guard.OUTPUT_CAP),
                                 ("tree_rss_bytes", float("inf"))):
            bad = {**value, field: bad_value}
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                guard.require_resources(bad)
        with self.assertRaisesRegex(RuntimeError, "RESOURCE_OBSERVATION_UNAVAILABLE"):
            guard.require_resources({})
        with self.assertRaisesRegex(RuntimeError, "RESOURCE_ENVELOPE"):
            guard.require_resources({**value, "host_available_bytes": guard.HOST_ADMISSION - 1}, preload=True)

    def test_observation_counts_job_members_including_orphaned_descendant(self):
        class Missing(Exception):
            pass
        class Denied(Exception):
            pass
        processes = {1: types.SimpleNamespace(create_time=lambda: 1.0,
                       memory_info=lambda: types.SimpleNamespace(rss=20, private=30), cpu_affinity=lambda: [0]),
                     2: types.SimpleNamespace(create_time=lambda: 2.0,
                       memory_info=lambda: types.SimpleNamespace(rss=40, private=50), cpu_affinity=lambda: [0])}
        psutil = types.SimpleNamespace(Process=lambda pid: processes[pid], NoSuchProcess=Missing,
                 AccessDenied=Denied, virtual_memory=lambda: types.SimpleNamespace(available=guard.HOST_ADMISSION))
        with patch.object(guard, "output_bytes", return_value=7):
            result = guard.observe_resources(psutil, Path("fixture"), [1, 2], 0)
        self.assertEqual(result["tree_rss_bytes"], 60)
        self.assertEqual(result["tree_private_commit_bytes"], 80)
        self.assertEqual([p["pid"] for p in result["job_members"]], [1, 2])

    def test_cleanup_waits_for_orphaned_job_member_before_release(self):
        events, clock = [], FakeClock()
        job, child, hardware = FakeJob(events), FakeChild(events), FakeHardware(events)
        states = iter(([777], [], []))
        def members(handle):
            state = next(states)
            events.append("members:" + str(state))
            return state
        result = guard.cleanup_owned(job, child, hardware, members=members, clock=clock,
                                     sleep=clock.sleep, deadline=1)
        self.assertTrue(result["whole_job_exit_verified"])
        self.assertTrue(result["cleanup_reconciled"])
        self.assertLess(events.index("terminate_owned_job"), events.index("hardware_close"))
        self.assertLess(events.index("reap_exact_retained_launcher"), events.index("job_close"))
        self.assertLess(events.index("job_accounting"), events.index("job_close"))
        self.assertLess(events.index("job_close"), events.index("hardware_close"))

    def test_unknown_job_membership_retains_both_handles_and_cannot_reconcile(self):
        events, clock = [], FakeClock()
        job, child, hardware = FakeJob(events), FakeChild(events), FakeHardware(events)
        def unknown(handle):
            raise OSError("fixture unobservable Job")
        result = guard.cleanup_owned(job, child, hardware, members=unknown,
                                     clock=clock, sleep=clock.sleep, deadline=1)
        self.assertFalse(result["whole_job_exit_verified"])
        self.assertFalse(result["cleanup_reconciled"])
        self.assertIsNone(result["job_active_pids_after_cleanup"])
        self.assertNotIn("job_close", events)
        self.assertNotIn("hardware_close", events)
        self.assertIn(job, guard.RETAINED_HANDLES)
        self.assertIn(hardware, guard.RETAINED_HANDLES)

    def test_remaining_member_at_cleanup_deadline_retains_lease(self):
        events, clock = [], FakeClock()
        job, child, hardware = FakeJob(events), FakeChild(events), FakeHardware(events)
        result = guard.cleanup_owned(job, child, hardware, members=lambda h: [777],
                                     clock=clock, sleep=clock.sleep, deadline=.1)
        self.assertFalse(result["whole_job_exit_verified"])
        self.assertEqual(result["job_active_pids_after_cleanup"], [777])
        self.assertNotIn("hardware_close", events)

    def test_assignment_failure_kills_only_exact_retained_launcher(self):
        events, clock = [], FakeClock()
        child, hardware = FakeChild(events, live=True), FakeHardware(events)
        result = guard.cleanup_owned(None, child, hardware, clock=clock, sleep=clock.sleep, deadline=1)
        self.assertTrue(result["exact_launcher_reaped"])
        self.assertEqual(result["launcher_returncode"], 125)
        self.assertLess(events.index("kill_exact_retained_launcher"), events.index("hardware_close"))

    def test_unconfirmed_launcher_reap_retains_hardware_even_with_empty_job(self):
        events, clock = [], FakeClock()
        job, child, hardware = FakeJob(events), FakeChild(events, wait_failure=True), FakeHardware(events)
        result = guard.cleanup_owned(job, child, hardware, members=lambda h: [],
                                     clock=clock, sleep=clock.sleep, deadline=1)
        self.assertFalse(result["whole_job_exit_verified"])
        self.assertNotIn("hardware_close", events)

    def test_job_close_failure_keeps_hardware_and_accounting_failure_cannot_pass(self):
        for close_failure, accounting_failure in ((True, False), (False, True)):
            events, clock = [], FakeClock()
            job = FakeJob(events, close_failure=close_failure, accounting_failure=accounting_failure)
            result = guard.cleanup_owned(job, FakeChild(events), FakeHardware(events), members=lambda h: [],
                                         clock=clock, sleep=clock.sleep, deadline=1)
            self.assertFalse(result["cleanup_reconciled"])
            if close_failure:
                self.assertNotIn("hardware_close", events)

    def test_hardware_close_failure_is_measured_and_not_success(self):
        events, clock = [], FakeClock()
        hardware = FakeHardware(events, fail=True)
        result = guard.cleanup_owned(FakeJob(events), FakeChild(events), hardware, members=lambda h: [],
                                     clock=clock, sleep=clock.sleep, deadline=1)
        self.assertFalse(result["cleanup_reconciled"])
        self.assertFalse(result["hardware_handle_closed"])
        self.assertIn(hardware, guard.RETAINED_HANDLES)

    def test_forced_containment_pass_requires_measured_termination_and_full_cleanup(self):
        record = {"stage": "containment", "status": "SUPERVISOR_STOP", "error": "STOP_CPU_HARD_WALL",
                  "hard_wall_watchdog_fired": True, "source_unchanged_during_worker": True,
                  "containment_fixture": {"forced_owned_job_termination": True}}
        cleanup = {"launcher_returncode": 125, "whole_job_exit_verified": True, "cleanup_reconciled": True}
        self.assertTrue(guard.containment_pass(record, cleanup))
        for field, value in (("status", "WORKER_EXITED"), ("hard_wall_watchdog_fired", False),
                             ("source_unchanged_during_worker", False), ("stage", "tests")):
            with self.subTest(field=field):
                self.assertFalse(guard.containment_pass({**record, field: value}, cleanup))
        for field, value in (("launcher_returncode", 0), ("whole_job_exit_verified", False), ("cleanup_reconciled", False)):
            with self.subTest(field=field):
                self.assertFalse(guard.containment_pass(record, {**cleanup, field: value}))

    def test_create_only_complete_receipt_publication_and_byte_pin(self):
        with self.fixture_directory() as directory:
            path = Path(directory) / "fixture.json"
            guard.write_new(path, {"fixture": True})
            original = path.read_bytes()
            self.assertEqual(guard.read_pinned(path, guard.digest(original)), original)
            with self.assertRaises(FileExistsError):
                guard.write_new(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.staging-*")), [])
            with self.assertRaisesRegex(ValueError, "SOURCE_BINDING"):
                guard.read_pinned(path, "0" * 64)

    def test_repository_output_and_reused_external_output_are_refused(self):
        with self.assertRaisesRegex(ValueError, "FRESH_EXTERNAL_OUTPUT_ONLY"):
            guard.owned_output(guard.HERE / "forbidden-output")
        with self.fixture_directory() as directory:
            with self.assertRaises(FileExistsError):
                guard.owned_output(Path(directory), fresh=True)


if __name__ == "__main__":
    unittest.main()
