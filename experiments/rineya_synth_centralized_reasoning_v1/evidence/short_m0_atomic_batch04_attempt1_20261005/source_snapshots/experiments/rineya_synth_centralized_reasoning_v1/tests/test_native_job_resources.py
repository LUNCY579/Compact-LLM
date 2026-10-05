"""Owned-Job resource controls; fake APIs plus a contained CPU-only child.

No model framework, GPU query, checkpoint or qualification worker is launched.
The Windows integration test keeps tiny CPU interpreters in a kill-on-close Job.
"""
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import native_worker as worker

NAME = "Local\\SYNTH_NQ_JOB_" + "a" * 32


class FakeKernel:
    def __init__(self, pids=(42, 43), *, open_error=0, query_error=0, close_error=0,
                 assigned=None, count=None, always_more=False):
        self.pids = list(pids)
        self.open_error, self.query_error, self.close_error = open_error, query_error, close_error
        self.assigned, self.count, self.always_more = assigned, count, always_more
        self.error = 0
        self.opened, self.queried, self.closed = [], [], []
    def OpenJobObjectW(self, access, inherit, name):
        self.opened.append((access, inherit, name))
        self.error = self.open_error
        return 0 if self.open_error else 123
    def QueryInformationJobObject(self, handle, info_class, pointer, byte_count, returned):
        value = pointer._obj
        capacity = len(value.pids)
        assert handle == 123 and info_class == 3 and returned is None
        assert byte_count == ctypes.sizeof(value)
        self.queried.append(capacity)
        if self.query_error:
            self.error = self.query_error
            return False
        value.assigned = len(self.pids) if self.assigned is None else self.assigned
        value.count = min(capacity, len(self.pids)) if self.count is None else self.count
        for index, pid in enumerate(self.pids[:capacity]): value.pids[index] = pid
        self.error = 234 if self.always_more or len(self.pids) > capacity else 0
        return not self.error
    def GetLastError(self): return self.error
    def CloseHandle(self, handle):
        self.closed.append(handle)
        self.error = self.close_error
        return not self.close_error


class FakeNoSuchProcess(Exception): pass
class FakeAccessDenied(Exception): pass


class FakePsutil:
    NoSuchProcess, AccessDenied = FakeNoSuchProcess, FakeAccessDenied
    def __init__(self, values=None, *, dead=()):
        self.values = {42: 100, 43: 200} if values is None else values
        self.dead, self.queries, self.existence = set(dead), [], []
    def Process(self, pid):
        self.queries.append(pid)
        value = self.values[pid]
        if isinstance(value, BaseException): raise value
        def children(*args, **kwargs): raise AssertionError("global process enumeration forbidden")
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=value), children=children)
    def pid_exists(self, pid):
        self.existence.append(pid)
        return pid not in self.dead


class JobPidTests(unittest.TestCase):
    def query(self, kernel, name=NAME):
        with patch.object(worker, "_job_query_api", lambda: kernel), patch.object(worker.os, "getpid", lambda: 42):
            return worker.owned_job_pids(name)

    def test_query_only_rights_noninheritance_exact_pid_membership_and_immediate_close(self):
        kernel = FakeKernel()
        self.assertEqual([42, 43], self.query(kernel))
        self.assertEqual([(4, False, NAME)], kernel.opened)
        self.assertEqual([16], kernel.queried)
        self.assertEqual([123], kernel.closed)

    def test_growth_uses_assigned_count_and_remains_bounded(self):
        kernel = FakeKernel([42] + list(range(100, 132)))
        result = self.query(kernel)
        self.assertEqual(kernel.pids, result)
        self.assertEqual([16, 33], kernel.queried)
        self.assertEqual([123], kernel.closed)
        kernel = FakeKernel([42] + list(range(100, 355)))
        self.assertEqual(256, len(self.query(kernel)))
        self.assertEqual([16, 256], kernel.queried)

    def test_malformed_name_never_opens_query_handle(self):
        for name in (None, "Global\\SYNTH_NQ_JOB_" + "a" * 32, "Local\\other", NAME + "extra"):
            kernel = FakeKernel()
            with self.subTest(name=name), self.assertRaises(ValueError): self.query(kernel, name)
            self.assertEqual([], kernel.opened)

    def test_missing_actual_interpreter_empty_duplicate_zero_and_wide_pid_fail_closed(self):
        for pids in ([43, 44], [], [42, 42], [42, 0], [42, 0x100000000]):
            kernel = FakeKernel(pids)
            with self.subTest(pids=pids), self.assertRaises(ValueError): self.query(kernel)
            self.assertEqual([123], kernel.closed)

    def test_incomplete_success_or_count_over_capacity_is_rejected(self):
        for assigned, count in ((3, 2), (1, 2), (20, 20), (257, 2)):
            kernel = FakeKernel(assigned=assigned, count=count)
            with self.subTest(assigned=assigned, count=count), self.assertRaises(ValueError): self.query(kernel)
            self.assertEqual([123], kernel.closed)

    def test_open_query_and_close_failures_are_not_silently_ignored(self):
        kernel = FakeKernel(open_error=5)
        with self.assertRaises(OSError): self.query(kernel)
        self.assertEqual([], kernel.closed)
        for code in (5, 6, 87):
            kernel = FakeKernel(query_error=code)
            with self.subTest(error=code), self.assertRaises(OSError): self.query(kernel)
            self.assertEqual([123], kernel.closed)
        kernel = FakeKernel(close_error=6)
        with self.assertRaises(OSError): self.query(kernel)
        self.assertEqual([123], kernel.closed)

    def test_more_than256_or_repeated_growth_never_returns_partial_list(self):
        kernel = FakeKernel([42] + list(range(100, 356)))
        with self.assertRaisesRegex(ValueError, "containment bound"): self.query(kernel)
        self.assertEqual([16], kernel.queried)
        self.assertEqual([123], kernel.closed)
        kernel = FakeKernel(always_more=True)
        with self.assertRaisesRegex(ValueError, "containment bound"): self.query(kernel)
        self.assertEqual([16, 32, 64, 128, 256], kernel.queried)
        self.assertEqual([123], kernel.closed)


class ResourceTests(unittest.TestCase):
    def resources(self, psutil, pids=(42, 43)):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"psutil": psutil}))
            stack.enter_context(patch.object(worker, "owned_job_pids", lambda name: list(pids)))
            stack.enter_context(patch.object(worker.os, "getpid", lambda: 42))
            return worker.owned_job_resources(NAME)

    def test_only_queried_pids_are_measured_without_children_or_enumeration(self):
        psutil = FakePsutil({42: 100, 43: 200, 99: 999999})
        result = self.resources(psutil)
        self.assertEqual(300, result["tree_rss_bytes"])
        self.assertEqual([42, 43], psutil.queries)
        self.assertEqual([42, 43], result["observed_pids"])
        self.assertEqual([], result["vanished_pids"])
        self.assertTrue(result["query_handle_closed"])

    def test_only_confirmed_dead_noncurrent_NoSuchProcess_race_is_ignored(self):
        psutil = FakePsutil({42: 100, 43: FakeNoSuchProcess("gone")}, dead=[43])
        result = self.resources(psutil)
        self.assertEqual(100, result["tree_rss_bytes"])
        self.assertEqual([43], result["vanished_pids"])
        self.assertEqual([43], psutil.existence)
        psutil = FakePsutil({42: 100, 43: FakeNoSuchProcess("not confirmed")})
        with self.assertRaisesRegex(ValueError, "not confirmed terminal"): self.resources(psutil)

    def test_current_interpreter_disappearance_access_denial_and_rss_errors_fail_closed(self):
        psutil = FakePsutil({42: FakeNoSuchProcess("current cannot be dead"), 43: 200}, dead=[42])
        with self.assertRaises(ValueError): self.resources(psutil)
        for value in (FakeAccessDenied("access denied"), OSError("query failure")):
            psutil = FakePsutil({42: 100, 43: value})
            with self.subTest(value=value), self.assertRaises(type(value)): self.resources(psutil)
        for value in (-1, True, 1.5, None):
            psutil = FakePsutil({42: 100, 43: value})
            with self.subTest(value=value), self.assertRaises(ValueError): self.resources(psutil)

    def test_missing_current_rss_is_rejected_even_if_injected_reader_omits_membership_check(self):
        with self.assertRaisesRegex(ValueError, "interpreter RSS was not measured"):
            self.resources(FakePsutil({43: 200}), [43])

    def test_guard_retains_original_limits_and_has_no_children_enumeration(self):
        source = Path(worker.__file__).read_text()
        guard = source[source.index("    def guard(stage):"):source.index("    journal.guard = guard")]
        self.assertIn('owned_job_resources(owner["job_name"])["tree_rss_bytes"]', guard)
        self.assertNotIn("children(", guard)
        self.assertIn("rss <= 2 * 1024**3 and available >= 1.25 * 1024**3", guard)
        self.assertIn("time.monotonic() - start <= 600", guard)
        self.assertIn("free >= 0.5 * 1024**3", guard)


CPU_CHILD = r'''
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import native_worker as worker
import psutil
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
job_name, output = sys.argv[2], Path(sys.argv[3])
child = subprocess.Popen([sys.executable, '-I', '-B', '-c',
    'import sys; payload=bytearray(2*1024*1024); print("READY",flush=True); sys.stdin.read(1)'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    creationflags=0x08000000)
assert child.stdout.readline().strip() == 'READY'
# The original expensive guard path must not be available in this fixture.
def forbidden(*args, **kwargs): raise AssertionError('global children enumeration forbidden')
psutil.Process.children = forbidden
samples = []
for _ in range(5):
    started = time.perf_counter()
    resources = worker.owned_job_resources(job_name)
    elapsed = time.perf_counter()-started
    assert os.getpid() in resources['observed_pids']
    assert child.pid in resources['queried_job_pids']
    assert resources['tree_rss_bytes'] >= psutil.Process().memory_info().rss
    assert resources['query_handle_closed'] and len(resources['queried_job_pids']) <= 256
    samples.append({'elapsed_seconds':elapsed, **resources})
with output.open('x',encoding='utf-8') as stream:
    json.dump({'status':'CPU_ONLY_OWNED_JOB_QUERY_PASSED', 'actual_interpreter_pid':os.getpid(),
               'descendant_pid':child.pid, 'samples':samples, 'model_frameworks_imported':False}, stream)
    stream.flush()
sys.stdin.read(1)
'''


@unittest.skipUnless(os.name == "nt", "real Windows Job integration requires Windows")
class WindowsJobIntegrationTests(unittest.TestCase):
    def test_contained_CPU_interpreter_and_descendant_query_then_kill_on_close(self):
        import native_supervisor as supervisor
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cpu_job_query.json"
            job = supervisor._WindowsJob()
            process = None
            try:
                process = subprocess.Popen([sys.executable, "-I", "-B", "-c", CPU_CHILD,
                                            str(PACKAGE), job.name, str(output)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, creationflags=0x08000000 | 0x00000004)
                job.assign(int(process._handle))
                supervisor._resume_primary_thread(process.pid)
                deadline = time.monotonic() + 15
                while not output.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(output.exists(), "contained CPU child produced no query receipt")
                # Wait for the brief exclusive write to finish without inspecting
                # or launching any model/GPU process.
                while True:
                    try:
                        report = json.loads(output.read_text())
                        break
                    except (json.JSONDecodeError, PermissionError):
                        if time.monotonic() >= deadline: raise
                        time.sleep(0.01)
                self.assertEqual("CPU_ONLY_OWNED_JOB_QUERY_PASSED", report["status"])
                self.assertFalse(report["model_frameworks_imported"])
                self.assertEqual(5, len(report["samples"]))
                self.assertTrue(all(sample["query_handle_closed"] for sample in report["samples"]))
                # A retained child query handle would keep this named Job alive
                # and prevent the supervisor's kill-on-close from terminating it.
                self.assertIsNone(process.poll())
                job.close()
                process.wait(timeout=5)
                self.assertIsNotNone(process.returncode)
                import psutil
                deadline = time.monotonic() + 5
                while psutil.pid_exists(report["actual_interpreter_pid"]) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(psutil.pid_exists(report["actual_interpreter_pid"]))
            finally:
                job.close()
                if process is not None:
                    if process.poll() is None: process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__": unittest.main()
