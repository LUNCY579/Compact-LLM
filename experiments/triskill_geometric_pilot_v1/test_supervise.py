"""CPU tests of the section H supervisor and WMI launcher; nothing is launched."""
import hashlib
import json
import os
import subprocess
import tempfile
import time
import types
import unittest
from pathlib import Path

import launch_wmi
import supervise
import worker

GRANT = "operator grant 2026-10-01: run the frozen section H screen once"


class FakeLease:
    """Mirrors gpu_lease's lock convention and API on a private lock path."""

    def __init__(self, lock, holder=None, release_failures=0):
        self.LOCK, self.holder, self.release_failures = lock, holder, release_failures
        self.released, self.acquired = [], 0

    def foreign_gpu_holder(self):
        return self.holder

    def acquire(self, label, job, timeout_seconds=3600):
        if self.LOCK.exists():
            raise RuntimeError("could not acquire")
        self.acquired += 1
        self.LOCK.write_text(json.dumps({"pid": 4242, "token": "tok", "label": label, "worker_pids": []}),
                             encoding="utf-8")
        return "tok"

    def release(self, token):
        self.released.append(token)
        if self.release_failures:
            self.release_failures -= 1
            return
        if json.loads(self.LOCK.read_text(encoding="utf-8")).get("token") == token:
            self.LOCK.unlink()


class FakeChild:
    def __init__(self, exits_after=None, dies_on_kill=True):
        self.pid, self.exits_after, self.dies_on_kill, self.killed = 777, exits_after, dies_on_kill, False

    def wait(self, timeout=None):
        if self.exits_after is not None and timeout >= self.exits_after:
            return 0
        if self.killed and self.dies_on_kill:
            return 1
        raise subprocess.TimeoutExpired("worker", timeout)

    def poll(self):
        if self.exits_after is not None or (self.killed and self.dies_on_kill):
            return 0
        return None


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.here = self.repo / "experiments" / "triskill_geometric_pilot_v1"
        self.here.mkdir(parents=True)
        lease_source = self.repo / supervise.GPU_LEASE_RELATIVE
        lease_source.parent.mkdir(parents=True)
        lease_source.write_text("LOCK = None\n", encoding="utf-8")
        files = {}
        for name in supervise.PINNED_LOCAL:
            (self.here / name).write_text(f"# {name}\n", encoding="utf-8")
            files[name] = hashlib.sha256((self.here / name).read_bytes()).hexdigest()
        self.freeze = self.here / supervise.FREEZE_NAME
        self.freeze.write_text(json.dumps({"files": files, "supervisor_dependencies": {"gpu_lease": {
            "path": supervise.GPU_LEASE_RELATIVE, "sha256": hashlib.sha256(lease_source.read_bytes()).hexdigest()}}}),
            encoding="utf-8")
        self.sha = hashlib.sha256(self.freeze.read_bytes()).hexdigest()
        self.lock = self.repo / "gpu.lock"
        self.attempt = self.here / "native_attempts" / "a1"
        self.attempt.mkdir(parents=True)

    def run_supervisor(self, child, lease=None, grant=GRANT, kill=None, alive=None, worker_pid=None):
        lease = lease or FakeLease(self.lock)
        self.launched, self.killed = [], []
        def popen(command, **kwargs):
            self.launched.append((command, kwargs))
            if worker_pid is not None:
                (self.attempt / supervise.WORKER_PID_NAME).write_text(json.dumps({"pid": worker_pid}), encoding="utf-8")
            return child
        def default_kill(pid):
            self.killed.append(pid)
            if pid == child.pid:
                child.killed = True
            return {"returncode": 0}
        return lease, supervise.supervise(self.freeze, self.sha, grant, here=self.here, repo=self.repo, python="py",
                                          popen=popen, clock=lambda: 0.0, lease_module=lease,
                                          kill=kill or default_kill, alive=alive or (lambda pid: False),
                                          hard_wall=600, sleep=lambda s: None, attempt_dir=self.attempt)

    def receipt(self):
        return json.loads((self.attempt / supervise.RECEIPT_NAME).read_text(encoding="utf-8"))

    def test_grant_is_required(self):
        for grant in ("", "   ", "ok", None):
            with self.assertRaisesRegex(ValueError, "STOP_NO_OPERATOR_GRANT"):
                self.run_supervisor(FakeChild(exits_after=1), grant=grant)
        self.assertFalse(self.lock.exists())

    def test_normal_exit_records_worker_and_releases_own_lease(self):
        lease, receipt = self.run_supervisor(FakeChild(exits_after=5))
        self.assertEqual(receipt["status"], "WORKER_EXITED")
        self.assertTrue(receipt["lease_released"])
        self.assertFalse(self.lock.exists())
        command, kwargs = self.launched[0]
        self.assertEqual(command[command.index("--lease-token") + 1], "tok")
        self.assertEqual(command[command.index("--attempt-dir") + 1], str(self.attempt))
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(self.receipt()["operator_grant"], GRANT)
        start = json.loads((self.attempt / supervise.START_NAME).read_text(encoding="utf-8"))
        self.assertEqual(start["lease_token"], "tok")
        # The worker's own grant check accepts exactly this receipt.
        self.assertEqual(worker.verify_supervisor_start(self.attempt, "tok")["operator_grant"], GRANT)
        with self.assertRaisesRegex(RuntimeError, "STOP_NO_SUPERVISOR_GRANT"):
            worker.verify_supervisor_start(self.attempt, "other-token")

    def test_hard_wall_kills_own_worker_then_releases(self):
        child = FakeChild(exits_after=None)
        lease, receipt = self.run_supervisor(child)
        self.assertTrue(child.killed)
        self.assertEqual(receipt["status"], "STOP_SUPERVISOR_HARD_WALL")
        self.assertTrue(receipt["lease_released"])

    def test_real_worker_pid_is_killed_and_must_be_dead_before_release(self):
        alive_pids = {9001}
        def kill(pid):
            self.killed.append(pid)
            return {"returncode": 0}
        lease, receipt = self.run_supervisor(FakeChild(exits_after=1), worker_pid=9001, kill=kill,
                                             alive=lambda pid: pid in alive_pids)
        self.assertIn(9001, self.killed)
        self.assertFalse(receipt["worker_exited"])
        self.assertFalse(receipt["lease_released"])
        self.assertTrue(self.lock.exists())

    def test_failed_kill_keeps_lease_under_live_worker(self):
        child = FakeChild(exits_after=None, dies_on_kill=False)
        lease, receipt = self.run_supervisor(child)
        self.assertEqual(receipt["status"], "STOP_SUPERVISOR_HARD_WALL")
        self.assertFalse(receipt["worker_exited"])
        self.assertFalse(receipt["lease_released"])
        self.assertEqual(lease.released, [])
        self.assertTrue(self.lock.exists())

    def test_existing_lock_is_never_reclaimed_and_foreign_holder_refuses(self):
        self.lock.write_text(json.dumps({"pid": 1, "token": "theirs", "worker_pids": []}), encoding="utf-8")
        lease = FakeLease(self.lock)
        with self.assertRaisesRegex(RuntimeError, "STOP_LEASE_PRESENT"):
            self.run_supervisor(FakeChild(exits_after=1), lease=lease)
        self.assertEqual(lease.acquired, 0)
        self.assertEqual(json.loads(self.lock.read_text(encoding="utf-8"))["token"], "theirs")
        self.lock.unlink()
        with self.assertRaisesRegex(RuntimeError, "STOP_FOREIGN_GPU_HOLDER"):
            self.run_supervisor(FakeChild(exits_after=1), lease=FakeLease(self.lock, holder=(99, 3 << 30)))
        self.assertFalse(self.launched)

    def test_spent_screen_refuses_any_new_attempt(self):
        (self.here / supervise.OUT_NAME).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "STOP_SCREEN_ALREADY_SPENT"):
            self.run_supervisor(FakeChild(exits_after=1))
        self.assertFalse(self.lock.exists())

    def test_lock_rewrite_and_release_retry_through_sharing_violations(self):
        self.lock.write_text(json.dumps({"token": "tok", "worker_pids": []}), encoding="utf-8")
        real_replace, calls = os.replace, []
        def flaky(src, dst):
            calls.append(src)
            if len(calls) < 3:
                raise PermissionError(13, "Access is denied")
            return real_replace(src, dst)
        supervise.os.replace = flaky
        try:
            supervise.set_worker_pids(self.lock, "tok", [777], sleep=lambda s: None)
        finally:
            supervise.os.replace = real_replace
        self.assertEqual(json.loads(self.lock.read_text(encoding="utf-8"))["worker_pids"], [777])
        self.assertEqual(len(calls), 3)
        self.assertFalse(list(self.lock.parent.glob("*.tmp")))
        with self.assertRaisesRegex(RuntimeError, "STOP_LEASE_TOKEN_CHANGED"):
            supervise.set_worker_pids(self.lock, "other", [1], sleep=lambda s: None)
        lease = FakeLease(self.lock, release_failures=2)
        self.assertTrue(supervise.release_own_lease(lease, "tok", sleep=lambda s: None))
        self.assertEqual(len(lease.released), 3)

    def test_freeze_and_pinned_files_must_match(self):
        with self.assertRaisesRegex(ValueError, "STOP_FREEZE_DIGEST_MISMATCH"):
            supervise.verify_freeze(self.freeze, "0" * 64, self.here, self.repo)
        (self.here / "worker.py").write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "STOP_FREEZE_FILE_MISMATCH:worker.py"):
            supervise.verify_freeze(self.freeze, self.sha, self.here, self.repo)

    def test_gpu_lease_module_is_loaded_only_from_pinned_bytes(self):
        freeze = supervise.verify_freeze(self.freeze, self.sha, self.here, self.repo)
        self.assertTrue(hasattr(supervise.load_gpu_lease(freeze, self.repo), "LOCK"))
        (self.repo / supervise.GPU_LEASE_RELATIVE).write_text("LOCK = 'swapped'\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "STOP_GPU_LEASE_DIGEST_MISMATCH"):
            supervise.load_gpu_lease(freeze, self.repo)

    def test_refusals_leave_a_failure_receipt(self):
        try:
            raise RuntimeError("STOP_LEASE_PRESENT")
        except RuntimeError as exc:
            path = supervise._record_failure(self.here, exc)
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("STOP_LEASE_PRESENT", record["error"])
        self.assertTrue(path.name.startswith("SUPERVISOR_FAILURE_"))

    def test_names_agree_with_the_worker(self):
        self.assertEqual(supervise.LEASE_LABEL, worker.LEASE_LABEL)
        self.assertEqual(supervise.START_NAME, worker.START_NAME)
        self.assertEqual(supervise.WORKER_PID_NAME, worker.WORKER_PID_NAME)
        self.assertEqual(launch_wmi.START_NAME, supervise.START_NAME)


class LauncherTests(unittest.TestCase):
    def test_command_quotes_grant_hides_window_and_checks_wmi_result(self):
        command = launch_wmi.build_command("C:/py dir/python.exe", Path("C:/a b/F.json"), "ab" * 32,
                                           'grant "quoted" & piped | text', here=Path("C:/x"))
        self.assertIn('"C:/py dir/python.exe"', command)
        self.assertIn('\\"quoted\\"', command)
        self.assertIn("ShowWindow=[uint16]0", launch_wmi.POWERSHELL)
        self.assertIn("ProcessStartupInformation=$si", launch_wmi.POWERSHELL)
        calls = []
        def run(args, **kwargs):
            calls.append(kwargs["env"]["SECTIONH_WMI_COMMAND"])
            return types.SimpleNamespace(returncode=0, stdout='{"ProcessId":4321,"ReturnValue":0}', stderr="")
        self.assertEqual(launch_wmi.launch(command, Path("C:/x"), run=run)["ProcessId"], 4321)
        self.assertEqual(calls, [command])
        failing = lambda args, **kwargs: types.SimpleNamespace(returncode=0, stdout='{"ProcessId":0,"ReturnValue":9}', stderr="")
        with self.assertRaisesRegex(RuntimeError, "STOP_WMI_CREATE_FAILED"):
            launch_wmi.launch(command, Path("C:/x"), run=failing)

    def test_launch_confirmation_reports_start_failure_or_silence(self):
        with tempfile.TemporaryDirectory() as folder:
            here = Path(folder)
            since = time.time() - 5
            silent = launch_wmi.confirm_started(here, since, timeout=0, sleep=lambda s: None)
            self.assertEqual(silent["supervisor"], "UNCONFIRMED")
            attempts = here / "native_attempts"
            (attempts / "a1").mkdir(parents=True)
            (attempts / "SUPERVISOR_FAILURE_x.json").write_text(json.dumps({"error": "STOP_LEASE_PRESENT"}), encoding="utf-8")
            failed = launch_wmi.confirm_started(here, since, timeout=0, sleep=lambda s: None)
            self.assertEqual((failed["supervisor"], failed["error"]), ("FAILED", "STOP_LEASE_PRESENT"))
            (attempts / "SUPERVISOR_FAILURE_x.json").unlink()
            (attempts / "a1" / launch_wmi.START_NAME).write_text("{}", encoding="utf-8")
            self.assertEqual(launch_wmi.confirm_started(here, since, timeout=0, sleep=lambda s: None)["supervisor"], "STARTED")


if __name__ == "__main__":
    unittest.main()
