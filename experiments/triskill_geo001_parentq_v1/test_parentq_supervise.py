"""CPU tests of the PARENTQ supervisor and WMI launcher (adapted from GEO's test_supervise.py);
nothing is launched."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import types
import unittest
from pathlib import Path

import derive_from_geo
import parentq_launch_wmi as launch_wmi
import parentq_supervise as supervise
import parentq_worker as worker

HERE = Path(__file__).resolve().parent
GRANT = "operator grant 2026-10-02: run the frozen PARENTQ qualification"


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
        self.here = self.repo / "experiments" / "triskill_geo001_parentq_v1"
        self.here.mkdir(parents=True)
        geo_dir = self.repo / "experiments" / "triskill_geometric_pilot_v1"
        geo_dir.mkdir()
        shutil.copy(HERE.parent / "triskill_geometric_pilot_v1" / "FREEZE_SECTIONH.json", geo_dir / "FREEZE_SECTIONH.json")
        self.geo_freeze = geo_dir / "FREEZE_SECTIONH.json"
        lease_source = self.repo / supervise.GPU_LEASE_RELATIVE
        lease_source.parent.mkdir(parents=True)
        lease_source.write_text("LOCK = None\n", encoding="utf-8")
        files = {}
        for name in supervise.PINNED_LOCAL:
            (self.here / name).write_text(f"# {name}\n", encoding="utf-8")
            files[name] = hashlib.sha256((self.here / name).read_bytes()).hexdigest()
        self.freeze = self.here / supervise.FREEZE_NAME
        self.freeze.write_text(json.dumps({"files": files, "geo_freeze": {"sha256": supervise.GEO_FREEZE_SHA256},
                                           "supervisor_dependencies": {"gpu_lease": {
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

    def test_normal_exit_launches_the_parentq_worker_and_releases_own_lease(self):
        lease, receipt = self.run_supervisor(FakeChild(exits_after=5))
        self.assertEqual(receipt["status"], "WORKER_EXITED")
        self.assertEqual(receipt["experiment"], "TRISKILL-GEO-001-PARENTQ")
        self.assertTrue(receipt["lease_released"])
        self.assertFalse(self.lock.exists())
        command, kwargs = self.launched[0]
        self.assertTrue(command[2].endswith("parentq_worker.py"))
        self.assertEqual(command[command.index("--lease-token") + 1], "tok")
        self.assertEqual(command[command.index("--attempt-dir") + 1], str(self.attempt))
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        start = json.loads((self.attempt / supervise.START_NAME).read_text(encoding="utf-8"))
        self.assertEqual((start["lease_token"], start["operator_grant"]), ("tok", GRANT))
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
        def kill(pid):
            self.killed.append(pid)
            return {"returncode": 0}
        lease, receipt = self.run_supervisor(FakeChild(exits_after=1), worker_pid=9001, kill=kill,
                                             alive=lambda pid: pid == 9001)
        self.assertIn(9001, self.killed)
        self.assertFalse(receipt["lease_released"])
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

    def test_decided_result_and_exhausted_attempts_refuse(self):
        (self.here / supervise.OUT_NAME).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "STOP_PARENTQ_ALREADY_DECIDED"):
            self.run_supervisor(FakeChild(exits_after=1))
        (self.here / supervise.OUT_NAME).unlink()
        for index in range(supervise.MAX_LOADED_ATTEMPTS):
            folder = self.here / supervise.ATTEMPTS_DIR / f"loaded{index}"
            folder.mkdir()
            (folder / supervise.INCOMPLETE_NAME).write_text("{}", encoding="utf-8")
            if index < supervise.MAX_LOADED_ATTEMPTS - 1:
                self.assertEqual(supervise.loaded_incomplete_attempts(self.here), index + 1)
        with self.assertRaisesRegex(RuntimeError, "STOP_PARENTQ_ATTEMPTS_EXHAUSTED"):
            self.run_supervisor(FakeChild(exits_after=1))
        self.assertFalse(self.lock.exists())

    def test_freeze_pins_and_the_geo_freeze_must_match(self):
        with self.assertRaisesRegex(ValueError, "STOP_FREEZE_DIGEST_MISMATCH"):
            supervise.verify_freeze(self.freeze, "0" * 64, self.here, self.repo)
        supervise.verify_freeze(self.freeze, self.sha, self.here, self.repo)
        self.geo_freeze.write_bytes(self.geo_freeze.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "STOP_GEO_FREEZE_MISMATCH"):
            supervise.verify_freeze(self.freeze, self.sha, self.here, self.repo)
        (self.here / "parentq_worker.py").write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "STOP_FREEZE_FILE_MISMATCH:parentq_worker.py"):
            supervise.verify_freeze(self.freeze, self.sha, self.here, self.repo)

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
        lease = FakeLease(self.lock, release_failures=2)
        self.assertTrue(supervise.release_own_lease(lease, "tok", sleep=lambda s: None))

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
        self.assertIn("STOP_LEASE_PRESENT", json.loads(path.read_text(encoding="utf-8"))["error"])

    def test_derived_files_differ_from_geo_only_by_the_registered_replacements(self):
        self.assertTrue(derive_from_geo.check())
        self.assertEqual(supervise.HARD_WALL_SECONDS, 600)
        self.assertEqual(supervise.PINNED_LOCAL, ("parentq_worker.py", "parentq_supervise.py", "parentq_rows.py",
                                                  "parentq_decide.py", "parentq_bank.json", "parentq_canary.json"))


class LauncherTests(unittest.TestCase):
    def test_command_targets_the_parentq_supervisor_and_hides_the_window(self):
        command = launch_wmi.build_command("C:/py dir/python.exe", Path("C:/a b/F.json"), "ab" * 32,
                                           'grant "quoted" & piped | text', here=Path("C:/x"))
        self.assertIn("parentq_supervise.py", command)
        self.assertIn('\\"quoted\\"', command)
        self.assertIn("ShowWindow=[uint16]0", launch_wmi.POWERSHELL)
        calls = []
        def run(args, **kwargs):
            calls.append(kwargs["env"]["PARENTQ_WMI_COMMAND"])
            return types.SimpleNamespace(returncode=0, stdout='{"ProcessId":4321,"ReturnValue":0}', stderr="")
        self.assertEqual(launch_wmi.launch(command, Path("C:/x"), run=run)["ProcessId"], 4321)
        self.assertEqual(calls, [command])

    def test_launch_confirmation_reports_start_failure_or_silence(self):
        with tempfile.TemporaryDirectory() as folder:
            here = Path(folder)
            since = time.time() - 5
            self.assertEqual(launch_wmi.confirm_started(here, since, timeout=0, sleep=lambda s: None)["supervisor"],
                             "UNCONFIRMED")
            attempts = here / "native_attempts"
            (attempts / "a1").mkdir(parents=True)
            (attempts / "a1" / launch_wmi.START_NAME).write_text("{}", encoding="utf-8")
            self.assertEqual(launch_wmi.confirm_started(here, since, timeout=0, sleep=lambda s: None)["supervisor"],
                             "STARTED")


if __name__ == "__main__":
    unittest.main()
