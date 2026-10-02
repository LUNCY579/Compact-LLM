"""CPU tests of the section H supervisor and WMI launcher; nothing is launched."""
import hashlib
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import launch_wmi
import supervise

GRANT = "operator grant 2026-10-01: run the frozen section H screen once"


class FakeLease:
    """Mirrors gpu_lease's lock convention and API on a private lock path."""

    def __init__(self, lock, holder=None, held=False):
        self.LOCK, self.holder, self.held = lock, holder, held
        self.released = []

    def foreign_gpu_holder(self):
        return self.holder

    def acquire(self, label, job, timeout_seconds=3600):
        if self.held or self.LOCK.exists():
            raise RuntimeError("could not acquire")
        self.LOCK.write_text(json.dumps({"pid": 4242, "token": "tok", "label": label, "worker_pids": []}),
                             encoding="utf-8")
        return "tok"

    def release(self, token):
        self.released.append(token)
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

    def run_supervisor(self, child, lease=None, grant=GRANT, clock=None, kill=None):
        lease = lease or FakeLease(self.lock)
        self.launched = []
        def popen(command, **kwargs):
            self.launched.append(command)
            self.lock_during_run = json.loads(self.lock.read_text(encoding="utf-8")) if self.lock.exists() else None
            return child
        def default_kill(pid):
            child.killed = True
            return {"returncode": 0}
        times = iter(clock or [0.0] * 10)
        return lease, supervise.supervise(self.freeze, self.sha, grant, here=self.here, repo=self.repo, python="py",
                                          popen=popen, clock=lambda: next(times), lease_module=lease,
                                          kill=kill or default_kill, hard_wall=600)

    def receipt(self):
        return json.loads((self.here / supervise.RECEIPT_NAME).read_text(encoding="utf-8"))

    def test_grant_is_required(self):
        for grant in ("", "   ", "ok", None):
            with self.assertRaisesRegex(ValueError, "STOP_NO_OPERATOR_GRANT"):
                self.run_supervisor(FakeChild(exits_after=1), grant=grant)
        self.assertFalse(self.lock.exists())

    def test_normal_exit_records_worker_and_releases_own_lease(self):
        lease, receipt = self.run_supervisor(FakeChild(exits_after=5))
        self.assertEqual(receipt["status"], "WORKER_EXITED")
        self.assertTrue(receipt["lease_released"])
        self.assertEqual(lease.released, ["tok"])
        self.assertFalse(self.lock.exists())
        self.assertEqual(self.lock_during_run["worker_pids"], [])
        command = self.launched[0]
        self.assertEqual(command[command.index("--lease-token") + 1], "tok")
        self.assertEqual(command[command.index("--freeze-sha256") + 1], self.sha)
        self.assertEqual(self.receipt()["operator_grant"], GRANT)
        start = json.loads((self.here / supervise.START_NAME).read_text(encoding="utf-8"))
        self.assertEqual(start["lease_token"], "tok")

    def test_worker_pid_is_written_into_the_lease_under_our_token(self):
        lease, receipt = self.run_supervisor(FakeChild(exits_after=1))
        self.assertEqual(receipt["worker_launcher_pid"], 777)
        self.lock.write_text(json.dumps({"token": "tok", "worker_pids": []}), encoding="utf-8")
        supervise.set_worker_pids(self.lock, "tok", [777])
        self.assertEqual(json.loads(self.lock.read_text(encoding="utf-8"))["worker_pids"], [777])
        with self.assertRaisesRegex(RuntimeError, "STOP_LEASE_TOKEN_CHANGED"):
            supervise.set_worker_pids(self.lock, "other", [1])

    def test_hard_wall_kills_own_worker_then_releases(self):
        child = FakeChild(exits_after=None)
        lease, receipt = self.run_supervisor(child)
        self.assertTrue(child.killed)
        self.assertEqual(receipt["status"], "STOP_SUPERVISOR_HARD_WALL")
        self.assertTrue(receipt["lease_released"])

    def test_failed_kill_keeps_lease_under_live_worker(self):
        child = FakeChild(exits_after=None, dies_on_kill=False)
        lease, receipt = self.run_supervisor(child)
        self.assertEqual(receipt["status"], "STOP_SUPERVISOR_HARD_WALL")
        self.assertFalse(receipt["worker_exited"])
        self.assertFalse(receipt["lease_released"])
        self.assertEqual(lease.released, [])
        self.assertTrue(self.lock.exists())

    def test_held_lease_or_foreign_holder_refuses_without_waiting(self):
        with self.assertRaisesRegex(RuntimeError, "could not acquire"):
            self.run_supervisor(FakeChild(exits_after=1), lease=FakeLease(self.lock, held=True))
        with self.assertRaisesRegex(RuntimeError, "STOP_FOREIGN_GPU_HOLDER"):
            self.run_supervisor(FakeChild(exits_after=1), lease=FakeLease(self.lock, holder=(99, 3 << 30)))
        self.assertFalse(self.launched)

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

    def test_outputs_are_create_only(self):
        (self.here / supervise.OUT_NAME).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "STOP_CREATE_ONLY"):
            self.run_supervisor(FakeChild(exits_after=1))
        self.assertFalse(self.lock.exists())


class LauncherTests(unittest.TestCase):
    def test_command_quotes_grant_and_launch_checks_wmi_result(self):
        command = launch_wmi.build_command("C:/py dir/python.exe", Path("C:/a b/F.json"), "ab" * 32,
                                           'grant "quoted" & piped | text', here=Path("C:/x"))
        self.assertIn('"C:/py dir/python.exe"', command)
        self.assertIn('\\"quoted\\"', command)
        calls = []
        def run(args, **kwargs):
            calls.append(kwargs["env"]["SECTIONH_WMI_COMMAND"])
            return types.SimpleNamespace(returncode=0, stdout='{"ProcessId":4321,"ReturnValue":0}', stderr="")
        self.assertEqual(launch_wmi.launch(command, Path("C:/x"), run=run)["ProcessId"], 4321)
        self.assertEqual(calls, [command])
        failing = lambda args, **kwargs: types.SimpleNamespace(returncode=0, stdout='{"ProcessId":0,"ReturnValue":9}', stderr="")
        with self.assertRaisesRegex(RuntimeError, "STOP_WMI_CREATE_FAILED"):
            launch_wmi.launch(command, Path("C:/x"), run=failing)


if __name__ == "__main__":
    unittest.main()
