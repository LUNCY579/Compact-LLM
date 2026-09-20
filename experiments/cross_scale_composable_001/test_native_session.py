import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import native_session as n
from . import receipts as r


class SessionTests(unittest.TestCase):
    def test_deadline_expiry_and_bound(self):
        self.assertEqual(n.deadline_seconds({"deadline_utc": "2030-01-01T00:10:00Z"},
                                           dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc).timestamp()), 600)
        with self.assertRaises(r.Blocked):
            n.deadline_seconds({"deadline_utc": "2000-01-01T00:00:00Z"})

    def test_resource_and_lease_refusals_precede_native_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lease.lock"
            executable = Path(tmp) / "python.exe"
            executable.touch()
            spec = {"lease_path": str(lock), "runtime": {key: {"python_executable": str(executable)}
                                                       for key in ("student_base", "donor")}}
            with patch.object(n, "validate_source"):
                for role, memory in (("student_base", 6.99), ("donor_base", 11.99), ("donor_skilled", 11.99)):
                    with self.subTest(role=role), self.assertRaises(r.Blocked):
                        n.prelaunch(spec, role, {"available": True, "free_phys_gib": memory})
                n.prelaunch(spec, "student_base", {"available": True, "free_phys_gib": 7.0})
                lock.touch()
                with self.assertRaises(r.Blocked):
                    n.prelaunch(spec, "student_base", {"available": True, "free_phys_gib": 20})
                self.assertTrue(lock.exists())

    def test_missing_authority_and_widened_calls_refused(self):
        with self.assertRaises(r.Blocked):
            n.validate_source({"schema": 1, "experiment": "CSC-001", "stage": "C", "native_capture_authorized": False})


if __name__ == "__main__":
    unittest.main()
