"""Small failure-injection fixtures for the offline controller; no hardware launch."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import parentq_offline_supervise as controller


class OfflineControllerTests(unittest.TestCase):
    def test_model_imports_are_rejected(self):
        guard = controller.NoModelImports()
        for name in ("torch", "torch.cuda", "transformers", "tensorflow", "jax", "cupy"):
            with self.assertRaisesRegex(ImportError, "STOP_LIGHTWEIGHT_PROFILE_MODEL_IMPORT"):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec("json"))

    def test_output_locations_and_create_only_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            with self.assertRaisesRegex(ValueError, "STOP_OUTPUT"):
                controller.output_root(controller.REPO / "output")
            with self.assertRaisesRegex(ValueError, "STOP_OUTPUT"):
                controller.output_root(controller.BUSINESS / ".codex/output")
            checkout = parent / "fixture_checkout"
            checkout.mkdir()
            (checkout / ".git").write_text("fixture")
            with self.assertRaisesRegex(ValueError, "STOP_OUTPUT"):
                controller.output_root(checkout / "output")
            root = controller.output_root(parent / "output")
            with self.assertRaises(FileExistsError):
                controller.output_root(root)

    def test_failed_job_assignment_kills_owned_handle_before_releasing_hardware(self):
        order = []
        class Hardware:
            def __init__(self, *args):
                order.append("hardware_acquired")
            def close(self):
                order.append("hardware_closed")
        class Job:
            def assign(self, child):
                raise RuntimeError("fixture assignment failed")
            def terminate(self):
                order.append("job_terminated")
            def close(self):
                order.append("job_closed")
        class Child:
            pid, dead = 777, False
            def poll(self):
                return 0 if self.dead else None
            def kill(self):
                self.dead = True
                order.append("owned_handle_killed")
            def wait(self, timeout):
                order.append("owned_exit_confirmed")
                return 125
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "failure"
            child = Child()
            with patch.object(controller, "HardwareHandle", Hardware), patch.object(controller, "Job", Job), \
                    patch.object(controller.subprocess, "Popen", return_value=child):
                self.assertEqual(controller.supervise("tests", root), 1)
            receipt = json.loads((root / "OFFLINE_SUPERVISOR_RECEIPT.json").read_bytes())
            self.assertEqual(receipt["status"], "SUPERVISOR_FAILURE")
            self.assertNotIn("worker_exit_unconfirmed", receipt)
            self.assertLess(order.index("owned_exit_confirmed"), order.index("hardware_closed"))


if __name__ == "__main__":
    unittest.main()
