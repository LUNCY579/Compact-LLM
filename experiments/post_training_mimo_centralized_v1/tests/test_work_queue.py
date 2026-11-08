import copy
import unittest

from experiments.post_training_mimo_centralized_v1.scripts.work_queue import plan, validate


def task(key, gpu=0, deps=()):
    return dict(task_id=key, status="pending", lane="HEAVY" if gpu else "FAST", dependency=list(deps),
                cpu_cores=1, ram_gib=1, gpu_gib=gpu, expected_seconds=14401 if gpu else 30,
                native_gate_receipt="qualified.json" if gpu else None, blocker=None)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.q = dict(resource_policy=dict(host_free_gib=7.4, worker_free_gib=6.0, gpu_free_gib=4.0, disk_free_gib=4.0), tasks=[])
        self.r = dict(cpu_cores=4, ram_free_gib=12, gpu_free_gib=7, disk_free_gib=20)

    def test_independent_fast_and_single_heavy(self):
        self.q["tasks"] = [task("gpu1", 4), task("gpu2", 4), task("audit")]
        p = plan(self.q, self.r)
        self.assertEqual(p["admitted_wave"], ["audit", "gpu1"])
        self.assertIn("one GPU workload limit", p["held"]["gpu2"])

    def test_running_gpu_keeps_fast_work_live(self):
        first = task("running", 4)
        first["status"] = "running"
        self.q["tasks"] = [first, task("next", 4), task("audit")]
        self.assertEqual(plan(self.q, self.r)["admitted_wave"], ["audit"])

    def test_dependency_cannot_complete_within_planned_wave(self):
        self.q["tasks"] = [task("audit"), task("training", 4, ["audit"])]
        self.assertEqual(plan(self.q, self.r)["admitted_wave"], ["audit"])

    def test_floors_and_missing_receipts_fail_closed(self):
        job = task("training", 4)
        job["native_gate_receipt"] = None
        self.q["tasks"] = [job]
        self.assertEqual(plan(self.q, self.r)["admitted_wave"], [])
        self.q["resource_policy"]["host_free_gib"] = 7.0
        with self.assertRaises(ValueError):
            plan(self.q, self.r)

    def test_cycle_unknown_dependency_duplicate_and_fake_completion(self):
        for tasks in ([task("a", deps=["b"]), task("b", deps=["a"])],
                      [task("a", deps=["missing"])], [task("a"), task("a")]):
            with self.assertRaises(ValueError):
                validate(tasks)
        complete = task("a")
        complete["status"] = "complete"
        with self.assertRaises(ValueError):
            validate([complete])

    def test_unknown_nonfinite_and_negative_resource_refused(self):
        for value in (None, float("nan"), -1, True):
            r = copy.deepcopy(self.r)
            r["ram_free_gib"] = value
            with self.assertRaises(ValueError):
                plan(self.q, r)

    def test_fast_reservations_cannot_starve_worker_floor(self):
        self.q["tasks"] = [task("audit"), task("training", 4)]
        self.r["ram_free_gib"] = 8.0
        result = plan(self.q, self.r)
        self.assertEqual(result["admitted_wave"], ["audit"])
        self.assertIn("inherited RAM admission/worker floor", result["held"]["training"])


if __name__ == "__main__":
    unittest.main()
