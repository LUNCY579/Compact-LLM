"""Synthetic receipt tests only; never inspect the active run or model files."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_mixture_v1 import cost_report as c


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class NativeJournalTests(unittest.TestCase):
    def events(self, bodies):
        records = {}; previous = None
        for i, value in enumerate(bodies):
            body = {**value, "index": i, "previous": previous}
            sealed = {**body, "sha256": c.digest(body)}
            path = Path(f"{i:06d}.json"); records[path] = sealed; previous = sealed["sha256"]
        return records

    def test_settlement_replaces_both_wall_reservations(self):
        worker, owned = c.WALL_KEYS
        records = self.events([
            {"kind": "start", "run_id": "one", "charges": {worker: 2400, owned: 2460, "tokens": 0}},
            {"kind": "charge", "charges": {"tokens": 77}},
            {"kind": "settle", "run_id": "one", "charges": {}, "actual_wall": {worker: 101, owned: 105}},
        ])
        state, started, settled = c.replay_events(list(records), records.__getitem__, {worker: 2500, owned: 2700, "tokens": 100})
        self.assertEqual(state["used_or_reserved"], {worker: 101, owned: 105, "tokens": 77})
        self.assertEqual(started, {"one"}); self.assertEqual(set(settled), {"one"})

    def test_active_worker_retains_full_reservation(self):
        worker, owned = c.WALL_KEYS
        records = self.events([{"kind": "start", "run_id": "one", "charges": {worker: 2400, owned: 2460}}])
        state, _, _ = c.replay_events(list(records), records.__getitem__, {worker: 2500, owned: 2700})
        self.assertEqual(state["used_or_reserved"][worker], 2400)
        self.assertEqual(state["active_reservations"], ["one"])

    def test_reused_native_run_id_is_rejected_after_settlement(self):
        worker, owned = c.WALL_KEYS
        start = {"kind": "start", "run_id": "one", "charges": {worker: 100, owned: 100}}
        records = self.events([start, {"kind": "settle", "run_id": "one", "charges": {}, "actual_wall": {worker: 1, owned: 1}}, start])
        with self.assertRaisesRegex(ValueError, "reused"):
            c.replay_events(list(records), records.__getitem__, {worker: 2500, owned: 2700})

    def test_invalid_or_missing_supervisor_retains_900_seconds(self):
        for value in (None, {}, {"status": "EXITED", "returncode": 1, "wall_seconds": float("nan")},
                      {"status": "RUNNING", "returncode": 0, "wall_seconds": 1}):
            self.assertEqual(c.observed_worker_charge(value)[0], 900)
        self.assertEqual(c.observed_worker_charge({"status": "EXITED", "returncode": 1, "wall_seconds": 27.5})[0], 27.5)

    def test_resources_remain_sampled_telemetry(self):
        value = c.resource_summary([{"available_host_bytes": 9, "rss_bytes": 3,
            "gpu0": {"free": 8, "allocated": 2, "reserved": 4, "peak": 5}}])
        self.assertEqual(value["maximum_recorded_torch_peak_allocated_bytes"], 5)
        self.assertIn("not a device-wide", value["coverage"])
        self.assertNotIn("measured_gpu_seconds", value)


class ProbeCostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.envelope = {"probe_workers_all_attempts": 6, "probe_retries_total": 1,
                         "probe_worker_seconds_all_attempts": 5400, "probe_rows_reserved_all_attempts": 2000,
                         "probe_generation_tokens_reserved_all_attempts": 170000, "total_owned_work_wall_seconds": 12600}
        self.receipts = []
        for leg in c.LEGS:
            self.receipts.append(self.attempt(leg))
        self.terminal()

    def tearDown(self):
        self.temp.cleanup()

    def attempt(self, leg, *, retry=False, success=True):
        key = "mixed-" + leg; name = key + ("-retry1" if retry else "")
        out = self.root / "eval" / name; worker = out if leg == "binding" else out / "worker"
        supervisor = {"status": "EXITED", "returncode": 0 if success else 1, "wall_seconds": 10.0, "child_pid": 44}
        write(worker / "supervisor.json", supervisor)
        launch = {"out": str(worker), "timeout_seconds": 900, "supervisor_pid": 22}
        write(worker / "launch.json", launch)
        write(worker / "spawn.json", {"launch_sha256": c.digest(launch), "supervisor_pid": 22, "child_pid": 44})
        write(worker / "worker_claim.json", {"launch_sha256": c.digest(launch), "supervisor_pid": 22, "pid": 44})
        term = "worker_terminal.json" if leg == "binding" else "probe_terminal.json"
        write(worker / term, {"status": ("COMPLETE" if leg == "binding" else "COMPLETED") if success else "FAILED",
                              "complete": success, "failure": None if success else {"message": "synthetic"}})
        if success:
            binding = {"status": "COMPLETE", "complete": True, "supervisor_sha256": c.sha(worker / "supervisor.json"),
                       "controller_returncode" if leg == "binding" else "returncode": 0,
                       "worker_terminal_sha256" if leg == "binding" else "terminal_sha256": c.sha(worker / term)}
            write(out / ("BINDING.json" if leg == "binding" else "PROBE_WEIGHT_BINDING.json"), binding)
        n, maximum = c.LEGS[leg]
        receipt = {"leg": key, "out": str(out), "returncode": 0 if success else 1, "wall_seconds": 12,
                   "worker_seconds_charged": 10.0, "reserved_rows": n, "reserved_generation_tokens": n * maximum}
        write(self.root / "posttrain" / (name + ".json"), receipt)
        return receipt

    def terminal(self):
        value = {"status": "COMPLETE_FIVE_PROBES", "attempts": self.receipts,
                 "probe_worker_seconds": 50, "probe_rows_reserved": 1558,
                 "probe_generation_tokens_reserved": 117264, "session_elapsed_seconds": 1000}
        write(self.root / "posttrain/TERMINAL.json", value)

    def test_all_five_reservations_reproduce_registered_totals(self):
        value = c.probe_costs(self.root, self.envelope, read)
        self.assertEqual(value["totals"]["rows_reserved"], 1558)
        self.assertEqual(value["totals"]["generation_tokens_reserved"], 117264)
        self.assertIsNone(value["measured_gpu_active_seconds"])
        self.assertIsNone(value["actual_generated_tokens"])

    def test_unreceipted_attempt_directory_cannot_be_ignored(self):
        (self.root / "eval/mixed-dose-retry1").mkdir()
        with self.assertRaisesRegex(ValueError, "no cost receipt"):
            c.probe_costs(self.root, self.envelope, read)

    def test_global_generation_reservation_ceiling_is_enforced(self):
        self.envelope["probe_generation_tokens_reserved_all_attempts"] = 117263
        with self.assertRaisesRegex(ValueError, "global five-leg"):
            c.probe_costs(self.root, self.envelope, read)

    def test_complete_terminal_cannot_omit_attempt(self):
        self.receipts.pop(); self.terminal()
        with self.assertRaisesRegex(ValueError, "omitted an attempt"):
            c.probe_costs(self.root, self.envelope, read)

    def test_blocked_probe_completion_keeps_failed_charges_visible(self):
        (self.root / "eval/mixed-binding/BINDING.json").unlink()
        self.attempt("binding", success=False)
        partial = {}
        with self.assertRaisesRegex(ValueError, "five unique successes"):
            c.probe_costs(self.root, self.envelope, read, partial=partial)
        self.assertEqual(partial["status"], "PARTIAL_NOT_COMPLETION_VERIFIED")
        self.assertEqual(partial["observed_receipt_totals"]["reported_failed_attempts"], 1)
        self.assertEqual(partial["observed_receipt_totals"]["worker_seconds_charged"], 50)

    def test_unobserved_failed_probe_retains_full_charge_in_blocked_output(self):
        (self.root / "eval/mixed-binding/BINDING.json").unlink()
        failed = self.attempt("binding", success=False)
        (self.root / "eval/mixed-binding/supervisor.json").unlink()
        failed["worker_seconds_charged"] = 900
        write(self.root / "posttrain/mixed-binding.json", failed)
        partial = {}
        with self.assertRaises(ValueError):
            c.probe_costs(self.root, self.envelope, read, partial=partial)
        self.assertEqual(partial["observed_receipt_totals"]["worker_seconds_charged"], 940)

    def test_original_failed_attempt_remains_charged_after_recovery(self):
        # Replace the synthetic first binding success with a retained failure.
        (self.root / "eval/mixed-binding/BINDING.json").unlink()
        failed = self.attempt("binding", success=False)
        original = {"status": "FAILED_OR_BLOCKED", "attempts": [*self.receipts[:-1], failed]}
        write(self.root / "posttrain/TERMINAL.json", original)
        retry = self.attempt("binding", retry=True)
        write(self.root / "ATTEMPT_MAP.json", {"schema": 1, "replacements": {"mixed-binding": "eval/mixed-binding-retry1"}})
        bindings = {"mixed-" + leg: c.sha(self.root / "eval" / ("mixed-" + leg + ("-retry1" if leg == "binding" else "")) /
                                        ("BINDING.json" if leg == "binding" else "PROBE_WEIGHT_BINDING.json")) for leg in c.LEGS}
        write(self.root / "posttrain/RECOVERY_COMPLETE.json", {"status": "COMPLETE_FIVE_PROBES",
            "prior_terminal_sha256": c.sha(self.root / "posttrain/TERMINAL.json"), "attempt_map_sha256": c.sha(self.root / "ATTEMPT_MAP.json"),
            "final_binding_sha256": bindings, "probe_worker_seconds": 60, "probe_rows_reserved": 1942,
            "probe_generation_tokens_reserved": 126480, "session_elapsed_seconds": 1100})
        value = c.probe_costs(self.root, self.envelope, read)
        self.assertEqual(value["totals"]["attempts"], 6)
        self.assertEqual(value["totals"]["failed_attempts"], 1)
        self.assertEqual(value["totals"]["worker_seconds_charged"], 60)
        changed = {**failed, "wall_seconds": 13}
        write(self.root / "posttrain/mixed-binding.json", changed)
        with self.assertRaisesRegex(ValueError, "original attempt receipt changed"):
            c.probe_costs(self.root, self.envelope, read)


if __name__ == "__main__":
    unittest.main()
