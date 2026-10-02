"""CPU fault and competing-coordinator controls; no model/runtime is imported."""
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1.operations import (
    OperationsError, OperationsStore, EPOCH_SECONDS, ROLLING_SECONDS)


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = 100000.0
        self.store = OperationsStore(self.root / "host.sqlite3", self.root / "receipts",
                                     clock=lambda: self.now)
        self.witness = {"pid": 42, "creation_time": "test-fixture-only",
                        "command_hash": "a" * 64}
        self.reconciliation = {"host_process_check": "synthetic fixture: no worker",
                               "gpu_activity_check": "synthetic fixture: no CUDA",
                               "worker_status": "synthetic fixture: exited"}

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def job(self, key="job", epoch="epoch-1"):
        self.store.register_job(key, {"model": "CPU-fixture-only"}, epoch)
        self.store.ready(key)

    def lease(self, owner="coordinator-A"):
        return self.store.acquire_lease(owner, 100)

    def admitted(self, key="job", seconds=600, epoch="epoch-1"):
        self.job(key, epoch)
        lease = self.lease()
        self.store.reserve(key, lease["owner"], lease["fence"], seconds)
        return lease

    def resolve(self, key, outcome, seconds, evidence, *, stale=False):
        row = self.store.get_job(key)
        lease = self.store.db.execute("SELECT * FROM leases WHERE kind='model'").fetchone()
        if row["worker"] is None and row["state"] == "RESERVED":
            self.store.mark_running(key, lease["owner"], lease["fence"], self.witness)
            row = self.store.get_job(key)
        proof = {"observer": "CPU-fixture-only", "worker_identity": row["worker"],
                 "process_exited": True, "exit_observed_at": self.now,
                 "elapsed_native_seconds": seconds, "request_hash": row["request_hash"],
                 "owner": lease["owner"], "fence": lease["fence"]}
        extras = {}
        if stale:
            coordination = self.store.acquire_lease("reconciler", 100, kind="coordinator")
            proof["reconciliation_owner"] = "reconciler"
            extras = {"stale_reconciliation": True, "coordinator_owner": "reconciler",
                      "coordinator_fence": coordination["fence"]}
        ref = self.store.write_receipt("native_outcome_observation", key, proof)
        return self.store.reconcile(key, outcome, seconds,
                  {**evidence, "native_outcome_proof_ref": ref}, owner=lease["owner"],
                  fence=lease["fence"], **extras)

    def test_idempotency_survives_reopening_and_refuses_changed_request(self):
        self.job()
        self.store.close()
        self.store = OperationsStore(self.root / "host.sqlite3", self.root / "receipts",
                                     clock=lambda: self.now)
        self.assertEqual(self.store.register_job("job", {"model": "CPU-fixture-only"},
                                                "epoch-1")["state"], "READY")
        with self.assertRaises(OperationsError):
            self.store.register_job("job", {"model": "different"}, "epoch-1")
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0], 1)

    def test_dependencies_must_succeed_and_parent_graph_cannot_be_rewritten(self):
        self.store.register_job("a", {}, "e")
        self.store.register_job("b", {}, "e", ["a"])
        with self.assertRaises(OperationsError):
            self.store.ready("b")
        self.store.ready("a")
        self.store.finish_cpu("a", "SUCCEEDED", {"test_receipt": "fixture"})
        self.store.ready("b")
        with self.assertRaises(OperationsError):
            self.store.register_job("a", {}, "e", ["b"])
        with self.assertRaises(OperationsError):
            self.store.register_job("future-parent", {}, "e", ["does-not-exist"])

    def test_failed_dependency_keeps_dependent_unready(self):
        self.store.register_job("a", {}, "e")
        self.store.register_job("b", {}, "e", ["a"])
        self.store.finish_cpu("a", "FAILED", {"failure": "fixture"})
        with self.assertRaises(OperationsError):
            self.store.ready("b")

    def test_independent_connections_exclude_overlapping_coordinators(self):
        other = OperationsStore(self.root / "host.sqlite3", self.root / "receipts-2",
                                clock=lambda: self.now)
        try:
            first = self.store.acquire_lease("A", 30, kind="coordinator")
            with self.assertRaises(OperationsError):
                other.acquire_lease("B", 30, kind="coordinator")
            self.store.release_lease("A", first["fence"], kind="coordinator")
            second = other.acquire_lease("B", 30, kind="coordinator")
            self.assertGreater(second["fence"], first["fence"])
            with self.assertRaises(OperationsError):
                self.store.heartbeat_lease("A", first["fence"], 30, kind="coordinator")
        finally:
            other.close()

    def test_expired_lease_marks_unknown_and_never_auto_relaunches(self):
        lease = self.admitted()
        self.store.mark_running("job", lease["owner"], lease["fence"], self.witness)
        self.now += 101
        with self.assertRaises(OperationsError):
            self.store.acquire_lease("B", 100)
        self.assertEqual(self.store.get_job("job")["state"], "OUTCOME_UNKNOWN")
        with self.assertRaises(OperationsError):
            self.store.heartbeat_lease(lease["owner"], lease["fence"], 100)
        with self.assertRaises(OperationsError):
            self.store.reconcile_lease(lease["owner"], lease["fence"], self.reconciliation,
                                       processes_clear=True)
        self.resolve("job", "FAILED", 51, {"failure_receipt": "fixture"}, stale=True)
        self.store.reconcile_lease(lease["owner"], lease["fence"], self.reconciliation,
                                   processes_clear=True)
        fresh = self.store.acquire_lease("B", 100)
        self.assertGreater(fresh["fence"], lease["fence"])

    def test_no_process_evidence_cannot_reconcile_stale_lease(self):
        lease = self.lease()
        self.now += 101
        with self.assertRaises(OperationsError):
            self.store.reconcile_lease(lease["owner"], lease["fence"], {}, processes_clear=True)
        with self.assertRaises(OperationsError):
            self.store.reconcile_lease(lease["owner"], lease["fence"], self.reconciliation,
                                       processes_clear=False)

    def test_lost_launch_response_preserves_reservation_and_refuses_dispatch(self):
        lease = self.admitted()
        self.store.mark_unknown("job", "lost response")
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], 600)
        retried = self.store.reserve("job", lease["owner"], lease["fence"], 600)
        self.assertEqual(retried["state"], "OUTCOME_UNKNOWN")
        with self.assertRaises(OperationsError):
            self.store.mark_running("job", lease["owner"], lease["fence"], self.witness)
        with self.assertRaises(OperationsError):
            self.store.release_lease(lease["owner"], lease["fence"])

    def test_worker_identity_and_one_worker_guard(self):
        lease = self.admitted()
        with self.assertRaises(OperationsError):
            self.store.mark_running("job", lease["owner"], lease["fence"], {"pid": 42})
        self.job("second")
        with self.assertRaises(OperationsError):
            self.store.reserve("second", lease["owner"], lease["fence"], 1)

    def test_failures_count_and_epoch_budget_never_resets_on_restart(self):
        lease = self.admitted(seconds=EPOCH_SECONDS)
        self.resolve("job", "FAILED", EPOCH_SECONDS, {"oom": "fixture"})
        self.job("second")
        with self.assertRaises(OperationsError):
            self.store.reserve("second", lease["owner"], lease["fence"], 1)
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], EPOCH_SECONDS)

    def test_rolling_budget_spans_epochs_and_known_expired_costs_age_out(self):
        lease = self.lease()
        for key, epoch in [("a", "e-a"), ("b", "e-b")]:
            self.job(key, epoch)
            self.store.reserve(key, lease["owner"], lease["fence"], EPOCH_SECONDS)
            self.resolve(key, "FAILED", EPOCH_SECONDS, {"failure": key})
        self.job("c", "e-c")
        with self.assertRaises(OperationsError):
            self.store.reserve("c", lease["owner"], lease["fence"], 1)
        self.assertEqual(self.store.budget_snapshot("e-c")["rolling_charged_seconds"], ROLLING_SECONDS)
        self.now += 86401
        self.assertEqual(self.store.budget_snapshot("e-c")["rolling_charged_seconds"], 0)
        self.assertEqual(self.store.budget_snapshot("e-a")["epoch_charged_seconds"], EPOCH_SECONDS)

    def test_active_unknown_budget_does_not_age_out_with_wall_clock(self):
        self.admitted()
        self.store.mark_unknown("job", "orphan")
        self.now += 1000000
        self.assertEqual(self.store.budget_snapshot("epoch-1")["rolling_charged_seconds"], 600)

    def test_partial_known_failure_cost_and_unknown_failure_charge(self):
        lease = self.admitted()
        self.resolve("job", "FAILED", 73, {"failure_receipt": "fixture"})
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], 73)
        self.job("second")
        self.store.reserve("second", lease["owner"], lease["fence"], 100)
        self.resolve("second", "ABORTED", None, {"cost_unavailable": "fixture"})
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], 173)

    def test_overrun_is_preserved_and_blocks_further_budget(self):
        self.admitted(seconds=600)
        result = self.resolve("job", "FAILED", EPOCH_SECONDS + 1,
                                      {"watchdog_violation": "fixture"})
        self.assertTrue(result["result"]["reservation_exceeded"])
        self.assertGreater(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], EPOCH_SECONDS)

    def test_outcomes_are_immutable_and_cpu_cannot_erase_native_cost(self):
        self.admitted()
        with self.assertRaises(OperationsError):
            self.store.finish_cpu("job", "SUCCEEDED", {"fake": "fixture"})
        evidence = {"failure": "fixture"}
        self.resolve("job", "FAILED", 23, evidence)
        self.resolve("job", "FAILED", 23, evidence)
        with self.assertRaises(OperationsError):
            self.store.reconcile("job", "SUCCEEDED", 0, evidence)

    def test_success_requires_measured_cost(self):
        self.admitted()
        with self.assertRaises(OperationsError):
            self.store.reconcile("job", "SUCCEEDED", None, {"result": "fixture"})

    def test_receipt_idempotency_corruption_and_escape(self):
        ref = self.store.write_receipt("result", "../../key", {"answer": 7})
        self.assertEqual(ref, self.store.write_receipt("result", "../../key", {"answer": 7}))
        self.assertEqual(self.store.verify_receipt(ref)["payload"], {"answer": 7})
        self.assertEqual(Path(ref["path"]).parent, self.root / "receipts")
        with self.assertRaises(OperationsError):
            self.store.write_receipt("result", "../../key", {"answer": 8})
        Path(ref["path"]).write_text("broken")
        with self.assertRaises(OperationsError):
            self.store.verify_receipt(ref)
        outside = self.root / "outside.json"
        outside.write_text("{}")
        with self.assertRaises(OperationsError):
            self.store.verify_receipt({"path": str(outside), "sha256": hashlib.sha256(b"{}").hexdigest()})

    def test_interrupted_atomic_publication_keeps_partial_without_claimed_receipt(self):
        with patch("os.link", side_effect=OSError("simulated disk exhaustion")):
            with self.assertRaises(OperationsError):
                self.store.write_receipt("result", "a", {"cost": 3})
        self.assertEqual(len(list((self.root / "receipts").glob("*.json"))), 0)
        self.assertEqual(len(list((self.root / "receipts").glob(".partial-*"))), 1)
        ref = self.store.write_receipt("result", "a", {"cost": 3})
        self.assertEqual(self.store.verify_receipt(ref)["payload"], {"cost": 3})

    def test_nonfinite_budget_and_payload_rejected(self):
        for value in [float("nan"), float("inf"), -1, True, 0]:
            with self.assertRaises(OperationsError):
                self.store.acquire_lease("A", value)
        with self.assertRaises(ValueError):
            self.store.write_receipt("result", "a", {"nan": float("nan")})

    def test_corrupt_database_is_preserved_and_fails_closed(self):
        damaged = self.root / "damaged.sqlite3"
        damaged.write_bytes(b"corrupt original bytes")
        with self.assertRaises(OperationsError):
            OperationsStore(damaged, self.root / "failed-receipts")
        self.assertEqual(damaged.read_bytes(), b"corrupt original bytes")

    def test_unregistered_orphan_blocks_native_reservations_until_reconciled(self):
        orphan = self.store.orphan_worker(self.witness, "observed orphan fixture")
        self.assertEqual(orphan["state"], "OUTCOME_UNKNOWN")
        self.assertEqual(self.store.budget_snapshot("new")["rolling_charged_seconds"], ROLLING_SECONDS)
        lease = self.lease()
        self.job()
        with self.assertRaises(OperationsError):
            self.store.reserve("job", lease["owner"], lease["fence"], 1)
        self.resolve(orphan["key"], "ABORTED", 51, {"orphan_exited": "fixture"})
        self.store.reserve("job", lease["owner"], lease["fence"], 1)

    def test_unfenced_fake_zero_reconciliation_cannot_free_running_worker(self):
        lease = self.admitted()
        self.store.mark_running("job", lease["owner"], lease["fence"], self.witness)
        with self.assertRaises(OperationsError):
            self.store.reconcile("job", "FAILED", 0, {"note": "still running; no exit evidence"})
        self.assertEqual(self.store.get_job("job")["state"], "RUNNING")
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], 600)
        self.job("second")
        with self.assertRaises(OperationsError):
            self.store.reserve("second", lease["owner"], lease["fence"], 1)

    def test_observed_running_worker_zero_proof_is_rejected(self):
        lease = self.admitted()
        self.store.mark_running("job", lease["owner"], lease["fence"], self.witness)
        with self.assertRaises(OperationsError):
            self.resolve("job", "FAILED", 0, {"claimed_exit": "fixture"})
        self.assertEqual(self.store.get_job("job")["state"], "RUNNING")

    def test_reserved_proved_no_dispatch_may_release_zero_cost(self):
        lease = self.admitted()
        row = self.store.get_job("job")
        ref = self.store.write_receipt("native_outcome_observation", "job", {
            "observer": "CPU-fixture-only", "never_dispatched": True, "launch_attempted": False,
            "elapsed_native_seconds": 0, "request_hash": row["request_hash"],
            "owner": lease["owner"], "fence": lease["fence"]})
        self.store.reconcile("job", "ABORTED", 0, {"native_outcome_proof_ref": ref},
                             owner=lease["owner"], fence=lease["fence"])
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], 0)

    def test_null_stale_evidence_and_unowned_stale_reconciliation_rejected(self):
        lease = self.admitted()
        self.store.mark_running("job", lease["owner"], lease["fence"], self.witness)
        self.now += 101
        with self.assertRaises(OperationsError):
            self.store.acquire_lease("B", 100)
        nulls = {"host_process_check": None, "gpu_activity_check": None, "worker_status": None}
        with self.assertRaises(OperationsError):
            self.store.reconcile_lease(lease["owner"], lease["fence"], nulls, processes_clear=True)
        row = self.store.get_job("job")
        ref = self.store.write_receipt("native_outcome_observation", "job", {
            "observer": "CPU-fixture-only", "worker_identity": self.witness, "process_exited": True,
            "exit_observed_at": self.now, "elapsed_native_seconds": 51,
            "request_hash": row["request_hash"], "owner": lease["owner"], "fence": lease["fence"],
            "reconciliation_owner": "unleased claimant"})
        with self.assertRaises(OperationsError):
            self.store.reconcile("job", "FAILED", 51, {"native_outcome_proof_ref": ref},
                owner=lease["owner"], fence=lease["fence"], stale_reconciliation=True)

    def test_logical_database_mutations_fail_before_budget_or_ready(self):
        self.job()
        self.store.db.execute("UPDATE jobs SET payload='{}' WHERE key='job'")
        with self.assertRaises(OperationsError):
            self.store.get_job("job")
        self.job("other")
        self.store.register_job("dependent", {}, "e", ["other"])
        self.store.db.execute("UPDATE jobs SET state='SUCCEEDED' WHERE key='other'")
        with self.assertRaises(OperationsError):
            self.store.get_job("other")
        with self.assertRaises(OperationsError):
            self.store.ready("dependent")

    def test_identical_cpu_finish_retry_keeps_timestamp_and_event_count(self):
        self.job()
        first = self.store.finish_cpu("job", "SUCCEEDED", {"receipt": "fixture"})
        count = self.store.db.execute("SELECT count(*) FROM events").fetchone()[0]
        self.now += 100
        second = self.store.finish_cpu("job", "SUCCEEDED", {"receipt": "fixture"})
        self.assertEqual(first["finished"], second["finished"])
        self.assertEqual(count, self.store.db.execute("SELECT count(*) FROM events").fetchone()[0])

    def test_accounting_mutations_cannot_erase_verified_failed_cost(self):
        self.admitted()
        self.resolve("job", "FAILED", 23, {"failure": "CPU fixture"})
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], 23)
        mutations = {"actual_seconds": 0, "reserved_seconds": 0, "owner": "different", "fence": 999,
                     "reserved": 0, "created": 0, "worker": None, "finished": 0}
        original = self.store.db.execute("SELECT * FROM jobs WHERE key='job'").fetchone()
        for field, value in mutations.items():
            self.store.db.execute("UPDATE jobs SET " + field + "=? WHERE key='job'", (value,))
            with self.assertRaises(OperationsError, msg=field):
                self.store.budget_snapshot("epoch-1")
            self.store.db.execute("UPDATE jobs SET " + field + "=? WHERE key='job'", (original[field],))
        self.assertEqual(self.store.budget_snapshot("epoch-1")["epoch_charged_seconds"], 23)


if __name__ == "__main__":
    unittest.main()
