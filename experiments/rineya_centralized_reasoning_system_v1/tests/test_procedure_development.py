"""Supplied lifecycle, evaluator boundary and restart controls; native calls0."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import procedure_development as dev
from experiments.rineya_centralized_reasoning_system_v1 import runner
from experiments.rineya_centralized_reasoning_system_v1.core import ContractError, ProcedureSnapshot
from experiments.rineya_centralized_reasoning_system_v1.data import PUBLIC_FIELDS
from experiments.rineya_centralized_reasoning_system_v1.operations import OperationsError, OperationsStore


class ProcedureDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = OperationsStore(self.root / "state.sqlite3", self.root / "receipts")

    def tearDown(self):
        self.store.close(); self.tmp.cleanup()

    def test_supplied_ancestry_and_restricted_graph_only(self):
        first, second = dev.supplied_candidates()
        dev.validate_candidate(first)
        dev.validate_candidate(second, parent=first)
        bad = copy.deepcopy(first)
        bad["graph"][1]["op"] = "SHELL"
        with self.assertRaises(ContractError):
            dev.validate_candidate(bad)
        bad = copy.deepcopy(second); bad["parent_sha256"] = "0" * 64
        with self.assertRaises(dev.DevelopmentError):
            dev.validate_candidate(bad, parent=first)
        bad = copy.deepcopy(first); bad["python"] = "arbitrary code"
        with self.assertRaises(dev.DevelopmentError):
            dev.validate_candidate(bad)

    def test_unidentified_or_forged_model_origin_cannot_claim_native_discovery(self):
        candidate = dev.supplied_candidates()[0]
        candidate["origin"] = "MODEL_PROPOSED"
        with self.assertRaisesRegex(dev.DevelopmentError, "actual native"):
            dev.validate_candidate(candidate, store=self.store)
        candidate["proposal_evidence"] = {
            "proposal_receipt": self.store.write_receipt("fake-model", "proposal", {"status": "MODEL_PROPOSED"}),
            "qualification_receipt": self.store.write_receipt("fake-qualifier", "qualification", {"status": "APPROVED"})}
        with self.assertRaisesRegex(dev.DevelopmentError, "adapter unregistered"):
            dev.validate_candidate(candidate, store=self.store)

    def test_count_bound_and_no_confirmation_role(self):
        for count in (0, 1, 8, 17, 33, 64, True):
            with self.assertRaises(dev.DevelopmentError):
                dev.run_development(self.store, count=count)

    def test_complete_public_lifecycle_retains_counterexamples_and_freezes_supplied_revision(self):
        original = dev.execute_case
        calls = []
        def public_boundary(store, key, case, arm, snapshot, epoch):
            self.assertEqual(set(case), set(PUBLIC_FIELDS))
            self.assertEqual(case["split"], "DEV")
            self.assertNotIn("gold", case)
            self.assertNotIn("scoring_payload", case)
            self.assertEqual(arm, "C")
            calls.append(key)
            return original(store, key, case, arm, snapshot, epoch)
        with patch.object(dev, "execute_case", side_effect=public_boundary):
            result = dev.run_development(self.store, count=16)
        self.assertEqual(result["status"], "CPU_SUPPLIED_LIFECYCLE_COMPLETED")
        self.assertEqual(len(calls), 32)
        self.assertEqual(result["planned_cases"], result["accounted_case_receipts"])
        first, second = result["trials"]
        self.assertFalse(first["acceptance"])
        self.assertGreater(first["valid_semantic_counterexamples"], 0)
        self.assertTrue(second["acceptance"])
        failures = self.store.verify_receipt(first["counterexample_receipt"])["payload"]
        self.assertTrue(failures["counterexamples"])
        self.assertEqual(failures["scheduled_cases"], 16)
        frozen = self.store.verify_receipt(result["frozen_snapshot_receipt"])["payload"]
        snapshot = ProcedureSnapshot(frozen["entries"])
        self.assertEqual(snapshot.sha256, frozen["snapshot_sha256"])
        self.assertEqual(snapshot.entries()[0]["version"], 2)
        self.assertEqual(snapshot.entries()[0]["origin"], "CODEX_SUPPLIED")
        with self.assertRaises(ContractError):
            snapshot.add({"arbitrary": "write during evaluation"})
        previous = None
        for index, ref in enumerate(result["immutable_history_receipts"]):
            event = self.store.verify_receipt(ref)["payload"]
            self.assertEqual(event["index"], index)
            self.assertEqual(event["previous_history_sha256"], previous)
            self.assertEqual(event["proposer"]["kind"], "CODEX_ENGINEERING_AGENT")
            previous = ref["sha256"]
        self.assertEqual(result["history_tip_sha256"], previous)
        self.assertFalse(any(result["claims"].values()))
        self.assertEqual(result["cost"]["native_worker_seconds"], 0)
        self.assertEqual(result["cost"]["model_calls"], 0)
        with patch.object(dev, "execute_case", side_effect=AssertionError("duplicate worker launch")):
            cached = dev.run_development(self.store, count=16)
        self.assertEqual(result, cached)
        Path(first["counterexample_receipt"]["path"]).write_bytes(b"corrupt evidence")
        with patch.object(dev, "execute_case", side_effect=AssertionError("must not regenerate lost evidence")):
            with self.assertRaises(OperationsError):
                dev.run_development(self.store, count=16)

    def test_interrupted_controller_reuses_finished_case_without_duplicate_execution(self):
        original_execute, original_worker = dev.execute_case, runner.worker_request
        seen = []
        stopped = False
        def stop_after_one(store, key, case, arm, snapshot, epoch):
            nonlocal stopped
            if len(seen) == 1 and not stopped:
                stopped = True
                raise RuntimeError("CPU fixture controller interruption before next launch")
            return original_execute(store, key, case, arm, snapshot, epoch)
        def observed_worker(case, arm, snapshot):
            seen.append((case["id"], snapshot.sha256))
            return original_worker(case, arm, snapshot)
        with patch.object(runner, "worker_request", side_effect=observed_worker):
            with patch.object(dev, "execute_case", side_effect=stop_after_one):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    dev.run_development(self.store, count=16, seed=202610019)
            result = dev.run_development(self.store, count=16, seed=202610019)
        self.assertEqual(result["status"], "CPU_SUPPLIED_LIFECYCLE_COMPLETED")
        self.assertEqual(len(seen), 32)
        self.assertEqual(len(set(seen)), 32)
        self.assertTrue(list((self.root / "receipts").glob("procedure-development-failure-*.json")))

    def test_operational_failure_alone_is_not_a_graph_counterexample(self):
        original_worker = runner.worker_request
        def unavailable_first_candidate(case, arm, snapshot):
            if snapshot.entries()[0]["version"] == 1:
                return {"case_id": case["id"], "arm": arm, "error": "CPU unavailable-worker fixture",
                        "stop_reason": "WORKER_UNAVAILABLE", "worker_wall_seconds": 0}
            return original_worker(case, arm, snapshot)
        with patch.object(runner, "worker_request", side_effect=unavailable_first_candidate):
            result = dev.run_development(self.store, count=16, seed=202610022)
        self.assertEqual(result["status"], "INCONCLUSIVE_LIFECYCLE")
        self.assertEqual(result["trials"][0]["status"], "QUARANTINED_OPERATIONAL")
        self.assertEqual(result["trials"][0]["valid_semantic_counterexamples"], 0)
        self.assertTrue(result["trials"][1]["acceptance"])
        self.assertIsNone(result["frozen_snapshot_receipt"])
        self.assertEqual(result["accounted_case_receipts"], 32)


if __name__ == "__main__":
    unittest.main()
