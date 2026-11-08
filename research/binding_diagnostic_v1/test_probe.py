"""Model-free failure-boundary tests for diagnostic admission and supervision."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_diagnostic_v1 import common as c, probe


class AdmissionFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / "eval"; self.out.mkdir()
        self.plan_path = self.root / "PLAN.json"
        self.grant_path = self.root / "GRANT.json"
        self.rows = [{"prompt_ids": [1, 2, 3]} for _ in range(c.NROWS)]
        input_path = self.root / "INPUTS.jsonl"
        input_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.plan = {"output_root": str(self.out), "python_path": sys.executable,
                     "input_path": str(input_path), "input_sha256": c.sha(input_path)}
        self.put(self.plan_path, self.plan)
        self.put(self.grant_path, {"plan_sha256": c.sha(self.plan_path)})

    def tearDown(self):
        self.temp.cleanup()

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def reservation(self, role="source", attempt=1, **changes):
        value = {"schema": 1, "model_role": role, "attempt": attempt,
                 "plan_sha256": c.sha(self.plan_path), "grant_sha256": c.sha(self.grant_path),
                 "reserved_worker_seconds": 900, "rows": c.NROWS,
                 "reserved_generated_tokens": c.NROWS * 24, "forward_calls": c.NROWS * 24,
                 "processed_input_tokens": c.NROWS * (3 + 24), **changes}
        lease_path = self.out / "CONTROLLER_LEASE.json"
        if lease_path.exists():
            value.update(controller_owner=c.read(lease_path), controller_lease_sha256=c.sha(lease_path))
        path = self.out / ("RESERVATION_" + c.slot_name(role, attempt) + ".json")
        self.put(path, value)
        return value

    def worker_identity(self, role="source", attempt=1, owner_pid=None):
        lease = c.read(self.out / "CONTROLLER_LEASE.json")
        owner = lease["owner_pid"] if owner_pid is None else owner_pid
        directory = self.out / c.slot_name(role, attempt)
        self.put(directory / "launch.json", {"supervisor_pid": owner})
        self.put(directory / "worker_claim.json", {"supervisor_pid": owner})
        reservation_path = self.out / ("RESERVATION_" + c.slot_name(role, attempt) + ".json")
        if reservation_path.exists():
            reservation = c.read(reservation_path)
            reservation.update(controller_owner=lease, controller_lease_sha256=c.sha(self.out / "CONTROLLER_LEASE.json"))
            self.put(reservation_path, reservation)
        return directory

    def verify(self, role="source", attempt=1):
        return c.verify_worker_reservation(self.plan, self.plan_path, self.grant_path, role, attempt, self.rows)


class ControllerLeaseTests(AdmissionFixture):
    def test_concurrent_controller_cannot_enter_same_global_admission(self):
        with c.controller_lease(self.plan, self.plan_path, self.grant_path):
            before = (self.out / "CONTROLLER_LEASE.json").read_bytes()
            def contender():
                with c.controller_lease(self.plan, self.plan_path, self.grant_path):
                    return "incorrectly admitted"
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(contender)
                with self.assertRaises(FileExistsError):
                    future.result()
            self.assertEqual((self.out / "CONTROLLER_LEASE.json").read_bytes(), before)
        self.assertFalse((self.out / "CONTROLLER_LEASE.json").exists())

    def test_stale_lease_is_preserved_without_claiming_or_deleting_it(self):
        path = self.out / "CONTROLLER_LEASE.json"
        self.put(path, {"owner_pid": 123, "nonce": "preexisting"})
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            with c.controller_lease(self.plan, self.plan_path, self.grant_path):
                self.fail("stale lease admitted")
        self.assertEqual(path.read_bytes(), original)

    def test_replaced_owner_lease_is_not_deleted_during_cleanup(self):
        path = self.out / "CONTROLLER_LEASE.json"
        replacement = {"owner_pid": 987, "nonce": "different-owner"}
        with self.assertRaisesRegex(ValueError, "ownership changed"):
            with c.controller_lease(self.plan, self.plan_path, self.grant_path):
                self.put(path, replacement)
        self.assertEqual(c.read(path), replacement)

    def test_controller_exception_releases_only_its_own_lease(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with c.controller_lease(self.plan, self.plan_path, self.grant_path):
                raise RuntimeError("synthetic failure")
        self.assertFalse((self.out / "CONTROLLER_LEASE.json").exists())

    def fake_completed_stage(self, *, replace_lease=False):
        directory = self.out / "source"; directory.mkdir()
        receipt = {"status": "COMPLETE", "complete": True, "controller_returncode": 0, "worker_returncode": 0}
        def stage(args, plan):
            self.assertTrue((self.out / "CONTROLLER_LEASE.json").exists())
            self.assertFalse((directory / "DIAGNOSTIC_BINDING.json").exists())
            if replace_lease:
                value = c.read(self.out / "CONTROLLER_LEASE.json")
                value["nonce"] = "different-owner"
                self.put(self.out / "CONTROLLER_LEASE.json", value)
            return directory, receipt
        return directory, stage

    def test_cleanup_failure_cannot_leave_a_complete_binding(self):
        directory, stage = self.fake_completed_stage(replace_lease=True)
        args = SimpleNamespace(plan=self.plan_path, grant=self.grant_path, model="source", attempt=1)
        with patch.object(c, "validate_grant"), patch.object(probe, "controller_admitted", side_effect=stage), \
             patch.object(c, "phase_guard") as phase:
            with self.assertRaisesRegex(ValueError, "ownership changed"):
                probe.controller(args, self.plan)
        self.assertFalse((directory / "DIAGNOSTIC_BINDING.json").exists())
        phase.assert_not_called()
        self.assertTrue((self.out / "CONTROLLER_LEASE.json").exists())

    def test_completion_is_published_after_owned_cleanup(self):
        directory, stage = self.fake_completed_stage()
        args = SimpleNamespace(plan=self.plan_path, grant=self.grant_path, model="source", attempt=1)
        def final_phase(*args, **kwargs):
            self.assertFalse((self.out / "CONTROLLER_LEASE.json").exists())
            self.assertFalse((directory / "DIAGNOSTIC_BINDING.json").exists())
        with patch.object(c, "validate_grant"), patch.object(probe, "controller_admitted", side_effect=stage), \
             patch.object(c, "phase_guard", side_effect=final_phase), patch("builtins.print"):
            self.assertEqual(probe.controller(args, self.plan), 0)
        self.assertEqual(c.read(directory / "DIAGNOSTIC_BINDING.json")["status"], "COMPLETE")

    def test_final_deadline_or_disk_refusal_cannot_publish_completion(self):
        directory, stage = self.fake_completed_stage()
        args = SimpleNamespace(plan=self.plan_path, grant=self.grant_path, model="source", attempt=1)
        with patch.object(c, "validate_grant"), patch.object(probe, "controller_admitted", side_effect=stage), \
             patch.object(c, "phase_guard", side_effect=ValueError("late phase bound exceeded")):
            with self.assertRaisesRegex(ValueError, "late phase"):
                probe.controller(args, self.plan)
        self.assertFalse((directory / "DIAGNOSTIC_BINDING.json").exists())
        self.assertFalse((self.out / "CONTROLLER_LEASE.json").exists())


class WorkerReservationTests(AdmissionFixture):
    def test_valid_reservation_is_charged_before_any_model_use(self):
        self.reservation()
        with c.controller_lease(self.plan, self.plan_path, self.grant_path):
            self.worker_identity()
            result = self.verify()
        self.assertEqual(result["totals"]["workers"], 1)
        self.assertEqual(result["totals"]["total_worker_seconds"], 900)
        self.assertEqual(result["totals"]["processed_input_tokens"], c.NROWS * 27)

    def test_private_worker_requires_its_exact_reservation(self):
        with c.controller_lease(self.plan, self.plan_path, self.grant_path):
            self.worker_identity()
            with self.assertRaises(ValueError):
                self.verify()
            for change in ({"rows": c.NROWS - 1}, {"processed_input_tokens": 0}, {"reserved_worker_seconds": 0},
                           {"forward_calls": c.NROWS * 24 - 1}, {"plan_sha256": "0" * 64}):
                with self.subTest(change=change):
                    self.reservation(**change)
                    with self.assertRaises(ValueError):
                        self.verify()

    def test_private_worker_requires_matching_controller_plan_and_grant(self):
        self.reservation()
        path = self.out / "CONTROLLER_LEASE.json"
        self.put(path, {"plan_sha256": "0" * 64, "grant_sha256": c.sha(self.grant_path), "owner_pid": 123})
        with self.assertRaises(ValueError):
            self.verify()

    def test_private_worker_cannot_borrow_another_supervisors_lease(self):
        self.reservation()
        with c.controller_lease(self.plan, self.plan_path, self.grant_path):
            self.worker_identity(owner_pid=999999)
            with self.assertRaisesRegex(ValueError, "lease|owner|supervisor"):
                self.verify()

    def test_global_retry_limit_is_not_per_model(self):
        self.reservation("source", 1)
        self.reservation("lora", 2)
        self.reservation("mixed", 2)
        with c.controller_lease(self.plan, self.plan_path, self.grant_path):
            self.worker_identity()
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                self.verify()

    def test_four_worker_reservations_exhaust_the_registered_geometry(self):
        for role, attempt in (("source", 1), ("lora", 1), ("mixed", 1), ("source", 2)):
            self.reservation(role, attempt)
        values = c.attempt_costs(self.plan, self.grant_path)["totals"]
        self.assertEqual((values["workers"], values["rows"], values["reserved_generated_tokens"], values["retries"]),
                         (4, 3584, 86016, 1))
        self.reservation("lora", 2)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            c.attempt_costs(self.plan, self.grant_path)

    def test_failed_observed_time_counts_and_missing_terminal_reserves_900(self):
        self.reservation("source", 1); self.reservation("lora", 1)
        self.put(self.out / "source/supervisor.json", {"status": "EXITED", "returncode": 2, "wall_seconds": 117.25})
        value = c.attempt_costs(self.plan, self.grant_path)["totals"]
        self.assertEqual(value["total_worker_seconds"], 1017.25)
        self.assertEqual(value["workers"], 2)
        self.put(self.out / "source/supervisor.json", {"status": "RUNNING", "returncode": None, "wall_seconds": 1})
        self.assertEqual(c.attempt_costs(self.plan, self.grant_path)["totals"]["total_worker_seconds"], 1800)

    def test_reservation_failure_precedes_cuda_guard_and_weight_hashing(self):
        directory = self.out / "source"; directory.mkdir()
        args = SimpleNamespace(out=directory, plan=self.plan_path, grant=self.grant_path, model="source", attempt=1)
        supervisor = ModuleType("experiments.model_quality_gpu_v1.supervisor")
        supervisor.claim_worker = lambda output: None
        resources = ModuleType("experiments.model_quality_gpu_v1.resources")
        resources.lease = lambda path: nullcontext()
        resources.guard = lambda **kwargs: self.fail("CUDA/resource guard ran before reservation refusal")
        psutil = ModuleType("psutil")
        psutil.virtual_memory = lambda: SimpleNamespace(available=8 * 2**30)
        modules = {supervisor.__name__: supervisor, resources.__name__: resources, "psutil": psutil}
        with patch.dict(sys.modules, modules), patch.object(probe, "setup_pilot"), \
             patch.object(c, "validate_grant"), patch.object(c, "phase_guard"), \
             patch.object(c, "read_jsonl", return_value=self.rows), patch.object(c, "validate_inputs", return_value=self.rows), \
             patch.object(c, "snapshot", side_effect=AssertionError("weight hashing ran before reservation refusal")), \
             patch.object(c, "verify_worker_reservation", side_effect=ValueError("missing reservation")), patch.object(c, "write_new"):
            with self.assertRaisesRegex(ValueError, "missing reservation"):
                probe.worker(args, self.plan)
        self.assertNotIn("torch", sys.modules)


class TerminalTests(unittest.TestCase):
    def test_completion_cannot_hide_supervisor_or_timing_failure(self):
        terminal = {"status": "COMPLETE", "complete": True, "failure": None, "before": {"id": 1}, "after": {"id": 1}}
        for supervisor in ({"status": "EXITED", "returncode": 1, "wall_seconds": 1},
                           {"status": "TIMEOUT_OR_INTERRUPTED", "returncode": 0, "wall_seconds": 1},
                           {"status": "EXITED", "returncode": 0, "wall_seconds": 900.1}):
            with self.subTest(supervisor=supervisor):
                with self.assertRaises(ValueError):
                    c.verify_terminal(supervisor, terminal)


class GoldFreeInputTests(unittest.TestCase):
    def rows(self):
        values = []
        for family, worlds, views in (("binding", 32, 16), ("arithmetic", 96, 4)):
            for world in range(worlds):
                for view in range(views):
                    values.append({"id": f"{family}-{world}-{view}", "parent_id": f"{family}-{world}",
                                   "prompt": [{"role": "system", "content": "Synthetic fixture"},
                                              {"role": "user", "content": "Synthetic fixture"}], "prompt_ids": [1, 2]})
        return values

    def test_exact_parent_geometry_and_no_gold_field(self):
        rows = self.rows()
        self.assertEqual(len(c.validate_inputs(rows)), 896)
        rows[0]["reference"] = "must remain outside worker inputs"
        with self.assertRaisesRegex(ValueError, "gold/category"):
            c.validate_inputs(rows)
        del rows[0]["reference"]
        rows[0]["parent_id"] = rows[16]["parent_id"]
        with self.assertRaisesRegex(ValueError, "requires32"):
            c.validate_inputs(rows)

    def test_context_reserve_prevents_implicit_truncation(self):
        rows = self.rows()
        rows[0]["prompt_ids"] = [1] * (2048 - 24)
        c.validate_inputs(rows)
        rows[0]["prompt_ids"].append(1)
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            c.validate_inputs(rows)


class GrantAdmissionTests(AdmissionFixture):
    def setUp(self):
        super().setUp()
        self.registration = self.root / "PREREG.md"
        self.registration.write_text("Synthetic registration fixture", encoding="utf-8")
        self.admission_path = self.root / "DATA_ADMISSION.json"
        self.admission = {
            "schema": 1, "record": "BINDING_DIAGNOSTIC_SCOPED_DATA_ADMISSION",
            "status": "ADMITTED_FOR_NAMED_EXPLORATORY_DIAGNOSTIC", "training_permission": False,
            "model_execution_permission": False, "mandatory_checks_passed": True,
            "policy_input_path": self.plan["input_path"], "policy_input_sha256": self.plan["input_sha256"],
            "evidence_sha256": {"DIAGNOSTIC_INPUTS.jsonl": self.plan["input_sha256"]},
            "frozen_models_weight_sha256": dict(c.MODEL_SHAS), "rows_per_model": 896,
            "binding_parent_worlds": 32, "arithmetic_parent_worlds": 96,
        }

    def grant(self):
        self.put(self.admission_path, self.admission)
        return {"schema": 1, "record": "BINDING_DIAGNOSTIC_GRANT", "experiment": c.EXPERIMENT,
                "decision": "APPROVED", "authorization_reference": "synthetic fixture",
                "plan_path": str(self.plan_path), "plan_sha256": c.sha(self.plan_path), "limits": dict(c.LIMITS),
                "registration_path": str(self.registration), "registration_sha256": c.sha(self.registration),
                "data_admission_path": str(self.admission_path), "data_admission_sha256": c.sha(self.admission_path),
                "session_start_unix": 1000, "deadline_unix": 8200}

    def test_grant_accepts_only_its_admitted_compiled_bank(self):
        c.validate_grant(self.plan_path, self.grant())

    def test_pinning_an_admission_receipt_does_not_admit_another_input_bank(self):
        self.admission["policy_input_sha256"] = "0" * 64
        self.admission["evidence_sha256"]["DIAGNOSTIC_INPUTS.jsonl"] = "0" * 64
        with self.assertRaises(ValueError):
            c.validate_grant(self.plan_path, self.grant())

    def test_admission_for_different_models_cannot_authorize_this_plan(self):
        self.admission["frozen_models_weight_sha256"]["mixed"] = "0" * 64
        with self.assertRaises(ValueError):
            c.validate_grant(self.plan_path, self.grant())


if __name__ == "__main__":
    unittest.main()
