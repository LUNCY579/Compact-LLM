"""CPU rejection contracts only: no checkpoint, CUDA or model is loaded."""
import copy
import json
import math
import hashlib
import struct
import os
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import native
from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting
from experiments.rineya_centralized_reasoning_system_v1.operations import OperationsStore


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(native.CONTRACT.read_text())
        self.now = 1000
        self.sample = {"sampled_at": self.now, "host_available_gib": 4,
                       "gpu_free_gib": 5, "disk_free_gib": 10,
                       "unresolved_gpu_pids": [], "competing_gpu_pids": [],
                       "unclassified_gpu_pids": [],
                       "process_inventory_available": True, "gpu_uuid": self.cfg["gpu_uuid"]}

    def test_missing_and_invalid_telemetry_never_opens_admission(self):
        self.assertTrue(native.admission_reasons(self.cfg, {}, current_time=self.now))
        for field in ("host_available_gib", "gpu_free_gib", "disk_free_gib"):
            for invalid in (None, float("nan"), float("inf"), True):
                sample = dict(self.sample, **{field: invalid})
                self.assertTrue(native.admission_reasons(self.cfg, sample, current_time=self.now))

    def test_resource_floor_unknown_process_and_stale_sample_rejected(self):
        changes = ({"host_available_gib": 3.3}, {"gpu_free_gib": 3.99},
                   {"disk_free_gib": 3.99}, {"unresolved_gpu_pids": [42]},
                   {"unclassified_gpu_pids": [12345]},
                   {"competing_gpu_pids": [42]}, {"process_inventory_available": False},
                   {"sampled_at": 994}, {"sampled_at": 1001}, {"gpu_uuid": "other"})
        for change in changes:
            self.assertTrue(native.admission_reasons(self.cfg, dict(self.sample, **change), current_time=self.now))

    def test_postimport_growth_budget_rejects_false_ambient_fit(self):
        sample = dict(self.sample, host_available_gib=1.5)
        self.assertTrue(native.admission_reasons(self.cfg, sample, current_time=self.now, postimport_rss_gib=.7))
        self.assertTrue(native.admission_reasons(self.cfg, self.sample, current_time=self.now, postimport_rss_gib=2.01))

    def test_source_drift_is_blocked_before_framework_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.py"; source.write_text("original")
            cfg = {"source_sha256": {}, "runtime_file_sha256": {str(source): native.sha(source)}}
            source.write_text("changed")
            with self.assertRaises(native.NativeBlocked):
                native.verify_sources(cfg)

    def test_model_asset_identity_not_name_or_existence(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.safetensors"; source.write_bytes(b"CPU fixture, not a checkpoint")
            cfg = {"model_dir": tmp, "asset_sha256": {"model.safetensors": native.sha(source)}}
            source.write_bytes(b"different fixture")
            with self.assertRaises(native.NativeBlocked):
                native.verify_assets(cfg)

    def test_builder_and_stale_review_cannot_approve_native_work(self):
        snapshot = {"commit": "a" * 40}
        base = {"status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC", "source_commit": snapshot["commit"],
                "contract_sha256": "b" * 64, "native_source_sha256": "c" * 64, "reviewer": "runtime_audit"}
        self.assertTrue(native.review_reasons(base, "b" * 64, snapshot, "c" * 64))
        base["reviewer"] = "/root/runtime_audit"
        self.assertTrue(native.review_reasons(base, "b" * 64, snapshot, "c" * 64))
        base.update(reviewer="independent fixture reviewer", source_commit="stale")
        self.assertTrue(native.review_reasons(base, "b" * 64, snapshot, "c" * 64))

    def test_contract_change_cannot_silently_expand_limits(self):
        with self.assertRaises(native.NativeBlocked):
            native.checked_contract("0" * 64)
        cfg = copy.deepcopy(self.cfg)
        cfg["limits"]["worker_rss_gib"] = 20
        with patch.object(native, "sha", return_value="f" * 64), patch.object(native, "read_json", return_value=cfg):
            with self.assertRaises(native.NativeBlocked):
                native.checked_contract("f" * 64)

    def test_exact_causal_mutations_protect_prefix_and_keep_dose(self):
        rows = native.causal_plan(self.cfg["prompt_ids"], self.cfg["generated_ids"])
        self.assertEqual(sum(len(r["ids"]) for r in rows), 524)
        full = rows[0]["ids"]
        for row, start in zip(rows[1:], (128, 129, 130)):
            self.assertEqual(row["ids"][:start], full[:start])
            self.assertTrue(all(a != b for a, b in zip(row["ids"][start:], full[start:])))
        with self.assertRaises(native.NativeBlocked):
            native.causal_plan(self.cfg["prompt_ids"], [17])

    def test_failed_numerics_and_nan_do_not_become_qualification(self):
        measured = native.numerical_decision(.01, .001, bitwise=False)
        self.assertFalse(measured["passed"])
        self.assertFalse(measured["capture_qualified"])
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(native.NativeBlocked):
                native.numerical_decision(value, 0, bitwise=True)

    def test_signed_zero_numeric_equality_is_not_raw_bitwise_identity(self):
        positive, negative = struct.pack("<f", 0.0), struct.pack("<f", -0.0)
        self.assertEqual(0.0, -0.0)
        measured = native.numerical_decision(0, 0, bitwise=True,
            left_sha256=hashlib.sha256(positive).hexdigest(), right_sha256=hashlib.sha256(negative).hexdigest())
        self.assertFalse(measured["passed"])
        self.assertFalse(measured["raw_byte_identity"])

    def test_complete_mask_representation_replays_registered_raw_bytes(self):
        raw = native.causal_mask_reference_bytes()
        self.assertEqual(len(raw), 131 * 131 * 2)
        words = struct.unpack("<17161H", raw)
        self.assertEqual(sum(word == 0 for word in words), 8646)
        self.assertEqual(sum(word == 65407 for word in words), 8515)
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 65407)

    def test_unclassified_gpu_executable_is_blocked_even_without_python_name(self):
        outputs = [self.cfg["gpu_uuid"] + ",8192,90", "12345,unclassified_worker.exe"]
        with patch.object(native.subprocess, "check_output", side_effect=outputs):
            sample = native.gpu_snapshot()
        self.assertEqual(sample["unclassified_gpu_pids"], [12345])
        sample.update(host_available_gib=4, disk_free_gib=10, process_inventory_available=True)
        self.assertIn("unregistered GPU executable identity", native.admission_reasons(self.cfg, sample))

    @unittest.skipUnless(os.name == "nt", "actual Windows containment fixture")
    def test_actual_suspended_windows_parent_and_children_share_owned_job_and_exit(self):
        result = native.containment_cpu_smoke(self.cfg)
        self.assertEqual(result["status"], "PASS_CONTAINMENT_ONLY")
        self.assertEqual(len(result["before_resume"]), 1)
        self.assertEqual(result["resume"]["prior_suspend_count"], 1)
        self.assertGreaterEqual(len(result["runtime_tree"]), 4)
        self.assertTrue(all(p["exact_owned_job_member"] for p in result["runtime_tree"]))
        self.assertTrue(result["exit"]["owned_tree_exited"])
        self.assertEqual(result["exit"]["remaining_owned_pids"], [])
        self.assertEqual(result["model_loads"], 0)
        self.assertEqual(result["framework_imports"], 0)
        self.assertEqual(result["gpu_calls"], 0)

    @unittest.skipUnless(os.name == "nt", "actual Windows worker rejection boundary")
    def test_direct_worker_go_marker_cannot_replace_supervised_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / "results" / "direct-worker-fixture"
            output.mkdir(parents=True)
            go = output / "GO.json"
            go.write_text(json.dumps({"contract_sha256": "a" * 64, "owner": "fixture", "fence": 1,
                                      "supervisor_pid": os.getpid(), "worker_tree_pid": os.getpid()}))
            args = SimpleNamespace(output=output, go_path=go, host_db=root / "state.sqlite3",
                                   contract_sha256="a" * 64, owner="fixture", fence=1)
            with patch.object(native, "PACKAGE", root), patch.object(native, "host_state_path", return_value=args.host_db):
                with self.assertRaisesRegex(native.NativeBlocked, "ancestry"):
                    native.worker(args, self.cfg)
            self.assertFalse(args.host_db.exists())

    def test_positive_gpu_identity_requires_unchanged_executable_and_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "fixture.exe"; dependency = Path(tmp) / "fixture.dll"
            exe.write_bytes(b"CPU registry fixture only"); dependency.write_bytes(b"original engine fixture")
            registry = [{"path": str(exe), "sha256": native.sha(exe), "role": "CPU fixture",
                         "dependency_sha256": {str(dependency): native.sha(dependency)}}]
            dependency.write_bytes(b"different engine fixture")
            outputs = [self.cfg["gpu_uuid"] + ",8192,90", "12345," + str(exe)]
            with patch.object(native.subprocess, "check_output", side_effect=outputs):
                sample = native.gpu_snapshot(desktop_identities=registry)
            self.assertEqual(sample["unclassified_gpu_pids"], [12345])

    def test_slow_gpu_identity_checks_cannot_refresh_old_gpu_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "graphical-fixture.exe"; exe.write_bytes(b"CPU timing fixture")
            registry = [{"path": str(exe), "sha256": "a" * 64, "role": "CPU fixture"}]
            outputs = [self.cfg["gpu_uuid"] + ",8192,90", "12345," + str(exe)]
            with patch.object(native.subprocess, "check_output", side_effect=outputs), \
                 patch.object(native, "sha", return_value="a" * 64), \
                 patch.object(native.time, "time", side_effect=[1000, 1000.1, 1007]):
                sample = native.gpu_snapshot(desktop_identities=registry)
            self.assertEqual(sample["sampled_at"], 1000)
            self.assertEqual(sample["identity_checks_completed_at"], 1007)
            sample.update(host_available_gib=4, disk_free_gib=10, process_inventory_available=True)
            self.assertIn("telemetry stale or future-dated", native.admission_reasons(self.cfg, sample, current_time=1007))

    def test_oom_attempt_is_accounted_and_failed_not_zero_work(self):
        events = []
        accounting = CallAccounting(lambda kind, **detail: events.append({"kind": kind, **detail}),
                                    maximum_optimizer_calls=0)
        def fixture_oom():
            raise MemoryError("CPU fault fixture; no CUDA operation")
        with self.assertRaises(MemoryError):
            accounting.call("forward", fixture_oom, stage="fixture", input_positions=131)
        counts = accounting.snapshot()
        self.assertEqual(counts["operations"]["forward"]["attempted"], 1)
        self.assertEqual(counts["operations"]["forward"]["failed"], 1)
        self.assertEqual(counts["forward_input_positions_attempted"], 131)
        self.assertEqual(counts["forward_input_positions_returned"], 0)
        self.assertEqual([e["kind"] for e in events], ["operation_attempt", "operation_failed"])

    def test_reader512_plus64_and_training_stay_unqualified(self):
        request = {"input_ids": [16] * 512, "attention_mask": [1] * 512, "max_new_tokens": 64}
        with self.assertRaises(native.NativeBlocked):
            native.validate_reader_request(request)
        with self.assertRaises(native.NativeBlocked):
            native.validate_reader_request(request, {"status": "PASS", "scope": "D2"})
        self.assertFalse(native.training_eligibility()["eligible"])

    def test_evaluator_fields_and_unsupported_shapes_rejected_by_reader_interface(self):
        # A fabricated dictionary here exercises syntax rejection only, not admission.
        hypothetical = {"scope": "CRS_READER_576_V1", "status": "INDEPENDENTLY_VERIFIED_NATIVE_PASS"}
        base = {"input_ids": [16], "attention_mask": [1], "max_new_tokens": 64}
        for request in ({**base, "gold": "leak"}, {**base, "max_new_tokens": 65},
                        {**base, "attention_mask": [0]}, {**base, "input_ids": [True]}):
            with self.assertRaises(native.NativeBlocked):
                native.validate_token_fields(request)

    def test_valid_token_fields_are_only_boundary_validation_and_cannot_open_reader(self):
        request = {"input_ids": [16], "attention_mask": [1], "max_new_tokens": 64}
        validated = native.validate_token_fields(request)
        self.assertEqual(validated, request)
        self.assertIsNot(validated["input_ids"], request["input_ids"])
        with self.assertRaises(native.NativeBlocked):
            native.validate_reader_request(request, {"scope": "CRS_READER_576_V1",
                                                     "status": "INDEPENDENTLY_VERIFIED_NATIVE_PASS"})

    def test_missing_preflight_registration_returns_blocked_with_zero_launches(self):
        report = native.preflight()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["native_launches_this_preflight"], 0)
        self.assertFalse(report["reader_qualified"])

    def launch_failure_fixture(self, *, before_attempt):
        """Actual supervisor/SQLite flow, with subprocess creation replaced by fault.

        Only preflight telemetry is a named CPU fixture; no model subprocess,
        containment pass or native qualification is manufactured.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "results" / "fixture"
            db = root / "host.sqlite3"
            args = SimpleNamespace(host_db=db, contract_sha256="f" * 64,
                      review=root / "review-fixture.json", output=out,
                      job_key="cpu-launch-failure-fixture", epoch="fixture-epoch")
            report = {"status": "READY_DIAGNOSTIC_ONLY", "reasons": [],
                      "observations": {"source": {"commit": "a" * 40}}}
            sample = dict(self.sample, sampled_at=native.time.time())
            original_write = native.write_new
            def write_fault(path, value):
                if before_attempt and Path(path).name == "LAUNCH_INTENT.json":
                    raise OSError("CPU fixture: disk exhaustion before any spawn attempt")
                return original_write(path, value)
            with patch.object(native, "PACKAGE", root), patch.object(native, "host_state_path", return_value=db), \
                 patch.object(native, "preflight", return_value=report), patch.object(native, "collect_telemetry", return_value=sample), \
                 patch.object(native.time, "sleep"), patch.object(native, "write_new", side_effect=write_fault), \
                 patch.object(native.subprocess, "Popen", side_effect=OSError("CPU fixture: lost creation response")) as spawn:
                self.assertEqual(native.supervise(args, self.cfg), 2)
            store = OperationsStore(db, out / "receipts")
            try:
                state = store.get_job(args.job_key)
                budget = store.budget_snapshot(args.epoch)
                receipt = native.read_json(out / "SUPERVISOR_RESULT.json")
                if before_attempt:
                    spawn.assert_not_called()
                    self.assertEqual(state["state"], "FAILED")
                    self.assertEqual(state["actual_seconds"], 0)
                    self.assertEqual(budget["epoch_charged_seconds"], 0)
                    self.assertFalse(receipt["launch_attempted"])
                else:
                    spawn.assert_called_once()
                    self.assertTrue((out / "LAUNCH_ATTEMPT.json").exists())
                    self.assertEqual(state["state"], "OUTCOME_UNKNOWN")
                    self.assertIsNone(state["actual_seconds"])
                    self.assertEqual(budget["epoch_charged_seconds"], 600)
                    self.assertTrue(receipt["subprocess_creation_outcome_unknown"])
                    self.assertFalse(list((out / "receipts").glob("native_outcome_observation-*.json")))
            finally:
                store.close()

    def test_popen_lost_response_keeps_unknown_outcome_and_full_reservation(self):
        self.launch_failure_fixture(before_attempt=False)

    def test_proved_failure_before_any_popen_attempt_may_account_zero(self):
        self.launch_failure_fixture(before_attempt=True)


if __name__ == "__main__":
    unittest.main()
