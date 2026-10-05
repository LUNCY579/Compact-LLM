"""Independent injected CPU controls; no native or model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import fixture_support as fixtures
import pipeline


HERE = Path(__file__).resolve().parent
SUCCESS = "CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED"
FAILED = "CPU_PREFIX_INPUT_CHECKS_FAILED"
GUARDS = ["pre_import", "after_import", "after_growth", "before_context_init",
          "after_context_init", "before_input_construction", "after_input_validation", "complete"]
PREFIX_SHA = "e806d781c2c21c15180b6ec1bc714ed5acc7104948c2859ec6b411761a4a19d1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def returned(number=1):
    return {"attempted": number, "returned": number, "failed": 0, "pending": 0}


class PrefixInputControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="prefix_fixture_", dir=HERE)
        self.root = Path(self.directory.name).resolve()
        self.case = 0

    def tearDown(self):
        self.assertEqual(self.root.parent, HERE.resolve())
        self.directory.cleanup()

    def run_fixture(self, faults=None, guard=None):
        self.case += 1
        output = self.root / ("case%d" % self.case)
        output.mkdir()
        api = fixtures.InjectedAPI(faults)
        report = pipeline.run_cpu(api, api.guard if guard is None else guard, output)
        return report, api, output

    def assert_failed(self, report):
        self.assertEqual(report["status"], FAILED)
        self.assertFalse(report["native_fit_established"])
        self.assertFalse(report["native_launch_authorized"])
        self.assertFalse(report["M0_membership"])

    def test_success_exact_reads_guards_and_observed_phase_order(self):
        report, api, output = self.run_fixture()
        self.assertEqual(report["status"], SUCCESS)
        self.assertEqual(fixtures.check_reads(api.recorder), fixtures.EXPECTED_CHECK_READS)
        self.assertEqual(api.recorder.guards, GUARDS)
        self.assertEqual(api.recorder.counts["api.config_preflight"], 2)
        self.assertEqual(api.recorder.counts["cuda.is_initialized"], 8)
        self.assertEqual(api.recorder.counts["cuda.mem_get_info"], 6)
        records = [json.loads(line) for line in (output / "PHASE_OBSERVATIONS.jsonl").read_text().splitlines()]
        attempts = [record for record in records if record["event"] == "attempt"]
        self.assertEqual([record["ordinal"] for record in attempts], list(range(1, len(attempts) + 1)))
        self.assertLess(next(i for i, record in enumerate(attempts) if record["operation"] == "mask_ones_like_create"),
                        next(i for i, record in enumerate(attempts) if record["operation"] == "input_is_tensor"))
        self.assertEqual(report["operation_accounting"]["guard_cuda_is_initialized"], returned(7))
        self.assertEqual(report["operation_accounting"]["guard_cuda_mem_get_info"], returned(4))

    def test_success_dose_hashes_seed_and_durable_final_IO_are_closed(self):
        original_fsync = pipeline.os.fsync
        with patch.object(pipeline.os, "fsync", wraps=original_fsync) as fsync:
            report, api, output = self.run_fixture()
        self.assertEqual(report["status"], SUCCESS)
        summary = report["input_summary"]
        self.assertEqual(summary["prefix_ids_sha256"], PREFIX_SHA)
        self.assertEqual(summary["observed_ids_host_values_sha256"], digest(api.torch.tensor_arguments[0]["rows"]))
        self.assertEqual(summary["mask_values_sha256"], digest([[1] * 422]))
        self.assertEqual(summary["shape"], [1, 422])
        self.assertEqual(summary["mask_shape"], [1, 422])
        self.assertEqual(summary["second_ids_shape"], [1, 422])
        self.assertEqual(summary["logical_tensor_bytes"], 6752)
        self.assertTrue(summary["logical_bytes_not_allocator_reservation"])
        self.assertEqual(api.recorder.arguments["api.python_seed"], [((101,), {})])
        self.assertEqual(api.recorder.arguments["torch.manual_seed"], [((101,), {})])
        self.assertEqual(api.recorder.arguments["cuda.manual_seed_all"], [((101,), {})])
        operations = report["operation_accounting"]
        self.assertEqual(sum(item["attempted"] for name, item in operations.items() if not name.startswith("guard_")), 35)
        self.assertTrue(all(item["attempted"] == item["returned"] and item["failed"] == item["pending"] == 0 for item in operations.values()))
        final = json.loads((output / "PHASE_ACCOUNTING_FINAL.json").read_text())
        self.assertEqual(final["IO"], report["IO_accounting"])
        self.assertEqual(final["operations"], operations)
        self.assertEqual(final["IO"]["final_summary_write"], returned())
        self.assertEqual(final["IO"]["ledger_close"], returned())
        self.assertTrue(final["closed"])
        self.assertEqual(fsync.call_count, final["markers"] + 4)
        self.assertFalse(final["native_PASS"])
        for name in ("model_loaded", "model_constructed", "M0_membership", "native_launch_authorized",
                     "scoring_authorized", "native_fit_established", "cause_established"):
            self.assertIs(report[name], False)
        for name in ("forwards", "backwards", "optimizer_steps", "decodes", "grades"):
            self.assertEqual(report[name], 0)

    def test_ID_is_tensor_false_short_circuits_every_property_read(self):
        report, api, _ = self.run_fixture({"ids.is_tensor": False})
        self.assert_failed(report)
        self.assertEqual(fixtures.check_reads(api.recorder), ["ids.is_tensor"])
        self.assertIn("torch.ones_like", api.recorder.events)
        self.assertEqual(report["operation_accounting"]["input_reference_release"], returned())
        self.assertEqual(report["operation_accounting"]["mask_reference_release"], returned())

    def test_ID_shape_dtype_and_device_failures_preserve_source_short_circuit(self):
        cases = [("ids.shape", (1, 421), 2), ("ids.dtype", "wrong_dtype", 3), ("ids.device", "cpu", 5)]
        for name, value, expected in cases:
            with self.subTest(predicate=name):
                report, api, _ = self.run_fixture({name: value})
                self.assert_failed(report)
                self.assertEqual(fixtures.check_reads(api.recorder), fixtures.EXPECTED_CHECK_READS[:expected])
                self.assertNotIn("mask_is_tensor", report["operation_accounting"])

    def test_ID_toList_mismatch_never_enters_mask_validation(self):
        report, api, _ = self.run_fixture({"ids.tolist": [[-1] * 422]})
        self.assert_failed(report)
        self.assertEqual(fixtures.check_reads(api.recorder), fixtures.EXPECTED_CHECK_READS[:6])
        self.assertNotIn("mask_is_tensor", report["operation_accounting"])
        self.assertEqual(report["error"]["message"], "exact unaltered input tensor")

    def test_mask_is_tensor_false_short_circuits_mask_metadata(self):
        report, api, _ = self.run_fixture({"mask.is_tensor": False})
        self.assert_failed(report)
        self.assertEqual(fixtures.check_reads(api.recorder), fixtures.EXPECTED_CHECK_READS[:7])
        self.assertEqual(report["error"]["message"], "exact tensor attention mask")

    def test_mask_shape_comparison_always_reads_ID_shape_second_time(self):
        for name in ("mask.shape", "ids.shape:2"):
            with self.subTest(predicate=name):
                report, api, _ = self.run_fixture({name: (1, 421)})
                self.assert_failed(report)
                self.assertEqual(fixtures.check_reads(api.recorder), fixtures.EXPECTED_CHECK_READS[:9])
                self.assertEqual(api.recorder.counts["ids.shape"], 2)
                self.assertNotIn("mask_dtype_long", report["operation_accounting"])

    def test_mask_dtype_and_device_failures_skip_later_queries(self):
        for name, value, expected in (("mask.dtype", "wrong_dtype", 10), ("mask.device", "cpu", 12)):
            with self.subTest(predicate=name):
                report, api, _ = self.run_fixture({name: value})
                self.assert_failed(report)
                self.assertEqual(fixtures.check_reads(api.recorder), fixtures.EXPECTED_CHECK_READS[:expected])
                self.assertNotIn("mask_exact_tolist", report["operation_accounting"])

    def test_mask_toList_mismatch_preserves_full_reads_but_no_success_guard(self):
        report, api, _ = self.run_fixture({"mask.tolist": [[0] * 422]})
        self.assert_failed(report)
        self.assertEqual(fixtures.check_reads(api.recorder), fixtures.EXPECTED_CHECK_READS)
        self.assertNotIn("after_input_validation", api.recorder.guards)
        self.assertNotIn("complete", api.recorder.guards)

    def test_primary_error_survives_diagnostic_failure_and_hostile_stringification(self):
        original = fixtures.PrimaryFixtureError("ORIGINAL_CPU_FAILURE")
        with patch.object(pipeline.C, "frame_helper", side_effect=ValueError("SECONDARY_HELPER_FAILURE")):
            report, _, _ = self.run_fixture({"torch.tensor": original})
        self.assert_failed(report)
        self.assertEqual(report["error"], {"type": "PrimaryFixtureError", "message": "ORIGINAL_CPU_FAILURE"})
        self.assertIn("exception_frame_diagnostic_failed", report)
        self.assertEqual(report["operation_accounting"]["bounded_original_exception_frames_capture"]["failed"], 1)
        class PoisonedPrimary(RuntimeError):
            def __str__(self):
                raise AssertionError("SECONDARY_MESSAGE_FAILURE")
        report, _, _ = self.run_fixture({"torch.tensor": PoisonedPrimary()})
        self.assert_failed(report)
        self.assertEqual(report["error"]["type"], "PoisonedPrimary")
        self.assertIs(report["error"]["message_unavailable"], True)
        self.assertIn("exception_frames", report)

    def test_guard_query_budget_refuses_before_the_fifteenth_actual_query(self):
        def excessive(stage, view):
            if stage == "after_context_init":
                for _ in range(7):
                    view.cuda.is_initialized()
                for _ in range(7):
                    view.cuda.mem_get_info()
                view.cuda.is_initialized()
        report, api, _ = self.run_fixture(guard=excessive)
        self.assert_failed(report)
        self.assertEqual(report["operation_accounting"]["guard_cuda_is_initialized"]["attempted"], 7)
        self.assertEqual(report["operation_accounting"]["guard_cuda_mem_get_info"]["attempted"], 7)
        # Cleanup may make one separately registered terminal query.
        self.assertEqual(api.recorder.counts["cuda.is_initialized"], 8)
        self.assertEqual(api.recorder.counts["cuda.mem_get_info"], 8)
        self.assertNotIn("torch.tensor", api.recorder.events)

    def test_resource_and_terminal_floors_do_not_become_fixture_exemptions(self):
        faults = [{"api.disk_free": 4 * 1024**3 - 1},
                  {"api.growth_observation": {"available_bytes": 0, "worker_RSS_bytes": 0}},
                  {"cuda.mem_get_info": (0, 24 * 1024**3)},
                  {"cuda.max_memory_allocated": True}]
        for fault in faults:
            with self.subTest(fault=fault):
                report, _, _ = self.run_fixture(fault)
                self.assert_failed(report)

    def test_return_marker_IO_failure_does_not_erase_actual_tensor_return(self):
        original = pipeline.Ledger.marker
        def marker(ledger, name, event, ordinal, *args, **kwargs):
            if name == "input_tensor_create" and event == "returned":
                raise OSError("CPU_RETURN_MARKER_IO_FAILURE")
            return original(ledger, name, event, ordinal, *args, **kwargs)
        with patch.object(pipeline.Ledger, "marker", marker):
            report, _, _ = self.run_fixture()
        self.assert_failed(report)
        self.assertEqual(report["operation_accounting"]["input_tensor_create"], returned())
        self.assertTrue(report["publication_errors"])
        self.assertNotIn("mask_ones_like_create", report["operation_accounting"])

    def test_publication_close_and_final_IO_failure_never_return_CPU_complete(self):
        original_durable, original_close, original_marker = pipeline.Ledger.durable, pipeline.Ledger.close, pipeline.Ledger.marker
        for point in ("worker_result.json", "worker_result_return_marker", "PHASE_ACCOUNTING_FINAL.json", "close"):
            def durable(ledger, name, value, point=point):
                if name == point:
                    raise OSError("CPU_DURABLE_IO_FAILURE")
                return original_durable(ledger, name, value)
            def close(ledger):
                original_close(ledger)
                raise OSError("CPU_CLOSE_IO_FAILURE")
            def marker(ledger, name, event, ordinal, *args, point=point, **kwargs):
                if point == "worker_result_return_marker" and name == "publication_worker_result" and event == "returned":
                    raise OSError("CPU_WORKER_RETURN_MARKER_FAILURE")
                return original_marker(ledger, name, event, ordinal, *args, **kwargs)
            with self.subTest(point=point), patch.object(pipeline.Ledger, "durable", durable), patch.object(pipeline.Ledger, "marker", marker):
                if point == "close":
                    with patch.object(pipeline.Ledger, "close", close):
                        report, _, output = self.run_fixture()
                else:
                    report, _, output = self.run_fixture()
                self.assert_failed(report)
                self.assertIs(report["requires_complete_final_phase_accounting"], True)
                worker_path = output / "worker_result.json"
                if worker_path.exists():
                    preliminary = json.loads(worker_path.read_text())
                    self.assertEqual(preliminary["status"], SUCCESS)
                    self.assertIs(preliminary["requires_complete_final_phase_accounting"], True)
                final_path = output / "PHASE_ACCOUNTING_FINAL.json"
                if final_path.exists():
                    self.assertEqual(json.loads(final_path.read_text())["status"], FAILED)
                else:
                    self.assertIn("final_summary_error", report)

    def test_missing_or_changed_config_preflight_cannot_claim_completion(self):
        for fault in ({"api.config_preflight": None}, {"api.config_preflight:2": {"identity": "DIFFERENT"}}):
            with self.subTest(fault=fault):
                report, api, _ = self.run_fixture(fault)
                self.assert_failed(report)
                self.assertNotIn("complete", api.recorder.guards)
                if fault.get("api.config_preflight", "present") is None:
                    self.assertNotIn("api.torch_import", api.recorder.events)
        secondary = fixtures.PrimaryFixtureError("FINAL_PREFLIGHT_FAILED")
        report, _, _ = self.run_fixture({"ids.is_tensor": False, "api.config_preflight:2": secondary})
        self.assert_failed(report)
        self.assertEqual(report["error"], {"type": "ValueError", "message": "exact unaltered input tensor"})
        self.assertEqual(report["secondary_errors"][0]["error"],
                         {"type": "PrimaryFixtureError", "message": "FINAL_PREFLIGHT_FAILED"})
        for name in ("terminal_cuda_is_initialized", "terminal_cuda_max_memory_allocated",
                     "terminal_cuda_max_memory_reserved", "terminal_cuda_mem_get_info"):
            self.assertNotIn(name, report["operation_accounting"])

    def test_native_entry_points_are_inert_and_no_framework_is_imported(self):
        forbidden = {"torch", "transformers", "jax", "tensorflow", "flax"}
        self.assertFalse(forbidden.intersection(name.split(".", 1)[0] for name in sys.modules))
        with patch.object(pipeline.C, "load_input", side_effect=AssertionError("input was read")), \
             patch.object(pipeline, "Ledger", side_effect=AssertionError("output was touched")):
            for function in (pipeline.run_native, pipeline.dispatch):
                with self.subTest(function=function), self.assertRaises(ValueError):
                    function({}, lambda *args: None, self.root / "missing_output")


if __name__ == "__main__":
    unittest.main()
