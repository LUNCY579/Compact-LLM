"""Synthetic controls for receipt validation, never evidence of native model behavior."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import e1_gates as gates


ROOT = Path(__file__).resolve().parents[1]


def synthetic_result():
    """A values-only fixture. It deliberately has no model/bundle provenance."""
    stages = ["preload", "loaded"] + [f"base_parity_{i}" for i in range(32)]
    stages += [f"zero_parity_{i}" for i in range(32)]
    stages += ["native_padding_2_0", "native_padding_0_2", "native_padding_2_2", "sample"]
    for length in (256, 512, 1024):
        stages += [f"before_backward_{length}", f"backward_{length}", f"after_step_{length}"]
    stages += ["complete"]
    resources = [{"stage": stage, "elapsed_seconds": i, "host_available_gib": 8,
                  "rss_gib": 1, "gpu_free_gib": 5, "gpu_allocated_gib": 1,
                  "gpu_reserved_gib": 1, "peak_allocated_gib": 1, "disk_free_gib": 10}
                 for i, stage in enumerate(stages)]
    checks = {name: True for name in gates.TRUE_CHECKS}
    checks.update(zero_adapter_parity={"prompts": 32, "max_abs": 0.0},
                  sampled_logprobs={"max_abs": .001, "mean_abs": .0001, "tokens": [151645],
                                    "stop_reason": "eos", "backend": "full_prefix_no_cache_temperature1_topk0_topp1"})
    return {"schema_version": 1, "experiment_id": "RINEYA-0P6B-E0", "status": gates.PASS_STATUS,
            "evidence_level": "MEASURED / REPRODUCED", "dirty_at_launch": False,
            "pretrained_loads": 1, "optimizer_steps": 5, "training_tokens": 3840,
            "generation_tokens": 1, "teacher_calls": 0, "e1_authorized": False,
            "code_sha": "a" * 40, "config_sha256": gates.CONFIG_SHA256,
            "wall_seconds": 100, "checks": checks, "resources": resources,
            "resource_profiles": [{"context": length, "loss": 1, "gradient_norm": 1, "seconds": 2,
                                   "tokens_per_second": length / 2, "peak_allocated_bytes": 1024,
                                   "optimizer_tensor_bytes": 512} for length in (256, 512, 1024)],
            "load_seconds": 1, "generation_seconds": 1, "generation_tokens_per_second": 1,
            "checkpoint_save_seconds": 1, "checkpoint_reload_seconds": 1}


class E1GateControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/E0_TITAN_BF16.json").read_text())
        cls.primary = json.loads((ROOT / "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json").read_text())
        cls.inventory = json.loads((ROOT / "receipts/titan_static_v1/PARAMETER_INVENTORY.json").read_text())

    def test_synthetic_values_validate_but_do_not_qualify_a_bundle(self):
        self.assertIsNone(gates.validate_result(synthetic_result(), self.config))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "E0_RESULT.json").write_text(json.dumps(synthetic_result()))
            outcome = gates.inspect_e0(ROOT, e0_directory=path)
        self.assertEqual(outcome["status"], "BLOCKED")
        self.assertFalse(outcome["native_launch_authorized"])
        self.assertTrue(outcome["no_native_executor"])

    def test_real_blocked_e0_needs_no_downstream_files(self):
        report = gates.preflight(ROOT)
        e0 = report["gates"]["e0_native"]
        self.assertEqual(e0["status"], "BLOCKED")
        self.assertEqual([reason["code"] for reason in e0["reasons"]], ["E0_NOT_PASSED", "E0_RECORDED_FAILURE"])
        self.assertEqual(len(e0["bound_files"]), 1)
        self.assertIn("HOST_RAM_GUARD", e0["reasons"][1]["detail"])
        self.assertFalse(report["e1_launch_ready"])
        self.assertFalse(report["native_launch_authorized"])
        self.assertTrue(report["no_native_executor"])

    def test_status_string_alone_is_not_native_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "E0_RESULT.json").write_text(json.dumps({"status": gates.PASS_STATUS}))
            (path / "CONFIG.json").write_text(json.dumps(self.config))
            outcome = gates.inspect_e0(ROOT, e0_directory=path)
        self.assertEqual(outcome["status"], "BLOCKED")
        self.assertEqual(outcome["reasons"][0]["code"], "E0_RESULT_CONTRACT")

    def test_config_mismatch_rejected_before_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "E0_RESULT.json").write_text(json.dumps(synthetic_result()))
            config = deepcopy(self.config)
            config["weight_sha256"] = "b" * 64
            (path / "CONFIG.json").write_text(json.dumps(config))
            outcome = gates.inspect_e0(ROOT, e0_directory=path)
        self.assertEqual(outcome["reasons"][0]["code"], "E0_CONFIG_DRIFT")
        self.assertFalse(any("model.safetensors" in name for name in outcome["bound_files"]))

    def test_nonfinite_json_nan_and_overflow_rejected(self):
        for text in ('{"status":"BLOCKED_E0","x":NaN}', '{"x":1e999}', '{"x":Infinity}'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "E0_RESULT.json").write_text(text)
                outcome = gates.inspect_e0(ROOT, e0_directory=path)
                self.assertEqual(outcome["reasons"][0]["code"], "NONFINITE_JSON")

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "E0_RESULT.json").write_text('{"status":"BLOCKED_E0","status":"PASS"}')
            outcome = gates.inspect_e0(ROOT, e0_directory=path)
        self.assertEqual(outcome["reasons"][0]["code"], "DUPLICATE_JSON_KEY")

    def test_pass_result_counts_strict_and_exact(self):
        for field, value in (("pretrained_loads", True), ("optimizer_steps", 4), ("training_tokens", 0),
                             ("teacher_calls", 1), ("dirty_at_launch", True), ("e1_authorized", True)):
            with self.subTest(field=field):
                result = synthetic_result()
                result[field] = value
                with self.assertRaisesRegex(gates.GateError, field):
                    gates.validate_result(result, self.config)

    def test_check_missing_false_or_numeric_boolean_rejected(self):
        for name in gates.TRUE_CHECKS:
            for value in (False, 1):
                with self.subTest(name=name, value=value):
                    result = synthetic_result()
                    result["checks"][name] = value
                    with self.assertRaises(gates.GateError):
                        gates.validate_result(result, self.config)
        result = synthetic_result()
        del result["checks"]["exact_native_load"]
        with self.assertRaises(gates.GateError):
            gates.validate_result(result, self.config)

    def test_parity_tolerance_cannot_be_relaxed(self):
        result = synthetic_result()
        result["checks"]["zero_adapter_parity"]["max_abs"] = .0000001
        with self.assertRaises(gates.GateError):
            gates.validate_result(result, self.config)

    def test_likelihood_tolerance_backend_and_nan_rejected(self):
        for field, value in (("max_abs", .032), ("mean_abs", .011), ("mean_abs", float("nan")),
                             ("backend", "kv_cache"), ("stop_reason", "length")):
            with self.subTest(field=field):
                result = synthetic_result()
                result["checks"]["sampled_logprobs"][field] = value
                with self.assertRaises(gates.GateError):
                    gates.validate_result(result, self.config)

    def test_resource_floor_and_missing_stage_rejected(self):
        result = synthetic_result()
        result["resources"][0]["host_available_gib"] = 7.39
        with self.assertRaises(gates.GateError):
            gates.validate_result(result, self.config)
        result = synthetic_result()
        result["resources"].pop(5)
        with self.assertRaises(gates.GateError):
            gates.validate_result(result, self.config)

    def test_profile_context_or_fake_throughput_rejected(self):
        result = synthetic_result()
        result["resource_profiles"][2]["context"] = 4096
        with self.assertRaises(gates.GateError):
            gates.validate_result(result, self.config)
        result = synthetic_result()
        result["resource_profiles"][0]["tokens_per_second"] = 999
        with self.assertRaises(gates.GateError):
            gates.validate_result(result, self.config)

    def test_generation_throughput_and_timing_cannot_be_fabricated(self):
        for field, value in (("generation_tokens_per_second", 999), ("load_seconds", 10000),
                             ("wall_seconds", 10 ** 400)):
            with self.subTest(field=field):
                result = synthetic_result()
                result[field] = value
                with self.assertRaises(gates.GateError):
                    gates.validate_result(result, self.config)

    def test_resource_integer_fields_are_strict(self):
        for field, value in (("context", 256.0), ("peak_allocated_bytes", 1024.5), ("optimizer_tensor_bytes", True)):
            with self.subTest(field=field):
                result = synthetic_result()
                result["resource_profiles"][0][field] = value
                with self.assertRaises(gates.GateError):
                    gates.validate_result(result, self.config)

    def test_native_source_helpers_match_recorded_commit_without_imports(self):
        source = json.loads((ROOT / "results/e0_titan_v2/SOURCE_RECEIPT.json").read_text())
        bound = {}
        for relative in gates.NATIVE_HELPERS:
            gates._verify_committed_helper(ROOT.parents[1], source["code_sha"], relative, bound)
        self.assertEqual(len(bound), len(gates.NATIVE_HELPERS))
        with self.assertRaises(gates.GateError):
            gates._verify_source(ROOT.parents[1], source["code_sha"], gates.QUALIFIER, "0" * 64, {})

    def test_runtime_package_identity_rejected_on_synthetic_bundle(self):
        # Existing source metadata can establish source lineage, but does not
        # convert these synthetic results into observed native evidence.
        source = json.loads((ROOT / "results/e0_titan_v2/SOURCE_RECEIPT.json").read_text())
        runtime = json.loads((ROOT / "results/e0_titan_v2/RUNTIME_RECEIPT.json").read_text())
        runtime["packages"]["torch"] = "wrong-version"
        result = synthetic_result()
        result["code_sha"] = source["code_sha"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, value in (("E0_RESULT.json", result), ("CONFIG.json", self.config),
                                ("SOURCE_RECEIPT.json", source), ("RUNTIME_RECEIPT.json", runtime)):
                (path / name).write_text(json.dumps(value))
            outcome = gates.inspect_e0(ROOT, e0_directory=path)
        self.assertEqual(outcome["status"], "BLOCKED")
        self.assertEqual(outcome["reasons"][0]["code"], "RUNTIME_IDENTITY")

    def inventories(self):
        modules = [f"model.layers.{i}.self_attn.{projection}_proj" for i in range(28) for projection in ("q", "v")]
        parameters = {}
        for module in modules:
            parameters[module + ".lora_A"] = {"shape": [8, 1024], "dtype": "torch.float32"}
            parameters[module + ".lora_B"] = {"shape": [2048 if module.endswith("q_proj") else 1024, 8], "dtype": "torch.float32"}
        model = {"weight_sha256": gates.WEIGHT_SHA256, "primary_receipt_sha256": gates.PRIMARY_SHA256,
                 "unique_parameters": 596049920, "tied": True,
                 "loader": "stock Qwen3 allocated BF16 on CUDA; strict streamed safetensors copy",
                 "config": deepcopy(self.primary["architecture"]),
                 "tensor_sha256": {name: "a" * 64 for name in self.inventory["tensors"]}}
        return model, {"modules": modules, "parameters": parameters, "count": 1146880}

    def test_inventory_fixture_is_only_a_schema_check(self):
        model, trainable = self.inventories()
        self.assertIsNone(gates._check_inventory(model, trainable, self.primary, self.inventory))

    def test_model_identity_architecture_and_missing_tensor_rejected(self):
        for field, value in (("weight_sha256", "b" * 64), ("tied", False), ("unique_parameters", 344000000)):
            with self.subTest(field=field):
                model, trainable = self.inventories()
                model[field] = value
                with self.assertRaises(gates.GateError):
                    gates._check_inventory(model, trainable, self.primary, self.inventory)
        model, trainable = self.inventories()
        model["config"]["num_hidden_layers"] = 16
        with self.assertRaises(gates.GateError):
            gates._check_inventory(model, trainable, self.primary, self.inventory)
        model, trainable = self.inventories()
        model["tensor_sha256"].pop(next(iter(model["tensor_sha256"])))
        with self.assertRaises(gates.GateError):
            gates._check_inventory(model, trainable, self.primary, self.inventory)

    def test_adapter_scope_cannot_expand_or_change_dtype(self):
        model, trainable = self.inventories()
        first = next(iter(trainable["parameters"]))
        trainable["parameters"][first]["dtype"] = "torch.bfloat16"
        with self.assertRaises(gates.GateError):
            gates._check_inventory(model, trainable, self.primary, self.inventory)
        model, trainable = self.inventories()
        trainable["modules"].append("model.layers.0.self_attn.k_proj")
        with self.assertRaises(gates.GateError):
            gates._check_inventory(model, trainable, self.primary, self.inventory)

    def test_pins_match_existing_immutable_metadata(self):
        self.assertEqual(gates._digest(ROOT / "configs/E0_TITAN_BF16.json"), gates.CONFIG_SHA256)
        self.assertEqual(gates._digest(ROOT / "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json"), gates.PRIMARY_SHA256)
        self.assertEqual(gates._digest(ROOT / "receipts/titan_static_v1/PARAMETER_INVENTORY.json"), gates.INVENTORY_SHA256)

    def test_downstream_gates_cannot_accept_status_flags(self):
        with self.assertRaises(TypeError):
            gates.preflight(ROOT, source_enrollment={"status": "PASS", "qualified": True})
        report = gates.preflight(ROOT)
        for name in ("source_enrollment", "fresh_language_grading", "long_context_runtime"):
            self.assertEqual(report["gates"][name]["status"], "BLOCKED")

    def test_import_has_no_backend_or_preparation_side_effect(self):
        code = ("import os,sys; before=dict(os.environ); "
                "from experiments.post_training_mimo_centralized_v1.scripts import e1_gates; "
                "assert dict(os.environ)==before; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
                "assert not any('tokenize_bridge' in n for n in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT.parents[1], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
