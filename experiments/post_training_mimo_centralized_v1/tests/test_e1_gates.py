"""Synthetic controls for receipt validation, never evidence of native model behavior."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import e1_gates as gates


ROOT = Path(__file__).resolve().parents[1]


def synthetic_result(generated=1):
    """A values-only fixture. It deliberately has no model/bundle provenance."""
    stages = ["preload", "loaded"] + [f"base_parity_{i}" for i in range(32)]
    stages += [f"zero_parity_{i}" for i in range(32)]
    stages += ["native_padding_2_0", "native_padding_0_2", "native_padding_2_2"] + ["sample"] * generated
    for length in (256, 512, 1024):
        stages += [f"before_backward_{length}", f"backward_{length}", f"after_step_{length}"]
    stages += ["complete"]
    resources = [{"stage": stage, "elapsed_seconds": i, "host_available_gib": 8,
                  "rss_gib": 1, "gpu_free_gib": 5, "gpu_allocated_gib": 1,
                  "gpu_reserved_gib": 1, "peak_allocated_gib": 1, "disk_free_gib": 10}
                 for i, stage in enumerate(stages)]
    checks = {name: True for name in gates.TRUE_CHECKS}
    tokens = [42] * generated
    if generated < 32:
        tokens[-1] = 151645
    checks.update(zero_adapter_parity={"prompts": 32, "max_abs": 0.0},
                  sampled_logprobs={"max_abs": .001, "mean_abs": .0001, "tokens": tokens,
                                    "stop_reason": "eos" if tokens[-1] == 151645 else "length",
                                    "backend": "full_prefix_no_cache_temperature1_topk0_topp1"})
    operations = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310,
                  "forward": 75 + generated, "backward": 5, "optimizer_step": 5,
                  "checkpoint_save": 1, "checkpoint_reload": 1}
    positions = 41704 + 81 * generated + generated * (generated - 1) // 2
    accounting = {"accounting_version": 2, "max_optimizer_calls": 5, "last_call_id": 399 + generated,
                  "operations": {name: {"attempted": number, "returned": number, "failed": 0, "pending": 0}
                                 for name, number in operations.items()},
                  "pending_calls": [], "forward_input_positions_attempted": positions,
                  "forward_input_positions_returned": positions,
                  "training_forward_input_positions_attempted": 3840,
                  "training_forward_input_positions_returned": 3840}
    return {"schema_version": 1, "experiment_id": "RINEYA-0P6B-E0", "status": gates.PASS_STATUS,
            "evidence_level": "MEASURED / REPRODUCED", "dirty_at_launch": False,
            "pretrained_loads": 1, "optimizer_steps": 5, "training_tokens": 3840,
            "generation_tokens": generated, "teacher_calls": 0, "e1_authorized": False,
            "accounting_version": 2, "call_accounting": accounting,
            "code_sha": "a" * 40, "config_sha256": gates.CONFIG_SHA256,
            "wall_seconds": 1000, "checks": checks, "resources": resources,
            "resource_profiles": [{"context": length, "loss": 1, "gradient_norm": 1, "seconds": 2,
                                   "tokens_per_second": length / 2, "peak_allocated_bytes": 1024,
                                   "optimizer_tensor_bytes": 512} for length in (256, 512, 1024)],
            "load_seconds": 1, "generation_seconds": 1, "generation_tokens_per_second": generated,
            "checkpoint_save_seconds": 1, "checkpoint_reload_seconds": 1}


def synthetic_journal(result, tensor_names):
    """Independently stage the registered mechanics with synthetic times, never native calls."""
    events = []
    resource_index = call_index = 0

    def emit(kind, **fields):
        event = {"elapsed_seconds": len(events) / 10, "event": kind, **fields}
        events.append(event)

    def begin(operation, stage, positions=0, training=False):
        nonlocal call_index
        call_index += 1
        fields = {"accounting_version": 2, "call_id": call_index, "operation": operation,
                  "stage": stage, "input_positions": positions, "training": training}
        emit("operation_attempt", **fields)
        return fields

    def call(operation, stage, positions=0, training=False):
        fields = begin(operation, stage, positions, training)
        emit("operation_returned", **fields)

    def resource(stage):
        nonlocal resource_index
        sample = result["resources"][resource_index]
        assert sample["stage"] == stage
        resource_index += 1
        sample["elapsed_seconds"] = len(events) / 10
        emit("resource", **sample)

    emit("runtime_import_start")
    emit("runtime_import_complete")
    resource("preload")
    call("model_construct", "model_construct")
    checkpoint = begin("checkpoint_load", "checkpoint_load")
    for tensor in sorted(tensor_names):
        call("tensor_copy", "tensor_copy:" + tensor)
    emit("operation_returned", **checkpoint)
    resource("loaded")
    for phase in ("base_parity", "zero_parity"):
        for prompt in range(32):
            name = f"{phase}_{prompt}"
            call("forward", name, (256, 512, 1024)[prompt % 3])
            resource(name)
    call("forward", "padding_reference", 80)
    for name, length in (("native_padding_2_0", 82), ("native_padding_0_2", 82), ("native_padding_2_2", 84)):
        call("forward", name, length)
        resource(name)
    call("forward", "causality_perturbed", 80)
    for sample in range(result["generation_tokens"]):
        call("forward", f"sample_{sample}", 80 + sample)
        resource("sample")
    call("forward", "sampled_logprobs_offline", 80 + result["generation_tokens"])
    for profile in result["resource_profiles"]:
        length = profile["context"]
        stage = f"training_{length}"
        resource(f"before_backward_{length}")
        call("forward", stage, length, True)
        call("backward", stage)
        resource(f"backward_{length}")
        call("optimizer_step", stage)
        emit("resource_profile", profile=deepcopy(profile))
        resource(f"after_step_{length}")
    call("checkpoint_save", "adapter_resume_save")
    for stage in ("resume_uninterrupted", "resume_reloaded"):
        if stage == "resume_reloaded":
            call("checkpoint_reload", "adapter_resume_reload")
        call("forward", stage, 1024, True)
        call("backward", stage)
        call("optimizer_step", stage)
    resource("complete")
    return events


def synthetic_cost(result):
    return {name: deepcopy(result[name]) for name in (
        "wall_seconds", "training_tokens", "generation_tokens", "optimizer_steps", "pretrained_loads",
        "teacher_calls", "accounting_version", "call_accounting")}


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

    def test_historical_blocked_v3_needs_no_new_accounting_or_journal(self):
        result = json.loads((ROOT / "results/e0_titan_v3/E0_RESULT.json").read_bytes())
        self.assertNotIn("accounting_version", result)
        report = gates.inspect_e0(ROOT, e0_directory=ROOT / "results/e0_titan_v3")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual([item["code"] for item in report["reasons"]], ["E0_NOT_PASSED", "E0_RECORDED_FAILURE"])
        self.assertEqual(len(report["bound_files"]), 1)
        self.assertFalse(report["native_launch_authorized"])

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
        # The v2 source predates the laptop helper. Its actual three dependencies
        # remain verifiable; adding today's helper cannot rewrite that history.
        laptop_helpers = ("experiments/post_training_mimo_centralized_v1/scripts/laptop_profile.py",
                          "experiments/post_training_mimo_centralized_v1/scripts/stream_checkpoint.py")
        historical_helpers = tuple(path for path in gates.NATIVE_HELPERS if path not in laptop_helpers)
        for relative in historical_helpers:
            gates._verify_committed_helper(ROOT.parents[1], source["code_sha"], relative, bound)
        self.assertEqual(len(bound), len(historical_helpers))
        for laptop_helper in laptop_helpers:
            self.assertIn(laptop_helper, gates.NATIVE_HELPERS)
            with self.assertRaises(gates.GateError) as error:
                gates._verify_committed_helper(ROOT.parents[1], source["code_sha"], laptop_helper, {})
            self.assertEqual(error.exception.code, "SOURCE_HISTORY_DRIFT")
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
            # Isolate the runtime check while the new worker source is still
            # uncommitted; no mock grants a passing full evidence bundle.
            with patch.object(gates, "_verify_source"), patch.object(gates, "_verify_committed_helper"):
                outcome = gates.inspect_e0(ROOT, e0_directory=path)
        self.assertEqual(outcome["status"], "BLOCKED")
        self.assertEqual(outcome["reasons"][0]["code"], "RUNTIME_IDENTITY")

    def journal_bundle(self, generated=1):
        result = synthetic_result(generated)
        names = sorted(self.inventory["tensors"])
        events = synthetic_journal(result, names)
        return result, synthetic_cost(result), events, names

    def test_successful_accounting_and_full_journal_for_minimum_and_maximum_generation(self):
        for generated in (1, 7, 32):
            result, cost, events, names = self.journal_bundle(generated)
            with self.subTest(generated=generated):
                self.assertIsNone(gates.validate_accounting_bundle(result, cost, events, self.config, names))
                attempts = [event for event in events if event["event"] == "operation_attempt"]
                forwards = [event for event in attempts if event["operation"] == "forward"]
                self.assertEqual(len(attempts), 399 + generated)
                self.assertEqual(len(forwards), 75 + generated)
                self.assertEqual(sum(event["input_positions"] for event in forwards),
                                 41704 + 81 * generated + generated * (generated - 1) // 2)
                self.assertEqual(sum(event["input_positions"] for event in forwards if event["training"]), 3840)

    def test_independent_gate_fixture_agrees_with_worker_pure_journal_reconstruction(self):
        # Importing this module does not run its guarded native entrypoint.
        # The gate itself never imports it or delegates journal checking to it.
        from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import verify_call_journal
        for generated in (1, 32):
            result, cost, events, names = self.journal_bundle(generated)
            rebuilt = verify_call_journal(events, snapshot=result["call_accounting"], maximum_optimizer_calls=5)
            self.assertEqual(rebuilt, result["call_accounting"])
            gates.validate_accounting_bundle(result, cost, events, self.config, names)

    def test_inspection_rejects_corrupt_journal_before_weight_streaming(self):
        result, cost, events, names = self.journal_bundle()
        source = json.loads((ROOT / "results/e0_titan_v2/SOURCE_RECEIPT.json").read_bytes())
        runtime = json.loads((ROOT / "results/e0_titan_v2/RUNTIME_RECEIPT.json").read_bytes())
        result["code_sha"] = source["code_sha"]
        model, trainable = self.inventories()
        tokenizer = {"class_name": "Qwen2TokenizerFast", "template_sha256": gates.TEMPLATE_SHA256,
                     "eos_id": 151645, "original_pad_id": 151643, "raw_roundtrip": True,
                     "enable_thinking": False, "add_generation_prompt": True, "raw_ids": [42]}
        events.pop(next(index for index, event in enumerate(events)
                        if event.get("operation") == "optimizer_step" and event["event"] == "operation_attempt"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, value in {"E0_RESULT.json": result, "CONFIG.json": self.config,
                                "SOURCE_RECEIPT.json": source, "RUNTIME_RECEIPT.json": runtime,
                                "TOKENIZER_RECEIPT.json": tokenizer, "MODEL_RECEIPT.json": model,
                                "TRAINABLE_INVENTORY.json": trainable, "COST_LEDGER.json": cost,
                                "RESOURCE_BASELINE.json": {"status": gates.PASS_STATUS,
                                    "samples": result["resources"], "profiles": result["resource_profiles"]}}.items():
                (path / name).write_text(json.dumps(value))
            (path / "EVENTS.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
            with patch.object(gates, "_verify_source"), patch.object(gates, "_verify_committed_helper"), \
                    patch.object(gates, "_check_weight_tensor_bytes") as weights:
                report = gates.inspect_e0(ROOT, e0_directory=path)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["reasons"][0]["code"], "EVENTS_CONTRACT")
            self.assertIn(str(path / "EVENTS.jsonl"), report["bound_files"])
            self.assertFalse(report["native_launch_authorized"])
            weights.assert_not_called()

    def test_new_pass_without_version2_counts_is_rejected(self):
        for missing in ("accounting_version", "call_accounting"):
            result = synthetic_result()
            del result[missing]
            with self.subTest(missing=missing), self.assertRaises(gates.GateError):
                gates.validate_result(result, self.config)
        result = synthetic_result()
        result["accounting_version"] = 1
        with self.assertRaises(gates.GateError):
            gates.validate_result(result, self.config)

    def test_cost_missing_legacy_or_accounting_fields_and_extra_claims_rejected(self):
        for field in gates.COST_FIELDS:
            result, cost, events, names = self.journal_bundle()
            del cost[field]
            with self.subTest(field=field), self.assertRaisesRegex(gates.GateError, "costs and version-2"):
                gates.validate_accounting_bundle(result, cost, events, self.config, names)
        result, cost, events, names = self.journal_bundle()
        cost["native_qualified"] = True
        with self.assertRaises(gates.GateError):
            gates.validate_accounting_bundle(result, cost, events, self.config, names)

    def test_failed_pending_over_limit_or_undercounted_operations_cannot_pass(self):
        edits = [lambda acc: acc["operations"]["optimizer_step"].update(attempted=6),
                 lambda acc: acc["operations"]["forward"].update(returned=75, pending=1),
                 lambda acc: acc["operations"]["backward"].update(failed=1),
                 lambda acc: acc["operations"]["model_construct"].update(attempted=True),
                 lambda acc: acc.update(forward_input_positions_attempted=1),
                 lambda acc: acc.update(training_forward_input_positions_returned=3839),
                 lambda acc: acc.update(pending_calls=[{"call_id": 400, "operation": "optimizer_step"}]),
                 lambda acc: acc.update(max_optimizer_calls=6), lambda acc: acc.update(last_call_id=399)]
        for edit in edits:
            result, cost, events, names = self.journal_bundle()
            edit(result["call_accounting"])
            cost = synthetic_cost(result)
            with self.assertRaisesRegex(gates.GateError, "version-2 operation counts"):
                gates.validate_accounting_bundle(result, cost, events, self.config, names)

    def test_deleted_duplicate_reordered_or_failed_operation_record_rejected(self):
        for mutation in ("drop", "duplicate", "reorder", "failed", "early_parent_return"):
            result, cost, events, names = self.journal_bundle()
            index = next(i for i, event in enumerate(events) if event.get("operation") == "tensor_copy")
            if mutation == "drop":
                events.pop(index)
            elif mutation == "duplicate":
                events.insert(index, deepcopy(events[index]))
            elif mutation == "reorder":
                events[index], events[index + 1] = events[index + 1], events[index]
            elif mutation == "failed":
                events[index + 1]["event"] = "operation_failed"
            else:
                end = next(i for i, event in enumerate(events) if event.get("operation") == "checkpoint_load"
                           and event["event"] == "operation_returned")
                events[index], events[end] = events[end], events[index]
            with self.subTest(mutation=mutation), self.assertRaises(gates.GateError):
                gates.validate_accounting_bundle(result, cost, events, self.config, names)

    def test_stage_call_id_input_training_or_timestamp_drift_rejected(self):
        for field, value in (("stage", "sample_999"), ("call_id", 5000), ("input_positions", 79),
                             ("training", True), ("elapsed_seconds", -1), ("accounting_version", True)):
            result, cost, events, names = self.journal_bundle()
            event = next(event for event in events if event.get("stage") == "sample_0"
                         and event["event"] == "operation_attempt")
            event[field] = value
            with self.subTest(field=field), self.assertRaises(gates.GateError):
                gates.validate_accounting_bundle(result, cost, events, self.config, names)

    def test_full_resource_profile_events_and_guard_order_required(self):
        for mutation in ("drop_profile", "profile_cost", "drop_guard", "unknown_event"):
            result, cost, events, names = self.journal_bundle()
            index = next(i for i, event in enumerate(events) if event["event"] == "resource_profile")
            if mutation == "drop_profile":
                events.pop(index)
            elif mutation == "profile_cost":
                events[index]["profile"]["seconds"] = 1
            elif mutation == "drop_guard":
                events.pop(index + 1)
            else:
                events[index]["event"] = "unregistered_success"
            with self.subTest(mutation=mutation), self.assertRaises(gates.GateError):
                gates.validate_accounting_bundle(result, cost, events, self.config, names)

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
