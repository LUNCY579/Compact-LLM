"""Synthetic schema/failure controls, never native Titan measurements or grants."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import unpadded_evidence as evidence


ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": 72,
          "backward": 5, "optimizer_step": 5, "checkpoint_save": 1, "checkpoint_reload": 1}


def fixture_result(qualification_stage="unpadded_ce"):
    narrow = qualification_stage == "unpadded_ce512"
    contexts = (256, 512) if narrow else (256, 512, 1024)
    optimizer_calls = 4 if narrow else 5
    forward_positions, training_positions = (26608, 1792) if narrow else (41456, 3840)
    operation_calls = 393 if narrow else 396
    counts = {**COUNTS, "forward": 71 if narrow else 72, "backward": optimizer_calls, "optimizer_step": optimizer_calls}
    stages = ["preload", "loaded"]
    for prefix in ("base_parity", "zero_parity"):
        stages.extend(f"{prefix}_{index}" for index in range(32))
    stages.extend(["unpad_causality", "unpad_allones_equivalence"])
    for length in contexts:
        stages.extend([f"before_backward_{length}", f"backward_{length}", f"after_step_{length}"])
    stages.append("complete")
    accounting = {"accounting_version": 2, "max_optimizer_calls": optimizer_calls, "last_call_id": operation_calls,
                  "operations": {name: {"attempted": count, "returned": count, "failed": 0, "pending": 0}
                                 for name, count in counts.items()}, "pending_calls": [],
                  "forward_input_positions_attempted": forward_positions, "forward_input_positions_returned": forward_positions,
                  "training_forward_input_positions_attempted": training_positions,
                  "training_forward_input_positions_returned": training_positions}
    checks = {name: True for name in ("exact_native_load", "eos_pad_mask", "save_reload_resume_bitwise",
                                     "frozen_base_unchanged", "finite_scoped_backward")}
    checks.update(zero_adapter_parity={"prompts": 32, "max_abs": 0.0},
                  native_unpadded_causality={"tokens": 80, "perturbed_tail_tokens": 3, "prefix_tokens": 77, "bitwise": True},
                  explicit_allones_mask_equivalence={"tokens": 79, "finite": True, "max_abs": .001,
                    "mean_abs": .0001, "qualified": True, "comparison": "explicit_allones_mask_vs_omitted_same_80_tokens"})
    samples = [{"stage": stage, "elapsed_seconds": index, "host_available_gib": 5.0, "rss_gib": 1.0,
                "gpu_free_gib": 5.0, "gpu_allocated_gib": 2.0, "gpu_reserved_gib": 2.0,
                "peak_allocated_gib": 2.0, "disk_free_gib": 10.0, "admission_required_gib": 2.0}
               for index, stage in enumerate(stages)]
    profiles = [{"context": length, "loss": 1.0, "gradient_norm": .1, "seconds": 2.0,
                 "tokens_per_second": length / 2.0, "peak_allocated_bytes": 2 * 1024**3,
                 "optimizer_tensor_bytes": 9175488} for length in contexts]
    config_sha = "42a32cfa2fe4dc1250a288bacec2a60d150c5498d8dced9bba23b0c5037f5929" if narrow else evidence.CONFIG_SHA256
    return {"schema_version": 1, "experiment_id": "RINEYA-0P6B-E0",
            "status": "PASS_E0_UNPADDED_CE512_SCOPED" if narrow else "PASS_E0_UNPADDED_CE_SCOPED",
            "evidence_level": "MEASURED / REPRODUCED", "code_sha": "a" * 40, "dirty_at_launch": False,
            "config_sha256": config_sha, "pretrained_loads": 1, "optimizer_steps": optimizer_calls,
            "training_tokens": training_positions, "generation_tokens": 0, "teacher_calls": 0, "accounting_version": 2,
            "call_accounting": accounting, "resource_profile": "LAPTOP_STREAMED_V1", "qualification_stage": qualification_stage,
            "runtime_scope": "UNPADDED_SINGLE_SEQUENCE_CE512_V1" if narrow else "UNPADDED_SINGLE_SEQUENCE_CE_V1",
            "loader_mode": "stream_read_2MiB_v1",
            "checks": checks, "resources": samples, "resource_profiles": profiles, "loader_diagnostics": {},
            "load_seconds": 100.0, "checkpoint_save_seconds": 1.0, "checkpoint_reload_seconds": 1.0,
            "wall_seconds": 1000.0, "e1_authorized": False,
            "unqualified": ["padding", "packing", "batched inference", "KV-cache sampler",
                            "sampled online/offline likelihood", "evaluation capture", "preservation",
                            "E1 data/replay", "held-out capability", "long context beyond512" if narrow else "long context beyond1024"]}


def fixture_journal(result, names):
    """Stage synthetic intent/return/resource records independently of checker code."""
    narrow = result["qualification_stage"] == "unpadded_ce512"
    contexts = (256, 512) if narrow else (256, 512, 1024)
    events, call_id, resource_index = [], 0, 0
    def emit(kind, **values):
        events.append({"elapsed_seconds": len(events) / 10.0, "event": kind, **values})
    def begin(operation, stage, positions=0, training=False):
        nonlocal call_id
        call_id += 1
        record = {"accounting_version": 2, "call_id": call_id, "operation": operation,
                  "stage": stage, "input_positions": positions, "training": training}
        emit("operation_attempt", **record)
        return record
    def call(operation, stage, positions=0, training=False):
        record = begin(operation, stage, positions, training)
        emit("operation_returned", **record)
    def resource(stage):
        nonlocal resource_index
        sample = result["resources"][resource_index]
        resource_index += 1
        assert sample["stage"] == stage
        sample["elapsed_seconds"] = len(events) / 10.0
        emit("resource", **sample)
    emit("runtime_import_start")
    emit("runtime_import_complete")
    resource("preload")
    call("model_construct", "model_construct")
    checkpoint = begin("checkpoint_load", "checkpoint_load")
    for name in sorted(names):
        call("tensor_copy", "tensor_copy:" + name)
    emit("operation_returned", **checkpoint)
    resource("loaded")
    for prefix in ("base_parity", "zero_parity"):
        for index in range(32):
            name = f"{prefix}_{index}"
            call("forward", name, contexts[index % len(contexts)])
            resource(name)
    call("forward", "unpad_reference", 80)
    call("forward", "unpad_causality_perturbed", 80)
    resource("unpad_causality")
    call("forward", "unpad_allones_mask", 80)
    resource("unpad_allones_equivalence")
    for profile in result["resource_profiles"]:
        length = profile["context"]
        name = f"training_{length}"
        resource(f"before_backward_{length}")
        call("forward", name, length, True)
        call("backward", name)
        resource(f"backward_{length}")
        call("optimizer_step", name)
        emit("resource_profile", profile=deepcopy(profile))
        resource(f"after_step_{length}")
    call("checkpoint_save", "adapter_resume_save")
    for name in ("resume_uninterrupted", "resume_reloaded"):
        if name == "resume_reloaded":
            call("checkpoint_reload", "adapter_resume_reload")
        call("forward", name, 512 if narrow else 1024, True)
        call("backward", name)
        call("optimizer_step", name)
    resource("complete")
    assert (call_id, len(events), resource_index) == ((393, 865, 75) if narrow else (396, 875, 78))
    return events


def fixture_cost(result):
    return {name: deepcopy(result[name]) for name in ("wall_seconds", "training_tokens", "generation_tokens",
            "optimizer_steps", "pretrained_loads", "teacher_calls", "accounting_version", "call_accounting")}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class UnpaddedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.primary = json.loads((ROOT / "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json").read_text())
        self.inventory = json.loads((ROOT / "receipts/titan_static_v1/PARAMETER_INVENTORY.json").read_text())
        self.names = sorted(self.inventory["tensors"])

    def bundle_values(self, qualification_stage="unpadded_ce"):
        result = fixture_result(qualification_stage)
        events = fixture_journal(result, self.names)
        return result, fixture_cost(result), events

    def test_values_only_controls_are_not_bundle_qualification(self):
        result, cost, events = self.bundle_values()
        self.assertIsNone(evidence.validate_result(result))
        self.assertIsNone(evidence.validate_accounting_bundle(result, cost, events, self.names))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(path / "E0_RESULT.json", result)
            with patch.object(evidence, "_verify_assets", side_effect=AssertionError("root weights must not be read")):
                report = evidence.inspect_unpadded_scope(ROOT, e0_directory=path)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertFalse(report["native_launch_authorized"])
            self.assertFalse(report["e1_authorized"])
            self.assertTrue(report["no_native_executor"])

    def test_exact_workload_totals_and_records_are_independent_fixture_controls(self):
        result, cost, events = self.bundle_values()
        self.assertEqual(len(events), 875)
        attempts = [event for event in events if event["event"] == "operation_attempt"]
        self.assertEqual(len(attempts), 396)
        self.assertEqual(sum(event["input_positions"] for event in attempts), 41456)
        self.assertEqual(sum(event["input_positions"] for event in attempts if event["training"]), 3840)
        self.assertEqual(sum(event["event"] == "resource" for event in events), 78)
        self.assertEqual(sum(event["event"] == "resource_profile" for event in events), 3)
        self.assertEqual(attempts[376]["stage"], "unpad_reference")
        self.assertEqual(attempts[-1]["call_id"], 396)

    def test_scope_and_counters_fail_closed(self):
        cases = (("runtime_scope", "FULL"), ("resource_profile", "LAPTOP_GPU_RESIDENT_V1"),
                 ("qualification_stage", "backward_resume"), ("config_sha256", "0" * 64),
                 ("generation_tokens", 1), ("optimizer_steps", 4), ("training_tokens", 3839),
                 ("pretrained_loads", True), ("e1_authorized", True), ("dirty_at_launch", True),
                 ("wall_seconds", 1801), ("wall_seconds", float("nan")))
        for field, value in cases:
            result = fixture_result()
            result[field] = value
            with self.subTest(field=field), self.assertRaises((evidence.GateError, ValueError)):
                evidence.validate_result(result)
        for field in fixture_result():
            result = fixture_result()
            del result[field]
            with self.subTest(missing=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)
        for field in ("error", "traceback", "sampled_logprobs", "native_padding_and_causality"):
            result = fixture_result()
            result[field] = "unsupported"
            with self.subTest(extra=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)

    def test_all_eight_native_checks_and_thresholds_are_required(self):
        mutations = (("zero_adapter_parity", "prompts", 31), ("zero_adapter_parity", "max_abs", .001),
                     ("native_unpadded_causality", "prefix_tokens", 76), ("native_unpadded_causality", "bitwise", False),
                     ("explicit_allones_mask_equivalence", "max_abs", .031251),
                     ("explicit_allones_mask_equivalence", "mean_abs", .010001),
                     ("explicit_allones_mask_equivalence", "qualified", False),
                     ("explicit_allones_mask_equivalence", "finite", False))
        for check, field, value in mutations:
            result = fixture_result()
            result["checks"][check][field] = value
            with self.subTest(check=check, field=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)
        for name in fixture_result()["checks"]:
            result = fixture_result()
            del result["checks"][name]
            with self.subTest(missing=name), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)
        for name in ("exact_native_load", "eos_pad_mask", "save_reload_resume_bitwise", "frozen_base_unchanged", "finite_scoped_backward"):
            result = fixture_result()
            result["checks"][name] = 1
            with self.subTest(boolean=name), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)

    def test_attempt_return_cost_and_position_corruption_is_rejected(self):
        for field in ("last_call_id", "forward_input_positions_attempted", "forward_input_positions_returned",
                      "training_forward_input_positions_attempted", "training_forward_input_positions_returned"):
            result = fixture_result()
            result["call_accounting"][field] += 1
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)
        for operation in COUNTS:
            for field in ("attempted", "returned", "failed", "pending"):
                result = fixture_result()
                result["call_accounting"]["operations"][operation][field] += 1
                with self.subTest(operation=operation, field=field), self.assertRaises(evidence.GateError):
                    evidence.validate_result(result)
        for field in evidence.COST_FIELDS:
            result, cost, events = self.bundle_values()
            del cost[field]
            with self.subTest(cost_field=field), self.assertRaises(evidence.GateError):
                evidence.validate_accounting_bundle(result, cost, events, self.names)

    def test_complete_ordered_journal_rejects_missing_extra_metadata_and_reordering(self):
        for mutation in ("drop", "extra", "swap", "call_id", "training", "positions", "failed", "stage", "time", "resource_copy", "profile"):
            result, cost, events = self.bundle_values()
            index = next(i for i, event in enumerate(events) if event.get("stage") == "training_256" and event["event"] == "operation_attempt")
            if mutation == "drop":
                events.pop(index)
            elif mutation == "extra":
                events.insert(index, deepcopy(events[index]))
            elif mutation == "swap":
                events[index], events[index + 1] = events[index + 1], events[index]
            elif mutation == "call_id":
                events[index]["call_id"] -= 1
            elif mutation == "training":
                events[index]["training"] = False
            elif mutation == "positions":
                events[index]["input_positions"] = 255
            elif mutation == "failed":
                events[index + 1]["event"] = "operation_failed"
            elif mutation == "stage":
                events[index]["stage"] = "unpinned"
            elif mutation == "time":
                events[index]["elapsed_seconds"] = 0
            elif mutation == "resource_copy":
                event = next(event for event in events if event["event"] == "resource")
                event["rss_gib"] = .9
            else:
                event = next(event for event in events if event["event"] == "resource_profile")
                event["profile"]["loss"] = 2.0
            with self.subTest(mutation=mutation), self.assertRaises(evidence.GateError):
                evidence.validate_accounting_bundle(result, cost, events, self.names)

    def test_resource_floors_dynamic_budget_profiles_and_coverage_are_strict(self):
        for field, value in (("host_available_gib", 1.249), ("rss_gib", 1.751), ("gpu_free_gib", .499),
                             ("disk_free_gib", 3.999), ("admission_required_gib", 1.9)):
            result = fixture_result()
            result["resources"][1][field] = value
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)
        result = fixture_result()
        result["resources"][0]["host_available_gib"] = 1.999
        with self.assertRaises(evidence.GateError):
            evidence.validate_result(result)
        for field, value in (("context", 128), ("optimizer_tensor_bytes", 9175487),
                             ("peak_allocated_bytes", True), ("tokens_per_second", 1), ("loss", float("inf"))):
            result = fixture_result()
            result["resource_profiles"][0][field] = value
            with self.subTest(profile=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)
        result = fixture_result()
        result["resources"].reverse()
        with self.assertRaises(evidence.GateError):
            evidence.validate_result(result)

    def test_journal_durations_bound_profiles_load_save_and_reload(self):
        for field in ("load_seconds", "checkpoint_save_seconds", "checkpoint_reload_seconds", "profile_seconds"):
            result, cost, events = self.bundle_values()
            if field == "profile_seconds":
                profile = result["resource_profiles"][0]
                profile["seconds"] = .000001
                profile["tokens_per_second"] = profile["context"] / profile["seconds"]
                event = next(event for event in events if event["event"] == "resource_profile")
                event["profile"] = deepcopy(profile)
            else:
                result[field] = .000001
            # Profile/result/cost snapshots and throughput are otherwise coherent;
            # the same source's intent/return timestamps contradict the tiny timer.
            cost = fixture_cost(result)
            evidence.validate_result(result)
            with self.subTest(field=field), self.assertRaises(evidence.GateError) as caught:
                evidence.validate_accounting_bundle(result, cost, events, self.names)
            self.assertEqual(caught.exception.code, "TIMING_JOURNAL_CONTRADICTION")

    def test_journal_timing_epsilon_does_not_become_clock_resolution_relaxation(self):
        result, cost, events = self.bundle_values()
        save_attempt = next(event["elapsed_seconds"] for event in events
                            if event["event"] == "operation_attempt" and event.get("operation") == "checkpoint_save")
        save_return = next(event["elapsed_seconds"] for event in events
                           if event["event"] == "operation_returned" and event.get("operation") == "checkpoint_save")
        exact_span = save_return - save_attempt
        result["checkpoint_save_seconds"] = exact_span - 5e-10
        self.assertIsNone(evidence.validate_accounting_bundle(result, fixture_cost(result), events, self.names))
        result["checkpoint_save_seconds"] = exact_span - 1e-7
        with self.assertRaises(evidence.GateError) as caught:
            evidence.validate_accounting_bundle(result, fixture_cost(result), events, self.names)
        self.assertEqual(caught.exception.code, "TIMING_JOURNAL_CONTRADICTION")

    def metadata_fixture(self, base, qualification_stage="unpadded_ce"):
        """Complete synthetic metadata; source/assets are deliberately not native evidence."""
        selected = evidence.registration(qualification_stage)
        config_relative = selected["config_relative"]
        package = base / "repo/experiments/post_training_mimo_centralized_v1"
        bundle = base / "bundle"
        for relative in (config_relative, "receipts/PRIMARY_STUDENT_RECEIPT.json", "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json",
                         "receipts/titan_static_v1/PARAMETER_INVENTORY.json"):
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        result, cost, events = self.bundle_values(qualification_stage)
        diagnostics = {"loader_mode": "stream_read_2MiB_v1", "header_bytes": 35216, "file_bytes": 1192135064,
                       "payload_bytes": 1192099840, "serialized_tensors": 310, "read_buffer_bytes": 2097152,
                       "maximum_payload_read_bytes": 2097152, "payload_chunk_count": 682,
                       "memory_mapped": False, "cpu_model_allocated": False, "blocking_chunk_copies": True}
        result["loader_diagnostics"] = deepcopy(diagnostics)
        modules = [f"model.layers.{layer}.self_attn.{projection}_proj" for layer in range(28) for projection in ("q", "v")]
        parameters = {}
        for module in modules:
            parameters[module + ".lora_A"] = {"shape": [8, 1024], "dtype": "torch.float32"}
            parameters[module + ".lora_B"] = {"shape": [2048 if module.endswith("q_proj") else 1024, 8], "dtype": "torch.float32"}
        runtime = {"python": "3.11.15 (synthetic fixture)", "executable": evidence.EXECUTABLE,
                   "platform": "Windows-synthetic", "packages": deepcopy(evidence.PACKAGES),
                   "gpu": "Synthetic control, not hardware", "cuda": "12.6", "total_vram_bytes": 8 * 1024**3,
                   "attention": "eager", "base_dtype": "bfloat16", "adapter_dtype": "float32"}
        files = {"E0_RESULT.json": result, "CONFIG.json": json.loads((ROOT / config_relative).read_text()),
                 "SOURCE_RECEIPT.json": {"code_sha": "a" * 40, "dirty": False, "script_sha256": "b" * 64,
                    "dependency_sha256": {name: "b" * 64 for name in evidence.DEPENDENCIES}},
                 "RUNTIME_RECEIPT.json": runtime,
                 "TOKENIZER_RECEIPT.json": {"class_name": "Qwen2TokenizerFast", "template_sha256": evidence.low.TEMPLATE_SHA256,
                    "eos_id": 151645, "original_pad_id": 151643, "raw_roundtrip": True,
                    "enable_thinking": False, "add_generation_prompt": True,
                    "raw_ids": [31029, 2404, 25, 220, 22, 488, 220, 20, 284, 220, 16, 17, 13]},
                 "MODEL_RECEIPT.json": {"weight_sha256": evidence.low.WEIGHT_SHA256,
                    "primary_receipt_sha256": evidence.low.PRIMARY_SHA256, "unique_parameters": 596049920,
                    "tied": True, "loader": evidence.LOADER_LABEL, "config": deepcopy(self.primary["architecture"]),
                    "tensor_sha256": {name: "b" * 64 for name in self.names}, "loader_mode": evidence.LOADER_MODE,
                    "loader_diagnostics": deepcopy(diagnostics)},
                 "TRAINABLE_INVENTORY.json": {"modules": modules, "parameters": parameters, "count": 1146880},
                 "RESOURCE_BASELINE.json": {"status": result["status"], "samples": deepcopy(result["resources"]),
                                            "profiles": deepcopy(result["resource_profiles"])}, "COST_LEDGER.json": cost}
        for name, value in files.items():
            write_json(bundle / name, value)
        (bundle / "EVENTS.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
        (bundle / "adapter_resume.pt").write_bytes(b"Synthetic control; never Torch-loaded")
        supervision = bundle.with_name(bundle.name + "_supervisor")
        launch = {"command": [runtime["executable"], "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.qualify_native",
                   "--config", str(package / config_relative), "--output", str(bundle), "--go-path", str(supervision / "GO.json")],
                  "config_sha256": result["config_sha256"], "supervisor_sha256": "b" * 64, "watchdog_sha256": "b" * 64,
                  "observed_gpu_processes": [], "source_sha": "a" * 40, "resource_profile": evidence.RESOURCE_PROFILE,
                  "qualification_stage": qualification_stage, "resource_limits": deepcopy(evidence.RESOURCE_LIMITS), "loader_mode": evidence.LOADER_MODE}
        write_json(supervision / "LAUNCH.json", launch)
        write_json(supervision / "GO.json", {"config_sha256": result["config_sha256"]})
        write_json(supervision / "SUPERVISOR.json", {"status": "PASS_PROCESS_ONLY", "reason": None, "exit_code": 0,
                    "native_qualification": False, "elapsed_seconds": 1001.0, "limit_seconds": 1800,
                    "memory_poll_seconds": .1, "peak_observed_process_tree_rss_bytes": 1024**3})
        (supervision / "stdout.log").write_text(json.dumps({"status": result["status"], "error": None,
                    "wall_seconds": result["wall_seconds"], "optimizer_steps": result["optimizer_steps"]}) + "\n", encoding="utf-8")
        (supervision / "stderr.log").write_bytes(b"")
        return package, bundle

    def inspect_fixture(self, package, bundle, qualification_stage="unpadded_ce"):
        def no_assets(*args, **kwargs):
            raise evidence.GateError("SYNTHETIC_NO_ASSETS", "metadata controls are not native evidence")
        with patch.object(evidence.low, "_verify_source"), patch.object(evidence.low, "_verify_committed_helper"), \
             patch.object(evidence, "_verify_assets", side_effect=no_assets) as assets:
            report = evidence.inspect_unpadded_scope(package, e0_directory=bundle, qualification_stage=qualification_stage)
        return report, assets.call_count

    def test_complete_synthetic_metadata_never_claims_native_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            package, bundle = self.metadata_fixture(Path(directory))
            report, asset_calls = self.inspect_fixture(package, bundle)
            self.assertEqual(asset_calls, 1)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["reasons"][0]["code"], "SYNTHETIC_NO_ASSETS")

    def test_config_and_source_drift_reject_before_weight_streaming(self):
        with tempfile.TemporaryDirectory() as directory:
            package, bundle = self.metadata_fixture(Path(directory))
            with patch.object(evidence, "_verify_assets", side_effect=AssertionError("no root weights")):
                report = evidence.inspect_unpadded_scope(package, e0_directory=bundle)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["reasons"][0]["code"], "SOURCE_HISTORY_DRIFT")
            config = package / evidence.CONFIG_RELATIVE
            config.write_bytes(config.read_bytes() + b" ")
            report, calls = self.inspect_fixture(package, bundle)
            self.assertEqual(calls, 0)
            self.assertEqual(report["reasons"][0]["code"], "UNPADDED_CONFIG_IDENTITY")

    def test_current_source_drift_is_rejected_against_committed_bytes(self):
        relative = evidence.low.QUALIFIER
        raw = (ROOT.parents[1] / relative).read_bytes()
        expected_hash = evidence.hashlib.sha256(raw).hexdigest()
        historical = subprocess.CompletedProcess(args=["git", "show"], returncode=0, stdout=raw, stderr=b"")
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(raw + b"\n")
            with patch.object(evidence.low.subprocess, "run", return_value=historical), self.assertRaises(evidence.GateError) as caught:
                evidence.low._verify_source(repo, "a" * 40, relative, expected_hash, {})
            self.assertEqual(caught.exception.code, "SOURCE_CURRENT_DRIFT")

    def test_pinned_current_native_source_closure_matches_real_history(self):
        # Source identity alone is deliberately not native execution evidence.
        producer = "d5640306dac3ea33b862e77bb1eb932a980a9e4f"
        repo = ROOT.parents[1]
        for stage in ("unpadded_ce", "unpadded_ce512"):
            selected = evidence.registration(stage)
            config_source = "experiments/post_training_mimo_centralized_v1/" + selected["config_relative"]
            bound = {}
            for relative in (config_source, evidence.low.QUALIFIER, *evidence.DEPENDENCIES,
                             evidence.low.SUPERVISOR, evidence.low.WATCHDOG):
                digest = evidence.hashlib.sha256((repo / relative).read_bytes()).hexdigest()
                evidence.low._verify_source(repo, producer, relative, digest, bound)
            for relative in evidence.NATIVE_HELPERS:
                evidence.low._verify_committed_helper(repo, producer, relative, bound)
            self.assertEqual(len(bound), 11)
            self.assertEqual(bound[str((repo / config_source).resolve())]["sha256"], selected["config_sha256"])

    def test_bundle_metadata_mutations_fail_before_asset_reads(self):
        cases = (("RUNTIME_RECEIPT.json", "attention", "sdpa"), ("RUNTIME_RECEIPT.json", "cuda", "12.7"),
                 ("SOURCE_RECEIPT.json", "dirty", True), ("SOURCE_RECEIPT.json", "dependency_sha256", {}),
                 ("MODEL_RECEIPT.json", "loader", "stock Qwen3 allocated BF16 on CUDA; strict streamed safetensors copy"),
                 ("MODEL_RECEIPT.json", "tied", False), ("MODEL_RECEIPT.json", "loader_diagnostics", {}),
                 ("TOKENIZER_RECEIPT.json", "raw_roundtrip", False), ("TRAINABLE_INVENTORY.json", "count", 1146879))
        for name, field, value in cases:
            with self.subTest(name=name, field=field), tempfile.TemporaryDirectory() as directory:
                package, bundle = self.metadata_fixture(Path(directory))
                path = bundle / name
                changed = json.loads(path.read_text())
                changed[field] = value
                write_json(path, changed)
                report, calls = self.inspect_fixture(package, bundle)
                self.assertEqual(report["status"], "BLOCKED")
                self.assertEqual(calls, 0)

    def test_containment_go_and_launch_limits_are_required(self):
        cases = (("SUPERVISOR.json", "memory_poll_seconds", .2), ("SUPERVISOR.json", "native_qualification", True),
                 ("SUPERVISOR.json", "peak_observed_process_tree_rss_bytes", 2 * 1024**3),
                 ("SUPERVISOR.json", "exit_code", 1), ("GO.json", "config_sha256", "0" * 64),
                 ("LAUNCH.json", "resource_limits", {}), ("LAUNCH.json", "qualification_stage", "full"),
                 ("LAUNCH.json", "observed_gpu_processes", ["123, python.exe"]))
        for name, field, value in cases:
            with self.subTest(name=name, field=field), tempfile.TemporaryDirectory() as directory:
                package, bundle = self.metadata_fixture(Path(directory))
                path = bundle.with_name(bundle.name + "_supervisor") / name
                changed = json.loads(path.read_text())
                changed[field] = value
                write_json(path, changed)
                report, calls = self.inspect_fixture(package, bundle)
                self.assertEqual(report["status"], "BLOCKED")
                self.assertEqual(calls, 0)

    def test_stdout_and_resident_memory_contradictions_stop_before_assets(self):
        for mutation in ("stdout", "allocation", "peak", "raw_tokens", "reserved", "device_limit", "current_above_peak"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                package, bundle = self.metadata_fixture(Path(directory))
                if mutation == "stdout":
                    (bundle.with_name(bundle.name + "_supervisor") / "stdout.log").write_text("{}\n", encoding="utf-8")
                elif mutation == "raw_tokens":
                    path = bundle / "TOKENIZER_RECEIPT.json"
                    receipt = json.loads(path.read_text())
                    receipt["raw_ids"] = [10, 12]
                    write_json(path, receipt)
                else:
                    path = bundle / "E0_RESULT.json"
                    result = json.loads(path.read_text())
                    if mutation == "peak":
                        result["resource_profiles"][0]["peak_allocated_bytes"] = 1024
                    elif mutation == "allocation":
                        result["resources"][1]["gpu_allocated_gib"] = 0.0
                    elif mutation == "reserved":
                        result["resources"][1]["gpu_reserved_gib"] = 1.5
                    elif mutation == "device_limit":
                        result["resources"][1]["gpu_reserved_gib"] = 9.0
                    else:
                        result["resources"][1]["peak_allocated_gib"] = 1.5
                    events = fixture_journal(result, self.names)
                    write_json(bundle / "RESOURCE_BASELINE.json", {"status": evidence.PASS_STATUS,
                        "samples": result["resources"], "profiles": result["resource_profiles"]})
                    write_json(path, result)
                    (bundle / "EVENTS.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
                report, calls = self.inspect_fixture(package, bundle)
                self.assertEqual(report["status"], "BLOCKED")
                self.assertEqual(calls, 0)

    def test_actual_failed_and_resource_only_bundles_are_rejected_early(self):
        for name in ("e0_titan_v2", "e0_titan_v4", "e0_laptop_load_v1", "e0_laptop_full_v2", "e0_laptop_stream_load_v1"):
            with self.subTest(name=name), patch.object(evidence, "_verify_assets", side_effect=AssertionError("no root weights")), \
                 patch.object(evidence.low, "_verify_source", side_effect=AssertionError("nonpasses stop before source")):
                report = evidence.inspect_unpadded_scope(ROOT, e0_directory=ROOT / "results" / name)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["reasons"][0]["code"], "UNPADDED_NOT_PASSED")

    def test_duplicate_nonfinite_and_oversized_metadata_are_rejected(self):
        for raw in (b'{"status":"PASS_E0_UNPADDED_CE_SCOPED","status":"other"}',
                    b'{"status":"PASS_E0_UNPADDED_CE_SCOPED","wall_seconds":NaN}', b" " * (evidence.METADATA_LIMIT + 1)):
            with self.subTest(size=len(raw)), tempfile.TemporaryDirectory() as directory:
                bundle = Path(directory)
                (bundle / "E0_RESULT.json").write_bytes(raw)
                report = evidence.inspect_unpadded_scope(ROOT, e0_directory=bundle)
                self.assertEqual(report["status"], "BLOCKED")

    def test_checker_imports_no_native_qualifier_or_framework(self):
        code = "import sys; import experiments.post_training_mimo_centralized_v1.scripts.unpadded_evidence; assert not any(n == 'torch' or n.startswith('torch.') or n == 'transformers' or n.startswith('transformers.') or n.endswith('.qualify_native') for n in sys.modules)"
        run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_ce512_explicit_scope_has_independently_derived_counts_without_qualification(self):
        result, cost, events = self.bundle_values("unpadded_ce512")
        self.assertIsNone(evidence.validate_result(result, qualification_stage="unpadded_ce512"))
        self.assertIsNone(evidence.validate_accounting_bundle(result, cost, events, self.names,
                                                           qualification_stage="unpadded_ce512"))
        attempts = [event for event in events if event["event"] == "operation_attempt"]
        self.assertEqual(len(events), 865)
        self.assertEqual(len(attempts), 393)
        self.assertEqual(sum(event["operation"] == "forward" for event in attempts), 71)
        self.assertEqual(sum(event["operation"] == "backward" for event in attempts), 4)
        self.assertEqual(sum(event["operation"] == "optimizer_step" for event in attempts), 4)
        self.assertEqual(sum(event["input_positions"] for event in attempts), 26608)
        self.assertEqual(sum(event["input_positions"] for event in attempts if event["training"]), 1792)
        self.assertEqual(sum(event["event"] == "resource" for event in events), 75)
        self.assertEqual(sum(event["event"] == "resource_profile" for event in events), 2)
        self.assertEqual([profile["context"] for profile in result["resource_profiles"]], [256, 512])
        self.assertFalse(result["e1_authorized"])
        with self.assertRaises(evidence.GateError):
            evidence.validate_result(result)  # Historical default does not infer a successor.

    def test_ce512_and_historical_claims_cannot_cross_promote(self):
        for selected, supplied in (("unpadded_ce", "unpadded_ce512"), ("unpadded_ce512", "unpadded_ce")):
            with self.subTest(selected=selected), self.assertRaises(evidence.GateError):
                evidence.validate_result(fixture_result(supplied), qualification_stage=selected)
        cases = (("status", evidence.PASS_STATUS), ("runtime_scope", evidence.RUNTIME_SCOPE),
                 ("config_sha256", evidence.CONFIG_SHA256), ("qualification_stage", "unpadded_ce"),
                 ("optimizer_steps", 5), ("training_tokens", 3840),
                 ("unqualified", deepcopy(evidence.UNQUALIFIED)))
        for field, value in cases:
            result = fixture_result("unpadded_ce512")
            result[field] = value
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result, qualification_stage="unpadded_ce512")

    def test_ce512_rejects_513_and_1024_token_forward_or_profile_claims(self):
        for length in (513, 1024):
            result, cost, events = self.bundle_values("unpadded_ce512")
            for event in events:
                if event.get("stage") == "base_parity_1" and event.get("operation") == "forward":
                    event["input_positions"] = length
            with self.subTest(journal_tokens=length), self.assertRaises(evidence.GateError) as caught:
                evidence.validate_accounting_bundle(result, cost, events, self.names, qualification_stage="unpadded_ce512")
            self.assertEqual(caught.exception.code, "EVENTS_CONTRACT")
            result = fixture_result("unpadded_ce512")
            result["resource_profiles"][-1]["context"] = length
            with self.subTest(profile_tokens=length), self.assertRaises(evidence.GateError):
                evidence.validate_result(result, qualification_stage="unpadded_ce512")
        result, cost, events = self.bundle_values("unpadded_ce512")
        for event in events:
            if event.get("stage") == "resume_reloaded" and event.get("operation") == "forward":
                event["input_positions"] = 1024
        with self.assertRaises(evidence.GateError):
            evidence.validate_accounting_bundle(result, cost, events, self.names, qualification_stage="unpadded_ce512")

    def test_ce512_requires_its_complete_four_call_and_865_record_workload(self):
        for field, value in (("max_optimizer_calls", 5), ("last_call_id", 396),
                             ("forward_input_positions_returned", 41456),
                             ("training_forward_input_positions_attempted", 3840)):
            result = fixture_result("unpadded_ce512")
            result["call_accounting"][field] = value
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result, qualification_stage="unpadded_ce512")
        for mutation in ("drop", "duplicate", "extra_profile", "reordered_context"):
            result, cost, events = self.bundle_values("unpadded_ce512")
            if mutation == "drop":
                events.pop()
            elif mutation == "duplicate":
                events.append(deepcopy(events[-1]))
            elif mutation == "extra_profile":
                result["resource_profiles"].append(deepcopy(result["resource_profiles"][-1]))
            else:
                result["resource_profiles"].reverse()
            with self.subTest(mutation=mutation), self.assertRaises(evidence.GateError):
                evidence.validate_accounting_bundle(result, cost, events, self.names, qualification_stage="unpadded_ce512")

    def test_ce512_preserves_numerical_checks_resource_floors_and_timing_bounds(self):
        for check, field, value in (("explicit_allones_mask_equivalence", "mean_abs", .010001),
                                    ("native_unpadded_causality", "bitwise", False)):
            result = fixture_result("unpadded_ce512")
            result["checks"][check][field] = value
            with self.subTest(check=check), self.assertRaises(evidence.GateError):
                evidence.validate_result(result, qualification_stage="unpadded_ce512")
        for field, value in (("host_available_gib", 1.249), ("rss_gib", 1.751), ("gpu_free_gib", .499)):
            result = fixture_result("unpadded_ce512")
            result["resources"][1][field] = value
            with self.subTest(resource=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result, qualification_stage="unpadded_ce512")
        for field in ("load_seconds", "checkpoint_save_seconds", "checkpoint_reload_seconds", "profile"):
            result, cost, events = self.bundle_values("unpadded_ce512")
            if field == "profile":
                profile = result["resource_profiles"][-1]
                profile["seconds"] = .000001
                profile["tokens_per_second"] = profile["context"] / profile["seconds"]
                event = [event for event in events if event["event"] == "resource_profile"][-1]
                event["profile"] = deepcopy(profile)
            else:
                result[field] = .000001
            with self.subTest(timer=field), self.assertRaises(evidence.GateError) as caught:
                evidence.validate_accounting_bundle(result, fixture_cost(result), events, self.names,
                                                   qualification_stage="unpadded_ce512")
            self.assertEqual(caught.exception.code, "TIMING_JOURNAL_CONTRADICTION")

    def test_ce512_complete_synthetic_metadata_still_requires_all_asset_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            package, bundle = self.metadata_fixture(Path(directory), "unpadded_ce512")
            report, calls = self.inspect_fixture(package, bundle, "unpadded_ce512")
            self.assertEqual(calls, 1)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["reasons"][0]["code"], "SYNTHETIC_NO_ASSETS")
            self.assertFalse(report["native_launch_authorized"])
            self.assertFalse(report["e1_authorized"])
            self.assertTrue(report["no_native_executor"])
            with patch.object(evidence, "_verify_assets", side_effect=AssertionError("no silent successor")):
                default = evidence.inspect_unpadded_scope(package, e0_directory=bundle)
            self.assertEqual(default["status"], "BLOCKED")
            self.assertEqual(default["reasons"][0]["code"], "UNPADDED_NOT_PASSED")

    def test_ce512_launch_config_or_scope_drift_blocks_before_assets(self):
        for mutation in ("config", "launch_stage", "launch_hash", "stdout"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                package, bundle = self.metadata_fixture(Path(directory), "unpadded_ce512")
                supervision = bundle.with_name(bundle.name + "_supervisor")
                if mutation == "config":
                    (bundle / "CONFIG.json").write_bytes((ROOT / evidence.CONFIG_RELATIVE).read_bytes())
                elif mutation == "stdout":
                    (supervision / "stdout.log").write_text(json.dumps({"status": evidence.PASS_STATUS, "error": None,
                            "wall_seconds": 1000.0, "optimizer_steps": 5}) + "\n", encoding="utf-8")
                else:
                    path = supervision / "LAUNCH.json"
                    launch = json.loads(path.read_text())
                    launch["qualification_stage" if mutation == "launch_stage" else "config_sha256"] = (
                            "unpadded_ce" if mutation == "launch_stage" else evidence.CONFIG_SHA256)
                    write_json(path, launch)
                report, calls = self.inspect_fixture(package, bundle, "unpadded_ce512")
                self.assertEqual(report["status"], "BLOCKED")
                self.assertEqual(calls, 0)

    def test_ce512_cli_requires_explicit_selection_and_rejects_unknown_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            write_json(bundle / "E0_RESULT.json", fixture_result("unpadded_ce512"))
            command = [sys.executable, "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.unpadded_evidence",
                       "--experiment-root", str(ROOT), "--e0-directory", str(bundle)]
            default = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(default.returncode, 0, default.stderr)
            default_report = json.loads(default.stdout)
            self.assertEqual(default_report["reasons"][0]["code"], "UNPADDED_NOT_PASSED")
            selected = subprocess.run(command + ["--qualification-stage", "unpadded_ce512"], capture_output=True, text=True)
            self.assertEqual(selected.returncode, 0, selected.stderr)
            selected_report = json.loads(selected.stdout)
            self.assertEqual(selected_report["status"], "BLOCKED")
            self.assertNotEqual(selected_report["reasons"][0]["code"], "UNPADDED_NOT_PASSED")
            self.assertFalse(selected_report["e1_authorized"])
            invalid = subprocess.run(command + ["--qualification-stage", "unknown"], capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
        report = evidence.inspect_unpadded_scope(ROOT, qualification_stage="unknown")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["reasons"][0]["code"], "UNREGISTERED_QUALIFICATION_STAGE")


if __name__ == "__main__":
    unittest.main()
