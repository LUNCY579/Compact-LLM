"""Synthetic checker controls only; never native capture or model qualification."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import capture_evidence as evidence


ROOT = Path(__file__).resolve().parents[1]


def likelihood(online, offline):
    errors = [abs(left - right) for left, right in zip(online, offline)]
    return {"finite": True, "absolute_errors": errors, "max_abs": max(errors),
            "mean_abs": sum(errors) / len(errors), "passed": True}


def cpu_receipt():
    """Fixture values independently state the registered branch outcomes."""
    cases = []
    def add(name, inputs, expected):
        cases.append({"case_id": name, "designation": "SYNTHETIC_MECHANICS_ONLY_NOT_NATIVE_OR_FORCED_MODEL_EOS",
                      "input": inputs, "output": deepcopy(expected), "expected": expected, "passed": True,
                      "input_sha256": evidence.digest(inputs), "output_sha256": evidence.digest(expected)})
    add("raw_greedy_tie_lowest_id", {"scores": [-2.0, 4.0, 4.0]}, 1)
    for eos, marker in ((151645, "<|im_end|>"), (151643, "<|endoftext|>")):
        add("terminal_only_render_" + str(eos), {"prompt_ids": [eos, 10], "generated_ids": [7, 8, 9, eos],
                                                "cap": 128, "eos_ids": [151645, 151643]},
            {"stop": "eos", "visible_ids": [7, 8, 9], "raw": " \n<other-special> tail " + marker,
             "visible": " \n<other-special> tail "})
    add("prompt_EOS_does_not_stop_new_nonEOS", {"prompt_ids": [151645], "generated_ids": [7], "cap": 128}, None)
    add("EOS_at_cap_has_EOS_priority", {"generated_ids": [7] * 127 + [151643], "cap": 128}, "eos")
    add("exact_cap_stop_without_EOS", {"generated_ids": [7] * 128, "cap": 128}, "length")
    add("offline_Nplus1_firstN_shift", {"generated_length": 2, "kept_logits_rows": 3}, [0, 1])
    add("verified_greedy_behavior_zero_is_not_model_logprob",
        {"token_id": 1, "selected": 4.0, "maximum": 4.0, "first_maximum": 1, "model_logprob": -.5}, 0.0)
    return {"status": "PASS_CPU_CAPTURE_MECHANICS_ONLY", "cases": cases, "native_EOS_observations": [],
            "model_calls": 0, "training_authorized": False, "E1_authorized": False,
            "helper_sha256": "a" * 64, "test_source_sha256": "b" * 64}


def fixture(prompt_length, generated_length, eos=151645):
    """Complete producer field inventory, with deliberately synthetic token/text values."""
    prompt = [151645] + [10] * (prompt_length - 1)
    generated = [7] * generated_length
    if eos is not None:
        generated[-1] = eos
    visible = generated[:-1] if eos is not None else generated[:]
    text = " \n<other-special> tail " if visible else ""
    raw = text + ({151645: "<|im_end|>", 151643: "<|endoftext|>"}[eos] if eos is not None else "")
    online = [-.5] * generated_length
    offline = [-.501] * generated_length
    return {"fixture_id": "fixture_" + str(prompt_length), "prompt_tokens": prompt_length, "generation_cap": 128,
            "prompt_ids": prompt, "prompt_token_ids_sha256": evidence.digest(prompt),
            "generated_ids": generated, "generated_token_ids_sha256": evidence.digest(generated),
            "prefix_token_ids_sha256": [evidence.digest(prompt + generated[:i]) for i in range(generated_length)],
            "raw_argmax_evidence": [{"selected_raw_logit": 4.0, "raw_max_logit": 4.0,
                                      "first_maximum_token_id": token, "raw_argmax_verified": True} for token in generated],
            "online_model_logprobs": online, "offline_model_logprobs": offline,
            "behavior_logprobs": [0.0] * generated_length, "stop_reason": "eos" if eos else "length",
            "terminal_eos_id": eos, "visible_ids": visible, "visible_token_ids_sha256": evidence.digest(visible),
            "raw_response": raw, "visible_response": text,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "visible_response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "offline_input_token_ids_sha256": evidence.digest(prompt + generated),
            "offline_logits_to_keep": generated_length + 1, "offline_prediction_indices": list(range(generated_length)),
            "likelihood": likelihood(online, offline), "generation_seconds": 2.0,
            "generation_tokens_per_second": generated_length / 2.0,
            "generation_timing": "instrumented generation wall including durable token logging and resource guards",
            "offline_teacher_seconds": .1,
            "offline_timing": "instrumented teacher-forcing wall including selected model logprobs and resource guard"}


TENSORS = ["synthetic_tensor_" + str(i).zfill(3) for i in range(310)]


def synthetic_bundle(first=3, second=5, *, eos_first=151645, eos_second=151643):
    """Independent source of the full serialized worker/result/journal shapes.

    These values exercise a parser only: no native execution, real source or
    asset closure is represented, and they must never pass inspect_capture_scope.
    """
    fixtures = [fixture(128, first, eos_first), fixture(384, second, eos_second)]
    events, resources = [], []
    call_id = 0
    positions = 0
    def emit(kind, **values):
        event = {"event": kind, "elapsed_seconds": (len(events) + 1) / 1000, **values}
        events.append(event)
    def resource(stage):
        sample = {"stage": stage, "elapsed_seconds": (len(events) + 1) / 1000,
                  "host_available_gib": 4.0, "rss_gib": 1.0, "gpu_free_gib": 5.0,
                  "gpu_allocated_gib": 0.0 if stage == "preload" else 1.2,
                  "gpu_reserved_gib": 1.5, "peak_allocated_gib": 1.5,
                  "disk_free_gib": 10.0, "admission_required_gib": 2.25}
        resources.append(sample)
        emit("resource", **sample)
    def attempt(operation, stage, input_positions=0):
        nonlocal call_id, positions
        call_id += 1
        if operation == "forward":
            positions += input_positions
        record = {"accounting_version": 2, "call_id": call_id, "operation": operation, "stage": stage,
                  "input_positions": input_positions, "training": False}
        emit("operation_attempt", **record)
        return record
    def op(operation, stage, input_positions=0):
        record = attempt(operation, stage, input_positions)
        emit("operation_returned", **record)
    emit("runtime_import_start")
    emit("runtime_import_complete")
    resource("preload")
    op("model_construct", "model_construct")
    load = attempt("checkpoint_load", "checkpoint_load")
    for name in TENSORS:
        op("tensor_copy", "tensor_copy:" + name)
    emit("operation_returned", **load)
    resource("loaded")
    for length in (256, 512):
        op("forward", "finite_prefix_probe_" + str(length), length)
        resource("finite_prefix_probe_" + str(length))
    for row in fixtures:
        p = row["prompt_tokens"]
        emit("fixture_start", fixture_id=row["fixture_id"], prompt_tokens=p, generation_cap=128,
             prompt_token_ids_sha256=row["prompt_token_ids_sha256"])
        for i, token in enumerate(row["generated_ids"]):
            stage = "greedy_" + str(p) + "_" + str(i)
            op("forward", stage, p + i)
            emit("generated_token", fixture_id=row["fixture_id"], prompt_tokens=p, ordinal=i,
                 prefix_token_ids_sha256=row["prefix_token_ids_sha256"][i], token_id=token,
                 model_logprob=row["online_model_logprobs"][i], behavior_logprob=0.0, **row["raw_argmax_evidence"][i])
            resource(stage)
        op("forward", "teacher_force_" + str(p), p + len(row["generated_ids"]))
        resource("teacher_force_" + str(p))
        emit("fixture_complete", fixture_id=row["fixture_id"], generated_tokens=len(row["generated_ids"]),
             stop_reason=row["stop_reason"], terminal_eos_id=row["terminal_eos_id"], likelihood=row["likelihood"],
             generated_token_ids_sha256=row["generated_token_ids_sha256"], raw_response_sha256=row["raw_response_sha256"],
             visible_response_sha256=row["visible_response_sha256"])
    resource("complete")
    def time_of(kind, stage=None, fixture_id=None):
        return next(event["elapsed_seconds"] for event in events if event["event"] == kind
                    and (stage is None or event.get("stage") == stage)
                    and (fixture_id is None or event.get("fixture_id") == fixture_id))
    for row in fixtures:
        row["generation_seconds"] = (time_of("resource", "greedy_" + str(row["prompt_tokens"]) + "_" + str(len(row["generated_ids"]) - 1))
                                     - time_of("fixture_start", fixture_id=row["fixture_id"]) + .0001)
        row["generation_tokens_per_second"] = len(row["generated_ids"]) / row["generation_seconds"]
        row["offline_teacher_seconds"] = .0011
    load_seconds = time_of("operation_returned", "checkpoint_load") - time_of("operation_attempt", "model_construct") + .0001
    total = first + second
    operation_counts = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": 4 + total,
                        "backward": 0, "optimizer_step": 0, "checkpoint_save": 0, "checkpoint_reload": 0}
    accounting = {"accounting_version": 2, "max_optimizer_calls": 0, "last_call_id": call_id,
                  "operations": {name: {"attempted": count, "returned": count, "failed": 0, "pending": 0}
                                 for name, count in operation_counts.items()}, "pending_calls": [],
                  "forward_input_positions_attempted": positions, "forward_input_positions_returned": positions,
                  "training_forward_input_positions_attempted": 0, "training_forward_input_positions_returned": 0}
    errors = [value for row in fixtures for value in row["likelihood"]["absolute_errors"]]
    eos = sorted({value for value in (eos_first, eos_second) if value is not None})
    checks = {name: True for name in ("exact_native_load", "bare_frozen_eval", "fixture_input_boundaries",
              "greedy_selected_raw_argmax", "eos_stop_accounting", "response_rendering", "frozen_base_unchanged",
              "source_assets_unchanged")}
    checks.update(finite_prefix_probes={"contexts": [256, 512], "logits_to_keep": 1, "finite": True},
                  causal_shift_model_logprobs={"fixtures": 2, "finite": True, "max_abs": max(errors),
                                              "mean_abs": sum(errors) / len(errors)},
                  native_eos_coverage={"observed": bool(eos), "terminal_eos_ids": eos})
    result = {"schema_version": 1, "experiment_id": "RINEYA-0P6B-CAPTURE-QUAL-SHORT-V1",
              "status": "PASS_FROZEN_GREEDY_SHORT_CAPTURE_SCOPED", "runtime_scope": "FROZEN_GREEDY_PREFIX_CE512_CAPTURE_V1",
              "resource_profile": "LAPTOP_CAPTURE_RESIDENT_V1", "config_sha256": evidence.CONFIG_SHA256,
              "code_sha": "a" * 40, "dirty_at_launch": False, "evidence_level": "MEASURED / REPRODUCED", "seed": 20260928,
              "qualified_CE512_reference_sha256": evidence.CORE_REFERENCE_SHA256, "pretrained_loads": 1,
              "optimizer_steps": 0, "training_tokens": 0, "generation_tokens": total, "teacher_calls": 0,
              "accounting_version": 2, "call_accounting": accounting, "fixtures": fixtures, "checks": checks,
              "resources": resources, "native_evaluation_bank_access": False, "language_grading_authorized": False,
              "E1_authorized": False, "CHECK_FINAL_access": False, "load_seconds": load_seconds, "loader_diagnostics": {},
              "wall_seconds": 10.0,
              "actual_counts": {"generated_tokens": total, "forward_calls": 4 + total, "logical_calls": call_id,
                                "forward_input_positions": positions, "resource_samples": len(resources), "journal_records": len(events)},
              "unqualified": ["DEV/source enrollment", "long context beyond512", "adapter capture", "E1/training",
                              "human language grading", "stochastic/probability-based RL", "published-default generation"]}
    cost = {name: result[name] for name in ("wall_seconds", "pretrained_loads", "optimizer_steps", "training_tokens",
                                         "generation_tokens", "teacher_calls", "accounting_version", "call_accounting")}
    return result, deepcopy(cost), events


class CaptureEvidenceControls(unittest.TestCase):
    def test_complete_actual_field_inventory_and_independent_trace_reconstruction(self):
        for a, b, eos_a, eos_b in ((1, 1, 151645, 151643), (4, 17, 151643, 151645),
                                  (128, 3, None, 151645), (3, 128, 151643, None),
                                  (128, 128, 151645, 151643)):
            with self.subTest(a=a, b=b):
                result, cost, events = synthetic_bundle(a, b, eos_first=eos_a, eos_second=eos_b)
                self.assertEqual(evidence.validate_result(result), [a, b])
                self.assertIsNone(evidence.validate_accounting_bundle(result, cost, events, TENSORS))
                self.assertEqual(len(events), 645 + 4 * (a + b))
                self.assertEqual(result["fixtures"][0]["generation_tokens_per_second"], a / result["fixtures"][0]["generation_seconds"])
                self.assertNotIn("tokens_per_second", result["fixtures"][0])

    def test_timing_schema_is_exact_current_producer_not_previous_draft(self):
        for mutate in (lambda r: r["fixtures"][0].__setitem__("tokens_per_second", 1),
                       lambda r: r["fixtures"][0].pop("generation_tokens_per_second"),
                       lambda r: r["fixtures"][0].pop("generation_timing"),
                       lambda r: r["fixtures"][0].pop("offline_timing"),
                       lambda r: r["fixtures"][0].__setitem__("generation_timing", "GPU active time"),
                       lambda r: r["fixtures"][0].__setitem__("generation_tokens_per_second", 10),
                       lambda r: r["fixtures"][0].__setitem__("offline_teacher_seconds", 0)):
            result, _, _ = synthetic_bundle()
            mutate(result)
            with self.assertRaises(evidence.GateError):
                evidence.validate_result(result)

    def test_eos_first_and_eos_at_cap_preserve_terminal_id(self):
        for n in (1, 128):
            for eos in (151645, 151643):
                row = fixture(128, n, eos)
                self.assertEqual(evidence._fixture(row, 128, 10), n)
                self.assertEqual(row["generated_ids"][-1], eos)
                self.assertEqual(len(row["visible_ids"]), n - 1)
                if n == 1:
                    self.assertEqual(row["visible_response"], "")
                row["stop_reason"] = "length"
                with self.assertRaises(evidence.GateError):
                    evidence._fixture(row, 128, 10)

    def test_prompt_eos_ignored_but_generated_eos_ends_sequence(self):
        row = fixture(128, 128, None)
        self.assertIn(151645, row["prompt_ids"])
        self.assertEqual(evidence._fixture(row, 128, 10), 128)
        row["generated_ids"][5] = 151645
        row["generated_token_ids_sha256"] = evidence.digest(row["generated_ids"])
        with self.assertRaisesRegex(evidence.GateError, "after first generated EOS"):
            evidence._fixture(row, 128, 10)
        with self.assertRaisesRegex(evidence.GateError, "reach the cap"):
            evidence._fixture(fixture(128, 3, None), 128, 10)

    def test_cpu_coverage_cannot_substitute_for_natural_native_eos(self):
        evidence.validate_cpu_controls(cpu_receipt())
        result, _, _ = synthetic_bundle(128, 128, eos_first=None, eos_second=None)
        with self.assertRaisesRegex(evidence.GateError, "natural native EOS"):
            evidence.validate_result(result)
        result["checks"]["native_eos_coverage"] = {"observed": True, "terminal_eos_ids": [151645]}
        with self.assertRaises(evidence.GateError):
            evidence.validate_result(result)

    def test_unattempted_missing_extra_and_reordered_fixtures_rejected(self):
        for mutate in (lambda r: r["fixtures"].pop(), lambda r: r["fixtures"].append(deepcopy(r["fixtures"][0])),
                       lambda r: r["fixtures"].reverse(), lambda r: r["fixtures"][0].__setitem__("generated_ids", []),
                       lambda r: r.__setitem__("status", "RUNNING"),
                       lambda r: r.__setitem__("status", "BLOCKED_CAPTURE_QUALIFICATION")):
            result, _, _ = synthetic_bundle(); mutate(result)
            with self.assertRaises(evidence.GateError):
                evidence.validate_result(result)

    def test_prefix_and_offline_shift_have_independent_id_bindings(self):
        changes = (("prompt_token_ids_sha256", "b" * 64), ("generated_token_ids_sha256", "b" * 64),
                   ("prefix_token_ids_sha256", ["b" * 64] * 3), ("offline_input_token_ids_sha256", "b" * 64),
                   ("offline_logits_to_keep", 3), ("offline_prediction_indices", [1, 2, 3]),
                   ("offline_prediction_indices", [False, 1, 2]))
        for key, value in changes:
            row = fixture(128, 3); row[key] = value
            with self.subTest(key=key), self.assertRaises(evidence.GateError):
                evidence._fixture(row, 128, 10)

    def test_raw_argmax_and_behavior_claims_checked_separately_from_raw_model_nll(self):
        for key, value in (("selected_raw_logit", 3.0), ("first_maximum_token_id", 8),
                           ("first_maximum_token_id", True), ("raw_argmax_verified", False),
                           ("raw_max_logit", float("inf"))):
            row = fixture(128, 3); row["raw_argmax_evidence"][0][key] = value
            with self.subTest(key=key), self.assertRaises(evidence.GateError):
                evidence._fixture(row, 128, 10)
        for value in (-.5, True, float("nan")):
            row = fixture(128, 3); row["behavior_logprobs"][0] = value
            with self.assertRaises(evidence.GateError):
                evidence._fixture(row, 128, 10)

    def test_only_terminal_extraction_is_accepted_and_hashes_preserve_raw_text(self):
        row = fixture(128, 3)
        self.assertIn(" \n<other-special> tail ", row["raw_response"])
        evidence._fixture(row, 128, 10)
        for field in ("raw_response", "visible_response"):
            changed = deepcopy(row); changed[field] = changed[field].strip()
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence._fixture(changed, 128, 10)
        changed = deepcopy(row); changed["visible_ids"] = []
        changed["visible_token_ids_sha256"] = evidence.digest([])
        with self.assertRaises(evidence.GateError):
            evidence._fixture(changed, 128, 10)
        # Hash self-consistency does not purport to independently detokenize IDs.
        changed = fixture(128, 128, None); changed["visible_response"] = "changed"
        changed["visible_response_sha256"] = hashlib.sha256(b"changed").hexdigest()
        with self.assertRaises(evidence.GateError):
            evidence._fixture(changed, 128, 10)

    def test_fixed_probe_and_aggregate_metrics_are_not_expanded_or_self_asserted(self):
        for key, value in (("finite_prefix_probes", {"contexts": [256, 512], "logits_to_keep": 512, "finite": True}),
                           ("causal_shift_model_logprobs", {"fixtures": 2, "finite": True, "max_abs": 0.0, "mean_abs": 0.0}),
                           ("greedy_selected_raw_argmax", 1), ("response_rendering", False)):
            result, _, _ = synthetic_bundle(); result["checks"][key] = value
            with self.subTest(key=key), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)

    def test_ceilings_and_bool_values_cannot_replace_actual_completed_counts(self):
        for mutate in (lambda r: r.__setitem__("generation_tokens", 256),
                       lambda r: r["actual_counts"].__setitem__("forward_calls", 260),
                       lambda r: r["call_accounting"]["operations"]["forward"].__setitem__("pending", 1),
                       lambda r: r["call_accounting"].__setitem__("pending_calls", [{"call_id": 1}]),
                       lambda r: r.__setitem__("pretrained_loads", True),
                       lambda r: r.__setitem__("teacher_calls", 1),
                       lambda r: r.__setitem__("E1_authorized", True),
                       lambda r: r.__setitem__("optimizer_steps", 1)):
            result, _, _ = synthetic_bundle(); mutate(result)
            with self.assertRaises(evidence.GateError):
                evidence.validate_result(result)

    def test_resource_floor_admission_and_scope_cannot_be_weakened(self):
        for index, field, value in ((0, "host_available_gib", 2.24), (1, "host_available_gib", 1.24),
                                    (0, "gpu_free_gib", 3.99), (1, "gpu_free_gib", .49),
                                    (1, "rss_gib", 2.01), (1, "disk_free_gib", 3.99),
                                    (1, "admission_required_gib", 0), (1, "elapsed_seconds", 100),
                                    (1, "stage", "loaded_other")):
            result, _, _ = synthetic_bundle(); result["resources"][index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(evidence.GateError):
                evidence.validate_result(result)

    def test_cost_and_full_journal_must_preserve_each_attempt_return_and_token(self):
        result, cost, original = synthetic_bundle()
        for mutation in ("drop", "duplicate", "swap", "fail", "pending", "prefix", "token", "bool", "resource_time"):
            events = deepcopy(original)
            token_index = next(i for i, event in enumerate(events) if event["event"] == "generated_token")
            if mutation == "drop": events.pop(token_index)
            if mutation == "duplicate": events.insert(token_index, deepcopy(events[token_index]))
            if mutation == "swap": events[token_index - 1], events[token_index] = events[token_index], events[token_index - 1]
            if mutation == "fail": events[4]["event"] = "operation_failed"
            if mutation == "pending": events[4]["event"] = "operation_attempt"
            if mutation == "prefix": events[token_index]["prefix_token_ids_sha256"] = "f" * 64
            if mutation == "token": events[token_index]["token_id"] = 8
            if mutation == "bool": events[token_index]["behavior_logprob"] = False
            if mutation == "resource_time": events[2]["elapsed_seconds"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(evidence.GateError):
                evidence.validate_accounting_bundle(result, cost, events, TENSORS)
        changed = deepcopy(cost); changed["generation_tokens"] = 256
        with self.assertRaises(evidence.GateError):
            evidence.validate_accounting_bundle(result, changed, original, TENSORS)

    def test_reported_timing_cannot_be_shorter_than_observed_journal_intervals(self):
        for target in ("load_seconds", "generation_seconds", "offline_teacher_seconds"):
            result, cost, events = synthetic_bundle()
            if target == "load_seconds": result[target] = .000001
            else:
                result["fixtures"][0][target] = .000001
                if target == "generation_seconds": result["fixtures"][0]["generation_tokens_per_second"] = 3000000.0
            with self.subTest(target=target), self.assertRaisesRegex(evidence.GateError, "duration outside|fixture_128"):
                evidence.validate_accounting_bundle(result, cost, events, TENSORS)

    def test_reported_timing_cannot_exceed_enclosing_journal_intervals(self):
        for target in ("load_seconds", "generation_seconds", "offline_teacher_seconds"):
            result, cost, events = synthetic_bundle()
            if target == "load_seconds": result[target] = 9.0
            else:
                result["fixtures"][0][target] = 9.0
                if target == "generation_seconds": result["fixtures"][0]["generation_tokens_per_second"] = 3 / 9.0
            with self.subTest(target=target), self.assertRaisesRegex(evidence.GateError, "duration outside|fixture_128"):
                evidence.validate_accounting_bundle(result, cost, events, TENSORS)

    def test_failed_partial_and_synthetic_bundles_stop_before_any_asset_read(self):
        complete, _, _ = synthetic_bundle()
        samples = ({"status": "RUNNING"}, {"status": "BLOCKED_CAPTURE_QUALIFICATION", "error": "fixture unattempted"},
                   {"status": evidence.PASS_STATUS}, complete)
        for result in samples:
            with self.subTest(status=result["status"], fields=len(result)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "CAPTURE_RESULT.json").write_text(json.dumps(result), encoding="utf-8")
                with patch.object(evidence, "_verify_assets", side_effect=AssertionError("asset read forbidden")) as asset:
                    report = evidence.inspect_capture_scope(ROOT, capture_directory=path)
                asset.assert_not_called()
                self.assertEqual(report["status"], "BLOCKED")
                self.assertTrue(report["reasons"])
                for key in ("native_launch_authorized", "native_capture_authorized", "DEV_collection_authorized",
                            "E1_authorized", "e1_authorized", "language_grading_authorized", "training_authorized",
                            "independent_detokenization"):
                    self.assertIs(report[key], False)
                self.assertTrue(report["no_native_executor"])

    def test_registered_producer_pins_match_final_reviewed_source_bytes(self):
        for relative, pin in evidence.PRODUCER_PINS.items():
            raw = (ROOT.parents[1] / relative).read_bytes()
            self.assertEqual({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, pin)

    def test_real_committed_source_closure_without_producer_or_framework_imports(self):
        raw = (ROOT / evidence.CONFIG_RELATIVE).read_bytes()
        cfg = json.loads(raw)
        source = {"code_sha": "f3c6d723e094d2fe4dc892c8c07f32df820a7dd9", "dirty": False,
                  "source_files": {**cfg["reused_source_files"], **evidence.PRODUCER_PINS,
                                   evidence.PACKAGE + evidence.CONFIG_RELATIVE:
                                       {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}},
                  "installed_runtime_source_files": cfg["installed_runtime_source_files"]}
        bound = {}
        evidence._validate_source(source, {"code_sha": source["code_sha"]}, cfg, ROOT.parents[1], bound)
        self.assertEqual(len(source["source_files"]), 12)
        self.assertEqual(len(bound), 14)  # Twelve repo files plus two installed Python source files.
        self.assertFalse(any(path.endswith("model.safetensors") for path in bound))

    def test_source_closure_rejects_extra_missing_and_new_committed_producer_claims(self):
        cfg = json.loads((ROOT / evidence.CONFIG_RELATIVE).read_text(encoding="utf-8"))
        config_raw = (ROOT / evidence.CONFIG_RELATIVE).read_bytes()
        pins = {**cfg["reused_source_files"], **evidence.PRODUCER_PINS,
                evidence.PACKAGE + evidence.CONFIG_RELATIVE: {"sha256": evidence.CONFIG_SHA256, "bytes": len(config_raw)}}
        source = {"code_sha": "a" * 40, "dirty": False, "source_files": pins,
                  "installed_runtime_source_files": cfg["installed_runtime_source_files"]}
        for mode in ("missing", "extra", "producer_drift", "reused_drift", "dirty", "installed_drift"):
            changed = deepcopy(source)
            if mode == "missing": changed["source_files"].pop(evidence.CPU_TEST_SOURCE)
            if mode == "extra": changed["source_files"]["scripts/unregistered.py"] = {"sha256": "b" * 64, "bytes": 10}
            if mode == "producer_drift": changed["source_files"][evidence.NEW_SOURCE_FILES[0]]["sha256"] = "b" * 64
            if mode == "reused_drift": changed["source_files"][next(iter(cfg["reused_source_files"]))]["sha256"] = "b" * 64
            if mode == "dirty": changed["dirty"] = True
            if mode == "installed_drift": changed["installed_runtime_source_files"] = {}
            bound = {}
            def source_check(repo, code, relative, sha, output):
                output[str((repo / relative).resolve())] = deepcopy(pins[relative])
            with self.subTest(mode=mode), patch.object(evidence.low, "_verify_source", side_effect=source_check), \
                    patch.object(evidence, "_bind_binary", side_effect=AssertionError("installed read not needed")), \
                    self.assertRaises(evidence.GateError):
                evidence._validate_source(changed, {"code_sha": "a" * 40}, cfg, ROOT.parents[1], bound)

    def test_runtime_and_tokenizer_receipts_cannot_expand_qualified_scope(self):
        cfg = {"runtime": {"executable": evidence.EXECUTABLE}}
        runtime = {"python": "3.11.15 synthetic", "executable": evidence.EXECUTABLE, "platform": "Windows-synthetic",
                   "packages": deepcopy(evidence.PACKAGES), "gpu": "synthetic GPU", "cuda": "12.6", "total_vram_bytes": 8 * 2**30,
                   "attention": "eager", "base_dtype": "bfloat16", "adapter": None, "use_cache": False,
                   "inference_context": "torch.no_grad()", "torch_threads": 2, "allow_tf32": False, "deterministic_algorithms": True}
        evidence._validate_runtime(runtime, cfg)
        for field, value in (("attention", "sdpa"), ("use_cache", True), ("adapter", "lora"), ("base_dtype", "float16"),
                             ("torch_threads", True), ("allow_tf32", True), ("total_vram_bytes", True), ("python", "3.12.0 x"),
                             ("executable", "other.exe"), ("packages", {})):
            changed = deepcopy(runtime); changed[field] = value
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence._validate_runtime(changed, cfg)
        tokenizer = {"class_name": "Qwen2TokenizerFast", "template_sha256": evidence.low.TEMPLATE_SHA256,
                     "eos_id": 151645, "original_pad_id": 151643, "raw_ids": deepcopy(evidence.RAW_TOKEN_IDS),
                     "raw_roundtrip": True, "enable_thinking": False, "add_generation_prompt": True,
                     "skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        evidence._validate_tokenizer(tokenizer)
        for field, value in (("enable_thinking", True), ("skip_special_tokens", True), ("clean_up_tokenization_spaces", True),
                             ("add_generation_prompt", False), ("template_sha256", "a" * 64), ("raw_ids", [7])):
            changed = deepcopy(tokenizer); changed[field] = value
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence._validate_tokenizer(changed)

    def test_loader_model_receipts_bind_base_only_and_all_310_tensor_records(self):
        # These small pre-existing static receipts contain no tensor payloads.
        primary = json.loads((ROOT / "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json").read_text(encoding="utf-8"))
        inventory = json.loads((ROOT / "receipts/titan_static_v1/PARAMETER_INVENTORY.json").read_text(encoding="utf-8"))
        diagnostics = {"loader_mode": "stream_read_2MiB_v1", "header_bytes": 35216, "file_bytes": 1192135064,
                       "payload_bytes": 1192099840, "serialized_tensors": 310, "read_buffer_bytes": 2097152,
                       "maximum_payload_read_bytes": 2097152, "payload_chunk_count": 682, "memory_mapped": False,
                       "cpu_model_allocated": False, "blocking_chunk_copies": True}
        model = {"weight_sha256": evidence.low.WEIGHT_SHA256, "primary_receipt_sha256": evidence.low.PRIMARY_SHA256,
                 "unique_parameters": 596049920, "serialized_tensors": 310, "tied": True, "trainable_parameters": 0,
                 "adapter": None, "tensor_sha256": {name: "a" * 64 for name in inventory["tensors"]},
                 "loader": evidence.LOADER_LABEL, "loader_mode": "stream_read_2MiB_v1", "loader_diagnostics": diagnostics,
                 "config": deepcopy(primary["architecture"])}
        result = {"loader_diagnostics": diagnostics}
        evidence._validate_model(model, primary, inventory, result)
        for field, value in (("weight_sha256", "b" * 64), ("trainable_parameters", 1), ("adapter", {}),
                             ("tied", 1), ("tensor_sha256", {}), ("loader_diagnostics", {}), ("loader_mode", "mmap")):
            changed = deepcopy(model); changed[field] = value
            with self.subTest(field=field), self.assertRaises(evidence.GateError):
                evidence._validate_model(changed, primary, inventory, result)
        changed = deepcopy(model); changed["config"]["rope_theta"] = 10000
        with self.assertRaises(evidence.GateError):
            evidence._validate_model(changed, primary, inventory, result)

    def test_supervisor_contract_does_not_grant_native_authority_or_relax_resources(self):
        cfg = json.loads((ROOT / evidence.CONFIG_RELATIVE).read_text(encoding="utf-8"))
        result, _, _ = synthetic_bundle()
        path = ROOT / "results/synthetic_capture_no_native"
        runtime = {"executable": evidence.EXECUTABLE}
        source = {"source_files": evidence.PRODUCER_PINS}
        supervisor = {"status": "PASS_PROCESS_ONLY", "reason": None, "exit_code": 0, "native_qualification": False,
                      "elapsed_seconds": 11.0, "limit_seconds": 600, "memory_poll_seconds": .1,
                      "peak_observed_process_tree_rss_bytes": 2**30}
        launch = {"command": [evidence.EXECUTABLE, "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.qualify_capture",
                              "--config", str(ROOT / evidence.CONFIG_RELATIVE), "--output", str(path), "--go-path",
                              str(path.with_name(path.name + "_supervisor") / "GO.json")],
                  "config_sha256": evidence.CONFIG_SHA256, "source_sha": result["code_sha"],
                  "supervisor_sha256": evidence.PRODUCER_PINS[evidence.NEW_SOURCE_FILES[2]]["sha256"],
                  "watchdog_sha256": cfg["reused_source_files"][evidence.low.WATCHDOG]["sha256"],
                  "resource_limits": {"resource_profile": evidence.RESOURCE_PROFILE, **evidence.RESOURCE_LIMITS},
                  "observed_gpu_processes": []}
        go = {"config_sha256": evidence.CONFIG_SHA256}
        def check(sup, run):
            with patch.object(evidence.low, "_verify_source"):
                evidence._validate_supervisor(sup, run, go, runtime, result, cfg, source, ROOT / evidence.CONFIG_RELATIVE,
                                              path, ROOT.parents[1], {})
        check(supervisor, launch)
        for field, value in (("native_qualification", True), ("exit_code", True), ("elapsed_seconds", 9),
                             ("memory_poll_seconds", 1), ("peak_observed_process_tree_rss_bytes", 3 * 2**30)):
            changed = deepcopy(supervisor); changed[field] = value
            with self.subTest(field=field), self.assertRaises(evidence.GateError): check(changed, launch)
        for mode in ("RSS", "worker", "config", "source", "competitor"):
            changed = deepcopy(launch)
            if mode == "RSS": changed["resource_limits"]["worker_rss_limit_gib"] = 3.0
            if mode == "worker": changed["command"][3] = "arbitrary.executor"
            if mode == "config": changed["config_sha256"] = "b" * 64
            if mode == "source": changed["supervisor_sha256"] = "b" * 64
            if mode == "competitor": changed["observed_gpu_processes"] = ["1234, python.exe"]
            with self.subTest(mode=mode), self.assertRaises(evidence.GateError): check(supervisor, changed)

    def test_actual_lengths_determine_counts_instead_of_upper_ceiling(self):
        for first, second in ((1, 1), (4, 17), (128, 1), (1, 128), (128, 128)):
            with self.subTest(first=first, second=second):
                result = evidence.successful_accounting([first, second])
                expanded = 256 + 512 + sum(128 + i for i in range(first)) + sum(384 + i for i in range(second))
                expanded += 128 + first + 384 + second
                self.assertEqual(result["forward_input_positions_attempted"], expanded)
                self.assertEqual(result["forward_input_positions_returned"], expanded)
                self.assertEqual(result["operations"]["forward"]["returned"], 4 + first + second)
                self.assertEqual(result["last_call_id"], 316 + first + second)
                self.assertEqual(result["operations"]["optimizer_step"]["attempted"], 0)
                self.assertEqual(result["max_optimizer_calls"], 0)
        maximum = evidence.successful_accounting([128, 128])
        self.assertEqual(maximum["last_call_id"], 572)
        self.assertEqual(maximum["forward_input_positions_attempted"], 83328)
        self.assertEqual(maximum["operations"]["forward"]["returned"], 260)

    def test_unattempted_lengths_and_bool_counts_are_not_completed_records(self):
        for lengths in ([0, 1], [1, 0], [True, 1], [1, 129], [1], [1, 2, 3], [1.0, 2], None):
            with self.subTest(lengths=lengths), self.assertRaises(evidence.GateError):
                evidence.successful_accounting(lengths)

    def test_raw_logprob_errors_recomputed_and_fixed_thresholds_enforced(self):
        online, offline = [-.5, -1.0], [-.504, -.998]
        measured = likelihood(online, offline)
        self.assertEqual(evidence._likelihood(online, offline, measured)["max_abs"], max(measured["absolute_errors"]))
        for key, changed in (("absolute_errors", [0.0, 0.0]), ("max_abs", 0.0), ("mean_abs", 0.0),
                             ("passed", False), ("finite", 1)):
            report = deepcopy(measured); report[key] = changed
            with self.subTest(key=key), self.assertRaises(evidence.GateError):
                evidence._likelihood(online, offline, report)
        for offline in ([-.54, -1.0], [-.515, -1.015]):
            with self.assertRaises(evidence.GateError):
                evidence._likelihood(online, offline, likelihood(online, offline))

    def test_nonfinite_positive_or_mismatched_raw_logprobs_rejected(self):
        for online, offline in (([float("nan")], [-.5]), ([-.5], [float("inf")]), ([.1], [-.5]),
                                ([-.5], []), ([True], [-.5]), ([], [])):
            with self.subTest(online=online, offline=offline), self.assertRaises(evidence.GateError):
                evidence._likelihood(online, offline, {})

    def test_cpu_controls_are_literal_mechanics_not_native_eos_observations(self):
        receipt = cpu_receipt()
        self.assertIsNone(evidence.validate_cpu_controls(receipt))
        self.assertEqual(receipt["native_EOS_observations"], [])
        for key, value in (("native_EOS_observations", [151645]), ("model_calls", True),
                           ("training_authorized", True), ("E1_authorized", True)):
            changed = deepcopy(receipt); changed[key] = value
            with self.subTest(key=key), self.assertRaises(evidence.GateError):
                evidence.validate_cpu_controls(changed)

    def test_cpu_control_expected_values_cannot_be_rewritten_to_match_bad_output(self):
        original = cpu_receipt()
        for index, output in ((0, 2), (1, {"stop": "eos", "visible_ids": [7, 8, 9], "raw": "filtered", "visible": "filtered"}),
                              (3, "eos"), (4, "length"), (6, [1, 2]), (7, -.5)):
            changed = deepcopy(original)
            case = changed["cases"][index]
            case.update(expected=deepcopy(output), output=output, output_sha256=evidence.digest(output))
            with self.subTest(index=index), self.assertRaises(evidence.GateError):
                evidence.validate_cpu_controls(changed)
        changed = deepcopy(original); changed["cases"].pop()
        with self.assertRaises(evidence.GateError):
            evidence.validate_cpu_controls(changed)

    def test_import_has_no_qualifier_framework_tokenizer_or_environment_side_effects(self):
        code = ("import os,sys;before=dict(os.environ);"
                "from experiments.post_training_mimo_centralized_v1.scripts import capture_evidence;"
                "assert before==dict(os.environ);"
                "assert not any(n.startswith(('torch','transformers','tokenizers')) or n.endswith("
                "('qualify_native','qualify_capture','greedy_capture','stream_checkpoint')) for n in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT.parents[1],
                                env=dict(os.environ), capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
