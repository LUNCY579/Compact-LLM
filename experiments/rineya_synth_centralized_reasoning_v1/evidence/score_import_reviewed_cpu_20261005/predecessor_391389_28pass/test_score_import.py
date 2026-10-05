"""Synthetic CPU controls; captured responses are never graded."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import score_import as adapter

PACKAGE = Path(__file__).resolve().parents[2]


def fixture():
    parents = [{"parent_id": f"fixture_{skill}_{i}", "component_id": f"component_{skill}_{i}",
        "skill": skill, "stratum": "synthetic", "source_hash": adapter.digest([skill, i])}
        for skill in ("R", "S", "RS") for i in range(32)]
    panel = adapter._seal({"selected_parents": parents, "shortages": []})
    prompts, inputs = [], []
    for parent in parents:
        for condition in (0, 1):
            for rendering in ("familiar", "heldout_ledger"):
                row_id = adapter.digest({"panel": panel["sha256"], "parent": parent["parent_id"],
                    "condition": condition, "rendering": rendering})
                messages = [{"role": "user", "content": "CPU synthetic " + row_id}]
                pin = adapter.digest(messages)
                prompts.append({"row_id": row_id, "messages": messages, "prompt_sha256": pin})
                inputs.append({"row_id": row_id, "prompt_sha256": pin, "prompt_ids": [1],
                    "prompt_tokens": 1, "attention_mask": [1]})
    public = {"rows": prompts, "panel_sha256": panel["sha256"]}
    tokens = adapter._seal({"rows": inputs, "source_packet_sha256": adapter.digest(public),
        "context_cap": 512, "maximum_generated_tokens": 32, "rows_dropped": 0,
        "rows_truncated": 0, "overlong_row_ids": []})
    ref = {"path": "CPU_SYNTHETIC_NOT_NATIVE_AUTHORITY", "bytes": 1, "sha256": "a" * 64}
    registration = adapter._panel(panel, public, tokens,
        {"synthetic": ref}, {n: ref for n in adapter.SCORER_FILES},
        {n: ref for n in adapter.STATISTICS_FILES}, adapter.FIXTURE)
    batches = []
    for ordinal in range(1, 7):
        source = inputs[(ordinal - 1) * 64:ordinal * 64]; packet_hash = adapter.digest(source)
        outputs = [{"row_id": r["row_id"], "prompt_sha256": r["prompt_sha256"],
            "prompt_ids_sha256": adapter.digest(r["prompt_ids"]), "input_row_sha256": adapter.digest(r),
            "source_packet_sha256": packet_hash, "status": "RAW_RESPONSE_COLLECTED_UNSCORED",
            "scope": adapter.FIXTURE,
            "capability_correctness_scored": False, "oracle_inputs_read": False, "prompt_truncated": False,
            "raw_generated_ids": [7, 151645], "finish_reason": "eos", "truncated": False,
            "eos_token_id": 151645, "context_cap": 512, "maximum_generated_tokens": 32,
            "decoded_response": '{"answer":[0]}'} for r in source]
        refs = {n: ref for n in ("audit", "config", "preservation", "worker")}
        refs.update({"source_commit": "f" * 40, "source_pins": {"fixture.py": "b" * 64},
                     "model_pins": {"CPU_SYNTHETIC_NO_MODEL": "c" * 64}})
        batches.append({"scope": adapter.FIXTURE, "arm": "M0", "ordinal": ordinal,
            "source_kind": "atomic_RS_DEV", "inputs": source, "outputs": outputs,
            "source_packet_sha256": packet_hash, "audit_passed": True, "references": refs})
    return registration, batches


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.registration, self.batches = fixture(); self.calls = []

    def callback(self, messages, response, metadata):
        self.calls.append((messages, response, metadata)); return True

    def rejected_before_callback(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            adapter.score_cpu_fixture(self.registration, self.batches, arm="M0", grader=self.callback)
        self.assertEqual(self.calls, [])

    def test_full_fixture_population_before_callback(self):
        result = adapter.score_cpu_fixture(self.registration, self.batches, arm="M0", grader=self.callback)
        self.assertEqual(len(result["scores"]), 384); self.assertEqual(len(self.calls), 384)
        self.assertFalse(result["scoring_authorized"])

    def test_family_component_and_scorer_refs_survive(self):
        packet = adapter.prepare_import(self.registration, self.batches, arm="M0")
        self.assertEqual(len({r["component_id"] for r in packet["rows"]}), 96)
        self.assertEqual(packet["scorer_sources"], self.registration["scorer_sources"])
        self.assertEqual([r["source_hash"] for r in packet["rows"]], [r["source_hash"] for r in self.registration["rows"]])
        self.assertEqual(packet["batch_references"][0], self.batches[0]["references"])
        self.assertFalse(packet["retention_certified"])

    def test_cap_metadata_is_preserved_without_correctness_decision(self):
        raw = self.batches[0]["outputs"][0]
        raw.update({"raw_generated_ids": [7] * 32, "finish_reason": "generation_cap", "truncated": True,
                    "eos_token_id": None, "decoded_response": '{"answer":[0]}'})
        packet = adapter.prepare_import(self.registration, self.batches, arm="M0")
        self.assertEqual(packet["rows"][0]["raw"], raw)
        self.assertFalse(packet["scoring_authorized"])

    def test_missing_batches(self):
        self.batches.pop(); self.rejected_before_callback("whole384")

    def test_duplicate_batch(self):
        self.batches[1] = deepcopy(self.batches[0]); self.rejected_before_callback("batch/order")

    def test_reordered_batches(self):
        self.batches[0], self.batches[1] = self.batches[1], self.batches[0]
        self.rejected_before_callback("batch/order")

    def test_wrong_arm(self):
        self.batches[-1]["arm"] = "M2"; self.rejected_before_callback("arm/batch")

    def test_unaudited(self):
        self.batches[-1]["audit_passed"] = False; self.rejected_before_callback("audited")

    def test_missing_row(self):
        self.batches[-1]["outputs"].pop(); self.rejected_before_callback("whole frozen batch")

    def test_duplicate_row(self):
        self.batches[-1]["outputs"][-1] = deepcopy(self.batches[-1]["outputs"][-2])
        self.rejected_before_callback("raw row")

    def test_reordered_rows(self):
        outputs = self.batches[-1]["outputs"]; outputs[-1], outputs[-2] = outputs[-2], outputs[-1]
        self.rejected_before_callback("raw row")

    def test_wrong_prompt(self):
        self.batches[-1]["outputs"][-1]["prompt_sha256"] = "f" * 64
        self.rejected_before_callback("raw row")

    def test_partial_and_qualification_only(self):
        for status in ("RAW_TOKEN_PARTIAL_UNSCORED", "QUALIFICATION_ONLY"):
            self.batches[-1]["outputs"][-1]["status"] = status
            self.rejected_before_callback("qualification/partial")

    def test_wrong_model_pin(self):
        self.batches[-1]["references"]["model_pins"] = {"OTHER": "d" * 64}
        self.rejected_before_callback("same frozen model")

    def test_no_last_json_extraction(self):
        text = 'garbage {"answer":[0]}'; self.batches[0]["outputs"][0]["decoded_response"] = text
        adapter.score_cpu_fixture(self.registration, self.batches, arm="M0", grader=self.callback)
        self.assertEqual(self.calls[0][1], text)

    def test_resigned_native_registration_cannot_substitute_for_frozen_panel(self):
        self.registration["scope"] = adapter.NATIVE
        self.registration = adapter._seal({k: v for k, v in self.registration.items() if k != "sha256"})
        for batch in self.batches: batch["scope"] = adapter.NATIVE
        self.rejected_before_callback("native registration differs")

    def test_forged_native_descriptor_rejected_before_callback(self):
        self.registration = adapter.load_frozen_panel(PACKAGE)
        for batch in self.batches: batch["scope"] = adapter.NATIVE
        self.rejected_before_callback("independently pinned loader record")

    def test_native_record_factory_is_private(self):
        with self.assertRaisesRegex(ValueError, "pinned loader factory"):
            adapter._LoadedM0Batch(self.batches[0], {}, token=object())

    def test_native_policy_gate_has_no_grader_path(self):
        # Synthetic injected native-scope packet tests the final gate only.
        with patch.object(adapter, "prepare_import", return_value={"scope": adapter.NATIVE}):
            self.rejected_before_callback("correctness policy unregistered")

    def test_fixture_cannot_use_native_capture_scope(self):
        self.batches[0]["outputs"][0]["scope"] = "UNQUALIFIED_NATIVE_IMPLEMENTATION"
        self.rejected_before_callback("fixtures cannot import native")

    def test_native_candidate_schema_absent(self):
        self.registration["scope"] = adapter.NATIVE
        self.registration = adapter._seal({k: v for k, v in self.registration.items() if k != "sha256"})
        for arm in ("M1", "M2"):
            with self.assertRaisesRegex(ValueError, "schema unregistered"):
                adapter.score_cpu_fixture(self.registration, self.batches, arm=arm, grader=self.callback)
        self.assertEqual(self.calls, [])

    def test_binary_callback_required(self):
        with self.assertRaisesRegex(ValueError, "strict binary"):
            adapter.score_cpu_fixture(self.registration, self.batches, arm="M0", grader=lambda *args: 1)

    def test_byte_drift_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "small.json"; path.write_bytes(b'{"a":1,"a":2}')
            with self.assertRaisesRegex(ValueError, "byte drift"):
                adapter.read_pinned(path, "f" * 64)
            raw, _ = adapter.read_pinned(path, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                adapter._json(raw)


class ExistingEvidenceTests(unittest.TestCase):
    def test_frozen_real_panel_metadata_without_labels_or_grading(self):
        registration = adapter.load_frozen_panel(PACKAGE)
        self.assertEqual(len(registration["rows"]), 384)
        self.assertEqual(len({r["component_id"] for r in registration["rows"]}), 96)
        self.assertIn("UNREGISTERED", registration["scoring_policy_status"])

    def test_resigned_real_registration_mutation_rejected(self):
        registration = adapter.load_frozen_panel(PACKAGE)
        registration["rows"][0]["source_hash"] = "f" * 64
        registration = adapter._seal({k: v for k, v in registration.items() if k != "sha256"})
        calls = []
        with self.assertRaisesRegex(ValueError, "native registration differs"):
            adapter.score_cpu_fixture(registration, [], arm="M0", grader=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_mutated_loaded_native_descriptor_rejected_after_reread(self):
        registration = adapter.load_frozen_panel(PACKAGE)
        audit_path = PACKAGE / "evidence/short_m0_atomic_batch01_attempt1_independent_audit_20261005.json"
        audit_bytes = audit_path.read_bytes(); audit = json.loads(audit_bytes)
        batch = adapter.load_m0_batch(PACKAGE / "evidence/short_m0_atomic_batch01_attempt1_20261005", audit_path,
            expected_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(), expected_config_sha256=audit["config_file_sha256"])
        batch["outputs"][0]["decoded_response"] = "CPU MUTATION CONTROL"
        calls = []
        with self.assertRaisesRegex(ValueError, "independently pinned file joins"):
            adapter.score_cpu_fixture(registration, [batch] * 6, arm="M0", grader=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_actual192_rejected_without_callback_or_model_reads(self):
        registration = adapter.load_frozen_panel(PACKAGE); batches = []
        for ordinal in (1, 2, 3):
            stem = f"short_m0_atomic_batch{ordinal:02d}_attempt1"
            audit_path = PACKAGE / "evidence" / (stem + "_independent_audit_20261005.json")
            audit_bytes = audit_path.read_bytes(); audit = json.loads(audit_bytes)
            batches.append(adapter.load_m0_batch(PACKAGE / "evidence" / (stem + "_20261005"), audit_path,
                expected_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(), expected_config_sha256=audit["config_file_sha256"]))
        calls = []
        with self.assertRaisesRegex(ValueError, "whole384"):
            adapter.score_cpu_fixture(registration, batches, arm="M0", grader=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_actual_failed_batch04_audit_is_not_import_authority(self):
        audit_path = PACKAGE / "evidence/short_m0_atomic_batch04_attempt1_independent_failure_audit_20261005.json"
        raw = audit_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "successful independent raw collection audit required"):
            adapter.load_m0_batch(PACKAGE / "evidence/short_m0_atomic_batch04_attempt1_20261005", audit_path,
                expected_audit_sha256=hashlib.sha256(raw).hexdigest(), expected_config_sha256="f" * 64)


if __name__ == "__main__":
    unittest.main()
