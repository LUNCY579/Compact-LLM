"""Opaque metadata fixtures only; never model capability or execution authority."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_quality_queue_v1.contracts import REQUIRED_GATES
from experiments.rineya_quality_queue_v1.queue import (
    BUSINESS, IDS, NO_ACTIONS, ROOT, bind_metric, build_card, external_output, pointer,
    read_evidence, sha_file, strict_json, verify_packet, write_new,
)


class QueueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=BUSINESS)
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.receipt = self.temp / "receipt.json"
        self.receipt.write_text(json.dumps({"correct": 4, "total": 10, "source": "a" * 64,
                                           "gate": False, "status": "historical", "steps": 2}), encoding="utf-8")
        self.spec = {"id": "rsd", "priority": 5, "title": "Synthetic metadata fixture",
            "checkout": str(self.temp), "expected_branch": None,
            "owner": {"thread_id": "fixture-owner", "title": "Synthetic owner", "host_id": "local"},
            "candidate": {"model_sha256": "NOT_MEASURED", "adapter_sha256": None, "checkpoint": "NOT_MEASURED"},
            "evidence": [{"id": "history", "path": str(self.receipt), "classification": "historical",
                          "role": "receipt", "format": "json"}],
            "metrics": [{"name": "historical_success", "status": "MEASURED", "numerator": 4,
                         "denominator": 10, "unit": "count", "evidence_id": "history",
                         "json_pointer": "/correct", "denominator_json_pointer": "/total"}],
            "requirements": [{"id": "native_qualification", "status": "NOT_MEASURED",
                              "evidence_ids": ["history"], "reason": "Synthetic metadata cannot qualify native."}],
            "next_proposal": {"phase": "OFFLINE_REVIEW", "requires": ["native_qualification"],
                              "proposal_evidence_id": "history", "limits": {"native_calls": 0},
                              "native_execution_authorized": False},
            "quality_gates": [{"id": "source_specific_retention", "evidence_id": "history", "json_pointer": "/gate"}],
            "advancement": False}

    def card(self):
        return build_card(self.spec, inspect_git=False)

    def test_historical_measurement_cannot_fill_current_candidate_gates(self):
        card, _ = self.card()
        self.assertEqual(card["metrics"][0]["source_classification"], "historical")
        self.assertEqual(card["candidate_outcome_status"], "NOT_MEASURED")
        self.assertEqual(card["decision"]["status"], "HOLD")
        self.assertIn("registered:source_specific_retention", card["decision"]["blocking_gates"])
        self.assertFalse(card["next_proposal"]["ready_for_native_authorization"])
        self.assertTrue(all(card[key] is False for key in NO_ACTIONS))

    def test_pointer_value_or_denominator_drift_is_rejected(self):
        for name, value in (("numerator", 5), ("denominator", 9)):
            with self.subTest(name=name):
                spec = copy.deepcopy(self.spec)
                spec["metrics"][0][name] = value
                with self.assertRaisesRegex(ValueError, "differs from source"):
                    build_card(spec, inspect_git=False)

    def test_boolean_source_flag_cannot_be_numeric_measurement(self):
        self.spec["metrics"][0].update(numerator=0, json_pointer="/gate")
        with self.assertRaisesRegex(ValueError, "numerator differs"):
            self.card()

    def test_fixture_cannot_authorize_quality_even_with_measurement_fields(self):
        self.spec["evidence"][0]["classification"] = "fixture"
        with self.assertRaisesRegex(ValueError, "cannot support a quality"):
            self.card()

    def test_recorded_asset_hash_is_metadata_only_and_must_match(self):
        self.spec["source_assets"] = [{"name": "opaque source", "sha256": "a" * 64,
              "path": str(self.temp / "absent-model.safetensors"), "evidence_id": "history",
              "json_pointer": "/source", "custody_status": "RECORDED_VERIFIED_METADATA"}]
        card, _ = self.card()
        self.assertEqual(card["source_assets"][0]["sha256"], "a" * 64)
        self.assertFalse(Path(self.spec["source_assets"][0]["path"]).exists())
        self.spec["source_assets"][0]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "asset hash differs"):
            self.card()

    def test_weights_and_protected_paths_refused_before_read(self):
        for name in ("sealed/receipt.json", "CHECK/receipt.json", "FINAL.json", "model.safetensors", "rows.jsonl"):
            record = dict(self.spec["evidence"][0], path=str(self.temp / name))
            with patch.object(Path, "read_bytes", side_effect=AssertionError("forbidden read")):
                with self.assertRaises(ValueError):
                    read_evidence(record)

    def test_task_arrays_duplicate_keys_and_nonfinite_receipts_refused(self):
        for content in ('[{"prompt":"fixture"}]', '{"x":1,"x":2}', '{"x":NaN}'):
            self.receipt.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_evidence(self.spec["evidence"][0])

    def test_text_assertion_and_denominator_are_preserved(self):
        self.receipt.write_text("Historical accuracy 4/10. Retained /10.\n", encoding="utf-8")
        self.spec["evidence"][0]["format"] = "text"
        self.spec["metrics"][0].pop("json_pointer")
        self.spec["metrics"][0].pop("denominator_json_pointer")
        self.spec["metrics"][0]["text_anchor"] = "Historical accuracy 4/10."
        self.spec["quality_gates"][0].pop("json_pointer")
        self.spec["quality_gates"][0]["text_anchor"] = "Retained /10."
        card, _ = self.card()
        self.assertEqual(card["metrics"][0]["assertion_verification"], "BOUND_TEXT_EXCERPT")
        self.spec["metrics"][0]["numerator"] = 5
        with self.assertRaisesRegex(ValueError, "numerator absent"):
            self.card()

    def test_denominator_text_requires_exact_numeric_support(self):
        self.receipt.write_text("Historical accuracy 4/10. Retained /10.\n", encoding="utf-8")
        snap = read_evidence(dict(self.spec["evidence"][0], format="text"))
        metric = dict(self.spec["metrics"][0], text_anchor="Historical accuracy 4/10.",
                      denominator_text_anchor="Retained /10.")
        metric.pop("json_pointer")
        metric.pop("denominator_json_pointer")
        self.assertEqual(bind_metric(metric, {"history": snap})["denominator"], 10)
        for denominator in (9, 100, 1):
            with self.assertRaises(ValueError):
                bind_metric(dict(metric, denominator=denominator), {"history": snap})

    def test_unknown_metric_still_checks_supplied_citations(self):
        self.spec["metrics"][0].update(numerator=None, denominator=None, status="NOT_MEASURED", evidence_id="absent")
        with self.assertRaisesRegex(ValueError, "unknown/missing source"):
            self.card()

    def test_priority_owner_advance_and_capsule_identity_guards(self):
        for key, value in (("priority", 1), ("advancement", True), ("owner", {})):
            spec = dict(self.spec, **{key: value})
            with self.assertRaises(ValueError):
                build_card(spec, inspect_git=False)
        self.spec["evidence"][0]["id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "safe evidence identity"):
            self.card()

    def test_external_output_requires_new_non_git_businessproject_path(self):
        self.assertEqual(external_output(self.temp / "new"), self.temp / "new")
        for path in (self.temp, ROOT / "artifacts/new", BUSINESS.parent / ".codex/new"):
            with self.assertRaises(ValueError):
                external_output(path)

    def test_json_pointer_escapes_indices_and_missing_values(self):
        self.assertEqual(pointer({"a/b": {"~c": [9]}}, "/a~1b/~0c/0"), 9)
        for path in ("", "/bad", "/a~2b", "/a~1b/~0c/-1"):
            with self.assertRaises((ValueError, IndexError)):
                pointer({"a/b": {"~c": [9]}}, path)

    def test_model_free_package_import_and_restricted_cli(self):
        subprocess.run([sys.executable, "-B", "-S", "-c",
            "import sys; import experiments.rineya_quality_queue_v1.queue; "
            "assert not {'torch','transformers','safetensors'} & sys.modules.keys()"], check=True, cwd=ROOT)
        result = subprocess.run([sys.executable, "-B", "-S", "-m", "experiments.rineya_quality_queue_v1",
                                 "native"], capture_output=True, cwd=ROOT)
        self.assertNotEqual(result.returncode, 0)

    def make_packet(self):
        out = self.temp / "packet"
        out.mkdir()
        (out / "specs").mkdir()
        sources = self.temp / "source-specs"
        sources.mkdir()
        queue = {"ordered_candidates": [], "software_source_sha256": {}, "specs": [],
                 "ready_for_native_authorization": [], **NO_ACTIONS}
        for key in IDS:
            spec = copy.deepcopy(self.spec)
            spec["id"], spec["priority"] = key, IDS.index(key) + 1
            card, snapshots = build_card(spec, inspect_git=False)
            capsule = key + ".evidence.json"
            (out / capsule).write_bytes(self.receipt.read_bytes())
            card["evidence"][0]["capsule"] = capsule
            write_new(out / (key + ".quality-card.json"), card)
            raw = json.dumps(spec).encode()
            (sources / (key + ".json")).write_bytes(raw)
            (out / "specs" / (key + ".json")).write_bytes(raw)
            queue["specs"].append({"id": key, "path": str(sources / (key + ".json")),
                                   "capsule": "specs/" + key + ".json", "sha256": sha_file(sources / (key + ".json"))})
            queue["ordered_candidates"].append({"id": key, "card": key + ".quality-card.json",
                "priority": spec["priority"], "owner": spec["owner"], "decision": "HOLD",
                "next_phase": "OFFLINE_REVIEW", "ready_for_native_authorization": False})
        write_new(out / "QUEUE.json", queue)
        manifest = {"files": {p.relative_to(out).as_posix(): sha_file(p) for p in out.rglob("*") if p.is_file()}}
        write_new(out / "MANIFEST.json", manifest)
        return out

    def reseal(self, out):
        # Adversarial temporary metadata fixture only; never real evidence.
        manifest = {"files": {p.relative_to(out).as_posix(): sha_file(p) for p in out.rglob("*")
                              if p.is_file() and p.name != "MANIFEST.json"}}
        (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_bound_packet_detects_tamper_and_live_source_drift(self):
        out = self.make_packet()
        self.assertEqual(verify_packet(out)["cards_verified"], 5)
        self.receipt.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "live source drift"):
            verify_packet(out)
        self.assertFalse(verify_packet(out, check_live_sources=False)["live_sources_checked"])
        (out / "rsd.evidence.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "packet file drift"):
            verify_packet(out, check_live_sources=False)

    def test_resealed_card_assertions_and_classifications_are_rechecked(self):
        out = self.make_packet()
        path = out / "rsd.quality-card.json"
        original = strict_json(path.read_bytes())
        for field in ("numerator", "classification"):
            card = copy.deepcopy(original)
            if field == "numerator":
                card["metrics"][0]["numerator"] = 5
            else:
                card["evidence"][0]["classification"] = "fixture"
            path.write_text(json.dumps(card), encoding="utf-8")
            self.reseal(out)
            with self.assertRaises(ValueError):
                verify_packet(out, check_live_sources=False)

    def test_resealed_nested_permissions_or_readiness_are_rejected(self):
        out = self.make_packet()
        card_path = out / "rsd.quality-card.json"
        original = strict_json(card_path.read_bytes())
        card = copy.deepcopy(original)
        card["next_proposal"]["native_execution_authorized"] = True
        card_path.write_text(json.dumps(card), encoding="utf-8")
        self.reseal(out)
        with self.assertRaisesRegex(ValueError, "nested proposal"):
            verify_packet(out, check_live_sources=False)
        card_path.write_text(json.dumps(original), encoding="utf-8")
        queue_path = out / "QUEUE.json"
        queue = strict_json(queue_path.read_bytes())
        queue["ordered_candidates"][0]["ready_for_native_authorization"] = True
        queue_path.write_text(json.dumps(queue), encoding="utf-8")
        self.reseal(out)
        with self.assertRaisesRegex(ValueError, "readiness"):
            verify_packet(out, check_live_sources=False)

    def test_resealed_verification_paths_cannot_escape(self):
        out = self.make_packet()
        path = out / "QUEUE.json"
        original = strict_json(path.read_bytes())
        for change in ("card", "software"):
            queue = copy.deepcopy(original)
            if change == "card":
                queue["ordered_candidates"][0]["card"] = "../receipt.json"
            else:
                queue["software_source_sha256"] = {"../receipt.json": sha_file(self.receipt)}
            path.write_text(json.dumps(queue), encoding="utf-8")
            self.reseal(out)
            with self.assertRaisesRegex(ValueError, "path"):
                verify_packet(out, check_live_sources=False)

    def test_live_source_resolution_must_match_original_target(self):
        out = self.make_packet()
        original_resolver = __import__("experiments.rineya_quality_queue_v1.queue", fromlist=["source_path"]).source_path
        def redirected(path, business=BUSINESS):
            requested, resolved = original_resolver(path, business)
            if Path(path) == self.receipt:
                resolved = self.temp / "changed-target.json"
            return requested, resolved
        with patch("experiments.rineya_quality_queue_v1.queue.source_path", side_effect=redirected):
            with self.assertRaisesRegex(ValueError, "resolution drift"):
                verify_packet(out)
