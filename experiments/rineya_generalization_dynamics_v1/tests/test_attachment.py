"""Offline toy-file attachment checks; never native loading or Rineya evidence."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1 import attachment as attachment_module
from experiments.rineya_generalization_dynamics_v1.attachment import (
    SCHEMA, native_spec_from_attachment, validate_attachment,
)
from experiments.rineya_generalization_dynamics_v1.common import Blocked, file_hash, signed, verify
from experiments.rineya_generalization_dynamics_v1.integration import record_snapshot
from experiments.rineya_generalization_dynamics_v1.pipeline import DEFAULT_NATIVE
from experiments.rineya_generalization_dynamics_v1.registry import snapshot


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        # All fresh payloads stay outside tracked source and .codex, even when callers
        # have not overridden the operating system's default temporary directory.
        business_project = Path(__file__).resolve().parents[4]
        temp_root = business_project / "gd001-fixture-checks"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="attachment_", dir=temp_root)
        self.root = Path(self.temp.name)
        self.sequence = 0

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, kind="hf", evidence="mechanical_fixture"):
        self.sequence += 1
        root = self.root / str(self.sequence)
        root.mkdir()
        base, tokenizer = root / "base", root / "tokenizer"
        base.mkdir()
        tokenizer.mkdir()
        shape = {"model_type": "toy_fixture", "hidden_size": 4}
        save(base / "config.json", shape)
        (base / "model.safetensors").write_bytes(b"TOY FIXTURE ONLY: not native weights")
        save(tokenizer / "tokenizer_config.json", {})
        save(tokenizer / "tokenizer.json", {"toy": True})
        checkpoints = []
        for step in (1, 2):
            path = root / f"step-{step}"
            path.mkdir()
            if kind == "hf":
                save(path / "config.json", shape)
                weight_name = "model.safetensors"
            else:
                save(path / "adapter_config.json", {"peft_type": "LORA", "r": 2,
                                                     "base_model_name_or_path": str(base)})
                weight_name = "adapter_model.safetensors"
            (path / weight_name).write_bytes(f"TOY FIXTURE STEP {step}".encode())
            receipt = record_snapshot(str(root / "receipts"), str(path), step, kind)
            checkpoints.append({"step": step, "kind": kind, "path": str(path),
                                "content_sha256": receipt["asset"]["content_sha256"],
                                "snapshot_receipt": ref(root / "receipts" / f"step_{step:08d}.json")})
        contract = {"schema": SCHEMA, "evidence_kind": evidence,
                    "trajectory_id": "TOY_FIXTURE_NOT_RINEYA",
                    "expected_base_parameters": 596_049_920,
                    "producer": {"owner": "toy-fixture-owner", "study_id": "toy-fixture-study",
                                 "independently_approved": True, "artifact_role": "original_training_trajectory"},
                    "base": {k: snapshot(str(base), "hf")[k] for k in ("path", "content_sha256")},
                    "tokenizer": {k: snapshot(str(tokenizer), "tokenizer")[k] for k in ("path", "content_sha256")},
                    "checkpoints": checkpoints}
        bindings = {k: copy.deepcopy(contract[k]) for k in
                    ("trajectory_id", "expected_base_parameters", "producer", "base", "tokenizer")}
        bindings["checkpoint_schedule"] = [{k: c[k] for k in ("step", "kind", "path")} for c in checkpoints]
        plan = {**bindings, "status": "APPROVED_PROSPECTIVE", "purpose": "research_training",
                "approved_at_utc": "2026-10-01T00:00:00Z",
                "retention": "all_original_snapshots_and_provenance"}
        completion = {**copy.deepcopy(bindings), "status": "COMPLETED", "purpose": "research_training",
                      "training_started_at_utc": "2026-10-02T00:00:00Z", "completed_at_utc": "2026-10-03T00:00:00Z",
                      "final_step": 2, "source_count": 4, "rejected_count": 1,
                      "consumed_count": 3, "unused_accepted_count": 1, "consumed_per_step": {"1": 2, "2": 1}}
        completion["checkpoints"] = copy.deepcopy(checkpoints)
        completion["final_checkpoint"] = copy.deepcopy(checkpoints[-1])
        raw_rows = [{"text": "keep repeat", "target": "alpha"}, {"text": "keep truncate", "target": "beta"},
                    {"text": "reject actual", "target": "gamma"}, {"text": "unused finite plan", "target": "delta"}]
        source_path = root / "original-source.jsonl"
        jsonl(source_path, raw_rows)
        events = []
        for index, raw in enumerate(raw_rows, 1):
            rejected = index == 3
            events.append({"event_id": f"source-{index}", "event_type": "source",
                           "parent_id": f"parent-{index}", "source_row_id": f"row-{index}",
                           "source_row_number": index, "source_ref": ref(source_path), "raw_record": raw})
            events.append({"event_id": f"filter-{index}", "event_type": "filter",
                           "source_event_id": f"source-{index}", "decision": "rejected" if rejected else "accepted",
                           "reason": "toy explicit filter", "filtering_history": [{"operation": "toy-filter-v1",
                           "decision": "reject" if rejected else "keep", "reason": "toy explicit decision"}]})
        events.append({"event_id": "unused-4", "event_type": "unused", "source_event_id": "source-4",
                       "filter_event_id": "filter-4", "reason": "finite toy schedule did not sample this accepted row"})
        for number, index, step in ((1, 1, 1), (2, 1, 1), (3, 2, 2)):
            truncated = index == 2
            events.append({"event_id": f"consume-{number}", "event_type": "consume",
                           "source_event_id": f"source-{index}", "filter_event_id": f"filter-{index}",
                           "parent_id": f"parent-{index}", "source_row_id": f"row-{index}", "step": step,
                           "prompt": raw_rows[index - 1]["text"], "raw_rendered_prompt": "toy rendered full text",
                           "consumed_prompt": "toy rendered" if truncated else "toy rendered full text",
                           "input_ids": [7, 8] if truncated else [7, 8, 9],
                           "tokenizer_content_sha256": contract["tokenizer"]["content_sha256"],
                           "rendering_history": [{"operation": "toy-render-v1", "template": "toy"}],
                           "truncation": {"policy": "right" if truncated else "none", "original_input_ids": [7, 8, 9],
                                          "max_tokens": 2 if truncated else None,
                                          "history": [{"operation": "truncate", "side": "right", "removed_tokens": 1,
                                                       "max_tokens": 2}] if truncated else []}})
        fixture = {"root": root, "contract": contract, "plan": plan, "completion": completion,
                   "events": events, "manifest": [{"prompt": e["consumed_prompt"], "parent_id": e["parent_id"]}
                                                   for e in events if e["event_type"] == "consume"]}
        self.persist(fixture)
        return fixture

    def persist(self, fixture):
        root, contract = fixture["root"], fixture["contract"]
        save(root / "owner-plan.json", fixture["plan"])
        jsonl(root / "events.jsonl", fixture["events"])
        jsonl(root / "training-manifest.jsonl", fixture["manifest"])
        contract["provenance"] = {"events": ref(root / "events.jsonl"),
                                  "training_manifest": ref(root / "training-manifest.jsonl")}
        fixture["completion"]["provenance"] = copy.deepcopy(contract["provenance"])
        save(root / "completion.json", fixture["completion"])
        contract["approval_plan"] = ref(root / "owner-plan.json")
        contract["completion_receipt"] = ref(root / "completion.json")
        save(root / "attachment.json", contract)
        return str(root / "attachment.json")

    def blocked(self, fixture, message=None):
        path = self.persist(fixture)
        with self.assertRaisesRegex(Blocked, message or "."):
            validate_attachment(path)

    def test_hf_post_save_payloads_and_repeat_counts_validate_offline(self):
        fixture = self.fixture()
        with patch("experiments.rineya_generalization_dynamics_v1.native.NativeEvaluator.load",
                   side_effect=AssertionError("Attachment validation must not load a model")):
            receipt = verify(validate_attachment(self.persist(fixture)))
        self.assertEqual(receipt["provenance"]["consumed_count"], 3)
        self.assertEqual(receipt["provenance"]["consumed_per_step"], {"1": 2, "2": 1})
        self.assertEqual(receipt["provenance"]["unused_accepted_count"], 1)
        self.assertEqual(len(receipt["registry"]["checkpoints"]), 3)
        for field in ("native_loading_qualified", "export_parity_qualified", "checkpoint_selection_efficacy_established",
                      "authorizes_training", "authorizes_native_run", "authorizes_confirmation", "permits_model_promotion"):
            self.assertIs(receipt[field], False)

    def test_peft_trajectory_metadata_validates(self):
        receipt = validate_attachment(self.persist(self.fixture(kind="peft")))
        self.assertEqual({c["kind"] for c in receipt["registry"]["checkpoints"][1:]}, {"peft"})

    def test_missing_original_weight_payload_blocks(self):
        fixture = self.fixture()
        (Path(fixture["contract"]["checkpoints"][0]["path"]) / "model.safetensors").unlink()
        self.blocked(fixture, "weights required")

    def test_mutated_weight_and_tokenizer_payloads_block(self):
        for location in ("base", "tokenizer", "step-1"):
            with self.subTest(location=location):
                fixture = self.fixture()
                name = "tokenizer.json" if location == "tokenizer" else "model.safetensors"
                (fixture["root"] / location / name).write_bytes(b"mutated payload")
                self.blocked(fixture, "identity differs|content differs")

    def test_snapshot_receipt_file_hash_and_internal_digest_block(self):
        for internally_signed in (True, False):
            with self.subTest(internally_signed=internally_signed):
                fixture = self.fixture()
                checkpoint = fixture["contract"]["checkpoints"][0]
                path = Path(checkpoint["snapshot_receipt"]["path"])
                payload = json.loads(path.read_text())
                payload["entry"]["step"] = 5
                if internally_signed:
                    payload = signed({k: v for k, v in payload.items() if k != "sha256"})
                save(path, payload)
                checkpoint["snapshot_receipt"] = ref(path)
                self.blocked(fixture, "receipt must match|digest mismatch")
        fixture = self.fixture()
        Path(fixture["contract"]["checkpoints"][0]["snapshot_receipt"]["path"]).write_text("{}")
        self.blocked(fixture, "payload changed")

    def test_schedule_order_count_directory_reuse_and_base_alias_block(self):
        for mutation in (lambda c: c["checkpoints"].reverse(), lambda c: c["checkpoints"].pop(),
                         lambda c: c["checkpoints"][1].update(step=1),
                         lambda c: c["checkpoints"][1].update(path=c["checkpoints"][0]["path"]),
                         lambda c: c["checkpoints"][0].update(path=c["base"]["path"])):
            fixture = self.fixture()
            mutation(fixture["contract"])
            self.blocked(fixture)

    def test_original_independent_research_and_complete_receipt_required(self):
        mutations = [lambda f: f["contract"]["producer"].update(independently_approved=False),
                     lambda f: f["contract"]["producer"].update(artifact_role="published_derived_core"),
                     lambda f: f["contract"].update(expected_base_parameters=344_000_000),
                     lambda f: f["completion"].update(status="INCOMPLETE"),
                     lambda f: f["completion"].update(final_step=6),
                     lambda f: f["completion"].update(purpose="six_discarded_ce8_qualification_updates"),
                     lambda f: f["plan"].update(purpose="mechanical_qualification"),
                     lambda f: f["plan"].update(status="UNAPPROVED"),
                     lambda f: f["plan"].update(retention="metadata_only")]
        for mutation in mutations:
            fixture = self.fixture()
            mutation(fixture)
            self.blocked(fixture)

    def test_owner_plan_source_tokenizer_and_schedule_identity_bound(self):
        for field in ("base", "tokenizer", "producer", "trajectory_id", "checkpoint_schedule"):
            for target in ("plan", "completion"):
                with self.subTest(field=field, target=target):
                    fixture = self.fixture()
                    fixture[target][field] = "foreign identity"
                    self.blocked(fixture, "identity or preregistered schedule differs")

    def test_approval_must_precede_training_and_completion(self):
        for target, field, value in (("plan", "approved_at_utc", "2026-10-04T00:00:00Z"),
                                     ("completion", "completed_at_utc", "2026-10-01T00:00:00Z"),
                                     ("plan", "approved_at_utc", "2026-10-01T00:00:00")):
            fixture = self.fixture()
            fixture[target][field] = value
            self.blocked(fixture)

    def test_architecture_and_peft_configuration_changes_block(self):
        for kind in ("hf", "peft"):
            fixture = self.fixture(kind)
            checkpoint = fixture["contract"]["checkpoints"][1]
            path = Path(checkpoint["path"])
            name = "config.json" if kind == "hf" else "adapter_config.json"
            config = json.loads((path / name).read_text())
            config["hidden_size" if kind == "hf" else "r"] = 8
            save(path / name, config)
            asset = snapshot(str(path), kind)
            checkpoint["content_sha256"] = asset["content_sha256"]
            receipt_path = Path(checkpoint["snapshot_receipt"]["path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["asset"] = asset
            save(receipt_path, signed({k: v for k, v in receipt.items() if k != "sha256"}))
            checkpoint["snapshot_receipt"] = ref(receipt_path)
            fixture["completion"]["checkpoints"] = copy.deepcopy(fixture["contract"]["checkpoints"])
            fixture["completion"]["final_checkpoint"] = copy.deepcopy(checkpoint)
            self.blocked(fixture, "Architecture/config differs|Adapter configuration changed")

    def test_custom_readout_or_adapter_export_blocks(self):
        fixture = self.fixture()
        path = fixture["root"] / "step-1" / "config.json"
        save(path, {"model_type": "toy_fixture", "hidden_size": 4, "abstain_logit_bias": 1})
        self.blocked(fixture, "Custom/quantized/readout")

    def test_original_source_missing_or_mutated_blocks(self):
        for delete in (True, False):
            fixture = self.fixture()
            path = fixture["root"] / "original-source.jsonl"
            if delete:
                path.unlink()
            else:
                path.write_text('{"text":"changed"}\n')
            self.blocked(fixture, "Missing retained|payload changed")

    def test_raw_source_and_rejected_inventory_cannot_be_omitted(self):
        fixture = self.fixture()
        fixture["events"][0]["raw_record"] = {"text": "metadata only"}
        self.blocked(fixture, "Original source payload")
        fixture = self.fixture()
        fixture["events"] = [e for e in fixture["events"] if e["event_id"] not in {"source-3", "filter-3"}]
        fixture["completion"].update(source_count=3, rejected_count=0)
        self.blocked(fixture, "Source inventory rows omitted")

    def test_filtering_rendering_truncation_fields_cannot_be_omitted(self):
        for event_index, field in ((1, "filtering_history"), (9, "rendering_history"), (9, "prompt"),
                                   (9, "raw_rendered_prompt"), (9, "consumed_prompt"), (9, "truncation"),
                                   (9, "input_ids"), (9, "tokenizer_content_sha256")):
            with self.subTest(field=field):
                fixture = self.fixture()
                fixture["events"][event_index].pop(field)
                self.blocked(fixture)

    def test_silent_or_incorrect_truncation_blocks(self):
        for mutation in (lambda e: e.update(consumed_prompt="silently shortened"),
                         lambda e: e.update(input_ids=[7]),
                         lambda e: e["truncation"].update(history=[]),
                         lambda e: e["truncation"].update(max_tokens=1),
                         lambda e: e["truncation"].update(policy="unknown")):
            fixture = self.fixture()
            target = fixture["events"][-1]
            mutation(target)
            # First text change is invalid for a nontruncated occurrence.
            if target["consumed_prompt"] == "silently shortened":
                fixture["events"][9]["consumed_prompt"] = "silently shortened"
            self.blocked(fixture)

    def test_wrong_source_filter_parent_or_tokenizer_blocks(self):
        for field, value in (("source_event_id", "source-3"), ("filter_event_id", "filter-2"),
                             ("parent_id", "foreign-parent"), ("source_row_id", "foreign-row"),
                             ("tokenizer_content_sha256", "0" * 64)):
            fixture = self.fixture()
            fixture["events"][9][field] = value
            self.blocked(fixture)

    def test_event_duplicate_step_order_and_final_coverage_block(self):
        for mutation in (lambda f: f["events"][9].update(event_id="source-1"),
                         lambda f: f["events"][9].update(step=2),
                         lambda f: f["events"][-1].update(step=1),
                         lambda f: f["events"][-1].update(step=True)):
            fixture = self.fixture()
            mutation(fixture)
            self.blocked(fixture)

    def test_consumption_and_disposition_counts_cannot_silently_shrink(self):
        for field in ("source_count", "rejected_count", "consumed_count", "unused_accepted_count", "consumed_per_step"):
            fixture = self.fixture()
            fixture["completion"][field] = {} if field == "consumed_per_step" else 0
            self.blocked(fixture)
        fixture = self.fixture()
        fixture["events"].pop(9)
        self.blocked(fixture, "exactly project")
        fixture = self.fixture()
        fixture["completion"]["consumed_per_step"]["2"] = True
        self.blocked(fixture, "per-step consumption differs")

    def test_completion_binds_exact_snapshot_hashes_receipts_and_final_reference(self):
        for field in ("checkpoints", "final_checkpoint"):
            fixture = self.fixture()
            fixture["completion"].pop(field)
            self.blocked(fixture, "bind exact saved payloads")
        fixture = self.fixture()
        fixture["completion"]["final_checkpoint"]["content_sha256"] = "0" * 64
        self.blocked(fixture, "bind exact saved payloads")

    def test_original_source_identity_is_type_sensitive(self):
        for replacement in (True, 1.0):
            fixture = self.fixture()
            source_path = fixture["root"] / "original-source.jsonl"
            rows = [json.loads(line) for line in source_path.read_text().splitlines()]
            rows[0]["numeric_field"] = 1
            jsonl(source_path, rows)
            for event in fixture["events"]:
                if event["event_type"] == "source":
                    event["source_ref"] = ref(source_path)
                    if event["source_row_number"] == 1:
                        event["raw_record"] = {**rows[0], "numeric_field": replacement}
            self.blocked(fixture, "Original source payload")

    def test_approval_completion_and_receipt_identity_are_type_sensitive(self):
        for target in ("plan", "completion"):
            for replacement in (True, 1.0):
                fixture = self.fixture()
                fixture[target]["checkpoint_schedule"][0]["step"] = replacement
                self.blocked(fixture, "identity or preregistered schedule differs")
        fixture = self.fixture()
        fixture["completion"]["final_checkpoint"]["step"] = 2.0
        self.blocked(fixture, "bind exact saved payloads")
        fixture = self.fixture()
        fixture["plan"]["expected_base_parameters"] = float(fixture["contract"]["expected_base_parameters"])
        self.blocked(fixture, "identity or preregistered schedule differs")
        for replacement in (True, 1.0):
            fixture = self.fixture()
            checkpoint = fixture["contract"]["checkpoints"][0]
            receipt_path = Path(checkpoint["snapshot_receipt"]["path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["entry"]["step"] = replacement
            save(receipt_path, signed({k: v for k, v in receipt.items() if k != "sha256"}))
            checkpoint["snapshot_receipt"] = ref(receipt_path)
            self.blocked(fixture, "Post-save receipt must match")
        fixture = self.fixture()
        fixture["events"][-1]["truncation"]["history"][0]["removed_tokens"] = True
        self.blocked(fixture, "Truncation history does not match")

    def test_explicit_unused_disposition_required_and_not_consumed(self):
        fixture = self.fixture()
        fixture["events"].pop(8)
        fixture["completion"]["unused_accepted_count"] = 0
        self.blocked(fixture, "consumption omitted")
        fixture = self.fixture()
        fixture["events"][8]["source_event_id"] = "source-1"
        fixture["events"][8]["filter_event_id"] = "filter-1"
        self.blocked(fixture, "cannot be unused")

    def test_projection_preserves_text_parent_order_and_repeat_occurrences(self):
        for mutation in (lambda m: m.pop(0), lambda m: m.reverse(),
                         lambda m: m[0].update(prompt="raw prompt instead of consumed"),
                         lambda m: m[0].update(parent_id="other"),
                         lambda m: m[0].update(extra="not an exact projection")):
            fixture = self.fixture()
            mutation(fixture["manifest"])
            self.blocked(fixture, "exactly project")

    def test_hashed_events_completion_and_plan_payloads_cannot_change(self):
        for name in ("events.jsonl", "completion.json", "owner-plan.json", "training-manifest.jsonl"):
            fixture = self.fixture()
            path = self.persist(fixture)
            with (fixture["root"] / name).open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(Blocked, "payload changed"):
                validate_attachment(path)

    def test_source_payload_mutation_during_validation_blocks(self):
        fixture = self.fixture()
        path = self.persist(fixture)
        original_rows = attachment_module._rows

        def mutate_after_read(payload):
            rows = original_rows(payload)
            if payload.name == "original-source.jsonl":
                with payload.open("a") as stream:
                    stream.write(" ")
            return rows

        with patch.object(attachment_module, "_rows", side_effect=mutate_after_read):
            with self.assertRaisesRegex(Blocked, "changed during attachment validation"):
                validate_attachment(path)

    def test_actual_left_truncation_and_rejected_history_validate(self):
        fixture = self.fixture()
        occurrence = fixture["events"][-1]
        occurrence["input_ids"] = [8, 9]
        occurrence["truncation"]["policy"] = "left"
        occurrence["truncation"]["history"][0]["side"] = "left"
        receipt = validate_attachment(self.persist(fixture))
        self.assertEqual(receipt["provenance"]["rejected_count"], 1)

    def test_missing_field_duplicate_json_and_relative_path_fail_closed(self):
        fixture = self.fixture()
        fixture["contract"].pop("producer")
        self.blocked(fixture, "Invalid/incomplete")
        fixture = self.fixture()
        path = Path(self.persist(fixture))
        path.write_text('{"schema":"first","schema":"second"}')
        with self.assertRaisesRegex(Blocked, "Duplicate JSON field"):
            validate_attachment(str(path))
        with self.assertRaisesRegex(Blocked, "absolute stable path"):
            validate_attachment("relative-attachment.json")

    def test_fixture_cannot_prepare_native_registration(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(Blocked, "fixtures cannot become native"):
            native_spec_from_attachment(self.persist(fixture), dict(DEFAULT_NATIVE))

    def test_protected_contract_prefix_blocks_before_content_read(self):
        for component in ("sealed_eval", "SPENT_archive", "ANY-READ-bank", "confirmation.STARTED.json"):
            with self.subTest(component=component), patch.object(attachment_module, "file_hash") as hashed:
                with self.assertRaisesRegex(Blocked, "Protected banks"):
                    validate_attachment(str(self.root / component / "attachment.json"))
                hashed.assert_not_called()

    def test_protected_nested_original_payloads_block_before_access(self):
        for field in ("base", "tokenizer", "approval_plan", "completion_receipt"):
            fixture = self.fixture()
            fixture["contract"][field]["path"] = str(self.root / "sealed_eval" / "payload.json")
            save(fixture["root"] / "attachment.json", fixture["contract"])
            with self.subTest(field=field), self.assertRaisesRegex(Blocked, "Protected banks"):
                validate_attachment(str(fixture["root"] / "attachment.json"))
        fixture = self.fixture()
        fixture["events"][0]["source_ref"]["path"] = str(self.root / "confirmation-bank" / "source.jsonl")
        self.blocked(fixture, "Protected banks")

    def test_final_named_original_export_remains_admissible(self):
        fixture = self.fixture()
        checkpoint = fixture["contract"]["checkpoints"][-1]
        old_path = Path(checkpoint["path"])
        final_path = old_path.parent / "final"
        old_path.rename(final_path)
        checkpoint["path"] = str(final_path)
        receipt = record_snapshot(str(fixture["root"] / "final-receipts"), str(final_path), checkpoint["step"], checkpoint["kind"])
        checkpoint["snapshot_receipt"] = ref(fixture["root"] / "final-receipts" / f"{receipt['entry']['id']}.json")
        fixture["plan"]["checkpoint_schedule"][-1]["path"] = str(final_path)
        fixture["completion"]["checkpoint_schedule"][-1]["path"] = str(final_path)
        fixture["completion"]["checkpoints"][-1] = copy.deepcopy(checkpoint)
        fixture["completion"]["final_checkpoint"] = copy.deepcopy(checkpoint)
        self.assertEqual(validate_attachment(self.persist(fixture))["status"], "ATTACHMENT_METADATA_VALIDATED")

    def test_spec_preparation_retains_identity_and_explicit_budgets_without_execution(self):
        # This deliberately exercises the declaration branch with toy files, never
        # claims the declaration proves original training or native compatibility.
        fixture = self.fixture(evidence="producer_metadata")
        with patch("experiments.rineya_generalization_dynamics_v1.native.NativeEvaluator.load",
                   side_effect=AssertionError("No native calls during proposal preparation")):
            spec = native_spec_from_attachment(self.persist(fixture), dict(DEFAULT_NATIVE))
        self.assertEqual(spec["expected_base_parameters"], 596_049_920)
        self.assertEqual([c["step"] for c in spec["checkpoints"]], [0, 1, 2])
        self.assertEqual(spec["runtime"]["min_free_ram_gib"], 7.4)
        self.assertEqual(spec["runtime"]["max_native_forward_calls"], 16000)
        self.assertEqual(spec["attachment"], ref(fixture["root"] / "attachment.json"))
        for runtime in ({}, {**DEFAULT_NATIVE, "min_free_ram_gib": 1},
                        {**DEFAULT_NATIVE, "max_native_forward_calls": 0}):
            with self.assertRaises(Blocked):
                native_spec_from_attachment(self.persist(fixture), runtime)


def build_fixture(root: Path, *, evidence_kind="mechanical_fixture", kind="hf") -> Path:
    """Shared retained toy-payload factory for offline integration checks only.

    evidence_kind='producer_metadata' exercises caller declarations with toy bytes;
    it does not establish native loader compatibility or Rineya capability evidence.
    """
    builder = AttachmentTests()
    builder.root = Path(root).resolve()
    builder.root.mkdir(parents=True, exist_ok=True)
    builder.sequence = 0
    return Path(builder.persist(builder.fixture(kind=kind, evidence=evidence_kind)))


if __name__ == "__main__":
    unittest.main()
