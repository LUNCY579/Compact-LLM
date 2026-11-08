"""CPU-only post-save ordering and retention checks; fake weights are not models."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import Blocked, file_hash, read, write_new
from experiments.rineya_generalization_dynamics_v1.producer_hook_example import after_completed_save


class ProducerHookTests(unittest.TestCase):
    def setUp(self):
        external_root = Path(__file__).resolve().parents[4] / "gd001-records" / "hook-fixtures"
        external_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=external_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoint-10"
        self.plan_path = self.root / "approved-plan.json"
        self.plan = {"status": "APPROVED_PROSPECTIVE", "purpose": "research_training", "producer": {
            "owner": "mechanical_fixture", "study_id": "not_a_native_study",
            "independently_approved": True, "artifact_role": "original_training_trajectory"},
            "base": {"path": str(self.root / "base")},
            "retention": "all_original_snapshots_and_provenance",
            "checkpoint_schedule": [{"step": 10, "kind": "peft", "path": str(self.checkpoint)}]}
        self.write_plan()

    def write_plan(self):
        self.plan_path.write_text(__import__("json").dumps(self.plan), encoding="utf-8")
        self.plan_ref = {"path": str(self.plan_path), "sha256": file_hash(self.plan_path)}

    def save_fixture(self):
        self.checkpoint.mkdir()
        write_new(self.checkpoint / "adapter_config.json", {"peft_type": "LORA"})
        (self.checkpoint / "adapter_model.safetensors").write_bytes(b"FIXTURE_NOT_NATIVE_WEIGHTS")

    def invoke(self, **changes):
        arguments = dict(approval_plan=self.plan_ref, receipt_directory=str(self.root / "receipts"),
                         checkpoint_directory=str(self.checkpoint), step=10, kind="peft")
        arguments.update(changes)
        return after_completed_save(**arguments)

    def test_missing_save_cannot_create_receipt(self):
        with self.assertRaises(Blocked):
            self.invoke()
        self.assertFalse((self.root / "receipts").exists())

    def test_real_hook_runs_after_completed_fixture_export(self):
        self.save_fixture()
        original_bytes = (self.checkpoint / "adapter_model.safetensors").read_bytes()
        entry = self.invoke()
        receipt_path = Path(entry["snapshot_receipt"]["path"])
        receipt = read(receipt_path)
        self.assertEqual(receipt["entry"]["step"], 10)
        self.assertEqual(entry["content_sha256"], receipt["asset"]["content_sha256"])
        self.assertEqual(entry["snapshot_receipt"]["sha256"], file_hash(receipt_path))
        self.assertFalse(receipt["evaluated"])
        self.assertEqual(original_bytes, (self.checkpoint / "adapter_model.safetensors").read_bytes())
        with self.assertRaises(FileExistsError):
            self.invoke()

    def test_plan_mutation_blocks_before_hook(self):
        self.plan_path.write_text("{}", encoding="utf-8")
        with patch("experiments.rineya_generalization_dynamics_v1.producer_hook_example.record_snapshot") as hook:
            with self.assertRaises(Blocked):
                self.invoke()
            hook.assert_not_called()

    def test_wrong_step_or_path_blocks(self):
        for change in ({"step": 11}, {"kind": "hf"}, {"checkpoint_directory": str(self.root / "rolling")}):
            with self.subTest(change=change), self.assertRaises(Blocked):
                self.invoke(**change)

    def test_qualification_or_unapproved_plan_blocks(self):
        for status, role in (("DRAFT", "original_training_trajectory"),
                             ("APPROVED_PROSPECTIVE", "discarded_qualification_updates")):
            self.plan["status"] = status
            self.plan["producer"]["artifact_role"] = role
            self.write_plan()
            with self.subTest(status=status, role=role), self.assertRaises(Blocked):
                self.invoke()

    def test_rolling_path_and_payload_pruning_not_approved(self):
        self.plan["checkpoint_schedule"].append({"step": 20, "kind": "peft", "path": str(self.checkpoint)})
        self.write_plan()
        with self.assertRaises(Blocked):
            self.invoke()
        self.plan["checkpoint_schedule"].pop()
        self.plan["retention"] = "metadata_only"
        self.write_plan()
        with self.assertRaises(Blocked):
            self.invoke()

    def test_receipts_cannot_go_inside_source_or_codex(self):
        workspace = Path(__file__).resolve().parents[3]
        for destination in (workspace / "receipts", Path("relative-receipts"),
                            workspace.parent / ".codex" / "receipts",
                            workspace.parent / "gd001-records" / ".CODEX" / "receipts"):
            with self.subTest(destination=destination), self.assertRaises(Blocked):
                self.invoke(receipt_directory=str(destination))

    def test_protected_plan_checkpoint_or_receipts_block_before_observer(self):
        hidden = self.root / "confirmation-bank" / "payload"
        changes = ({"approval_plan": {"path": str(hidden), "sha256": "0" * 64}},
                   {"checkpoint_directory": str(hidden)}, {"receipt_directory": str(hidden)})
        for change in changes:
            with self.subTest(change=change), patch(
                    "experiments.rineya_generalization_dynamics_v1.producer_hook_example.record_snapshot") as hook:
                with self.assertRaisesRegex(Blocked, "Protected banks"):
                    self.invoke(**change)
                hook.assert_not_called()

    def test_final_named_original_save_remains_admissible(self):
        self.checkpoint = self.root / "final"
        self.plan["checkpoint_schedule"][0]["path"] = str(self.checkpoint)
        self.write_plan()
        self.save_fixture()
        self.assertEqual(self.invoke()["path"], str(self.checkpoint))


if __name__ == "__main__":
    unittest.main()
