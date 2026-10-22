"""Retained toy-attachment -> frozen DEV preparation checks; never model evidence."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.attachment import native_spec_from_attachment, validate_attachment
from experiments.rineya_generalization_dynamics_v1.attachment_cli import audit_producer_prompt_overlap, main, prepare_dev
from experiments.rineya_generalization_dynamics_v1.common import Blocked, code_hash, digest, file_hash, read, signed, verify, write_new
from experiments.rineya_generalization_dynamics_v1.pipeline import DEFAULT_NATIVE, DEFAULT_POLICY
from experiments.rineya_generalization_dynamics_v1.registry import freeze_registry
from experiments.rineya_generalization_dynamics_v1.tests.test_attachment import build_fixture


class PreparationTests(unittest.TestCase):
    def setUp(self):
        record_root = Path(__file__).resolve().parents[4] / "gd001-records" / "preparation-fixtures"
        record_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=record_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Caller-declared producer metadata exercises only offline file plumbing.
        # The retained weight bytes and source identity are explicitly toy fixtures.
        self.contract = build_fixture(self.root / "producer", evidence_kind="producer_metadata")
        self.proposal = {
            "status": "REVIEWED_PROPOSAL_NOT_EXECUTION_AUTHORIZATION",
            "stage": "DEV_ONLY_NATIVE_LOADING_SCORING",
            "separate_native_authorization_required": True, "confirmation_authorized": False,
            "code_sha256": code_hash(),
            "attachment_contract": {"path": str(self.contract), "sha256": file_hash(self.contract)},
            "checkpoint_steps": [0, 1, 2], "dev_seed": 17, "pairs_per_family": 24,
            "public_exposure_manifests": [],
            "policy": dict(DEFAULT_POLICY), "runtime": dict(DEFAULT_NATIVE),
            "parity_policy": signed({"schema": "gd001_export_parity_policy_v1",
                "registered_before_outcomes": True, "abs_tolerance": 0.00001,
                "rel_tolerance": 0.00001, "max_prompt_tokens": 512,
                "max_candidate_tokens": 16, "max_generated_tokens": 16,
                "max_probes": 2, "max_logit_probe_tokens": 8, "logit_probe_token_ids": [1, 2, 3]})}
        self.proposal_path = self.root / "reviewed-proposal.json"
        self.sequence = 0

    def save_proposal(self):
        self.sequence += 1
        self.proposal_path = self.root / f"reviewed-{self.sequence}.json"
        write_new(self.proposal_path, signed(self.proposal))

    def invoke(self, destination=None):
        self.save_proposal()
        return prepare_dev(str(self.contract), str(self.proposal_path),
                           str(destination or self.root / "prepared"))

    def test_post_save_contract_to_frozen_dev_sidecar_has_no_native_calls(self):
        with patch("experiments.rineya_generalization_dynamics_v1.native.NativeEvaluator.load",
                   side_effect=AssertionError("No model load authorized")), \
             patch("experiments.rineya_generalization_dynamics_v1.pipeline.confirm",
                   side_effect=AssertionError("Confirmation is not authorized")):
            result = verify(self.invoke())
        prepared = self.root / "prepared"
        self.assertEqual(result["checkpoint_steps"], [0, 1, 2])
        self.assertEqual(result["total_prompts"], 864)
        self.assertEqual(result["dev_worst_case_forward_calls"], 15552)
        self.assertEqual(result["parity_worst_case_forward_calls"], 216)
        self.assertEqual(result["worst_case_forward_calls"], 15768)
        self.assertEqual(result["native_spec_sha256"], digest(read(prepared / "native-spec.json")))
        self.assertEqual(result["attachment_contract"]["sha256"], file_hash(self.contract))
        self.assertEqual(len(read(prepared / "development-suite.json")), 288)
        self.assertTrue(result["shared_parity_dev_metering_required_before_execution"])
        self.assertFalse(result["confirmation_authorized"])
        self.assertFalse(result["permits_model_promotion"])
        self.assertEqual(result["native_model_loads"], 0)
        self.assertFalse(any("confirmation" in item.name for item in prepared.iterdir()))
        with self.assertRaises(Blocked):
            self.invoke()

    def test_budget_must_cover_dev_and_both_parity_traces(self):
        self.proposal["runtime"]["max_native_forward_calls"] = 15552
        with self.assertRaisesRegex(Blocked, "aggregate forward budget"):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_contract_replaced_between_validations_is_not_reviewed(self):
        def replace_contract(path, runtime):
            with self.contract.open("a", encoding="utf-8") as stream:
                stream.write("\n")  # Valid unchanged JSON; a new unreviewed file identity.
            return native_spec_from_attachment(path, runtime)
        with patch("experiments.rineya_generalization_dynamics_v1.attachment_cli.native_spec_from_attachment",
                   side_effect=replace_contract), self.assertRaisesRegex(Blocked, "specification changed"):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_contract_changed_after_specification_blocks_publication(self):
        def change_after_specification(path, runtime):
            spec = native_spec_from_attachment(path, runtime)
            with self.contract.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            return spec
        with patch("experiments.rineya_generalization_dynamics_v1.attachment_cli.native_spec_from_attachment",
                   side_effect=change_after_specification), self.assertRaisesRegex(Blocked, "contract changed"):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_proposal_changed_after_capture_blocks_publication(self):
        def change_proposal(path):
            attachment = validate_attachment(path)
            with self.proposal_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            return attachment
        with patch("experiments.rineya_generalization_dynamics_v1.attachment_cli.validate_attachment",
                   side_effect=change_proposal), self.assertRaisesRegex(Blocked, "proposal changed"):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_asset_changed_before_freeze_cannot_replace_validated_registry(self):
        def replace_before_freeze(spec):
            with (Path(spec["base_model"]) / "model.safetensors").open("ab") as stream:
                stream.write(b"EXCLUDED_FIXTURE_MUTATION")
            return freeze_registry(spec)
        with patch("experiments.rineya_generalization_dynamics_v1.attachment_cli.freeze_registry",
                   side_effect=replace_before_freeze), self.assertRaisesRegex(Blocked, "registry differs"):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_asset_changed_after_freeze_blocks_publication(self):
        def replace_after_freeze(spec):
            registry = freeze_registry(spec)
            with (Path(spec["base_model"]) / "model.safetensors").open("ab") as stream:
                stream.write(b"EXCLUDED_FIXTURE_MUTATION")
            return registry
        with patch("experiments.rineya_generalization_dynamics_v1.attachment_cli.freeze_registry",
                   side_effect=replace_after_freeze), self.assertRaisesRegex(Blocked, "changed"):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_code_identity_changed_during_preparation_blocks_publication(self):
        with patch("experiments.rineya_generalization_dynamics_v1.attachment_cli.code_hash",
                   side_effect=[self.proposal["code_sha256"], "0" * 64]), \
             self.assertRaisesRegex(Blocked, "GD code changed"):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_guard_and_review_identity_changes_block(self):
        for field, value in (("code_sha256", "0" * 64), ("checkpoint_steps", [0, 1, 3]),
                             ("confirmation_authorized", True)):
            old = self.proposal[field]
            self.proposal[field] = value
            with self.subTest(field=field), self.assertRaises(Blocked):
                self.invoke()
            self.proposal[field] = old
        self.proposal["runtime"]["min_free_ram_gib"] = 1.0
        with self.assertRaises(Blocked):
            self.invoke()
        self.assertFalse((self.root / "prepared").exists())

    def test_sibling_source_checkout_output_is_refused(self):
        business_root = Path(__file__).resolve().parents[4]
        for destination in (business_root / "ChatBubble-post-training-Mimo" / "gd001-forbidden-fixture-output",
                            business_root / "gd001-records" / ".CODEX" / "gd001-forbidden-fixture-output"):
            with self.subTest(destination=destination), self.assertRaisesRegex(
                    Blocked, "external BusinessProject/gd001-records"):
                self.invoke(destination)
            self.assertFalse(destination.exists())

    def test_mechanical_label_cannot_prepare_native_spec(self):
        self.contract = build_fixture(self.root / "mechanical")
        self.proposal["attachment_contract"] = {"path": str(self.contract), "sha256": file_hash(self.contract)}
        with self.assertRaisesRegex(Blocked, "Mechanical attachment fixtures"):
            self.invoke()

    def test_cli_reports_missing_payload_without_outputs(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["check", "--contract", str(self.root / "missing.json")])
        self.assertEqual(code, 2)
        self.assertIn('"status": "BLOCKED"', output.getvalue())

    def test_protected_proposal_or_contract_blocks_before_read(self):
        hidden = str(self.root / "sealed_eval" / "metadata.json")
        for contract, proposal in ((hidden, str(self.proposal_path)), (str(self.contract), hidden)):
            with self.subTest(contract=contract, proposal=proposal), \
                 patch.object(Path, "read_bytes") as reader:
                with self.assertRaisesRegex(Blocked, "Protected banks"):
                    prepare_dev(contract, proposal, str(self.root / "prepared"))
                reader.assert_not_called()
        self.assertFalse((self.root / "prepared").exists())

    def test_protected_exposure_inventory_blocks_before_hash_or_read(self):
        attachment = validate_attachment(str(self.contract))
        reference = {"path": str(self.root / "ANY_READ-bank" / "exposure.jsonl"), "sha256": "0" * 64}
        with self.assertRaisesRegex(Blocked, "Protected banks"):
            audit_producer_prompt_overlap([], attachment, [reference])

    def test_raw_question_overlap_blocks_despite_distinct_rendering_and_parent(self):
        attachment = validate_attachment(str(self.contract))
        # This question was consumed only inside a different rendered template.
        # The new DEV parent differs, so parent-only auditing also would miss it.
        rows = [{"prompt": "keep repeat", "parent_id": "fresh-dev-parent"}]
        with self.assertRaisesRegex(Blocked, "prompt appears"):
            audit_producer_prompt_overlap(rows, attachment, [])

    def test_public_qualification_and_dev_exposures_are_exact_exclusions(self):
        import json
        attachment = validate_attachment(str(self.contract))
        path = self.root / "open-producer-exposures.jsonl"
        path.write_text(json.dumps({"scope": "public_qualification", "public": True,
            "prompt": "Public producer question", "rendered_prompt": "<user>Public producer question</user>",
            "parent_id": "producer-parent", "parent_namespace": "producer-qualified-dev"}) + "\n", encoding="utf-8")
        ref = {"path": str(path), "sha256": file_hash(path)}
        for row in ({"prompt": "Public producer question", "parent_id": "new-dev-parent"},
                    {"prompt": "Unrelated question", "parent_id": "producer-parent"}):
            with self.subTest(row=row), self.assertRaises(Blocked):
                audit_producer_prompt_overlap([row], attachment, [ref])
        outcome = audit_producer_prompt_overlap(
            [{"prompt": "Unrelated question", "parent_id": "new-dev-parent"}], attachment, [ref])
        self.assertEqual(outcome["public_exposure_count"], 1)
        self.assertFalse(outcome["semantic_decontamination"])


if __name__ == "__main__":
    unittest.main()
