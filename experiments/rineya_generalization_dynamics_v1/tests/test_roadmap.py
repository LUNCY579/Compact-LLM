"""Offline aggregate custody and readiness graph checks; no model evidence."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import Blocked, code_hash, digest, file_hash, read, signed, write_new
from experiments.rineya_generalization_dynamics_v1.roadmap import DOMAINS, SCHEMA, prepare_roadmap, public_metadata, review_m1_handoff, review_supervised_handoff, verify_roadmap_packet
from experiments.rineya_generalization_dynamics_v1.roadmap_cli import main
from experiments.rineya_generalization_dynamics_v1.protected_paths import reject_protected_path


class RoadmapTests(unittest.TestCase):
    def setUp(self):
        records = Path(__file__).resolve().parents[4] / "gd001-records" / "roadmap-fixtures"
        records.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=records)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "public-summary.json"
        write_new(source, {"scope": "mechanical_fixture_public_summary", "passed": 1})
        self.ref = {"path": str(source), "sha256": file_hash(source), "access": "public_aggregate"}
        self.spec = {"schema": SCHEMA, "scope": "OFFLINE_PUBLIC_AGGREGATE_AND_HANDOFF_PREPARATION",
            "code_sha256": code_hash(), "training_authorized": False, "native_execution_authorized": False,
            "confirmation_authorized": False, "external_messages_authorized": False,
            "owners": [{"lane": "data", "chat_id": "toy-owner-only", "title": "Toy existing owner",
                        "workspace": str(self.root / "source")}],
            "evidence": [{"id": "toy_aggregate", "reference": self.ref}],
            "observations": [{"owner_lane": "data", "evidence_level": "mechanical", "effect_type": "data_mechanics",
                "facts": {"audited_rows": 1, "admitted_rows": 0}, "domains": ["instruction_following"],
                "evidence_ids": ["toy_aggregate"], "limitations": ["Fixture only; no learner evidence"],
                "quality_card_status": "BLOCKED_MISSING_FULL_IDENTITY_OR_QUALIFICATION"}],
            "lanes": [{"id": "data", "status": "BLOCKED_PREREQUISITES", "scope": "OWNER_HANDOFF_ONLY",
                "execution_authorized": False, "evidence_ids": ["toy_aggregate"], "depends_on": [],
                "next_action": "Original owner completes qualified review", "prerequisites": ["Independent native/mask review"]}]}
        self.sequence = 0

    def invoke(self, output=None):
        self.sequence += 1
        spec = self.root / f"spec-{self.sequence}.json"
        write_new(spec, signed(self.spec))
        return prepare_roadmap(str(spec), str(output or self.root / "packet"))

    def test_packet_retains_exact_metadata_and_explicit_unknowns(self):
        with patch("experiments.rineya_generalization_dynamics_v1.native.NativeEvaluator.load",
                   side_effect=AssertionError("No native execution")):
            result = self.invoke()
        output = self.root / "packet"
        verification = verify_roadmap_packet(str(output))
        self.assertEqual(verification["status"], "OFFLINE_PACKET_HASHES_VERIFIED")
        self.assertFalse(verification["execution_authorized"])
        card = read(output / "quality-scorecard.json")
        self.assertEqual(card["domains"], list(DOMAINS))
        self.assertIsNone(card["overall_winner"])
        self.assertEqual(card["fully_qualified_quality_cards"], [])
        self.assertFalse(card["balanced_quality_established"])
        self.assertEqual(result["data_rows_admitted"], 0)
        self.assertEqual(result["native_model_loads"], 0)
        snapshots = read(output / "snapshot-receipts.json")["snapshots"]
        self.assertEqual(file_hash(Path(snapshots[0]["path"])), self.ref["sha256"])
        self.assertFalse(any("confirmation" in p.name for p in output.iterdir()))
        with self.assertRaises(Blocked):
            self.invoke()

    def test_changed_upstream_metadata_blocks_before_packet_creation(self):
        Path(self.ref["path"]).write_text('{}', encoding="utf-8")
        with self.assertRaisesRegex(Blocked, "Public aggregate changed"):
            self.invoke()
        self.assertFalse((self.root / "packet").exists())

    def test_bank_ref_and_execution_flags_cannot_be_imported(self):
        self.spec["evidence"][0]["reference"]["access"] = "confirmation_bank"
        with self.assertRaises(Blocked):
            self.invoke()
        self.spec["evidence"][0]["reference"]["access"] = "public_aggregate"
        for flag in ("training_authorized", "native_execution_authorized", "confirmation_authorized", "external_messages_authorized"):
            self.spec[flag] = True
            with self.subTest(flag=flag), self.assertRaises(Blocked):
                self.invoke()
            self.spec[flag] = False

    def test_source_and_codex_outputs_are_rejected(self):
        workspace = Path(__file__).resolve().parents[3]
        for destination in (workspace / "bad-roadmap-output", workspace.parent / "ChatBubble-post-training-Mimo" / "bad-roadmap-output",
                            workspace.parent / "gd001-records" / ".CODEX" / "bad-roadmap-output"):
            with self.subTest(destination=destination), self.assertRaises(Blocked):
                self.invoke(destination)
            self.assertFalse(destination.exists())

    def test_unqualified_aggregate_cannot_become_confirmed_or_native_ready(self):
        self.spec["observations"][0]["evidence_level"] = "confirmed"
        with self.assertRaises(Blocked):
            self.invoke()
        self.spec["observations"][0]["evidence_level"] = "mechanical"
        self.spec["lanes"][0]["status"] = "NATIVE_READY"
        with self.assertRaises(Blocked):
            self.invoke()

    def test_boolean_fixture_facts_cannot_become_learned_exploratory_evidence(self):
        source = Path(self.ref["path"])
        source.write_text('{"scope":"public_aggregate"}', encoding="utf-8")
        self.ref["sha256"] = file_hash(source)
        observation = self.spec["observations"][0]
        observation.update(evidence_level="exploratory", effect_type="learned")
        for facts in ({"fixture": True}, {"nested": [{"fixture": True}]}):
            observation["facts"] = facts
            with self.subTest(facts=facts), self.assertRaisesRegex(Blocked, "fixture facts"):
                self.invoke()
            self.assertFalse((self.root / "packet").exists())

    def test_bound_boolean_fixture_receipts_cannot_become_exploratory_evidence(self):
        observation = self.spec["observations"][0]
        observation.update(evidence_level="exploratory", effect_type="learned")
        source = Path(self.ref["path"])
        for payload in ({"fixture": True}, {"nested": [{"fixture": True}]}):
            source.write_text(json.dumps(payload), encoding="utf-8")
            self.ref["sha256"] = file_hash(source)
            with self.subTest(payload=payload), self.assertRaisesRegex(Blocked, "Bound fixture receipts"):
                self.invoke()
            self.assertFalse((self.root / "packet").exists())

    def test_explicit_false_fixture_flag_does_not_create_a_fixture_claim(self):
        source = Path(self.ref["path"])
        source.write_text('{"fixture":false}', encoding="utf-8")
        self.ref["sha256"] = file_hash(source)
        self.spec["observations"][0].update(evidence_level="exploratory",
                                           facts={"fixture": False})
        result = self.invoke()
        self.assertEqual(result["data_rows_admitted"], 0)
        self.assertFalse(read(self.root / "packet" / "quality-scorecard.json")["balanced_quality_established"])

    def test_protected_paths_refuse_before_any_content_hash_or_read(self):
        bad = {**self.ref, "path": str(self.root / "ConfirMATion" / "CHECK.json")}
        with patch("experiments.rineya_generalization_dynamics_v1.roadmap.file_hash",
                   side_effect=AssertionError("Protected content must not be read")):
            with self.assertRaisesRegex(Blocked, "Protected banks"):
                public_metadata(bad)

    def test_protected_prefix_components_refuse_before_resolve_or_content_access(self):
        for component in ("sealed_eval", "confirmation-bank", "SPENT_2026", "ANY_READ-bank", "ANY-READ-bank", "barred-bank",
                          "CHECK-bank", "FINAL.instrument", "bank_controls"):
            bad = {**self.ref, "path": str(self.root / component / "public-summary.json")}
            with self.subTest(component=component), \
                    patch.object(Path, "resolve", side_effect=AssertionError("Reject lexical path before resolve")), \
                    patch("experiments.rineya_generalization_dynamics_v1.roadmap.file_hash",
                          side_effect=AssertionError("Protected content must not be hashed")), \
                    patch.object(Path, "read_bytes", side_effect=AssertionError("Protected content must not be read")):
                with self.assertRaisesRegex(Blocked, "Protected banks"):
                    public_metadata(bad)

    def test_resolved_protected_target_refuses_before_content_access(self):
        target = self.root / "sealed_eval" / "public-summary.json"
        with patch.object(Path, "resolve", return_value=target), \
                patch("experiments.rineya_generalization_dynamics_v1.roadmap.file_hash",
                      side_effect=AssertionError("Protected target must not be hashed")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("Protected target must not be read")):
            with self.assertRaisesRegex(Blocked, "Protected banks"):
                public_metadata(self.ref)

    def test_core_path_guard_permits_original_final_exports_and_checkpoint_names(self):
        for component in ("final", "checkpoint-1200"):
            path = self.root / component / "model.safetensors"
            self.assertEqual(reject_protected_path(path), path.resolve())
        self.assertEqual(reject_protected_path(self.root / "checkpoint-1200", include_instruments=True),
                         (self.root / "checkpoint-1200").resolve())

    def test_card_cli_refuses_protected_input_before_json_read(self):
        result = io.StringIO()
        with contextlib.redirect_stdout(result), \
                patch("experiments.rineya_generalization_dynamics_v1.roadmap_cli.read",
                      side_effect=AssertionError("Protected card must not be read")):
            code = main(["check-card", "--input", str(self.root / "confirmation-bank" / "summary.json")])
        self.assertEqual(code, 2)
        self.assertIn("Protected banks", result.getvalue())

    def test_card_cli_refuses_model_payload_before_content_read(self):
        path = self.root / "model.safetensors"
        path.write_bytes(b"SYNTHETIC_NOT_MODEL_WEIGHTS")
        result = io.StringIO()
        with contextlib.redirect_stdout(result), \
                patch("experiments.rineya_generalization_dynamics_v1.roadmap_cli.read",
                      side_effect=AssertionError("Model payload must not be read by card validation")):
            code = main(["check-card", "--input", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("bounded absolute regular JSON", result.getvalue())

    def test_explicit_fixture_provenance_cannot_be_upgraded(self):
        self.spec["observations"][0]["evidence_level"] = "exploratory"
        self.spec["observations"][0]["effect_type"] = "learned"
        with self.assertRaisesRegex(Blocked, "Bound fixture receipts"):
            self.invoke()

    def test_excluded_query_witness_facts_cannot_become_native_observations(self):
        from experiments.rineya_generalization_dynamics_v1.query_dependence_witness import audit_witness, build_witness
        witness = build_witness()
        for payload in (witness, audit_witness(witness)):
            for level in ("exploratory", "native_qualification"):
                with self.subTest(kind=payload["evidence_kind"], level=level):
                    self.spec["observations"][0].update(evidence_level=level, effect_type="learned", facts=payload)
                    # The original bound summary has no fixture marker; facts still cannot promote a witness.
                    Path(self.ref["path"]).write_text('{"scope":"public_aggregate_metadata"}', encoding="utf-8")
                    self.ref["sha256"] = file_hash(Path(self.ref["path"]))
                    with self.assertRaisesRegex(Blocked, "Explicit fixture facts"):
                        self.invoke()

    def test_excluded_query_witness_receipts_cannot_become_native_observations(self):
        from experiments.rineya_generalization_dynamics_v1.query_dependence_witness import audit_witness, build_witness
        witness = build_witness()
        for payload in (witness, audit_witness(witness)):
            for level in ("exploratory", "native_qualification"):
                with self.subTest(kind=payload["evidence_kind"], level=level):
                    Path(self.ref["path"]).write_text(__import__("json").dumps(payload), encoding="utf-8")
                    self.ref["sha256"] = file_hash(Path(self.ref["path"]))
                    self.spec["observations"][0].update(evidence_level=level, effect_type="learned", facts={"declared": "native"})
                    with self.assertRaisesRegex(Blocked, "Bound fixture receipts"):
                        self.invoke()

    def test_quality_card_interface_integrates_without_native_execution(self):
        from experiments.rineya_generalization_dynamics_v1.tests.test_quality_card import build_fixture_card
        path = self.root / "mechanical-card.json"
        write_new(path, build_fixture_card())
        self.spec["evidence"].append({"id": "mechanical_card", "reference": {
            "path": str(path), "sha256": file_hash(path), "access": "public_aggregate"}})
        self.spec["quality_card_evidence_ids"] = ["mechanical_card"]
        self.invoke()
        cards = read(self.root / "packet" / "quality-scorecard.json")["validated_quality_cards"]
        self.assertEqual(len(cards), 1)
        self.assertFalse(cards[0]["native_execution_verified_by_this_module"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["check-card", "--input", str(path)])
        self.assertEqual(code, 0)

    def test_quality_card_packet_uses_captured_bytes_without_reopening_json(self):
        from experiments.rineya_generalization_dynamics_v1.tests.test_quality_card import build_fixture_card
        path = self.root / "captured-mechanical-card.json"
        write_new(path, build_fixture_card())
        self.spec["evidence"].append({"id": "captured_card", "reference": {
            "path": str(path), "sha256": file_hash(path), "access": "public_aggregate"}})
        self.spec["quality_card_evidence_ids"] = ["captured_card"]
        with patch("experiments.rineya_generalization_dynamics_v1.roadmap.read",
                   side_effect=AssertionError("Do not reopen captured owner JSON")):
            self.invoke()

    def test_cycle_in_owner_readiness_is_rejected(self):
        owner = copy.deepcopy(self.spec["owners"][0])
        owner["lane"] = "m1"
        self.spec["owners"].append(owner)
        lane = copy.deepcopy(self.spec["lanes"][0])
        lane["id"], lane["depends_on"] = "m1", ["data"]
        self.spec["lanes"].append(lane)
        self.spec["lanes"][0]["depends_on"] = ["m1"]
        with self.assertRaisesRegex(Blocked, "Cyclic"):
            self.invoke()

    def test_net_gain_keeps_independent_damage_and_zero_support_undefined(self):
        self.spec["observations"][0]["paired_counts"] = [{"domain": "instruction_following", "endpoint": "fixture_exact",
            "denominator": 10, "source_successes": 5, "candidate_successes": 6,
            "gained": 3, "lost": 2, "retained": 3}]
        self.invoke()
        endpoint = read(self.root / "packet" / "quality-scorecard.json")["observations"][0]["paired_counts"][0]
        self.assertEqual(endpoint["net_gain"], 1)
        self.assertEqual(endpoint["lost"], 2)
        self.assertEqual(endpoint["retained_fraction"], .6)

    def test_cross_domain_aggregate_does_not_become_domain_quality_coverage(self):
        self.spec["observations"][0]["paired_counts"] = [{"domain": "cross_domain_aggregate", "endpoint": "toy_mixed_panel",
            "denominator": 10, "source_successes": 5, "candidate_successes": 6,
            "gained": 3, "lost": 2, "retained": 3}]
        self.invoke()
        card = read(self.root / "packet" / "quality-scorecard.json")
        self.assertTrue(all(item["status"] == "UNMEASURED" and item["score"] is None for item in card["domain_coverage"].values()))

    def test_invalid_paired_count_and_resigned_inventory_tamper_block(self):
        self.spec["observations"][0]["paired_counts"] = [{"domain": "instruction_following", "endpoint": "fixture_exact",
            "denominator": 10, "source_successes": True, "candidate_successes": 6,
            "gained": 3, "lost": 2, "retained": 3}]
        with self.assertRaises(Blocked):
            self.invoke()
        self.spec["observations"][0].pop("paired_counts")
        self.invoke()
        path = self.root / "packet" / "snapshot-receipts.json"
        changed = read(path)
        changed.pop("sha256")
        changed["snapshots"] = []
        path.write_text(__import__("json").dumps(signed(changed)), encoding="utf-8")
        with self.assertRaisesRegex(Blocked, "Snapshot inventory changed"):
            verify_roadmap_packet(str(self.root / "packet"))

    def test_snapshot_or_scorecard_tamper_is_detected(self):
        self.invoke()
        output = self.root / "packet"
        snapshots = read(output / "snapshot-receipts.json")["snapshots"]
        Path(snapshots[0]["path"]).write_text('{}', encoding="utf-8")
        with self.assertRaises(Blocked):
            verify_roadmap_packet(str(output))

    def test_missing_cli_input_returns_structured_blocked(self):
        result = io.StringIO()
        with contextlib.redirect_stdout(result):
            status = main(["verify", "--output", str(self.root / "missing")])
        self.assertEqual(status, 2)
        self.assertIn('"status": "BLOCKED"', result.getvalue())

    def m1_fixture(self):
        executable = self.root / "python.exe"
        executable.write_bytes(b"FIXTURE_NOT_AN_EXECUTABLE")
        runtime = {"python.exe": {"path": str(executable), "bytes": executable.stat().st_size,
                                  "sha256": file_hash(executable)}}
        fixture = {"schema": "mlp_ship.launcher_identity_fixture/1", "base_runtime_files": runtime,
            "status": "PASS_OWNED_PID_FIXTURE_ONLY", "child_exited": True, "child_returncode": 0,
            "observation": {"guard_passed": True, "ppid": 123, "pid": 456, "supervisor": "123", "executable": str(executable)},
            "parent": {"pid": 123}, "child": {"pid": 456}, "model_loads": 0, "native_forwards": 0,
            "backward_calls": 0, "optimizer_steps": 0, "task_rows_read": 0, "new_native_execution_authorized": False}
        fixture_path = self.root / "launcher-fixture.json"
        write_new(fixture_path, fixture)
        proposal = {"schema": "mlp_ship.prospective_retry_proposal/1", "status": "PREPARED_OFFLINE_NOT_AUTHORIZED",
            "native_attempts_authorized_now": 0, "original_authorized_attempt_consumed": True,
            "existing_runner_unchanged": True, "source_pins_tolerances_guards_unchanged": True,
            "cal_check_authorized": False, "deployment_authorized": False, "task_bank_reads_allowed": False,
            "permissions": {"training": False, "sealed_evaluation": False},
            "command": [str(executable), "-B", "-m", "experiments.matched_mlp_shipping_v1.run", "qualify",
                        "--out", str(self.root / "future-native-output"), "--device", "cuda"],
            "base_runtime_files": runtime, "launch_identity_fixture_sha256": file_hash(fixture_path),
            "source_snapshot": {"branch": "toy-owned-fixture", "head": "1"*40, "model_commit": "2"*40,
                "source_sha256": digest("toy-source"), "adapter_manifest_sha256": digest("toy-manifest"),
                "adapter_tensor_sha256": digest("toy-adapter"), "model_files": {
                    name: digest("toy-"+name) for name in ("model.safetensors", "tokenizer.json", "config.json")}},
            "limits": {"wall_seconds": 900}, "environment": {"HF_HUB_OFFLINE": "1"},
            "requirements_before_execution": ["New original-owner authorization; fixture is not a native qualification"]}
        return proposal, fixture_path, executable

    def test_m1_exact_launcher_handoff_grants_no_retry(self):
        proposal, fixture, _ = self.m1_fixture()
        path = self.root / "m1-proposal.json"
        write_new(path, proposal)
        ref = lambda p: {"path": str(p), "sha256": file_hash(p), "access": "public_aggregate"}
        with patch("experiments.rineya_generalization_dynamics_v1.roadmap.read",
                   side_effect=AssertionError("Do not reopen captured owner JSON")):
            result = review_m1_handoff(ref(path), ref(fixture))
        self.assertEqual(result["native_attempts_authorized"], 0)
        self.assertFalse(result["native_model_qualification"])
        self.assertFalse(result["cal_check_authorized"])

    def test_m1_wrong_launcher_or_mutated_runtime_cannot_reuse_fixture(self):
        proposal, fixture, executable = self.m1_fixture()
        proposal["command"][0] = str(self.root / "venv-redirector.exe")
        path = self.root / "wrong-m1-proposal.json"
        write_new(path, proposal)
        ref = lambda p: {"path": str(p), "sha256": file_hash(p), "access": "public_aggregate"}
        with self.assertRaisesRegex(Blocked, "Direct launch"):
            review_m1_handoff(ref(path), ref(fixture))
        proposal["command"][0] = str(executable)
        another = self.root / "mutated-runtime-proposal.json"
        write_new(another, proposal)
        executable.write_bytes(b"CHANGED_TOY_BYTES")
        with self.assertRaisesRegex(Blocked, "bytes changed"):
            review_m1_handoff(ref(another), ref(fixture))

    def test_m1_nested_runtime_protected_path_refuses_before_content_hash(self):
        proposal, fixture, _ = self.m1_fixture()
        protected = self.root / "ANY_READ-bank" / "python312.dll"
        runtime = proposal["base_runtime_files"]
        runtime["python312.dll"] = {"path": str(protected), "bytes": 2, "sha256": digest("fixture bytes")}
        fixture_value = read(fixture)
        fixture_value["base_runtime_files"] = copy.deepcopy(runtime)
        fixture.write_text(__import__("json").dumps(fixture_value), encoding="utf-8")
        proposal["launch_identity_fixture_sha256"] = file_hash(fixture)
        path = self.root / "protected-runtime-proposal.json"
        write_new(path, proposal)
        ref = lambda p: {"path": str(p), "sha256": file_hash(p), "access": "public_aggregate"}
        actual_hash = file_hash
        def guarded_hash(value):
            self.assertNotEqual(Path(value), protected, "Protected runtime body must not be hashed")
            return actual_hash(value)
        with patch("experiments.rineya_generalization_dynamics_v1.roadmap.file_hash", side_effect=guarded_hash):
            with self.assertRaisesRegex(Blocked, "Protected banks"):
                review_m1_handoff(ref(path), ref(fixture))

    def test_resigned_supplemental_owner_proposal_cannot_escape_packet_binding(self):
        proposal, fixture, _ = self.m1_fixture()
        path = self.root / "bound-m1-proposal.json"
        write_new(path, proposal)
        for identifier, p in (("m1_proposal", path), ("m1_fixture", fixture)):
            self.spec["evidence"].append({"id": identifier, "reference": {
                "path": str(p), "sha256": file_hash(p), "access": "public_aggregate"}})
        self.spec["m1_review"] = {"proposal_id": "m1_proposal", "fixture_id": "m1_fixture"}
        self.invoke()
        handoff = self.root / "packet" / "M1_OWNER_HANDOFF.json"
        value = read(handoff)
        value.pop("sha256")
        value["original_limits"]["wall_seconds"] = 10000
        handoff.write_text(__import__("json").dumps(signed(value)), encoding="utf-8")
        with self.assertRaisesRegex(Blocked, "Supplemental owner proposal changed"):
            verify_roadmap_packet(str(self.root / "packet"))

    def supervised_fixture(self):
        draft = {"status": "CPU_SCHEDULE_PREPARATION_NOT_EXECUTION_PERMIT", "native_launch_authorized": False,
            "confirmation_authorized": False, "permissions": {"training": False, "native_inference": False},
            "student": {"name": "TOY_NOT_RINEYA", "base_parameters": 596049920,
                        **{key: digest("toy-"+key) for key in ("weight_sha256", "tokenizer_sha256", "config_sha256", "tokenizer_config_sha256")}},
            "seed": 101, "dose": {"whole_sequences_per_arm": 540, "full_updates_per_adapted_arm": 27,
                "sequences_per_update": 20, **{key: False for key in
                    ("RS_training", "dropping", "packing", "padding", "replay_added", "truncation", "warmstart")}},
            "adaptation": {"method": "qv_lora", "rank": 8, "trainable_parameters": 1146880, "trainable_tensors": 112,
                "loss": "completion_plus_EOS_mean_per_sequence_then_mean_over_accumulation"},
            "optimizer": {"name": "AdamW", "optimizer_state_dtype": "float32", "learning_rates_by_1based_update": [1e-4]*26+[0.0]},
            "source_and_decision_pins": {"toy-original": digest("toy")}, "pending_gates": ["Real native scoring and retention"]}
        qualifier = {"native_launch_authorized": False, "production_training_authorized": False,
            "protected_bank_access": False, "quality_gain_established": False, "runtime_candidate": {"backend": "TOY"},
            "resources": {"whole_scope_seconds": 600}, "remaining_prerequisites": ["Native physical fit"]}
        return draft, qualifier

    def supervised_invoke(self, draft, qualifier):
        self.sequence += 1
        path, qualification = self.root / f"draft-{self.sequence}.json", self.root / f"qualification-{self.sequence}.json"
        write_new(path, draft)
        write_new(qualification, qualifier)
        ref = lambda p: {"path": str(p), "sha256": file_hash(p), "access": "public_aggregate"}
        return review_supervised_handoff(ref(path), ref(qualification))

    def test_owner_seed_and_full_zero_lr_update_schedule_remain_exact(self):
        draft, qualifier = self.supervised_fixture()
        with patch("experiments.rineya_generalization_dynamics_v1.roadmap.read",
                   side_effect=AssertionError("Do not reopen captured owner JSON")):
            result = self.supervised_invoke(draft, qualifier)
        self.assertEqual(result["owner_draft_seed"], 101)
        self.assertEqual(result["optimizer"]["learning_rates_by_1based_update"], draft["optimizer"]["learning_rates_by_1based_update"])
        self.assertEqual(result["checkpoint_observer_proposal"]["first_gd_dev_steps"], [0, 9, 27])
        self.assertFalse(result["qualification_states_can_warmstart_study"])
        self.assertFalse(result["training_authorized"])

    def test_shortened_schedule_or_qualification_permission_cannot_enable_training(self):
        draft, qualifier = self.supervised_fixture()
        draft["optimizer"]["learning_rates_by_1based_update"].pop()
        with self.assertRaisesRegex(Blocked, "Retain every exact LR"):
            self.supervised_invoke(draft, qualifier)
        draft, qualifier = self.supervised_fixture()
        qualifier["production_training_authorized"] = True
        with self.assertRaises(Blocked):
            self.supervised_invoke(draft, qualifier)

    def test_pending_data_packet_cli_keeps_all_admission_and_native_gates_closed(self):
        from experiments.rineya_generalization_dynamics_v1.data_admission import make_pending_packet
        contract = self.root / "public-diagnostic-contract.json"
        write_new(contract, {"parents": 28, "assigned_trials": 224, "renderings": ["A", "B"],
                             "roles": ["semantic", "adversarial"], "candidates_per_packet": 2})
        packet = make_pending_packet(owner="existing-data-owner", packet_id="toy-pending-only",
            public_metadata_refs=[{"path": self.ref["path"], "sha256": self.ref["sha256"], "role": "PUBLIC_PENDING_METADATA"}],
            diagnostic_contract_ref={"path": str(contract), "sha256": file_hash(contract), "role": "PUBLIC_DIAGNOSTIC_CONTRACT"})
        path = self.root / "pending-data-packet.json"
        write_new(path, packet)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["check-data", "--packet", str(path)])
        self.assertEqual(status, 2)
        self.assertIn('"status": "BLOCKED"', output.getvalue())
        self.assertIn('completed224', output.getvalue())


if __name__ == "__main__":
    unittest.main()
