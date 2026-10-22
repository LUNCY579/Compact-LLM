"""Toy retained packets test offline contracts, never real admission or model quality."""
from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import digest, file_hash, verify
from experiments.rineya_generalization_dynamics_v1.data_admission import (
    GATES, SCHEMA, assess_data_admission, make_pending_packet,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def reference(path, role):
    return {"path": str(path.resolve()), "sha256": file_hash(path), "role": role}


def build_data_packet(root: Path, *, fixture=True) -> Path:
    """Minimal schema factory. All bytes are toy data, even when testing declarations.

    fixture=False simulates caller-declared reviewed production metadata solely to
    exercise READY_FOR_OWNER_ADMISSION_REVIEW. It is never native evidence, a
    dataset release, or a permission to use these test controls for training.
    """
    root.mkdir(parents=True, exist_ok=True)
    kind = "mechanical_fixture" if fixture else "real_observation"
    sources = {"evidence_kind": kind, "source_id": "toy-source", "split": "TRAIN", "rows": [
        {"source_row_id": "s1", "parent_id": "p1", "family": "arithmetic", "raw_record": {"text": "1+1"}},
        {"source_row_id": "s2", "parent_id": "p2", "family": "instruction", "raw_record": {"text": "empty"}},
        {"source_row_id": "s3", "parent_id": "p3", "family": "arithmetic", "raw_record": {"text": "ambiguous"}}]}
    rows = [{"row_id": "t1", "parent_id": "p1", "family": "arithmetic", "split": "TRAIN",
             "source_id": "toy-source", "source_row_id": "s1", "prompt": "Return 1+1.", "target": "2"},
            {"row_id": "t2", "parent_id": "p2", "family": "instruction", "split": "TRAIN",
             "source_id": "toy-source", "source_row_id": "s2", "prompt": "Output the empty string.", "target": ""}]
    write(root / "sources.json", sources)
    write(root / "tasks.json", {"evidence_kind": kind, "rows": rows})
    (root / "tokenizer.bin").write_bytes(b"TOY TOKENIZER BYTES; NO TOKENIZER EXECUTION")
    write(root / "tokenizer.json", {"evidence_kind": kind, "vocab_size": 10, "eos_token_id": 9,
          "files": [reference(root / "tokenizer.bin", "TOKENIZER_BYTES")]})
    filters = []
    for index in range(1, 4):
        filters.append({"source_id": "toy-source", "source_row_id": f"s{index}",
            "decision": "rejected" if index == 3 else "accepted", "task_ids": [] if index == 3 else [f"t{index}"],
            "history": [{"operation": "toy-complete-source-review", "reason": "retained toy disposition",
                         "decision": "reject" if index == 3 else "keep"}]})
    write(root / "filtering.json", {"evidence_kind": kind, "rows": filters})
    tokenizer_ref = reference(root / "tokenizer.json", "TOKENIZER_METADATA")
    occurrences = []
    for ordinal, rid, step in ((0, "t1", 1), (1, "t2", 1), (2, "t1", 2)):
        row = next(row for row in rows if row["row_id"] == rid)
        response = [9] if row["target"] == "" else [5, 9]
        occurrences.append({"occurrence_id": f"o{ordinal}", "row_id": rid, "step": step,
            "tokenizer_sha256": tokenizer_ref["sha256"], "prompt": row["prompt"],
            "response_text": row["target"], "raw_rendered_prompt": "chat:" + row["prompt"],
            "rendered_prompt": "chat:" + row["prompt"], "rendering_history": [{"operation": "toy-chat-template"}],
            "original_prompt_ids": [1, 2], "prompt_ids": [1, 2], "response_ids": response,
            "input_ids": [1, 2] + response, "attention_mask": [1] * (2 + len(response)),
            "response_mask": [0, 0] + [1] * len(response), "labels": [-100, -100] + response,
            "truncation": {"policy": "none", "max_tokens": None, "history": []}})
    write(root / "consumption.json", {"evidence_kind": kind, "status": "PROPOSED_CONSUMPTION_NOT_TRAINING",
                                       "occurrences": occurrences})
    assets = {"source_refs": [reference(root / "sources.json", "TRAIN_SIDE_PAYLOAD")],
              "task_ref": reference(root / "tasks.json", "TRAIN_SIDE_PAYLOAD"), "tokenizer_ref": tokenizer_ref,
              "filtering_ref": reference(root / "filtering.json", "TRAIN_SIDE_PAYLOAD"),
              "consumption_ref": reference(root / "consumption.json", "TRAIN_SIDE_PAYLOAD")}
    dose = {"evidence_kind": kind, "purpose": "proposed_training_dose", "frozen_before_outcomes": True,
            "assets": assets, "occurrence_schedule": [{k: event[k] for k in ("occurrence_id", "row_id", "step")}
                                                       for event in occurrences],
            "counts": {"unique_tasks": 2, "occurrences": 3, "input_tokens": 11, "supervised_tokens": 5}}
    write(root / "dose.json", dose)
    packet = {"schema": SCHEMA, "evidence_kind": kind, "packet_id": "TOY_NOT_A_DATASET_RELEASE", "owner": "toy-owner",
              "assets": assets, "dose_ref": reference(root / "dose.json", "PROSPECTIVE_DOSE"), "gates": {}}
    binding = digest({"packet_id": packet["packet_id"], "dose_ref": packet["dose_ref"], "assets": assets})
    write(root / "rights-basis.json", {"evidence_kind": kind, "scope": "toy permission artifact"})
    write(root / "production-protocol.json", {"evidence_kind": kind, "scope": "toy separate production calibration protocol",
          "status": "PREREGISTERED", "assigned": 128, "max_error_upper": 0.05, "min_coverage": 0.9})
    write(root / "production-calibration.json", {"evidence_kind": kind, "status": "COMPLETED",
          "scope": "separately_reserved_production_grader", "registered_before_outcomes": True,
          "assigned": 128, "eligible_decisive": 128, "error_upper": 0.01, "registered_max_error_upper": 0.05,
          "registered_min_coverage": 0.9, "protocol_ref": reference(root / "production-protocol.json", "PUBLIC_PRODUCTION_PROTOCOL")})
    for gate in GATES:
        evidence = {"gate": gate, "binding_sha256": binding, "status": "PASS", "evidence_kind": kind,
                    "producer": "toy-method-author", "method": "invented toy observations; not native evidence"}
        if gate == "source_use_rights":
            evidence.update(source_refs=assets["source_refs"], permitted_use="model_training",
                            rights_basis_refs=[reference(root / "rights-basis.json", "PUBLIC_RIGHTS_EVIDENCE")])
        elif gate == "split_parent_family_closure":
            evidence.update(row_inventory=[{k: row[k] for k in ("row_id", "parent_id", "family", "split")} for row in rows],
                            cross_split_parent_collisions=[], unresolved_ancestry=[], exclusion_registry_sha256=digest("toy registry"))
        elif gate == "raw_rendered_overlap":
            evidence.update(row_ids=[r["row_id"] for r in rows], raw_prompt_collisions=[], rendered_prompt_collisions=[],
                            parent_collisions=[], scope="all_reserved_roles", reservation_manifest_sha256=digest("toy reservations"))
        elif gate in {"independent_solver", "independent_scorer"}:
            evidence["rows"] = [{"row_id": row["row_id"], "public_prompt_sha256": digest(row["prompt"]),
                                 "derived_target": row["target"], "exact_target_score": 1,
                                 "wrong_target": "wrong", "wrong_target_score": 0} for row in rows]
        elif gate == "native_rendering_eos_loss":
            evidence.update(runtime_sha256=digest("toy runtime"), tokenizer_execution_sha256=digest("toy tokenizer execution"),
                occurrences=[{"occurrence_id": event["occurrence_id"], "rendered_payload_sha256": digest(event),
                    "supervised_positions": list(range(2, len(event["input_ids"]))), "prompt_loss_tokens": 0,
                    "observed_response_loss": 0.7} for event in occurrences])
        else:
            evidence["calibration_ref"] = reference(root / "production-calibration.json", "PUBLIC_PRODUCTION_CALIBRATION")
        write(root / f"{gate}-evidence.json", evidence)
        evidence_ref = reference(root / f"{gate}-evidence.json", "PUBLIC_GATE_EVIDENCE")
        write(root / f"{gate}-review.json", {"gate": gate, "binding_sha256": binding, "status": "PASS",
              "evidence_kind": kind, "reviewer": "toy-nonauthor-reviewer", "scope": "production_data_readiness",
              "evidence_sha256": evidence_ref["sha256"]})
        write(root / f"{gate}.json", {"gate": gate, "binding_sha256": binding, "status": "PASS", "evidence_kind": kind,
              "evidence_ref": evidence_ref, "review_ref": reference(root / f"{gate}-review.json", "PUBLIC_GATE_REVIEW")})
        packet["gates"][gate] = reference(root / f"{gate}.json", "PUBLIC_GATE_RECEIPT")
    contract = {"parents": 28, "renderings": ["A", "B"], "candidates_per_packet": 2,
                "roles": ["semantic", "adversarial"], "assigned_trials": 224, "fresh_context_per_trial": True,
                "tools_permitted": False, "no_source_or_gold_to_reviewers": True, "evidence_kind": kind}
    write(root / "diagnostic-contract.json", contract)
    (root / "diagnostic-protocol.md").write_text("TOY fixed224 EXCLUDED diagnostic protocol", encoding="utf-8")
    parents = [f"opaque-parent-{n:02d}" for n in range(28)]
    assignments, outcomes = [], []
    for index, (parent, rendering, candidate, role) in enumerate(itertools.product(parents, ("A", "B"), ("clean", "fault"), ("semantic", "adversarial"))):
        aid, context = f"trial-{index:03d}", f"fresh-context-{index:03d}"
        slot = {"assignment_id": aid, "context_id": context, "parent_id": parent, "rendering": rendering,
                "candidate": candidate, "role": role, "public_packet_sha256": digest([parent, rendering]),
                "candidate_sha256": digest([parent, rendering, candidate])}
        assignments.append(slot)
        history = [{"kind": name, "ordinal": ordinal, "sha256": slot["public_packet_sha256"] if ordinal == 0 else
                    slot["candidate_sha256"] if ordinal == 3 else digest([aid, name]),
                    "at_utc": f"2026-10-05T00:00:0{ordinal}Z"} for ordinal, name in enumerate(
                        ("public_packet", "public_rubric", "commitment", "candidate", "assessment"))]
        outcomes.append({"assignment_id": aid, "context_id": context, "disposition": "completed",
             "history_observable": True, "reviewer_is_author": False, "observed_history": history,
             "candidate_qualification": "QUALIFIED_BY_CUSTODIAN", "vote": "ACCEPT" if candidate == "clean" else "REJECT"})
    write(root / "assignments.json", {"evidence_kind": kind, "frozen_before_candidates": True,
          "role": "EXCLUDED_DIAGNOSTIC", "parent_ids": parents, "assignments": assignments})
    assignment_ref = reference(root / "assignments.json", "PUBLIC_DIAGNOSTIC_ASSIGNMENTS")
    write(root / "outcomes.json", {"evidence_kind": kind, "status": "COMPLETED", "role": "EXCLUDED_DIAGNOSTIC",
          "producer": "toy-outcome-producer",
          "assignment_sha256": assignment_ref["sha256"], "outcomes": outcomes})
    write(root / "custodian.json", {"evidence_kind": kind, "status": "COMPLETED", "role": "EXCLUDED_DIAGNOSTIC",
          "producer": "toy-custodian", "assignment_sha256": assignment_ref["sha256"], "candidate_checks": [
              {"assignment_id": s["assignment_id"], "public_packet_sha256": s["public_packet_sha256"],
               "candidate_sha256": s["candidate_sha256"], "qualification": "QUALIFIED",
               "public_verification_sha256": digest([s["assignment_id"], "toy-public-verification"]),
               "failure_witness_sha256": digest([s["assignment_id"], "toy-fault-witness"])} for s in assignments]})
    custodian_ref = reference(root / "custodian.json", "PUBLIC_DIAGNOSTIC_CUSTODIAN")
    write(root / "diagnostic-review.json", {"evidence_kind": kind, "status": "COMPLETED", "role": "EXCLUDED_DIAGNOSTIC",
          "reviewer": "toy-nonauthor-history-reviewer", "assignment_sha256": assignment_ref["sha256"],
          "outcome_sha256": file_hash(root / "outcomes.json"), "custodian_sha256": custodian_ref["sha256"]})
    packet["diagnostic"] = {"contract_ref": reference(root / "diagnostic-contract.json", "PUBLIC_DIAGNOSTIC_CONTRACT"),
        "protocol_ref": reference(root / "diagnostic-protocol.md", "PUBLIC_DIAGNOSTIC_PROTOCOL"),
        "assignment_ref": assignment_ref, "outcome_ref": reference(root / "outcomes.json", "PUBLIC_DIAGNOSTIC_OUTCOMES"),
        "custodian_ref": custodian_ref, "review_ref": reference(root / "diagnostic-review.json", "PUBLIC_DIAGNOSTIC_REVIEW")}
    write(root / "packet.json", packet)
    return root / "packet.json"


class DataAdmissionTests(unittest.TestCase):
    def setUp(self):
        external = Path(__file__).resolve().parents[4] / "gd001-records" / "quality-roadmap-fixtures" / "data"
        external.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=external)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = build_data_packet(self.root, fixture=False)

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def change_gate(self, gate, mutation):
        evidence = self.read(f"{gate}-evidence.json")
        mutation(evidence)
        write(self.root / f"{gate}-evidence.json", evidence)
        receipt = self.read(f"{gate}.json")
        receipt["evidence_ref"] = reference(self.root / f"{gate}-evidence.json", "PUBLIC_GATE_EVIDENCE")
        review = self.read(f"{gate}-review.json")
        review["evidence_sha256"] = receipt["evidence_ref"]["sha256"]
        write(self.root / f"{gate}-review.json", review)
        receipt["review_ref"] = reference(self.root / f"{gate}-review.json", "PUBLIC_GATE_REVIEW")
        write(self.root / f"{gate}.json", receipt)
        packet = self.read("packet.json")
        packet["gates"][gate] = reference(self.root / f"{gate}.json", "PUBLIC_GATE_RECEIPT")
        write(self.path, packet)

    def change_diagnostic(self, name, mutation):
        value = self.read(name)
        mutation(value)
        write(self.root / name, value)
        packet = self.read("packet.json")
        key = "assignment_ref" if name == "assignments.json" else "outcome_ref"
        packet["diagnostic"][key] = reference(self.root / name, "PUBLIC_DIAGNOSTIC_ASSIGNMENTS" if key == "assignment_ref" else "PUBLIC_DIAGNOSTIC_OUTCOMES")
        if key == "assignment_ref":
            outcomes = self.read("outcomes.json")
            outcomes["assignment_sha256"] = packet["diagnostic"][key]["sha256"]
            write(self.root / "outcomes.json", outcomes)
            packet["diagnostic"]["outcome_ref"] = reference(self.root / "outcomes.json", "PUBLIC_DIAGNOSTIC_OUTCOMES")
        custodian = self.read("custodian.json")
        custodian["assignment_sha256"] = packet["diagnostic"]["assignment_ref"]["sha256"]
        write(self.root / "custodian.json", custodian)
        packet["diagnostic"]["custodian_ref"] = reference(self.root / "custodian.json", "PUBLIC_DIAGNOSTIC_CUSTODIAN")
        review = self.read("diagnostic-review.json")
        review.update(assignment_sha256=packet["diagnostic"]["assignment_ref"]["sha256"],
                      outcome_sha256=packet["diagnostic"]["outcome_ref"]["sha256"],
                      custodian_sha256=packet["diagnostic"]["custodian_ref"]["sha256"])
        write(self.root / "diagnostic-review.json", review)
        packet["diagnostic"]["review_ref"] = reference(self.root / "diagnostic-review.json", "PUBLIC_DIAGNOSTIC_REVIEW")
        write(self.path, packet)

    def change_asset(self, name, key, mutation):
        value = self.read(name)
        mutation(value)
        write(self.root / name, value)
        packet = self.read("packet.json")
        packet["assets"][key] = reference(self.root / name, "TRAIN_SIDE_PAYLOAD")
        dose = self.read("dose.json")
        dose["assets"] = packet["assets"]
        write(self.root / "dose.json", dose)
        packet["dose_ref"] = reference(self.root / "dose.json", "PROSPECTIVE_DOSE")
        write(self.path, packet)

    def blocked(self, reason):
        result = verify(assess_data_admission(str(self.path)))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn(reason, "\n".join(result["blockers"]))
        self.assertFalse(result["actual_dataset_admission"])
        return result

    def test_complete_caller_declarations_only_prepare_owner_review(self):
        result = verify(assess_data_admission(str(self.path)))
        self.assertEqual(result["status"], "READY_FOR_OWNER_ADMISSION_REVIEW")
        for field in ("actual_dataset_admission", "authorizes_training", "authorizes_native_run", "authorizes_confirmation",
                      "model_quality_established", "permits_model_promotion", "native_execution_independently_verified"):
            self.assertFalse(result[field])
        self.assertEqual(result["diagnostic"]["assigned_trials"], 224)
        self.assertFalse(result["diagnostic"]["production_grader_qualification"])

    def test_fixture_marked_retained_inputs_cannot_be_ready(self):
        self.path = build_data_packet(self.root / "explicit-fixture")
        self.blocked("fixture provenance")

    def test_positive_diagnostic_does_not_replace_seven_production_gates(self):
        packet = self.read("packet.json")
        packet["gates"] = {}
        write(self.path, packet)
        result = self.blocked("production_grader")
        self.assertTrue(result["diagnostic"]["positive_control_results"])
        self.assertGreaterEqual(len(result["blockers"]), 7)

    def test_mutated_retained_payload_blocks_before_metadata_claim(self):
        (self.root / "tasks.json").write_text("{}", encoding="utf-8")
        self.blocked("retained payload hash changed")

    def test_independent_solver_exact_target_mismatch_blocks(self):
        self.change_gate("independent_solver", lambda e: e["rows"][0].update(derived_target="3"))
        self.blocked("exact target verification")

    def test_scorer_requires_actual_wrong_target_control(self):
        self.change_gate("independent_scorer", lambda e: e["rows"][1].update(wrong_target=""))
        self.blocked("correct/wrong-target controls")

    def test_raw_and_rendered_overlap_are_separate_gates(self):
        self.change_gate("raw_rendered_overlap", lambda e: e.update(rendered_prompt_collisions=["heldout-parent"]))
        self.blocked("raw/rendered/parent checks")

    def test_null_target_is_not_legitimate_empty_output(self):
        tasks = self.read("tasks.json")
        tasks["rows"][1]["target"] = None
        write(self.root / "tasks.json", tasks)
        packet = self.read("packet.json")
        packet["assets"]["task_ref"] = reference(self.root / "tasks.json", "TRAIN_SIDE_PAYLOAD")
        dose = self.read("dose.json")
        dose["assets"] = packet["assets"]
        write(self.root / "dose.json", dose)
        packet["dose_ref"] = reference(self.root / "dose.json", "PROSPECTIVE_DOSE")
        write(self.path, packet)
        self.blocked("null is not empty output")

    def test_empty_output_still_requires_native_eos_supervision(self):
        self.change_gate("native_rendering_eos_loss", lambda e: e["occurrences"][1].update(supervised_positions=[]))
        self.blocked("native response/EOS supervision")

    def test_native_prompt_loss_blocks(self):
        self.change_gate("native_rendering_eos_loss", lambda e: e["occurrences"][0].update(prompt_loss_tokens=1))
        self.blocked("prompt tokens were supervised")

    def test_actual_response_mask_and_prompt_labels_are_required(self):
        self.change_asset("consumption.json", "consumption_ref", lambda e: e["occurrences"][0]["labels"].__setitem__(0, 1))
        self.blocked("prompt loss exclusion and EOS label")

    def test_actual_empty_response_cannot_omit_eos(self):
        self.change_asset("consumption.json", "consumption_ref", lambda e: e["occurrences"][1].update(response_ids=[5]))
        self.blocked("final EOS required")

    def test_repeated_consumption_cannot_be_collapsed_to_unique_tasks(self):
        self.change_asset("consumption.json", "consumption_ref", lambda e: e["occurrences"].pop())
        self.blocked("original dose/consume repeats")

    def test_rejected_source_filtering_history_cannot_be_omitted(self):
        self.change_asset("filtering.json", "filtering_ref", lambda e: e["rows"].pop())
        self.blocked("including rejected inputs")

    def test_silent_truncation_cannot_hide_behind_original_ids(self):
        self.change_asset("consumption.json", "consumption_ref", lambda e: e["occurrences"][0].update(original_prompt_ids=[1, 2, 3]))
        self.blocked("silent truncation")

    def test_fixture_marked_nested_rights_basis_cannot_pass(self):
        write(self.root / "rights-basis.json", {"evidence_kind": "mechanical_fixture"})
        self.change_gate("source_use_rights", lambda e: e.update(rights_basis_refs=[reference(self.root / "rights-basis.json", "PUBLIC_RIGHTS_EVIDENCE")]))
        self.blocked("fixture-marked rights basis")

    def test_explicit_finite_mechanical_markers_cannot_establish_readiness(self):
        for field in ("scope", "evidence_kind", "evidence_level", "classification", "status"):
            for kind in ("finite_mechanical_witness", "EXCLUDED_MECHANICAL_FIXTURE",
                         "PASS_FINITE_MECHANICAL_QUERY_DEPENDENCE_WITNESS"):
                write(self.root / "rights-basis.json", {field: kind})
                self.change_gate("source_use_rights", lambda e: e.update(
                    rights_basis_refs=[reference(self.root / "rights-basis.json", "PUBLIC_RIGHTS_EVIDENCE")]))
                with self.subTest(field=field, kind=kind):
                    self.blocked("fixture-marked rights basis")

    def test_fixture_marked_nested_production_protocol_cannot_pass(self):
        protocol = self.read("production-protocol.json")
        protocol["evidence_kind"] = "mechanical_fixture"
        write(self.root / "production-protocol.json", protocol)
        calibration = self.read("production-calibration.json")
        calibration["protocol_ref"] = reference(self.root / "production-protocol.json", "PUBLIC_PRODUCTION_PROTOCOL")
        write(self.root / "production-calibration.json", calibration)
        self.change_gate("production_grader", lambda e: e.update(calibration_ref=reference(self.root / "production-calibration.json", "PUBLIC_PRODUCTION_CALIBRATION")))
        self.blocked("retained real prospective protocol")

    def test_production_calibration_cannot_loosen_frozen_error_bound(self):
        calibration = self.read("production-calibration.json")
        calibration["registered_max_error_upper"] = 0.5
        write(self.root / "production-calibration.json", calibration)
        self.change_gate("production_grader", lambda e: e.update(calibration_ref=reference(self.root / "production-calibration.json", "PUBLIC_PRODUCTION_CALIBRATION")))
        self.blocked("frozen error bound")

    def test_pending_public_metadata_never_reads_train_or_control_payloads(self):
        pending = make_pending_packet(owner="existing-owner", packet_id="current-public-readiness",
            public_metadata_refs=[reference(self.root / "tasks.json", "PUBLIC_PENDING_METADATA")],
            diagnostic_contract_ref=reference(self.root / "diagnostic-contract.json", "PUBLIC_DIAGNOSTIC_CONTRACT"),
            proposed_dose_metadata_ref=reference(self.root / "dose.json", "PUBLIC_PENDING_METADATA"),
            source_metadata_refs=[reference(self.root / "sources.json", "PUBLIC_PENDING_METADATA")],
            tokenizer_metadata_ref=reference(self.root / "tokenizer.json", "PUBLIC_PENDING_METADATA"),
            task_metadata_ref=reference(self.root / "tasks.json", "PUBLIC_PENDING_METADATA"))
        write(self.path, pending)
        # Only the aggregate public diagnostic contract may be parsed.
        original = __import__("experiments.rineya_generalization_dynamics_v1.data_admission", fromlist=["_json"])._json
        def guarded_json(path):
            self.assertIn(path.name, {"packet.json", "diagnostic-contract.json"})
            return original(path)
        with patch("experiments.rineya_generalization_dynamics_v1.data_admission._json", side_effect=guarded_json):
            result = self.blocked("completed224 outcomes and observed exposure history")
        self.assertEqual(result["diagnostic"]["completed_trials"], 0)
        self.assertEqual(result["diagnostic"]["exposure_history"], "UNKNOWN")

    def test_pending_missing_metadata_lists_concrete_owner_prerequisites(self):
        write(self.path, make_pending_packet(owner="existing-owner", packet_id="pending", public_metadata_refs=[]))
        result = self.blocked("fixed224 diagnostic contract")
        self.assertIn("native_rendering_eos_loss", "\n".join(result["blockers"]))

    def test_native_fixture_receipt_cannot_substitute_real_observation(self):
        self.change_gate("native_rendering_eos_loss", lambda e: e.update(evidence_kind="mechanical_fixture"))
        self.blocked("real completed reviewed production evidence")

    def test_missing_diagnostic_context_and_duplicate_assignment_block(self):
        self.change_diagnostic("assignments.json", lambda e: e["assignments"][1].update(context_id=e["assignments"][0]["context_id"]))
        self.blocked("duplicate assignment/context")

    def test_all224_assignment_denominator_is_mandatory(self):
        self.change_diagnostic("assignments.json", lambda e: e["assignments"].pop())
        self.blocked("all224 frozen assignments")

    def test_missing_uncertain_slots_stay_in_assigned_denominator(self):
        self.change_diagnostic("outcomes.json", lambda e: e["outcomes"][0].update(disposition="missing"))
        result = self.blocked("remain in assigned224 denominator")
        self.assertEqual(result["diagnostic"]["reported_slots"], 224)
        self.assertEqual(result["diagnostic"]["counts"]["missing"], 1)
        self.assertEqual(result["diagnostic"]["counts"]["completed"], 223)

    def test_omitting_missing_slot_is_not_a_complete_diagnostic(self):
        self.change_diagnostic("outcomes.json", lambda e: e["outcomes"].pop())
        self.blocked("missing slots must remain explicitly")

    def test_tools_or_gold_exposure_invalidates_blind_control(self):
        self.change_diagnostic("outcomes.json", lambda e: e["outcomes"][0]["observed_history"][1].update(kind="tool"))
        self.blocked("source/gold/tool exposure")

    def test_unobserved_history_cannot_be_replaced_by_innocence_declaration(self):
        self.change_diagnostic("outcomes.json", lambda e: e["outcomes"][0].update(history_observable=False))
        self.blocked("observed access history missing")

    def test_author_cannot_review_own_native_evidence(self):
        review = self.read("native_rendering_eos_loss-review.json")
        review["reviewer"] = "toy-method-author"
        write(self.root / "native_rendering_eos_loss-review.json", review)
        self.change_gate("native_rendering_eos_loss", lambda e: None)
        self.blocked("author cannot independently accept")

    def test_protected_bank_reference_is_refused_before_read(self):
        packet = self.read("packet.json")
        packet["assets"]["task_ref"]["path"] = str(self.root / "confirmation" / "hidden.json")
        dose = self.read("dose.json")
        dose["assets"] = packet["assets"]
        write(self.root / "dose.json", dose)
        packet["dose_ref"] = reference(self.root / "dose.json", "PROSPECTIVE_DOSE")
        write(self.path, packet)
        self.blocked("Protected banks")

    def test_protected_prefix_packet_blocks_before_read(self):
        for component in ("sealed_eval", "SPENT_archive", "ANY-READ-bank", "confirmation.STARTED.json"):
            with self.subTest(component=component), patch(
                    "experiments.rineya_generalization_dynamics_v1.data_admission._json") as reader:
                result = assess_data_admission(str(self.root / component / "packet.json"))
                self.assertEqual(result["status"], "BLOCKED")
                self.assertIn("Protected banks", " ".join(result["blockers"]))
                reader.assert_not_called()

    def test_protected_prefix_nested_payload_blocks(self):
        packet = self.read("packet.json")
        packet["assets"]["task_ref"]["path"] = str(self.root / "sealed_eval" / "hidden.json")
        dose = self.read("dose.json")
        dose["assets"] = packet["assets"]
        write(self.root / "dose.json", dose)
        packet["dose_ref"] = reference(self.root / "dose.json", "PROSPECTIVE_DOSE")
        write(self.path, packet)
        self.blocked("Protected banks")

    def test_admission_assessment_imports_no_model_runtime(self):
        original_import = __import__
        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], {"torch", "transformers", "peft", "safetensors"})
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import):
            self.assertEqual(assess_data_admission(str(self.path))["status"], "READY_FOR_OWNER_ADMISSION_REVIEW")


if __name__ == "__main__":
    unittest.main()
