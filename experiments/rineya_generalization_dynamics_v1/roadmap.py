"""Offline balanced-quality evidence ledger and owner handoff preparation.

This module never dispatches a reviewer, launches a worker, opens an evaluation
bank, admits training rows or changes another experiment. Inputs are explicitly
listed public aggregate metadata and owner proposals, not model predictions.
"""
from __future__ import annotations

import re
import hashlib
import json
import math
from pathlib import Path

from .common import code_hash, digest, file_hash, read, require, signed, verify, write_new
from .protected_paths import reject_protected_path

SCHEMA = "gd001_balanced_quality_roadmap_v1"
DOMAINS = ("instruction_following", "grounded_retrieval", "arithmetic_state_tracking",
           "skill_composition", "answerability_abstention", "language_long_context_preservation")
LEVELS = {"mechanical", "native_qualification", "exploratory", "confirmed"}
HEX = re.compile(r"[0-9a-f]{64}")


def _fixture_marked(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"scope", "evidence_kind", "evidence_level", "classification", "status"} and isinstance(item, str):
                marker = item.lower()
                if marker in {"mechanical", "fixture", "mechanical_fixture", "mechanical_only", "toy_fixture", "synthetic_fixture",
                              "excluded_mechanical_fixture", "finite_mechanical_witness",
                              "pass_finite_mechanical_query_dependence_witness"} \
                        or marker.startswith(("mechanical_fixture", "toy_fixture", "synthetic_fixture", "pass_mechanical_only")):
                    return True
            if _fixture_marked(item):
                return True
    elif isinstance(value, list):
        return any(_fixture_marked(item) for item in value)
    return False


def _hash(value, label):
    require(isinstance(value, str) and HEX.fullmatch(value) is not None, f"Invalid {label} hash")


def _captured_json(payload: bytes):
    """Parse captured bytes without reopening an owner-controlled source file."""
    def reject(value):
        require(False, f"Non-finite JSON number: {value}")
    return json.loads(payload.decode("utf-8"), parse_constant=reject)


def public_metadata(ref: dict) -> tuple[dict, bytes]:
    """Read exactly one declared open aggregate, never follow references inside it."""
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "access"},
            "Public aggregate reference needs path/hash/access")
    require(ref["access"] == "public_aggregate", "Only open public aggregate metadata is permitted")
    path = Path(ref["path"]).expanduser()
    resolved = reject_protected_path(path, "Public aggregate metadata", include_instruments=True)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            "Retain an absolute regular public aggregate file")
    require(path.suffix.lower() in {".json", ".md", ".txt"} and path.stat().st_size <= 2 * 1024 * 1024,
            "Only bounded aggregate documents are supported; no banks or model payloads")
    _hash(ref["sha256"], "public aggregate")
    require(file_hash(path) == ref["sha256"], "Public aggregate changed; obtain a fresh owner review")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == ref["sha256"] and file_hash(path) == ref["sha256"],
            "Public aggregate changed during capture")
    return {"path": str(resolved), "sha256": ref["sha256"], "access": ref["access"]}, payload


def review_m1_handoff(proposal_ref: dict, identity_fixture_ref: dict) -> dict:
    """Review the preserved direct-interpreter proposal without executing it."""
    proposal_binding, proposal_bytes = public_metadata(proposal_ref)
    fixture_binding, fixture_bytes = public_metadata(identity_fixture_ref)
    proposal, fixture = _captured_json(proposal_bytes), _captured_json(fixture_bytes)
    require(proposal["schema"] == "mlp_ship.prospective_retry_proposal/1"
            and proposal["status"] == "PREPARED_OFFLINE_NOT_AUTHORIZED", "Unexpected M1 proposal")
    require(type(proposal["native_attempts_authorized_now"]) is int
            and proposal["native_attempts_authorized_now"] == 0,
            "This offline handoff does not consume or import native authorization")
    require(proposal["original_authorized_attempt_consumed"] is True
            and proposal["existing_runner_unchanged"] is True
            and proposal["source_pins_tolerances_guards_unchanged"] is True,
            "Preserve consumed attempts, runner, pins, tolerances and guards")
    require(proposal["cal_check_authorized"] is False and proposal["deployment_authorized"] is False
            and proposal["task_bank_reads_allowed"] is False
            and all(value is False for value in proposal["permissions"].values()),
            "M1 review cannot authorize training, task banks or deployment")
    command = proposal["command"]
    require(isinstance(command, list) and len(command) >= 8 and all(isinstance(v, str) for v in command)
            and command[1:5] == ["-B", "-m", "experiments.matched_mlp_shipping_v1.run", "qualify"],
            "Review the exact existing qualification command; no new launcher")
    source_snapshot = proposal["source_snapshot"]
    require(isinstance(source_snapshot, dict) and {"branch", "head", "model_commit", "source_sha256",
            "adapter_manifest_sha256", "adapter_tensor_sha256", "model_files"} <= set(source_snapshot),
            "M1 source/checkpoint/sidecar identity is incomplete")
    for key in ("source_sha256", "adapter_manifest_sha256", "adapter_tensor_sha256"):
        _hash(source_snapshot[key], key)
    for key in ("head", "model_commit"):
        require(isinstance(source_snapshot[key], str) and re.fullmatch(r"[0-9a-f]{40}", source_snapshot[key]),
                "Bind the exact owner source and model revision")
    for key in ("model.safetensors", "tokenizer.json", "config.json"):
        _hash(source_snapshot["model_files"][key], key)
    runtime = proposal["base_runtime_files"]
    executable = reject_protected_path(runtime["python.exe"]["path"], "M1 interpreter custody")
    require(reject_protected_path(command[0], "M1 interpreter command") == executable
            and "--out" in command and "--device" in command,
            "Direct launch must use the custody-bound base interpreter")
    require(digest(runtime) == digest(fixture["base_runtime_files"])
            and file_hash(Path(identity_fixture_ref["path"])) == proposal["launch_identity_fixture_sha256"],
            "M1 proposal/launcher fixture custody differs")
    for item in runtime.values():
        path = Path(item["path"])
        reject_protected_path(path, "M1 runtime custody")
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"],
                "M1 direct-interpreter bytes changed; fresh owner review required")
    require(fixture["status"] == "PASS_OWNED_PID_FIXTURE_ONLY" and fixture["child_exited"] is True
            and type(fixture["child_returncode"]) is int and fixture["child_returncode"] == 0,
            "No completed direct-parent identity fixture")
    observation, parent, child = fixture["observation"], fixture["parent"], fixture["child"]
    require(observation["guard_passed"] is True
            and type(observation["ppid"]) is int and type(parent["pid"]) is int
            and type(observation["pid"]) is int and type(child["pid"]) is int
            and observation["ppid"] == parent["pid"] and observation["pid"] == child["pid"]
            and observation["supervisor"] == str(parent["pid"])
            and reject_protected_path(observation["executable"], "M1 observed interpreter") == executable,
            "M1 fixture does not establish the direct-parent ownership guard")
    require(all(type(fixture[key]) is int and fixture[key] == 0 for key in
            ("model_loads", "native_forwards", "backward_calls", "optimizer_steps", "task_rows_read"))
            and fixture["new_native_execution_authorized"] is False,
            "A launcher fixture is not native model qualification")
    require(isinstance(proposal["requirements_before_execution"], list)
            and bool(proposal["requirements_before_execution"]), "Owner execution prerequisites required")
    return signed({"status": "OWNER_RETRY_HANDOFF_REVIEWED_NOT_AUTHORIZED",
        "proposal": proposal_binding, "launcher_fixture": fixture_binding,
        "runtime_custody_verified": True, "existing_owner_command": command,
        "existing_owner_environment": proposal["environment"],
        "original_limits": proposal["limits"], "source_snapshot": proposal["source_snapshot"],
        "requirements_before_execution": proposal["requirements_before_execution"],
        "native_model_qualification": False, "native_attempts_authorized": 0,
        "cal_check_authorized": False, "training_authorized": False, "deployment_authorized": False})


def review_supervised_handoff(draft_ref: dict, qualification_ref: dict) -> dict:
    """Bind the existing 27-update draft; neither change its dose nor launch it."""
    draft_binding, draft_bytes = public_metadata(draft_ref)
    qualifier_binding, qualifier_bytes = public_metadata(qualification_ref)
    draft, qualifier = _captured_json(draft_bytes), _captured_json(qualifier_bytes)
    require(draft["status"] == "CPU_SCHEDULE_PREPARATION_NOT_EXECUTION_PERMIT"
            and draft["native_launch_authorized"] is False and draft["confirmation_authorized"] is False
            and all(value is False for value in draft["permissions"].values()),
            "Preserve the owner's source/schedule proposal and closed execution gates")
    require(qualifier["native_launch_authorized"] is False and qualifier["production_training_authorized"] is False
            and qualifier["protected_bank_access"] is False and qualifier["quality_gain_established"] is False,
            "Disposable source qualification cannot become research training or quality evidence")
    source = draft["student"]
    require(type(source["base_parameters"]) is int and 500000000 <= source["base_parameters"] <= 700000000,
            "First supervised source must remain exact 0.6B class")
    for field in ("weight_sha256", "tokenizer_sha256", "config_sha256", "tokenizer_config_sha256"):
        _hash(source[field], field)
    dose, adaptation, optimizer = draft["dose"], draft["adaptation"], draft["optimizer"]
    require(type(dose["whole_sequences_per_arm"]) is int and dose["whole_sequences_per_arm"] == 540
            and type(dose["full_updates_per_adapted_arm"]) is int and dose["full_updates_per_adapted_arm"] == 27
            and type(dose["sequences_per_update"]) is int and dose["sequences_per_update"] == 20,
            "Do not silently change the frozen 540-sequence/27-update dose")
    require(all(dose[key] is False for key in
                ("RS_training", "dropping", "packing", "padding", "replay_added", "truncation", "warmstart")),
            "Keep atomic-only training, whole-sequence consumption and independent preservation guards")
    require(adaptation["method"] == "qv_lora" and type(adaptation["rank"]) is int and adaptation["rank"] == 8
            and adaptation["trainable_parameters"] == 1146880 and adaptation["trainable_tensors"] == 112
            and adaptation["loss"] == "completion_plus_EOS_mean_per_sequence_then_mean_over_accumulation",
            "Original target/EOS and trainable-scope registration differs")
    require(optimizer["name"] == "AdamW" and optimizer["optimizer_state_dtype"] == "float32",
            "Owner optimizer schema must be frozen explicitly")
    rates = optimizer["learning_rates_by_1based_update"]
    require(isinstance(rates, list) and len(rates) == 27
            and all(type(value) in {int, float} and math.isfinite(value) and value >= 0 for value in rates)
            and rates[-1] == 0 and all(value > 0 for value in rates[:-1]),
            "Retain every exact LR, including the consumed zero-LR final update")
    require(type(draft["seed"]) is int and draft["seed"] >= 0, "Preserve the actual owner draft seed")
    return signed({"status": "SUPERVISED_ACQUISITION_DRAFT_BOUND_OWNER_GATES_PENDING",
        "draft": draft_binding, "qualification_proposal": qualifier_binding, "source": source,
        "owner_draft_seed": draft["seed"], "dose": dose, "adaptation": adaptation, "optimizer": optimizer,
        "original_runtime_candidate": qualifier["runtime_candidate"],
        "original_qualification_resources": qualifier["resources"], "source_and_decision_pins": draft["source_and_decision_pins"],
        "pending_owner_gates": draft["pending_gates"], "qualification_pending_gates": qualifier["remaining_prerequisites"],
        "checkpoint_observer_proposal": {"saved_steps": [9, 18, 27], "first_gd_dev_steps": [0, 9, 27],
            "absolute_stable_paths": None, "post_save_hook": "integration.py::record_snapshot AFTER producer save",
            "retain_all_original_payloads": True, "owner_prospective_approval_required": True,
            "standard_export_and_native_parity_qualified": False},
        "qualification_states_can_warmstart_study": False, "training_authorized": False,
        "native_execution_authorized": False, "confirmation_authorized": False, "quality_gain_established": False})


def _observation(item: dict, owners: dict) -> dict:
    require(item["owner_lane"] in owners, "Observation needs its existing owner")
    require(item["evidence_level"] in LEVELS and item["evidence_level"] != "confirmed",
            "Aggregate imports cannot mint confirmed evidence; validate an owner quality_card separately")
    require(item["effect_type"] in {"learned", "serving", "architecture", "qualification", "data_mechanics"},
            "Distinguish learned capability, serving effects and mechanics")
    require(isinstance(item["limitations"], list) and bool(item["limitations"]), "Keep evidence limitations")
    require(isinstance(item["facts"], dict), "Preserve original aggregate facts, not an invented overall score")
    require(item["evidence_level"] == "mechanical" or not _fixture_marked(item["facts"]),
            "Explicit fixture facts cannot become exploratory/native evidence")
    require(isinstance(item["domains"], list) and set(item["domains"]) <= set(DOMAINS), "Unknown quality domain")
    require(item["quality_card_status"] == "BLOCKED_MISSING_FULL_IDENTITY_OR_QUALIFICATION",
            "An incomplete aggregate cannot become a fully qualified quality_card")
    paired = []
    for endpoint in item.get("paired_counts", []):
        require(endpoint["domain"] in set(DOMAINS) | {"cross_domain_aggregate"}
                and isinstance(endpoint["endpoint"], str), "Unknown paired endpoint")
        keys = ("denominator", "source_successes", "candidate_successes", "gained", "lost", "retained")
        require(all(type(endpoint[key]) is int and endpoint[key] >= 0 for key in keys)
                and endpoint["denominator"] > 0, "Invalid paired aggregate count")
        n, source, candidate = (endpoint[key] for key in keys[:3])
        require(max(source, candidate) <= n and endpoint["retained"] + endpoint["lost"] == source
                and endpoint["retained"] + endpoint["gained"] == candidate
                and endpoint["gained"] <= n - source, "Inconsistent paired gain/loss/retention accounting")
        paired.append({**endpoint, "source_accuracy": source / n, "candidate_accuracy": candidate / n,
                       "net_gain": endpoint["gained"] - endpoint["lost"],
                       "retained_fraction": endpoint["retained"] / source if source else None,
                       "source_membership_not_verified_by_aggregate_counts": True})
    return {**item, "paired_counts": paired, "balanced_quality_established": False, "permits_model_promotion": False}


def prepare_roadmap(spec_path: str, output_directory: str) -> dict:
    """Snapshot reviewed public facts and prepare immutable owner packets only."""
    spec_file = reject_protected_path(spec_path, "Roadmap specification", include_instruments=True)
    spec_bytes = spec_file.read_bytes()
    spec = verify(_captured_json(spec_bytes))
    spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
    require(spec["schema"] == SCHEMA and spec["scope"] == "OFFLINE_PUBLIC_AGGREGATE_AND_HANDOFF_PREPARATION",
            "Only offline aggregate/handoff preparation is implemented")
    require(spec["code_sha256"] == code_hash(), "Roadmap specification needs current code review")
    require(spec["training_authorized"] is False and spec["native_execution_authorized"] is False
            and spec["confirmation_authorized"] is False and spec["external_messages_authorized"] is False,
            "The roadmap does not import execution or messaging permission")
    owners = {owner["lane"]: owner for owner in spec["owners"]}
    require(len(owners) == len(spec["owners"]) and bool(owners), "Duplicate or missing owner lane")
    for owner in owners.values():
        require(all(isinstance(owner[key], str) and owner[key].strip()
                    for key in ("chat_id", "title", "workspace")), "Exact existing owner identity required")
    evidence, captured = {}, {}
    for item in spec["evidence"]:
        require(isinstance(item["id"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item["id"])
                and item["id"] not in evidence, "Unsafe/duplicate evidence ID")
        binding, payload = public_metadata(item["reference"])
        evidence[item["id"]] = {**item, "reference": binding}
        captured[item["id"]] = payload
    observations = []
    for item in spec["observations"]:
        require(item["evidence_ids"] and set(item["evidence_ids"]) <= set(evidence), "Unbound observation")
        if item["evidence_level"] != "mechanical":
            for identifier in item["evidence_ids"]:
                if Path(evidence[identifier]["reference"]["path"]).suffix.lower() == ".json":
                    require(not _fixture_marked(_captured_json(captured[identifier])),
                            "Bound fixture receipts cannot become exploratory/native evidence")
        observations.append(_observation(item, owners))
    lanes = {}
    for lane in spec["lanes"]:
        require(lane["id"] in owners and lane["id"] not in lanes, "Lane must reuse one existing owner")
        require(set(lane["evidence_ids"]) <= set(evidence), "Unbound lane proposal")
        require(isinstance(lane["prerequisites"], list) and bool(lane["prerequisites"]), "Precise gates required")
        require(lane["execution_authorized"] is False and lane["scope"] == "OWNER_HANDOFF_ONLY",
                "No competing owner operation is implemented")
        require(lane["status"] in {"BLOCKED_PREREQUISITES", "HOLD", "DEFERRED"}, "No native-ready claim from a roadmap")
        lanes[lane["id"]] = lane
    require(set(lanes) == set(owners), "Every owner lane needs an exact handoff")
    for lane in lanes.values():
        require(set(lane["depends_on"]) <= set(lanes) and lane["id"] not in lane["depends_on"],
                "Invalid roadmap dependency")
    def visit(identifier, stack):
        require(identifier not in stack, "Cyclic readiness dependencies")
        for dependency in lanes[identifier]["depends_on"]:
            visit(dependency, stack | {identifier})
    for identifier in lanes:
        visit(identifier, set())
    m1 = None
    if "m1_review" in spec:
        m1 = review_m1_handoff(evidence[spec["m1_review"]["proposal_id"]]["reference"],
                              evidence[spec["m1_review"]["fixture_id"]]["reference"])
    supervised = None
    if "supervised_review" in spec:
        supervised = review_supervised_handoff(evidence[spec["supervised_review"]["draft_id"]]["reference"],
            evidence[spec["supervised_review"]["qualification_id"]]["reference"])
    workspace = Path(__file__).resolve().parents[2]
    output = reject_protected_path(output_directory, "Roadmap output", include_instruments=True)
    require(output.is_relative_to(workspace.parent / "gd001-records")
            and all(part.lower() != ".codex" for part in output.parts),
            "Roadmap outputs must remain in external BusinessProject/gd001-records")
    require(not output.exists(), "Choose a fresh packet directory; preserve earlier outcomes")
    validated_cards = []
    if spec.get("quality_card_evidence_ids"):
        from .quality_card import validate_quality_card
        for identifier in spec["quality_card_evidence_ids"]:
            require(identifier in evidence, "Unbound quality-card aggregate")
            validated_cards.append(validate_quality_card(_captured_json(captured[identifier])))
    planned_snapshots = [{"evidence_id": identifier,
        "path": str(output / "public-metadata-snapshots" / (identifier + Path(item["reference"]["path"]).suffix)),
        "sha256": item["reference"]["sha256"]} for identifier, item in evidence.items()]
    snapshot_receipt = signed({"snapshots": planned_snapshots, "scope": "public_aggregate_metadata_only"})
    handoffs = signed({"owners": list(owners.values()), "lanes": list(lanes.values()),
                      "messages_sent": 0, "execution_authorized": False})
    scorecard = signed({"schema": "gd001_public_quality_scorecard_v1", "domains": list(DOMAINS),
        "observations": observations, "unmeasured_cells": "UNMEASURED, not zero",
        "domain_coverage": {domain: {"status": "UNMEASURED", "score": None,
            "reason": "No full native quality-card identity/endpoint custody inferred from aggregate imports"} for domain in DOMAINS},
        "overall_winner": None, "cross_source_ranking_permitted": False,
        "fully_qualified_quality_cards": [], "incomplete_aggregate_observations_only": True,
        "validated_quality_cards": validated_cards,
        "balanced_quality_established": False, "permits_model_promotion": False})
    packet = signed({"schema": SCHEMA, "status": "OFFLINE_ROADMAP_IMPLEMENTED_NATIVE_AND_ADMISSION_GATED",
        "code_sha256": code_hash(), "spec": {"path": str(spec_file), "sha256": spec_sha256},
        "owners": list(owners.values()), "evidence": list(evidence.values()),
        "quality_scorecard_sha256": scorecard["sha256"], "lanes": list(lanes.values()),
        "owner_handoffs_sha256": handoffs["sha256"], "snapshot_receipts_sha256": snapshot_receipt["sha256"],
        "m1_handoff_review": m1, "source_parameters_range": [500000000, 700000000],
        "supervised_acquisition_handoff": supervised,
        "one_owned_native_worker_at_a_time": True, "gd_dev_parity_limits": {"wall_seconds": 1800, "forward_calls": 16000},
        "budgets_are_study_specific": True, "native_model_loads": 0, "native_forward_calls": 0,
        "training_runs": 0, "confirmation_accesses": 0, "external_messages_sent": 0,
        "data_rows_admitted": 0, "permits_model_promotion": False})
    # Recheck original aggregate bytes before writing a new exclusive packet.
    require(file_hash(spec_file) == spec_sha256, "Roadmap specification changed during capture")
    for item in evidence.values():
        require(file_hash(Path(item["reference"]["path"])) == item["reference"]["sha256"],
                "Owner evidence changed; keep historical snapshot and prepare a new review")
    output.mkdir(parents=True)
    snapshots = []
    for identifier, payload in captured.items():
        destination = output / "public-metadata-snapshots" / (identifier + Path(evidence[identifier]["reference"]["path"]).suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
        snapshots.append({"evidence_id": identifier, "path": str(destination), "sha256": file_hash(destination)})
    write_new(output / "quality-scorecard.json", scorecard)
    write_new(output / "owner-handoffs.json", handoffs)
    write_new(output / "ROADMAP_PACKET.json", packet)
    if supervised is not None:
        write_new(output / "SUPERVISED_ACQUISITION_PROPOSAL.json", supervised)
    if m1 is not None:
        write_new(output / "M1_OWNER_HANDOFF.json", m1)
    require(digest(snapshots) == digest(planned_snapshots), "Captured metadata differs from registered snapshot inventory")
    write_new(output / "snapshot-receipts.json", snapshot_receipt)
    report = ["# Rineya balanced-quality implementation packet", "", "Offline preparation only. No rows admitted or native execution authorized.", "",
              "| Lane | Existing owner chat | State | Next action |", "|---|---|---|---|"]
    for identifier, lane in lanes.items():
        report.append(f"| {identifier} | {owners[identifier]['title']} | {lane['status']} | {lane['next_action']} |")
    report += ["", "## Required owner gates", ""]
    for identifier, lane in lanes.items():
        report += [f"### {identifier}", "", *["- " + gate for gate in lane["prerequisites"]], ""]
    report += ["## Evidence limits", "", "Incomplete aggregate observations do not become qualified quality cards.",
               "Missing domains remain unmeasured. Preserve each study's original thresholds, source and runtime.",
               "Complete reviewer diagnostics, qualification, acquisition and retention require owner evidence."]
    with (output / "OWNER_HANDOFFS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    return packet


def verify_roadmap_packet(output_directory: str) -> dict:
    """Verify retained packet bytes; changed upstream owners do not rewrite snapshots."""
    output = reject_protected_path(output_directory, "Roadmap packet", include_instruments=True)
    packet = verify(read(output / "ROADMAP_PACKET.json"))
    require(packet["code_sha256"] == code_hash(), "Roadmap code changed; fresh review required")
    scorecard = verify(read(output / "quality-scorecard.json"))
    require(scorecard["sha256"] == packet["quality_scorecard_sha256"], "Scorecard changed")
    handoffs = verify(read(output / "owner-handoffs.json"))
    require(handoffs["sha256"] == packet["owner_handoffs_sha256"], "Owner handoffs changed")
    snapshots = verify(read(output / "snapshot-receipts.json"))
    require(snapshots["sha256"] == packet["snapshot_receipts_sha256"], "Snapshot inventory changed")
    for key, name in (("m1_handoff_review", "M1_OWNER_HANDOFF.json"),
                      ("supervised_acquisition_handoff", "SUPERVISED_ACQUISITION_PROPOSAL.json")):
        if packet[key] is not None:
            require(verify(read(output / name))["sha256"] == packet[key]["sha256"],
                    "Supplemental owner proposal changed")
    for item in snapshots["snapshots"]:
        path = reject_protected_path(item["path"], "Retained public snapshot", include_instruments=True)
        require(file_hash(path) == item["sha256"], "Retained public snapshot changed")
    return signed({"status": "OFFLINE_PACKET_HASHES_VERIFIED", "roadmap_packet_sha256": packet["sha256"],
                   "execution_authorized": False, "upstream_evidence_recheck_required_before_owner_execution": True})
