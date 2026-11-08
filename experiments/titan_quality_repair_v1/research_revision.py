"""Additive research continuation; prepare reviewed qualification source, no permits."""
from __future__ import annotations
import json
from pathlib import Path

from . import custody, registry, schema
from .native_proposal import proposal as original_proposal
from .native_qualification import proposal_payload_digest, validate_proposal
from .opposite_analysis import analyze_opposite_directions


BASE_PACKET = custody.ARTIFACT_ROOT / "20261005-offline/packet-002"
RESEARCH_ROOT = custody.ARTIFACT_ROOT / "20261005-research-v2"


def build_qualification_proposal(packet_dir=BASE_PACKET):
    packet_dir = custody.external(packet_dir)
    validation_bytes = (packet_dir / "CPU_VALIDATION.json").read_bytes()
    validation = json.loads(validation_bytes)
    if (validation.get("offline_validation_passed") is not True
            or validation.get("registry_sha256") != registry.build_registry()["registry_sha256"]
            or validation.get("current_model_loads") != 0 or validation.get("current_training_updates") != 0):
        raise ValueError("STOP_RESEARCH_BASE_PACKET_OR_REGISTRY")
    # V1 packets stay historical at their exact source hashes. Data and recipe
    # identities are reused, while this new source revision is bound separately.
    custody.read_pinned(validation["development"]["path"], validation["development"]["sha256"])
    current = custody.source_manifest()
    revised = dict(validation)
    revised["source_manifest"] = current
    body = original_proposal(revised)
    body.pop("proposal_sha256", None)
    source = custody.source_identity()
    original_identity_path = packet_dir / "SOURCE_IDENTITY.json"
    original_identity_bytes = original_identity_path.read_bytes()
    original_identity = json.loads(original_identity_bytes)
    if source["sidecar_sha256"] != original_identity["sidecar_sha256"]:
        raise ValueError("STOP_RESEARCH_CORE_SIDECAR_DRIFT")
    body.update(proposal_version=2, stage="native_qualification",
        status="SOURCE_PREPARED_REQUIRES_INDEPENDENT_REVIEW_AND_OPERATOR_SLOT",
        native_worker_implementation_in_this_milestone=True,
        source_manifest_sha256=schema.digest(current), runtime_packages=source["runtime_packages"],
        runtime_site_packages=["C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages"],
        python_startup_policy="ISOLATED_-I_-B_-S_EXPLICIT_PINNED_SITE_PATHS_NO_PYC_PACKET_IMPORTS",
        source_sidecar_sha256=original_identity["sidecar_sha256"],
        source_identity_descriptor={"path": str(original_identity_path), "sha256": custody.sha_bytes(original_identity_bytes)},
        metadata_descriptor={"path": str(original_identity_path), "sha256": custody.sha_bytes(original_identity_bytes)},
        original_data_validation_sha256=custody.sha_bytes(validation_bytes),
        scientific_recipe_choices_unchanged=True,
        human_authorization_required_separately=True, native_execution_authorized=False,
        reviewed_source_scope="GUARDED_DISPOSABLE_QUALIFICATION_NOT_TRAINING_SWEEP_OR_FINAL",
        implementation_options={"adamw_foreach": False, "adamw_fused": False,
            "checkpoint_use_reentrant": True, "loss_normalization": "per_microbatch_positive_target_token_mean_divided_by_eight",
            "inference_padding": "unpadded_generation_and_exact_1984_slot_left_padding_probes"},
        environment={"CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "cache_policy": "OWN_ATTEMPT_DIRECTORY_ONLY"},
        unresolved=["No issued operator grant or named hardware slot",
            "Native model memory/optimizer/HF serialization/reload determinism have not been measured",
            "Source-only decode timing cannot prove all LoRA DEV and dense final throughput fits 24 hours",
            "No model quality gain has been observed; fixed 16-trial program remains unrun"])
    body["attempt"].update(declared_attempt_cost_seconds=720,
        padding_layout="left_singleton_vs_unpadded", padding_input_slots=1984,
        optimizer_state_reuse=False, deliverable_candidate=False,
        source_checkpoint_loads_max=1, dense_reload_loads_max=1,
        total_model_constructions_max=2, concurrent_model_instances_max=1)
    body["proposal_payload_sha256"] = proposal_payload_digest(body)
    validate_proposal(body)
    return body


def prepare_research_spec(output_dir):
    root = custody.external(output_dir)
    body = build_qualification_proposal()
    custody.write_json(root / "NATIVE_QUALIFICATION_V2.json", body)
    custody.write_json(root / "OPPOSITE_DIRECTIONS.json", analyze_opposite_directions())
    from .operating_characteristics import audit
    design = audit()
    custody.write_json(root / "OPERATING_CHARACTERISTICS.json", design)
    report = {"experiment": schema.EXPERIMENT, "status": "RESEARCH_IN_PROGRESS_NO_VERIFIED_MODEL_GAIN",
        "goal_completed": False, "registry_sha256": registry.build_registry()["registry_sha256"],
        "qualification_payload_sha256": body["proposal_payload_sha256"],
        "source_manifest_sha256": body["source_manifest_sha256"],
        "current_model_loads": 0, "current_model_forwards": 0, "current_training_updates": 0,
        "native_execution_authorized": False, "operator_permit_file_created": False,
        "opposite_model_direction_status": "UNMEASURED",
        "new_sources": ["Guarded qualification body and own-Job guardian", "Opposite-direction factorial DEV diagnostics", "Prospective exact decision-rule audit"],
        "next_step": "Independent source/CPU review, then an explicit exclusive-slot grant for the bounded qualification only",
        "source_manifest": custody.source_manifest()}
    custody.write_json(root / "RESEARCH_STATUS.json", report)
    print(json.dumps({"status": report["status"], "native_model_gain": False,
                      "qualification_payload_sha256": body["proposal_payload_sha256"]}, indent=2))
    return report
