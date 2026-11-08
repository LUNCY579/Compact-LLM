"""Machine-enforced coordination contract for the TGSA v5.1 programme.

V5.1 deliberately does not collapse the sequential Phase 2A predictor and the
independent-offset Phase 2B decoder into one ambiguous intervention.  It freezes
how their evidence may be produced, promoted, and interpreted while preserving
both implementations as separately identified scientific stages.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROGRAMME_SCHEMA_VERSION = "tgsa.v5-1-programme.v1"
VERIFICATION_SCHEMA_VERSION = "tgsa.v5-1-programme-verification.v1"
PROGRAMME_ID = "tgsa-v5-1-unified-programme-2026-09-02"
COORDINATION_BRANCH = "research/tgsa-v5-1-unified-programme"

V4_RESULT_COMMIT = "8b5737e260b5478427b8bb033cffa394a5f3fc5d"
V4_RESULT_ARCHIVE_TAG = "archive/tgsa-phase1-v4-regime-rejected"
V5_EXECUTION_COMMIT = "de3bd632b491df294f9b8736079bdcbbdd64e4d3"
V5_MECHANISM_COMMIT = "b09149002a7892a87a461af0b3c852ed14842905"
PHASE2B_COMMIT = "49aa961265c7b248be5c81244cc5d80d49db13c7"
PRETRAINED_RETROFIT_COMMIT = "ef5194b36583f31bd59508cfe04aec1f5dcfdaec"
SOURCE_V5_PLAN_PATH = (
    "Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/"
    "configs/phase1_v5_plan.json"
)
SOURCE_V5_PLAN_BLOB = "0538faf4d2ad142dbce9c408910f9dcbcf1fd064"

REQUIRED_STAGE_ORDER = (
    "v5_apparatus",
    "mtp_free_mechanism",
    "phase2a_trajectory_mtp",
    "phase2b_system_factorial",
    "rineya_checkpoint_qualification",
    "wall_clock_benchmark",
)
APPARATUS_ARMS = ("cascade", "history_control", "learned_delta")
MECHANISM_CONDITIONS = (
    "cascade",
    "history_control",
    "learned_delta",
    "replay_control",
    "deranged_control",
)
PHASE2A_MODES = ("dormant", "backbone", "trajectory")
PHASE2A_GRADIENT_ORDER = ("frozen_probe", "stop_gradient", "joint")
PHASE2B_CELLS = ("parallel_ar", "parallel_mtp", "tgsa_ar", "tgsa_mtp")
PHASE2A_SEMANTICS = "sequential_teacher_forced_recursive_v1"
PHASE2B_SEMANTICS = "parallel_independent_offsets_v1"


class V51ContractError(ValueError):
    """Raised when the v5.1 scientific programme contract is weakened or stale."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V51ContractError(f"expected a JSON object at {path}")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise V51ContractError(f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise V51ContractError(f"{field} must be an array")
    return list(value)


def validate_contract(contract: Mapping[str, object]) -> dict[str, object]:
    """Validate every evidence boundary that makes v5.1 scientifically distinct."""

    errors: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool, message: str) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(message)

    check(
        "schema_version",
        contract.get("schema_version") == PROGRAMME_SCHEMA_VERSION,
        f"schema_version must be {PROGRAMME_SCHEMA_VERSION!r}",
    )
    check(
        "programme_id",
        contract.get("programme_id") == PROGRAMME_ID,
        f"programme_id must be {PROGRAMME_ID!r}",
    )
    check(
        "coordination_branch",
        contract.get("coordination_branch") == COORDINATION_BRANCH,
        f"coordination_branch must be {COORDINATION_BRANCH!r}",
    )
    check(
        "status",
        contract.get("status") == "prospectively_frozen_coordination_contract",
        "v5.1 must remain a prospectively frozen coordination contract",
    )

    registration = _mapping(contract.get("registration_lock"), "registration_lock")
    source_plan = _mapping(
        registration.get("source_v5_plan"), "registration_lock.source_v5_plan"
    )
    attestation = _mapping(
        registration.get("operator_attestation"),
        "registration_lock.operator_attestation",
    )
    check(
        "prospective_registration",
        registration.get("status") == "frozen_before_nondevelopment_probe",
        "nondevelopment v5 execution must be prospectively frozen before its probe",
    )
    check(
        "operator_attestation_required",
        attestation.get("required_for_nondevelopment_execution") is True,
        "nondevelopment execution must require a no-prior-outcome-inspection attestation",
    )
    check(
        "source_plan_path",
        source_plan.get("path") == SOURCE_V5_PLAN_PATH,
        "the v5.1 lock points to the wrong source plan",
    )
    check(
        "source_plan_blob",
        source_plan.get("git_blob_sha1") == SOURCE_V5_PLAN_BLOB,
        "the v5.1 lock does not pin the reviewed v5 plan bytes",
    )
    check(
        "source_plan_status_disclosed",
        source_plan.get("declared_registration_status")
        == "implementation_complete_not_registered",
        "the inherited v5 plan's unregistered status must remain disclosed",
    )

    lineage = _mapping(contract.get("lineage"), "lineage")
    expected_lineage = {
        "v4_result_boundary": V4_RESULT_COMMIT,
        "v5_execution_source": V5_EXECUTION_COMMIT,
        "v5_mechanism_source": V5_MECHANISM_COMMIT,
        "phase2a_source": V5_EXECUTION_COMMIT,
        "phase2b_source": PHASE2B_COMMIT,
        "pretrained_retrofit_source": PRETRAINED_RETROFIT_COMMIT,
    }
    for name, expected_commit in expected_lineage.items():
        entry = _mapping(lineage.get(name), f"lineage.{name}")
        check(
            f"lineage_{name}",
            entry.get("commit") == expected_commit,
            f"lineage.{name}.commit must remain pinned to {expected_commit}",
        )
    v4_boundary = _mapping(lineage.get("v4_result_boundary"), "lineage.v4_result_boundary")
    check(
        "v4_archive_tag",
        v4_boundary.get("archive_tag") == V4_RESULT_ARCHIVE_TAG,
        f"the retired v4 source must remain discoverable through {V4_RESULT_ARCHIVE_TAG!r}",
    )
    mechanism_source = _mapping(
        lineage.get("v5_mechanism_source"), "lineage.v5_mechanism_source"
    )
    check(
        "v5_contains_execution",
        mechanism_source.get("contains_commit") == V5_EXECUTION_COMMIT,
        "the v5 mechanism source must disclose that it contains the execution source",
    )

    evidence = _mapping(contract.get("global_evidence_rules"), "global_evidence_rules")
    for field in (
        "semantic_relabeling_allowed",
        "development_evidence_can_certify_confirmatory",
    ):
        check(
            f"{field}_false",
            evidence.get(field) is False,
            f"global_evidence_rules.{field} must be false",
        )
    check(
        "cross_stage_pooling_forbidden",
        evidence.get("cross_stage_pooling_allowed") is False,
        "Phase 1, Phase 2A, and Phase 2B evidence may not be pooled",
    )
    for field in (
        "stage_specific_checkpoints_required",
        "stage_specific_evidence_roots_required",
        "scope_hash_match_required",
        "environment_lock_required",
        "sealed_evaluation_is_single_use",
        "wall_clock_is_never_inferred_from_serial_rounds",
    ):
        check(
            field,
            evidence.get(field) is True,
            f"global_evidence_rules.{field} must remain true",
        )

    stage_order = tuple(str(value) for value in _sequence(contract.get("stage_order"), "stage_order"))
    check(
        "stage_order",
        stage_order == REQUIRED_STAGE_ORDER,
        f"stage_order must be exactly {list(REQUIRED_STAGE_ORDER)}",
    )
    stages = _mapping(contract.get("stages"), "stages")
    check(
        "stage_set",
        set(stages) == set(REQUIRED_STAGE_ORDER),
        "stages must contain exactly the registered v5.1 stages",
    )

    apparatus = _mapping(stages.get("v5_apparatus"), "stages.v5_apparatus")
    check(
        "apparatus_has_no_mtp",
        apparatus.get("mtp_semantics") == "none",
        "v5 apparatus qualification must remain MTP-free",
    )
    check(
        "apparatus_arms",
        tuple(apparatus.get("required_arms", ())) == APPARATUS_ARMS,
        f"v5 apparatus arms must be exactly {list(APPARATUS_ARMS)}",
    )
    check(
        "apparatus_capacity_ladder",
        tuple(apparatus.get("capacity_ladder", ())) == ("C1", "C2", "C3"),
        "the registered depth-before-width capacity ladder changed",
    )
    check(
        "apparatus_exposure_grid",
        tuple(apparatus.get("exposure_grid", ())) == ("E1", "E2", "E3", "E4"),
        "the registered diversity-by-presentation exposure grid changed",
    )
    probe = _mapping(apparatus.get("probe"), "stages.v5_apparatus.probe")
    check(
        "one_hop_probe",
        probe.get("family") == "pointer_chase"
        and int(probe.get("hops", 0)) == 1
        and probe.get("stop_at_first_passing_capacity") is True,
        "capacity selection must use the registered one-hop pointer probe and stop at first pass",
    )
    roles = _mapping(apparatus.get("family_roles"), "stages.v5_apparatus.family_roles")
    check(
        "family_roles",
        roles.get("primary") == ["pointer_chase"]
        and roles.get("guardrail") == ["direct_lookup"]
        and roles.get("canary") == ["state_update"],
        "primary, guardrail, and canary family roles changed",
    )
    check(
        "apparatus_claim_boundary",
        apparatus.get("mechanism_interpretation_allowed") is False
        and apparatus.get("rineya_retrofit_authorized") is False,
        "apparatus qualification may not authorize mechanism or Rineya claims",
    )

    mechanism = _mapping(stages.get("mtp_free_mechanism"), "stages.mtp_free_mechanism")
    check(
        "mechanism_is_mtp_free",
        mechanism.get("mtp_semantics") == "none",
        "the primary mechanism study must remain MTP-free",
    )
    check(
        "mechanism_controls",
        tuple(mechanism.get("required_conditions", ())) == MECHANISM_CONDITIONS,
        f"mechanism controls must be exactly {list(MECHANISM_CONDITIONS)}",
    )
    check(
        "mechanism_requires_apparatus",
        mechanism.get("requires_v5_apparatus_go") is True,
        "the mechanism study may not bypass apparatus qualification",
    )

    phase2a = _mapping(stages.get("phase2a_trajectory_mtp"), "stages.phase2a_trajectory_mtp")
    phase2b = _mapping(stages.get("phase2b_system_factorial"), "stages.phase2b_system_factorial")
    check(
        "phase2a_semantics",
        phase2a.get("study_kind") == "mechanism_bridge"
        and phase2a.get("mtp_semantics") == PHASE2A_SEMANTICS,
        "Phase 2A must retain sequential teacher-forced/recursive MTP semantics",
    )
    check(
        "phase2a_modes",
        tuple(phase2a.get("conditioning_modes", ())) == PHASE2A_MODES,
        f"Phase 2A conditioning modes must be exactly {list(PHASE2A_MODES)}",
    )
    check(
        "phase2a_gradient_order",
        tuple(phase2a.get("gradient_mode_order", ())) == PHASE2A_GRADIENT_ORDER
        and phase2a.get("primary_gradient_mode") == "frozen_probe",
        "Phase 2A must identify existing information before stop-gradient or joint co-training",
    )
    readout = _mapping(
        phase2a.get("trajectory_readout"),
        "stages.phase2a_trajectory_mtp.trajectory_readout",
    )
    check(
        "phase2a_readout_isolation",
        readout.get("telemetry") == "control_only"
        and int(readout.get("registered_block_count", 0)) == 1
        and readout.get("quadratic_attention_maps_allowed") is False
        and readout.get("cascade_trajectory_side_channel_allowed") is False,
        "Phase 2A readout must remain one-block, linear-memory, and free of a cascade trajectory side channel",
    )
    check(
        "phase2a_mechanism_prerequisite",
        phase2a.get("requires_mtp_free_mechanism_go_for_mechanism_interpretation") is True,
        "Phase 2A may not create a mechanism claim after an MTP-free mechanism NO_GO",
    )

    check(
        "phase2b_semantics",
        phase2b.get("study_kind") == "system_factorial"
        and phase2b.get("mtp_semantics") == PHASE2B_SEMANTICS,
        "Phase 2B must retain independent-offset exact-verification semantics",
    )
    check(
        "phase2_namespaces_distinct",
        phase2a.get("implementation_namespace")
        != phase2b.get("implementation_namespace"),
        "Phase 2A and Phase 2B must have distinct implementation namespaces",
    )
    check(
        "phase2b_cells",
        tuple(phase2b.get("primary_cells", ())) == PHASE2B_CELLS,
        f"Phase 2B primary cells must be exactly {list(PHASE2B_CELLS)}",
    )
    check(
        "phase2b_capacity_controls",
        phase2b.get("all_cells_have_same_mtp_parameter_structure") is True
        and phase2b.get("ar_cells_keep_mtp_dormant_and_frozen") is True,
        "Phase 2B must preserve dormant same-structure AR controls",
    )
    exact = set(str(value) for value in phase2b.get("exact_decode_gates", ()))
    check(
        "phase2b_exact_decode",
        exact
        == {
            "token_id_sequence",
            "termination_reason",
            "accepted_prefix_order",
            "first_mismatch_rollback",
            "final_greedy_sequence",
        },
        "Phase 2B exact decoding gates were weakened",
    )
    check(
        "phase2b_quality_and_serial_gates",
        phase2b.get("quality_gate")
        == "registered_noninferiority_of_tgsa_mtp_to_parallel_ar"
        and phase2b.get("serial_round_gate")
        == "registered_minimum_reduction_for_mtp_on_tgsa",
        "Phase 2B must preserve the registered quality and serial-round gates",
    )
    check(
        "phase2b_claim_boundary",
        phase2b.get("wall_clock_gating") is False
        and phase2b.get("mechanism_interpretation_allowed") is False
        and phase2b.get("may_proceed_after_mechanism_no_go") is True,
        "Phase 2B is a system lane: it may proceed independently but cannot claim mechanism or wall-clock speed",
    )

    rineya = _mapping(
        stages.get("rineya_checkpoint_qualification"),
        "stages.rineya_checkpoint_qualification",
    )
    check(
        "rineya_cells",
        tuple(rineya.get("required_cells", ())) == PHASE2B_CELLS,
        "real-checkpoint qualification must preserve the complete paired four-cell block",
    )
    required_gates = set(str(value) for value in rineya.get("required_gates", ()))
    check(
        "rineya_gates",
        {
            "source_and_data_hashes",
            "paired_initialization",
            "main_path_initialization_drift",
            "exact_verified_decoding",
            "closed_evidence_verifier",
            "within_scope_competence",
            "registered_quality_noninferiority",
            "registered_serial_round_reduction",
        }.issubset(required_gates),
        "real-checkpoint qualification is missing a required integrity, quality, or decoding gate",
    )
    check(
        "rineya_claim_boundary",
        rineya.get("wall_clock_claim_allowed") is False
        and rineya.get("broad_quality_claim_allowed") is False
        and rineya.get("mechanism_claim_allowed_without_mechanism_lane_go") is False,
        "a single checkpoint qualification may not create wall-clock, broad-quality, or unlicensed mechanism claims",
    )

    wall_clock = _mapping(stages.get("wall_clock_benchmark"), "stages.wall_clock_benchmark")
    check(
        "wall_clock_separate",
        wall_clock.get("implementation_status") == "separate_preregistration_required"
        and wall_clock.get("requires_prior_checkpoint_qualification_go") is True
        and wall_clock.get("serial_round_proxy_may_substitute") is False,
        "wall-clock benchmarking must remain a separate post-qualification study",
    )

    lanes = _mapping(contract.get("promotion_lanes"), "promotion_lanes")
    system_lane = _mapping(lanes.get("system_utility_lane"), "promotion_lanes.system_utility_lane")
    mechanism_lane = _mapping(
        lanes.get("mechanism_interpretable_lane"),
        "promotion_lanes.mechanism_interpretable_lane",
    )
    check(
        "independent_system_lane",
        system_lane.get("does_not_require") == "a positive TGSA mechanism result"
        and "system-level result" in str(system_lane.get("forbids", "")),
        "system utility may be tested independently but must not be laundered into mechanism evidence",
    )
    check(
        "mechanism_lane_order",
        mechanism_lane.get("ordered_requirements")
        == [
            "v5_apparatus:GO",
            "mtp_free_mechanism:GO",
            "phase2a_trajectory_mtp:GO",
            "rineya_checkpoint_qualification:GO",
        ],
        "the mechanism-interpretable promotion lane was reordered or weakened",
    )

    if errors:
        raise V51ContractError("; ".join(errors))
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "passed": True,
        "checks": dict(sorted(checks.items())),
        "errors": [],
        "programme_sha256": canonical_sha256(contract),
        "claim_scope": "programme identity and promotion-boundary validation only",
    }


def _git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def verify_repository_contract(
    *,
    repo_root: Path,
    contract_path: Path,
) -> dict[str, object]:
    """Verify the frozen contract against the actual Git checkout before output."""

    repo_root = Path(repo_root).resolve()
    contract_path = Path(contract_path).resolve()
    checks: dict[str, bool] = {}
    errors: list[str] = []
    contract: dict[str, Any] | None = None
    contract_validation: dict[str, object] | None = None
    head: str | None = None
    source_plan_blob: str | None = None
    source_plan_worktree_blob: str | None = None

    try:
        contract = load_object(contract_path)
        contract_validation = validate_contract(contract)
        checks["contract_valid"] = True
    except Exception as error:  # noqa: BLE001 - report every fail-closed reason
        checks["contract_valid"] = False
        errors.append(f"contract: {type(error).__name__}: {error}")

    git_dir = _git(repo_root, "rev-parse", "--git-dir")
    checks["git_checkout"] = git_dir.returncode == 0
    if not checks["git_checkout"]:
        errors.append("repo_root is not a Git checkout")
    else:
        head_result = _git(repo_root, "rev-parse", "HEAD")
        if head_result.returncode == 0:
            head = head_result.stdout.strip()
        checks["head_recorded"] = bool(head)
        if not checks["head_recorded"]:
            errors.append("Git HEAD could not be resolved")

        status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
        clean = status.returncode == 0 and not status.stdout.strip()
        checks["clean_worktree"] = clean
        if not clean:
            errors.append("v5.1 execution requires a clean worktree before any output is created")

        ancestor = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            V5_MECHANISM_COMMIT,
            "HEAD",
        )
        checks["head_descends_from_v5_mechanism"] = ancestor.returncode == 0
        if ancestor.returncode != 0:
            errors.append("HEAD does not descend from the reviewed v5 mechanism source")

        contains = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            V5_EXECUTION_COMMIT,
            V5_MECHANISM_COMMIT,
        )
        checks["v5_mechanism_contains_execution"] = contains.returncode == 0
        if contains.returncode != 0:
            errors.append("the reviewed v5 mechanism commit does not contain the execution commit")

        pinned = (
            V4_RESULT_COMMIT,
            V5_EXECUTION_COMMIT,
            V5_MECHANISM_COMMIT,
            PHASE2B_COMMIT,
            PRETRAINED_RETROFIT_COMMIT,
        )
        missing_commits: list[str] = []
        for commit in pinned:
            exists = _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}")
            if exists.returncode != 0:
                missing_commits.append(commit)
        checks["pinned_commit_objects_present"] = not missing_commits
        if missing_commits:
            errors.append(
                "full lineage is not present locally; fetch all branches before execution: "
                + ", ".join(missing_commits)
            )

        source_plan_path = repo_root / SOURCE_V5_PLAN_PATH
        checks["source_plan_exists"] = source_plan_path.is_file()
        if not source_plan_path.is_file():
            errors.append(f"locked source plan is missing: {SOURCE_V5_PLAN_PATH}")
        else:
            payload = source_plan_path.read_bytes()
            source_plan_worktree_blob = git_blob_sha1(payload)
            checks["source_plan_worktree_blob"] = (
                source_plan_worktree_blob == SOURCE_V5_PLAN_BLOB
            )
            if source_plan_worktree_blob != SOURCE_V5_PLAN_BLOB:
                errors.append("working-tree source plan bytes do not match the v5.1 registration lock")

            tree_blob = _git(repo_root, "rev-parse", f"HEAD:{SOURCE_V5_PLAN_PATH}")
            if tree_blob.returncode == 0:
                source_plan_blob = tree_blob.stdout.strip()
            checks["source_plan_committed_blob"] = source_plan_blob == SOURCE_V5_PLAN_BLOB
            if source_plan_blob != SOURCE_V5_PLAN_BLOB:
                errors.append("HEAD does not carry the exact reviewed v5 source-plan blob")

    if contract_validation is None:
        programme_sha = None
    else:
        programme_sha = contract_validation["programme_sha256"]
    report = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "passed": not errors and bool(checks) and all(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": errors,
        "programme_sha256": programme_sha,
        "contract_path": str(contract_path),
        "repo_root": str(repo_root),
        "repo_head": head,
        "source_plan_path": SOURCE_V5_PLAN_PATH,
        "source_plan_expected_blob_sha1": SOURCE_V5_PLAN_BLOB,
        "source_plan_committed_blob_sha1": source_plan_blob,
        "source_plan_worktree_blob_sha1": source_plan_worktree_blob,
        "claim_scope": (
            "prospective registration, lineage, and promotion-boundary verification only; "
            "this report contains no competence, mechanism, Rineya-quality, serial-round, "
            "or wall-clock result"
        ),
    }
    return report


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


__all__ = [
    "COORDINATION_BRANCH",
    "PROGRAMME_ID",
    "PROGRAMME_SCHEMA_VERSION",
    "SOURCE_V5_PLAN_BLOB",
    "SOURCE_V5_PLAN_PATH",
    "V51ContractError",
    "VERIFICATION_SCHEMA_VERSION",
    "atomic_write_json",
    "canonical_sha256",
    "git_blob_sha1",
    "load_object",
    "validate_contract",
    "verify_repository_contract",
]
