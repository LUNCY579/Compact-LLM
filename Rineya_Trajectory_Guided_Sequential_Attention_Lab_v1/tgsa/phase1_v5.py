"""Versioned contracts and arm-blind analysis for TGSA Phase 1 v5.

V5 is an apparatus-qualification programme, not a TGSA mechanism result.  This
module turns the null-anchored preregistration into executable, fail-closed
objects while keeping three boundaries explicit:

* the prerequisite probe can select a capacity rung but cannot authorize a
  mechanism experiment;
* the qualification grid can establish an informative regime but cannot be
  interpreted as evidence for or against TGSA;
* only a separately frozen, MTP-free mechanism study may compare TGSA arms.

The functions here are deliberately pure where possible.  Training and artifact
orchestration live in :mod:`tgsa.phase1_v5_runner`; this module owns scientific
identity, registered ordering, shortcut nulls, gates, and promotion decisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from .phase1_competence import (
    ArmCompetence,
    CompetenceThresholds,
    evaluate_competence,
)
from .phase1_data import (
    Phase1Example,
    SyntheticVocabulary,
    canonical_json_bytes,
    sha256_bytes,
)
from .phase1_difficulty import shortcut_baselines

V5_PLAN_SCHEMA_VERSION = "tgsa.phase1-v5-plan.v1"
V5_PROBE_REPORT_SCHEMA_VERSION = "tgsa.phase1-v5-probe-report.v1"
V5_QUALIFICATION_REPORT_SCHEMA_VERSION = "tgsa.phase1-v5-qualification-report.v1"
V5_FROZEN_BAND_SCHEMA_VERSION = "tgsa.phase1-v5-frozen-band.v1"
V5_SELECTION_REPORT_SCHEMA_VERSION = "tgsa.phase1-v5-selection-report.v1"

V5_PHASE = "phase1_v5"
V5_STUDY_KIND = "apparatus_qualification"
V5_MECHANISM_SEMANTICS = "none"
V5_REQUIRED_ARMS = ("cascade", "history_control", "learned_delta")
V5_PRIMARY_FAMILIES = ("pointer_chase",)
V5_GUARDRAIL_FAMILIES = ("direct_lookup",)
V5_CANARY_FAMILIES = ("state_update",)
V5_SPLIT_GENERATION_ORDER = ("test", "dev", "train")

REGISTRATION_STATES = {
    "development_only",
    "implementation_complete_not_registered",
    "frozen_after_probe",
}
SELECTION_POLICY = (
    "minimum_sample_presentations",
    "maximum_unique_train_examples",
    "lexicographic_exposure_id",
)


class V5ContractError(ValueError):
    """Raised when a v5 object would weaken or blur the registered design."""


@dataclass(frozen=True, slots=True)
class CapacityRung:
    rung_id: str
    embed_dim: int
    num_layers: int
    num_heads: int
    trajectory_rank: int

    def __post_init__(self) -> None:
        if not self.rung_id:
            raise V5ContractError("capacity rung_id must be non-empty")
        if self.embed_dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise V5ContractError("capacity dimensions must be positive")
        if self.embed_dim % self.num_heads:
            raise V5ContractError(
                f"capacity rung {self.rung_id}: embed_dim must be divisible by num_heads"
            )
        if self.trajectory_rank <= 0:
            raise V5ContractError("trajectory_rank must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CapacityRung":
        return cls(
            rung_id=str(value["id"]),
            embed_dim=int(value["embed_dim"]),
            num_layers=int(value["num_layers"]),
            num_heads=int(value["num_heads"]),
            trajectory_rank=int(value.get("trajectory_rank", 16)),
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["id"] = value.pop("rung_id")
        return value


@dataclass(frozen=True, slots=True)
class ExposureCell:
    exposure_id: str
    unique_train_examples: int
    sample_presentations: int
    batch_size: int

    def __post_init__(self) -> None:
        if not self.exposure_id:
            raise V5ContractError("exposure id must be non-empty")
        if self.unique_train_examples <= 0:
            raise V5ContractError("unique_train_examples must be positive")
        if self.sample_presentations <= 0:
            raise V5ContractError("sample_presentations must be positive")
        if self.batch_size <= 0:
            raise V5ContractError("batch_size must be positive")
        if self.sample_presentations % self.batch_size:
            raise V5ContractError(
                f"exposure {self.exposure_id}: sample_presentations must be exactly "
                "divisible by batch_size"
            )

    @property
    def optimizer_steps(self) -> int:
        return self.sample_presentations // self.batch_size

    @property
    def effective_epochs(self) -> float:
        return self.sample_presentations / self.unique_train_examples

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ExposureCell":
        return cls(
            exposure_id=str(value["id"]),
            unique_train_examples=int(value["unique_train_examples"]),
            sample_presentations=int(value["sample_presentations"]),
            batch_size=int(value["batch_size"]),
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["id"] = value.pop("exposure_id")
        value["optimizer_steps"] = self.optimizer_steps
        value["effective_epochs"] = self.effective_epochs
        return value


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible object under the repository canonical encoding."""

    return sha256_bytes(canonical_json_bytes(value))


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise V5ContractError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V5ContractError(f"{field} must be a list")
    return cast(Sequence[object], value)


def _require_exact_tuple(value: object, expected: Sequence[str], *, field: str) -> None:
    actual = tuple(str(item) for item in _as_sequence(value, field=field))
    if actual != tuple(expected):
        raise V5ContractError(f"{field} must be {list(expected)!r}; got {list(actual)!r}")


def _require_boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise V5ContractError(f"{field} must be an explicit boolean")
    return value


def _validate_weights(weights: Mapping[str, object], roles: Sequence[str], *, field: str) -> None:
    if set(weights) != set(roles):
        raise V5ContractError(
            f"{field} must name exactly {sorted(roles)}; got {sorted(str(k) for k in weights)}"
        )
    values = [float(weights[name]) for name in roles]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise V5ContractError(f"{field} values must be finite and positive")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise V5ContractError(f"{field} values must sum to exactly one")


def _validate_family_spec(
    spec: Mapping[str, object],
    *,
    family: str,
    development_only: bool,
) -> None:
    if not bool(spec.get("reject_trivial", False)):
        raise V5ContractError(f"dataset.families.{family}.reject_trivial must be true")
    if family in {"pointer_chase", "direct_lookup"}:
        for key in ("min_nodes", "max_nodes"):
            if int(spec.get(key, 0)) <= 0:
                raise V5ContractError(f"dataset.families.{family}.{key} must be positive")
        if int(spec["min_nodes"]) > int(spec["max_nodes"]):
            raise V5ContractError(f"dataset.families.{family} node range is reversed")
    if family == "pointer_chase":
        if int(spec.get("min_hops", 0)) != 1 or int(spec.get("max_hops", 0)) != 3:
            if not development_only:
                raise V5ContractError(
                    "registered pointer_chase must vary hops 1..3 within one family"
                )
    elif family == "direct_lookup":
        if int(spec.get("min_hops", 1)) != 1 or int(spec.get("max_hops", 1)) != 1:
            raise V5ContractError("direct_lookup must remain a one-hop guardrail")
    elif family == "state_update":
        for key in ("min_steps", "max_steps"):
            if int(spec.get(key, 0)) <= 0:
                raise V5ContractError(f"dataset.families.state_update.{key} must be positive")
        if int(spec["min_steps"]) > int(spec["max_steps"]):
            raise V5ContractError("state_update step range is reversed")


def validate_v5_plan(value: Mapping[str, object]) -> dict[str, Any]:
    """Validate and return a defensive copy of a v5 programme plan.

    The validator intentionally rejects scientifically ambiguous states rather
    than trying to coerce them.  In particular, an apparatus plan may never be
    marked confirmatory, enable MTP, authorize a mechanism test, rename the
    family roles, or change split generation order.
    """

    plan = copy.deepcopy(dict(value))
    if plan.get("schema_version") != V5_PLAN_SCHEMA_VERSION:
        raise V5ContractError(
            f"unsupported v5 plan schema: {plan.get('schema_version')!r}"
        )
    if plan.get("phase") != V5_PHASE:
        raise V5ContractError(f"phase must be {V5_PHASE!r}")
    if plan.get("study_kind") != V5_STUDY_KIND:
        raise V5ContractError(f"study_kind must be {V5_STUDY_KIND!r}")
    if plan.get("mechanism_semantics") != V5_MECHANISM_SEMANTICS:
        raise V5ContractError("v5 apparatus qualification has no mechanism semantics")
    if _require_boolean(plan.get("confirmatory"), field="confirmatory"):
        raise V5ContractError("v5 apparatus qualification may not be confirmatory")
    if _require_boolean(
        plan.get("mechanism_test_authorized"), field="mechanism_test_authorized"
    ):
        raise V5ContractError("a plan cannot pre-authorize its downstream mechanism study")
    development_only = _require_boolean(
        plan.get("development_only"), field="development_only"
    )
    registration_status = str(plan.get("registration_status", ""))
    if registration_status not in REGISTRATION_STATES:
        raise V5ContractError(
            f"registration_status must be one of {sorted(REGISTRATION_STATES)}"
        )
    if development_only and registration_status != "development_only":
        raise V5ContractError(
            "development_only plans must carry registration_status=development_only"
        )
    if not development_only and registration_status == "development_only":
        raise V5ContractError("non-development plan cannot use a development-only status")

    study_id = str(plan.get("study_id", ""))
    if not study_id.startswith("tgsa-phase1-v5-"):
        raise V5ContractError("study_id must start with 'tgsa-phase1-v5-'")
    if not str(plan.get("claim_ceiling", "")).strip():
        raise V5ContractError("claim_ceiling must be non-empty")

    _require_exact_tuple(
        plan.get("registered_arms"), V5_REQUIRED_ARMS, field="registered_arms"
    )
    roles = _as_mapping(plan.get("family_roles"), field="family_roles")
    _require_exact_tuple(
        roles.get("primary"), V5_PRIMARY_FAMILIES, field="family_roles.primary"
    )
    _require_exact_tuple(
        roles.get("guardrail"),
        V5_GUARDRAIL_FAMILIES,
        field="family_roles.guardrail",
    )
    _require_exact_tuple(
        roles.get("canary"), V5_CANARY_FAMILIES, field="family_roles.canary"
    )

    mtp = _as_mapping(plan.get("mtp"), field="mtp")
    if _require_boolean(mtp.get("enabled"), field="mtp.enabled"):
        raise V5ContractError("MTP is forbidden during v5 apparatus qualification")

    dataset = _as_mapping(plan.get("dataset"), field="dataset")
    _require_exact_tuple(
        dataset.get("split_generation_order"),
        V5_SPLIT_GENERATION_ORDER,
        field="dataset.split_generation_order",
    )
    vocabulary = _as_mapping(dataset.get("vocabulary"), field="dataset.vocabulary")
    SyntheticVocabulary(
        domain_size=int(vocabulary.get("domain_size", 0)),
        max_hops=int(vocabulary.get("max_hops", 0)),
        max_abs_step=int(vocabulary.get("max_abs_step", 0)),
    )
    families = _as_mapping(dataset.get("families"), field="dataset.families")
    expected_families = (
        *V5_PRIMARY_FAMILIES,
        *V5_GUARDRAIL_FAMILIES,
        *V5_CANARY_FAMILIES,
    )
    if set(families) != set(expected_families):
        raise V5ContractError(
            f"dataset.families must be exactly {sorted(expected_families)}"
        )
    for family in expected_families:
        _validate_family_spec(
            _as_mapping(families[family], field=f"dataset.families.{family}"),
            family=family,
            development_only=development_only,
        )
    _validate_weights(
        _as_mapping(dataset.get("family_weights"), field="dataset.family_weights"),
        expected_families,
        field="dataset.family_weights",
    )
    for key in ("dev_examples", "test_examples", "seed"):
        if int(dataset.get(key, 0)) <= 0:
            raise V5ContractError(f"dataset.{key} must be positive")

    ladder_raw = _as_sequence(plan.get("capacity_ladder"), field="capacity_ladder")
    ladder = [
        CapacityRung.from_mapping(_as_mapping(item, field="capacity_ladder[]"))
        for item in ladder_raw
    ]
    if not ladder or len({rung.rung_id for rung in ladder}) != len(ladder):
        raise V5ContractError("capacity_ladder must be non-empty with unique ids")
    if not development_only:
        expected_ladder = (
            CapacityRung("C1", 96, 2, 6, 16),
            CapacityRung("C2", 96, 4, 6, 16),
            CapacityRung("C3", 192, 4, 6, 16),
        )
        if tuple(ladder) != expected_ladder:
            raise V5ContractError(
                "non-development v5 capacity ladder is frozen as "
                "C1=96x2, C2=96x4, C3=192x4 with six heads"
            )

    probe = _as_mapping(plan.get("probe"), field="probe")
    if str(probe.get("arm")) != "cascade":
        raise V5ContractError("probe.arm must be cascade")
    if str(probe.get("family")) != "pointer_chase":
        raise V5ContractError("probe.family must be pointer_chase")
    seeds = tuple(int(seed) for seed in _as_sequence(probe.get("seeds"), field="probe.seeds"))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise V5ContractError("probe requires at least three unique seeds")
    probe_exposure = ExposureCell.from_mapping(
        _as_mapping(probe.get("exposure"), field="probe.exposure")
    )
    probe_thresholds = _as_mapping(probe.get("thresholds"), field="probe.thresholds")
    required_probe_thresholds = (
        "minimum_margin_over_family_null",
        "maximum_mean_accuracy",
        "maximum_seed_coefficient_of_variation",
        "maximum_generalization_gap",
        "minimum_seed_runs",
    )
    for key in required_probe_thresholds:
        if key not in probe_thresholds:
            raise V5ContractError(f"probe.thresholds is missing {key}")
    if float(probe_thresholds["minimum_margin_over_family_null"]) < 0.15:
        raise V5ContractError("probe null margin may not be weaker than 0.150")
    if float(probe_thresholds["maximum_mean_accuracy"]) > 0.95:
        raise V5ContractError("probe saturation ceiling may not exceed 0.95")
    if float(probe_thresholds["maximum_seed_coefficient_of_variation"]) > 0.35:
        raise V5ContractError("probe seed-CV bound may not exceed 0.35")
    if float(probe_thresholds["maximum_generalization_gap"]) > 0.10:
        raise V5ContractError("probe generalization-gap bound may not exceed 0.10")
    if int(probe_thresholds["minimum_seed_runs"]) < 3:
        raise V5ContractError("probe requires at least three seed runs")
    if not development_only:
        if (
            probe_exposure.unique_train_examples != 50_000
            or probe_exposure.sample_presentations != 1_280_000
        ):
            raise V5ContractError(
                "registered probe exposure is 50,000 unique examples and "
                "1,280,000 sample presentations"
            )

    exposure_raw = _as_sequence(plan.get("exposure_grid"), field="exposure_grid")
    exposures = [
        ExposureCell.from_mapping(_as_mapping(item, field="exposure_grid[]"))
        for item in exposure_raw
    ]
    if not exposures or len({cell.exposure_id for cell in exposures}) != len(exposures):
        raise V5ContractError("exposure_grid must be non-empty with unique ids")
    if not development_only:
        expected_pairs = {
            (50_000, 1_280_000),
            (200_000, 1_280_000),
            (50_000, 2_560_000),
            (200_000, 2_560_000),
        }
        actual_pairs = {
            (cell.unique_train_examples, cell.sample_presentations)
            for cell in exposures
        }
        if actual_pairs != expected_pairs or len(exposures) != 4:
            raise V5ContractError(
                "non-development exposure_grid must cross 50k/200k diversity "
                "with 1.28M/2.56M presentations"
            )

    qualification = _as_mapping(plan.get("qualification"), field="qualification")
    qualification_seeds = tuple(
        int(seed)
        for seed in _as_sequence(
            qualification.get("seeds"), field="qualification.seeds"
        )
    )
    if len(qualification_seeds) < 3 or len(set(qualification_seeds)) != len(
        qualification_seeds
    ):
        raise V5ContractError("qualification requires at least three unique seeds")
    thresholds = _as_mapping(
        qualification.get("thresholds"), field="qualification.thresholds"
    )
    required_competence = {
        "minimum_margin_over_trivial",
        "minimum_absolute_accuracy",
        "maximum_answer_nll_fraction_of_uniform",
        "minimum_distinct_predictions",
        "maximum_family_accuracy",
        "maximum_generalization_gap",
        "maximum_seed_coefficient_of_variation",
        "minimum_family_margin_over_null",
        "minimum_seed_runs",
    }
    missing = required_competence - set(thresholds)
    if missing:
        raise V5ContractError(
            f"qualification.thresholds is missing {sorted(missing)}"
        )
    if float(thresholds["minimum_family_margin_over_null"]) < 0.030:
        raise V5ContractError("family-null margin may not be weaker than 0.030")
    if float(thresholds["maximum_family_accuracy"]) > 0.95:
        raise V5ContractError("primary-family saturation ceiling may not exceed 0.95")
    if float(thresholds["maximum_generalization_gap"]) > 0.10:
        raise V5ContractError("generalization-gap bound may not exceed 0.10")
    if float(
        thresholds["maximum_seed_coefficient_of_variation"]
    ) > 0.35:
        raise V5ContractError("seed-CV bound may not exceed 0.35")
    if int(thresholds["minimum_seed_runs"]) < 3:
        raise V5ContractError("qualification requires at least three seed runs")
    if float(qualification.get("guardrail_minimum_margin_over_null", 0.0)) < 0.15:
        raise V5ContractError(
            "direct-lookup guardrail margin may not be weaker than 0.150"
        )
    if int(qualification.get("seen_examples", 0)) <= 0:
        raise V5ContractError("qualification.seen_examples must be positive")
    if int(qualification.get("evaluation_batch_size", 0)) <= 0:
        raise V5ContractError("qualification.evaluation_batch_size must be positive")
    if not _require_boolean(
        qualification.get("require_complete_exposure_grid"),
        field="qualification.require_complete_exposure_grid",
    ):
        raise V5ContractError(
            "qualification must complete the registered exposure grid before selection"
        )
    _require_exact_tuple(
        qualification.get("selection_policy"),
        SELECTION_POLICY,
        field="qualification.selection_policy",
    )

    training = _as_mapping(plan.get("training"), field="training")
    if float(training.get("learning_rate", 0.0)) <= 0.0:
        raise V5ContractError("training.learning_rate must be positive")
    if not 0.0 <= float(training.get("warmup_fraction", -1.0)) < 1.0:
        raise V5ContractError("training.warmup_fraction must be in [0, 1)")
    if float(training.get("gradient_clip_norm", 0.0)) <= 0.0:
        raise V5ContractError("training.gradient_clip_norm must be positive")
    betas = _as_sequence(training.get("betas"), field="training.betas")
    if len(betas) != 2 or any(not 0.0 < float(value) < 1.0 for value in betas):
        raise V5ContractError("training.betas must contain two values in (0, 1)")
    if float(training.get("dropout", -1.0)) != 0.0:
        raise V5ContractError("v5 requires dropout=0 for paired determinism")

    runtime = _as_mapping(plan.get("runtime"), field="runtime")
    _require_boolean(
        runtime.get("deterministic_algorithms"),
        field="runtime.deterministic_algorithms",
    )

    # Make the computed identities visible to every caller without changing the
    # hash of the source plan itself.
    plan["_validated_capacity_ladder"] = [rung.to_dict() for rung in ladder]
    plan["_validated_probe_exposure"] = probe_exposure.to_dict()
    plan["_validated_exposure_grid"] = [cell.to_dict() for cell in exposures]
    return cast(dict[str, Any], plan)


def load_v5_plan(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V5ContractError("v5 plan must be a JSON object")
    return validate_v5_plan(cast(Mapping[str, object], value))


def source_plan(plan: Mapping[str, object]) -> dict[str, object]:
    """Strip validator-only fields before hashing or snapshotting a plan."""

    return {
        str(key): copy.deepcopy(value)
        for key, value in plan.items()
        if not str(key).startswith("_validated_")
    }


def plan_sha256(plan: Mapping[str, object]) -> str:
    return canonical_sha256(source_plan(plan))


def capacity_ladder(plan: Mapping[str, object]) -> tuple[CapacityRung, ...]:
    raw = _as_sequence(plan["capacity_ladder"], field="capacity_ladder")
    return tuple(
        CapacityRung.from_mapping(_as_mapping(item, field="capacity_ladder[]"))
        for item in raw
    )


def exposure_grid(plan: Mapping[str, object]) -> tuple[ExposureCell, ...]:
    raw = _as_sequence(plan["exposure_grid"], field="exposure_grid")
    return tuple(
        ExposureCell.from_mapping(_as_mapping(item, field="exposure_grid[]"))
        for item in raw
    )


def probe_exposure(plan: Mapping[str, object]) -> ExposureCell:
    probe = _as_mapping(plan["probe"], field="probe")
    return ExposureCell.from_mapping(
        _as_mapping(probe["exposure"], field="probe.exposure")
    )


def resolve_rung(plan: Mapping[str, object], rung_id: str) -> CapacityRung:
    for rung in capacity_ladder(plan):
        if rung.rung_id == rung_id:
            return rung
    raise V5ContractError(f"unknown capacity rung {rung_id!r}")


def resolve_exposure(plan: Mapping[str, object], exposure_id: str) -> ExposureCell:
    for cell in exposure_grid(plan):
        if cell.exposure_id == exposure_id:
            return cell
    raise V5ContractError(f"unknown exposure cell {exposure_id!r}")


def allocate_family_counts(
    total: int,
    weights: Mapping[str, object],
) -> dict[str, int]:
    """Allocate an exact total using deterministic largest remainders."""

    if total <= 0:
        raise V5ContractError("allocation total must be positive")
    names = sorted(str(name) for name in weights)
    _validate_weights(weights, names, field="family_weights")
    quotas = {name: total * float(weights[name]) for name in names}
    counts = {name: int(math.floor(quotas[name])) for name in names}
    remaining = total - sum(counts.values())
    order = sorted(names, key=lambda name: (-(quotas[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != total:
        raise AssertionError("family allocation did not preserve the requested total")
    if total >= len(names) and any(value <= 0 for value in counts.values()):
        raise V5ContractError("positive family weights unexpectedly received zero examples")
    return counts


def stage_dataset_config(
    plan: Mapping[str, object],
    *,
    stage: str,
    exposure: ExposureCell,
) -> dict[str, object]:
    """Build the exact materialized dataset configuration for one v5 cell.

    Evaluation splits are generated before training so they are unbiased draws.
    All qualification exposure cells use the same dataset seed; therefore the
    50k train set is a deterministic prefix of the 200k train set, and cells
    differing only in presentation count consume byte-identical data.
    """

    if stage not in {"probe", "qualification"}:
        raise V5ContractError("stage must be 'probe' or 'qualification'")
    dataset = _as_mapping(plan["dataset"], field="dataset")
    vocabulary = copy.deepcopy(
        dict(_as_mapping(dataset["vocabulary"], field="dataset.vocabulary"))
    )
    base_families = {
        str(name): copy.deepcopy(dict(_as_mapping(spec, field=f"dataset.families.{name}")))
        for name, spec in _as_mapping(dataset["families"], field="dataset.families").items()
    }
    if stage == "probe":
        pointer = dict(base_families["pointer_chase"])
        pointer["min_hops"] = 1
        pointer["max_hops"] = 1
        families = {"pointer_chase": pointer}
        train_counts = {"pointer_chase": exposure.unique_train_examples}
        probe = _as_mapping(plan["probe"], field="probe")
        dev_counts = {"pointer_chase": int(probe["dev_examples"])}
        test_counts = {"pointer_chase": int(probe["test_examples"])}
        seed = int(probe["dataset_seed"])
    else:
        families = base_families
        weights = _as_mapping(dataset["family_weights"], field="dataset.family_weights")
        train_counts = allocate_family_counts(exposure.unique_train_examples, weights)
        dev_counts = allocate_family_counts(int(dataset["dev_examples"]), weights)
        test_counts = allocate_family_counts(int(dataset["test_examples"]), weights)
        seed = int(dataset["seed"])
    return {
        "vocabulary": vocabulary,
        "seed": seed,
        "split_generation_order": list(V5_SPLIT_GENERATION_ORDER),
        "splits": {
            "train": train_counts,
            "dev": dev_counts,
            "test": test_counts,
        },
        "families": families,
    }


def _candidate_chance(
    example: Phase1Example,
    *,
    family_spec: Mapping[str, object],
    domain_size: int,
) -> float:
    if example.family in {"pointer_chase", "direct_lookup"}:
        node_count = int(example.metadata.get("node_count", 0))
        if node_count <= 0:
            raise V5ContractError(
                f"{example.id}: pointer-family example has no positive node_count"
            )
        candidates = node_count - 1 if bool(family_spec.get("reject_trivial")) else node_count
        return 1.0 / float(max(candidates, 1))
    return 1.0 / float(domain_size)


def registered_family_null_report(
    train: Sequence[Phase1Example],
    holdout: Sequence[Phase1Example],
    *,
    family_specs: Mapping[str, object],
    vocab: SyntheticVocabulary,
) -> dict[str, dict[str, object]]:
    """Compute each family's strongest no-competence strategy.

    V5 exposes every component instead of returning only a scalar.  The maximum
    includes candidate chance, copy-start, copy-last-value, train majority, and
    evaluation majority.  The final term is intentionally explicit: the
    historical v4 helper documented it but omitted it from the implementation.
    """

    if not train or not holdout:
        raise V5ContractError("family nulls require non-empty train and holdout data")
    shortcuts = shortcut_baselines(train, holdout, vocab=vocab)
    report: dict[str, dict[str, object]] = {}
    for family in sorted({example.family for example in holdout}):
        selection = [example for example in holdout if example.family == family]
        if not selection:
            continue
        spec = _as_mapping(
            family_specs.get(family), field=f"family_specs.{family}"
        )
        candidate = statistics.fmean(
            _candidate_chance(
                example,
                family_spec=spec,
                domain_size=vocab.domain_size,
            )
            for example in selection
        )
        shortcut = _as_mapping(
            shortcuts.get(family, {}), field=f"shortcut_baselines.{family}"
        )
        eval_counts = Counter(example.answer_token for example in selection)
        eval_majority = max(eval_counts.values()) / len(selection)
        components = {
            "candidate_chance": candidate,
            "copy_start": float(shortcut.get("copy_start_accuracy", 0.0)),
            "copy_last_value": float(
                shortcut.get("copy_last_value_accuracy", 0.0)
            ),
            "train_majority": float(
                shortcut.get("train_majority_accuracy", 0.0)
            ),
            "evaluation_majority": eval_majority,
        }
        registered = max(components.values())
        winners = sorted(
            name
            for name, score in components.items()
            if math.isclose(score, registered, rel_tol=0.0, abs_tol=1.0e-12)
        )
        report[family] = {
            "examples": len(selection),
            "components": components,
            "registered_null": registered,
            "winning_strategies": winners,
        }
    return report


def family_null_values(
    report: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    return {
        str(family): float(entry["registered_null"])
        for family, entry in report.items()
    }


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean <= 0.0:
        return math.inf
    return statistics.stdev(values) / mean


def evaluate_prerequisite_probe(
    *,
    rung_id: str,
    rung_index: int,
    rung_count: int,
    per_seed_holdout_accuracy: Sequence[float],
    per_seed_seen_accuracy: Sequence[float],
    family_null: float,
    thresholds: Mapping[str, object],
    development_only: bool,
) -> dict[str, object]:
    """Apply the registered one-hop capacity probe without mechanism inference."""

    holdout = [float(value) for value in per_seed_holdout_accuracy]
    seen = [float(value) for value in per_seed_seen_accuracy]
    minimum_runs = int(thresholds["minimum_seed_runs"])
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    checks["minimum_seed_runs_recorded"] = (
        len(holdout) >= minimum_runs and len(holdout) == len(seen)
    )
    if not checks["minimum_seed_runs_recorded"]:
        reasons.append(
            f"recorded {len(holdout)} holdout and {len(seen)} seen runs; "
            f"{minimum_runs} aligned runs are required"
        )

    mean = statistics.fmean(holdout) if holdout else 0.0
    weakest = min(holdout) if holdout else 0.0
    margin = weakest - float(family_null)
    required_margin = float(thresholds["minimum_margin_over_family_null"])
    checks["weakest_seed_beats_family_null"] = margin >= required_margin
    if not checks["weakest_seed_beats_family_null"]:
        reasons.append(
            f"weakest seed clears the family null by {margin:+.4f}, below "
            f"the required {required_margin:+.4f}"
        )

    ceiling = float(thresholds["maximum_mean_accuracy"])
    checks["one_hop_not_saturated"] = mean <= ceiling
    if not checks["one_hop_not_saturated"]:
        reasons.append(
            f"mean one-hop accuracy {mean:.4f} exceeds saturation ceiling {ceiling:.4f}"
        )

    cv = _coefficient_of_variation(holdout)
    maximum_cv = float(thresholds["maximum_seed_coefficient_of_variation"])
    checks["reliable_across_seeds"] = cv is not None and cv <= maximum_cv
    if not checks["reliable_across_seeds"]:
        reasons.append(
            f"seed coefficient of variation {cv!r} exceeds {maximum_cv:.4f} "
            "or cannot be estimated"
        )

    gaps = [
        seen_value - holdout_value
        for seen_value, holdout_value in zip(seen, holdout, strict=False)
    ]
    maximum_gap = float(thresholds["maximum_generalization_gap"])
    checks["generalization_gap_acceptable"] = bool(gaps) and max(gaps) <= maximum_gap
    if not checks["generalization_gap_acceptable"]:
        reasons.append(
            f"maximum seen-minus-holdout gap "
            f"{max(gaps) if gaps else math.inf:.4f} exceeds {maximum_gap:.4f}"
        )

    verdict = "GO" if all(checks.values()) else "NO_GO"
    terminal_saturation = not checks["one_hop_not_saturated"]
    if verdict == "GO":
        decision = "SELECT_CAPACITY_RUNG"
        next_action = "freeze this rung and execute the registered qualification grid"
    elif terminal_saturation:
        decision = "REJECT_SATURATED_BAND"
        next_action = "return to task design under a new registration"
    elif rung_index + 1 < rung_count:
        decision = "ADVANCE_CAPACITY_RUNG"
        next_action = "run the same probe on the next registered rung"
    else:
        decision = "REJECT_TASK_FAMILY"
        next_action = "return to task design; no unregistered fourth rung is permitted"

    return {
        "schema_version": V5_PROBE_REPORT_SCHEMA_VERSION,
        "stage": "prerequisite_probe",
        "rung_id": rung_id,
        "rung_index": rung_index,
        "rung_count": rung_count,
        "verdict": verdict,
        "decision": decision,
        "checks": dict(sorted(checks.items())),
        "reasons": reasons,
        "metrics": {
            "per_seed_holdout_accuracy": holdout,
            "per_seed_seen_accuracy": seen,
            "weakest_holdout_accuracy": weakest,
            "mean_holdout_accuracy": mean,
            "family_null": float(family_null),
            "weakest_margin_over_null": margin,
            "seed_coefficient_of_variation": cv,
            "per_seed_generalization_gap": gaps,
        },
        "development_only": bool(development_only),
        "authorization": {
            "qualification_grid": verdict == "GO",
            "mechanism_test": False,
            "rineya_retrofit": False,
        },
        "next_action": next_action,
        "claim_scope": (
            "apparatus-only one-hop capacity qualification; no comparison among "
            "TGSA mechanisms and no evidence for or against TGSA"
        ),
    }


def hop_count_modulation_diagnostic(
    prediction_rows: Iterable[Mapping[str, object]],
    *,
    family: str = "pointer_chase",
    low_hop: int = 1,
    high_hop: int = 3,
    minimum_examples_per_cell: int = 20,
) -> dict[str, object]:
    """Compare hop depths only within context buckets.

    No pooled hop contrast is produced.  This prevents the node-count shift
    induced by rejection sampling from being reintroduced through aggregation.
    """

    cells: dict[tuple[str, int], list[bool]] = {}
    for row in prediction_rows:
        if str(row.get("family")) != family:
            continue
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        hop = int(metadata.get("hops", 0))
        bucket = str(metadata.get("context_bucket", ""))
        if hop not in {low_hop, high_hop} or not bucket:
            continue
        cells.setdefault((bucket, hop), []).append(bool(row.get("correct", False)))

    per_bucket: dict[str, object] = {}
    eligible: list[str] = []
    directional: list[bool] = []
    for bucket in sorted({key[0] for key in cells}):
        low = cells.get((bucket, low_hop), [])
        high = cells.get((bucket, high_hop), [])
        eligible_cell = (
            len(low) >= minimum_examples_per_cell
            and len(high) >= minimum_examples_per_cell
        )
        low_accuracy = (
            sum(1 for value in low if value) / len(low) if low else None
        )
        high_accuracy = (
            sum(1 for value in high if value) / len(high) if high else None
        )
        difference = (
            float(low_accuracy) - float(high_accuracy)
            if low_accuracy is not None and high_accuracy is not None
            else None
        )
        supports = bool(eligible_cell and difference is not None and difference > 0.0)
        if eligible_cell:
            eligible.append(bucket)
            directional.append(supports)
        per_bucket[bucket] = {
            f"hop_{low_hop}_examples": len(low),
            f"hop_{low_hop}_accuracy": low_accuracy,
            f"hop_{high_hop}_examples": len(high),
            f"hop_{high_hop}_accuracy": high_accuracy,
            "difference_low_minus_high": difference,
            "eligible": eligible_cell,
            "supports_registered_direction": supports,
        }

    return {
        "family": family,
        "low_hop": low_hop,
        "high_hop": high_hop,
        "minimum_examples_per_cell": minimum_examples_per_cell,
        "per_context_bucket": per_bucket,
        "eligible_context_buckets": eligible,
        "all_eligible_buckets_support_direction": bool(eligible) and all(directional),
        "gating": False,
        "pooled_comparison_permitted": False,
        "interpretation": (
            "diagnostic only: within a context bucket, lower-hop accuracy is "
            "expected to exceed higher-hop accuracy. Failure weakens the task-"
            "composition interpretation but cannot erase or create a result."
        ),
    }


def competence_thresholds_from_plan(
    plan: Mapping[str, object],
) -> CompetenceThresholds:
    qualification = _as_mapping(plan["qualification"], field="qualification")
    thresholds = _as_mapping(
        qualification["thresholds"], field="qualification.thresholds"
    )
    return CompetenceThresholds(
        minimum_margin_over_trivial=float(
            thresholds["minimum_margin_over_trivial"]
        ),
        minimum_absolute_accuracy=float(thresholds["minimum_absolute_accuracy"]),
        maximum_answer_nll_fraction_of_uniform=float(
            thresholds["maximum_answer_nll_fraction_of_uniform"]
        ),
        minimum_distinct_predictions=int(
            thresholds["minimum_distinct_predictions"]
        ),
        maximum_family_accuracy=float(thresholds["maximum_family_accuracy"]),
        maximum_generalization_gap=float(
            thresholds["maximum_generalization_gap"]
        ),
        maximum_seed_coefficient_of_variation=float(
            thresholds["maximum_seed_coefficient_of_variation"]
        ),
        minimum_family_margin_over_null=float(
            thresholds["minimum_family_margin_over_null"]
        ),
        minimum_seed_runs=int(thresholds["minimum_seed_runs"]),
    )


def evaluate_v5_qualification(
    *,
    arms: Sequence[ArmCompetence],
    thresholds: CompetenceThresholds,
    family_null_report: Mapping[str, Mapping[str, object]],
    uniform_answer_nll: float,
    guardrail_minimum_margin_over_null: float,
    canary_metrics: Mapping[str, object],
    hop_diagnostic: Mapping[str, object],
    development_only: bool,
) -> dict[str, object]:
    """Apply the v5 primary gate plus role-specific guardrail semantics."""

    nulls = family_null_values(family_null_report)
    missing = set((*V5_PRIMARY_FAMILIES, *V5_GUARDRAIL_FAMILIES)) - set(nulls)
    if missing:
        raise V5ContractError(f"missing registered family nulls: {sorted(missing)}")
    primary_nulls = {family: nulls[family] for family in V5_PRIMARY_FAMILIES}
    base = evaluate_competence(
        arms,
        thresholds=thresholds,
        trivial_baselines={
            f"{family}_registered_null": null
            for family, null in primary_nulls.items()
        },
        uniform_answer_nll=float(uniform_answer_nll),
        primary_families=V5_PRIMARY_FAMILIES,
        family_null=primary_nulls,
    )
    checks = cast(dict[str, bool], dict(base["checks"]))
    reasons = cast(list[str], list(base["reasons"]))

    guardrail_detail: dict[str, object] = {}
    guardrail_ok = True
    for family in V5_GUARDRAIL_FAMILIES:
        values = {
            arm.arm: float(arm.per_family_accuracy.get(family, 0.0))
            for arm in arms
        }
        weakest_arm, weakest_value = min(values.items(), key=lambda item: item[1])
        null = nulls[family]
        margin = weakest_value - null
        passed = margin >= guardrail_minimum_margin_over_null
        guardrail_ok = guardrail_ok and passed
        guardrail_detail[family] = {
            "per_arm_accuracy": values,
            "weakest_arm": weakest_arm,
            "weakest_accuracy": weakest_value,
            "registered_null": null,
            "margin_over_null": margin,
            "required_margin": guardrail_minimum_margin_over_null,
            "pass": passed,
        }
        if not passed:
            reasons.append(
                f"guardrail family {family!r}: weakest arm {weakest_arm} clears "
                f"its own null by only {margin:+.4f}, below the required "
                f"{guardrail_minimum_margin_over_null:+.4f}"
            )
    checks["guardrail_families_beat_their_own_null"] = guardrail_ok

    verdict = "GO" if all(checks.values()) else "NO_GO"
    return {
        "schema_version": V5_QUALIFICATION_REPORT_SCHEMA_VERSION,
        "stage": "qualification_cell",
        "verdict": verdict,
        "decision": (
            "QUALIFIED_CELL" if verdict == "GO" else "UNQUALIFIED_CELL"
        ),
        "checks": dict(sorted(checks.items())),
        "reasons": reasons,
        "thresholds": thresholds.to_dict(),
        "family_null_report": copy.deepcopy(dict(family_null_report)),
        "primary_gate_detail": base["detail"],
        "guardrail_detail": guardrail_detail,
        "canary_metrics": copy.deepcopy(dict(canary_metrics)),
        "hop_count_modulation": copy.deepcopy(dict(hop_diagnostic)),
        "arms": [arm.to_dict() for arm in arms],
        "development_only": bool(development_only),
        "authorization": {
            # One cell never authorizes the mechanism matrix.  Selection requires
            # a frozen band and the complete registered exposure grid.
            "mechanism_test": False,
            "rineya_retrofit": False,
        },
        "claim_scope": (
            "apparatus qualification only; establishes whether this capacity and "
            "exposure cell can support an interpretable future mechanism contrast"
        ),
    }


def _report_digest(report: Mapping[str, object]) -> str:
    return canonical_sha256(report)


def freeze_band_after_probe(
    *,
    plan: Mapping[str, object],
    probe_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select the first passing registered rung and emit a frozen-band record."""

    validated = validate_v5_plan(source_plan(plan))
    if bool(validated["development_only"]):
        raise V5ContractError("development-only probe reports cannot freeze a real band")
    ladder = capacity_ladder(validated)
    by_rung = {str(report.get("rung_id")): report for report in probe_reports}
    selected: CapacityRung | None = None
    ordered_reports: list[dict[str, object]] = []
    for index, rung in enumerate(ladder):
        report = by_rung.get(rung.rung_id)
        if report is None:
            raise V5ContractError(
                f"probe report for rung {rung.rung_id} is missing; rung selection "
                "must follow the registered order"
            )
        if report.get("schema_version") != V5_PROBE_REPORT_SCHEMA_VERSION:
            raise V5ContractError(f"rung {rung.rung_id} has an incompatible probe report")
        if bool(report.get("development_only")):
            raise V5ContractError("development-only report cannot freeze a band")
        # Symmetric with `select_qualified_exposure`: the band a whole 36-cell
        # grid is run against must be frozen from probe evidence registered
        # against this exact plan, not merely from a well-shaped report.
        if report.get("plan_sha256") != plan_sha256(validated):
            raise V5ContractError(
                f"rung {rung.rung_id} probe report was produced from a different "
                "v5 plan; a band may only be frozen from reports registered "
                "against this plan"
            )
        ordered_reports.append(
            {
                "rung_id": rung.rung_id,
                "verdict": report.get("verdict"),
                "decision": report.get("decision"),
                "sha256": _report_digest(report),
            }
        )
        if report.get("verdict") == "GO":
            selected = rung
            # A later rung may not be inspected after the first registered pass.
            if any(other.rung_id in by_rung for other in ladder[index + 1 :]):
                raise V5ContractError(
                    "probe reports exist after the first passing rung; the "
                    "registered ladder must stop at first pass"
                )
            break
        if report.get("decision") in {
            "REJECT_SATURATED_BAND",
            "REJECT_TASK_FAMILY",
        }:
            break

    if selected is None:
        terminal = ordered_reports[-1] if ordered_reports else None
        return {
            "schema_version": V5_SELECTION_REPORT_SCHEMA_VERSION,
            "selection_kind": "capacity_rung",
            "verdict": "NO_GO",
            "decision": (
                terminal["decision"] if terminal is not None else "INCOMPLETE"
            ),
            "plan_sha256": plan_sha256(validated),
            "probe_reports": ordered_reports,
            "selected_rung": None,
            "authorization": {
                "qualification_grid": False,
                "mechanism_test": False,
                "rineya_retrofit": False,
            },
            "claim_scope": "apparatus-only capacity selection",
        }

    return {
        "schema_version": V5_FROZEN_BAND_SCHEMA_VERSION,
        "registration_status": "frozen_after_probe",
        "plan_sha256": plan_sha256(validated),
        "study_id": validated["study_id"],
        "selected_rung": selected.to_dict(),
        "probe_reports": ordered_reports,
        "qualification_exposure_grid": [
            cell.to_dict() for cell in exposure_grid(validated)
        ],
        "selection_policy": list(SELECTION_POLICY),
        "mechanism_test_authorized": False,
        "authorization": {
            "qualification_grid": True,
            "mechanism_test": False,
            "rineya_retrofit": False,
        },
        "claim_scope": (
            "frozen apparatus band after one-hop capacity qualification; no "
            "mechanism comparison has been run"
        ),
    }


def _qualification_sort_key(
    exposure: ExposureCell,
) -> tuple[int, int, str]:
    return (
        exposure.sample_presentations,
        -exposure.unique_train_examples,
        exposure.exposure_id,
    )


def select_qualified_exposure(
    *,
    plan: Mapping[str, object],
    frozen_band: Mapping[str, object],
    qualification_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select a qualified exposure only after the complete frozen grid exists."""

    validated = validate_v5_plan(source_plan(plan))
    # This function is what emits AUTHORIZE_MTP_FREE_MECHANISM_STUDY, so it must
    # establish where its evidence came from. It previously validated the frozen
    # band's `plan_sha256` but accepted any report carrying a matching
    # `schema_version`, `rung_id` and `exposure_id` -- so a development-only or
    # foreign-plan report could reach the authorization. Every other layer of
    # this apparatus is hash-pinned and fail-closed; this one was not.
    # `freeze_band_after_probe` already refuses a development-only plan and
    # development-only reports, and the two are now symmetric.
    if bool(validated["development_only"]):
        raise V5ContractError(
            "development-only plan cannot select a qualified exposure or authorize "
            "a mechanism study"
        )
    if frozen_band.get("schema_version") != V5_FROZEN_BAND_SCHEMA_VERSION:
        raise V5ContractError("frozen_band has an incompatible schema")
    if frozen_band.get("plan_sha256") != plan_sha256(validated):
        raise V5ContractError("frozen_band was produced from a different plan")
    if frozen_band.get("registration_status") != "frozen_after_probe":
        raise V5ContractError("frozen_band is not registered after a passing probe")
    selected_rung = _as_mapping(
        frozen_band.get("selected_rung"), field="frozen_band.selected_rung"
    )
    selected_rung_id = str(selected_rung["id"])

    reports_by_exposure: dict[str, Mapping[str, object]] = {}
    for report in qualification_reports:
        if report.get("schema_version") != V5_QUALIFICATION_REPORT_SCHEMA_VERSION:
            raise V5ContractError("qualification report has an incompatible schema")
        if bool(report.get("development_only")):
            raise V5ContractError(
                "development-only qualification report cannot authorize a mechanism "
                "study"
            )
        if report.get("plan_sha256") != plan_sha256(validated):
            raise V5ContractError(
                "qualification report was produced from a different v5 plan; "
                "selection may only read reports registered against this plan"
            )
        if str(report.get("rung_id")) != selected_rung_id:
            raise V5ContractError("qualification report uses a non-frozen capacity rung")
        exposure_id = str(report.get("exposure_id", ""))
        if not exposure_id or exposure_id in reports_by_exposure:
            raise V5ContractError("qualification exposure ids must be non-empty and unique")
        reports_by_exposure[exposure_id] = report

    expected = {cell.exposure_id for cell in exposure_grid(validated)}
    missing = expected - set(reports_by_exposure)
    extra = set(reports_by_exposure) - expected
    if missing or extra:
        raise V5ContractError(
            f"qualification grid is incomplete or contains extras; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    qualified = [
        cell
        for cell in exposure_grid(validated)
        if reports_by_exposure[cell.exposure_id].get("verdict") == "GO"
    ]
    selected = min(qualified, key=_qualification_sort_key) if qualified else None
    report_index = [
        {
            "exposure_id": cell.exposure_id,
            "verdict": reports_by_exposure[cell.exposure_id].get("verdict"),
            "sha256": _report_digest(reports_by_exposure[cell.exposure_id]),
        }
        for cell in exposure_grid(validated)
    ]
    authorized = selected is not None
    return {
        "schema_version": V5_SELECTION_REPORT_SCHEMA_VERSION,
        "selection_kind": "qualification_exposure",
        "verdict": "GO" if authorized else "NO_GO",
        "decision": (
            "AUTHORIZE_MTP_FREE_MECHANISM_STUDY"
            if authorized
            else "REJECT_REGISTERED_REGIME_FAMILY"
        ),
        "plan_sha256": plan_sha256(validated),
        "frozen_band_sha256": canonical_sha256(frozen_band),
        "selected_rung_id": selected_rung_id,
        "selection_policy": list(SELECTION_POLICY),
        "qualification_reports": report_index,
        "selected_exposure": None if selected is None else selected.to_dict(),
        "authorization": {
            "qualification_grid": True,
            "mechanism_test": authorized,
            # Rineya remains downstream of a positive mechanism result.
            "rineya_retrofit": False,
        },
        "claim_scope": (
            "authorizes only a separately preregistered, MTP-free scratch-model "
            "mechanism study; it is not evidence for TGSA and does not authorize "
            "a Rineya retrofit"
        ),
    }
