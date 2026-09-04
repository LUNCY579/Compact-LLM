"""Contracts for the executable TGSA Phase 1 v5 qualification path."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tgsa.phase1_competence import ArmCompetence
from tgsa.phase1_data import EXAMPLE_SCHEMA_VERSION, Phase1Example, SyntheticVocabulary
from tgsa.phase1_v5 import (
    V5ContractError,
    V5_PROBE_REPORT_SCHEMA_VERSION,
    V5_QUALIFICATION_REPORT_SCHEMA_VERSION,
    allocate_family_counts,
    competence_thresholds_from_plan,
    evaluate_prerequisite_probe,
    evaluate_v5_qualification,
    exposure_grid,
    freeze_band_after_probe,
    hop_count_modulation_diagnostic,
    load_v5_plan,
    plan_sha256,
    probe_exposure,
    registered_family_null_report,
    resolve_rung,
    select_qualified_exposure,
    stage_dataset_config,
)

LAB_ROOT = Path(__file__).resolve().parents[1]
FULL_PLAN = LAB_ROOT / "configs" / "phase1_v5_plan.json"
SMOKE_PLAN = LAB_ROOT / "configs" / "phase1_v5_smoke.json"
PlanMutation = Callable[[dict[str, Any]], None]


def _load_raw(path: Path = FULL_PLAN) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _pointer_example(
    *,
    identifier: str,
    split: str,
    answer_domain: int,
    start_domain: int = 2,
    last_value_domain: int = 1,
) -> Phase1Example:
    vocab = SyntheticVocabulary()
    tokens = (
        vocab.BOS,
        vocab.TASK_POINTER,
        vocab.PAIR,
        vocab.key(0),
        vocab.value(last_value_domain),
        vocab.QUERY,
        vocab.key(start_domain),
        vocab.hop(1),
        vocab.ANSWER,
        vocab.value(answer_domain),
        vocab.EOS,
    )
    return Phase1Example(
        schema_version=EXAMPLE_SCHEMA_VERSION,
        id=identifier,
        split=split,
        family="pointer_chase",
        stratum="pointer_chase:h1:short",
        tokens=tokens,
        answer_token=vocab.value(answer_domain),
        answer_input_index=8,
        latent_signature=f"signature-{identifier}",
        metadata={
            "node_count": 32,
            "hops": 1,
            "context_bucket": "short",
            "trivial": False,
        },
    )


def _probe(
    *,
    rung: str,
    index: int,
    count: int = 3,
    holdout: tuple[float, ...] = (0.50, 0.52, 0.51),
) -> dict[str, object]:
    return evaluate_prerequisite_probe(
        rung_id=rung,
        rung_index=index,
        rung_count=count,
        per_seed_holdout_accuracy=holdout,
        per_seed_seen_accuracy=tuple(value + 0.04 for value in holdout),
        family_null=0.18,
        thresholds={
            "minimum_margin_over_family_null": 0.15,
            "maximum_mean_accuracy": 0.95,
            "maximum_seed_coefficient_of_variation": 0.35,
            "maximum_generalization_gap": 0.10,
            "minimum_seed_runs": 3,
        },
        development_only=False,
    ) | {"plan_sha256": plan_sha256(load_v5_plan(FULL_PLAN))}


def _arm(name: str, *, direct: float = 0.50) -> ArmCompetence:
    return ArmCompetence(
        arm=name,
        holdout_accuracy=0.50,
        seen_accuracy=0.55,
        answer_nll=1.0,
        distinct_predictions=12,
        per_family_accuracy={
            "pointer_chase": 0.50,
            "direct_lookup": direct,
            "state_update": 0.90,
        },
        per_seed_holdout_accuracy=[0.49, 0.50, 0.51],
    )


def _null_report() -> dict[str, dict[str, object]]:
    return {
        family: {
            "examples": 100,
            "components": {"candidate_chance": value},
            "registered_null": value,
            "winning_strategies": ["candidate_chance"],
        }
        for family, value in {
            "pointer_chase": 0.18,
            "direct_lookup": 0.18,
            "state_update": 0.04,
        }.items()
    }


def _enable_confirmatory(value: dict[str, Any]) -> None:
    value["confirmatory"] = True


def _enable_mtp(value: dict[str, Any]) -> None:
    value["mtp"]["enabled"] = True


def _reverse_split_order(value: dict[str, Any]) -> None:
    value["dataset"]["split_generation_order"] = ["train", "dev", "test"]


def _weaken_probe_margin(value: dict[str, Any]) -> None:
    value["probe"]["thresholds"]["minimum_margin_over_family_null"] = 0.149


def _weaken_family_margin(value: dict[str, Any]) -> None:
    value["qualification"]["thresholds"]["minimum_family_margin_over_null"] = 0.029


def test_registered_and_smoke_plans_validate() -> None:
    full = load_v5_plan(FULL_PLAN)
    smoke = load_v5_plan(SMOKE_PLAN)
    assert full["development_only"] is False
    assert [entry["id"] for entry in full["_validated_capacity_ladder"]] == [
        "C1",
        "C2",
        "C3",
    ]
    assert {entry["id"] for entry in full["_validated_exposure_grid"]} == {
        "E1",
        "E2",
        "E3",
        "E4",
    }
    assert smoke["development_only"] is True
    assert smoke["registration_status"] == "development_only"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_enable_confirmatory, "confirmatory"),
        (_enable_mtp, "MTP"),
        (_reverse_split_order, "split_generation_order"),
        (_weaken_probe_margin, "0.150"),
        (_weaken_family_margin, "0.030"),
    ],
)
def test_plan_rejects_semantic_weakening(
    mutation: PlanMutation,
    message: str,
    tmp_path: Path,
) -> None:
    value = copy.deepcopy(_load_raw())
    mutation(value)
    temporary = tmp_path / "invalid_v5_plan.json"
    temporary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(V5ContractError, match=message):
        load_v5_plan(temporary)


def test_family_allocation_preserves_exact_total() -> None:
    counts = allocate_family_counts(
        101,
        {
            "pointer_chase": 0.70,
            "direct_lookup": 0.20,
            "state_update": 0.10,
        },
    )
    assert counts == {
        "direct_lookup": 20,
        "pointer_chase": 71,
        "state_update": 10,
    }
    assert sum(counts.values()) == 101


def test_stage_dataset_configs_preserve_registered_boundaries() -> None:
    plan = load_v5_plan(FULL_PLAN)
    e1, e2, *_ = exposure_grid(plan)
    probe = stage_dataset_config(
        plan,
        stage="probe",
        exposure=probe_exposure(plan),
    )
    assert set(probe["families"]) == {"pointer_chase"}
    assert probe["families"]["pointer_chase"]["min_hops"] == 1
    assert probe["families"]["pointer_chase"]["max_hops"] == 1
    assert probe["split_generation_order"] == ["test", "dev", "train"]

    small = stage_dataset_config(plan, stage="qualification", exposure=e1)
    large = stage_dataset_config(plan, stage="qualification", exposure=e2)
    assert small["seed"] == large["seed"]
    assert small["splits"]["test"] == large["splits"]["test"]
    assert sum(small["splits"]["train"].values()) == 50_000
    assert sum(large["splits"]["train"].values()) == 200_000


def test_registered_null_includes_evaluation_majority() -> None:
    vocab = SyntheticVocabulary()
    train = [
        _pointer_example(identifier="train-a", split="train", answer_domain=4),
        _pointer_example(identifier="train-b", split="train", answer_domain=5),
    ]
    holdout = [
        _pointer_example(
            identifier=f"test-{index}",
            split="test",
            answer_domain=3,
        )
        for index in range(5)
    ]
    report = registered_family_null_report(
        train,
        holdout,
        family_specs={
            "pointer_chase": {
                "min_nodes": 6,
                "max_nodes": 12,
                "min_hops": 1,
                "max_hops": 3,
                "reject_trivial": True,
            }
        },
        vocab=vocab,
    )
    pointer = report["pointer_chase"]
    assert pointer["components"]["evaluation_majority"] == 1.0
    assert pointer["registered_null"] == 1.0
    assert pointer["winning_strategies"] == ["evaluation_majority"]


def test_probe_selects_capacity_without_authorizing_mechanism() -> None:
    result = _probe(rung="C1", index=0)
    assert result["schema_version"] == V5_PROBE_REPORT_SCHEMA_VERSION
    assert result["verdict"] == "GO"
    assert result["decision"] == "SELECT_CAPACITY_RUNG"
    assert result["authorization"]["qualification_grid"] is True
    assert result["authorization"]["mechanism_test"] is False


def test_probe_advances_then_rejects_only_at_registered_boundary() -> None:
    weak = (0.20, 0.21, 0.22)
    first = _probe(rung="C1", index=0, holdout=weak)
    last = _probe(rung="C3", index=2, holdout=weak)
    assert first["decision"] == "ADVANCE_CAPACITY_RUNG"
    assert last["decision"] == "REJECT_TASK_FAMILY"


def test_probe_saturation_is_terminal() -> None:
    result = _probe(
        rung="C1",
        index=0,
        holdout=(0.97, 0.98, 0.99),
    )
    assert result["verdict"] == "NO_GO"
    assert result["decision"] == "REJECT_SATURATED_BAND"


def test_hop_diagnostic_never_pools_context_buckets() -> None:
    rows: list[dict[str, object]] = []
    for bucket, low_hits, high_hits in (
        ("short", 30, 20),
        ("medium", 20, 30),
    ):
        for hop, hits in ((1, low_hits), (3, high_hits)):
            for index in range(40):
                rows.append(
                    {
                        "family": "pointer_chase",
                        "correct": index < hits,
                        "metadata": {
                            "hops": hop,
                            "context_bucket": bucket,
                        },
                    }
                )
    result = hop_count_modulation_diagnostic(rows)
    assert result["pooled_comparison_permitted"] is False
    assert result["gating"] is False
    assert result["per_context_bucket"]["short"][
        "supports_registered_direction"
    ] is True
    assert result["per_context_bucket"]["medium"][
        "supports_registered_direction"
    ] is False
    assert result["all_eligible_buckets_support_direction"] is False


def test_qualification_adds_a_gating_direct_lookup_guardrail() -> None:
    plan = load_v5_plan(FULL_PLAN)
    result = evaluate_v5_qualification(
        arms=[_arm(name) for name in ("cascade", "history_control", "learned_delta")],
        thresholds=competence_thresholds_from_plan(plan),
        family_null_report=_null_report(),
        uniform_answer_nll=math.log(32),
        guardrail_minimum_margin_over_null=0.15,
        canary_metrics={"gating": False},
        hop_diagnostic={"gating": False},
        development_only=False,
    )
    assert result["verdict"] == "GO"
    assert result["checks"]["guardrail_families_beat_their_own_null"] is True
    assert result["authorization"]["mechanism_test"] is False

    failed = evaluate_v5_qualification(
        arms=[
            _arm(name, direct=0.20)
            for name in ("cascade", "history_control", "learned_delta")
        ],
        thresholds=competence_thresholds_from_plan(plan),
        family_null_report=_null_report(),
        uniform_answer_nll=math.log(32),
        guardrail_minimum_margin_over_null=0.15,
        canary_metrics={"gating": False},
        hop_diagnostic={"gating": False},
        development_only=False,
    )
    assert failed["verdict"] == "NO_GO"
    assert failed["checks"]["guardrail_families_beat_their_own_null"] is False


def test_capacity_freeze_stops_at_first_registered_pass() -> None:
    plan = load_v5_plan(FULL_PLAN)
    c1 = _probe(rung="C1", index=0)
    frozen = freeze_band_after_probe(plan=plan, probe_reports=[c1])
    assert frozen["selected_rung"]["id"] == "C1"
    assert frozen["mechanism_test_authorized"] is False
    assert frozen["authorization"]["qualification_grid"] is True

    c2 = _probe(rung="C2", index=1)
    with pytest.raises(V5ContractError, match="after the first passing rung"):
        freeze_band_after_probe(plan=plan, probe_reports=[c1, c2])


def test_exposure_selection_requires_full_grid_and_uses_resource_order() -> None:
    plan = load_v5_plan(FULL_PLAN)
    frozen = freeze_band_after_probe(
        plan=plan,
        probe_reports=[_probe(rung="C1", index=0)],
    )
    reports = []
    for cell in exposure_grid(plan):
        reports.append(
            {
                "schema_version": V5_QUALIFICATION_REPORT_SCHEMA_VERSION,
                "plan_sha256": plan_sha256(plan),
                "development_only": False,
                "rung_id": "C1",
                "exposure_id": cell.exposure_id,
                "verdict": (
                    "GO" if cell.exposure_id in {"E1", "E2", "E3"} else "NO_GO"
                ),
            }
        )
    selected = select_qualified_exposure(
        plan=plan,
        frozen_band=frozen,
        qualification_reports=reports,
    )
    # E1 and E2 use the same presentation budget; E2 wins the registered
    # secondary criterion because it has greater data diversity.
    assert selected["selected_exposure"]["id"] == "E2"
    assert selected["authorization"]["mechanism_test"] is True
    assert selected["authorization"]["rineya_retrofit"] is False

    with pytest.raises(V5ContractError, match="incomplete"):
        select_qualified_exposure(
            plan=plan,
            frozen_band=frozen,
            qualification_reports=reports[:-1],
        )


# ---------------------------------------------------------------------------
# Selection provenance: the function that emits
# AUTHORIZE_MTP_FREE_MECHANISM_STUDY must establish where its evidence came from.
# ---------------------------------------------------------------------------


def _grid_reports(plan, *, verdict: str = "GO", **overrides) -> list[dict[str, object]]:
    reports = []
    for cell in exposure_grid(plan):
        report = {
            "schema_version": V5_QUALIFICATION_REPORT_SCHEMA_VERSION,
            "plan_sha256": plan_sha256(plan),
            "development_only": False,
            "rung_id": "C1",
            "exposure_id": cell.exposure_id,
            "verdict": verdict,
        }
        report.update(overrides)
        reports.append(report)
    return reports


def test_selection_refuses_a_development_only_qualification_report() -> None:
    """A toy report must not be able to authorize the mechanism study.

    `freeze_band_after_probe` already refused development-only reports; this
    function accepted any report with a matching schema, rung and exposure, so
    the two were asymmetric at the point that issues the authorization.
    """

    plan = load_v5_plan(FULL_PLAN)
    frozen = freeze_band_after_probe(plan=plan, probe_reports=[_probe(rung="C1", index=0)])
    with pytest.raises(V5ContractError, match="development-only qualification report"):
        select_qualified_exposure(
            plan=plan,
            frozen_band=frozen,
            qualification_reports=_grid_reports(plan, development_only=True),
        )


def test_selection_refuses_a_report_from_a_different_plan() -> None:
    """Evidence must be registered against the plan being selected for."""

    plan = load_v5_plan(FULL_PLAN)
    frozen = freeze_band_after_probe(plan=plan, probe_reports=[_probe(rung="C1", index=0)])
    with pytest.raises(V5ContractError, match="different v5 plan"):
        select_qualified_exposure(
            plan=plan,
            frozen_band=frozen,
            qualification_reports=_grid_reports(plan, plan_sha256="0" * 64),
        )


def test_selection_refuses_a_development_only_plan() -> None:
    """The smoke plan cannot reach a mechanism-study authorization at all."""

    plan = load_v5_plan(FULL_PLAN)
    frozen = freeze_band_after_probe(plan=plan, probe_reports=[_probe(rung="C1", index=0)])
    smoke = load_v5_plan(SMOKE_PLAN)
    with pytest.raises(V5ContractError, match="development-only plan cannot select"):
        select_qualified_exposure(
            plan=smoke,
            frozen_band=frozen,
            qualification_reports=_grid_reports(plan),
        )


def test_band_freezing_refuses_a_probe_report_from_a_different_plan() -> None:
    """A 36-cell grid must not be run against a band frozen from foreign evidence."""

    plan = load_v5_plan(FULL_PLAN)
    foreign = _probe(rung="C1", index=0) | {"plan_sha256": "0" * 64}
    with pytest.raises(V5ContractError, match="different v5 plan"):
        freeze_band_after_probe(plan=plan, probe_reports=[foreign])


def test_the_cell_identity_binds_the_cublas_workspace_setting() -> None:
    """`configure_runtime` only `setdefault`s CUBLAS_WORKSPACE_CONFIG.

    An inherited value therefore wins and selects different cuBLAS GEMM
    behaviour. Unless it is in the fingerprint, a cell can resume from a
    checkpoint produced under different CUDA arithmetic and still match.
    """

    import torch

    from tgsa.phase1_runner import environment_snapshot
    from tgsa.phase1_v5_runner import _cell_identity

    env = environment_snapshot(torch.device("cpu"))
    assert "cublas_workspace_config" in env

    plan = load_v5_plan(FULL_PLAN)
    identity = _cell_identity(
        plan_hash=plan_sha256(plan),
        stage="qualification",
        rung=resolve_rung(plan, "C1"),
        exposure=exposure_grid(plan)[0],
        dataset_manifest_sha256="0" * 64,
        arm="cascade",
        seed=17,
        git={"head": "0" * 40},
        environment={**env, "cublas_workspace_config": ":4096:8"},
    )
    assert identity["environment"]["cublas_workspace_config"] == ":4096:8"

    other = _cell_identity(
        plan_hash=plan_sha256(plan),
        stage="qualification",
        rung=resolve_rung(plan, "C1"),
        exposure=exposure_grid(plan)[0],
        dataset_manifest_sha256="0" * 64,
        arm="cascade",
        seed=17,
        git={"head": "0" * 40},
        environment={**env, "cublas_workspace_config": ":16:8"},
    )
    assert identity != other, "a different cuBLAS setting must change the fingerprint"
