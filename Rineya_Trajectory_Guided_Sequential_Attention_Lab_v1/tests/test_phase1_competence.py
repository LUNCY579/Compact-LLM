"""Tests for the v4 competence gate.

The gate exists because v3 produced an uninformative result: three arms that had
memorized the training set were compared, and nothing in the registered design
stopped it. Each test below pins one way that failure could recur.

The most important are the refusal tests: the confirmatory runner must be unable
to launch when the gate says NO_GO, when the report is missing, or when it is
malformed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tgsa.phase1_competence import (
    COMPETENCE_SCHEMA_VERSION,
    ArmCompetence,
    CompetenceThresholds,
    evaluate_competence,
    load_verdict,
    require_go,
)

UNIFORM_NLL = math.log(32)
PRIMARY = ("pointer_chase", "state_update")
BASELINES = {"uniform": 0.0312, "train_majority": 0.0360, "eval_majority": 0.0407}


def _thresholds(**overrides) -> CompetenceThresholds:
    base = dict(
        minimum_margin_over_trivial=0.03,
        minimum_absolute_accuracy=0.07,
        maximum_answer_nll_fraction_of_uniform=0.97,
        minimum_distinct_predictions=4,
        maximum_family_accuracy=0.95,
        maximum_generalization_gap=0.10,
        maximum_seed_sd=0.04,
        minimum_seed_runs=3,
    )
    base.update(overrides)
    return CompetenceThresholds(**base)


def _competent_arm(name: str, accuracy: float = 0.094, **overrides) -> ArmCompetence:
    base = dict(
        arm=name,
        holdout_accuracy=accuracy,
        seen_accuracy=accuracy + 0.01,
        answer_nll=3.18,
        distinct_predictions=30,
        per_family_accuracy={"pointer_chase": accuracy, "state_update": accuracy},
        per_seed_holdout_accuracy=[accuracy - 0.004, accuracy, accuracy + 0.004],
    )
    base.update(overrides)
    return ArmCompetence(**base)


def _arms(**overrides) -> list[ArmCompetence]:
    return [
        _competent_arm("cascade", **overrides.get("cascade", {})),
        _competent_arm("history_control", **overrides.get("history_control", {})),
        _competent_arm("learned_delta", **overrides.get("learned_delta", {})),
    ]


def _evaluate(arms) -> dict:
    return evaluate_competence(
        arms,
        thresholds=_thresholds(),
        trivial_baselines=BASELINES,
        uniform_answer_nll=UNIFORM_NLL,
        primary_families=PRIMARY,
    )


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_competent_regime_returns_go() -> None:
    report = _evaluate(_arms())
    assert report["verdict"] == "GO"
    assert all(report["checks"].values())
    assert report["reasons"] == []


# --------------------------------------------------------------------------
# Each way v3 could recur
# --------------------------------------------------------------------------


def test_the_actual_v3_regime_is_refused() -> None:
    """The exact v3 numbers must not pass. This is the whole point of the gate.

    v3 measured: holdout ~0.045 against a 0.0407 majority baseline, seen ~0.235,
    answer-NLL ~4.03 against ln(32)=3.466.
    """

    v3_arms = [
        ArmCompetence(
            arm=name,
            holdout_accuracy=0.045,
            seen_accuracy=0.235,
            answer_nll=4.03,
            distinct_predictions=33,
            per_family_accuracy={"pointer_chase": 0.044, "state_update": 0.028},
            per_seed_holdout_accuracy=[0.043, 0.045, 0.047],
        )
        for name in ("cascade", "history_control", "learned_delta")
    ]
    report = _evaluate(v3_arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["all_arms_beat_trivial"] is False
    assert report["checks"]["answer_nll_beats_uniform"] is False
    assert report["checks"]["generalization_gap_acceptable"] is False


def test_memorization_alone_is_refused() -> None:
    """High seen accuracy must never substitute for generalization."""

    arms = _arms(cascade={"seen_accuracy": 0.40})
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["generalization_gap_acceptable"] is False
    assert any("memorization" in r for r in report["reasons"])


def test_accuracy_above_baseline_but_nll_worse_than_uniform_is_refused() -> None:
    """v3's exact trap: marginally above baseline while confidently wrong."""

    arms = _arms()
    arms[0] = _competent_arm("cascade", answer_nll=UNIFORM_NLL + 0.5)
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["answer_nll_beats_uniform"] is False


def test_one_weak_arm_blocks_the_whole_regime() -> None:
    """Arm-blind: the WEAKEST arm must clear the bar, not the average.

    Otherwise a regime could qualify because the candidate mechanism alone
    happens to work, which would bias the confirmatory comparison.
    """

    arms = _arms()
    arms[2] = _competent_arm("learned_delta", accuracy=0.30)  # the candidate is strong
    arms[0] = _competent_arm("cascade", accuracy=0.045)       # a control is at floor
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["detail"]["weakest_arm"] == "cascade"


def test_a_dead_family_is_not_hidden_by_a_healthy_one() -> None:
    """v3's state_update sat below chance while pointer_chase was near baseline."""

    arms = _arms()
    for index, arm in enumerate(arms):
        arms[index] = _competent_arm(
            arm.arm, per_family_accuracy={"pointer_chase": 0.15, "state_update": 0.03}
        )
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["every_primary_family_competent"] is False


def test_a_constant_predictor_is_refused() -> None:
    arms = _arms()
    arms[1] = _competent_arm("history_control", distinct_predictions=1)
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["predictions_are_input_dependent"] is False


def test_a_saturated_task_is_refused() -> None:
    """A ceiling is as uninformative as a floor: no room for a decrement."""

    arms = [
        _competent_arm(name, accuracy=0.99,
                       per_family_accuracy={"pointer_chase": 0.99, "state_update": 0.99})
        for name in ("cascade", "history_control", "learned_delta")
    ]
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["no_primary_family_saturated"] is False


def test_margin_is_measured_against_the_strongest_trivial_baseline() -> None:
    """Beating `uniform` while losing to `majority` must not qualify."""

    arms = [
        _competent_arm(name, accuracy=0.035,
                       per_family_accuracy={"pointer_chase": 0.035, "state_update": 0.035})
        for name in ("cascade", "history_control", "learned_delta")
    ]
    report = _evaluate(arms)
    assert report["detail"]["strongest_trivial_baseline"] == max(BASELINES.values())
    assert report["verdict"] == "NO_GO"


# --------------------------------------------------------------------------
# The runner must refuse to launch
# --------------------------------------------------------------------------


def test_require_go_accepts_a_frozen_go_report(tmp_path: Path) -> None:
    path = tmp_path / "competence.json"
    path.write_text(json.dumps(_evaluate(_arms())), encoding="utf-8")
    assert require_go(path)["verdict"] == "GO"


def test_require_go_refuses_a_no_go_report(tmp_path: Path) -> None:
    arms = _arms()
    arms[0] = _competent_arm("cascade", accuracy=0.04)
    path = tmp_path / "competence.json"
    path.write_text(json.dumps(_evaluate(arms)), encoding="utf-8")
    with pytest.raises(ValueError, match="NO_GO"):
        require_go(path)


def test_require_go_refuses_a_missing_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="competence report not found"):
        require_go(tmp_path / "absent.json")


def test_require_go_refuses_a_malformed_report(tmp_path: Path) -> None:
    path = tmp_path / "competence.json"
    path.write_text(json.dumps({"schema_version": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported competence report schema"):
        require_go(path)


def test_require_go_refuses_a_report_with_no_verdict(tmp_path: Path) -> None:
    path = tmp_path / "competence.json"
    path.write_text(
        json.dumps({"schema_version": COMPETENCE_SCHEMA_VERSION, "verdict": "MAYBE"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no usable verdict"):
        require_go(path)


def test_evaluate_requires_at_least_one_arm() -> None:
    with pytest.raises(ValueError, match="at least one arm"):
        evaluate_competence(
            [], thresholds=_thresholds(), trivial_baselines=BASELINES,
            uniform_answer_nll=UNIFORM_NLL, primary_families=PRIMARY,
        )


def test_report_never_claims_anything_about_the_mechanism() -> None:
    report = _evaluate(_arms())
    scope = str(report["claim_scope"]).lower()
    assert "says nothing about the mechanism" in scope


# --------------------------------------------------------------------------
# The confirmatory runner itself must refuse to launch
# --------------------------------------------------------------------------


def test_confirmatory_runner_refuses_to_launch_on_no_go(tmp_path: Path) -> None:
    """End-to-end: run_study_matrix must abort before touching any data.

    This is the check that would have prevented the v3 spend.
    """

    from tgsa.phase1_runner import run_study_matrix

    lab = Path(__file__).resolve().parents[1]
    study = json.loads((lab / "configs" / "phase1_identifiability_v3.json").read_text(encoding="utf-8"))
    study["competence_report"] = "competence.json"
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(study), encoding="utf-8")

    arms = _arms()
    arms[0] = _competent_arm("cascade", accuracy=0.04)  # a control at floor
    (tmp_path / "competence.json").write_text(json.dumps(_evaluate(arms)), encoding="utf-8")

    with pytest.raises(ValueError, match="NO_GO"):
        run_study_matrix(
            study_config_path=config_path,
            dataset_dir=tmp_path / "dataset",
            output_root=tmp_path / "study",
            repo_root=lab.parent,
            phase="train",
        )
    assert not (tmp_path / "study").exists(), "runner must abort before creating an output root"


def test_confirmatory_runner_refuses_a_missing_competence_report(tmp_path: Path) -> None:
    from tgsa.phase1_runner import run_study_matrix

    lab = Path(__file__).resolve().parents[1]
    study = json.loads((lab / "configs" / "phase1_identifiability_v3.json").read_text(encoding="utf-8"))
    study["competence_report"] = "absent.json"
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(study), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="competence report not found"):
        run_study_matrix(
            study_config_path=config_path,
            dataset_dir=tmp_path / "dataset",
            output_root=tmp_path / "study",
            repo_root=lab.parent,
            phase="train",
        )


def test_a_smoke_config_is_not_blocked_by_the_gate(tmp_path: Path) -> None:
    """The gate applies to confirmatory studies only; smoke must still run."""

    from tgsa.phase1_competence import require_go

    lab = Path(__file__).resolve().parents[1]
    study = json.loads((lab / "configs" / "phase1_identifiability_v3_smoke.json").read_text(encoding="utf-8"))
    assert study["confirmatory"] is False
    # No competence_report key and confirmatory False -> the gate is not consulted.
    assert "competence_report" not in study


# --------------------------------------------------------------------------
# Reliability: the v4 pilot failure this check was added for
# --------------------------------------------------------------------------


def test_a_bimodal_regime_is_refused_despite_a_healthy_mean() -> None:
    """The v4 `easy_50k_long` regime, with its real numbers.

    Every arm's MEAN cleared every other check — 0.263 against a 0.040 baseline,
    NLL 2.06 against ln(32)=3.47, memorization gap 0.068. But the individual runs
    were 0.117 and 0.410 for the same arm: a training phase transition that some
    seeds completed and others were truncated mid-descent. Without this check the
    gate returns GO on a regime with sigma_a ~ 0.2.
    """

    arms = [
        ArmCompetence(
            arm="cascade", holdout_accuracy=0.2633, seen_accuracy=0.3325,
            answer_nll=2.0532, distinct_predictions=32,
            per_family_accuracy={"pointer_chase": 0.25, "state_update": 0.24},
            per_seed_holdout_accuracy=[0.1167, 0.4100, 0.2600],
        ),
        ArmCompetence(
            arm="history_control", holdout_accuracy=0.2637, seen_accuracy=0.3325,
            answer_nll=2.0628, distinct_predictions=32,
            per_family_accuracy={"pointer_chase": 0.25, "state_update": 0.24},
            per_seed_holdout_accuracy=[0.3993, 0.1280, 0.2600],
        ),
        ArmCompetence(
            arm="learned_delta", holdout_accuracy=0.3737, seen_accuracy=0.4405,
            answer_nll=1.6885, distinct_predictions=32,
            per_family_accuracy={"pointer_chase": 0.36, "state_update": 0.35},
            per_seed_holdout_accuracy=[0.3167, 0.4307, 0.3737],
        ),
    ]
    report = _evaluate(arms)

    # Everything except reliability passes -- that is precisely the danger.
    assert report["checks"]["all_arms_beat_trivial"] is True
    assert report["checks"]["answer_nll_beats_uniform"] is True
    assert report["checks"]["generalization_gap_acceptable"] is True
    assert report["checks"]["competence_is_reliable_across_seeds"] is False
    assert report["verdict"] == "NO_GO"
    # The 0.1167 run still clears the 0.07 absolute floor, so the dispersion
    # bound is the load-bearing one here -- a floor alone would not have caught it.
    assert all(v >= 0.07 for arm in arms for v in arm.per_seed_holdout_accuracy)
    assert sum("between-seed SD" in r for r in report["reasons"]) == 3  # all three arms
    assert report["detail"]["per_seed_holdout_accuracy"]["cascade"]["worst"] == 0.1167


def test_a_single_truncated_run_blocks_the_regime() -> None:
    """One unconverged seed is enough: the mean would hide it."""

    arms = _arms()
    arms[0] = _competent_arm("cascade", per_seed_holdout_accuracy=[0.094, 0.094, 0.030])
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["competence_is_reliable_across_seeds"] is False


def test_too_few_development_runs_fails_closed() -> None:
    """Two seeds cannot establish that a third would behave the same way."""

    arms = _arms()
    arms[1] = _competent_arm("history_control", per_seed_holdout_accuracy=[0.094, 0.096])
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert any("at least 3 are required" in r for r in report["reasons"])


def test_unrecorded_per_seed_evidence_fails_closed() -> None:
    """A gate cannot certify reliability it has no data for."""

    arms = _arms()
    arms[2] = _competent_arm("learned_delta", per_seed_holdout_accuracy=[])
    report = _evaluate(arms)
    assert report["verdict"] == "NO_GO"
    assert report["checks"]["competence_is_reliable_across_seeds"] is False


def test_seed_sd_threshold_is_the_registered_robustness_value() -> None:
    """The bound is traceable to the power analysis, not chosen for convenience."""

    assert CompetenceThresholds(
        minimum_margin_over_trivial=0.03, minimum_absolute_accuracy=0.07
    ).maximum_seed_sd == 0.04


def test_a_v1_report_is_rejected_on_load(tmp_path: Path) -> None:
    """Reports predating the reliability check must not satisfy the gate."""

    path = tmp_path / "competence.json"
    stale = _evaluate(_arms())
    stale["schema_version"] = "tgsa.phase1-competence-gate.v1"
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported competence report schema"):
        require_go(path)
