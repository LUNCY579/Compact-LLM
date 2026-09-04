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
#: A properly specified study gives each primary family its own null. These
#: are the uniform answer space, correct for families with no in-context
#: candidate set. A family whose answer is one of the values shown in the
#: context needs 1/candidates instead -- see the v4 pointer families below.
DEFAULT_FAMILY_NULL = {"pointer_chase": 0.0312, "state_update": 0.0312}


def _thresholds(**overrides) -> CompetenceThresholds:
    base = dict(
        minimum_margin_over_trivial=0.03,
        minimum_absolute_accuracy=0.07,
        maximum_answer_nll_fraction_of_uniform=0.97,
        minimum_distinct_predictions=4,
        maximum_family_accuracy=0.95,
        maximum_generalization_gap=0.10,
        maximum_seed_coefficient_of_variation=0.35,
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


def _evaluate(arms, family_null=None) -> dict:
    return evaluate_competence(
        arms,
        thresholds=_thresholds(),
        trivial_baselines=BASELINES,
        uniform_answer_nll=UNIFORM_NLL,
        primary_families=PRIMARY,
        family_null=DEFAULT_FAMILY_NULL if family_null is None else family_null,
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
    # Every run clears the 0.07 absolute floor, so the floor alone would not have
    # caught this -- the mixture signature is what does.
    assert all(v >= 0.07 for arm in arms for v in arm.per_seed_holdout_accuracy)
    assert sum("coefficient of variation" in r for r in report["reasons"]) == 2
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


def test_the_cv_bound_separates_a_mixture_from_a_merely_noisy_regime() -> None:
    """The real pilot arms, with their real per-seed numbers.

    The two bimodal arms sit near CV 0.7-0.8; every arm that was not a mixture --
    including easy_50k_long's learned_delta, whose runs were both post-transition
    but 0.11 apart -- sits at or below 0.28. The 0.35 bound is a structural
    argument (a balanced two-point mixture has CV ~ 0.6) that the data confirms,
    not a value fitted to the data.
    """

    import statistics

    observed = {
        "easy_50k_long/cascade": [0.1167, 0.4100],           # mixture
        "easy_50k_long/history_control": [0.3993, 0.1280],   # mixture
        "easy_50k_long/learned_delta": [0.3167, 0.4307],     # both converged
        "easy_50k/cascade": [0.1067, 0.0807],
        "easy_50k/history_control": [0.1120, 0.0753],
        "v3_replica/cascade": [0.0447, 0.0493],
    }
    cv = {k: statistics.stdev(v) / statistics.fmean(v) for k, v in observed.items()}
    mixtures = [k for k in cv if k.endswith(("long/cascade", "long/history_control"))]
    assert all(cv[k] > 0.35 for k in mixtures), cv
    assert all(cv[k] < 0.35 for k in cv if k not in mixtures), cv


def test_dispersion_alone_is_not_a_competence_failure() -> None:
    """A spread but all-competent regime is a POWER problem, not a gate failure.

    Rejecting it would be wrong: the remedy is more seeds, and the power audit
    decides how many. The realised SD is exported for exactly that purpose.
    """

    arms = _arms()
    for index, arm in enumerate(arms):
        arms[index] = _competent_arm(
            arm.arm, accuracy=0.30,
            per_seed_holdout_accuracy=[0.24, 0.30, 0.36],  # sd 0.06, cv 0.20
            per_family_accuracy={"pointer_chase": 0.30, "state_update": 0.29},
        )
    report = _evaluate(arms)
    assert report["verdict"] == "GO"
    assert report["checks"]["competence_is_reliable_across_seeds"] is True
    exported = report["detail"]["realised_seed_sd_for_power_audit"]
    assert exported == pytest.approx(0.06, abs=1e-9)
    assert exported > 0.04, "and it exceeds v3's robustness sigma_a, which is the point"


def test_the_realised_sd_is_exported_for_the_power_audit() -> None:
    report = _evaluate(_arms())
    assert "realised_seed_sd_for_power_audit" in report["detail"]
    per_arm = report["detail"]["per_seed_holdout_accuracy"]
    assert report["detail"]["realised_seed_sd_for_power_audit"] == max(
        entry["sd"] for entry in per_arm.values()
    )


def test_a_superseded_schema_is_rejected_on_load(tmp_path: Path) -> None:
    """Reports predating the current checks must not satisfy the gate."""

    path = tmp_path / "competence.json"
    stale = _evaluate(_arms())
    stale["schema_version"] = "tgsa.phase1-competence-gate.v2"
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported competence report schema"):
        require_go(path)


# --------------------------------------------------------------------------
# A rehearsal writes a genuine GO report to a fixed path. It must not launch.
# --------------------------------------------------------------------------


def test_a_development_only_report_is_refused_however_well_formed(tmp_path: Path) -> None:
    """The pipeline rehearsal produces a real GO at toy scale to exercise this
    very code path, and that artifact persists on disk afterwards. Without the
    marker it satisfies every structural check the config builder makes."""

    report = _evaluate(_arms())
    assert report["verdict"] == "GO"
    report["regime"] = {"train_examples": 2400, "steps": 150,
                        "split_generation_order": ["holdout", "train"],
                        "development_only": True}
    path = tmp_path / "competence.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="development_only"):
        require_go(path)


def test_a_real_report_is_not_caught_by_the_marker(tmp_path: Path) -> None:
    report = _evaluate(_arms())
    report["regime"] = {"train_examples": 50_000, "steps": 20_000,
                        "split_generation_order": ["holdout", "train"]}
    path = tmp_path / "competence.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert require_go(path)["verdict"] == "GO"


def test_the_marker_is_checked_before_the_verdict(tmp_path: Path) -> None:
    """A development-only NO_GO should still name the marker, not the verdict,
    so an operator is told the artifact is the wrong kind rather than that the
    regime failed."""

    arms = _arms()
    arms[0] = _competent_arm("cascade", accuracy=0.04)
    report = _evaluate(arms)
    report["regime"] = {"development_only": True}
    path = tmp_path / "competence.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="development_only"):
        require_go(path)


def test_the_runner_accepts_a_marked_report_only_for_a_marked_study(tmp_path: Path) -> None:
    """The two ends of the chain must agree, or the marker is launderable."""

    from tgsa.phase1_runner import run_study_matrix

    lab = Path(__file__).resolve().parents[1]
    report = _evaluate(_arms())
    report["regime"] = {"train_examples": 2400, "steps": 150,
                        "split_generation_order": ["holdout", "train"],
                        "development_only": True}
    (tmp_path / "competence.json").write_text(json.dumps(report), encoding="utf-8")

    study = json.loads((lab / "configs" / "phase1_identifiability_v3.json").read_text(encoding="utf-8"))
    study["competence_report"] = "competence.json"
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(study), encoding="utf-8")

    # Unmarked study + marked report -> refused.
    with pytest.raises(ValueError, match="development_only"):
        run_study_matrix(
            study_config_path=config_path, dataset_dir=tmp_path / "dataset",
            output_root=tmp_path / "study", repo_root=lab.parent, phase="train",
        )
    assert not (tmp_path / "study").exists()


# --------------------------------------------------------------------------
# Each primary family against ITS OWN null
# --------------------------------------------------------------------------
#
# The v4 measurement that forced this check. `pointer_chase` answers are one of
# the values shown in the context, so a model that has learned only "the answer
# is one of these" scores 1/node_count. On the registered holdout that is 0.1558
# for pointer_chase and 0.1587 for direct_lookup. v4 measured 0.147-0.177 on
# both -- at chance -- and the gate certified them, because it compared against a
# global 0.070 floor and a global 0.0377 majority baseline.

V4_POINTER_NULL = {"pointer_chase": 0.1558, "direct_lookup": 0.1587}


def _null_evaluate(arms, family_null=None):
    """Deliberately passes family_null straight through, including None, so the
    fail-closed path can be tested."""

    return evaluate_competence(
        arms, thresholds=_thresholds(), trivial_baselines=BASELINES,
        uniform_answer_nll=UNIFORM_NLL, primary_families=PRIMARY,
        family_null=family_null,
    )


def test_the_v4_pointer_families_are_refused_against_their_own_null() -> None:
    """The exact numbers the old gate passed."""

    arms = [
        ArmCompetence(
            arm=name, holdout_accuracy=0.2244, seen_accuracy=0.2582,
            answer_nll=2.086, distinct_predictions=32,
            per_family_accuracy={"pointer_chase": value, "state_update": 0.36},
            per_seed_holdout_accuracy=[0.22, 0.224, 0.229],
        )
        for name, value in (("cascade", 0.1590), ("history_control", 0.1703),
                            ("learned_delta", 0.1733))
    ]
    report = _null_evaluate(arms, V4_POINTER_NULL)
    assert report["checks"]["every_primary_family_beats_its_own_null"] is False
    assert any("at or near chance" in r for r in report["reasons"])
    detail = report["detail"]["family_null"]["pointer_chase"]
    assert detail["margin_over_null"] < 0.01


def test_the_old_global_floor_would_still_have_passed_them() -> None:
    """Establishes that the new check is doing work the old ones did not.

    If the global checks also rejected these numbers, the new check would be
    redundant rather than load-bearing.
    """

    arms = [
        ArmCompetence(
            arm=name, holdout_accuracy=0.2244, seen_accuracy=0.2582,
            answer_nll=2.086, distinct_predictions=32,
            per_family_accuracy={"pointer_chase": value, "state_update": 0.36},
            per_seed_holdout_accuracy=[0.22, 0.224, 0.229],
        )
        for name, value in (("cascade", 0.1590), ("history_control", 0.1703),
                            ("learned_delta", 0.1733))
    ]
    report = _null_evaluate(arms, V4_POINTER_NULL)
    assert report["checks"]["every_primary_family_competent"] is True
    assert report["checks"]["all_arms_beat_trivial"] is True


def test_a_family_genuinely_above_its_null_passes() -> None:
    arms = _arms()
    for index, arm in enumerate(arms):
        arms[index] = _competent_arm(
            arm.arm, accuracy=0.42,
            per_family_accuracy={"pointer_chase": 0.42, "state_update": 0.40},
            per_seed_holdout_accuracy=[0.41, 0.42, 0.43],
        )
    report = _null_evaluate(arms, V4_POINTER_NULL)
    assert report["checks"]["every_primary_family_beats_its_own_null"] is True


def test_an_absent_null_fails_closed() -> None:
    """A gate cannot compare against a null it was never given."""

    report = _null_evaluate(_arms(), None)
    assert report["checks"]["every_primary_family_beats_its_own_null"] is False
    assert any("no per-family null was supplied" in r for r in report["reasons"])


def test_the_null_is_per_family_not_global() -> None:
    """One family clearing its null must not cover another that does not."""

    arms = _arms()
    for index, arm in enumerate(arms):
        arms[index] = _competent_arm(
            arm.arm, accuracy=0.30,
            per_family_accuracy={"pointer_chase": 0.16, "state_update": 0.44},
            per_seed_holdout_accuracy=[0.29, 0.30, 0.31],
        )
    report = _null_evaluate(arms, {"pointer_chase": 0.1558, "state_update": 0.0312})
    assert report["checks"]["every_primary_family_beats_its_own_null"] is False
    assert any("'pointer_chase'" in r for r in report["reasons"])
    assert not any("'state_update'" in r for r in report["reasons"])


#: The null a family whose answer is an in-context candidate actually has.
#: Measured at 0.13184 for `pointer_chase` on the registered v5 E1 holdout.
#: `DEFAULT_FAMILY_NULL` above uses the uniform answer space, which is correct
#: only for families with no candidate set -- the distinction v5 exists to make.
CANDIDATE_FAMILY_NULL = {"pointer_chase": 0.13184, "state_update": 0.0312}


def _one_below_null_arm(name: str) -> ArmCompetence:
    """An arm whose mean is competent and one of whose three runs is not.

    Every number here is chosen so that EVERY other registered check passes:
    the mean clears the trivial baseline, the absolute floor, the family null by
    +0.0302, the NLL bound and the memorization gap, and the seed CV is 0.3498
    against a bound of 0.35. The only thing wrong with it is that one run sits
    below chance for what the primary family asks.
    """

    return ArmCompetence(
        arm=name,
        holdout_accuracy=0.16207,
        seen_accuracy=0.20000,
        answer_nll=2.0,
        distinct_predictions=32,
        per_family_accuracy={"pointer_chase": 0.16207, "state_update": 0.30},
        per_seed_family="pointer_chase",
        per_seed_holdout_accuracy=[0.0966, 0.1948, 0.1948],
    )


def test_a_run_below_its_family_null_is_refused_despite_a_healthy_mean() -> None:
    """A seed that learned nothing must not be averaged into a passing arm.

    This is the v3 and v4 failure in its last remaining hiding place. The gate
    corrected the ARM-level comparison to each family's own null, but the
    per-RUN comparison was left against `minimum_absolute_accuracy` = 0.070 --
    a global floor placed in v4 against a 0.0377 majority baseline. Measured
    against the real `pointer_chase` null of 0.13184 that floor is 0.53x chance,
    so a run at 0.0966 clears it while sitting 0.0352 BELOW chance.

    Without `every_seed_beats_its_family_null` this regime returns GO with an
    empty `reasons` list, and the arms then compared are mixtures of learners
    and non-learners in unknown proportion.
    """

    arms = [_one_below_null_arm(n) for n in ("cascade", "history_control", "learned_delta")]
    report = _evaluate(arms, family_null=CANDIDATE_FAMILY_NULL)

    checks = report["checks"]
    assert checks["every_seed_beats_its_family_null"] is False
    assert report["verdict"] == "NO_GO"

    # The point of the test: every OTHER check passes, so this one is load-bearing.
    others = {k: v for k, v in checks.items() if k != "every_seed_beats_its_family_null"}
    assert all(others.values()), f"expected only the seed-null check to fail, got {others}"

    # Including the two that look like they should have caught it.
    assert checks["competence_is_reliable_across_seeds"] is True
    assert checks["every_primary_family_beats_its_own_null"] is True

    assert any("below chance" in r or "at or below chance" in r for r in report["reasons"])


def test_the_absolute_floor_alone_cannot_detect_a_below_chance_run() -> None:
    """Pins WHY the check is against the null and not a raised global floor.

    0.0966 clears the registered absolute floor of 0.070 by +0.0266 and is still
    0.0352 below the family's null. No value of a single global floor is correct
    for both `pointer_chase` (candidate chance ~0.132) and `state_update`
    (uniform chance ~0.031) at once, which is exactly why the comparison, not
    the threshold, had to change.
    """

    thresholds = _thresholds()
    worst_run = 0.0966
    assert worst_run >= thresholds.minimum_absolute_accuracy
    assert worst_run < CANDIDATE_FAMILY_NULL["pointer_chase"]
    assert CANDIDATE_FAMILY_NULL["pointer_chase"] > thresholds.minimum_absolute_accuracy


def test_the_seed_null_check_fails_closed_without_a_family_null() -> None:
    """A gate cannot make a comparison it was never given the terms for.

    Only meaningful for an arm that DECLARES which family its per-run scores
    measure. A pooled arm needs no family null -- it is held to the strongest
    trivial baseline instead -- so it must not be used to test this.
    """

    report = evaluate_competence(
        _arms(cascade={"per_seed_family": "pointer_chase"},
              history_control={"per_seed_family": "pointer_chase"},
              learned_delta={"per_seed_family": "pointer_chase"}),
        thresholds=_thresholds(),
        trivial_baselines=BASELINES,
        uniform_answer_nll=UNIFORM_NLL,
        primary_families=PRIMARY,
        family_null=None,
    )
    assert report["checks"]["every_seed_beats_its_family_null"] is False
    assert report["verdict"] == "NO_GO"


def test_a_regime_whose_every_run_clears_its_null_still_passes() -> None:
    """The check must not reject a usable regime: it only adds a floor at chance.

    Note the accuracy has to be raised to 0.25 here. The module's default
    `_competent_arm` sits at 0.094, which clears the uniform null of 0.0312 that
    `DEFAULT_FAMILY_NULL` supplies but is BELOW the 0.13184 candidate-chance null
    a family like `pointer_chase` actually has. That fixture is calibrated for a
    family with no in-context candidate set, and reusing it here would assert
    the wrong thing.
    """

    arms = _arms(
        cascade={"accuracy": 0.25, "per_seed_family": "pointer_chase"},
        history_control={"accuracy": 0.25, "per_seed_family": "pointer_chase"},
        learned_delta={"accuracy": 0.25, "per_seed_family": "pointer_chase"},
    )
    report = _evaluate(arms, family_null=CANDIDATE_FAMILY_NULL)
    assert report["checks"]["every_seed_beats_its_family_null"] is True
    assert report["verdict"] == "GO", report["reasons"]
    margin = report["detail"]["per_seed_family_null_margin"]["cascade"]
    assert margin["margin_over_baseline"] > 0
    assert margin["per_seed_family"] == "pointer_chase"


def test_the_seed_null_check_fails_closed_when_the_primary_family_has_no_null() -> None:
    """A missing null must not default to 0.0 and quietly lower the bar.

    `family_null.get(family, 0.0)` would put the bar at
    `minimum_family_margin_over_null` alone -- 0.030 -- which almost any run
    clears, so the check would certify exactly what it exists to catch. The
    arm-level check has the same shape; this pins the per-seed one.
    """

    report = evaluate_competence(
        _arms(cascade={"per_seed_family": "pointer_chase"},
              history_control={"per_seed_family": "pointer_chase"},
              learned_delta={"per_seed_family": "pointer_chase"}),
        thresholds=_thresholds(),
        trivial_baselines=BASELINES,
        uniform_answer_nll=UNIFORM_NLL,
        primary_families=PRIMARY,
        family_null={"state_update": 0.0312},  # pointer_chase deliberately absent
    )
    assert report["checks"]["every_seed_beats_its_family_null"] is False
    assert report["verdict"] == "NO_GO"
    assert any("absent from the supplied mapping" in r for r in report["reasons"])


def test_pooled_per_run_scores_are_held_to_the_trivial_baseline() -> None:
    """The v4-era callers pass POOLED per-run accuracy, not one family's.

    `phase1_v4_competence_run`, `regate_competence_report` and
    `phase1_v4_pipeline_rehearsal` all fill `per_seed_holdout_accuracy` from the
    across-family holdout accuracy. Comparing a pooled number against one
    family's null would be fail-open on this task, because `state_update`
    saturates while `pointer_chase` sits near chance. An arm that declares no
    family is therefore measured against the strongest trivial baseline, which
    is the comparison that number actually supports.
    """

    report = _evaluate(_arms(), family_null=CANDIDATE_FAMILY_NULL)
    detail = report["detail"]["per_seed_family_null_margin"]["cascade"]
    assert detail["per_seed_family"] is None
    assert detail["baseline"] == max(BASELINES.values())
    assert "pooled" in detail["comparison_basis"]
    # 0.090 is the worst default run; the bar is 0.0407 + 0.030.
    assert report["checks"]["every_seed_beats_its_family_null"] is True

    # And a pooled run below THAT bar is still caught.
    weak = _evaluate(
        _arms(cascade={"per_seed_holdout_accuracy": [0.05, 0.094, 0.098]}),
        family_null=CANDIDATE_FAMILY_NULL,
    )
    assert weak["checks"]["every_seed_beats_its_family_null"] is False
