"""Competence gate: the launch precondition v3 did not have.

v3 was rigorous about blinding, evidence chains, estimator calibration, and
power, and still produced an uninformative result — because it never asked
whether any arm could perform the task above a trivial baseline. All three arms
memorized 5,000 training examples and generalized at the majority baseline, so
the mechanism contrast compared three non-learners.

This module makes that failure impossible to repeat. It computes an arm-blind
competence verdict from **development-only** evidence and emits a machine-readable
``GO`` / ``NO_GO``. The confirmatory runner refuses to dispatch on ``NO_GO``.

## Why every check is here

``all_arms_beat_trivial``
    The *weakest* arm must clear the strongest registered trivial baseline by a
    preregistered margin. Using the weakest arm is what makes it arm-blind: a
    regime cannot qualify because the candidate mechanism alone happens to work.

``answer_nll_beats_uniform``
    v3 had accuracy marginally above the majority baseline while its answer-NLL
    was *worse* than a uniform guess over the answer space — confidently wrong.
    Accuracy alone would have passed a naive gate; NLL catches it.

``every_primary_family_competent``
    v3's `state_update` sat below uniform chance while `pointer_chase` was near
    baseline. A pooled score can hide a dead family.

``predictions_are_input_dependent``
    Guards against a degenerate constant predictor scoring at the majority
    baseline and being mistaken for competence.

``no_arm_at_floor`` / ``no_family_saturated``
    The confirmatory band must be in a sensitivity region: far enough above the
    floor that a decrement is visible, far enough below ceiling that an increment
    is. A saturated task cannot show a mechanism effect either.

``generalization_gap_acceptable``
    The v3 signature was a large gap between seen and held-out accuracy. A regime
    that only memorizes must not qualify however high its seen accuracy is.

``competence_is_reliable_across_seeds``
    Added after the v4 pilot produced a regime whose *mean* held-out accuracy
    cleared every other check while its individual seeds were bimodal — 0.12 and
    0.41 for the same arm, because a phase transition in training completed for
    some seeds and was truncated mid-descent for others. A mean that averages a
    converged and an unconverged run describes no model that exists.

    The check draws a line this module previously blurred. **Dispersion among
    runs that are all competent is a power problem, and its remedy is more
    seeds.** **Some runs failing to become competent at all is a competence
    problem, and no seed count fixes it** — the arms are no longer comparable,
    because each is a different mixture of learners and non-learners.

    The gate owns the second and hands the first to the power audit. So it
    requires the worst individual *run* to clear the absolute floor, and bounds
    the within-arm **coefficient of variation** rather than the raw SD. A
    balanced two-point mixture of a competent and an at-floor population has
    CV ≈ 0.6 by construction, while ordinary dispersion among converged runs sits
    near 0.1–0.3, so the bound separates a mixture from a merely noisy regime
    without rejecting a usable one.

    The realised SD is *reported*, not gated, under
    ``detail.realised_seed_sd_for_power_audit``. The registered design must be
    re-powered at that value before the seed count is fixed; see
    ``PHASE1_V4_PREREGISTRATION.md`` §Power.

Thresholds are supplied by the caller and must be frozen from the outcome-blind
development pilot **before** any confirmatory run. This module does not invent
them, and it never reads a confirmatory outcome.

``require_go`` additionally refuses any report whose regime is marked
``development_only``. The pipeline rehearsal produces a genuine, well-formed
``GO`` at toy scale in order to exercise this very code path, and that artifact
persists on disk afterwards; without the marker it would satisfy every
structural check the config builder makes.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

# v2 added competence_is_reliable_across_seeds. v3 splits that check's two
# concerns: the gate keeps competence (every run clears the floor, bounded by
# coefficient of variation) and hands dispersion to the power audit. Older
# reports are rejected on load: they cannot certify what they never tested.
COMPETENCE_SCHEMA_VERSION = "tgsa.phase1-competence-gate.v3"


@dataclass(frozen=True, slots=True)
class CompetenceThresholds:
    """Frozen gate parameters. Derived from the pilot, never from outcomes."""

    minimum_margin_over_trivial: float
    minimum_absolute_accuracy: float
    maximum_answer_nll_fraction_of_uniform: float = 1.0
    minimum_distinct_predictions: int = 4
    maximum_family_accuracy: float = 0.95
    maximum_generalization_gap: float = 0.10
    # A balanced two-point mixture of a competent and an at-floor population has
    # CV ~ 0.6; ordinary dispersion among converged runs sits near 0.1-0.3. This
    # sits between them, and is a structural bound rather than a fitted one.
    maximum_seed_coefficient_of_variation: float = 0.35
    #: Each primary family must beat ITS OWN null by this much. A global
    #: baseline cannot serve: see `every_primary_family_beats_its_own_null`.
    minimum_family_margin_over_null: float = 0.030
    minimum_seed_runs: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ArmCompetence:
    """Development-only competence evidence for one arm."""

    arm: str
    holdout_accuracy: float
    seen_accuracy: float
    answer_nll: float
    distinct_predictions: int
    per_family_accuracy: dict[str, float]
    # Held-out accuracy of every individual development run for this arm.
    # Empty means the evidence was never recorded, which fails the gate.
    per_seed_holdout_accuracy: list[float] = field(default_factory=list)
    #: Which family `per_seed_holdout_accuracy` measures, when it measures one.
    #:
    #: Callers disagree, and the difference decides which null those numbers may
    #: be compared against. `phase1_v5_runner._aggregate_qualification_arms`
    #: fills them from the FIRST primary family, so they carry that family's
    #: null. The v4-era callers -- `phase1_v4_competence_run`,
    #: `regate_competence_report` and `phase1_v4_pipeline_rehearsal` -- fill them
    #: with POOLED across-family accuracy, which no single family's null
    #: describes: on this task `state_update` saturates while `pointer_chase`
    #: sits near chance, so a pooled number compared against the pointer null
    #: would be fail-open by a wide margin.
    #:
    #: Declaring it makes that difference machine-checked instead of an
    #: assumption in a comment. `None` means pooled, and
    #: `every_seed_beats_its_family_null` then measures each run against the
    #: strongest trivial baseline instead -- the comparison that is correct for a
    #: pooled number -- and records which one it used.
    per_seed_family: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_competence(
    arms: Sequence[ArmCompetence],
    *,
    thresholds: CompetenceThresholds,
    trivial_baselines: Mapping[str, float],
    uniform_answer_nll: float,
    primary_families: Sequence[str],
    family_null: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Return a machine-readable GO / NO_GO verdict with per-check detail."""

    if not arms:
        raise ValueError("competence evaluation requires at least one arm")
    strongest_trivial = max(trivial_baselines.values())
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}
    reasons: list[str] = []

    weakest = min(arms, key=lambda a: a.holdout_accuracy)
    margin = weakest.holdout_accuracy - strongest_trivial
    checks["all_arms_beat_trivial"] = margin >= thresholds.minimum_margin_over_trivial
    detail["weakest_arm"] = weakest.arm
    detail["weakest_holdout_accuracy"] = weakest.holdout_accuracy
    detail["strongest_trivial_baseline"] = strongest_trivial
    detail["margin_over_trivial"] = margin
    if not checks["all_arms_beat_trivial"]:
        reasons.append(
            f"weakest arm {weakest.arm} clears the strongest trivial baseline by only "
            f"{margin:+.4f}, below the required {thresholds.minimum_margin_over_trivial:+.4f}"
        )

    checks["all_arms_reach_absolute_floor"] = all(
        a.holdout_accuracy >= thresholds.minimum_absolute_accuracy for a in arms
    )
    if not checks["all_arms_reach_absolute_floor"]:
        low = [a.arm for a in arms if a.holdout_accuracy < thresholds.minimum_absolute_accuracy]
        reasons.append(f"arms below the absolute accuracy floor: {low}")

    limit = uniform_answer_nll * thresholds.maximum_answer_nll_fraction_of_uniform
    checks["answer_nll_beats_uniform"] = all(a.answer_nll < limit for a in arms)
    detail["uniform_answer_nll"] = uniform_answer_nll
    detail["answer_nll_by_arm"] = {a.arm: a.answer_nll for a in arms}
    if not checks["answer_nll_beats_uniform"]:
        bad = [a.arm for a in arms if a.answer_nll >= limit]
        reasons.append(
            f"arms whose answer-NLL is not better than a uniform guess over the answer "
            f"space (>= {limit:.4f}): {bad}"
        )

    family_ok = True
    family_detail: dict[str, dict[str, float]] = {}
    for family in primary_families:
        values = {a.arm: a.per_family_accuracy.get(family, 0.0) for a in arms}
        family_detail[family] = values
        if min(values.values()) < thresholds.minimum_absolute_accuracy:
            family_ok = False
            reasons.append(f"family {family!r} is below the absolute floor for some arm")
    checks["every_primary_family_competent"] = family_ok
    detail["per_family_accuracy"] = family_detail

    # Each primary family against ITS OWN null, not a global constant.
    #
    # v4 certified `every_primary_family_competent` for `pointer_chase` at
    # 0.159-0.173 against a 0.070 global floor and a 0.0377 global majority
    # baseline. But that family's answer is one of the values shown in the
    # context, so a model that has learned only "the answer is one of these"
    # scores 1/node_count -- measured at 0.1558 on the registered holdout, and
    # 0.1587 for `direct_lookup`. Both families sat AT CHANCE for the correct
    # null while clearing every global bar the gate had.
    #
    # A global baseline systematically overstates competence on any family whose
    # answer space is an in-context candidate set. That is not a threshold that
    # was set too low; it is the wrong comparison.
    if family_null is None:
        checks["every_primary_family_beats_its_own_null"] = False
        reasons.append(
            "no per-family null was supplied. A family whose answers come from an "
            "in-context candidate set can sit at chance and still clear a global "
            "floor, which is exactly what v4 certified. Supply family_null"
        )
        detail["family_null"] = None
    else:
        null_ok = True
        null_detail: dict[str, dict[str, float]] = {}
        for family in primary_families:
            null = float(family_null.get(family, 0.0))
            weakest_value = min(
                a.per_family_accuracy.get(family, 0.0) for a in arms
            )
            margin = weakest_value - null
            null_detail[family] = {
                "null": null,
                "weakest_arm_accuracy": weakest_value,
                "margin_over_null": margin,
            }
            if margin < thresholds.minimum_family_margin_over_null:
                null_ok = False
                reasons.append(
                    f"family {family!r} clears its own null by only {margin:+.4f} "
                    f"(weakest arm {weakest_value:.4f} against null {null:.4f}), "
                    f"below the required "
                    f"{thresholds.minimum_family_margin_over_null:+.4f}; it is at or "
                    f"near chance for what that family actually asks"
                )
        checks["every_primary_family_beats_its_own_null"] = null_ok
        detail["family_null"] = null_detail

    checks["predictions_are_input_dependent"] = all(
        a.distinct_predictions >= thresholds.minimum_distinct_predictions for a in arms
    )
    if not checks["predictions_are_input_dependent"]:
        reasons.append("at least one arm produces a near-constant prediction")

    saturated = [
        (a.arm, family)
        for a in arms
        for family in primary_families
        if a.per_family_accuracy.get(family, 0.0) > thresholds.maximum_family_accuracy
    ]
    checks["no_primary_family_saturated"] = not saturated
    if saturated:
        reasons.append(f"saturated at an uninformative ceiling: {saturated}")

    gaps = {a.arm: a.seen_accuracy - a.holdout_accuracy for a in arms}
    checks["generalization_gap_acceptable"] = all(
        gap <= thresholds.maximum_generalization_gap for gap in gaps.values()
    )
    detail["generalization_gap"] = gaps
    if not checks["generalization_gap_acceptable"]:
        bad = {k: round(v, 4) for k, v in gaps.items() if v > thresholds.maximum_generalization_gap}
        reasons.append(f"memorization: seen-minus-holdout gap too large: {bad}")

    # Reliability. The mean is not the unit of competence: a regime whose seeds
    # straddle a training phase transition has a healthy mean and no reproducible
    # behaviour. Dispersion among all-competent runs is a POWER problem and is
    # exported below rather than gated; runs that never became competent are a
    # COMPETENCE problem and are gated here.
    reliability_ok = True
    seed_detail: dict[str, dict[str, object]] = {}
    realised_sds: list[float] = []
    for arm in arms:
        values = list(arm.per_seed_holdout_accuracy)
        mean = statistics.fmean(values) if values else 0.0
        sd = statistics.stdev(values) if len(values) > 1 else None
        cv = (sd / mean) if (sd is not None and mean > 0) else None
        entry: dict[str, object] = {
            "runs": values,
            "count": len(values),
            "worst": min(values) if values else None,
            "mean": mean if values else None,
            "sd": sd,
            "coefficient_of_variation": cv,
        }
        seed_detail[arm.arm] = entry
        if len(values) < thresholds.minimum_seed_runs:
            reliability_ok = False
            reasons.append(
                f"arm {arm.arm} recorded {len(values)} development run(s); at least "
                f"{thresholds.minimum_seed_runs} are required to estimate seed variance"
            )
            continue
        realised_sds.append(sd)  # type: ignore[arg-type]
        if min(values) < thresholds.minimum_absolute_accuracy:
            reliability_ok = False
            reasons.append(
                f"arm {arm.arm} has an individual run at {min(values):.4f}, below the "
                f"absolute floor {thresholds.minimum_absolute_accuracy:.4f}; the mean "
                f"{arm.holdout_accuracy:.4f} averages runs that behave differently"
            )
        if cv is not None and cv > thresholds.maximum_seed_coefficient_of_variation:
            reliability_ok = False
            reasons.append(
                f"arm {arm.arm} has a between-seed coefficient of variation {cv:.3f} "
                f"(sd {sd:.4f} on mean {mean:.4f}), above the bound "
                f"{thresholds.maximum_seed_coefficient_of_variation:.3f}; this is the "
                f"signature of a mixture of converged and unconverged runs rather than "
                f"a single noisy population"
            )
    detail["per_seed_holdout_accuracy"] = seed_detail
    # Not a gate. The registered design must be re-powered at this value before
    # the seed count is fixed -- carrying a stale sigma_a forward is exactly how
    # an underpowered study gets registered without anyone noticing.
    detail["realised_seed_sd_for_power_audit"] = max(realised_sds) if realised_sds else None

    checks["competence_is_reliable_across_seeds"] = reliability_ok
    detail["per_seed_holdout_accuracy"] = seed_detail

    # The same correction this module already applies at the arm level, applied
    # at the level the reliability check actually operates on.
    #
    # `competence_is_reliable_across_seeds` above compares the worst individual
    # run to `minimum_absolute_accuracy`, a GLOBAL floor. That floor was
    # calibrated in v4 against a global majority baseline of 0.0377, where 0.070
    # was a sane ~1.9x bar. v5 corrected the baseline to each family's own null
    # -- measured at ~0.1318 for `pointer_chase` -- but the per-run floor was
    # never re-derived, leaving it at roughly HALF of chance. A run at 0.10 is
    # below chance for what the primary family actually asks and still clears
    # 0.070, so a seed that demonstrably learned nothing can be averaged into a
    # passing arm mean whenever the CV bound happens to hold.
    #
    # That is the identical error described at the top of the family-null block:
    # not a threshold set too low, but the wrong comparison. The module's own
    # stated contract is that "some runs failing to become competent at all is a
    # competence problem, and no seed count fixes it". A run at chance has not
    # become competent, so it is gated here against the family's own null, by
    # the same registered margin the arm-level check uses.
    #
    # This is added rather than substituted: the absolute floor and the CV bound
    # both still apply unchanged. It can only ever turn a GO into a NO_GO.
    # Which baseline an individual run must clear depends on what that run's
    # number MEASURES, and callers differ -- see `ArmCompetence.per_seed_family`.
    # An arm that declares a family carries that family's null; an arm that
    # declares none carries a pooled across-family score, which no single
    # family's null describes, so it is held to the strongest trivial baseline
    # instead. Both are real comparisons; picking the wrong one would reproduce
    # the wrong-comparison error this module exists to correct.
    seed_null_ok = True
    seed_null_detail: dict[str, dict[str, object]] = {}
    for arm in arms:
        values = list(arm.per_seed_holdout_accuracy)
        worst = min(values) if values else None
        declared = arm.per_seed_family

        if declared is None:
            baseline: float | None = strongest_trivial
            basis = "strongest trivial baseline (per-run scores are pooled)"
        elif family_null is None:
            baseline, basis = None, f"{declared!r} null (not supplied)"
        elif declared not in family_null:
            # Fail closed rather than defaulting the null to 0.0. A missing null
            # would put the bar at `minimum_family_margin_over_null` alone --
            # 0.030, which almost any run clears -- so the check would silently
            # certify what it exists to catch.
            baseline, basis = None, (
                f"{declared!r} null (absent from the supplied mapping "
                f"{sorted(family_null)})"
            )
        else:
            baseline = float(family_null[declared])
            basis = f"{declared!r} null"

        required = (
            None if baseline is None
            else baseline + thresholds.minimum_family_margin_over_null
        )
        seed_null_detail[arm.arm] = {
            "worst_run": worst,
            "per_seed_family": declared,
            "comparison_basis": basis,
            "baseline": baseline,
            "required_minimum": required,
            "margin_over_baseline": (
                None if (worst is None or baseline is None) else worst - baseline
            ),
        }
        if baseline is None or required is None:
            seed_null_ok = False
            reasons.append(
                f"arm {arm.arm} carries per-run scores that cannot be measured: the "
                f"comparison basis is the {basis}, and a gate may not make a "
                f"comparison it was not given the terms for"
            )
        elif worst is None or worst < required:
            seed_null_ok = False
            shown = "no runs" if worst is None else f"{worst:.4f}"
            margin = (
                "n/a" if worst is None else format(worst - baseline, "+.4f")
            )
            reasons.append(
                f"arm {arm.arm} has an individual run at {shown}, which clears the "
                f"{basis} of {baseline:.4f} by only {margin} and so falls below the "
                f"required {required:.4f}. That run is at or below chance for what "
                f"it was scored on; averaging it into the arm mean would compare "
                f"mixtures of learners and non-learners, which is what v3 and v4 "
                f"each failed on"
            )
    checks["every_seed_beats_its_family_null"] = seed_null_ok
    detail["per_seed_family_null_margin"] = seed_null_detail

    verdict = "GO" if all(checks.values()) else "NO_GO"
    return {
        "schema_version": COMPETENCE_SCHEMA_VERSION,
        "verdict": verdict,
        "checks": dict(sorted(checks.items())),
        "reasons": reasons,
        "thresholds": thresholds.to_dict(),
        "trivial_baselines": dict(trivial_baselines),
        "primary_families": list(primary_families),
        "arms": [a.to_dict() for a in arms],
        "detail": detail,
        "claim_scope": (
            "development-only competence evidence; establishes that a between-arm "
            "mechanism contrast would be interpretable, and says nothing about the "
            "mechanism itself"
        ),
    }


def load_verdict(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != COMPETENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported competence report schema: {value.get('schema_version')}")
    if value.get("verdict") not in {"GO", "NO_GO"}:
        raise ValueError(f"competence report has no usable verdict: {value.get('verdict')!r}")
    return value


def require_go(path: Path, *, allow_development_only: bool = False) -> dict[str, object]:
    """Fail closed unless a frozen competence report says GO.

    Called by the confirmatory runner. A missing, malformed, or NO_GO report
    must stop the launch — that is the entire point of the gate.

    ``allow_development_only`` exists solely so the pipeline rehearsal can
    exercise this code path with its own toy report. It is never set from a
    real study configuration: ``run_study_matrix`` passes it only when the
    study itself declares ``development_only``, and ``build_phase1_v4_config``
    only emits that declaration when it was given a marked report. The marker
    therefore travels with the artifact chain and cannot be laundered by
    passing a marked report to an unmarked config, or the reverse.
    """

    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"competence report not found at {report_path}; a confirmatory matrix may not "
            f"be dispatched without a frozen GO verdict"
        )
    report = load_verdict(report_path)

    # A rehearsal produces a real, well-formed GO report at toy scale so the
    # pipeline's gate path can be exercised end to end. That artifact then sits
    # on disk at a path this function would otherwise accept, and it now passes
    # every structural check the config builder makes. Refusing it here is the
    # only thing standing between a toy report and a dispatched matrix.
    regime = report.get("regime")
    if isinstance(regime, Mapping) and regime.get("development_only") and not allow_development_only:
        raise ValueError(
            f"competence report at {report_path} is marked development_only; it was "
            f"produced by a rehearsal or a smoke run and certifies nothing. A "
            f"confirmatory matrix requires a report from a real measurement"
        )

    if report["verdict"] != "GO":
        failed = [name for name, ok in report["checks"].items() if not ok]  # type: ignore[union-attr]
        raise ValueError(
            f"competence gate returned NO_GO; failed checks: {failed}; "
            f"reasons: {report['reasons']}"
        )
    return report
