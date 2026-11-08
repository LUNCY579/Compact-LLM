"""Frozen paired analysis for the TGSA Phase-1 model study.

The primary estimand is a macro-average over registered non-guardrail strata,
then over paired training seeds. Seed and example are CROSSED factors: every
training seed is evaluated on the same materialized test split. The registered
bootstrap therefore resamples seed IDs and example IDs and applies the same
resampled examples to every selected seed, preserving that shared structure.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from .phase1_data import sha256_file
from .phase1_inference import (
    INTERVAL_ESTIMATORS,
    PRIMARY_ESTIMATOR,
    estimate_interval,
    resolve_interval_estimator,
)
from .phase1_response import difficulty_response
from .phase1_verify import verify_evidence_chain
from .phase1_runner import (
    MATRIX_MANIFEST_VERSION,
    REQUIRED_ARMS,
    RUN_MANIFEST_VERSION,
    atomic_write_json,
    load_study_config,
)

ANALYSIS_SCHEMA_VERSION = "tgsa.phase1-model-analysis.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    observed: float
    confidence_low: float
    confidence_high: float
    probability_positive: float
    samples: int
    confidence: float
    seed_count: int
    stratum_count: int
    example_seed_pairs: int

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "probability_positive": self.probability_positive,
            "samples": self.samples,
            "confidence": self.confidence,
            "seed_count": self.seed_count,
            "stratum_count": self.stratum_count,
            "example_seed_pairs": self.example_seed_pairs,
        }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile input is empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"prediction row is not an object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate prediction IDs: {path}")
    return rows


def _index_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {str(row["example_id"]): row for row in rows}


def _validate_pair(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
) -> list[tuple[Mapping[str, object], Mapping[str, object]]]:
    candidate_index = _index_rows(candidate)
    baseline_index = _index_rows(baseline)
    if candidate_index.keys() != baseline_index.keys():
        missing_candidate = sorted(baseline_index.keys() - candidate_index.keys())[:10]
        missing_baseline = sorted(candidate_index.keys() - baseline_index.keys())[:10]
        raise ValueError(
            f"prediction ID mismatch; missing candidate={missing_candidate}, "
            f"missing baseline={missing_baseline}"
        )
    paired: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for example_id in sorted(candidate_index):
        left = candidate_index[example_id]
        right = baseline_index[example_id]
        for field in ("family", "stratum", "answer_token", "split", "seed"):
            if left[field] != right[field]:
                raise ValueError(f"paired prediction metadata mismatch for {example_id}: {field}")
        paired.append((left, right))
    return paired


def _accuracy_delta(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> float:
    return float(bool(candidate["correct"])) - float(bool(baseline["correct"]))


def _nll_improvement(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> float:
    return float(baseline["answer_nll"]) - float(candidate["answer_nll"])


def paired_seed_stratum_values(
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    baseline_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    score: Callable[[Mapping[str, object], Mapping[str, object]], float],
    include_families: set[str] | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """Paired per-example differences keyed by seed, stratum, and example ID.

    Example identity is carried explicitly rather than implied by list position.
    The crossed bootstrap needs it: seed and example are crossed factors, so a
    resampled example must be the *same* example in every selected seed.
    """

    if candidate_by_seed.keys() != baseline_by_seed.keys():
        raise ValueError("candidate and baseline seed sets differ")
    result: dict[int, dict[str, dict[str, float]]] = {}
    for seed in sorted(candidate_by_seed):
        paired = _validate_pair(candidate_by_seed[seed], baseline_by_seed[seed])
        strata: dict[str, dict[str, float]] = {}
        for candidate, baseline in paired:
            family = str(candidate["family"])
            if include_families is not None and family not in include_families:
                continue
            stratum = str(candidate["stratum"])
            strata.setdefault(stratum, {})[str(candidate["example_id"])] = score(candidate, baseline)
        if not strata:
            raise ValueError(f"family filter removed all examples for seed {seed}")
        result[seed] = dict(sorted(strata.items()))
    reference = next(iter(result.values()))
    reference_strata = set(reference)
    for seed, strata in result.items():
        if set(strata) != reference_strata:
            raise ValueError(f"stratum set differs for seed {seed}")
        for stratum in reference_strata:
            # All seeds are evaluated on the same materialized test split, so
            # the example set inside a stratum must be identical. If it is not,
            # the design is not crossed and the paired estimand is undefined.
            if set(strata[stratum]) != set(reference[stratum]):
                raise ValueError(
                    f"seed {seed} stratum {stratum!r} does not share the reference example set; "
                    "seed and example are not crossed"
                )
    return result


def crossed_paired_bootstrap(
    values: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> BootstrapEstimate:
    """Two-way paired bootstrap over the crossed seed x example design.

    Every training seed is evaluated on the *same* materialized test examples,
    so seed and example are crossed, not nested. Each draw therefore resamples
    seed IDs and, independently, example IDs within each stratum — and then
    evaluates every selected seed on the *same* resampled examples, preserving
    the shared-example structure across seeds.

    The pre-remediation estimator resampled examples independently inside each
    selected seed, which destroys that structure and can understate uncertainty
    when example-level effects are correlated across seeds
    (``PHASE1_LAUNCH_AUDIT.md`` finding P0-7).

    Ten thousand draws do not create information: with five training seeds the
    seed-level resolution remains coarse, which is why the registered gates also
    require a minimum observed effect and why an outcome-blind power simulation
    is part of the launch gates.
    """

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    seed_ids = sorted(values)
    if not seed_ids:
        raise ValueError("no paired seeds")
    strata = sorted(next(iter(values.values())))
    if not strata:
        raise ValueError("no paired strata")
    example_ids: dict[str, list[str]] = {}
    for stratum in strata:
        ids = sorted(values[seed_ids[0]][stratum])
        if not ids:
            raise ValueError("all seed/stratum cells must be non-empty")
        example_ids[stratum] = ids
    for seed_id in seed_ids:
        if sorted(values[seed_id]) != strata:
            raise ValueError("all seeds must share the same strata")
        for stratum in strata:
            if sorted(values[seed_id][stratum]) != example_ids[stratum]:
                raise ValueError("all seeds must share the same examples inside each stratum")

    def macro(seed_id: int, selection: Mapping[str, Sequence[str]]) -> float:
        total = 0.0
        for stratum in strata:
            cell = values[seed_id][stratum]
            chosen = selection[stratum]
            total += sum(cell[example_id] for example_id in chosen) / len(chosen)
        return total / len(strata)

    identity = {stratum: example_ids[stratum] for stratum in strata}
    observed = sum(macro(seed_id, identity) for seed_id in seed_ids) / len(seed_ids)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected_seeds = [rng.choice(seed_ids) for _ in seed_ids]
        # One example resample per draw, shared by every selected seed. This is
        # what keeps the design crossed.
        selection = {
            stratum: [rng.choice(example_ids[stratum]) for _ in example_ids[stratum]]
            for stratum in strata
        }
        draws.append(
            sum(macro(seed_id, selection) for seed_id in selected_seeds) / len(selected_seeds)
        )
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    return BootstrapEstimate(
        observed=observed,
        confidence_low=_quantile(draws, alpha),
        confidence_high=_quantile(draws, 1.0 - alpha),
        probability_positive=sum(value > 0.0 for value in draws) / len(draws),
        samples=samples,
        confidence=confidence,
        seed_count=len(seed_ids),
        stratum_count=len(strata),
        example_seed_pairs=sum(
            len(values[seed_id][stratum]) for seed_id in seed_ids for stratum in strata
        ),
    )


def seed_cluster_bootstrap(
    values: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> BootstrapEstimate:
    """Registered alternative: treat the generated test set as fixed.

    Only training seeds are resampled. The resulting interval covers variation
    across training seeds for *this* materialized test split, so the claim is
    correspondingly narrower.
    """

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    seed_ids = sorted(values)
    strata = sorted(next(iter(values.values())))

    def macro(seed_id: int) -> float:
        total = 0.0
        for stratum in strata:
            cell = values[seed_id][stratum]
            total += sum(cell.values()) / len(cell)
        return total / len(strata)

    per_seed = {seed_id: macro(seed_id) for seed_id in seed_ids}
    observed = sum(per_seed.values()) / len(seed_ids)
    rng = random.Random(seed)
    draws = sorted(
        sum(per_seed[rng.choice(seed_ids)] for _ in seed_ids) / len(seed_ids)
        for _ in range(samples)
    )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapEstimate(
        observed=observed,
        confidence_low=_quantile(draws, alpha),
        confidence_high=_quantile(draws, 1.0 - alpha),
        probability_positive=sum(value > 0.0 for value in draws) / len(draws),
        samples=samples,
        confidence=confidence,
        seed_count=len(seed_ids),
        stratum_count=len(strata),
        example_seed_pairs=sum(
            len(values[seed_id][stratum]) for seed_id in seed_ids for stratum in strata
        ),
    )


BOOTSTRAP_SCHEMES: dict[str, Callable[..., BootstrapEstimate]] = {
    "crossed_paired": crossed_paired_bootstrap,
    "seed_cluster": seed_cluster_bootstrap,
}


def resolve_bootstrap(scheme: str) -> Callable[..., BootstrapEstimate]:
    if scheme not in BOOTSTRAP_SCHEMES:
        raise ValueError(
            f"unsupported bootstrap scheme {scheme!r}; registered schemes are "
            f"{sorted(BOOTSTRAP_SCHEMES)}"
        )
    return BOOTSTRAP_SCHEMES[scheme]


def _prediction_path(
    output_root: Path,
    *,
    arm: str,
    seed: int,
    split: str,
    intervention: str,
) -> Path:
    return output_root / "runs" / arm / f"seed_{seed}" / f"predictions_{split}__{intervention}.jsonl"


def _load_prediction_matrix(
    output_root: Path,
    *,
    arm: str,
    seeds: Sequence[int],
    split: str,
    intervention: str,
) -> dict[int, list[dict[str, object]]]:
    return {
        seed: load_jsonl(
            _prediction_path(
                output_root,
                arm=arm,
                seed=seed,
                split=split,
                intervention=intervention,
            )
        )
        for seed in seeds
    }


def _validate_run_artifacts(
    output_root: Path,
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, object]:
    manifests: dict[str, object] = {}
    dataset_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for arm in arms:
        for seed in seeds:
            run_dir = output_root / "runs" / arm / f"seed_{seed}"
            path = run_dir / "run_manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != RUN_MANIFEST_VERSION or value.get("status") != "complete":
                raise ValueError(f"run manifest is not complete: {path}")
            if value.get("arm") != arm or int(value.get("seed")) != seed:
                raise ValueError(f"run manifest identity mismatch: {path}")
            dataset_hashes.add(str(value["dataset_manifest_sha256"]))
            config_hashes.add(str(value["study_config_sha256"]))
            evaluations = cast(Mapping[str, Mapping[str, object]], value["evaluations"])
            for evaluation in evaluations.values():
                prediction_path = run_dir / str(evaluation["path"])
                if sha256_file(prediction_path) != evaluation["sha256"]:
                    raise ValueError(f"prediction hash mismatch: {prediction_path}")
            manifests[f"{arm}:{seed}"] = {
                "path": str(path.relative_to(output_root)),
                "sha256": sha256_file(path),
                "initial_model_state_sha256": value["initial_model_state_sha256"],
                "final_model_state_sha256": value["final_model_state_sha256"],
                "trainable_parameters": value["parameter_audit"]["trainable_parameters"],
                "tokens_seen": value["training"]["tokens_seen"],
                "completed_steps": value["training"]["completed_steps"],
            }
    if len(dataset_hashes) != 1 or len(config_hashes) != 1:
        raise ValueError("runs do not share exactly one dataset/config hash")
    return {
        "runs": manifests,
        "shared_dataset_manifest_sha256": next(iter(dataset_hashes)),
        "shared_study_config_sha256": next(iter(config_hashes)),
    }


def _per_family_observed(
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    baseline_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
) -> dict[str, float]:
    families = sorted({str(row["family"]) for rows in candidate_by_seed.values() for row in rows})
    result: dict[str, float] = {}
    for family in families:
        values = paired_seed_stratum_values(
            candidate_by_seed,
            baseline_by_seed,
            score=_accuracy_delta,
            include_families={family},
        )
        seed_values = []
        for seed, strata in values.items():
            seed_values.append(
                sum(sum(cell.values()) / len(cell) for cell in strata.values()) / len(strata)
            )
        result[family] = sum(seed_values) / len(seed_values)
    return result


def _estimate_comparison(
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    baseline_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    include_families: set[str],
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
    scheme: str = PRIMARY_ESTIMATOR,
    sensitivity_schemes: Sequence[str] = (),
) -> dict[str, object]:
    """Estimate one registered contrast.

    The primary interval uses the registered estimator. Every additional
    registered estimator is reported alongside it as a preregistered sensitivity
    analysis, so a reader can see immediately whether the conclusion depends on
    the inferential choice. Only the primary enters a gate.
    """

    resolve_interval_estimator(scheme)
    accuracy_values = paired_seed_stratum_values(
        candidate_by_seed,
        baseline_by_seed,
        score=_accuracy_delta,
        include_families=include_families,
    )
    nll_values = paired_seed_stratum_values(
        candidate_by_seed,
        baseline_by_seed,
        score=_nll_improvement,
        include_families=include_families,
    )
    sensitivity: dict[str, object] = {}
    for name in sensitivity_schemes:
        if name == scheme:
            continue
        try:
            sensitivity[name] = estimate_interval(
                accuracy_values,
                estimator=name,
                confidence=confidence,
                samples=bootstrap_samples,
                rng_seed=bootstrap_seed + 500,
            ).to_dict()
        except ValueError as exc:
            # A sensitivity analysis that cannot be computed is reported as such.
            # It must never abort, and must never silently vanish either.
            sensitivity[name] = {"status": "unavailable", "reason": str(exc)}
    return {
        "interval_estimator": scheme,
        "accuracy_delta": estimate_interval(
            accuracy_values,
            estimator=scheme,
            confidence=confidence,
            samples=bootstrap_samples,
            rng_seed=bootstrap_seed,
        ).to_dict(),
        "answer_nll_improvement": estimate_interval(
            nll_values,
            estimator=scheme,
            confidence=confidence,
            samples=bootstrap_samples,
            rng_seed=bootstrap_seed + 1,
        ).to_dict(),
        "sensitivity_estimators": sensitivity,
        "per_family_accuracy_delta": _per_family_observed(candidate_by_seed, baseline_by_seed),
    }


def _markdown_report(analysis: Mapping[str, object]) -> str:
    comparisons = cast(Mapping[str, Mapping[str, object]], analysis["comparisons"])
    gates = cast(Mapping[str, Mapping[str, object]], analysis["gates"])
    lines = [
        "# TGSA Phase 1 model-identifiability analysis",
        "",
        f"**Decision:** `{analysis['decision']}`",
        "",
        f"**Confirmatory configuration:** `{analysis['confirmatory_config']}`  ",
        f"**Registered confirmatory matrix:** `{analysis['registered_confirmatory_matrix']}`  ",
        f"**Interval estimator:** `{analysis['interval_estimator']}`  ",
        f"**Alignment falsifier:** `{analysis['alignment_condition']}`",
        "",
        str(analysis["claim_ceiling"]),
        "",
        "## Registered comparisons",
        "",
        "| Comparison | Mean accuracy delta | Confidence interval | P(delta > 0) |",
        "|---|---:|---:|---:|",
    ]
    for name, result in comparisons.items():
        estimate = cast(Mapping[str, object], result["accuracy_delta"])
        lines.append(
            f"| `{name}` | {float(estimate['observed']):+.4f} | "
            f"[{float(estimate['confidence_low']):+.4f}, {float(estimate['confidence_high']):+.4f}] | "
            f"{float(estimate['probability_positive']):.4f} |"
        )
    lines.extend(["", "## Advancement gates", ""])
    for name, gate in gates.items():
        lines.append(
            f"- **{name}:** {'PASS' if gate['pass'] else 'FAIL'} — {gate['reason']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "A positive decision supports only the controlled Phase-1 statement that an aligned "
            "finite-difference control signal improved these trained synthetic tasks beyond an "
            "information-matched sequential-history control, and that replaying another "
            "same-length example's aligned control reduced the learned-delta model. It does not "
            "establish Qwen-scale, natural-language, long-context, efficiency, novelty, or "
            "publication-primary claims. A `smoke_only` decision is never evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_study(
    *,
    study_config_path: Path,
    output_root: Path,
    dataset_dir: Path | None = None,
    repo_root: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, object]:
    study = load_study_config(study_config_path)
    confirmatory_config = bool(study["confirmatory"])
    if confirmatory_config and dataset_dir is None:
        raise ValueError(
            "confirmatory analysis requires --dataset-dir so the evidence chain can be "
            "bound to the materialized dataset actually used"
        )
    matrix_path = output_root / "matrix_manifest.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != MATRIX_MANIFEST_VERSION:
        raise ValueError("unsupported matrix manifest")
    if matrix.get("status") != "complete":
        raise ValueError("matrix manifest is not complete")
    configured_arms = tuple(str(arm) for arm in study["arms"])
    configured_seeds = tuple(int(seed) for seed in study["seeds"])
    selected_arms = tuple(str(arm) for arm in matrix["selected_arms"])
    selected_seeds = tuple(int(seed) for seed in matrix["selected_seeds"])
    is_full = set(selected_arms) == set(configured_arms) and set(selected_seeds) == set(configured_seeds)
    if not is_full and not allow_partial:
        raise ValueError("refusing confirmatory analysis of a partial arm/seed matrix")
    for required in REQUIRED_ARMS:
        if required not in selected_arms:
            raise ValueError(f"selected matrix lacks required arm {required}")

    preflight_path = output_root / str(matrix["preflight"])
    if sha256_file(preflight_path) != matrix["preflight_sha256"]:
        raise ValueError("preflight hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    initialization_audits = cast(Mapping[str, Mapping[str, object]], preflight["paired_initialization_by_seed"])
    for seed in selected_seeds:
        audit = initialization_audits[str(seed)]
        if not audit["history_delta_exact_initialization"]:
            raise ValueError(f"history/delta initialization mismatch for seed {seed}")
        if not audit["history_delta_parameter_count_equal"]:
            raise ValueError(f"history/delta parameter-count mismatch for seed {seed}")
        if not cast(Mapping[str, object], audit["cascade_delta_shared_initialization"])["identical"]:
            raise ValueError(f"cascade/delta shared initialization mismatch for seed {seed}")

    artifact_audit = _validate_run_artifacts(
        output_root,
        arms=selected_arms,
        seeds=selected_seeds,
    )
    # Bind the whole chain — current config, current materialized data, current
    # source checkout, preflight, matrix, run manifests, checkpoints, train logs,
    # and prediction files — with exact hashes. The integrity gate below is
    # derived from this report instead of being hard-coded to pass.
    chain = verify_evidence_chain(
        study_config_path=study_config_path,
        output_root=output_root,
        dataset_dir=dataset_dir,
        repo_root=repo_root,
    )
    if str(matrix.get("study_config_sha256")) != sha256_file(study_config_path):
        raise ValueError("matrix manifest was produced from a different study configuration")
    if str(artifact_audit["shared_study_config_sha256"]) != sha256_file(study_config_path):
        raise ValueError("run manifests were produced from a different study configuration")
    if str(artifact_audit["shared_dataset_manifest_sha256"]) != str(
        matrix.get("dataset_manifest_sha256")
    ):
        raise ValueError("run manifests and matrix manifest disagree about the dataset")

    analysis_cfg = cast(Mapping[str, object], study["analysis"])
    bootstrap_samples = int(analysis_cfg.get("bootstrap_samples", 10000))
    confidence = float(analysis_cfg.get("confidence", 0.95))
    bootstrap_seed = int(analysis_cfg.get("bootstrap_seed", 91273))
    bootstrap_scheme = str(analysis_cfg.get("interval_estimator", PRIMARY_ESTIMATOR))
    resolve_interval_estimator(bootstrap_scheme)
    sensitivity_schemes = [
        str(name) for name in analysis_cfg.get("sensitivity_estimators", list(INTERVAL_ESTIMATORS))
    ]
    for name in sensitivity_schemes:
        resolve_interval_estimator(name)
    primary_families = {str(value) for value in analysis_cfg["primary_families"]}
    guardrail_family = str(analysis_cfg.get("guardrail_family", "direct_lookup"))
    alignment_condition = str(analysis_cfg.get("alignment_condition", "replay_derangement"))
    minimum_coverage = float(analysis_cfg.get("minimum_derangement_coverage", 0.95))

    learned = _load_prediction_matrix(
        output_root,
        arm="learned_delta",
        seeds=selected_seeds,
        split="test",
        intervention="identity",
    )
    history = _load_prediction_matrix(
        output_root,
        arm="history_control",
        seeds=selected_seeds,
        split="test",
        intervention="identity",
    )
    cascade = _load_prediction_matrix(
        output_root,
        arm="cascade",
        seeds=selected_seeds,
        split="test",
        intervention="identity",
    )
    rolled = _load_prediction_matrix(
        output_root,
        arm="learned_delta",
        seeds=selected_seeds,
        split="test",
        intervention=alignment_condition,
    )
    # Coverage of the length-matched derangement. Examples with no valid
    # same-length donor keep their own control and contribute a zero paired
    # difference, which biases the contrast toward zero; the gate below refuses
    # to call the falsifier informative if too many examples were skipped.
    coverage_by_seed: dict[str, float] = {}
    for seed in selected_seeds:
        rows = rolled[seed]
        applied = [bool(row.get("intervention_applied", True)) for row in rows]
        coverage_by_seed[str(seed)] = sum(applied) / len(applied)
    observed_coverage = min(coverage_by_seed.values()) if coverage_by_seed else 0.0

    comparisons: dict[str, object] = {
        "learned_delta_minus_history_control": _estimate_comparison(
            learned,
            history,
            include_families=primary_families,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
            scheme=bootstrap_scheme,
            sensitivity_schemes=sensitivity_schemes,
        ),
        "learned_delta_minus_cascade": _estimate_comparison(
            learned,
            cascade,
            include_families=primary_families,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 1000,
            scheme=bootstrap_scheme,
            sensitivity_schemes=sensitivity_schemes,
        ),
        "aligned_delta_minus_replayed_delta": _estimate_comparison(
            learned,
            rolled,
            include_families=primary_families,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 2000,
            scheme=bootstrap_scheme,
            sensitivity_schemes=sensitivity_schemes,
        ),
        "direct_lookup_delta_minus_history_control": _estimate_comparison(
            learned,
            history,
            include_families={guardrail_family},
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 3000,
            scheme=bootstrap_scheme,
            sensitivity_schemes=sensitivity_schemes,
        ),
    }

    history_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["learned_delta_minus_history_control"])["accuracy_delta"])
    cascade_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["learned_delta_minus_cascade"])["accuracy_delta"])
    alignment_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["aligned_delta_minus_replayed_delta"])["accuracy_delta"])
    guardrail_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["direct_lookup_delta_minus_history_control"])["accuracy_delta"])

    # v3 gate definition. The registered primary test is against zero; the
    # "minimum effect" values are the effects the study is POWERED to detect and
    # are reported descriptively alongside the interval.
    #
    # v2 additionally required `observed >= threshold`. That clause caps
    # detection probability at 50% when the true effect equals the threshold,
    # because an unbiased symmetric estimator exceeds its own mean half the time
    # — a property of the rule, not of the sample size. See
    # PHASE1_V3_POWER_REPORT.md.
    gate_rule = str(analysis_cfg.get("primary_gate", "lower_bound_above_zero"))
    if gate_rule not in {"lower_bound_above_zero", "conjunction"}:
        raise ValueError(f"unsupported primary_gate {gate_rule!r}")
    # v3 names these `powered_for_*` because that is what they are: the effect
    # the replication level was chosen to detect. The v2 `minimum_*` names are
    # accepted as a fallback so a v2 configuration still analyses.
    min_history = float(
        analysis_cfg.get("powered_for_history_delta", analysis_cfg.get("minimum_history_delta", 0.02))
    )
    min_cascade = float(
        analysis_cfg.get("powered_for_cascade_delta", analysis_cfg.get("minimum_cascade_delta", 0.02))
    )
    min_alignment = float(
        analysis_cfg.get("powered_for_alignment_drop", analysis_cfg.get("minimum_alignment_drop", 0.015))
    )

    def _passes(estimate: Mapping[str, object], powered_for: float) -> bool:
        if float(estimate["confidence_low"]) <= 0.0:
            return False
        if gate_rule == "conjunction":
            return float(estimate["observed"]) >= powered_for
        return True

    def _reason(estimate: Mapping[str, object], powered_for: float) -> str:
        return (
            f"mean={float(estimate['observed']):+.4f}, "
            f"lower={float(estimate['confidence_low']):+.4f}, "
            f"upper={float(estimate['confidence_high']):+.4f}, "
            f"powered_for={powered_for:+.4f}, "
            f"observed_at_or_above_powered_effect="
            f"{float(estimate['observed']) >= powered_for}"
        )
    noninferiority = float(analysis_cfg.get("direct_lookup_noninferiority_margin", -0.01))
    gates: dict[str, dict[str, object]] = {
        "full_registered_matrix": {
            "pass": is_full,
            "reason": f"selected arms={selected_arms}; selected seeds={selected_seeds}",
        },
        "H1a_finite_difference_beats_information_matched_history": {
            "pass": _passes(history_estimate, min_history),
            "reason": _reason(history_estimate, min_history),
        },
        "H1b_finite_difference_beats_cascade": {
            "pass": _passes(cascade_estimate, min_cascade),
            "reason": _reason(cascade_estimate, min_cascade),
        },
        "H1c_trajectory_alignment_is_causal": {
            "pass": _passes(alignment_estimate, min_alignment),
            "reason": "aligned-minus-replayed " + _reason(alignment_estimate, min_alignment),
        },
        "direct_lookup_noninferiority": {
            "pass": float(guardrail_estimate["confidence_low"]) >= noninferiority,
            "reason": (
                f"lower={float(guardrail_estimate['confidence_low']):+.4f}, "
                f"margin={noninferiority:+.4f}"
            ),
        },
        "alignment_falsifier_coverage": {
            "pass": observed_coverage >= minimum_coverage,
            "reason": (
                f"minimum per-seed length-matched derangement coverage={observed_coverage:.4f}, "
                f"threshold={minimum_coverage:.4f}"
            ),
        },
        "paired_initialization_and_artifact_integrity": {
            "pass": bool(chain["passed"]),
            "reason": (
                "closed evidence chain verified: "
                f"{sum(1 for value in cast(Mapping[str, bool], chain['checks']).values() if value)}"
                f"/{len(cast(Mapping[str, bool], chain['checks']))} checks passed"
                + ("" if chain["passed"] else f"; failures={list(cast(Sequence[str], chain['errors']))[:5]}")
            ),
        },
    }

    all_pass = all(bool(gate["pass"]) for gate in gates.values())
    # A non-confirmatory configuration can never produce an evidentiary
    # decision, regardless of scores or matrix completeness. Before the launch
    # audit a complete one-seed smoke matrix could be labelled
    # `supports_phase1_mechanism_hypothesis` because `is_full` was evaluated
    # relative to the smoke configuration itself (finding P0-4).
    registered_matrix = (
        confirmatory_config
        and is_full
        and str(matrix.get("study_id")) == str(study["study_id"])
        and str(matrix.get("study_config_sha256")) == sha256_file(study_config_path)
    )
    if not registered_matrix:
        decision = "smoke_only"
    elif all_pass:
        decision = "supports_phase1_mechanism_hypothesis"
    elif float(history_estimate["confidence_high"]) < 0.0 or float(alignment_estimate["confidence_high"]) < 0.0:
        decision = "contradicted"
    else:
        decision = "inconclusive"

    # Secondary, explicitly non-gating. Bins come from registered generator
    # metadata, so no binning decision can be made after seeing outcomes.
    secondary: dict[str, object] = {
        "status": "skipped",
        "reason": "requires --dataset-dir to read registered example metadata",
    }
    if dataset_dir is not None:
        from .phase1_data import load_examples, validate_dataset_manifest

        dataset_manifest = validate_dataset_manifest(dataset_dir / "dataset_manifest.json")
        splits = cast(Mapping[str, Mapping[str, object]], dataset_manifest["splits"])
        test_examples = load_examples(dataset_dir / str(splits["test"]["path"]))
        secondary = difficulty_response(
            learned,
            history,
            examples_metadata={
                example.id: dict(example.metadata) for example in test_examples
            },
            example_families={example.id: example.family for example in test_examples},
            estimator=bootstrap_scheme,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            rng_seed=bootstrap_seed + 4000,
        )

    h1a = bool(gates["H1a_finite_difference_beats_information_matched_history"]["pass"])
    h1b = bool(gates["H1b_finite_difference_beats_cascade"]["pass"])
    h1c = bool(gates["H1c_trajectory_alignment_is_causal"]["pass"])
    if h1a and h1b and h1c:
        interpretation = (
            "controlled evidence that trajectory-specific, example-aligned information adds "
            "capability beyond the sequential cascade and an information-matched history control"
        )
    elif h1b and not h1a:
        interpretation = (
            "evidence for an extra learned recurrent pathway, NOT for trajectory-specific "
            "capability: learned_delta beat cascade but not the information-matched history control"
        )
    elif h1a and h1b and not h1c:
        interpretation = (
            "the finite-difference parameterization may help, but example-aligned trajectory "
            "content has NOT been shown causal: the replay-derangement falsifier did not fire"
        )
    elif float(history_estimate["confidence_high"]) < 0.0:
        interpretation = "evidence AGAINST the mechanism in this setting: the primary effect is negative"
    elif float(history_estimate["confidence_low"]) <= 0.0 <= float(history_estimate["confidence_high"]):
        interpretation = (
            "inconclusive, not success: the primary confidence interval includes zero"
        )
    else:
        interpretation = "inconclusive: the registered hypothesis hierarchy was not satisfied"

    analysis: dict[str, object] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "primary_gate": gate_rule,
        "hypothesis_hierarchy": {
            "H1a_learned_delta_gt_history_control": h1a,
            "H1b_learned_delta_gt_cascade": h1b,
            "H1c_aligned_gt_replay_deranged": h1c,
            "interpretation": interpretation,
        },
        "secondary_difficulty_response": secondary,
        "secondary_analyses_are_non_gating": True,
        "created_at_utc": utc_now(),
        "study_id": study["study_id"],
        "decision": decision,
        "claim_ceiling": (
            "At most: controlled tiny-decoder Phase-1 evidence that an aligned finite-difference "
            "control signal adds capability beyond an information-matched sequential-history "
            "control on the registered synthetic task families. This analysis cannot establish "
            "Qwen/GQA, natural-language, long-context, efficiency, generalization, novelty, or "
            "paper-primary claims."
        ),
        "is_full_registered_matrix": is_full,
        "confirmatory_config": confirmatory_config,
        "registered_confirmatory_matrix": registered_matrix,
        "interval_estimator": bootstrap_scheme,
        "sensitivity_estimators": sensitivity_schemes,
        "alignment_condition": alignment_condition,
        "derangement_coverage_by_seed": coverage_by_seed,
        "evidence_chain": chain,
        "configured_arms": list(configured_arms),
        "configured_seeds": list(configured_seeds),
        "selected_arms": list(selected_arms),
        "selected_seeds": list(selected_seeds),
        "primary_families": sorted(primary_families),
        "guardrail_family": guardrail_family,
        "analysis_config": dict(analysis_cfg),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "study_config_sha256": sha256_file(study_config_path),
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "preflight_sha256": sha256_file(preflight_path),
        "artifact_audit": artifact_audit,
        "comparisons": comparisons,
        "gates": gates,
    }
    atomic_write_json(output_root / "analysis.json", analysis)
    (output_root / "analysis.md").write_text(_markdown_report(analysis), encoding="utf-8", newline="\n")
    return analysis
