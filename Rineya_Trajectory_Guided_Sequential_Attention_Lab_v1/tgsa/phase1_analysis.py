"""Frozen paired analysis for the TGSA Phase-1 model study.

The primary estimand is a macro-average over registered non-guardrail strata,
then over paired training seeds. Bootstrap draws resample seeds and, within each
selected seed, examples inside each stratum. This keeps arm, seed, task, and
example pairing intact while avoiding domination by a populous stratum.
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
) -> dict[int, dict[str, list[float]]]:
    if candidate_by_seed.keys() != baseline_by_seed.keys():
        raise ValueError("candidate and baseline seed sets differ")
    result: dict[int, dict[str, list[float]]] = {}
    for seed in sorted(candidate_by_seed):
        paired = _validate_pair(candidate_by_seed[seed], baseline_by_seed[seed])
        strata: dict[str, list[float]] = {}
        for candidate, baseline in paired:
            family = str(candidate["family"])
            if include_families is not None and family not in include_families:
                continue
            stratum = str(candidate["stratum"])
            strata.setdefault(stratum, []).append(score(candidate, baseline))
        if not strata:
            raise ValueError(f"family filter removed all examples for seed {seed}")
        result[seed] = dict(sorted(strata.items()))
    reference_strata = set(next(iter(result.values())))
    for seed, strata in result.items():
        if set(strata) != reference_strata:
            raise ValueError(f"stratum set differs for seed {seed}")
    return result


def hierarchical_paired_bootstrap(
    values: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> BootstrapEstimate:
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
    for seed_id in seed_ids:
        if sorted(values[seed_id]) != strata:
            raise ValueError("all seeds must share the same strata")
        if any(not values[seed_id][stratum] for stratum in strata):
            raise ValueError("all seed/stratum cells must be non-empty")

    def cell_macro(seed_id: int) -> float:
        return sum(sum(values[seed_id][stratum]) / len(values[seed_id][stratum]) for stratum in strata) / len(strata)

    observed = sum(cell_macro(seed_id) for seed_id in seed_ids) / len(seed_ids)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected_seeds = [rng.choice(seed_ids) for _ in seed_ids]
        seed_draws: list[float] = []
        for seed_id in selected_seeds:
            stratum_draws: list[float] = []
            for stratum in strata:
                cell = values[seed_id][stratum]
                resampled = [rng.choice(cell) for _ in cell]
                stratum_draws.append(sum(resampled) / len(resampled))
            seed_draws.append(sum(stratum_draws) / len(stratum_draws))
        draws.append(sum(seed_draws) / len(seed_draws))
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
                sum(sum(cell) / len(cell) for cell in strata.values()) / len(strata)
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
) -> dict[str, object]:
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
    return {
        "accuracy_delta": hierarchical_paired_bootstrap(
            accuracy_values,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed,
        ).to_dict(),
        "answer_nll_improvement": hierarchical_paired_bootstrap(
            nll_values,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed + 1,
        ).to_dict(),
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
            "information-matched sequential-history control, and that disrupting signal alignment "
            "reduced the learned-delta model. It does not establish Qwen-scale, natural-language, "
            "long-context, efficiency, novelty, or publication-primary claims.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_study(
    *,
    study_config_path: Path,
    output_root: Path,
    allow_partial: bool = False,
) -> dict[str, object]:
    study = load_study_config(study_config_path)
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
    analysis_cfg = cast(Mapping[str, object], study["analysis"])
    bootstrap_samples = int(analysis_cfg.get("bootstrap_samples", 10000))
    confidence = float(analysis_cfg.get("confidence", 0.95))
    bootstrap_seed = int(analysis_cfg.get("bootstrap_seed", 91273))
    primary_families = {str(value) for value in analysis_cfg["primary_families"]}
    guardrail_family = str(analysis_cfg.get("guardrail_family", "direct_lookup"))

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
        intervention="roll_batch",
    )

    comparisons: dict[str, object] = {
        "learned_delta_minus_history_control": _estimate_comparison(
            learned,
            history,
            include_families=primary_families,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        ),
        "learned_delta_minus_cascade": _estimate_comparison(
            learned,
            cascade,
            include_families=primary_families,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 1000,
        ),
        "aligned_delta_minus_batch_rolled_delta": _estimate_comparison(
            learned,
            rolled,
            include_families=primary_families,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 2000,
        ),
        "direct_lookup_delta_minus_history_control": _estimate_comparison(
            learned,
            history,
            include_families={guardrail_family},
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 3000,
        ),
    }

    history_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["learned_delta_minus_history_control"])["accuracy_delta"])
    cascade_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["learned_delta_minus_cascade"])["accuracy_delta"])
    alignment_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["aligned_delta_minus_batch_rolled_delta"])["accuracy_delta"])
    guardrail_estimate = cast(Mapping[str, object], cast(Mapping[str, object], comparisons["direct_lookup_delta_minus_history_control"])["accuracy_delta"])

    min_history = float(analysis_cfg.get("minimum_history_delta", 0.02))
    min_cascade = float(analysis_cfg.get("minimum_cascade_delta", 0.02))
    min_alignment = float(analysis_cfg.get("minimum_alignment_drop", 0.015))
    noninferiority = float(analysis_cfg.get("direct_lookup_noninferiority_margin", -0.01))
    gates: dict[str, dict[str, object]] = {
        "full_registered_matrix": {
            "pass": is_full,
            "reason": f"selected arms={selected_arms}; selected seeds={selected_seeds}",
        },
        "finite_difference_beats_information_matched_history": {
            "pass": float(history_estimate["observed"]) >= min_history
            and float(history_estimate["confidence_low"]) > 0.0,
            "reason": (
                f"mean={float(history_estimate['observed']):+.4f}, "
                f"lower={float(history_estimate['confidence_low']):+.4f}, "
                f"threshold={min_history:+.4f}"
            ),
        },
        "finite_difference_beats_cascade": {
            "pass": float(cascade_estimate["observed"]) >= min_cascade
            and float(cascade_estimate["confidence_low"]) > 0.0,
            "reason": (
                f"mean={float(cascade_estimate['observed']):+.4f}, "
                f"lower={float(cascade_estimate['confidence_low']):+.4f}, "
                f"threshold={min_cascade:+.4f}"
            ),
        },
        "trajectory_alignment_is_causal": {
            "pass": float(alignment_estimate["observed"]) >= min_alignment
            and float(alignment_estimate["confidence_low"]) > 0.0,
            "reason": (
                f"aligned-minus-rolled mean={float(alignment_estimate['observed']):+.4f}, "
                f"lower={float(alignment_estimate['confidence_low']):+.4f}, "
                f"threshold={min_alignment:+.4f}"
            ),
        },
        "direct_lookup_noninferiority": {
            "pass": float(guardrail_estimate["confidence_low"]) >= noninferiority,
            "reason": (
                f"lower={float(guardrail_estimate['confidence_low']):+.4f}, "
                f"margin={noninferiority:+.4f}"
            ),
        },
        "paired_initialization_and_artifact_integrity": {
            "pass": True,
            "reason": "all registered seed audits and recorded file hashes validated",
        },
    }

    all_pass = all(bool(gate["pass"]) for gate in gates.values())
    if not is_full:
        decision = "smoke_only"
    elif all_pass:
        decision = "supports_phase1_mechanism_hypothesis"
    elif float(history_estimate["confidence_high"]) < 0.0 or float(alignment_estimate["confidence_high"]) < 0.0:
        decision = "contradicted"
    else:
        decision = "inconclusive"

    analysis: dict[str, object] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
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
