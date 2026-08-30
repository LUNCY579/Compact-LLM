"""Paired factorial analysis for TGSA x multi-token prediction.

The estimands are fixed by design rather than chosen after outcomes are visible.
Every seed must contain the complete four-cell block.  Missing cells fail closed
because an unpaired comparison would confound seed variance with the mechanism.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

PRIMARY_ARM_ORDER = ("parallel_ar", "parallel_mtp", "tgsa_ar", "tgsa_mtp")
EFFECT_NAMES = (
    "mtp_on_parallel",
    "mtp_on_tgsa",
    "tgsa_without_mtp",
    "tgsa_with_mtp",
    "interaction",
    "combined_vs_parallel_ar",
)


@dataclass(frozen=True, slots=True)
class IntervalSummary:
    mean: float
    lower: float
    upper: float
    per_seed: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "per_seed": list(self.per_seed),
        }


def _metric_value(record: Mapping[str, object], metric: str) -> float:
    if metric in record:
        return float(record[metric])
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping) and metric in metrics:
        return float(metrics[metric])
    raise KeyError(f"record has no metric {metric!r}")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_bootstrap_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 20260830,
) -> IntervalSummary:
    if not values:
        raise ValueError("paired bootstrap requires at least one seed effect")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if samples <= 0:
        raise ValueError("samples must be positive")
    values = tuple(float(value) for value in values)
    mean = statistics.fmean(values)
    if len(values) == 1:
        return IntervalSummary(mean=mean, lower=mean, upper=mean, per_seed=values)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(statistics.fmean(resample))
    draws.sort()
    alpha = 1.0 - confidence
    return IntervalSummary(
        mean=mean,
        lower=_percentile(draws, alpha / 2.0),
        upper=_percentile(draws, 1.0 - alpha / 2.0),
        per_seed=values,
    )


def paired_factorial_effects(
    records: Sequence[Mapping[str, object]],
    *,
    metric: str,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260830,
) -> dict[str, object]:
    """Compute preregistered main/simple effects and the TGSA x MTP interaction.

    Signs are oriented so positive always means improvement.  For a lower-is-
    better metric such as serial rounds or NLL, raw differences are multiplied
    by -1 before summarization.
    """

    by_seed: dict[int, dict[str, float]] = {}
    duplicates: list[tuple[int, str]] = []
    for record in records:
        seed = int(record["seed"])
        arm = str(record["arm"])
        if arm not in PRIMARY_ARM_ORDER:
            continue
        if arm in by_seed.setdefault(seed, {}):
            duplicates.append((seed, arm))
        by_seed[seed][arm] = _metric_value(record, metric)
    if duplicates:
        raise ValueError(f"duplicate primary cells: {duplicates}")
    if not by_seed:
        raise ValueError("no primary factorial records were supplied")
    incomplete = {
        seed: sorted(set(PRIMARY_ARM_ORDER) - set(cells))
        for seed, cells in by_seed.items()
        if set(cells) != set(PRIMARY_ARM_ORDER)
    }
    if incomplete:
        raise ValueError(f"incomplete paired seed blocks: {incomplete}")

    sign = 1.0 if higher_is_better else -1.0
    per_effect: dict[str, list[float]] = {name: [] for name in EFFECT_NAMES}
    per_seed_rows: list[dict[str, object]] = []
    for seed in sorted(by_seed):
        cells = by_seed[seed]
        p_ar = cells["parallel_ar"]
        p_mtp = cells["parallel_mtp"]
        t_ar = cells["tgsa_ar"]
        t_mtp = cells["tgsa_mtp"]
        effects = {
            "mtp_on_parallel": sign * (p_mtp - p_ar),
            "mtp_on_tgsa": sign * (t_mtp - t_ar),
            "tgsa_without_mtp": sign * (t_ar - p_ar),
            "tgsa_with_mtp": sign * (t_mtp - p_mtp),
            "interaction": sign * ((t_mtp - t_ar) - (p_mtp - p_ar)),
            "combined_vs_parallel_ar": sign * (t_mtp - p_ar),
        }
        for name, value in effects.items():
            per_effect[name].append(value)
        per_seed_rows.append(
            {
                "seed": seed,
                "cells": dict(cells),
                "effects": effects,
            }
        )

    summaries = {
        name: paired_bootstrap_interval(
            values,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        ).to_dict()
        for index, (name, values) in enumerate(per_effect.items())
    }
    return {
        "schema_version": "tgsa.mtp-factorial-analysis.v1",
        "metric": metric,
        "higher_is_better": higher_is_better,
        "positive_means": "improvement",
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "paired_seed_count": len(by_seed),
        "arm_order": list(PRIMARY_ARM_ORDER),
        "estimands": {
            "mtp_on_parallel": "parallel_mtp - parallel_ar",
            "mtp_on_tgsa": "tgsa_mtp - tgsa_ar",
            "tgsa_without_mtp": "tgsa_ar - parallel_ar",
            "tgsa_with_mtp": "tgsa_mtp - parallel_mtp",
            "interaction": "(tgsa_mtp - tgsa_ar) - (parallel_mtp - parallel_ar)",
            "combined_vs_parallel_ar": "tgsa_mtp - parallel_ar",
        },
        "effects": summaries,
        "per_seed": per_seed_rows,
    }


def evaluate_complementarity_gates(
    *,
    exact_decode_equivalence: bool,
    main_path_initialization_neutral: bool,
    evidence_integrity: bool,
    competence_go: bool,
    quality_analysis: Mapping[str, object],
    efficiency_analysis: Mapping[str, object],
    quality_noninferiority_margin: float,
    minimum_serial_round_gain: float,
    require_positive_interaction: bool = False,
) -> dict[str, object]:
    """Apply fixed gates to already-frozen paired analyses.

    The gate intentionally separates **usable complementarity** from a positive
    statistical interaction.  MTP can compensate for TGSA latency even when its
    gain is additive rather than super-additive.  A positive interaction may be
    required by a later preregistration, but it is not silently assumed here.
    """

    quality_effects = quality_analysis["effects"]
    efficiency_effects = efficiency_analysis["effects"]
    combined_quality_lower = float(quality_effects["combined_vs_parallel_ar"]["lower"])
    mtp_on_tgsa_efficiency_lower = float(efficiency_effects["mtp_on_tgsa"]["lower"])
    interaction_lower = float(efficiency_effects["interaction"]["lower"])

    checks = {
        "exact_decode_equivalence": bool(exact_decode_equivalence),
        "main_path_initialization_neutral": bool(main_path_initialization_neutral),
        "evidence_integrity": bool(evidence_integrity),
        "competence_go": bool(competence_go),
        "combined_quality_noninferior": combined_quality_lower >= -quality_noninferiority_margin,
        "mtp_reduces_tgsa_serial_rounds": (
            mtp_on_tgsa_efficiency_lower >= minimum_serial_round_gain
        ),
        "positive_efficiency_interaction": interaction_lower > 0.0,
    }
    required = [
        "exact_decode_equivalence",
        "main_path_initialization_neutral",
        "evidence_integrity",
        "competence_go",
        "combined_quality_noninferior",
        "mtp_reduces_tgsa_serial_rounds",
    ]
    if require_positive_interaction:
        required.append("positive_efficiency_interaction")
    verdict = "GO" if all(checks[name] for name in required) else "NO_GO"
    return {
        "schema_version": "tgsa.mtp-complementarity-gate.v1",
        "verdict": verdict,
        "checks": checks,
        "required_checks": required,
        "thresholds": {
            "quality_noninferiority_margin": quality_noninferiority_margin,
            "minimum_serial_round_gain": minimum_serial_round_gain,
            "require_positive_interaction": require_positive_interaction,
        },
        "observed_bounds": {
            "combined_quality_lower": combined_quality_lower,
            "mtp_on_tgsa_efficiency_lower": mtp_on_tgsa_efficiency_lower,
            "efficiency_interaction_lower": interaction_lower,
        },
        "claim_scope": (
            "GO supports a quality-preserving reduction in serial decoding rounds "
            "for the tested regime; it does not establish wall-clock speedup, broad "
            "capability superiority, or scaling to other model sizes"
        ),
    }
