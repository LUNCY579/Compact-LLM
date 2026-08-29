"""Outcome-blind power simulation for the registered Phase-1 gates.

Ten thousand bootstrap draws do not manufacture information: the design has only
five independent training seeds. Before the matrix is dispatched we therefore
simulate the registered decision rule under plausible seed variance and report
the probability that each gate fires, so the launch decision is made with a
known — not assumed — detection probability
(``PHASE1_LAUNCH_AUDIT.md`` finding P0-7).

Nothing here reads a checkpoint, a prediction file, or any real outcome. The
simulator generates synthetic paired differences from an explicit generative
model, so it can be run and frozen before launch without unblinding anything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .phase1_analysis import resolve_bootstrap

POWER_SCHEMA_VERSION = "tgsa.phase1-power-simulation.v1"


@dataclass(frozen=True, slots=True)
class PowerScenario:
    """One synthetic world the registered decision rule is run against."""

    name: str
    true_effect: float
    seed_sd: float
    stratum_sd: float
    disagreement_rate: float = 0.30

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "true_effect": self.true_effect,
            "seed_sd": self.seed_sd,
            "stratum_sd": self.stratum_sd,
            "disagreement_rate": self.disagreement_rate,
        }


def _simulate_values(
    scenario: PowerScenario,
    *,
    seeds: Sequence[int],
    strata: Sequence[str],
    examples_per_stratum: int,
    rng: random.Random,
) -> dict[int, dict[str, dict[str, float]]]:
    """Draw one synthetic crossed seed x stratum x example paired difference set.

    Paired accuracy differences take values in {-1, 0, +1}. The per-cell mean
    effect is the true effect plus a seed offset and a stratum offset; an
    example-level offset shared across seeds reproduces the correlation that
    makes the crossed structure matter.
    """

    seed_offsets = {seed: rng.gauss(0.0, scenario.seed_sd) for seed in seeds}
    stratum_offsets = {stratum: rng.gauss(0.0, scenario.stratum_sd) for stratum in strata}
    # Example difficulty is a property of the shared test split, so its offset is
    # drawn once and applied to every seed.
    example_offsets = {
        stratum: [rng.gauss(0.0, scenario.stratum_sd) for _ in range(examples_per_stratum)]
        for stratum in strata
    }
    values: dict[int, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        per_stratum: dict[str, dict[str, float]] = {}
        for stratum in strata:
            cell: dict[str, float] = {}
            for index in range(examples_per_stratum):
                mean = (
                    scenario.true_effect
                    + seed_offsets[seed]
                    + stratum_offsets[stratum]
                    + example_offsets[stratum][index]
                )
                # Map the per-example advantage onto the {-1, 0, +1} support of a
                # paired correctness difference, calibrated so that the expected
                # value IS `mean`. `disagreement_rate` is the fraction of examples
                # on which the two arms disagree at all:
                #   P(+1) + P(-1) = disagreement_rate
                #   P(+1) - P(-1) = mean
                rate = scenario.disagreement_rate
                positive = max(0.0, min(rate, 0.5 * (rate + mean)))
                negative = max(0.0, min(rate, rate - positive))
                draw = rng.random()
                if draw < positive:
                    value = 1.0
                elif draw < positive + negative:
                    value = -1.0
                else:
                    value = 0.0
                cell[f"ex_{stratum}_{index:05d}"] = value
            per_stratum[stratum] = cell
        values[seed] = per_stratum
    return values


def simulate_power(
    *,
    scenarios: Sequence[PowerScenario],
    seeds: Sequence[int],
    strata_count: int,
    examples_per_stratum: int,
    threshold: float,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_scheme: str,
    trials: int,
    rng_seed: int,
) -> dict[str, object]:
    """Estimate P(gate fires) for each scenario under the registered rule."""

    estimator = resolve_bootstrap(bootstrap_scheme)
    strata = [f"stratum_{index}" for index in range(strata_count)]
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        rng = random.Random(rng_seed + abs(hash(scenario.name)) % 10_000)
        fired = 0
        observed_values: list[float] = []
        lower_bounds: list[float] = []
        for trial in range(trials):
            values = _simulate_values(
                scenario,
                seeds=seeds,
                strata=strata,
                examples_per_stratum=examples_per_stratum,
                rng=rng,
            )
            estimate = estimator(
                values,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=rng_seed + trial,
            )
            observed_values.append(estimate.observed)
            lower_bounds.append(estimate.confidence_low)
            # This is exactly the registered gate: observed effect at or above
            # the threshold AND a lower confidence bound strictly above zero.
            if estimate.observed >= threshold and estimate.confidence_low > 0.0:
                fired += 1
        results.append(
            {
                "scenario": scenario.to_dict(),
                "trials": trials,
                "gate_fire_rate": fired / trials,
                "mean_observed_effect": sum(observed_values) / len(observed_values),
                "mean_lower_bound": sum(lower_bounds) / len(lower_bounds),
            }
        )
    return {
        "schema_version": POWER_SCHEMA_VERSION,
        "bootstrap_scheme": bootstrap_scheme,
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "threshold": threshold,
        "seeds": list(seeds),
        "strata_count": strata_count,
        "examples_per_stratum": examples_per_stratum,
        "trials": trials,
        "rng_seed": rng_seed,
        "scenarios": results,
        "interpretation": (
            "Outcome-blind simulation of the registered decision rule. A null scenario "
            "(true_effect=0) reports the false-positive rate; a scenario at the registered "
            "threshold reports detection probability. No real outcome is read."
        ),
    }


def default_scenarios(threshold: float) -> tuple[PowerScenario, ...]:
    """Registered scenario ladder: null, threshold, and a comfortable effect."""

    return (
        PowerScenario(name="null", true_effect=0.0, seed_sd=0.02, stratum_sd=0.05),
        PowerScenario(name="at_threshold", true_effect=threshold, seed_sd=0.02, stratum_sd=0.05),
        PowerScenario(name="double_threshold", true_effect=2.0 * threshold, seed_sd=0.02, stratum_sd=0.05),
        PowerScenario(
            name="at_threshold_high_seed_variance",
            true_effect=threshold,
            seed_sd=0.06,
            stratum_sd=0.05,
        ),
    )


def render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# TGSA Phase 1 outcome-blind power simulation",
        "",
        f"**Bootstrap scheme:** `{report['bootstrap_scheme']}`  ",
        f"**Registered threshold:** {float(report['threshold']):+.4f}  ",
        f"**Seeds:** {report['seeds']}  ",
        f"**Trials per scenario:** {report['trials']}",
        "",
        "| Scenario | True effect | Seed SD | Gate fire rate | Mean observed | Mean lower bound |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for entry in report["scenarios"]:  # type: ignore[index]
        scenario = entry["scenario"]
        lines.append(
            f"| `{scenario['name']}` | {float(scenario['true_effect']):+.4f} | "
            f"{float(scenario['seed_sd']):.3f} | {float(entry['gate_fire_rate']):.3f} | "
            f"{float(entry['mean_observed_effect']):+.4f} | "
            f"{float(entry['mean_lower_bound']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "The `null` row is the empirical false-positive rate of the registered gate. ",
            "The `at_threshold` row is its detection probability at the registered minimum ",
            "effect. A low detection probability is a design finding, not a reason to move ",
            "the threshold after launch.",
            "",
        ]
    )
    return "\n".join(lines)
