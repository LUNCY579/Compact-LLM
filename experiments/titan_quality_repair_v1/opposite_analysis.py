"""Development-only factorial diagnostics of the existing eight fixed recipes.

No data/model/file access, outcome-driven search, selection-rule changes or final
decision endpoints live here. Consistency across two seeds generates development
hypotheses; it is neither deterministic proof nor verified model improvement.
"""
from __future__ import annotations

import itertools
import math

from . import registry as fixed_registry
from .schema import EXPERIMENT, FAMILIES, REPAIR_FAMILIES, RETENTION_FAMILIES, SEEDS, digest
from .selection import development_gates
from .statistics import paired_worlds

FACTORS = {
    "prompt_coverage": ("canonical", "diverse"),
    "difficulty": ("regular", "counterfactual"),
    "replay_fraction": (0, 0.2),
}
METRIC_BENEFIT_DIRECTION = {
    "repair_macro_semantic_accuracy": 1,
    "supported_answer_accuracy": 1,
    "unsupported_answer_rate": -1,
    "unnecessary_abstention_rate": -1,
    "retention_population_semantic_damage": -1,
    "retention_population_strict_damage": -1,
}
_ROUNDING_EPSILON = 1e-12


def _direction(value):
    return "positive" if value > _ROUNDING_EPSILON else "negative" if value < -_ROUNDING_EPSILON else "zero"


def _seed_agreement(by_seed):
    agreement = {}
    for metric in METRIC_BENEFIT_DIRECTION:
        directions = [_direction(by_seed[str(seed)][metric]) for seed in SEEDS]
        if directions[0] == directions[1]:
            agreement[metric] = "both_seeds_" + directions[0]
        elif "zero" in directions:
            agreement[metric] = "one_seed_zero"
        else:
            agreement[metric] = "opposed_seed_directions"
    return agreement


def _tradeoffs(effects):
    favorable, unfavorable = [], []
    for metric, beneficial_sign in METRIC_BENEFIT_DIRECTION.items():
        signed = effects[metric] * beneficial_sign
        if signed > _ROUNDING_EPSILON:
            favorable.append(metric)
        elif signed < -_ROUNDING_EPSILON:
            unfavorable.append(metric)
    return {"favorable_metrics": favorable, "unfavorable_metrics": unfavorable,
            "has_opposite_quality_directions": bool(favorable and unfavorable)}


def _metrics(baseline, candidate):
    # Reuse the unchanged selector's complete DEV/paired-world validation. A
    # failed gate remains observable: diagnostics do not filter losing recipes.
    gates = development_gates(baseline, candidate)
    retention_pairs = [pair for family in RETENTION_FAMILIES
                       for pair in paired_worlds(baseline, candidate, family)]
    repair_pairs = [pair for family in REPAIR_FAMILIES
                    for pair in paired_worlds(baseline, candidate, family)]
    supported_n = sum(right["answerable"] for _, right in repair_pairs)
    unsupported_n = len(repair_pairs) - supported_n
    rates = gates["repair_candidate"]
    if any(rates[name] is None for name in ("supported", "unsupported", "unnecessary")):
        raise ValueError("STOP_DIAGNOSTICS_ZERO_METRIC_DENOMINATOR")
    metrics = {
        "repair_macro_semantic_accuracy": math.fsum(
            gates["families"][family]["candidate"]["semantic"] for family in REPAIR_FAMILIES) / 4,
        "supported_answer_accuracy": rates["supported"],
        "unsupported_answer_rate": rates["unsupported"],
        "unnecessary_abstention_rate": rates["unnecessary"],
        "retention_population_semantic_damage": sum(
            left["semantic_correct"] and not right["semantic_correct"] for left, right in retention_pairs) / len(retention_pairs),
        "retention_population_strict_damage": sum(
            left["strict_correct"] and not right["strict_correct"] for left, right in retention_pairs) / len(retention_pairs),
    }
    return {"metrics": metrics, "development_gates_passed": gates["passed"],
            "development_gate_failures": gates["failures"],
            "family_metrics": gates["families"],
            "denominators": {"repair_canonical_worlds": len(repair_pairs), "supported_repair_worlds": supported_n,
                             "unsupported_repair_worlds": unsupported_n,
                             "retention_canonical_worlds": len(retention_pairs), "variant_independent_n": 0},
            "summary_sha256": digest(candidate)}


def _difference(high, low):
    return {metric: high[metric] - low[metric] for metric in METRIC_BENEFIT_DIRECTION}


def _average(effects):
    return {metric: math.fsum(effect[metric] for effect in effects) / len(effects)
            for metric in METRIC_BENEFIT_DIRECTION}


def analyze_opposite_directions(baseline=None, trial_summaries=None, recipe_registry=None):
    """Compare all fixed cells, always level 2 minus level 1, on complete DEV.

    trial_summaries maps R01..R08 to {17: aggregate_rows summary, 29: summary}.
    Empty observations return UNMEASURED with no numerical gains. Populated
    input requires one complete baseline and the exact frozen recipe/seed grid;
    missing cells, unequal worlds and any FINAL instrument fail closed.
    """
    recipes = fixed_registry.build_registry() if recipe_registry is None else fixed_registry.validate_registry(recipe_registry)
    report = {
        "experiment": EXPERIMENT, "analysis_version": 1, "development_only": True,
        "registry_sha256": recipes["registry_sha256"], "seeds": list(SEEDS),
        "factor_levels_in_contrast_order": {factor: list(levels) for factor, levels in FACTORS.items()},
        "metric_benefit_direction": dict(METRIC_BENEFIT_DIRECTION),
        "model_gain_verified": False, "native_execution_authorized": False,
        "changes_selection_rules": False, "changes_final_endpoint_registry": False,
        "pattern_claim": "DEVELOPMENT_HYPOTHESES_ONLY_NOT_DETERMINISTIC_OR_CAPABILITY_PROOF",
        "replication_scope": "Frozen final measures the selected recipe at seeds 17 and 29; it cannot confirm all factorial contrasts.",
        "replay_estimand": "Fixed 2000-example budget: 400 replay occurrences replace 400 fresh repair examples, not an isolated equal-repair-dose replay effect.",
        "limitations": ["Only two training seeds; sign agreement is descriptive, without a generalization or significance claim.",
            "Input summaries are not authenticated native receipts by this pure diagnostic function.",
            "Interactions and directions cannot authorize another final recipe, alter thresholds or expand samples.",
            "Attached contract variants are repeated observations and do not contribute independent worlds."],
    }
    if baseline is not None:
        if not isinstance(baseline, dict) or baseline.get("split") != "dev":
            raise ValueError("STOP_DIAGNOSTICS_REQUIRES_DEVELOPMENT_BASELINE")
        development_gates(baseline, baseline)
    if trial_summaries is None or trial_summaries == {}:
        return {**report, "status": "UNMEASURED", "analyzed_trial_count": 0,
                "baseline_summary_sha256": digest(baseline) if baseline is not None else None,
                "trial_metrics": {}, "matched_cell_contrasts": [], "main_effects": {},
                "pairwise_interactions": [], "triple_interaction": None,
                "sign_reversals": [], "hypothesis_candidates": []}
    if baseline is None:
        raise ValueError("STOP_DIAGNOSTICS_MISSING_BASELINE")
    recipe_ids = [recipe["recipe_id"] for recipe in recipes["recipes"]]
    if not isinstance(trial_summaries, dict) or set(trial_summaries) != set(recipe_ids):
        raise ValueError("STOP_DIAGNOSTICS_INCOMPLETE_EIGHT_CELLS")
    trial_metrics, cell_ids = {}, {}
    for recipe in recipes["recipes"]:
        recipe_id = recipe["recipe_id"]
        observations = trial_summaries[recipe_id]
        if not isinstance(observations, dict) or set(observations) != set(SEEDS):
            raise ValueError("STOP_DIAGNOSTICS_FIXED_SEED_COVERAGE")
        cell = tuple(recipe[factor] for factor in FACTORS)
        if cell in cell_ids:
            raise ValueError("STOP_DIAGNOSTICS_DUPLICATE_FACTOR_CELL")
        cell_ids[cell] = recipe_id
        trial_metrics[recipe_id] = {}
        for seed in SEEDS:
            candidate = observations[seed]
            if not isinstance(candidate, dict) or candidate.get("split") != "dev":
                raise ValueError("STOP_DIAGNOSTICS_FINAL_OR_NONDEVELOPMENT_RESULT")
            trial_metrics[recipe_id][str(seed)] = _metrics(baseline, candidate)

    def cell_recipe(values):
        return cell_ids[tuple(values[factor] for factor in FACTORS)]

    def cell_metrics(values, seed):
        return trial_metrics[cell_recipe(values)][str(seed)]["metrics"]

    contrasts, main_effects, reversals, hypotheses = [], {}, [], []
    for factor, levels in FACTORS.items():
        others = [name for name in FACTORS if name != factor]
        effects = []
        for held_levels in itertools.product(*(FACTORS[name] for name in others)):
            held = dict(zip(others, held_levels))
            low, high = {**held, factor: levels[0]}, {**held, factor: levels[1]}
            by_seed = {str(seed): _difference(cell_metrics(high, seed), cell_metrics(low, seed)) for seed in SEEDS}
            item = {"factor": factor, "held_factors": held, "low_recipe_id": cell_recipe(low),
                    "high_recipe_id": cell_recipe(high), "effect_definition": "LEVEL_2_MINUS_LEVEL_1",
                    "by_seed": by_seed, "seed_agreement": _seed_agreement(by_seed),
                    "tradeoffs_by_seed": {str(seed): _tradeoffs(by_seed[str(seed)]) for seed in SEEDS}}
            effects.append(item)
            contrasts.append(item)
        by_seed = {str(seed): _average([item["by_seed"][str(seed)] for item in effects]) for seed in SEEDS}
        main_effects[factor] = {"definition": "Equal mean of the four matched other-factor-cell contrasts",
                                "by_seed": by_seed, "seed_agreement": _seed_agreement(by_seed),
                                "tradeoffs_by_seed": {str(seed): _tradeoffs(by_seed[str(seed)]) for seed in SEEDS}}
        for metric in METRIC_BENEFIT_DIRECTION:
            for seed in SEEDS:
                positive = [item["held_factors"] for item in effects if _direction(item["by_seed"][str(seed)][metric]) == "positive"]
                negative = [item["held_factors"] for item in effects if _direction(item["by_seed"][str(seed)][metric]) == "negative"]
                if positive and negative:
                    reversals.append({"factor": factor, "metric": metric, "seed": seed,
                                      "positive_held_cells": positive, "negative_held_cells": negative,
                                      "interpretation": "Conditional direction reversal; averaging is not a universal improvement rule."})
            agreement = main_effects[factor]["seed_agreement"][metric]
            if agreement in {"both_seeds_positive", "both_seeds_negative"}:
                direction = 1 if agreement == "both_seeds_positive" else -1
                hypotheses.append({"factor": factor, "metric": metric,
                    "averaged_direction": "higher" if direction > 0 else "lower",
                    "favorable_direction": direction == METRIC_BENEFIT_DIRECTION[metric],
                    "status": "CONSISTENT_DEV_HYPOTHESIS_PENDING_PROSPECTIVE_REPLICATION",
                    "model_gain_verified": False,
                    "has_conditional_sign_reversal": any(item["factor"] == factor and item["metric"] == metric for item in reversals),
                    "has_quality_tradeoff": any(main_effects[factor]["tradeoffs_by_seed"][str(seed)]["has_opposite_quality_directions"] for seed in SEEDS)})

    interactions = []
    for first, second in itertools.combinations(FACTORS, 2):
        other = next(factor for factor in FACTORS if factor not in {first, second})
        for held_level in FACTORS[other]:
            by_seed = {}
            for seed in SEEDS:
                cells = {(a, b): cell_metrics({first: FACTORS[first][a], second: FACTORS[second][b], other: held_level}, seed)
                         for a, b in itertools.product(range(2), repeat=2)}
                by_seed[str(seed)] = {metric: math.fsum((cells[1, 1][metric], -cells[0, 1][metric],
                                                        -cells[1, 0][metric], cells[0, 0][metric]))
                                     for metric in METRIC_BENEFIT_DIRECTION}
            interactions.append({"factors": [first, second], "held_factors": {other: held_level},
                "definition": "(high-first minus low-first at high-second) minus (high-first minus low-first at low-second)",
                "by_seed": by_seed, "seed_agreement": _seed_agreement(by_seed)})
    triple = {}
    for seed in SEEDS:
        cells = {indices: cell_metrics({factor: FACTORS[factor][index] for factor, index in zip(FACTORS, indices)}, seed)
                 for indices in itertools.product(range(2), repeat=3)}
        triple[str(seed)] = {metric: math.fsum(((-1) ** (3 - sum(indices))) * values[metric]
                                              for indices, values in cells.items())
                            for metric in METRIC_BENEFIT_DIRECTION}
    return {**report, "status": "COMPLETE_DEVELOPMENT_DIAGNOSTICS_NO_VERIFIED_GAIN",
            "analyzed_trial_count": 16, "baseline_summary_sha256": digest(baseline),
            "trial_metrics": trial_metrics, "matched_cell_contrasts": contrasts,
            "main_effects": main_effects, "pairwise_interactions": interactions,
            "triple_interaction": {"definition": "Alternating signed contrast of all eight cells",
                                    "by_seed": triple, "seed_agreement": _seed_agreement(triple)},
            "sign_reversals": reversals, "hypothesis_candidates": hypotheses}
