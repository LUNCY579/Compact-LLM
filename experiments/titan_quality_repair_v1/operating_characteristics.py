"""Prospective exact audit of existing final margins; no observations or I/O.

The frozen selector and its 48 endpoints remain unchanged. This module evaluates
declared binomial marginal models and dependence-safe bounds, never model power
from fabricated outcomes or an independent-endpoint product.
"""
from __future__ import annotations

import math

from .schema import EXPERIMENT, digest
from .statistics import ENDPOINT_ALPHA, ENDPOINTS, FAMILYWISE_ALPHA, binomial_cdf, clopper_pearson_upper

FINAL_WORLDS_PER_FAMILY = 512
POPULATION_CEILING = 0.05
CONDITIONAL_CEILING = 0.10
LOSS_PROBABILITIES = (0, .005, .01, .02, .03, .04, .05)
SOURCE_CORRECT_DENOMINATORS = (0, 32, 64, 66, 128, 256, 512)
MATCHING_MARGIN_ENDPOINTS = 14  # two fixed seeds × seven families, one margin type


def _valid_rule(n, ceiling, alpha):
    if type(n) is not int or n < 0 or not 0 < ceiling < 1 or not 0 < alpha < 1:
        raise ValueError("STOP_INVALID_PROSPECTIVE_MARGIN_RULE")


def maximum_allowed_failures(n, ceiling, alpha=ENDPOINT_ALPHA):
    """Largest k with exact upper(k,n) <= ceiling; -1 means no k can pass.

    No denominator returns None, matching the frozen inconclusive-bound policy.
    Binary search uses monotonicity of the exact upper bound in observed losses.
    """
    _valid_rule(n, ceiling, alpha)
    if n == 0:
        return None
    low, high = -1, n  # k=n has upper=1 and cannot pass a ceiling below one.
    while high - low > 1:
        middle = (low + high) // 2
        if clopper_pearson_upper(middle, n, alpha) <= ceiling:
            low = middle
        else:
            high = middle
    return low


def minimum_n_with_zero_losses(ceiling, alpha=ENDPOINT_ALPHA):
    """Smallest positive n permitting zero-loss acceptance, not a sample change."""
    _valid_rule(1, ceiling, alpha)
    candidate = max(1, math.ceil(math.log(alpha) / math.log1p(-ceiling)))
    while clopper_pearson_upper(0, candidate, alpha) > ceiling:
        candidate += 1
    while candidate > 1 and clopper_pearson_upper(0, candidate - 1, alpha) <= ceiling:
        candidate -= 1
    return candidate


def dependence_safe_all_pass_bounds(probabilities):
    """Union-bound lower and minimum-marginal upper for exactly 14 matching margins.

    The events may share worlds, baseline errors and seed outcomes. These are
    mathematical bounds, not joint acceptance-power estimates.
    """
    values = list(probabilities)
    if len(values) != MATCHING_MARGIN_ENDPOINTS or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("STOP_REQUIRE_14_VALID_MARGINAL_PROBABILITIES")
    return {"matching_margin_endpoints": MATCHING_MARGIN_ENDPOINTS,
            "all_pass_lower_union_bound": max(0.0, 1 - math.fsum(1 - value for value in values)),
            "all_pass_upper_minimum_marginal": min(values),
            "independence_assumed": False, "joint_power_estimate": None}


def _curve(n, ceiling):
    maximum = maximum_allowed_failures(n, ceiling)
    curve = []
    for probability in LOSS_PROBABILITIES:
        passing = 0.0 if maximum is None or maximum < 0 else binomial_cdf(maximum, n, probability)
        curve.append({"declared_true_loss_probability_per_this_denominator": probability,
                      "exact_binomial_marginal_acceptance_probability": passing,
                      "all_14_matching_margins_bounds_under_equal_declared_marginals":
                          dependence_safe_all_pass_bounds([passing] * MATCHING_MARGIN_ENDPOINTS)})
    return {"n": n, "ceiling": ceiling, "endpoint_alpha": ENDPOINT_ALPHA,
            "maximum_allowed_failure_count": maximum,
            "zero_loss_upper_bound": clopper_pearson_upper(0, n),
            "minimum_n_even_with_zero_losses": minimum_n_with_zero_losses(ceiling),
            "status": "INCONCLUSIVE_ZERO_DENOMINATOR" if n == 0 else
                      "NO_FAILURE_COUNT_CAN_PASS" if maximum < 0 else "DECLARED_MARGINAL_RULE_AUDITED",
            "marginal_curve": curve}


def audit():
    """JSON-compatible prospective record of the already frozen bound rules."""
    population = _curve(FINAL_WORLDS_PER_FAMILY, POPULATION_CEILING)
    conditional = [_curve(n, CONDITIONAL_CEILING) for n in SOURCE_CORRECT_DENOMINATORS]
    examples = []
    population_maximum = population["maximum_allowed_failure_count"]
    for source_n in SOURCE_CORRECT_DENOMINATORS:
        feasible_counts = sorted({0, min(1, source_n), min(population_maximum, source_n)})
        for losses in feasible_counts:
            population_upper = clopper_pearson_upper(losses, FINAL_WORLDS_PER_FAMILY)
            conditional_upper = clopper_pearson_upper(losses, source_n)
            population_pass = population_upper is not None and population_upper <= POPULATION_CEILING
            conditional_pass = conditional_upper is not None and conditional_upper <= CONDITIONAL_CEILING
            examples.append({"source_correct_n": source_n, "same_actual_damage_count": losses,
                "population_n": FINAL_WORLDS_PER_FAMILY, "population_upper": population_upper,
                "conditional_upper": conditional_upper, "population_bound_passes": population_pass,
                "conditional_bound_passes": conditional_pass,
                "population_pass_conditional_fail": population_pass and not conditional_pass,
                "conditional_status": "INCONCLUSIVE" if source_n == 0 else "PASS" if conditional_pass else "FAIL"})
    rules = {"final_worlds_per_family": FINAL_WORLDS_PER_FAMILY,
             "population_damage_upper_ceiling": POPULATION_CEILING,
             "conditional_source_correct_damage_upper_ceiling": CONDITIONAL_CEILING,
             "familywise_alpha": FAMILYWISE_ALPHA, "bonferroni_endpoints": ENDPOINTS,
             "endpoint_alpha": ENDPOINT_ALPHA,
             "declared_true_loss_probability_grid": list(LOSS_PROBABILITIES),
             "declared_source_correct_denominator_grid": list(SOURCE_CORRECT_DENOMINATORS)}
    return {"experiment": EXPERIMENT, "audit_version": 1,
        "status": "PROSPECTIVE_DESIGN_MECHANICS_NO_MODEL_EVIDENCE",
        "classification": "SYNTHETIC_DECLARED_BINOMIAL_MODELS_NOT_OBSERVED_OUTCOMES",
        "rules": rules, "rules_sha256": digest(rules),
        "semantic_and_strict_population_margins": population,
        "population_margin_note": "Semantic and strict margins use identical count rules but may have different actual losses; this curve does not make them independent.",
        "conditional_semantic_margins_by_declared_source_correct_n": conditional,
        "same_damage_count_examples": examples,
        "probability_interpretation": "For population margins p is P(source-correct AND candidate-wrong) over all worlds. For conditional margins p is P(candidate-wrong GIVEN source-correct), conditional on the declared fixed denominator. The same numerical p in two curves is not one joint model.",
        "dependence_scope": "Bounds apply separately to 14 corresponding margins (2 seeds x 7 families). They do not multiply probabilities across semantic, strict, conditional, supported-loss, gain, risk or retention endpoints.",
        "three_percentage_point_floor_note": "At a true gain exactly at the 3-percentage-point point-estimate floor, a nondegenerate approximately unbiased asymptotically symmetric estimator exceeds its center approximately half the time, before other gates. Discreteness, skew and dependence can change finite-sample behavior; this is not a numeric joint-power calculation.",
        "joint_power_estimate": None, "monte_carlo_draws": 0,
        "observed_model_outcomes_used": 0, "native_model_loads": 0, "native_model_forwards": 0,
        "final_payload_accessed": False, "quality_claim": False,
        "frozen_thresholds_changed": False, "sample_sizes_changed": False,
        "design_interpretation": "Any restrictive acceptance regions are prospective design limitations, not Rineya failures and not authorization to change frozen rules after outcomes."}
