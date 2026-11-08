"""Prospectively fixed world-level uncertainty; stdlib only, no bank access."""
from __future__ import annotations

import math
import random

from .schema import REPAIR_FAMILIES

FAMILYWISE_ALPHA = 0.05
ENDPOINTS = 48
ENDPOINT_ALPHA = FAMILYWISE_ALPHA / ENDPOINTS
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261005
ENDPOINT_REGISTRY = tuple(
    f"seed{seed}:{family}:{endpoint}"
    for seed in (17, 29)
    for family in ("ledger", "register", "answerability", "grounded_qa", "arithmetic", "instruction", "concise_grounded")
    for endpoint in ("semantic_population_damage", "strict_population_damage", "conditional_semantic_damage")
) + tuple(
    f"seed{seed}:{family}:supported_population_loss"
    for seed in (17, 29) for family in ("answerability", "grounded_qa")
) + tuple(f"seed{seed}:repair_macro_gain_lower" for seed in (17, 29))
assert len(ENDPOINT_REGISTRY) == ENDPOINTS


def binomial_cdf(k, n, probability):
    """Log-sum-exp binomial CDF, stable at the small one-sided tail used here."""
    if type(k) is not int or type(n) is not int or n < 0 or not 0 <= probability <= 1:
        raise ValueError("STOP_INVALID_BINOMIAL_ARGUMENT")
    if k < 0:
        return 0.0
    if k >= n or probability == 0:
        return 1.0
    if probability == 1:
        return 0.0
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * math.log(probability) + (n - i) * math.log1p(-probability)
             for i in range(k + 1)]
    maximum = max(terms)
    return min(1.0, math.exp(maximum) * math.fsum(math.exp(term - maximum) for term in terms))


def clopper_pearson_upper(failures, n, alpha=ENDPOINT_ALPHA):
    """Exact one-sided upper confidence limit. No denominator is inconclusive."""
    if type(failures) is not int or type(n) is not int or not 0 <= failures <= n or not 0 < alpha < 1:
        raise ValueError("STOP_INVALID_EXACT_BOUND_ARGUMENT")
    if n == 0:
        return None
    if failures == n:
        return 1.0
    if failures == 0:
        return -math.expm1(math.log(alpha) / n)
    lower, upper = failures / n, 1.0
    for _ in range(64):
        midpoint = (lower + upper) / 2
        if binomial_cdf(failures, n, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def paired_worlds(baseline, candidate, family):
    """Align canonical worlds and refuse comparison across different instruments."""
    def index(summary):
        items = summary.get("primary", {}).get(family, [])
        result = {}
        for item in items:
            if item.get("variant_id") != "canonical" or item.get("family") != family:
                raise ValueError("STOP_NONCANONICAL_PRIMARY")
            world = item.get("world_id")
            if not world or world in result:
                raise ValueError("STOP_DUPLICATE_OR_EMPTY_WORLD")
            for field in ("semantic_correct", "strict_correct", "answerable", "abstained",
                          "unsupported_answer", "unnecessary_abstention"):
                if type(item.get(field)) is not bool:
                    raise ValueError("STOP_INVALID_SCORE_FLAG")
            result[world] = item
        return result

    base, cand = index(baseline), index(candidate)
    if not base or base.keys() != cand.keys() or baseline.get("split") != candidate.get("split"):
        raise ValueError("STOP_UNPAIRED_WORLD_MEMBERSHIP")
    pairs = []
    for world in sorted(base):
        left, right = base[world], cand[world]
        if any(left.get(field) != right.get(field) for field in (
                "row_id", "answerable", "support_expansion")):
            raise ValueError("STOP_UNPAIRED_WORLD_IDENTITY")
        pairs.append((left, right))
    return pairs


def macro_gain_interval(baseline, candidate, *, draws=BOOTSTRAP_DRAWS,
                        seed=BOOTSTRAP_SEED, alpha=ENDPOINT_ALPHA):
    """Paired family-stratified canonical-world bootstrap percentile lower bound.

    The 20,000-draw frozen implementation retains only 20,000 scalar draws and
    each family's {-1,0,1} differences; no variants inflate sample size.
    """
    if draws != BOOTSTRAP_DRAWS or seed != BOOTSTRAP_SEED or alpha != ENDPOINT_ALPHA:
        raise ValueError("STOP_UNFROZEN_BOOTSTRAP_CONFIGURATION")
    differences = []
    for family in REPAIR_FAMILIES:
        differences.append([int(right["semantic_correct"]) - int(left["semantic_correct"])
                            for left, right in paired_worlds(baseline, candidate, family)])
    point = math.fsum(math.fsum(values) / len(values) for values in differences) / len(differences)
    generator = random.Random(seed)
    samples = []
    for _ in range(draws):
        sample_means = []
        for values in differences:
            sample_means.append(math.fsum(generator.choices(values, k=len(values))) / len(values))
        samples.append(math.fsum(sample_means) / len(differences))
    samples.sort()
    # Conservative lower order statistic: never interpolate upward.
    lower = samples[max(0, math.floor(alpha * draws) - 1)]
    return {"point": point, "lower": lower, "alpha": alpha, "draws": draws,
            "seed": seed, "method": "paired_family_stratified_world_percentile_bootstrap",
            "variant_independent_n": 0, "family_world_n": dict(zip(REPAIR_FAMILIES, map(len, differences)))}


def damage_bounds(pairs):
    n = len(pairs)
    semantic = sum(left["semantic_correct"] and not right["semantic_correct"] for left, right in pairs)
    strict = sum(left["strict_correct"] and not right["strict_correct"] for left, right in pairs)
    baseline_correct = sum(left["semantic_correct"] for left, _ in pairs)
    return {
        "semantic_population_damage": {"failures": semantic, "n": n,
                                       "upper": clopper_pearson_upper(semantic, n)},
        "strict_population_damage": {"failures": strict, "n": n,
                                     "upper": clopper_pearson_upper(strict, n)},
        "conditional_semantic_damage": {"failures": semantic, "n": baseline_correct,
                                        "upper": clopper_pearson_upper(semantic, baseline_correct)},
    }


def supported_population_loss(pairs):
    supported = [(left, right) for left, right in pairs if left["answerable"]]
    failures = sum(left["semantic_correct"] and not right["semantic_correct"] for left, right in supported)
    return {"failures": failures, "n": len(supported),
            "upper": clopper_pearson_upper(failures, len(supported))}
