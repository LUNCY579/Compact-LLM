"""Exploratory Monte Carlo intervals from declared paired-cluster metadata.

Clusters are sampled within each fixed family. Every sampled cluster contributes
all its cases, so each family replicate is a case-weighted ratio; the final
statistic is the equal-family macro mean. Declared cluster IDs do not establish
semantic ancestry or scientific independence. No data, model or file is loaded,
and this prototype selects no acceptance rule or practical margin.

Percentile intervals describe the empirical resampling distribution, not exact
frequentist coverage. In particular, zero observed differences can produce [0, 0]
without ruling out unobserved future harms or establishing quality/native authority.
Rank thresholds use the exact ratio of the supplied binary-float confidence;
this is distinct from interpreting a printed decimal as an exact rational.
Resampled rates themselves use floating-point arithmetic.
"""

from __future__ import annotations

import math
import random

MAX_ROWS = 384
MAX_FAMILIES = 16
MAX_ID_CHARACTERS = 128
MIN_DRAWS = 1000
MAX_DRAWS = 50000
MAX_SEED = 2**63 - 1

_ENROLLMENT_KEYS = frozenset(("case_id", "family", "cluster_id"))
_OUTCOME_KEYS = frozenset(("case_id", "complete", "parent_correct", "candidate_correct"))
_COUNT_KEYS = ("n", "parent_correct", "candidate_correct", "repairs", "losses", "paired_delta")


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= MAX_ID_CHARACTERS:
        raise ValueError(name + " must be a nonempty plain string of at most 128 characters")


def _record(value: dict, keys: frozenset[str], name: str) -> None:
    if type(value) is not dict or not all(type(key) is str for key in value) or set(value) != keys:
        raise ValueError(name + " must be a plain dict with exactly the required plain-string keys")


def _counts() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _inverse_rational_cdf(sorted_values: list[float], numerator: int, denominator: int) -> float:
    """Inverse empirical CDF at an exact rational rank threshold."""
    rank = (numerator * len(sorted_values) + denominator - 1) // denominator
    return sorted_values[max(0, min(len(sorted_values) - 1, rank - 1))]


def _inverse_empirical_cdf(sorted_values: list[float], probability: float) -> float:
    """Smallest observed value with empirical CDF at least the supplied float."""
    return _inverse_rational_cdf(sorted_values, *probability.as_integer_ratio())


def stratified_cluster_bootstrap_interval(
    enrollment: list[dict],
    outcomes: list[dict],
    *,
    seed: int,
    draws: int,
    confidence: float = .90,
) -> dict:
    """Sample whole declared clusters within fixed families, then macro-average.

    Every enrolled case must have exactly one complete paired outcome. Correctness,
    family and cluster identities are externally supplied; this function verifies
    their typed joins, not their provenance or independence. Observed family point
    estimates retain the original case denominator, even with unequal cluster sizes.
    The simulation uses a local RNG and leaves inputs and global RNG state unchanged.
    """
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be a plain integer from 0 through 2**63-1")
    if type(draws) is not int or not MIN_DRAWS <= draws <= MAX_DRAWS:
        raise ValueError("draws must be a plain integer from 1000 through 50000")
    if (type(confidence) not in (int, float) or not 0 < confidence < 1
            or not math.isfinite(confidence)):
        raise ValueError("confidence must be a finite plain number strictly between zero and one")
    if type(enrollment) is not list or not 1 <= len(enrollment) <= MAX_ROWS:
        raise ValueError("enrollment must be a nonempty plain list of at most 384 records")
    if type(outcomes) is not list or not 1 <= len(outcomes) <= MAX_ROWS:
        raise ValueError("outcomes must be a nonempty plain list of at most 384 records")

    enrolled = {}
    cluster_families = {}
    families = set()
    for row in enrollment:
        _record(row, _ENROLLMENT_KEYS, "enrollment record")
        for key in _ENROLLMENT_KEYS:
            _identifier(row[key], key)
        case_id, family, cluster_id = row["case_id"], row["family"], row["cluster_id"]
        if case_id in enrolled:
            raise ValueError("duplicate enrolled case_id")
        if cluster_id in cluster_families and cluster_families[cluster_id] != family:
            raise ValueError("the same cluster_id cannot belong to different families")
        families.add(family)
        if len(families) > MAX_FAMILIES:
            raise ValueError("enrollment exceeds 16 fixed families")
        enrolled[case_id] = (family, cluster_id)
        cluster_families[cluster_id] = family

    paired = {}
    for row in outcomes:
        _record(row, _OUTCOME_KEYS, "outcome record")
        _identifier(row["case_id"], "case_id")
        if (type(row["complete"]) is not bool or row["complete"] is not True
                or type(row["parent_correct"]) is not bool
                or type(row["candidate_correct"]) is not bool):
            raise ValueError("every outcome must have complete=True and exact boolean correctness flags")
        case_id = row["case_id"]
        if case_id in paired:
            raise ValueError("duplicate outcome case_id")
        if case_id not in enrolled:
            raise ValueError("outcome case_id is not enrolled")
        paired[case_id] = (row["parent_correct"], row["candidate_correct"])
    if set(paired) != set(enrolled):
        raise ValueError("every enrolled case requires one complete paired outcome")

    grouped = {}
    for case_id in sorted(enrolled):
        family, cluster_id = enrolled[case_id]
        cluster = grouped.setdefault(family, {}).setdefault(cluster_id, _counts())
        parent, candidate = paired[case_id]
        cluster["n"] += 1
        cluster["parent_correct"] += int(parent)
        cluster["candidate_correct"] += int(candidate)
        cluster["repairs"] += int(candidate and not parent)
        cluster["losses"] += int(parent and not candidate)
        cluster["paired_delta"] += int(candidate) - int(parent)

    family_summaries, strata = [], []
    totals = _counts()
    for family in sorted(grouped):
        clusters = [{"cluster_id": cluster_id, **grouped[family][cluster_id]}
                    for cluster_id in sorted(grouped[family])]
        counts = {key: sum(cluster[key] for cluster in clusters) for key in _COUNT_KEYS}
        family_summaries.append({"family": family, **counts, "cluster_count": len(clusters),
                                 "net_rate": counts["paired_delta"] / counts["n"],
                                 "clusters": clusters})
        strata.append(tuple((cluster["paired_delta"], cluster["n"]) for cluster in clusters))
        for key in _COUNT_KEYS:
            totals[key] += counts[key]

    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        family_rates = []
        for clusters in strata:
            delta = denominator = 0
            for _ in range(len(clusters)):
                cluster_delta, cluster_n = clusters[rng.randrange(len(clusters))]
                delta += cluster_delta
                denominator += cluster_n
            family_rates.append(delta / denominator)
        samples.append(math.fsum(family_rates) / len(strata))
    samples.sort()
    confidence_numerator, confidence_denominator = confidence.as_integer_ratio()
    lower_numerator = confidence_denominator - confidence_numerator
    upper_numerator = confidence_denominator + confidence_numerator
    tail_denominator = 2 * confidence_denominator
    return {
        "net_rate": math.fsum(family["net_rate"] for family in family_summaries) / len(strata),
        "lower": _inverse_rational_cdf(samples, lower_numerator, tail_denominator),
        "upper": _inverse_rational_cdf(samples, upper_numerator, tail_denominator),
        "confidence": confidence,
        "confidence_probability_convention": "EXACT_SUPPLIED_BINARY_FLOAT_RATIO",
        "confidence_ratio": [confidence_numerator, confidence_denominator],
        "seed": seed,
        "draws": draws,
        "method": "monte_carlo_fixed_family_cluster_percentile",
        "simulation": "MONTE_CARLO",
        "coverage": "EXPLORATORY_NOT_CONFIRMATION",
        "native_execution_authorized": False,
        "quality_improvement_established": False,
        "parent_independence_verified": False,
        **totals,
        "family_count": len(strata),
        "cluster_count": sum(len(clusters) for clusters in strata),
        "families": family_summaries,
        "cluster_identity": "EXTERNALLY_DECLARED_PARENT_IDS_NOT_VERIFIED_INDEPENDENCE",
        "limitations": [
            "Case, family, correctness and semantic-parent identities are externally supplied metadata.",
            "Monte Carlo empirical percentiles do not establish exact frequentist coverage.",
            "Zero observed differences do not exclude unobserved future harms.",
            "This prototype grants no native or quality authority and selects no acceptance rule or practical margin.",
        ],
    }
