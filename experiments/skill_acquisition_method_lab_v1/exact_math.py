"""Finite enumeration and quadratic counterexamples; no native model claims.

The scalar policy is Bernoulli with logit theta, reward r and trajectory score
r-p. It isolates estimator expectation/variance exactly. Language-model score
vectors, optimizer dynamics and skill generalization are outside this toy.
"""
from __future__ import annotations

from itertools import combinations
from math import comb, fsum, isfinite, log, sqrt

METHODS = ("reinforce", "centered", "rloo", "grpo_sample", "grpo_population",
           "maxrl_practical", "maxrl_success", "maxrl_full_cv")


def validate_probability(p: float) -> None:
    if not isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability outside [0,1]")


def mixed_probability(p: float, n: int) -> float:
    validate_probability(p)
    if type(n) is not int or n < 2:
        raise ValueError("group size must be >=2")
    return max(0.0, 1 - (1-p)**n - p**n)


def group_gradient(method: str, p: float, n: int, k: int, epsilon: float = 0.0) -> float:
    """Realized full-sequence score update under the Bernoulli-logit toy."""
    validate_probability(p)
    if method not in METHODS or type(n) is not int or n < 2 or type(k) is not int or not 0 <= k <= n:
        raise ValueError("invalid method/group")
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("invalid epsilon")
    u = k / n
    if method == "reinforce":
        return u * (1-p)
    if method == "maxrl_success":
        return 1-p if k else 0.0
    if method == "maxrl_full_cv":
        return (1-p if k else 0.0) - (u-p)
    if k in (0, n):
        return 0.0
    numerator = u * (1-u)
    if method == "centered":
        return numerator
    if method == "rloo":
        return n / (n-1) * numerator
    if method == "maxrl_practical":
        return numerator / (u+epsilon)
    ddof = 1 if method == "grpo_sample" else 0
    std = sqrt(n / (n-ddof) * numerator)
    return numerator / (std+epsilon)


def gradient_statistics(method: str, p: float, n: int, epsilon: float = 0.0) -> dict:
    terms = []
    for k in range(n+1):
        probability = comb(n, k) * p**k * (1-p)**(n-k)
        gradient = group_gradient(method, p, n, k, epsilon)
        terms.append((probability, gradient))
    mean = fsum(probability * g for probability, g in terms)
    variance = fsum(probability * (g-mean)**2 for probability, g in terms)
    derivative = p * (1-p)
    return {"method": method, "p": p, "n": n, "epsilon": epsilon,
            "probability_sum": fsum(prob for prob, _ in terms),
            "expected_logit_gradient": mean, "variance": variance,
            "population_weight": mean / derivative if derivative else None,
            "mixed_probability": mixed_probability(p, n)}


def maxrl_closed_gradient(p: float, n: int, *, practical: bool) -> float:
    validate_probability(p)
    order = n-1 if practical else n
    return (1-p) * (1-(1-p)**order)


def bounded_failure_continuation(p: float, n: int, blocks: int, gamma: float) -> dict:
    """Exact block sampling, stop on any success or probabilistic failure exit.

Every prior block is all-failure. The final group contains k successes among
n samples; pooling yields random total T=n*j, so fixed-N identities do not
apply. This is an original bounded same-snapshot toy, not an NGU replication.
"""
    validate_probability(p)
    validate_probability(gamma)
    if type(n) is not int or n < 2 or type(blocks) is not int or blocks < 1:
        raise ValueError("invalid block budget")
    outcomes = []
    for j in range(1, blocks+1):
        reach = (gamma * (1-p)**n)**(j-1)
        for k in range(n+1):
            probability = reach * comb(n, k) * p**k * (1-p)**(n-k)
            if k == 0 and j < blocks:
                probability *= 1-gamma
            if not probability:
                continue
            total = n*j
            practical_gradient = 1-k/total if k else 0.0
            outcomes.append((probability, total, k, practical_gradient))
    mean = fsum(prob*g for prob, _, _, g in outcomes)
    draws = fsum(prob*t for prob, t, _, _ in outcomes)
    closed_draws = n * fsum((gamma * (1-p)**n)**j for j in range(blocks))
    return {"p": p, "initial_group_size": n, "max_blocks": blocks, "gamma": gamma,
            "max_draws": n*blocks, "expected_draws": draws,
            "expected_draws_closed": closed_draws,
            "probability_sum": fsum(prob for prob, _, _, _ in outcomes),
            "probability_any_success": fsum(prob for prob, _, k, _ in outcomes if k),
            "expected_pooled_practical_gradient": mean,
            "fixed_initial_group_gradient": maxrl_closed_gradient(p, n, practical=True),
            "fixed_max_group_gradient": maxrl_closed_gradient(p, n*blocks, practical=True)}


def rejection_mixture(pass_rates: list[float], rho: list[float], n: int) -> dict:
    """Distinguish global rejection from a fixed accepted quota per prompt."""
    if len(pass_rates) != len(rho) or not pass_rates or abs(fsum(rho)-1) > 1e-12 or any(x <= 0 for x in rho):
        raise ValueError("invalid prompt distribution")
    acceptance = [mixed_probability(p, n) for p in pass_rates]
    a = fsum(w*m for w, m in zip(rho, acceptance))
    unconditioned = [maxrl_closed_gradient(p, n, practical=True) for p in pass_rates]
    if not a:
        return {"acceptance": 0.0, "status": "NO_ACCEPTED_GROUPS"}
    conditional = [g/m if m else None for g, m in zip(unconditioned, acceptance)]
    return {"acceptance": a, "accepted_prompt_distribution": [w*m/a for w, m in zip(rho, acceptance)],
            "fixed_proposal_expected_gradient": fsum(w*g for w, g in zip(rho, unconditioned)),
            "global_rejection_expected_gradient": fsum(w*g for w, g in zip(rho, unconditioned))/a,
            "per_prompt_quota_expected_gradient": fsum(w*g for w, g in zip(rho, conditional) if g is not None),
            "conditional_gradients": conditional,
            "expected_proposal_groups_per_accepted": 1/a}


def dot(x, y) -> float:
    if len(x) != len(y):
        raise ValueError("vector shape mismatch")
    return fsum(a*b for a, b in zip(x, y))


def matvec(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def add(x, y):
    if len(x) != len(y):
        raise ValueError("vector shape mismatch")
    return tuple(a+b for a, b in zip(x, y))


def scale(x, value):
    return tuple(value*a for a in x)


def quadratic_loss(theta, hessian, linear):
    return 0.5*dot(theta, matvec(hessian, theta)) + dot(linear, theta)


def descent(theta, hessian, linear, eta):
    return add(theta, scale(add(matvec(hessian, theta), linear), -eta))


def sequential_counterexample(eta_r=0.1, eta_s=0.1) -> dict:
    """Orthogonal initial gradients; exact nonzero one-step order commutator."""
    zero = (0.0, 0.0)
    hr, hs = ((2.0, 0.3), (0.3, 1.0)), ((1.0, 0.8), (0.8, 2.0))
    gr, gs = (1.0, 0.0), (0.0, 1.0)
    r = descent(zero, hr, gr, eta_r)
    s = descent(zero, hs, gs, eta_s)
    rs = descent(r, hs, gs, eta_s)
    sr = descent(s, hr, gr, eta_r)
    composed = add(r, s)
    observed = add(rs, scale(sr, -1))
    predicted = scale(add(matvec(hs, gr), scale(matvec(hr, gs), -1)), eta_r*eta_s)
    return {"gradient_dot": dot(gr, gs), "R_then_S": rs, "S_then_R": sr,
            "independent_sum": composed, "order_difference": observed,
            "hessian_commutator_prediction": predicted,
            "R_loss": {"sum": quadratic_loss(composed, hr, gr), "RS": quadratic_loss(rs, hr, gr),
                       "SR": quadratic_loss(sr, hr, gr)},
            "S_loss": {"sum": quadratic_loss(composed, hs, gs), "RS": quadratic_loss(rs, hs, gs),
                       "SR": quadratic_loss(sr, hs, gs)}}


def composition_counterexample(a=0.5, b=0.25) -> dict:
    """Euclidean orthogonality does not cancel Hessian/activation interactions."""
    r, s = (1.0, 0.0), (0.0, 1.0)
    h = ((1.0, 0.7), (0.7, 1.0))
    f = lambda theta: quadratic_loss(theta, h, (0.0, 0.0))
    interaction = f(add(scale(r, a), scale(s, b))) - f(scale(r, a)) - f(scale(s, b)) + f((0.0, 0.0))
    dr, ds, covariance = (1.0, 1.0), (1.0, -1.0), ((100.0, 0.0), (0.0, 1.0))
    weighted = dot(dr, matvec(covariance, ds))
    return {"delta_dot": dot(r, s), "functional_interaction": interaction,
            "hessian_prediction": a*b*dot(r, matvec(h, s)),
            "matrix_frobenius_dot": dot(dr, ds), "activation_weighted_cross": weighted,
            "activation_weighted_cosine": weighted/sqrt(dot(dr, matvec(covariance, dr))*dot(ds, matvec(covariance, ds)))}


def mask_objective(mask, delta, skill_fisher, legacy_fisher, lam):
    return fsum((fs*(1-m)+lam*fl*m)*d*d for m, d, fs, fl in zip(mask, delta, skill_fisher, legacy_fisher))


def dual_fisher_allocation(delta, skill_fisher, legacy_fisher, k: int, lam: float) -> dict:
    n = len(delta)
    if n != len(skill_fisher) or n != len(legacy_fisher) or not 0 <= k <= n or lam < 0:
        raise ValueError("invalid mask shape/budget")
    scores = [(skill_fisher[i]-lam*legacy_fisher[i])*delta[i]**2 for i in range(n)]
    chosen = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
    mask = tuple(int(i in chosen) for i in range(n))
    exhaustive = []
    for keep in combinations(range(n), k):
        candidate = tuple(int(i in keep) for i in range(n))
        exhaustive.append((mask_objective(candidate, delta, skill_fisher, legacy_fisher, lam), keep))
    best = min(exhaustive)
    return {"score": scores, "keep": chosen, "mask": mask,
            "proxy_objective": mask_objective(mask, delta, skill_fisher, legacy_fisher, lam),
            "exhaustive_minimum": best[0], "exhaustive_keep": best[1],
            "note": "Original two-distribution diagonal proxy; not the published F-WANDA algorithm."}


def row_ranking_invariance(scores, row_multipliers, keep_per_row: int) -> dict:
    before, after = [], []
    for row, multiplier in zip(scores, row_multipliers):
        if multiplier <= 0:
            raise ValueError("row multiplier must be positive")
        before.append(sorted(range(len(row)), key=lambda j: (-row[j], j))[:keep_per_row])
        after.append(sorted(range(len(row)), key=lambda j: (-multiplier*row[j], j))[:keep_per_row])
    return {"before": before, "after": after, "unchanged": before == after}


def outer(b, a):
    return tuple(tuple(x*y for y in a) for x in b)


def matrix_add(x, y):
    return tuple(add(a, b) for a, b in zip(x, y))


def matrix_scale(x, c):
    return tuple(scale(row, c) for row in x)


def lora_gauge_counterexample(gauge=10.0) -> dict:
    if gauge <= 0:
        raise ValueError("gauge must be positive")
    b1, a1, b2, a2 = (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, -1.0)
    effective = matrix_scale(matrix_add(outer(b1, a1), outer(b2, a2)), 0.5)
    factors = outer(scale(add(b1, b2), 0.5), scale(add(a1, a2), 0.5))
    gauge_b1, gauge_a1 = scale(b1, 1/gauge), scale(a1, gauge)
    gauge_factors = outer(scale(add(gauge_b1, b2), 0.5), scale(add(gauge_a1, a2), 0.5))
    return {"donor1_effective_unchanged": outer(b1, a1) == outer(gauge_b1, gauge_a1),
            "effective_delta_mean": effective, "factor_mean": factors, "gauge_changed_factor_mean": gauge_factors}


def bernoulli_kl(reference_p: float, current_p: float) -> float:
    validate_probability(reference_p)
    validate_probability(current_p)
    if not 0 < reference_p < 1 or not 0 < current_p < 1:
        raise ValueError("KL toy requires interior probabilities")
    return reference_p*log(reference_p/current_p) + (1-reference_p)*log((1-reference_p)/(1-current_p))


def replay_signal(reference_p=0.99, current_p=0.9, group_size=8) -> dict:
    return {"uniform_success_rl_gradient": group_gradient("maxrl_practical", current_p, group_size, group_size),
            "supervised_success_logit_gradient": 1-current_p,
            "reference_kl": bernoulli_kl(reference_p, current_p),
            "reference_kl_logit_gradient": current_p-reference_p,
            "fisher_local_coefficient": reference_p*(1-reference_p)}
