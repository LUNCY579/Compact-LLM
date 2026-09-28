"""Exact confirmatory statistics for COMP-RS-001 (Amendment 1, sections 3, 9 and 11).

Pure python plus numpy; no scipy. Every verdict-bearing quantity here is exact:
binomial tails by integer arithmetic (p = 1/2) or log-space sums, one-sided
Clopper-Pearson bounds by bisection on the binomial tail, exact conditional
McNemar. The only Monte Carlo quantities are the registered H3b sign-flip
p-value and its design-randomization companion, both with the +1 correction.
No model output, TEST or CHALLENGE file is opened by this module.

Section 15 clarifications implemented here: C1 (limb_state), C2
(H4_FAMILY_SIZE, GATE_FAILED), C3 (cross_draw_floor), C4 (int list is
content_scoring.is_int_list), C5 (h3b_candidate); and the section 5 cap on
H3c (h3c_state).
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math
import numbers
import re

import numpy as np

from experiments.maxrl_comp002_pretraining_v1.content_scoring import PARSE_ERRORS, is_int_list, parse_candidate
from experiments.maxrl_composable_skills_v2.verifier import concrete

PASS = "PASS"
FAIL = "FAIL"
UNCERTIFIABLE = "UNCERTIFIABLE"
NOT_REACHED = "NOT_REACHED"
DESCRIPTIVE = "DESCRIPTIVE"
GATE_FAILED = "GATE_FAILED"
NOT_CLAIMED = "NOT_CLAIMED"

H4_FAMILY_SIZE = 12        # section 3: "The family has m = 12 limbs" (clarification C2)
H4_FLOOR_LEVEL = 0.05      # section 3 F_E: minimum detectable |net| at one-sided 0.05
PLACEBO_LABELS = (re.compile(r"PLACEBO_UNDERSIZED(\(\d+(\.\d+)?%\))?"), re.compile(r"PLACEBO_UNDERENGAGED"))
CP_TOLERANCE = 1e-12
SIGNFLIP_CHUNK = 1 << 14   # rows of sign patterns per draw; part of the p-value's identity
COLUMN_BLOCK = 256         # columns per matmul block in the batched sign-flip test
H_BLIND = (Fraction(1), Fraction(2, 3), Fraction(1, 3), Fraction(0))


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _unit(value, name: str, *, closed_high: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not (0.0 < value < 1.0 or (closed_high and value == 1.0)):
        raise ValueError(f"{name} out of range")
    return value


def _tie_tolerance(total_abs: float) -> float:
    """Distinct sums of registered e-differences differ by >= 3**-5; float error is ~1e-13."""
    return 1e-9 * (1.0 + total_abs)


# ---------------------------------------------------------------- binomial

@lru_cache(maxsize=128)
def _log_comb(n: int) -> tuple:
    return tuple(math.log(math.comb(n, j)) for j in range(n + 1))


def _log_sum(terms: list[float]) -> float:
    if not terms:
        return -math.inf
    top = max(terms)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(t - top) for t in terms))


def _log_terms(n: int, p: float, lo: int, hi: int) -> list[float]:
    logs = _log_comb(n)
    lp, lq = math.log(p), math.log1p(-p)
    return [logs[j] + j * lp + (n - j) * lq for j in range(lo, hi + 1)]


def binom_pmf(j: int, n: int, p: float) -> float:
    """P(X = j) for X ~ Bin(n, p), log-space (section 9 enumeration)."""
    j, n = _count(j, "j"), _count(n, "n")
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not 0.0 <= p <= 1.0:
        raise ValueError("p out of range")
    if j > n:
        return 0.0
    if p == 0.0 or p == 1.0:
        return float(j == (0 if p == 0.0 else n))
    return math.exp(_log_terms(n, float(p), j, j)[0])


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p), summed in log space from math.comb (section 9)."""
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError("k must be an integer")
    n = _count(n, "n")
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not 0.0 <= p <= 1.0:
        raise ValueError("p out of range")
    k = int(k)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    return min(1.0, math.exp(_log_sum(_log_terms(n, float(p), 0, k))))


def binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """Upper tail P(X >= k), X ~ Bin(n, p) (section 3 McNemar limbs).

    For p = 1/2 (every registered McNemar limb) the tail is the exact rational
    sum(C(n, j), j >= k) / 2**n, rounded once. Otherwise it is summed directly
    over the upper terms in log space (never as 1 - cdf).
    """
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError("k must be an integer")
    n = _count(n, "n")
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not 0.0 <= p <= 1.0:
        raise ValueError("p out of range")
    k = int(k)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p == 0.5:
        return sum(math.comb(n, j) for j in range(k, n + 1)) / (1 << n)
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    return min(1.0, math.exp(_log_sum(_log_terms(n, float(p), k, n))))


def _bisect(decreasing, target: float) -> tuple[float, float]:
    """Bracket the root of decreasing(p) = target on [0, 1] to CP_TOLERANCE."""
    lo, hi = 0.0, 1.0
    while hi - lo > CP_TOLERANCE:
        mid = 0.5 * (lo + hi)
        if decreasing(mid) > target:
            lo = mid
        else:
            hi = mid
    return lo, hi


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact Clopper-Pearson upper bound (sections 3 H2, legacy; 9).

    The p with P(X <= k | p) = 1 - conf, by bisection (tolerance 1e-12, the
    conservative end returned). k = 0 uses the closed form 1 - (1-conf)**(1/n)
    that the attainability rule of section 3 is written in; n = 0 gives 1.
    """
    k, n = _count(k, "k"), _count(n, "n")
    alpha = 1.0 - _unit(conf, "conf")
    if k > n:
        raise ValueError("k exceeds n")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    return _bisect(lambda p: binom_cdf(k, n, p), alpha)[1]


def cp_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact Clopper-Pearson lower bound (section 9): P(X >= k | p) = 1 - conf."""
    k, n = _count(k, "k"), _count(n, "n")
    alpha = 1.0 - _unit(conf, "conf")
    if k > n:
        raise ValueError("k exceeds n")
    if k == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)
    return _bisect(lambda p: -binom_sf(k, n, p), -alpha)[0]


def paired_tie_bound(nd: int, conf: float = 0.95) -> float:
    """|net| an exact 95% bound admits on a perfect tie of nd discordant rows (section 3 H4 floor sizing).

    nd * (2 * cp_upper(ceil(nd/2), nd) - 1): the net the upper bound on the
    better-share allows when the observed split is as even as possible.
    """
    nd = _count(nd, "nd")
    if nd == 0:
        raise ValueError("nd must be positive")
    return nd * (2.0 * cp_upper(math.ceil(nd / 2), nd, conf) - 1.0)


def percentile_bootstrap_coverage(true_count: float, n: int = 250, conf: float = 0.95) -> dict:
    """Exact coverage of the nominal percentile-bootstrap interval for a rate (section 9 [V]).

    Enumerates the observed count x ~ Bin(n, true_count/n); the infinite-draw
    bootstrap of x is Bin(n, x/n), whose inverse-cdf quantiles (smallest y with
    cdf(y) >= q) give the interval. Provenance check only; never a verdict.
    """
    n = _count(n, "n")
    if n == 0 or not 0.0 < true_count < n:
        raise ValueError("true_count must lie strictly inside (0, n)")
    tail = (1.0 - _unit(conf, "conf")) / 2.0
    p = true_count / n
    coverage = zero = 0.0
    for x in range(n + 1):
        weight = binom_pmf(x, n, p)
        if x in (0, n):
            lo = hi = x
        else:
            cdf = np.cumsum([math.exp(t) for t in _log_terms(n, x / n, 0, n)])
            lo = int(np.searchsorted(cdf, tail, side="left"))
            hi = int(np.searchsorted(cdf, 1.0 - tail, side="left"))
        coverage += weight * (lo <= true_count <= hi)
        zero += weight * (lo == hi == 0)
    return {"true_count": true_count, "n": n, "coverage": coverage, "p_interval_zero_zero": zero}


# ---------------------------------------------------------------- McNemar and limbs

def mcnemar_one_sided_p(better: int, worse: int) -> float:
    """Exact one-sided conditional McNemar P(Bin(nd, 1/2) >= better), nd = better + worse (section 3)."""
    better, worse = _count(better, "better"), _count(worse, "worse")
    nd = better + worse
    return 1.0 if nd == 0 else binom_sf(better, nd)


def nd_min(level: float) -> int:
    """Smallest nd with 0.5**nd <= level (section 3): 5 at 0.05, 8 at 0.05/12."""
    level = _unit(level, "level", closed_high=True)
    nd = 0
    while 0.5 ** nd > level:
        nd += 1
    return nd


def min_detectable_net(nd: int, level: float = 0.05):
    """Smallest net = b - c with b + c = nd whose one-sided p <= level; None if unattainable (section 3 F_E)."""
    nd = _count(nd, "nd")
    level = _unit(level, "level", closed_high=True)
    for better in range(nd + 1):
        if binom_sf(better, nd) <= level:
            return 2 * better - nd
    return None


def limb_state(significant: bool, nd: int, nd_floor: int, below_floor: bool = False) -> str:
    """The state of a McNemar limb (section 3 as settled by clarification C1).

    A limb significant under its registered procedure is never UNCERTIFIABLE:
    it is PASS, or FAIL when its net is below the floor. A non-significant limb
    is UNCERTIFIABLE iff nd < nd_min at the procedure's smallest level, else FAIL.
    """
    if type(significant) is not bool or type(below_floor) is not bool:
        raise ValueError("significant and below_floor must be bools")
    _count(nd, "nd")
    _count(nd_floor, "nd_floor")
    if significant:
        return FAIL if below_floor else PASS
    return UNCERTIFIABLE if nd < nd_floor else FAIL


def mcnemar_limb(better: int, worse: int, level: float = 0.05) -> dict:
    """One-sided exact McNemar limb (section 3). UNCERTIFIABLE iff nd < nd_min(level), never null.

    At a single level p <= level is unattainable below nd_min(level), so C1's
    significance-first order (limb_state) and the nd-first reading agree here.
    """
    level = _unit(level, "level")
    p = mcnemar_one_sided_p(better, worse)
    nd = int(better) + int(worse)
    floor = nd_min(level)
    state = limb_state(p <= level, nd, floor)
    return {"state": state, "p": p, "nd": nd, "nd_min": floor, "better": int(better),
            "worse": int(worse), "net": int(better) - int(worse), "level": level}


def loss_rate_limb(losses: int, denominator: int, margin: float, conf: float = 0.95) -> dict:
    """One-sample loss limb (section 3 H2-X and legacy retention).

    PASS iff the one-sided CP upper bound on losses/denominator is <= margin.
    UNCERTIFIABLE iff cp_upper(0, denominator) > margin: attainability depends
    on the denominator alone (|D| < 29 at margin 0.10).
    """
    losses, denominator = _count(losses, "losses"), _count(denominator, "denominator")
    margin = _unit(margin, "margin")
    if losses > denominator:
        raise ValueError("losses exceed denominator")
    attainable = cp_upper(0, denominator, conf) <= margin
    upper = cp_upper(losses, denominator, conf)
    state = UNCERTIFIABLE if not attainable else PASS if upper <= margin else FAIL
    return {"state": state, "upper": upper, "attainable": attainable, "losses": losses,
            "denominator": denominator, "margin": margin, "conf": conf}


def holm(pvalues: dict, level: float = 0.05) -> dict:
    """Holm step-down over the registered family (section 3 H4); ties ordered by name."""
    level = _unit(level, "level")
    if not pvalues:
        raise ValueError("empty family")
    for name, p in pvalues.items():
        if isinstance(p, bool) or not isinstance(p, numbers.Real) or not 0.0 <= p <= 1.0:
            raise ValueError(f"invalid p-value for {name}")
    order = sorted(pvalues, key=lambda name: (pvalues[name], str(name)))
    rejected = {name: False for name in pvalues}
    m = len(order)
    for step, name in enumerate(order):
        if pvalues[name] > level / (m - step):
            break
        rejected[name] = True
    return rejected


def fixed_sequence(limbs: list) -> dict:
    """Fixed-sequence testing (section 3 H3): every limb after the first non-PASS is NOT_REACHED.

    Each limb is (name, item) where item is a result dict with 'state', a state
    string, or a zero-argument callable returning either; callables after the
    first non-PASS are never invoked.
    """
    results = {}
    reached = True
    for entry in limbs:
        name, item = entry
        if name in results:
            raise ValueError(f"duplicate limb {name}")
        if not reached:
            results[name] = {"state": NOT_REACHED}
            continue
        result = item() if callable(item) else item
        if type(result) is str:
            result = {"state": result}
        if type(result) is not dict or type(result.get("state")) is not str:
            raise ValueError(f"limb {name} has no state")
        results[name] = result
        reached = result["state"] == PASS
    return results


def h3_supported(results: dict) -> bool:
    """H3 is supported iff H3a, H3b and H3d all PASS (section 3)."""
    return all(results.get(name, {}).get("state") == PASS for name in ("H3a", "H3b", "H3d"))


def _placebo_labels(labels) -> list[str]:
    if type(labels) not in (list, tuple, set, frozenset) or any(type(label) is not str for label in labels):
        raise ValueError("labels must be a collection of strings")
    for label in labels:
        if not any(pattern.fullmatch(label) for pattern in PLACEBO_LABELS):
            raise ValueError("not a section 5 label: " + label)
    return sorted(set(labels))


def h3c_state(result, labels) -> dict:
    """H3c subject to the section 5 labels: under either label H3c can be read only as FAIL.

    labels are the mechanically applied section 5 labels of this comparison
    (PLACEBO_UNDERSIZED, optionally with its shortfall as arm_build.placebo_label
    formats it, and PLACEBO_UNDERENGAGED); any other string is refused. Under
    either label a PASS becomes NOT_CLAIMED, never PASS; FAIL, UNCERTIFIABLE and
    NOT_REACHED are unchanged. The uncapped state is kept as 'uncapped_state'.
    """
    if type(result) is str:
        result = {"state": result}
    if type(result) is not dict or type(result.get("state")) is not str:
        raise ValueError("H3c result has no state")
    applied = _placebo_labels(labels)
    uncapped = result["state"]
    state = NOT_CLAIMED if applied and uncapped == PASS else uncapped
    return {**result, "state": state, "uncapped_state": uncapped, "section5_labels": applied,
            "capped": state != uncapped}


def cross_draw_floor(replicate_nd, level: float = H4_FLOOR_LEVEL) -> dict:
    """F_E from the RS_a100 vs RS_a100' discordance on endpoint E (section 3 H4; clarification C3).

    None (no replicate) gives floor_net None: the limb is DESCRIPTIVE. When no
    net at that discordance can be significant (min_detectable_net is None),
    F_E = 0: draw noise on E is negligible and Holm significance alone decides.
    """
    if replicate_nd is None:
        return {"floor_net": None, "floor_basis": "no_replicate", "replicate_nd": None}
    nd = _count(replicate_nd, "replicate_nd")
    net = min_detectable_net(nd, level)
    if net is None:
        return {"floor_net": 0, "floor_basis": "replicate_nd_below_nd_min", "replicate_nd": nd}
    return {"floor_net": net, "floor_basis": "min_detectable_net", "replicate_nd": nd}


def _floor(floor) -> tuple:
    """(floor_net, floor_basis) from a number, None, or a cross_draw_floor record."""
    if type(floor) is dict:
        if set(floor) != {"floor_net", "floor_basis", "replicate_nd"}:
            raise ValueError("floor record must come from cross_draw_floor")
        return floor["floor_net"], floor["floor_basis"]
    return floor, None


def h4_limb(better: int, worse: int, family_size: int, floor_net, *, level: float = 0.05,
            holm_rejected=None, floor_basis: str | None = None) -> dict:
    """One H4 limb (section 3; clarifications C1 and C2).

    family_size must be the registered m = 12. floor_net None (no replicate, no
    claim) gives DESCRIPTIVE. Pass the family's Holm decision as holm_rejected
    (see h4_family); without it the first Holm step, p <= level/m, is used,
    which implies Holm rejection but is conservative. State by limb_state: a
    significant limb is PASS if net >= floor_net, else FAIL with
    below_floor=True; a non-significant limb is UNCERTIFIABLE iff
    nd < nd_min(level/m) (8 at m = 12), else FAIL.
    """
    family_size = _count(family_size, "family_size")
    if family_size != H4_FAMILY_SIZE:
        raise ValueError(f"the H4 family size is fixed at m = {H4_FAMILY_SIZE}")
    level = _unit(level, "level")
    p = mcnemar_one_sided_p(better, worse)
    nd, net = int(better) + int(worse), int(better) - int(worse)
    first_step = level / family_size
    if holm_rejected is None:
        significant, basis = p <= first_step, "first_step_bound"
    elif type(holm_rejected) is bool:
        significant, basis = holm_rejected, "family"
    else:
        raise ValueError("holm_rejected must be a bool or None")
    floor = nd_min(first_step)
    result = {"p": p, "nd": nd, "net": net, "better": int(better), "worse": int(worse),
              "family_size": family_size, "nd_min": floor, "holm_significant": significant,
              "holm_basis": basis, "floor_net": floor_net, "floor_basis": floor_basis}
    if floor_net is None:
        return {**result, "state": DESCRIPTIVE, "below_floor": None}
    if isinstance(floor_net, bool) or not isinstance(floor_net, numbers.Real) or floor_net < 0:
        raise ValueError("floor_net must be a nonnegative number or None")
    below = bool(net < floor_net)
    return {**result, "state": limb_state(significant, nd, floor, below), "below_floor": below}


def h4_family(limbs: dict, level: float = 0.05) -> dict:
    """The H4 family (section 3): Holm across ALL m = 12 registered limbs (clarification C2).

    limbs maps each of exactly H4_FAMILY_SIZE names to (better, worse, floor),
    where floor is None (descriptive), a nonnegative number, or a
    cross_draw_floor record; or to GATE_FAILED for a comparator that failed its
    gate. A GATE_FAILED limb stays in the family, enters Holm as p = 1 and is
    never rejected, so m never shrinks. Descriptive limbs are Holm members too.
    """
    if type(limbs) is not dict or len(limbs) != H4_FAMILY_SIZE:
        raise ValueError(f"the H4 family has exactly m = {H4_FAMILY_SIZE} limbs")
    pvalues, tested = {}, {}
    for name, limb in limbs.items():
        if type(limb) is str and limb == GATE_FAILED:
            pvalues[name] = 1.0
            continue
        if type(limb) is not tuple or len(limb) != 3:
            raise ValueError(f"limb {name} must be (better, worse, floor) or GATE_FAILED")
        tested[name] = limb
        pvalues[name] = mcnemar_one_sided_p(limb[0], limb[1])
    rejected = holm(pvalues, level)
    out = {}
    for name in limbs:
        if name not in tested:
            out[name] = {"state": GATE_FAILED, "p": None, "holm_significant": False,
                         "holm_basis": "gate_failed", "family_size": H4_FAMILY_SIZE}
            continue
        better, worse, floor = tested[name]
        floor_net, basis = _floor(floor)
        out[name] = h4_limb(better, worse, H4_FAMILY_SIZE, floor_net, level=level,
                            holm_rejected=rejected[name], floor_basis=basis)
    return out


# ---------------------------------------------------------------- H3b: pi_blind and e

def h3b_candidate(response: str):
    """H3b's candidate: the content-contract candidate (clarification C5).

    content_scoring.parse_candidate with the composition contract, the parse
    classify_rs uses, so a wrapper cannot hide lookup use. A multiple-candidate
    object, UNKNOWN, an unparseable response or a non-int-list answer (C4) gives
    None, and e_value(None, spec) = 0.
    """
    if type(response) is not str:
        raise TypeError("response must be a string")
    try:
        answer, _, multiple = parse_candidate(response)
    except PARSE_ERRORS:
        return None
    if multiple or not is_int_list(answer):
        return None
    return answer


def _check_triples(triples) -> list[tuple[int, ...]]:
    seen = set()
    out = []
    for triple in triples:
        members = tuple(triple)
        if len(members) != 3 or any(type(x) is not int for x in members) or len(set(members)) != 3:
            raise ValueError("every value class must be a triple of distinct ints")
        if seen & set(members):
            raise ValueError("triples overlap")
        seen.update(members)
        out.append(members)
    return out


def pi_blind(candidate: list, triples: list) -> float:
    """Exact pass probability of a lookup-blind list (section 3 H3b).

    Product over triples g of h(|L & g|), h = (1, 2/3, 1/3, 0), over DISTINCT
    indices; 0 if any index lies outside every triple.
    """
    if not is_int_list(candidate):
        raise ValueError("candidate must be an int list (non-empty, int elements)")
    checked = _check_triples(triples)
    chosen = set(candidate)
    covered = set().union(*map(set, checked)) if checked else set()
    if not chosen <= covered:
        return 0.0
    value = Fraction(1)
    for triple in checked:
        value *= H_BLIND[len(chosen & set(triple))]
    return float(value)


def triples_from_spec(spec: dict) -> list[list[int]]:
    """Value classes of spec['initial'], each required to be a triple (sections 3, 11)."""
    initial = spec.get("initial") if type(spec) is dict else None
    if type(initial) is not list or not initial:
        raise ValueError("spec has no initial values")
    classes: dict = {}
    for index, value in enumerate(initial):
        classes.setdefault(value, []).append(index)
    triples = sorted(classes.values(), key=lambda members: members[0])
    if any(len(members) != 3 for members in triples):
        raise ValueError("value class is not a triple")
    return triples


def resolved_set(spec: dict) -> frozenset:
    """The lookup-resolved candidate set of an RS spec via verifier.concrete (section 3 H3b)."""
    if type(spec) is not dict or "lookup" not in spec or "updates" in spec:
        raise ValueError("resolved_set needs a lookup spec without updates (RS)")
    candidates, _ = concrete(spec)
    if candidates is None:
        raise ValueError("lookup does not resolve")
    return frozenset(candidates)


def design(spec: dict) -> tuple[list[list[int]], frozenset]:
    """Triples and resolved set, with the 2-of-3 structure verified (section 11 triple gate)."""
    triples = triples_from_spec(spec)
    resolved = resolved_set(spec)
    if any(len(resolved & set(triple)) != 2 for triple in triples):
        raise ValueError("resolved set is not exactly two members of every triple")
    if not resolved <= set().union(*map(set, triples)):
        raise ValueError("resolved index outside every triple")
    return triples, resolved


def e_value(candidate, spec: dict) -> float:
    """e = 1[int list subset of resolved] - pi_blind (section 3 H3b); 0 for None or non-int-list.

    candidate is h3b_candidate(response). "Int list" is is_int_list (C4), so []
    has e = 0 (as the vacuous subset would also give: 1 - pi_blind([]) = 0).
    """
    triples, resolved = design(spec)
    if not is_int_list(candidate):
        return 0.0
    return float(set(candidate) <= resolved) - pi_blind(candidate, triples)


# ---------------------------------------------------------------- Monte Carlo tests

def _as_columns(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1 or not np.all(np.isfinite(array)):
        raise ValueError("differences must be a finite nonempty vector or matrix")
    return array


def signflip_test_columns(differences, n_flips: int = 1_000_000, *, seed) -> np.ndarray:
    """One-sided sign-flip p-values for each column of an (rows x columns) matrix (section 3 H3b).

    Every column is tested against the SAME sign patterns: the stream is
    default_rng(seed), drawn SIGNFLIP_CHUNK patterns at a time as packed bits,
    so column j reproduces signflip_test(differences[:, j], n_flips, seed=seed).
    p = (1 + #(null >= observed)) / (n_flips + 1); sums within a 1e-9 relative
    tolerance of the observed value count as ties (>=).
    """
    columns = _as_columns(differences)
    n_flips = _count(n_flips, "n_flips")
    if n_flips < 1:
        raise ValueError("n_flips must be positive")
    rng = np.random.default_rng(_count(seed, "seed"))
    rows, width = columns.shape
    observed = columns.sum(axis=0)
    threshold = observed - np.array([_tie_tolerance(t) for t in np.abs(columns).sum(axis=0)])
    doubled = 2.0 * columns
    exceed = np.zeros(width, dtype=np.int64)
    packed = (rows + 7) // 8
    for start in range(0, n_flips, SIGNFLIP_CHUNK):
        size = min(SIGNFLIP_CHUNK, n_flips - start)
        bits = np.unpackbits(rng.integers(0, 256, size=(size, packed), dtype=np.uint8),
                             axis=1, count=rows).astype(np.float64)
        for left in range(0, width, COLUMN_BLOCK):
            right = min(width, left + COLUMN_BLOCK)
            null = bits @ doubled[:, left:right] - observed[left:right]
            exceed[left:right] += (null >= threshold[left:right]).sum(axis=0)
    return (1.0 + exceed) / (n_flips + 1.0)


def signflip_test(d, n_flips: int = 1_000_000, *, seed) -> float:
    """Registered H3b paired sign-flip test of sum(d) > 0 (section 3): Monte Carlo, +1 correction.

    Null: per-row exchangeability of the two arms, i.e. independent fair signs
    on each d_i. Vectorised in chunks of SIGNFLIP_CHUNK patterns from
    numpy default_rng(seed); the p-value is a deterministic function of
    (d, n_flips, seed, SIGNFLIP_CHUNK, numpy's bit generator).
    """
    array = np.asarray(d, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("d must be a vector")
    return float(signflip_test_columns(array, n_flips, seed=seed)[0])


def _row_design(candidate, triples: list[list[int]]) -> tuple[bool, np.ndarray, float]:
    """(eligible, member mask T x 3, pi_blind); ineligible rows have e = 0 under every redraw."""
    mask = np.zeros((len(triples), 3), dtype=bool)
    if not is_int_list(candidate):
        return False, mask, 0.0
    chosen = set(candidate)
    covered = set().union(*map(set, triples))
    if not chosen <= covered:
        return False, mask, 0.0
    for t, triple in enumerate(triples):
        for slot, index in enumerate(triple):
            mask[t, slot] = index in chosen
    return True, mask, pi_blind(candidate, triples)


def redraw_e(candidate_sets: list, specs: list, n_draws: int, rng: np.random.Generator) -> list[np.ndarray]:
    """e under n_draws design re-draws, one (n_draws x rows) matrix per candidate list (section 3).

    Per row and per triple the excluded member is re-drawn uniformly; every
    candidate list sees the SAME re-drawn world, responses held fixed.
    """
    n_draws = _count(n_draws, "n_draws")
    if n_draws < 1 or not specs or any(len(c) != len(specs) for c in candidate_sets):
        raise ValueError("candidates and specs must align and n_draws be positive")
    out = [np.zeros((n_draws, len(specs))) for _ in candidate_sets]
    for i, spec in enumerate(specs):
        triples, _ = design(spec)
        excluded = rng.integers(0, 3, size=(n_draws, len(triples)))
        slots = np.arange(len(triples))[None, :]
        for matrix, candidates in zip(out, candidate_sets):
            eligible, mask, pi = _row_design(candidates[i], triples)
            if eligible:
                inside = ~mask[slots, excluded].any(axis=1)
                matrix[:, i] = inside.astype(np.float64) - pi
    return out


def design_randomization_test(candidates_rs: list, candidates_s, specs: list, n_draws: int, seed) -> float:
    """Design-randomization p-value for sum(e_RS - e_S) > 0 (section 3 H3b rule and companion).

    Observed statistic uses each row's actual resolved set; the reference
    re-draws the excluded member per row and per triple (uniform), with the
    responses held fixed. candidates_s None tests RS alone (descriptive
    companion). p = (1 + #(reference >= observed)) / (n_draws + 1).
    """
    if candidates_s is None:
        candidates_s = [None] * len(specs)
    if len(candidates_rs) != len(specs) or len(candidates_s) != len(specs):
        raise ValueError("candidates and specs must align")
    observed = math.fsum(e_value(a, s) - e_value(b, s)
                         for a, b, s in zip(candidates_rs, candidates_s, specs))
    e_rs, e_s = redraw_e([candidates_rs, candidates_s], specs, n_draws, np.random.default_rng(_count(seed, "seed")))
    reference = (e_rs - e_s).sum(axis=1)
    bound = 2.0 * len(specs)
    exceed = int((reference >= observed - _tie_tolerance(bound)).sum())
    return (1.0 + exceed) / (n_draws + 1.0)
