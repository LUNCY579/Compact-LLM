"""COMPOSE4B-001 analysis. Written and committed BEFORE any score exists.

Primary contrast: TH_s vs SCALED_T on select-views-001, acquisition, exact paired McNemar.
Both arms are transforms of the SAME two deltas, so this is a WITHIN-DRAW comparison and the
cross-training |net| >= 20 floor does not apply.

MDE is computed from the OBSERVED discordance, never imported from the 596M run's nd=16:
borrowing a floor across a change of structure is a recorded failure in this programme. An
underpowered null is reported UNINFORMATIVE, not as a null.

Endpoint is `canonical` per the scorer's own rule; strict / canonical / contract_tolerant are
never pooled.
"""
from pathlib import Path
from math import comb
import json
import sys

OUT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/compose4b001-20260921")
SCORED = OUT / "scored"
ALPHA = 0.05
ENDPOINT = "canonical"


def exact_mcnemar(b, c):
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mde_at(nd):
    """Smallest |net| that would reach p <= ALPHA at this observed discordance."""
    for net in range(0, nd + 1):
        if (nd - net) % 2:
            continue
        b = (nd + net) // 2
        if exact_mcnemar(b, nd - b) <= ALPHA:
            return net
    return None


def _betacf(a, bb, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + bb, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (bb - m) * x / ((qam + m2) * (a + m2)); d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < EPS: break
    return h


def _betai(a, bb, x):
    import math
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + bb) - math.lgamma(a) - math.lgamma(bb)
                  + a * math.log(x) + bb * math.log1p(-x))
    return bt * _betacf(a, bb, x) / a if x < (a + 1.0) / (a + bb + 2.0) else 1.0 - bt * _betacf(bb, a, 1.0 - x) / bb


def cp_upper(k, n, alpha=0.05):
    """One-sided 95% Clopper-Pearson upper bound on pi given k of n."""
    if n == 0: return 1.0
    if k >= n: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m = (lo + hi) / 2
        if 1.0 - _betai(k + 1, n - k, m) > alpha: lo = m
        else: hi = m
    return (lo + hi) / 2


def equivalence_bound_rows(losses, nd):
    """Exact one-sided 95% upper bound on the LOSS-SIDE excess (c-b), in rows.

    A powered null says a difference of >= MDE would have been detected. It does NOT
    say the difference is small. This is the separate, much weaker thing the data can
    actually bound, and at n=384 even a perfect tie at nd=20 only reaches ~7.9 rows.
    """
    if nd == 0: return 0.0
    return nd * (2 * cp_upper(losses, nd) - 1)


def load(label):
    p = SCORED / f"{label}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def rows(d):
    return {r["id"]: bool(r[ENDPOINT]) for r in d["detail"]}


def contrast(name, a, b):
    """a minus b, paired on row id."""
    ra, rb = rows(a), rows(b)
    ids = sorted(set(ra) & set(rb))
    if len(ids) != len(ra) or len(ids) != len(rb):
        raise SystemExit(f"{name}: row id sets differ ({len(ra)} vs {len(rb)}, {len(ids)} shared)")
    gains = sum(1 for i in ids if ra[i] and not rb[i])
    losses = sum(1 for i in ids if rb[i] and not ra[i])
    nd = gains + losses
    p = exact_mcnemar(gains, losses)
    m = mde_at(nd)
    return {
        "contrast": name, "n": len(ids),
        "score_a": sum(ra.values()), "score_b": sum(rb.values()),
        "gains": gains, "losses": losses, "net": gains - losses, "nd": nd,
        "exact_two_sided_p": p, "alpha": ALPHA,
        "mde_at_observed_nd": m,
        # best achievable p at this discordance: if min_p > alpha the contrast CANNOT
        # reach significance however the pairs fall, and a null from it is unattainable
        # rather than informative. This is the power statement, MDE is its net analogue.
        "min_attainable_p_at_observed_nd": exact_mcnemar(nd, 0) if nd else 1.0,
        "significance_attainable": (exact_mcnemar(nd, 0) if nd else 1.0) <= ALPHA,
        "significant": p <= ALPHA,
        "equivalence_upper_bound_rows": equivalence_bound_rows(losses, nd),
        "equivalence_note": ("one-sided 95% CP upper bound on the loss-side excess. A powered null "
                             "bounds DETECTION, not EQUIVALENCE; these need different sample sizes "
                             "on the same rows."),
        "relationship": "within_draw (transforms of the same two deltas)",
        "cross_training_floor_applies": False,
    }


def main():
    arms = {k: load(k) for k in ("th_s", "scaled_t", "h_only")}
    missing = [k for k, v in arms.items() if v is None]
    if "th_s" in missing or "scaled_t" in missing:
        raise SystemExit(f"primary arms not scored yet: {missing}")

    # comparability gate: same instrument, same bank, before any endpoint is read
    banks = {k: v["views_sha256"] for k, v in arms.items() if v}
    insts = {k: v["instrument"] for k, v in arms.items() if v}
    if len(set(banks.values())) != 1 or len(set(insts.values())) != 1:
        raise SystemExit(f"BLOCKED_ON_COMPARABILITY: banks={banks} instruments={insts}")

    primary = contrast("TH_s - SCALED_T", arms["th_s"], arms["scaled_t"])

    if primary["nd"] == 0:
        verdict, reading = "UNINFORMATIVE", "zero discordant pairs; the instrument saw no difference to test"
    elif not primary["significance_attainable"]:
        verdict, reading = "UNINFORMATIVE", (
            f"significance is UNATTAINABLE at nd={primary['nd']}: even an all-one-way split gives "
            f"p={primary['min_attainable_p_at_observed_nd']:.4g} > {ALPHA}. A null here is a property "
            "of the discordance, not evidence of equivalence")
    elif primary["mde_at_observed_nd"] is None:
        verdict, reading = "UNINFORMATIVE", f"no |net| reaches p<={ALPHA} at nd={primary['nd']}; underpowered by construction"
    elif primary["significant"] and primary["net"] < 0:
        verdict, reading = "P2", ("interference appears under harsher attenuation on a base with headroom. "
                                  "NOT 'headroom causes interference' - three alternative causes are "
                                  "registered (headroom, s_t 0.587 vs 0.675, H's absolute strength) and this "
                                  "design separates none of them")
    elif primary["significant"] and primary["net"] > 0:
        verdict, reading = "P3", "H's presence HELPS at matched magnitude; unregistered direction, claimed as nothing"
    elif abs(primary["net"]) <= primary["mde_at_observed_nd"]:
        verdict, reading = "P1", (
            "POWERED NULL, not an equivalence certification. No cost detected, and a cost of "
            f">= MDE rows would have been detected. The null REPLICATES on a base with headroom "
            "under HARSHER attenuation (s_t 0.587 vs the 596M's 0.675), with delta_H functional "
            "at 2.35x its own operating point (amendments 3, 4). It does NOT establish that the "
            "cost is zero or small: see equivalence_upper_bound_rows for what the data actually "
            "bound, which at this n is loose")
    else:
        verdict, reading = "UNINFORMATIVE", "not significant and |net| exceeds the MDE; cannot be called a null"

    result = {
        "experiment": "COMPOSE4B-001",
        "registration_sha256_of_prereg": None,   # filled by the caller's receipt step
        "instrument": insts["th_s"], "views_sha256": banks["th_s"],
        "endpoint": ENDPOINT, "pooling_forbidden": True,
        "scores": {k: (v["aggregate"][f"{ENDPOINT}_correct"] if v else None) for k, v in arms.items()},
        "reference_delta_T_qwenft001": 270,
        "primary": primary,
        "verdict": verdict, "reading": reading,
        "secondary": [contrast("TH_s - H_only", arms["th_s"], arms["h_only"])] if arms["h_only"] else [],
        "quality_claim": False, "release_permitted": False, "promotes": None,
    }
    (OUT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("scores", "primary", "verdict", "reading")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
