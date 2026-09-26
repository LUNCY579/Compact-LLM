"""ABST-001 -- the registered DESCRIPTIVE quantities (ABST001_PREREG.md §7), computed ONLY from the stored
per-row records (so they can be recomputed from ABST_RESULT.json without a model). Never a verdict.

records[row_id] = {stratum ('unk'|'known'), family, tier, reference, input_equal_V0 {V4, V2},
                   m[config][input_vocab][support] = {nats, per_token, first{UNKNOWN,0..3,im_end}, first_logmass},
                   gen[cell] = greedy text}
Cells: A S20/V4, B S20/V2, C S18/V4, D S18/V2, E S20/V0, F S18/V0; teacher T4 / T2 / T0 in the same vocabulary.
"""
from __future__ import annotations

import math
import statistics as st

import abst_verdict as av

LC_CELLS = {"A": ("S20", "V4"), "B": ("S20", "V2"), "C": ("S18", "V4"), "D": ("S18", "V2"),
            "E": ("S20", "V0"), "F": ("S18", "V0")}
AUROC_MARGIN = 0.03
TEACHER_AUROC_NEAR_HALF = 0.05     # |AUROC_T - 0.5| <= this -> "UNKNOWN-token propensity" wording


def nats(rec, cfg, vin, sup=None):
    try:
        return rec["m"][cfg][vin][sup or vin]["nats"]
    except KeyError:
        return None


def first(rec, cfg, v):
    try:
        return rec["m"][cfg][v][v]["first"]
    except KeyError:
        return None


def lse(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def margin(fr):
    """log p(UNKNOWN) - logsumexp log p(0..3) at the first answer position."""
    if fr is None or fr.get("UNKNOWN") is None:
        return None
    d = lse([fr.get(k) for k in ("0", "1", "2", "3")])
    return None if d is None else fr["UNKNOWN"] - d


def cp_interval(k, n, alpha=0.05):
    """Clopper-Pearson by bisection on the exact binomial tails (no scipy)."""
    if n == 0:
        return [0.0, 1.0]

    def upper_tail(p, kk):          # P(X >= kk)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk, n + 1))

    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    # both tails are INCREASING in p, so the bisection moves up while f(mid) < target:
    #   lower: P(X >= k | p_L) = alpha/2          upper: P(X <= k | p_U) = alpha/2  <=>  P(X >= k+1 | p_U) = 1 - alpha/2
    lo = 0.0 if k == 0 else solve(lambda p: upper_tail(p, k), alpha / 2)
    hi = 1.0 if k == n else solve(lambda p: upper_tail(p, k + 1), 1 - alpha / 2)
    return [lo, hi]


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); None if either side is empty."""
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def med(xs):
    xs = [x for x in xs if x is not None and x == x and not math.isinf(x)]
    return st.median(xs) if xs else None


def summarise(records: dict) -> dict:
    ids = list(records)
    by = {s: [i for i in ids if records[i]["stratum"] == s] for s in ("unk", "known")}
    out = {"rows": {s: len(v) for s, v in by.items()}, "cells": {}, "teacher": {}}
    for tv in ("V4", "V2", "V0"):
        out["teacher"]["T" + tv[1]] = {s: {"median_nats": med([nats(records[i], "T", tv) for i in by[s]]),
                                           "median_margin": med([margin(first(records[i], "T", tv)) for i in by[s]])}
                                       for s in by}
    for X, (depth, v) in LC_CELLS.items():
        plain_c, lc_c = depth + "_plain", depth + "_lc"
        if not all(plain_c in records[i]["m"] and v in records[i]["m"][plain_c] for i in ids):
            continue
        cell = {}
        for s, rows in by.items():
            P = [nats(records[i], plain_c, v) for i in rows]
            Q = [nats(records[i], lc_c, v) for i in rows]
            T = [nats(records[i], "T", v) for i in rows]
            sg = av.signs_of_difference(P, Q)
            worse, better = sum(x < 0 for x in sg), sum(x > 0 for x in sg)
            an = av.anchor_signs(P, Q, T)
            crossed = sum(1 for p, q, t in zip(P, Q, T) if None not in (p, q, t) and (p - t) * (q - t) < 0)
            gap = [(q - p) / (t - p) for p, q, t in zip(P, Q, T) if None not in (p, q, t) and t != p]
            worse_rows = [k for k, x in enumerate(sg) if x < 0]
            mp = [margin(first(records[i], plain_c, v)) for i in rows]
            ml = [margin(first(records[i], lc_c, v)) for i in rows]
            fP = [first(records[i], plain_c, v) for i in rows]
            fQ = [first(records[i], lc_c, v) for i in rows]
            F = lambda fr: lse([fr.get(k) for k in ("UNKNOWN", "0", "1", "2", "3")]) if fr else None
            cell[s] = {
                "lc_better": better, "lc_worse": worse, "ties_or_undefined": len(sg) - better - worse,
                "worse_fraction_cp95": cp_interval(worse, better + worse),
                "median_d_log_plain_over_lc": med([av.log_ratio(p, q) for p, q in zip(P, Q)]),
                "plain_nll_below_teacher": sum(1 for p, t in zip(P, T) if None not in (p, t) and p < t),
                "lc_closer_to_teacher": sum(x > 0 for x in an), "lc_farther_from_teacher": sum(x < 0 for x in an),
                "crossed_the_teacher": crossed, "median_gap_closure": med(gap),
                "closer_among_lc_worse_rows": sum(an[k] > 0 for k in worse_rows),
                "lc_worse_rows": len(worse_rows),
                "median_margin_plain": med(mp), "median_margin_lc": med(ml),
                "median_delta_margin": med([b - a for a, b in zip(mp, ml) if None not in (a, b)]),
                "median_delta_answer_mass_F": med([F(b) - F(a) for a, b in zip(fP, fQ) if a and b and F(a) is not None and F(b) is not None]),
                "median_delta_unknown_given_answer": med([(b["UNKNOWN"] - F(b)) - (a["UNKNOWN"] - F(a))
                                                         for a, b in zip(fP, fQ) if a and b and F(a) is not None and F(b) is not None]),
            }
        au_p = auroc([margin(first(records[i], plain_c, v)) for i in by["unk"]], [margin(first(records[i], plain_c, v)) for i in by["known"]])
        au_l = auroc([margin(first(records[i], lc_c, v)) for i in by["unk"]], [margin(first(records[i], lc_c, v)) for i in by["known"]])
        au_t = auroc([margin(first(records[i], "T", v)) for i in by["unk"]], [margin(first(records[i], "T", v)) for i in by["known"]])
        dm_u, dm_k = cell["unk"]["median_delta_margin"], cell["known"]["median_delta_margin"]
        if au_p is not None and au_l is not None and au_l < au_p - AUROC_MARGIN:
            reading = "DISCRIMINATION_LOSS"
        elif None not in (dm_u, dm_k, au_p, au_l) and dm_u < 0 and dm_k < 0 and abs(au_l - au_p) <= AUROC_MARGIN:
            reading = "BIAS_SHIFT"
        else:
            reading = "OTHER"
        cell["auroc_unknown_vs_known"] = {"plain": au_p, "lc": au_l, "teacher": au_t, "reading_M2": reading,
                                          "wording": ("UNKNOWN-token propensity" if au_t is not None and
                                                      abs(au_t - 0.5) <= TEACHER_AUROC_NEAR_HALF else "abstention")}
        cw = cell["unk"]["lc_worse_rows"]
        frac = cell["unk"]["closer_among_lc_worse_rows"] / cw if cw else None
        cell["reading_M1_teacher_tracking_rowwise"] = (None if frac is None else
                                                       "TEACHER_TRACKING" if frac >= 0.75 else
                                                       "LC_ARTEFACT" if frac <= 0.25 else "MIXED")
        out["cells"][X] = cell
    # per family and tokenisation identity, cell A, UNKNOWN rows
    fam = {}
    for i in by["unk"]:
        r = records[i]
        p, q = nats(r, "S20_plain", "V4"), nats(r, "S20_lc", "V4")
        if p is None or q is None:
            continue
        f = fam.setdefault(r["family"], {"worse": 0, "better": 0, "n": 0})
        f["n"] += 1
        f["worse"] += q > p
        f["better"] += q < p
    out["A_unk_by_family"] = fam
    # the UNKNOWN token alone (per_token[0]) vs the registered 2-token endpoint, cell A, UNKNOWN rows
    tok_s, end_s = [], []
    for i in by["unk"]:
        try:
            pt = records[i]["m"]["S20_plain"]["V4"]["V4"]["per_token"][0]
            lt = records[i]["m"]["S20_lc"]["V4"]["V4"]["per_token"][0]
        except (KeyError, IndexError):
            continue
        tok_s.append(av.sgn(pt - lt))
        end_s.append(av.sgn(nats(records[i], "S20_plain", "V4") - nats(records[i], "S20_lc", "V4")))
    out["A_unk_unknown_token_only"] = {"lc_worse": sum(x < 0 for x in tok_s), "lc_better": sum(x > 0 for x in tok_s),
                                       "agrees_with_endpoint": sum(1 for a, b in zip(tok_s, end_s) if a == b), "rows": len(tok_s)}
    tid = {}
    for flag in (True, False):
        rows = [i for i in by["unk"] if records[i]["input_equal_V0"]["V4"] == flag]
        s = av.signs_of_difference([nats(records[i], "S20_plain", "V4") for i in rows],
                                   [nats(records[i], "S20_lc", "V4") for i in rows])
        tid["V4_input_equals_V0" if flag else "V4_input_differs"] = {"rows": len(rows), "worse": sum(x < 0 for x in s),
                                                                     "better": sum(x > 0 for x in s)}
    out["A_unk_by_tokenisation_identity"] = tid
    # baseline vs endpoint (A vs D), UNKNOWN rows
    b = [av.log_ratio(nats(records[i], "S20_plain", "V4"), nats(records[i], "S18_plain", "V2")) for i in by["unk"]]
    e = [av.log_ratio(nats(records[i], "S20_lc", "V4"), nats(records[i], "S18_lc", "V2")) for i in by["unk"]]
    out["baseline_vs_endpoint_A_vs_D"] = {"median_log_plain_A_over_plain_D": med(b), "median_log_lc_A_over_lc_D": med(e)}
    # vocabulary factor at depth 20, UNKNOWN rows: output support vs input tokenisation
    def d_mixed(i, vin, sup):
        return av.log_ratio(nats(records[i], "S20_plain", vin, sup), nats(records[i], "S20_lc", vin, sup))
    out["vocab_split_depth20_unk"] = {
        "median_output_support_term_V4": med([d_mixed(i, "V4", "V4") - d_mixed(i, "V4", "V0") for i in by["unk"]
                                              if "V0" in records[i]["m"].get("S20_plain", {})]),
        "median_input_term_V4": med([d_mixed(i, "V4", "V0") - d_mixed(i, "V0", "V0") for i in by["unk"]
                                     if "V0" in records[i]["m"].get("S20_plain", {})])}
    # greedy behaviour
    gen = {}
    for cell in ("A_plain", "A_lc", "T4", "T0"):
        texts = {i: records[i]["gen"].get(cell) for i in ids}
        if any(t is None for t in texts.values()):
            continue
        normed = {i: t.strip().strip(".").upper() for i, t in texts.items()}
        gen[cell] = {"abstains_on_unk": sum(normed[i] == "UNKNOWN" for i in by["unk"]),
                     "false_abstain_on_known": sum(normed[i] == "UNKNOWN" for i in by["known"]),
                     "correct_on_known": sum(normed[i] == records[i]["reference"] for i in by["known"]),
                     "n_unk": len(by["unk"]), "n_known": len(by["known"])}
    out["greedy"] = gen
    return out
