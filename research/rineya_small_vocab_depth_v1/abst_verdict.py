"""ABST-001 -- the registered verdict (ABST001_PREREG.md §6), exact and distribution-free.

Endpoint per row i and cell X: the teacher-forced NLL (nats) of the reference + <|im_end|>
(UNKNOWN rows: 'UNKNOWN' + <|im_end|>; known rows: the digit + <|im_end|>).
  s_X,i = sign(NLL_plain - NLL_lc)   > 0: LC makes the reference MORE likely, < 0: LESS likely
Signs are taken on the DIFFERENCE, never on a ratio, so no value can raise after the panel is read:
bitwise-equal pairs are ties (dropped, counted); NaN or a missing value (None) is a tie ('undefined').

FIXED SEQUENCE (alpha carried forward; steps not reached are UNTESTED):
  Step 1  H1, cell A (S20/V4), UNKNOWN rows: exact sign test, one-sided 0.025 in EACH direction
            LC_LOWERS_UNKNOWN  #(LC worse) significant      -- the HLCC lead replicated
            LC_RAISES_UNKNOWN  #(LC better) significant
            NOT_CONFIRMED      neither
          (the two directional labels together have size 0.039 at n = 256; the chain continues only on
          LC_LOWERS_UNKNOWN, with alpha 0.025.)
  Step 2  UNKNOWN-row anchor, cell A vs the teacher T4 (same vocabulary):
            g_i = sign(|log NLL_plain - log NLL_T| - |log NLL_lc - log NLL_T|)   (> 0: LC closer)
          two-sided 0.025 (0.0125 per tail): TOWARD_TEACHER / AWAY_FROM_TEACHER / ANCHOR_UNRESOLVED (stop).
          NOTE (review): with the spent geometry (plain more UNKNOWN-prone than the teacher on every row),
          TOWARD_TEACHER is nearly implied by step 1 unless LC overshoots -- it is a statement of DIRECTION
          toward the teacher, not of mechanism.
  After step 2 rejects in either direction, the remaining 0.025 is split equally (Bonferroni):
  Step 2b (0.0125, only after TOWARD_TEACHER): KNOWN-row anchor, cell A vs T4, one-sided toward the teacher.
          On known rows the geometry is reversed (plain LESS confident in the gold digit than the teacher),
          so generic damage (every NLL up) moves AWAY there.
            TEACHER_CONSISTENT_DIRECTION    step 2 TOWARD, step 2b rejects, and fewer than half of the UNKNOWN
                                            rows with defined values OVER-SUPPRESS (plain < T < LC: LC's UNKNOWN
                                            NLL passes above the teacher's)
            TOWARD_TEACHER_OVER_SUPPRESSION TOWARD, but at least half of those rows over-suppress
            TOWARD_TEACHER_NOT_CORROBORATED TOWARD, no over-suppression majority, step 2b does not reject
          R1 (cold review M1): this label is DIRECTION-CONSISTENT, not teacher-specific -- a teacher-blind
          shift of probability mass from UNKNOWN to the digits also passes steps 1, 2 and 2b. No label
          licenses abstention; a row-level tracking statistic is reported descriptively (abst_describe).
  Step 3  (0.0125) Holm over eight one-sided hypotheses (valid under any dependence):
            D+ / D-  LC makes UNKNOWN more / less likely in D (S18/V2 -- HLCC's secondary)
            B+ / B-  ... in B (S20/V2 -- vocabulary swapped)     C+ / C-  ... in C (S18/V4 -- depth swapped)
            dV       d_B - d_A > 0      dD   d_C - d_A > 0        (d_X,i = log(NLL_plain / NLL_lc))
Everything else is descriptive (abst_describe).
"""
from __future__ import annotations

import math

import hlcc_verdict as hv

ALPHA = 0.025
ALPHA_ANCHOR_TAIL = 0.0125
ALPHA_AFTER_ANCHOR = 0.0125          # each of step 2b and step 3 (Bonferroni split of the carried 0.025)


def sgn(x) -> int:
    """+1 / -1 / 0; NaN or None -> 0 (a tie)."""
    if x is None or x != x:
        return 0
    return (x > 0) - (x < 0)


def signs_of_difference(plain: list, lc: list) -> list[int]:
    """sign(plain - lc). inf - inf is a tie; a finite value against inf has a sign; None/NaN -> tie."""
    out = []
    for a, b in zip(plain, lc):
        if a is None or b is None or a != a or b != b or a == b:
            out.append(0)
            continue
        out.append(1 if a > b else -1)
    return out


def log_ratio(a, b) -> float:
    """log(a / b) with the conventions: a == b -> 0; either None or NaN, 0/0 or inf/inf -> NaN;
    a = 0 < b -> -inf; a > 0 = b -> +inf. Never raises."""
    if a is None or b is None or a != a or b != b:
        return float("nan")
    if a == b:
        return 0.0
    if a <= 0 and b <= 0:
        return float("nan")
    if a <= 0:
        return float("-inf")
    if b <= 0:
        return float("inf")
    if math.isinf(a) and math.isinf(b):
        return float("nan")
    if math.isinf(a):
        return float("inf")
    if math.isinf(b):
        return float("-inf")
    return math.log(a) - math.log(b)


def signed_test(signs: list[int]) -> dict:
    """Exact sign test on a list of +1 / -1 / 0 (0 = tie, dropped)."""
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = sum(1 for s in signs if s < 0)
    n = n_pos + n_neg
    return {"rows": len(signs), "ties_or_undefined": len(signs) - n, "n": n, "positive": n_pos, "negative": n_neg,
            "p_positive": hv.binom_upper_tail(n, n_pos) if n else 1.0,
            "p_negative": hv.binom_upper_tail(n, n_neg) if n else 1.0}


def anchor_signs(plain: list, lc: list, teacher: list) -> list[int]:
    """sign(|log plain - log T| - |log lc - log T|): +1 LC closer to the teacher; undefined -> 0."""
    out = []
    for a, b, t in zip(plain, lc, teacher):
        ga, gb = log_ratio(a, t), log_ratio(b, t)
        if ga != ga or gb != gb:
            out.append(0)
            continue
        da, db = abs(ga), abs(gb)
        if da == db:
            out.append(0)          # includes inf == inf
            continue
        out.append(1 if da > db else -1)
    return out


def over_suppression(plain: list, lc: list, teacher: list) -> tuple[int, int]:
    """(rows where plain < T < LC -- LC's NLL passes ABOVE the teacher's, i.e. over-suppresses the
    reference --, rows with all three values finite). Crossings in the other direction are not counted."""
    k = n = 0
    for a, b, t in zip(plain, lc, teacher):
        if None in (a, b, t) or any(x != x or math.isinf(x) for x in (a, b, t)):
            continue
        n += 1
        k += a < t < b
    return k, n


def contrast_signs(d_ref: list, d_other: list) -> list[int]:
    """sign(d_other - d_ref) per row; a tie if either is None/NaN or they are equal (incl. inf == inf)."""
    out = []
    for a, b in zip(d_ref, d_other):
        if a is None or b is None or a != a or b != b or a == b:
            out.append(0)
            continue
        out.append(1 if b > a else -1)
    return out


def holm(pvals: dict, alpha: float) -> dict:
    """Holm step-down: sort ascending; reject while p_(k) < alpha / (m - k); stop at the first failure."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    rej = {k: False for k in pvals}
    for k, name in enumerate(order):
        if pvals[name] < alpha / (m - k):
            rej[name] = True
        else:
            break
    return rej


ATTRIBUTION = {  # (B sign, C sign) -> label, used only when D+ is rejected (the 18/V2 sign replicated)
    (-1, 1): "SIGN_FOLLOWS_DEPTH",
    (1, -1): "SIGN_FOLLOWS_VOCABULARY",
    (-1, -1): "A_SIGN_EITHER_FEATURE_SUFFICIENT__REVERSAL_NEEDS_BOTH",
    (1, 1): "A_SIGN_NEEDS_BOTH_FEATURES__EITHER_CHANGE_REVERSES",
}


def verdict(cells: dict) -> dict:
    """cells: 'A' {'plain','lc'} and 'T4' {'teacher'} on the UNKNOWN rows; 'A_known' {'plain','lc'} and
    'T4_known' {'teacher'} on the known rows; optionally 'B','C','D' {'plain','lc'} on the UNKNOWN rows.
    All lists over the same rows in the same order."""
    A = cells["A"]
    t1 = signed_test(signs_of_difference(A["plain"], A["lc"]))
    worse, better = t1["p_negative"] < ALPHA, t1["p_positive"] < ALPHA
    h1 = "LC_LOWERS_UNKNOWN" if worse else "LC_RAISES_UNKNOWN" if better else "NOT_CONFIRMED"
    dA = [log_ratio(p, q) for p, q in zip(A["plain"], A["lc"])]
    rise = [-x for x in dA if x == x]           # log(NLL_lc / NLL_plain)
    lb = hv.median_lower_bound(rise, ALPHA) if rise else None
    out = {"H1": {**t1, "label": h1, "alpha_per_direction": ALPHA,
                  "size_median_row_ratio_lc_over_plain_lower_bound_log": lb,
                  "size_median_row_ratio_lc_over_plain_at_least": None if lb is None else math.exp(lb)},
           "H2": {"label": "UNTESTED"}, "H3": {"label": "UNTESTED"}}
    if h1 != "LC_LOWERS_UNKNOWN":
        out["chain_stop"] = "step 1"
        return out
    T = cells["T4"]["teacher"]
    t2 = signed_test(anchor_signs(A["plain"], A["lc"], T))
    toward, away = t2["p_positive"] < ALPHA_ANCHOR_TAIL, t2["p_negative"] < ALPHA_ANCHOR_TAIL
    n_over, n_def = over_suppression(A["plain"], A["lc"], T)
    h2 = {**t2, "alpha_per_tail": ALPHA_ANCHOR_TAIL, "over_suppressing_rows": n_over, "rows_defined": n_def,
          "over_suppression_majority": n_over * 2 >= max(n_def, 1)}
    if toward:
        tk = signed_test(anchor_signs(cells["A_known"]["plain"], cells["A_known"]["lc"], cells["T4_known"]["teacher"]))
        k_ok = tk["p_positive"] < ALPHA_AFTER_ANCHOR
        h2["step_2b_known_anchor"] = {**tk, "alpha": ALPHA_AFTER_ANCHOR, "rejects_toward_teacher": k_ok}
        if h2["over_suppression_majority"]:
            h2["label"] = "TOWARD_TEACHER_OVER_SUPPRESSION"
        elif k_ok:
            h2["label"] = "TEACHER_CONSISTENT_DIRECTION"
        else:
            h2["label"] = "TOWARD_TEACHER_NOT_CORROBORATED"
    elif away:
        h2["label"] = "AWAY_FROM_TEACHER"
    else:
        h2["label"] = "ANCHOR_UNRESOLVED"
    out["H2"] = h2
    if h2["label"] == "ANCHOR_UNRESOLVED":
        out["chain_stop"] = "step 2"
        return out
    if any(X not in cells or not cells[X].get("plain") for X in ("B", "C", "D")):
        out["H3"] = {"label": "NOT_MEASURED"}
        out["chain_stop"] = "step 3 cells missing"
        return out
    tests = {}
    for X in ("D", "B", "C"):
        t = signed_test(signs_of_difference(cells[X]["plain"], cells[X]["lc"]))
        tests[X + "+"] = {**t, "p": t["p_positive"]}
        tests[X + "-"] = {**t, "p": t["p_negative"]}
    for name, X in (("dV", "B"), ("dD", "C")):
        dX = [log_ratio(p, q) for p, q in zip(cells[X]["plain"], cells[X]["lc"])]
        t = signed_test(contrast_signs(dA, dX))
        tests[name] = {**t, "p": t["p_positive"]}
    rej = holm({k: v["p"] for k, v in tests.items()}, ALPHA_AFTER_ANCHOR)
    sign_of = {X: (1 if rej[X + "+"] else -1 if rej[X + "-"] else 0) for X in ("D", "B", "C")}
    if sign_of["D"] == 1:
        key = (sign_of["B"], sign_of["C"])
        att = ATTRIBUTION.get(key, "PARTIAL" if any(key) else "UNRESOLVED")
    elif sign_of["D"] == -1:
        att = "NO_REVERSAL"
    else:
        att = "D_UNRESOLVED"
    out["H3"] = {"label": att, "alpha": ALPHA_AFTER_ANCHOR, "tests": tests, "holm_rejected": rej, "sign_of": sign_of,
                 "contributions": {"vocabulary_dV": rej["dV"], "depth_dD": rej["dD"]}}
    out["chain_stop"] = None
    return out
