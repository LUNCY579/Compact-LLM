"""ABST-001 verdict: every label reachable, never raises on degenerate values, and the REGISTERED
statistic (abst_verdict.verdict, not a proxy) holds alpha under finite-pool, stratified nulls."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import abst_verdict as av  # noqa: E402

FAMILY_POOLS = (112, 109, 109, 102)      # fresh RTG1-dev UNKNOWN pool sizes (ABST001_PREREG.md §3)
PER_FAMILY = 64


def teacher_for(plain, lc, anchor, overshoot=False):
    """closer (+1): teacher beyond lc on lc's side; farther (-1): teacher on the other side of plain;
    overshoot: teacher just short of lc (the row CROSSES the teacher yet is closer)."""
    out = []
    for p, q, a in zip(plain, lc, anchor):
        if overshoot and a > 0:
            out.append(q / 1.05 if q > p else q * 1.05)
        else:
            out.append(q * (1.5 if q > p else 1 / 1.5) if a > 0 else p * (1 / 1.5 if q > p else 1.5))
    return out


def cells_from(worse_A, n=256, anchor=None, D=None, B=None, C=None, known_anchor=None, overshoot=False):
    """UNKNOWN rows: plain = 4.0 + small; lc = plain * 1.3 if worse else / 1.3; teacher placed so the
    anchor sign per row is 'anchor'. Known rows: plain 2.0, LC lower (/1.3), teacher per 'known_anchor'.
    D/B/C: lists of +1/-1 (LC makes UNKNOWN more / less likely)."""
    plain = [4.0 + 0.01 * i for i in range(n)]
    lc = [p * (1.3 if w else 1 / 1.3) for p, w in zip(plain, worse_A)]
    anchor = anchor or [1] * n
    known_anchor = known_anchor or [1] * n
    kp = [2.0 + 0.01 * i for i in range(n)]
    kl = [p / 1.3 for p in kp]
    cells = {"A": {"plain": plain, "lc": lc}, "T4": {"teacher": teacher_for(plain, lc, anchor, overshoot)},
             "A_known": {"plain": kp, "lc": kl}, "T4_known": {"teacher": teacher_for(kp, kl, known_anchor)}}
    for X, s in (("D", D), ("B", B), ("C", C)):
        if s is not None:
            cells[X] = {"plain": plain[:], "lc": [p / 1.3 if v > 0 else p * 1.3 for p, v in zip(plain, s)]}
    return cells


def test_primary_labels_reachable_and_thresholds():
    assert av.verdict(cells_from([True] * 145 + [False] * 111))["H1"]["label"] == "LC_LOWERS_UNKNOWN"
    assert av.verdict(cells_from([True] * 144 + [False] * 112))["H1"]["label"] == "NOT_CONFIRMED"
    assert av.verdict(cells_from([False] * 145 + [True] * 111))["H1"]["label"] == "LC_RAISES_UNKNOWN"
    v = av.verdict(cells_from([False] * 200 + [True] * 56))
    assert v["H2"]["label"] == "UNTESTED" and v["chain_stop"] == "step 1"


def test_anchor_labels_reachable():
    w = [True] * 200 + [False] * 56
    assert av.verdict(cells_from(w, anchor=[1] * 256))["H2"]["label"] == "FIDELITY"
    assert av.verdict(cells_from(w, anchor=[1] * 256, known_anchor=[-1] * 256))["H2"]["label"] == "TOWARD_TEACHER_NOT_CORROBORATED"
    assert av.verdict(cells_from(w, anchor=[1] * 256, known_anchor=[1, -1] * 128))["H2"]["label"] == "TOWARD_TEACHER_NOT_CORROBORATED"
    v = av.verdict(cells_from(w, anchor=[1] * 256, overshoot=True))
    assert v["H2"]["label"] == "TOWARD_TEACHER_OVERSHOOT" and v["H2"]["overshoot_majority"]
    v = av.verdict(cells_from(w, anchor=[-1] * 256))
    assert v["H2"]["label"] == "AWAY_FROM_TEACHER" and "step_2b_known_anchor" not in v["H2"]
    v = av.verdict(cells_from(w, anchor=[1, -1] * 128))
    assert v["H2"]["label"] == "ANCHOR_UNRESOLVED" and v["H3"]["label"] == "UNTESTED"


def test_step3_is_reached_after_away_too_and_uses_the_split_alpha():
    w = [True] * 220 + [False] * 36
    v = av.verdict(cells_from(w, anchor=[-1] * 256, D=[1] * 256, B=[-1] * 256, C=[1] * 256))
    assert v["H2"]["label"] == "AWAY_FROM_TEACHER" and v["H3"]["label"] == "DEPTH" and v["H3"]["alpha"] == 0.0125


def test_attribution_labels_reachable():
    w = [True] * 220 + [False] * 36
    good, bad, mixed = [1] * 256, [-1] * 256, [1, -1] * 128
    lab = lambda **k: av.verdict(cells_from(w, **k))["H3"]["label"]
    assert lab(D=good, B=bad, C=good) == "DEPTH"
    assert lab(D=good, B=good, C=bad) == "VOCABULARY"
    assert lab(D=good, B=bad, C=bad) == "EITHER_SUFFICIENT"
    assert lab(D=good, B=good, C=good) == "BOTH_REQUIRED"
    assert lab(D=good, B=bad, C=mixed) == "PARTIAL"
    assert lab(D=good, B=mixed, C=mixed) == "UNRESOLVED"
    assert lab(D=bad, B=bad, C=bad) == "NO_REVERSAL"
    assert lab(D=mixed, B=bad, C=good) == "D_UNRESOLVED"
    assert av.verdict(cells_from(w))["H3"]["label"] == "NOT_MEASURED"


def test_degenerate_values_never_raise_and_count_as_ties():
    nan, inf = float("nan"), float("inf")
    plain = [0.0, inf, nan, 2.0, 0.0, 3.0]
    lc = [0.0, inf, 1.0, 0.0, 1.0, 3.0]
    s = av.signs_of_difference(plain, lc)
    assert s == [0, 0, 0, 1, -1, 0]
    assert av.log_ratio(0.0, 0.0) == 0.0 and av.log_ratio(2.0, 0.0) == inf and av.log_ratio(0.0, 2.0) == -inf
    assert av.log_ratio(inf, inf) == 0.0 and math.isnan(av.log_ratio(nan, 1.0))
    t = av.anchor_signs([0.0, 1.0], [0.0, 2.0], [0.0, 2.0])
    assert t == [0, 1]
    v = av.verdict({"A": {"plain": plain, "lc": lc}, "T4": {"teacher": [1.0] * 6}})
    assert v["H1"]["ties_or_undefined"] == 4 and v["H1"]["label"] == "NOT_CONFIRMED"


def test_holm_stepdown():
    assert av.holm({"a": 0.001, "b": 0.02, "c": 0.5}, 0.025) == {"a": True, "b": False, "c": False}
    assert av.holm({"a": 0.001, "b": 0.012, "c": 0.024}, 0.025) == {"a": True, "b": True, "c": True}
    assert av.holm({"a": 0.001, "b": 0.012, "c": 0.024}, 0.0125) == {"a": True, "b": False, "c": False}


def finite_pool_draw(rng, p_worse_by_family, effect=1.3):
    """Draw 64 rows per family WITHOUT replacement from finite pools of the real sizes, where each pool
    holds round(p * size) 'worse' rows; NLL log-ratios skewed and heavy-tailed."""
    plain, lc = [], []
    for size, p in zip(FAMILY_POOLS, p_worse_by_family):
        k = round(p * size)
        pool = [True] * k + [False] * (size - k)
        for w in rng.sample(pool, PER_FAMILY):
            a = math.exp(rng.gauss(1.4, 0.6))
            r = math.exp(abs(rng.lognormvariate(0, 1.0) - 0.5) + 1e-9) * (effect / 1.3)
            plain.append(a)
            lc.append(a * r if w else a / r)
    return plain, lc


def test_null_calibration_on_the_registered_statistic():
    """Finite pools with EXACTLY half the rows worse (and the heterogeneous 2/3,2/3,2/3,0 null): the
    false LC_LOWERS and false LC_RAISES rates of verdict() stay <= alpha + 3 SE."""
    rng = random.Random(20260926)
    sims = 2000
    se3 = 3 * math.sqrt(av.ALPHA * (1 - av.ALPHA) / sims)
    for fam in ((0.5, 0.5, 0.5, 0.5), (2 / 3, 2 / 3, 2 / 3, 0.0)):
        low = high = 0
        for _ in range(sims):
            plain, lc = finite_pool_draw(rng, fam)
            lab = av.verdict({"A": {"plain": plain, "lc": lc}, "T4": {"teacher": plain}})["H1"]["label"]
            low += lab == "LC_LOWERS_UNKNOWN"
            high += lab == "LC_RAISES_UNKNOWN"
        assert low / sims <= av.ALPHA + se3, (fam, low / sims)
        assert high / sims <= av.ALPHA + se3, (fam, high / sims)


def anchor_null_teacher(rng, plain, lc):
    """An EXACT anchor null: the teacher's log NLL sits at the midpoint of log plain and log lc plus
    symmetric noise, so 'LC closer' and 'LC farther' are equally likely for every row. (A first version
    placed the teacher at plain * exp(+-|log r| * U(0.2, 1.8)): that is P(closer) = 0.41, NOT a null, and
    the test failed -- correctly -- with AWAY_FROM_TEACHER firing 73% of the time.)"""
    return [math.exp((math.log(p) + math.log(q)) / 2 + rng.gauss(0, 1.0)) for p, q in zip(plain, lc)]


def known_cells(rng, toward=True, n=256):
    """Known rows: plain less confident in the digit than the teacher (the spent geometry); LC moves the
    digit NLL toward the teacher (toward=True) or away from it."""
    kp = [math.exp(rng.gauss(1.0, 0.4)) for _ in range(n)]
    kt = [p * 0.3 for p in kp]
    kl = [p * (0.7 if toward else 1.3) for p in kp]
    return {"A_known": {"plain": kp, "lc": kl}, "T4_known": {"teacher": kt}}


TOWARD_LABELS = ("FIDELITY", "TOWARD_TEACHER_OVERSHOOT", "TOWARD_TEACHER_NOT_CORROBORATED")


def test_power_at_the_lead_shape_and_exact_anchor_null():
    """Lead-shaped truth (three families 0.8 worse, prerequisite_schedule 0.1): H1 fires almost always.
    Under the exact UNKNOWN-anchor null, any step-2 rejection (a TOWARD label or AWAY, 0.0125 per tail)
    stays <= 0.025 + 3 SE; with the teacher beyond LC and known rows moving toward it, FIDELITY fires."""
    rng = random.Random(7)
    sims, fired, anchor_false, fidelity = 1000, 0, 0, 0
    for _ in range(sims):
        plain, lc = finite_pool_draw(rng, (0.8, 0.1, 0.8, 0.8))
        kc = known_cells(rng)
        v = av.verdict({"A": {"plain": plain, "lc": lc}, "T4": {"teacher": anchor_null_teacher(rng, plain, lc)}, **kc})
        fired += v["H1"]["label"] == "LC_LOWERS_UNKNOWN"
        anchor_false += v["H2"]["label"] in TOWARD_LABELS + ("AWAY_FROM_TEACHER",)
        beyond = [q * math.exp(0.5 * math.log(q / p)) for p, q in zip(plain, lc)]
        fidelity += av.verdict({"A": {"plain": plain, "lc": lc}, "T4": {"teacher": beyond}, **kc})["H2"]["label"] == "FIDELITY"
    assert fired / sims > 0.95
    assert anchor_false / sims <= 0.025 + 3 * math.sqrt(0.025 * 0.975 / sims), anchor_false / sims
    assert fidelity / sims > 0.95


def test_teacher_blind_generic_damage_does_not_earn_fidelity():
    """The review's scenario: LC inflates every NLL (UNKNOWN AND known rows) regardless of the teacher, with
    the spent geometry (UNKNOWN: plain below the teacher by x1.1-2.9; known: plain above it). Step 2 then
    says TOWARD on UNKNOWN rows -- but step 2b must not corroborate, so FIDELITY stays <= 0.025."""
    rng = random.Random(11)
    sims, fid, toward = 1000, 0, 0
    for _ in range(sims):
        plain = [math.exp(rng.gauss(1.4, 0.3)) for _ in range(256)]
        teacher = [p * rng.uniform(1.14, 2.85) for p in plain]
        lc = [p * math.exp(abs(rng.gauss(0, 0.4))) if rng.random() < 0.7 else p / math.exp(abs(rng.gauss(0, 0.4)))
              for p in plain]
        kc = known_cells(rng, toward=False)
        v = av.verdict({"A": {"plain": plain, "lc": lc}, "T4": {"teacher": teacher}, **kc})
        toward += v["H2"]["label"] in TOWARD_LABELS
        fid += v["H2"]["label"] == "FIDELITY"
    assert toward / sims > 0.9                   # the UNKNOWN anchor alone cannot tell (the review was right)
    assert fid / sims <= 0.025 + 3 * math.sqrt(0.025 * 0.975 / sims), fid / sims
