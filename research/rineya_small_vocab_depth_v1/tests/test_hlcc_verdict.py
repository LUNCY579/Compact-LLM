"""HLCC-001 verdict: the sign test and the order-statistic bound are exact; controls both ways."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import hlcc_verdict as hv  # noqa: E402


def test_binomial_tail_known_values():
    assert hv.binom_upper_tail(10, 10) == 1 / 1024
    assert abs(hv.binom_upper_tail(10, 8) - (45 + 10 + 1) / 1024) < 1e-15
    assert hv.binom_upper_tail(5, 0) == 1.0 and hv.binom_upper_tail(5, 6) == 0.0


def test_negative_control_identity_is_all_ties_and_never_confirms():
    """Plain vs plain (identity fold): every d is exactly 0 -> no signed rows, p = 1, NOT_CONFIRMED."""
    v = hv.verdict([0.0] * 300, [0.0] * 180)
    assert v["verdict"] == "NOT_CONFIRMED" and v["kl"]["n"] == 0 and v["kl"]["p_greater"] == 1.0


def test_positive_control_smallvd001_shape():
    """SMALLVD-001's REPORT/HOLDOUT signs (253/257 KL, 171/186 bpb) must confirm, large."""
    d_kl = [0.88] * 253 + [-0.2] * 4
    d_bpb = [0.3] * 171 + [-0.1] * 15
    v = hv.verdict(d_kl, d_bpb)
    assert v["verdict"] == "CONFIRMED" and v["size_large"] is True
    assert v["kl"]["p_greater"] < 1e-60


def test_harmful_and_partial_branches():
    assert hv.verdict([-0.5] * 60, [0.5] * 60)["verdict"] == "HARMFUL"
    assert hv.verdict([0.5] * 60, [0.5] * 3 + [-0.5] * 3)["verdict"] == "PARTIAL"
    assert hv.verdict([0.5, -0.5] * 30, [0.5, -0.5] * 30)["verdict"] == "NOT_CONFIRMED"


def test_sign_test_is_calibrated_under_a_skewed_null_by_simulation():
    """d = LogNormal(0, 1.5) - 1: median exactly 0 but heavily RIGHT-skewed with MEAN ~ +2.1 (the shape
    that broke SMALLVD-001's bootstrap). The sign test tests the median, so its false-CONFIRM rate must
    stay <= alpha even though every mean-based test would reject. (A first version multiplied by a
    random sign -- that is SYMMETRIC, and never tested the case that matters.)"""
    rng = random.Random(3)
    alpha, sims = hv.ALPHA_EACH, 4000
    for sgn in (1, -1):                                  # right- and left-skewed
        rej = mean_on_skew_side = 0
        for _ in range(sims):
            d = [sgn * (rng.lognormvariate(0, 1.5) - 1.0) for _ in range(250)]
            mean_on_skew_side += sgn * sum(d) > 0
            rej += hv.sign_test(d)["p_greater"] < alpha
        assert mean_on_skew_side / sims > 0.99            # the null really has a non-zero MEAN
        assert rej / sims <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / sims), (sgn, rej / sims)


def test_median_lower_bound_coverage_by_simulation():
    """The order-statistic bound must cover the true median with probability >= 1 - alpha, for a
    symmetric mixture and for a skewed shape (median 0.8 + LogNormal(0, 1.2) - 1 has median 0.8)."""
    rng = random.Random(5)
    true_med, sims = 0.8, 3000
    for draw in (lambda: true_med + rng.gauss(0, 1) * rng.choice((0.2, 2.0)),
                 lambda: true_med + rng.lognormvariate(0, 1.2) - 1.0):
        miss = 0
        for _ in range(sims):
            lb = hv.median_lower_bound([draw() for _ in range(150)])
            miss += lb is not None and lb > true_med
        assert miss / sims <= hv.ALPHA_EACH + 3 * math.sqrt(hv.ALPHA_EACH / sims)


def test_clustered_rows_break_the_row_level_test_and_units_fix_it():
    """The panel's real cluster sizes (19 books for 32 prose rows, 11 tasks for 32 long-context rows,
    singletons elsewhere). Under H0 each CLUSTER has a shared shift with median 0 and rows add small
    noise: the ROW-level sign test must be measurably anticonservative (so the unit rule is not
    decoration) and the UNIT-level test must hold alpha."""
    rng = random.Random(11)
    sizes = [3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [6, 4, 4, 4, 3, 2, 2, 2, 2, 2, 1] + [1] * 220
    assert sum(sizes[:19]) == 32 and sum(sizes[19:30]) == 32
    alpha, sims, rej_row, rej_unit = hv.ALPHA_EACH, 3000, 0, 0
    for _ in range(sims):
        d_row, unit, d_by_id = [], {}, {}
        for c, n in enumerate(sizes):
            u = rng.gauss(0, 1.0)
            for j in range(n):
                x = u + rng.gauss(0, 0.05)
                d_row.append(x)
                d_by_id[f"{c}:{j}"], unit[f"{c}:{j}"] = x, str(c)
        rej_row += hv.sign_test(d_row)["p_greater"] < alpha
        rej_unit += hv.sign_test(list(hv.collapse(d_by_id, unit).values()))["p_greater"] < alpha
    tol = 3 * math.sqrt(alpha * (1 - alpha) / sims)
    assert rej_unit / sims <= alpha + tol
    assert rej_row / sims > alpha + tol, rej_row / sims          # the dependence is real, and measurable


def test_unit_of_on_the_panel_id_shapes():
    t = lambda rid: {"id": rid, "kind": "text", "source": "x"}
    assert hv.unit_of(t("confirm:story:Mary_Shelley_Frankenstein:62")) == "doc:story:Mary_Shelley_Frankenstein"
    assert hv.unit_of(t("confirm:story:Mary_Shelley_Frankenstein:7")) == hv.unit_of(t("confirm:story:Mary_Shelley_Frankenstein:62"))
    assert hv.unit_of(t("confirm:md:43d41905c53ed309:3")) == "doc:md:43d41905c53ed309"
    assert hv.unit_of(t("confirm:py:b036a3c6a59557e3")) != hv.unit_of(t("confirm:py:a5bc854d90a9eb9a"))
    lc = lambda rid: {"id": rid, "kind": "chat", "source": "cc0:validation"}
    assert hv.unit_of(lc("rineya_v3_sft__linked_fact__validation_000__00390")) == \
        hv.unit_of(lc("rineya_v3_sft__linked_fact__validation_003__00012")) == "task:linked_fact"
    assert hv.unit_of({"id": "RTG1-9433dd", "kind": "chat", "source": "rtg1:dev"}) == "row:RTG1-9433dd"
    assert hv.collapse({"a": 1.0, "b": 3.0, "c": -2.0}, {"a": "u", "b": "u", "c": "v"}) == {"u": 2.0, "v": -2.0}


def test_power_is_high_at_the_planned_sizes_even_for_a_weak_effect():
    assert hv.power_sign_test(180, 0.65) > 0.9          # bpb rows, a far weaker effect than observed
    assert hv.power_sign_test(300, 0.62) > 0.9          # KL rows


def test_r2_even_unit_median_is_biased_under_a_skewed_null_and_majority_sign_is_not():
    """R2's reason, on the statistic itself (the first skewed-null test never called collapse()).
    150 two-row units, rows i.i.d. LogNormal(0, 1.5) - 1 (median 0): the unit MEDIAN of an even unit is
    the MEAN of its rows, positive ~63.5% of the time, so the unit-median test false-confirms far
    above alpha. The ROW-level test and the MAJORITY-SIGN units must hold alpha under the same null."""
    rng = random.Random(21)
    alpha, sims = hv.ALPHA_EACH, 1500
    rej_row = rej_med = rej_maj = 0
    unit = {f"{u}:{j}": str(u) for u in range(150) for j in range(2)}
    for _ in range(sims):
        d = {i: rng.lognormvariate(0, 1.5) - 1.0 for i in unit}
        rej_row += hv.sign_test(list(d.values()))["p_greater"] < alpha
        rej_med += hv.sign_test(list(hv.collapse(d, unit).values()))["p_greater"] < alpha
        rej_maj += hv.sign_test(list(hv.majority_sign_collapse(d, unit).values()))["p_greater"] < alpha
    tol = 3 * math.sqrt(alpha * (1 - alpha) / sims)
    assert rej_row / sims <= alpha + tol and rej_maj / sims <= alpha + tol, (rej_row / sims, rej_maj / sims)
    assert rej_med / sims > 0.5, rej_med / sims          # grossly anticonservative, not marginal


def test_majority_sign_collapse_and_direction_only_verdict():
    d = {"a": 0.5, "b": -0.1, "c": 0.2, "x": -1.0, "y": 2.0, "z": 0.0}
    u = {"a": "U", "b": "U", "c": "U", "x": "V", "y": "V", "z": "W"}
    assert hv.majority_sign_collapse(d, u) == {"U": 1, "V": 0, "W": 0}
    v = hv.verdict([1] * 60, [1] * 60, with_size=False)
    assert v["verdict"] == "CONFIRMED" and v["size_large"] is None and v["kl"]["median_lower_bound"] is None
