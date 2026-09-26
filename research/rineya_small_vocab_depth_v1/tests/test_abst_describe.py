"""ABST-001 descriptive helpers: known values, and summarise() on a constructed record set."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import abst_describe as ad  # noqa: E402


def test_clopper_pearson_known_values():
    lo, hi = ad.cp_interval(13, 16)
    assert abs(lo - 0.5435) < 1e-3 and abs(hi - 0.9595) < 1e-3          # a first version returned hi = 4e-19
    lo, hi = ad.cp_interval(0, 10)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 10))) < 1e-6
    lo, hi = ad.cp_interval(10, 10)
    assert hi == 1.0 and abs(lo - 0.025 ** (1 / 10)) < 1e-6
    lo, hi = ad.cp_interval(50, 100)
    assert abs(lo - 0.3983) < 1e-3 and abs(hi - 0.6017) < 1e-3


def test_auroc_margin_lse():
    assert ad.auroc([1, 2, 3], [0, 2]) == 0.75
    assert ad.auroc([], [1]) is None
    assert abs(ad.lse([0.0, 0.0]) - 0.6931471805599453) < 1e-12
    fr = {"UNKNOWN": -1.0, "0": -2.0, "1": -2.0, "2": -3.0, "3": -3.0}
    assert abs(ad.margin(fr) - (-1.0 - ad.lse([-2.0, -2.0, -3.0, -3.0]))) < 1e-12


def test_tracking_partial_spearman_null_and_signal():
    """Uniform multiplicative shift: delta independent of the teacher given plain -> partial rho ~ 0.
    Row-level tracking (LC closes a fixed fraction of the log gap) -> partial rho strongly positive."""
    import math
    import random
    rng = random.Random(3)
    P = [math.exp(rng.gauss(1.4, 0.4)) for _ in range(256)]
    T = [p * math.exp(rng.uniform(0.1, 1.0)) for p in P]
    Q_uniform = [p * math.exp(0.3 + rng.gauss(0, 0.05)) for p in P]
    Q_track = [p * math.exp(0.5 * (math.log(t) - math.log(p)) + rng.gauss(0, 0.02)) for p, t in zip(P, T)]
    u = ad.tracking(P, Q_uniform, T)
    k = ad.tracking(P, Q_track, T)
    assert abs(u["partial_spearman"]) < 0.15
    assert k["partial_spearman"] > 0.8 and k["reading"] == "TRACKING_SIGNAL"
    assert ad.ranks([3.0, 1.0, 3.0]) == [2.5, 1.0, 2.5]
    # the 'given plain' null (R1 review): delta depends on z, the teacher is independent of both -> the partial
    # is ~0 while the raw rho(delta, x - z) is strongly negative (confounded through z)
    T_c = [math.exp(rng.gauss(3.0, 0.2)) for _ in P]
    Q_z = [p * math.exp(0.3 + 0.8 * (math.log(p) - 1.4) + rng.gauss(0, 0.3)) for p in P]
    g = ad.tracking(P, Q_z, T_c)
    assert abs(g["partial_spearman"]) < 0.15 and g["raw_spearman_delta_gap"] < -0.3


def test_tracking_is_within_family():
    """R1 review: a teacher-blind shift that is uniform within each family but differs between families (the
    split F4 forecasts), aligned with per-family teacher offsets, makes the POOLED partial look like tracking.
    The registered within-family statistic stays near zero."""
    import math
    import random
    rng = random.Random(11)
    P, Q, T, F = [], [], [], []
    for f, (dm, off) in enumerate([(-0.3, 0.3), (0.3, 0.8), (0.3, 0.8), (0.3, 0.8)]):
        for _ in range(64):
            p = math.exp(rng.gauss(1.45, 0.3))
            P.append(p)
            T.append(p * math.exp(off + rng.gauss(0, 0.1)))
            Q.append(p * math.exp(dm + rng.gauss(0, 0.25)))
            F.append(f)
    r = ad.tracking(P, Q, T, F)
    assert r["pooled_partial_spearman"] > 0.3 and abs(r["partial_spearman"]) < 0.15
    assert r["reading"] != "TRACKING_SIGNAL" and len(r["per_family"]) == 4


def rec(stratum, fam, p, q, t, mp, ml, mt, same=True, cue=False):
    first = lambda m: {"UNKNOWN": m, "0": 0.0, "1": -9.0, "2": -9.0, "3": -9.0, "im_end": -1.0}
    cell = lambda n, m: {"V4": {"V4": {"nats": n, "per_token": [n], "first": first(m), "first_logmass": 0.0}}}
    return {"stratum": stratum, "family": fam, "tier": None, "reference": "UNKNOWN" if stratum == "unk" else "0",
            "input_equal_V0": {"V4": same, "V2": same}, "cue_not_recorded": cue,
            "m": {"S20_plain": cell(p, mp), "S20_lc": cell(q, ml), "T": cell(t, mt)}, "gen": {}}


def test_summarise_teacher_anchor_and_readings():
    R = {}
    for k in range(10):     # UNKNOWN rows: plain 4, LC 6 (worse), teacher 8 -> LC closer, no crossing
        R[f"u{k}"] = rec("unk", "evidence_chain", 4.0, 6.0, 8.0, -1.0, -2.0, -3.0, cue=k < 8)
    for k in range(10):     # known rows
        R[f"k{k}"] = rec("known", "evidence_chain", 2.0, 1.5, 1.8, -4.0, -5.0, -6.0, cue=k < 3)
    s = ad.summarise(R)
    a = s["cells"]["A"]
    assert a["unk"]["lc_worse"] == 10 and a["unk"]["lc_closer_to_teacher"] == 10 and a["unk"]["crossed_the_teacher"] == 0
    assert abs(a["unk"]["median_gap_closure"] - 0.5) < 1e-12
    assert a["reading_M1_teacher_tracking_rowwise"] == "TEACHER_TRACKING"
    assert a["unk"]["plain_nll_below_teacher"] == 10
    assert a["auroc_unknown_vs_known"]["plain"] == 1.0 and a["auroc_unknown_vs_known"]["reading_M2"] == "BIAS_SHIFT"
    assert s["A_unk_by_family"]["evidence_chain"] == {"worse": 10, "better": 0, "n": 10}
    assert "B" not in s["cells"]           # cells not measured are simply absent
    assert s["A_unk_unknown_token_only"] == {"lc_worse": 10, "lc_better": 0, "agrees_with_endpoint": 10, "rows": 10}
    assert a["auroc_unknown_vs_known"]["wording"] == "abstention"
    # R1 outputs (R1 review: none was asserted): the cue, the digit-only reading and anchor, and tracking
    assert s["cue_not_recorded"] == {"unk_rows_with_cue": 8, "known_rows_with_cue": 3,
                                     "cue_only_auroc_unknown_vs_known": 0.5 + 0.5 * (8 / 10 - 3 / 10)}
    assert s["A_known_digit_token_only"] == {"lc_worse": 0, "lc_better": 10, "rows": 10}
    # known rows: plain 2.0 -> LC 1.5 past the teacher's 1.8, so LC is FARTHER on the digit (|log 1.5/1.8| > |log 2/1.8|)
    assert s["A_known_digit_anchor"] == {"closer": 0, "farther": 10, "rows": 10, "plain_digit_nll_below_teacher": 0}
    tr = s["A_tracking"]["unk"]
    assert tr["rows"] == 10 and tr["within_family"] and set(tr["per_family"]) == {"evidence_chain"}
