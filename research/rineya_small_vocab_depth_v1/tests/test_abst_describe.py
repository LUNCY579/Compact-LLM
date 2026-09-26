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


def rec(stratum, fam, p, q, t, mp, ml, mt, same=True):
    first = lambda m: {"UNKNOWN": m, "0": 0.0, "1": -9.0, "2": -9.0, "3": -9.0, "im_end": -1.0}
    cell = lambda n, m: {"V4": {"V4": {"nats": n, "per_token": [n], "first": first(m), "first_logmass": 0.0}}}
    return {"stratum": stratum, "family": fam, "tier": None, "reference": "UNKNOWN" if stratum == "unk" else "0",
            "input_equal_V0": {"V4": same, "V2": same},
            "m": {"S20_plain": cell(p, mp), "S20_lc": cell(q, ml), "T": cell(t, mt)}, "gen": {}}


def test_summarise_teacher_anchor_and_readings():
    R = {}
    for k in range(10):     # UNKNOWN rows: plain 4, LC 6 (worse), teacher 8 -> LC closer, no crossing
        R[f"u{k}"] = rec("unk", "evidence_chain", 4.0, 6.0, 8.0, -1.0, -2.0, -3.0)
    for k in range(10):     # known rows
        R[f"k{k}"] = rec("known", "evidence_chain", 2.0, 1.5, 1.8, -4.0, -5.0, -6.0)
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
