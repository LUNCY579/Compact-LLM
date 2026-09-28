from __future__ import annotations

import builtins
from fractions import Fraction
import io
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pytest

from experiments.maxrl_comp002_pretraining_v1 import content_scoring, h3b_null_cell, stats_exact as st
from experiments.maxrl_comp002_pretraining_v1.failure_taxonomy import classify_rs
from experiments.maxrl_composable_skills_v2.generator import view, world

DATA = Path(os.environ.get("COMP_RS_DATA", r"C:\Users\clclc\BusinessProject\Rineya-Dataset\data"))
needs_dev = pytest.mark.skipif(not (DATA / "manifest.json").is_file()
                               or not (DATA / "canonical" / "dev.jsonl").is_file(),
                               reason="Rineya-Dataset DEV data absent")
NC = h3b_null_cell
LAST, RANDOM = NC.PAIR_NAMES


def synthetic_specs(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    specs = []
    while len(specs) < count:
        level = rng.choice((0, 1, 2))
        spec = view(world(rng, level), "RS", rng, level)
        if spec is not None:
            specs.append(spec)
    return specs


def blind(spec: dict) -> list[list[int]]:
    return h3b_null_cell.blind_subsets(spec)


def terminal_aliases(spec: dict) -> dict:
    """index -> the one alias no other alias points at (the chain head the generator requests)."""
    links = spec["lookup"]["links"]
    referenced = {v for v in links.values() if type(v) is str}
    out = {}
    for alias in links:
        if alias in referenced:
            continue
        node = alias
        while type(node) is str:
            node = links[node]
        out[node] = alias
    return out


@pytest.fixture
def open_log(monkeypatch):
    """Every file path opened through builtins.open or io.open (Path.open and read_bytes use io.open)."""
    opened: list[str] = []
    real_builtin, real_io = builtins.open, io.open

    def recorder(real):
        def wrapped(file, *args, **kwargs):
            if not isinstance(file, int):
                opened.append(str(Path(os.fsdecode(os.fspath(file))).resolve()))
            return real(file, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(builtins, "open", recorder(real_builtin))
    monkeypatch.setattr(io, "open", recorder(real_io))
    return opened


# ------------------------------------------------------------ canaries from the amendment

def test_nd_min_canaries():
    assert st.nd_min(0.05) == 5
    assert st.nd_min(0.05 / 12) == 8
    assert st.nd_min(1.0) == 0
    for bad in (0, -0.1, 1.5, True):
        with pytest.raises(ValueError):
            st.nd_min(bad)


def test_cp_upper_zero_losses_closed_form():
    assert abs(st.cp_upper(0, 150) - (1 - 0.05 ** (1 / 150))) < 1e-9
    assert abs(st.cp_lower(150, 150) - 0.05 ** (1 / 150)) < 1e-9
    assert st.cp_upper(7, 7) == 1.0 and st.cp_lower(0, 7) == 0.0
    assert st.cp_upper(0, 0) == 1.0


def test_cp_bounds_solve_their_tail_equations():
    for k, n in ((1, 10), (3, 29), (40, 250), (249, 250)):
        upper, lower = st.cp_upper(k, n), st.cp_lower(k, n)
        assert abs(st.binom_cdf(k, n, upper) - 0.05) < 1e-9
        assert abs(st.binom_sf(k, n, lower) - 0.05) < 1e-9
        assert lower < k / n < upper


def test_cp_bounds_return_the_conservative_end_of_the_bracket():
    """No tolerance: the upper bound's tail is <= 1 - conf, the lower bound's < 1 - conf (TA-07).

    Both come from bisection brackets whose invariants make these exact; the
    bracket is also tight, 2 * CP_TOLERANCE inside the bound is past the root.
    k = 0 (upper) and k = n (lower) use the closed forms and are excluded.
    """
    step = 2 * st.CP_TOLERANCE
    for n in (1, 2, 7, 29, 150, 250, 400, 2000):
        for k in sorted({1, n // 3, n // 2, n - 1} - {0, n}):
            for conf in (0.95, 0.975):
                alpha = 1.0 - conf
                upper, lower = st.cp_upper(k, n, conf), st.cp_lower(k, n, conf)
                assert st.binom_cdf(k, n, upper) <= alpha, (k, n, conf)
                assert st.binom_cdf(k, n, upper - step) > alpha, (k, n, conf)
                assert st.binom_sf(k, n, lower) < alpha, (k, n, conf)
                assert st.binom_sf(k, n, lower + step) >= alpha, (k, n, conf)


def test_cp_bounds_match_scipy_beta():
    stats = pytest.importorskip("scipy.stats")
    for n in (1, 7, 28, 29, 150, 250, 1200):
        for k in sorted({0, 1, n // 7, n // 2, n - 1, n}):
            for conf in (0.95, 0.975):
                upper = 1.0 if k == n else stats.beta.ppf(conf, k + 1, n - k)
                lower = 0.0 if k == 0 else stats.beta.ppf(1 - conf, k, n - k + 1)
                assert abs(st.cp_upper(k, n, conf) - upper) < 1e-9, (k, n, conf)
                assert abs(st.cp_lower(k, n, conf) - lower) < 1e-9, (k, n, conf)


def test_loss_rate_attainability_flips_between_28_and_29_at_ten_percent():
    at28 = st.loss_rate_limb(0, 28, 0.10)
    at29 = st.loss_rate_limb(0, 29, 0.10)
    assert at28["attainable"] is False and at28["state"] == st.UNCERTIFIABLE
    assert at29["attainable"] is True and at29["state"] == st.PASS
    assert st.loss_rate_limb(10, 29, 0.10)["state"] == st.FAIL
    assert st.loss_rate_limb(0, 0, 0.05)["state"] == st.UNCERTIFIABLE
    assert st.loss_rate_limb(1, 150, 0.05)["state"] == st.PASS   # upper ~0.031
    with pytest.raises(ValueError):
        st.loss_rate_limb(3, 2, 0.10)


def test_paired_tie_bound_canary():
    expected = {4: 3.22, 10: 5.55, 20: 7.92, 30: 9.67, 40: 11.11, 60: 13.51}
    assert {nd: round(st.paired_tie_bound(nd), 2) for nd in expected} == expected


def test_percentile_bootstrap_coverage_canary():
    expected = {0.5: 0.39, 1: 0.63, 2: 0.86, 4: 0.91, 8: 0.95}
    results = {c: st.percentile_bootstrap_coverage(c, 250) for c in expected}
    assert {c: round(r["coverage"], 2) for c, r in results.items()} == expected
    assert round(results[0.5]["p_interval_zero_zero"], 2) == 0.61


# ------------------------------------------------------------ exact McNemar and limbs

def test_binom_sf_is_exact_for_half_and_consistent_otherwise():
    for n in range(0, 16):
        for k in range(-1, n + 2):
            exact = sum(Fraction(math.comb(n, j), 2 ** n) for j in range(max(k, 0), n + 1))
            assert st.binom_sf(k, n) == float(exact)
    for p in (0.01, 0.3, 0.77):
        for k in range(0, 13):
            assert abs(st.binom_sf(k, 12, p) + st.binom_cdf(k - 1, 12, p) - 1) < 1e-12
            brute = sum(math.comb(12, j) * p ** j * (1 - p) ** (12 - j) for j in range(k, 13))
            assert abs(st.binom_sf(k, 12, p) - brute) < 1e-12


def test_mcnemar_p_and_min_detectable_net():
    assert st.mcnemar_one_sided_p(0, 0) == 1.0
    assert st.mcnemar_one_sided_p(5, 0) == 1 / 32
    assert st.mcnemar_one_sided_p(0, 5) == 1.0
    assert st.min_detectable_net(0) is None and st.min_detectable_net(4) is None
    assert st.min_detectable_net(5) == 5
    assert st.min_detectable_net(10) == 8
    for nd in range(1, 60):
        net = st.min_detectable_net(nd)
        passing = [2 * b - nd for b in range(nd + 1) if st.mcnemar_one_sided_p(b, nd - b) <= 0.05]
        assert net == (min(passing) if passing else None)


def test_mcnemar_limb_states():
    assert st.mcnemar_limb(4, 0)["state"] == st.UNCERTIFIABLE
    assert st.mcnemar_limb(5, 0)["state"] == st.PASS
    limb = st.mcnemar_limb(3, 2)
    assert limb["state"] == st.FAIL and limb["nd"] == 5 and limb["nd_min"] == 5 and limb["p"] == 0.5
    assert st.mcnemar_limb(7, 0, 0.05 / 12)["state"] == st.UNCERTIFIABLE
    assert st.mcnemar_limb(8, 0, 0.05 / 12)["state"] == st.PASS


def test_limb_state_is_significance_first():
    """Clarification C1: a significant limb is never UNCERTIFIABLE."""
    assert st.limb_state(True, 3, 8) == st.PASS
    assert st.limb_state(True, 3, 8, True) == st.FAIL
    assert st.limb_state(True, 20, 8, True) == st.FAIL
    assert st.limb_state(False, 7, 8) == st.UNCERTIFIABLE
    assert st.limb_state(False, 7, 8, True) == st.UNCERTIFIABLE
    assert st.limb_state(False, 8, 8) == st.FAIL
    for bad in ((1, 3, 8), (True, -1, 8), (True, 3, 8, None)):
        with pytest.raises(ValueError):
            st.limb_state(*bad)


def test_holm_semantics():
    rejected = st.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, 0.05)
    assert rejected == {"a": True, "b": False, "c": False, "d": True}
    # step-down: all rejected when every p clears its own step level
    assert all(st.holm({"a": 0.01, "b": 0.02, "c": 0.04}, 0.05).values())
    # Holm stops at the first failure even if later ones would clear alpha
    assert st.holm({"a": 0.03, "b": 0.03}, 0.05) == {"a": False, "b": False}
    with pytest.raises(ValueError):
        st.holm({}, 0.05)
    with pytest.raises(ValueError):
        st.holm({"a": 1.5}, 0.05)


def test_fixed_sequence_stops_at_first_non_pass_and_never_calls_later_limbs():
    called = []

    def later():
        called.append(True)
        return st.PASS

    results = st.fixed_sequence([("H3a", {"state": st.PASS, "p": 0.01}),
                                 ("H3b", lambda: st.mcnemar_limb(3, 1)),
                                 ("H3d", later), ("H3c", st.PASS)])
    assert [r["state"] for r in results.values()] == [st.PASS, st.UNCERTIFIABLE, st.NOT_REACHED,
                                                     st.NOT_REACHED]
    assert called == [] and results["H3a"]["p"] == 0.01
    assert not st.h3_supported(results)
    full = st.fixed_sequence([("H3a", st.PASS), ("H3b", st.PASS), ("H3d", st.PASS), ("H3c", st.FAIL)])
    assert st.h3_supported(full) and full["H3c"]["state"] == st.FAIL
    with pytest.raises(ValueError):
        st.fixed_sequence([("x", st.PASS), ("x", st.PASS)])


def test_h3c_state_caps_a_pass_under_either_section5_label():
    """Section 5: under PLACEBO_UNDERSIZED or PLACEBO_UNDERENGAGED H3c can be read only as FAIL (F11)."""
    passed = {"state": st.PASS, "p": 0.01}
    for labels in (["PLACEBO_UNDERSIZED(50.00%)"], ("PLACEBO_UNDERSIZED",), {"PLACEBO_UNDERENGAGED"},
                   ["PLACEBO_UNDERENGAGED", "PLACEBO_UNDERSIZED(3.25%)"]):
        capped = st.h3c_state(passed, labels)
        assert capped["state"] == st.NOT_CLAIMED and capped["uncapped_state"] == st.PASS
        assert capped["capped"] is True and capped["p"] == 0.01
        assert capped["section5_labels"] == sorted(set(labels))
    assert st.h3c_state(passed, [])["state"] == st.PASS and st.h3c_state(passed, [])["capped"] is False
    for state in (st.FAIL, st.UNCERTIFIABLE, st.NOT_REACHED):
        assert st.h3c_state(state, ["PLACEBO_UNDERENGAGED", "PLACEBO_UNDERSIZED(10.00%)"])["state"] == state
    # through the fixed sequence: H3c's own PASS under a label is not claimable
    sequence = st.fixed_sequence([("H3a", st.PASS), ("H3b", st.PASS), ("H3d", st.PASS), ("H3c", st.PASS)])
    assert st.h3c_state(sequence["H3c"], ["PLACEBO_UNDERENGAGED"])["state"] == st.NOT_CLAIMED
    for bad in (["MAGNITUDE_UNCONTROLLED"], ["PLACEBO_UNDERSIZED(50%"], ["placebo_underengaged"], [1],
                "PLACEBO_UNDERENGAGED", None):
        with pytest.raises(ValueError):
            st.h3c_state(passed, bad)
    with pytest.raises(ValueError):
        st.h3c_state({"p": 0.01}, [])


def test_h4_limb_states_and_family_holm():
    assert st.h4_limb(20, 0, 12, None)["state"] == st.DESCRIPTIVE
    assert st.h4_limb(20, 0, 12, 10)["state"] == st.PASS
    below = st.h4_limb(20, 0, 12, 25)
    assert below["state"] == st.FAIL and below["below_floor"] is True
    assert st.h4_limb(7, 0, 12, 0)["state"] == st.UNCERTIFIABLE   # nd_min(0.05/12) = 8
    assert st.h4_limb(8, 0, 12, 0)["state"] == st.PASS            # 1/256 <= 0.05/12
    assert st.h4_limb(6, 4, 12, 0)["state"] == st.FAIL
    limbs = {f"strong{i}": (20, 0, 5) for i in range(11)}
    limbs["weak"] = (9, 1, 5)                                     # p = 11/1024
    family = st.h4_family(limbs)
    assert len(family) == 12 and all(r["state"] == st.PASS for r in family.values())
    assert family["weak"]["holm_basis"] == "family"
    assert st.h4_limb(9, 1, 12, 5)["state"] == st.FAIL            # first-step bound only
    limbs["weak"] = (9, 1, None)
    assert st.h4_family(limbs)["weak"]["state"] == st.DESCRIPTIVE


def test_h4_limb_passes_at_net_equal_to_the_floor():
    """Section 3 claims a limb iff |net| >= F_E; nets and F_E are integers, so equality occurs (TA-07)."""
    at = st.h4_limb(20, 0, 12, 20)
    assert at["state"] == st.PASS and at["below_floor"] is False
    assert st.h4_limb(20, 0, 12, 21)["state"] == st.FAIL
    limbs = {f"other{i}": (0, 0, None) for i in range(11)}
    limbs["edge"] = (20, 0, 20)
    assert st.h4_family(limbs)["edge"]["state"] == st.PASS
    limbs["edge"] = (20, 0, 21)
    assert st.h4_family(limbs)["edge"]["below_floor"] is True


def test_h4_holm_significant_limbs_are_never_uncertifiable():
    """C1 in the family: Holm's later steps can reject a limb with nd < 8 (F4, NUM-2)."""
    limbs = {f"strong{i}": (40, 0, 5) for i in range(11)}
    limbs["x"] = (6, 0, 8)                                        # p = 1/64: rejected at Holm's last step
    x = st.h4_family(limbs)["x"]
    assert x["holm_significant"] is True and x["nd"] == 6 < x["nd_min"] == 8
    assert x["state"] == st.FAIL and x["below_floor"] is True     # below its floor, not UNCERTIFIABLE
    limbs["x"] = (6, 0, 5)
    assert st.h4_family(limbs)["x"]["state"] == st.PASS           # significant, net 6 >= 5
    limbs["x"] = (7, 0, 5)
    assert st.h4_family(limbs)["x"]["state"] == st.PASS
    # not significant with nd < nd_min(0.05/12) is UNCERTIFIABLE; with nd >= 8 it is FAIL
    weak = st.h4_family({f"w{i}": (7, 0, 0) for i in range(12)})
    assert all(r["holm_significant"] is False and r["state"] == st.UNCERTIFIABLE for r in weak.values())
    mixed = st.h4_family({**{f"w{i}": (7, 0, 0) for i in range(11)}, "n": (6, 4, 0)})
    assert mixed["n"]["state"] == st.FAIL


def test_h4_family_size_is_fixed_and_gate_failed_limbs_keep_m():
    """C2: m = 12 always; a gate-failed comparator's limbs stay in as GATE_FAILED (F5)."""
    assert st.H4_FAMILY_SIZE == 12
    nulls = {f"null{i}": (5, 5, 0) for i in range(7)}
    for size in (8, 11, 13):
        limbs = {f"l{i}": (10, 1, 0) for i in range(size)}
        with pytest.raises(ValueError, match="m = 12"):
            st.h4_family(limbs)
    with pytest.raises(ValueError, match="m = 12"):
        st.h4_limb(10, 1, 8, 0)
    gated = {**nulls, "target": (10, 1, 0), **{f"MT:{e}": st.GATE_FAILED for e in ("rs", "r", "s", "legacy")}}
    family = st.h4_family(gated)
    assert len(family) == 12
    assert family["target"]["p"] == 0.005859375 and family["target"]["holm_significant"] is False
    assert family["target"]["state"] == st.FAIL                   # 0.00586 > 0.05/12
    for name in ("MT:rs", "MT:r", "MT:s", "MT:legacy"):
        assert family[name]["state"] == st.GATE_FAILED and family[name]["holm_significant"] is False
    # the same target would have been rejected had the gate-failed limbs been dropped (m = 8)
    dropped = {name: st.mcnemar_one_sided_p(b, w) for name, (b, w, _) in {**nulls, "target": (10, 1, 0)}.items()}
    assert st.holm(dropped)["target"] is True
    # a GATE_FAILED limb is never rejected, even when every tested limb is
    strong = st.h4_family({**{f"s{i}": (40, 0, 0) for i in range(11)}, "gated": st.GATE_FAILED})
    assert strong["gated"]["state"] == st.GATE_FAILED and strong["gated"]["holm_significant"] is False
    assert all(strong[f"s{i}"]["state"] == st.PASS for i in range(11))
    for bad in ([10, 1, 0], (10, 1), "PASS"):
        with pytest.raises(ValueError):
            st.h4_family({**{f"s{i}": (40, 0, 0) for i in range(11)}, "bad": bad})


def test_cross_draw_floor_is_zero_when_no_net_is_significant_at_the_replicate_nd():
    """C3: F_E = 0 when min_detectable_net(replicate nd) is None, with its basis recorded."""
    assert st.cross_draw_floor(None) == {"floor_net": None, "floor_basis": "no_replicate", "replicate_nd": None}
    for nd in range(0, 5):
        assert st.cross_draw_floor(nd) == {"floor_net": 0, "floor_basis": "replicate_nd_below_nd_min",
                                           "replicate_nd": nd}
    assert st.cross_draw_floor(5)["floor_net"] == 5 and st.cross_draw_floor(10)["floor_net"] == 8
    assert st.cross_draw_floor(10)["floor_basis"] == "min_detectable_net"
    for nd in range(5, 60):
        assert st.cross_draw_floor(nd)["floor_net"] == st.min_detectable_net(nd)
    limbs = {f"strong{i}": (40, 0, 0) for i in range(11)}
    limbs["e"] = (5, 0, st.cross_draw_floor(3))
    result = st.h4_family(limbs)["e"]
    assert result["state"] == st.PASS and result["floor_net"] == 0
    assert result["floor_basis"] == "replicate_nd_below_nd_min"
    limbs["e"] = (5, 0, st.cross_draw_floor(None))
    assert st.h4_family(limbs)["e"]["state"] == st.DESCRIPTIVE
    limbs["e"] = (5, 0, {"floor_net": 0})
    with pytest.raises(ValueError):
        st.h4_family(limbs)
    with pytest.raises(ValueError):
        st.cross_draw_floor(-1)


# ------------------------------------------------------------ pi_blind, e and the H3b candidate

def test_int_list_is_the_one_content_scoring_definition():
    """C4: stats_exact uses content_scoring.is_int_list; [] is not an int list (F10)."""
    assert st.is_int_list is content_scoring.is_int_list
    assert not hasattr(st, "_int_list")
    triples = [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(ValueError):
        st.pi_blind([], triples)
    spec = synthetic_specs(1, 11)[0]
    assert st.e_value([], spec) == 0.0
    (matrix,) = st.redraw_e([[[]]], [spec], 50, np.random.default_rng(1))
    assert not matrix.any()


def test_pi_blind_hand_cases():
    triples = [[0, 1, 2], [3, 4, 5]]
    assert st.pi_blind([0], triples) == pytest.approx(2 / 3)
    assert st.pi_blind([0, 1], triples) == pytest.approx(1 / 3)
    assert st.pi_blind([0, 1, 2], triples) == 0.0
    assert st.pi_blind([0, 3], triples) == pytest.approx(4 / 9)
    assert st.pi_blind([2, 0, 4, 5], triples) == pytest.approx(1 / 9)
    assert st.pi_blind([0, 0, 0], triples) == pytest.approx(2 / 3)   # distinct indices
    assert st.pi_blind([0, 9], triples) == 0.0                       # outside every triple
    for bad in ([0, True], [0.0], "UNKNOWN", None, []):
        with pytest.raises(ValueError):
            st.pi_blind(bad, triples)
    with pytest.raises(ValueError):
        st.pi_blind([0], [[0, 1], [2, 3, 4]])


def test_triples_resolved_set_and_design_on_synthetic_specs():
    for spec in synthetic_specs(12, 3):
        triples, resolved = st.design(spec)
        assert sorted(i for t in triples for i in t) == list(range(len(spec["initial"])))
        assert len(resolved) == 2 * len(triples)
        assert resolved == st.resolved_set(spec)
    with pytest.raises(ValueError, match="triple"):
        st.triples_from_spec({"initial": [1, 1, 1, 2, 2]})
    with pytest.raises(ValueError):
        st.resolved_set({"initial": [1, 1, 1], "selection": {"k": 1, "target": 1}})


def test_design_refuses_a_broken_two_of_three_structure(monkeypatch):
    """The at-freeze structure check: 1 or 3 resolved members, or a resolved index outside (TA-04)."""
    spec = synthetic_specs(1, 17)[0]
    triples, resolved = st.design(spec)
    heads = terminal_aliases(spec)
    excluded = next(i for i in triples[0] if i not in resolved)
    kept = next(i for i in triples[0] if i in resolved)
    three = json.loads(json.dumps(spec))
    three["lookup"]["requested"].append(heads[excluded])
    one = json.loads(json.dumps(spec))
    one["lookup"]["requested"].remove(heads[kept])
    candidate = sorted(resolved)[:1]
    for broken in (three, one):
        assert st.triples_from_spec(broken) == triples
        with pytest.raises(ValueError, match="exactly two"):
            st.design(broken)
        with pytest.raises(ValueError, match="exactly two"):
            st.e_value(candidate, broken)
        with pytest.raises(ValueError, match="exactly two"):
            st.redraw_e([[candidate]], [broken], 5, np.random.default_rng(0))
        with pytest.raises(ValueError, match="exactly two"):
            st.design_randomization_test([candidate], None, [broken], 5, 0)
    outside = resolved | {len(spec["initial"]) + 3}
    monkeypatch.setattr(st, "resolved_set", lambda s: frozenset(outside))
    with pytest.raises(ValueError, match="outside"):
        st.design(spec)
    with pytest.raises(ValueError, match="outside"):
        st.e_value(candidate, spec)
    with pytest.raises(ValueError, match="outside"):
        st.redraw_e([[candidate]], [spec], 5, np.random.default_rng(0))


def test_e_value_cases():
    spec = synthetic_specs(1, 11)[0]
    triples, resolved = st.design(spec)
    inside = sorted(resolved)[:2]
    outside = [next(i for i in range(len(spec["initial"])) if i not in resolved)]
    for none_like in (None, "UNKNOWN", {"a": 1}, [0, "1"], [True], 3, []):
        assert st.e_value(none_like, spec) == 0.0
    assert st.e_value(inside, spec) == pytest.approx(1 - st.pi_blind(inside, triples))
    assert st.e_value(outside, spec) == pytest.approx(-2 / 3)
    assert st.e_value([len(spec["initial"]) + 5], spec) == 0.0


def test_e_value_on_the_full_resolved_set():
    """The FULL_RESOLVED_SET case (R's contract leaking) is inside: e = 1 - (1/3)**T (TA-06)."""
    for spec in synthetic_specs(8, 23):
        triples, resolved = st.design(spec)
        full = sorted(resolved)
        assert st.pi_blind(full, triples) == float(Fraction(1, 3) ** len(triples))
        assert st.e_value(full, spec) == 1.0 - st.pi_blind(full, triples) > 0.0
        assert st.e_value(full[::-1], spec) == st.e_value(full, spec)


class _ActualDesign:
    """A stand-in generator whose every re-draw is the actual excluded member of each triple."""

    def __init__(self, specs):
        self.slots = []
        for spec in specs:
            triples, resolved = st.design(spec)
            self.slots.append(np.array([next(s for s, i in enumerate(t) if i not in resolved) for t in triples]))
        self.calls = iter(self.slots)

    def integers(self, low, high, size):
        assert (low, high) == (0, 3)
        slots = next(self.calls)
        assert size[1] == len(slots)
        return np.tile(slots, (size[0], 1))


def test_redraw_at_the_actual_design_reproduces_e_value():
    """redraw_e's inside-ness and e_value agree row by row, the full resolved set included (TA-06)."""
    specs = synthetic_specs(10, 29)
    lists = []
    for spec in specs:
        triples, resolved = st.design(spec)
        lists.append([sorted(resolved), blind(spec)[0], blind(spec)[-1], [triples[0][0]], list(triples[0]),
                      sorted(resolved)[:3], [len(spec["initial"]) + 1], "UNKNOWN", []])
    candidate_sets = [[row[j] for row in lists] for j in range(len(lists[0]))]
    matrices = st.redraw_e(candidate_sets, specs, 3, _ActualDesign(specs))
    for matrix, candidates in zip(matrices, candidate_sets):
        expected = [st.e_value(c, s) for c, s in zip(candidates, specs)]
        assert np.allclose(matrix, np.array(expected)[None, :], rtol=0, atol=1e-15)


def test_e_is_exactly_centred_over_the_full_design():
    """Enumerate every excluded-member pattern: the pass rate of any list equals pi_blind exactly."""
    for spec in synthetic_specs(6, 5):
        triples, resolved = st.design(spec)
        if len(triples) > 4:
            continue
        candidates = blind(spec)[:6] + [[0], [0, 1, 2], triples[0][:2] + triples[1][:1], sorted(resolved)]
        for candidate in candidates:
            chosen, passes = set(candidate), 0
            patterns = list(product(range(3), repeat=len(triples)))
            for pattern in patterns:
                world_resolved = {i for t, x in zip(triples, pattern) for s, i in enumerate(t) if s != x}
                passes += chosen <= world_resolved
            assert Fraction(passes, len(patterns)) == Fraction(st.pi_blind(candidate, triples)).limit_denominator(3 ** 5)


def test_e_has_mean_zero_under_design_randomization():
    specs = synthetic_specs(40, 21)
    candidates = [blind(s)[0] for s in specs]
    (matrix,) = st.redraw_e([candidates], specs, 5000, np.random.default_rng(4))
    per_draw = matrix.mean(axis=1)
    assert abs(per_draw.mean()) < 4 * per_draw.std(ddof=1) / math.sqrt(len(per_draw))
    assert np.allclose(matrix.mean(axis=0), 0.0, atol=4 * 0.5 / math.sqrt(5000))


def test_h3b_candidate_is_the_content_contract_candidate():
    """C5: parse_candidate's candidate; multiple, UNKNOWN, unparseable and non-int-list give None (F11)."""
    spec = synthetic_specs(1, 41)[0]
    triples, resolved = st.design(spec)
    inside = sorted(resolved)[:2]
    obj = json.dumps({"answer": inside})
    accepted = {obj: inside, "```json\n" + obj + "\n```": inside, obj + ".": inside,
                json.dumps(inside): inside, json.dumps({"answer": inside, "reason": "ok"}): inside,
                "```\n" + json.dumps(inside) + ".\n```": inside}
    refused = ['{"answer":[1],"alt":[2]}', '{"answer":[1],"why":{"deep":"UNKNOWN"}}', "UNKNOWN",
               '{"answer":"UNKNOWN"}', '{"answer":[]}', "[]", '{"answer":[true]}', '{"answer":[1.0]}',
               '{"answer":"[1, 2]"}', "not json", "", "   ", '{"answer":[1],"answer":[2]}',
               '{"answer":[NaN]}', "[" + "1," * 3000 + "1]", '{"result":[1]}', '[1] [2]']
    for response, expected in accepted.items():
        assert st.h3b_candidate(response) == expected, response
        assert classify_rs(spec, response) not in ("UNPARSEABLE", "UNKNOWN_ANSWER", "NOT_INT_LIST")
        # a wrapper cannot hide lookup use: e is the unwrapped list's e
        assert st.e_value(st.h3b_candidate(response), spec) == 1.0 - st.pi_blind(inside, triples)
    for response in refused:
        assert st.h3b_candidate(response) is None, response
        assert classify_rs(spec, response) in ("UNPARSEABLE", "UNKNOWN_ANSWER", "NOT_INT_LIST"), response
        assert st.e_value(st.h3b_candidate(response), spec) == 0.0
    for bad in (None, b"[1]", ["[1]"]):
        with pytest.raises(TypeError):
            st.h3b_candidate(bad)


# ------------------------------------------------------------ Monte Carlo tests

def test_signflip_matches_exact_enumeration_and_handles_ties():
    d = np.array([0.9, -0.2, 0.4, 1 / 3, -1 / 3, 0.0, 0.7, -0.5, 0.1, 2 / 3])
    signs = np.array(list(product((-1.0, 1.0), repeat=len(d))))
    exact = float(((signs @ d) >= d.sum() - 1e-12).mean())
    p = st.signflip_test(d, 200_000, seed=1)
    assert abs(p - exact) < 0.005
    ties = [1 / 3] * 6
    assert abs(st.signflip_test(ties, 100_000, seed=2) - 1 / 64) < 0.002
    assert st.signflip_test([0.0] * 5, 1000, seed=3) == 1.0
    assert st.signflip_test([0.0] * 5, 999, seed=3) == 1.0
    # +1 correction: an unreachable observed sum (P = 2**-30 per draw) gives exactly 1/(n+1)
    assert st.signflip_test([1.0] * 30, 999, seed=4) == 1 / 1000
    with pytest.raises(ValueError):
        st.signflip_test(d, 100, seed=None)


def test_signflip_known_answer_pins_the_registered_stream(monkeypatch):
    """The p-value is a function of (d, n_flips, seed, SIGNFLIP_CHUNK, numpy's generator) (TA-05).

    20,000 flips span two chunks. A power-of-two chunk change leaves numpy's
    uint8 stream unchanged, so the chunk constant is pinned directly; a chunk
    that splits the byte stream off a 4-byte boundary changes the p-value.
    """
    d = [0.9, -0.2, 0.4, 1 / 3, -1 / 3, 0.0, 0.7, -0.5, 0.1, 2 / 3, -0.4, 0.25]
    assert st.SIGNFLIP_CHUNK == 1 << 14
    assert st.signflip_test(d, 20_000, seed=7) == 2614 / 20_001
    assert st.signflip_test(d, 20_000, seed=8) == 2581 / 20_001
    assert st.signflip_test_columns(np.column_stack([d, d]), 20_000, seed=7).tolist() == [2614 / 20_001] * 2
    monkeypatch.setattr(st, "SIGNFLIP_CHUNK", 999)
    assert st.signflip_test(d, 20_000, seed=7) == 2613 / 20_001


def test_signflip_columns_reproduce_single_tests():
    matrix = np.random.default_rng(8).normal(size=(30, 5)) + 0.2
    batched = st.signflip_test_columns(matrix, 3000, seed=9)
    single = [st.signflip_test(matrix[:, j], 3000, seed=9) for j in range(5)]
    assert batched.tolist() == single


def test_signflip_is_uniformish_under_a_symmetric_null():
    rng = np.random.default_rng(12)
    continuous = [st.signflip_test(rng.normal(size=40), 499, seed=100 + i) for i in range(400)]
    rate = np.mean(np.array(continuous) <= 0.05)
    assert 0.02 <= rate <= 0.09
    assert 0.42 <= np.mean(np.array(continuous) <= 0.5) <= 0.58
    atoms = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    discrete = [st.signflip_test(rng.choice(atoms, 40) * rng.choice((-1, 1), 40), 499, seed=900 + i)
                for i in range(300)]
    assert np.mean(np.array(discrete) <= 0.05) <= 0.09


def test_design_randomization_test_detects_lookup_use_and_is_one_on_identity():
    specs = synthetic_specs(60, 31)
    uses_lookup, blind_first = [], []
    for spec in specs:
        _, resolved = st.design(spec)
        subsets = blind(spec)
        uses_lookup.append(next(c for c in subsets if set(c) <= resolved))
        blind_first.append(subsets[0])
    assert st.design_randomization_test(uses_lookup, blind_first, specs, 2000, 5) < 0.01
    assert st.design_randomization_test(uses_lookup, None, specs, 2000, 5) < 0.01
    # +1 correction: every lookup-user row is inside the actual set; no re-draw reaches that sum
    assert st.design_randomization_test(uses_lookup, blind_first, specs, 99, 5) == 1 / 100
    assert st.design_randomization_test(blind_first, blind_first, specs, 500, 5) == 1.0
    null_p = st.design_randomization_test(blind_first, None, specs, 2000, 6)
    assert 0.0 < null_p <= 1.0
    with pytest.raises(ValueError):
        st.design_randomization_test(uses_lookup[:-1], None, specs, 10, 5)


# ------------------------------------------------------------ the null cell's switch rule (C7)

def test_switch_rule_decides_on_the_cp_upper_bound_of_any_pair():
    """TA-01: SWITCH at 25/400, NO SWITCH at 87/2000; one exceeding pair alone switches."""
    assert (NC.LEVEL, NC.SWITCH_BOUND) == (0.05, 0.075)
    assert (NC.REGISTERED_MIN_REDRAWS, NC.REGISTERED_FLIPS) == (2000, 1_000_000)
    flips = NC.REGISTERED_FLIPS
    at400 = NC.switch_rule({LAST: 25, RANDOM: 0}, 400, flips)
    pair = at400["pairs"][LAST]
    assert pair["false_rejection_rate"] == 0.0625 < NC.SWITCH_BOUND < pair["upper95"] == st.cp_upper(25, 400)
    assert pair["exceeds_switch_bound"] is True and at400["rule"]["switch_indicated"] is True
    assert at400["rule"]["outcome"] == NC.NOT_A_REGISTERED_RUN      # 400 < 2000: not a registered run
    dev = NC.switch_rule({LAST: 87, RANDOM: 93}, 2000, flips)
    assert dev["rule"]["switch_indicated"] is False and dev["rule"]["rule_compliant"] is True
    assert dev["rule"]["outcome"] == NC.SIGNFLIP_REFERENCE
    assert round(dev["pairs"][LAST]["upper95"], 4) == 0.0518 and round(dev["pairs"][RANDOM]["upper95"], 4) == 0.0550
    # the bound, not the rate: 131/2000 = 0.0655 switches (upper 0.0753); 130/2000 does not (0.0748)
    assert NC.switch_rule({LAST: 130, RANDOM: 130}, 2000, flips)["rule"]["outcome"] == NC.SIGNFLIP_REFERENCE
    for counts in ({LAST: 131, RANDOM: 0}, {LAST: 0, RANDOM: 131}):
        decided = NC.switch_rule(counts, 2000, flips)
        assert decided["rule"]["outcome"] == NC.DESIGN_RANDOMIZATION_REFERENCE
        assert sum(p["exceeds_switch_bound"] for p in decided["pairs"].values()) == 1
        assert all(p["false_rejection_rate"] < NC.SWITCH_BOUND for p in decided["pairs"].values())
    for name in (LAST, RANDOM):
        assert dev["pairs"][name]["mc_se"] == math.sqrt(dev["pairs"][name]["false_rejection_rate"]
                                                        * (1 - dev["pairs"][name]["false_rejection_rate"]) / 2000)


def test_switch_rule_emits_an_outcome_only_at_the_registered_counts():
    """C7: >= 2,000 re-draws and exactly 10^6 flips; anything else is NOT_A_REGISTERED_RUN (F3, NUM-1)."""
    for redraws, flips, compliant in ((2000, 1_000_000, True), (5000, 1_000_000, True),
                                      (1999, 1_000_000, False), (2000, 999_999, False),
                                      (2000, 1_000_001, False), (20, 500, False)):
        rule = NC.switch_rule({LAST: 0, RANDOM: 0}, redraws, flips)["rule"]
        assert rule["rule_compliant"] is compliant and NC.rule_compliant(redraws, flips) is compliant
        expected = NC.SIGNFLIP_REFERENCE if compliant else NC.NOT_A_REGISTERED_RUN
        if redraws == 20:
            assert rule["switch_indicated"] is True                  # cp_upper(0, 20) = 0.139
        assert rule["outcome"] == expected
    for bad in ({LAST: 0}, {LAST: 0, RANDOM: 0, "extra": 0}, {LAST: 2001, RANDOM: 0}, {LAST: -1, RANDOM: 0},
                {LAST: True, RANDOM: 0}, {LAST: 1.0, RANDOM: 0}):
        with pytest.raises(ValueError):
            NC.switch_rule(bad, 2000, 1_000_000)
    with pytest.raises(ValueError):
        NC.switch_rule({LAST: 0, RANDOM: 0}, 0, 1_000_000)


def test_count_rejections_is_p_at_most_the_nominal_level():
    assert NC.count_rejections([0.05, 0.049, np.nextafter(0.05, 1.0), 1.0, 0.025, 0.0251]) == 4


def test_policies_are_first_last_and_a_held_random_choice():
    """TA-01: lex_last and fixed_random differ from lex_first on some rows."""
    specs = synthetic_specs(30, 43)
    chosen = NC.policies(specs, np.random.default_rng(5))
    again = NC.policies(specs, np.random.default_rng(5))
    assert chosen == again
    for i, spec in enumerate(specs):
        subsets = blind(spec)
        assert chosen["lex_first"][i] == subsets[0] and chosen["lex_last"][i] == subsets[-1]
        assert chosen["fixed_random"][i] in subsets
    assert any(a != b for a, b in zip(chosen["lex_first"], chosen["lex_last"]))
    assert any(a != b for a, b in zip(chosen["lex_first"], chosen["fixed_random"]))


def test_null_cell_counts_rejections_and_uses_the_given_flip_seed(monkeypatch):
    """The null cell feeds (p <= 0.05) counts into switch_rule and tests with --flip-seed (NUM-3)."""
    specs = synthetic_specs(8, 47)
    calls = []
    plan = iter([np.array([0.05] * 40 + [0.01] * 47 + [np.nextafter(0.05, 1.0)] * 13 + [0.9] * 1900),
                 np.array([0.9] * 2000)])

    def fake(differences, n_flips, *, seed):
        calls.append((differences.shape, n_flips, seed))
        return next(plan)

    monkeypatch.setattr(NC, "signflip_test_columns", fake)
    report = NC.null_cell(specs, 2000, NC.REGISTERED_FLIPS, 3, flip_seed=987654321)
    assert calls == [((len(specs), 2000), NC.REGISTERED_FLIPS, 987654321)] * 2
    assert report["seeds"]["flip_seed"] == 987654321 and "registered" in report["seeds"]["flip_seed_source"]
    assert report["pairs"][LAST]["rejections"] == 87 and report["pairs"][RANDOM]["rejections"] == 0
    assert report["rule"]["rule_compliant"] is True and report["rule"]["outcome"] == NC.SIGNFLIP_REFERENCE
    assert report["pairs"][LAST]["rows_differing"] > 0 and report["pairs"][RANDOM]["rows_differing"] > 0
    plan = iter([np.array([0.9] * 2000), np.array([0.04] * 131 + [0.5] * 1869)])
    switched = NC.null_cell(specs, 2000, NC.REGISTERED_FLIPS, 3)
    assert switched["rule"]["outcome"] == NC.DESIGN_RANDOMIZATION_REFERENCE
    assert switched["seeds"]["flip_seed"] == NC.derive_seeds(3)["flip_seed"] != 987654321
    assert switched["seeds"]["flip_seed_source"].startswith("derived")
    with pytest.raises(ValueError):
        NC.derive_seeds(3, -1)


def test_null_cell_below_the_registered_counts_emits_no_rule_outcome():
    specs = synthetic_specs(12, 53)
    report = NC.null_cell(specs, 20, 200, 1)
    assert report["rule"]["rule_compliant"] is False
    assert report["rule"]["outcome"] == NC.NOT_A_REGISTERED_RUN
    assert report["rule"]["switch_indicated"] is True          # 0 or more of 20 always indicates a switch


# ------------------------------------------------------------ data access (sealed-role gates, single read)

def write_synthetic_bank(root: Path, roles, specs: list[dict], monkeypatch, *, mutate=None) -> dict:
    """A synthetic dataset (never under a 'data' directory): manifest plus canonical/<role>.jsonl."""
    files = {}
    (root / "canonical").mkdir(parents=True)
    for role in roles:
        rows = [{"id": f"{role}-rs-{i:03d}", "parent_id": f"{role}-p-{i:03d}", "role": role, "tag": "core",
                 "skills": ["R", "S"], "kind": "composition", "verifier": {"kind": "composition", "spec": spec}}
                for i, spec in reversed(list(enumerate(specs)))]
        rows.append({**rows[0], "id": f"{role}-b-000", "tag": "boundary"})
        rows.append({**rows[0], "id": f"{role}-s-000", "skills": ["S"]})
        lines = [json.dumps(row, sort_keys=True) for row in rows]
        if mutate is not None:
            lines[0] = mutate(lines[0])
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        (root / "canonical" / f"{role}.jsonl").write_bytes(raw)
        files[f"canonical/{role}.jsonl"] = {"sha256": hashlib.sha256(raw).hexdigest()}
    manifest = json.dumps({"files": files, "counts": {r: len(specs) + 2 for r in roles}}).encode("utf-8")
    (root / "manifest.json").write_bytes(manifest)
    monkeypatch.setattr(NC, "MANIFEST_SHA", hashlib.sha256(manifest).hexdigest())
    monkeypatch.setattr(NC, "EXPECTED_ROWS", {"dev": len(specs), "test": len(specs)})
    return files


def test_load_core_rs_refuses_role_test_before_touching_any_path(tmp_path, monkeypatch, open_log):
    """DS-1: the library gate, not only the CLI. The synthetic bank holds a (synthetic) test.jsonl."""
    root = tmp_path / "bank"
    write_synthetic_bank(root, ("dev", "test"), synthetic_specs(3, 59), monkeypatch)
    del open_log[:]
    for flag in ({}, {"at_freeze": False}, {"at_freeze": 1}, {"at_freeze": "yes"}):
        with pytest.raises(ValueError, match="at_freeze"):
            NC.load_core_rs(root, "test", **flag)
    with pytest.raises(ValueError, match="at_freeze"):
        NC.load_core_rs(tmp_path / "empty", "test")
    assert open_log == []
    with pytest.raises(TypeError):
        NC.load_core_rs(root, "test", True)                    # keyword-only
    for role in ("challenge", "guard_test", "calibration", None):
        with pytest.raises(ValueError):
            NC.load_core_rs(root, role, at_freeze=True)
    assert open_log == []
    rows, _ = NC.load_core_rs(root, "test", at_freeze=True)      # the flag is the gate (synthetic file)
    assert [row["id"] for row in rows] == ["test-rs-000", "test-rs-001", "test-rs-002"]


def test_load_core_rs_hashes_and_parses_one_strict_read_per_file(tmp_path, monkeypatch, open_log):
    """DS-7: each file is read once; the hashed bytes are the parsed bytes, parsed by strict_json."""
    specs = synthetic_specs(4, 61)
    root = tmp_path / "bank"
    files = write_synthetic_bank(root, ("dev",), specs, monkeypatch)
    del open_log[:]
    rows, file_sha = NC.load_core_rs(root, "dev")
    assert open_log == [str((root / "manifest.json").resolve()), str((root / "canonical" / "dev.jsonl").resolve())]
    assert file_sha == files["canonical/dev.jsonl"]["sha256"]
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows) and len(rows) == 4
    assert [row["spec"] for row in rows] == specs
    for mutate, message in ((lambda line: line[:-1] + ', "tag": "core"}', "duplicate key"),
                            (lambda line: line[:-1] + ', "noise": NaN}', "nonfinite")):
        other = tmp_path / ("bank_" + message.split()[0])
        write_synthetic_bank(other, ("dev",), specs, monkeypatch, mutate=mutate)
        with pytest.raises(ValueError, match=message):
            NC.load_core_rs(other, "dev")


def test_null_cell_role_test_refuses_without_freeze_before_touching_data(tmp_path):
    # tmp_path holds no dataset: even a broken guard could not reach a TEST file.
    with pytest.raises(SystemExit, match="at-freeze"):
        h3b_null_cell.main(["--data", str(tmp_path), "--role", "test", "--redraws", "1",
                            "--flips", "1", "--seed", "0", "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("redraws,flips", [("20", "10"), ("400", "1000000"), ("1999", "1000000"),
                                           ("2000", "999999"), ("2000", "1000001")])
def test_null_cell_role_test_refuses_unregistered_counts_before_touching_data(tmp_path, open_log, redraws, flips):
    """TA-02 / F3 / NUM-1: at freeze the registered counts are required, checked before any file access."""
    out = tmp_path / "x.json"
    with pytest.raises(SystemExit, match="registered counts"):
        NC.main(["--data", str(tmp_path), "--role", "test", "--at-freeze", "--redraws", redraws,
                 "--flips", flips, "--seed", "1", "--flip-seed", "5", "--out", str(out)])
    assert open_log == [] and not out.exists()


def test_null_cell_role_test_requires_the_registered_flip_seed(tmp_path, open_log):
    out = tmp_path / "x.json"
    base = ["--data", str(tmp_path), "--role", "test", "--at-freeze", "--redraws", "2000",
            "--flips", "1000000", "--seed", "1", "--out", str(out)]
    with pytest.raises(SystemExit, match="flip-seed"):
        NC.main(base)
    with pytest.raises(SystemExit, match="nonnegative"):
        NC.main(base + ["--flip-seed", "-3"])
    assert open_log == []
    # every flag present: the run gets past the gates and fails on the empty directory's missing manifest
    with pytest.raises(FileNotFoundError):
        NC.main(base + ["--flip-seed", "5"])
    assert open_log == [str((tmp_path / "manifest.json").resolve())] and not out.exists()


def test_code_map_is_hashed_at_import_and_rechecked_before_writing(tmp_path, monkeypatch):
    """DS-6: the null cell binds its whole dependency set, hashed at import, re-checked at write."""
    assert set(NC.CODE_FILES) == {"h3b_null_cell.py", "stats_exact.py", "content_scoring.py", "preflight.py",
                                  "maxrl_composable_skills_v2/verifier.py", "maxrl_composable_skills_v2/corpus.py",
                                  "maxrl_composable_skills_v2/contracts.py"}
    for name, path in NC.CODE_FILES.items():
        assert path.name == name.split("/")[-1] and path.is_file()
        assert NC._IMPORT_CODE[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert NC.code_sha256() == NC._IMPORT_CODE
    out = tmp_path / "r.json"
    drifted = {**NC._IMPORT_CODE, "content_scoring.py": "0" * 64}
    monkeypatch.setattr(NC, "_IMPORT_CODE", drifted)
    with pytest.raises(SystemExit, match="changed since import"):
        NC.write_receipt({"a": 1}, out)
    assert not out.exists()
    monkeypatch.undo()
    NC.write_receipt({"a": 1}, out)
    assert out.read_bytes() == b'{\n  "a": 1\n}\n'
    with pytest.raises(FileExistsError):
        NC.write_receipt({"a": 2}, out)


# ------------------------------------------------------------ DEV-backed (spec-only) checks

@needs_dev
def test_dev_floor_canaries_and_triple_structure():
    rows, _ = h3b_null_cell.load_core_rs(DATA.resolve(), "dev")
    specs = [row["spec"] for row in rows]
    for spec in specs:
        st.design(spec)
    canaries = h3b_null_cell.floor_canaries(specs)
    assert canaries["rows"] == 250 and canaries["lex_first_passes"] == 83
    assert round(canaries["floor_mean"], 3) == 0.344
    assert round(canaries["floor_by_k"]["2"]["mean"], 3) == 0.444
    assert round(canaries["floor_by_k"]["3"]["mean"], 3) == 0.293


@needs_dev
def test_null_cell_cli_on_dev_writes_a_create_only_receipt(tmp_path, monkeypatch):
    out = tmp_path / "null_cell.json"
    argv = ["--data", str(DATA), "--role", "dev", "--redraws", "20", "--flips", "500",
            "--seed", "11", "--out", str(out)]
    with monkeypatch.context() as patch:
        patch.setattr(NC, "_IMPORT_CODE", {**NC._IMPORT_CODE, "stats_exact.py": "0" * 64})
        with pytest.raises(SystemExit, match="changed since import"):
            h3b_null_cell.main(argv)
    assert not out.exists()
    assert h3b_null_cell.main(argv) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["dataset_manifest_sha256"] == h3b_null_cell.MANIFEST_SHA
    assert receipt["canonical_file_sha256"] == manifest["files"]["canonical/dev.jsonl"]["sha256"]
    assert receipt["core_rs_rows"] == receipt["parents"] == 250 and receipt["redraws"] == 20
    assert receipt["test_challenge_read"] is False and receipt["seeds"]["seed"] == 11
    assert receipt["canaries"]["lex_first_passes"] == 83
    code = receipt["code_sha256"]
    assert code == NC._IMPORT_CODE and receipt["code_bundle_sha256"] == NC.digest(NC._IMPORT_CODE)
    for name, path in (("h3b_null_cell.py", h3b_null_cell.__file__), ("stats_exact.py", st.__file__),
                       ("content_scoring.py", content_scoring.__file__)):
        assert code[name] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for pair in receipt["pairs"].values():
        assert 0 <= pair["rejections"] <= 20 and pair["upper95"] == st.cp_upper(pair["rejections"], 20)
    # 20 re-draws and 500 flips are not the registered counts: no rule outcome is emitted
    assert receipt["rule"]["rule_compliant"] is False
    assert receipt["rule"]["outcome"] == NC.NOT_A_REGISTERED_RUN
    for policy in receipt["policies"].values():
        assert abs(policy["mean_e"]) < 0.1
    assert b"\r\n" not in out.read_bytes()
    with pytest.raises(SystemExit, match="create-only"):
        h3b_null_cell.main(argv)
