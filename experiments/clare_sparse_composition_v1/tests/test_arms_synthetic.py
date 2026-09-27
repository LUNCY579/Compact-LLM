"""All six arms end to end on the tiny 4-skill problem: STRUCTURAL invariants only (no performance
claims), plus the functional-interaction estimators on inputs with known answers."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from clare.allocation import ArmSpec, run_arm
from clare.functional import accuracy_effects, interaction_rows, kappa, summarise_vocab, vocab_interaction
from clare.geometry import by_layer, layer_energy, rank_energy
from clare.schedule import budgets
from tinymodel import RECIPE, SKILLS, problem

ORDER = tuple(SKILLS)


@pytest.fixture(scope="module")
def all_arms():
    model, fp, host, data, loss_fn = problem(hidden=8, inter=16)
    out = {"fp": fp}
    out["B"] = run_arm(ArmSpec("B_independent_sum", ORDER, rho="0.8"), host, data, loss_fn, RECIPE)
    for a in ("A_dense_sequential", "D_random_disjoint", "E_learned_disjoint"):
        out[a[0]] = run_arm(ArmSpec(a, ORDER, rho="0.8"), host, data, loss_fn, RECIPE)
    out["C"] = run_arm(ArmSpec("C_magnitude_sparse", ORDER, rho="0.8"), host, data, loss_fn, RECIPE,
                       independent=out["B"].deltas())
    out["F"] = run_arm(ArmSpec("F_shared_private", ORDER, rho="0.8", shared_fraction=0.1), host, data, loss_fn, RECIPE)
    # B's first delta must equal A's first stage (same data, recipe, base): a free consistency check
    return out


def test_every_arm_records_every_skill_in_order(all_arms):
    for k in "ABCDEF":
        assert [r.skill for r in all_arms[k].records] == list(ORDER)
        assert all_arms[k].prefix_sum(len(ORDER)).footprint.digest() == all_arms["fp"].digest()


def test_dense_arms_touch_everything_and_B_first_equals_A_first(all_arms):
    fp = all_arms["fp"]
    for k in "AB":
        for r in all_arms[k].records:
            assert r.mask is None and r.delta.nonzero() > 0.9 * fp.total
    assert all_arms["A"].records[0].delta.checksum() == all_arms["B"].records[0].delta.checksum()


def test_C_is_magnitude_pruned_B_at_clare_budgets(all_arms):
    fp = all_arms["fp"]
    sched = budgets(fp.total, ORDER, "0.8")
    for r, b, src in zip(all_arms["C"].records, sched, all_arms["B"].records):
        assert r.mask.count() == b.budget
        kept = np.concatenate([np.abs(src.delta.dense(s.name))[r.mask.bool(s.name)] for s in fp])
        dropped = np.concatenate([np.abs(src.delta.dense(s.name))[~r.mask.bool(s.name)] for s in fp])
        assert kept.min() >= dropped.max()
        for s in fp:  # values are B's own, unscaled
            m = r.mask.bool(s.name)
            assert r.delta.dense(s.name)[m].tobytes() == src.delta.dense(s.name)[m].tobytes()


def test_F_private_masks_disjoint_from_each_other_and_from_shared(all_arms):
    F = all_arms["F"]
    shared = F.shared_mask
    assert shared is not None and shared.count() == round(0.1 * all_arms["fp"].total)
    for i, a in enumerate(F.records):
        assert a.mask.overlap(shared) == 0
        for b in F.records[i + 1:]:
            assert a.mask.overlap(b.mask) == 0
        # a skill's delta lives on its private support plus the shared support
        for s in all_arms["fp"]:
            nz = a.delta.dense(s.name) != 0
            assert not (nz & ~(a.mask.bool(s.name) | shared.bool(s.name))).any()


def test_layer_and_rank_diagnostics_are_well_formed(all_arms):
    d = all_arms["E"].records[0].delta
    rows = layer_energy(d, all_arms["E"].records[0].mask)
    assert sum(r["energy_share"] for r in rows) == pytest.approx(1.0)
    layers = by_layer(rows)
    assert sum(l["allocated"] for l in layers) == all_arms["E"].records[0].mask.count()
    rk = rank_energy(all_arms["A"].records[0].delta, ks=(1, 2, 100))
    assert all(r["top100_energy_fraction"] == pytest.approx(1.0) for r in rk)
    assert all(0 < r["top1_energy_fraction"] <= 1 for r in rk)


def test_interaction_is_zero_for_additive_effects_and_detected_otherwise():
    base = np.array([-3.0, -2.0, -5.0])
    gA, gB = np.array([1.0, 0.5, 2.0]), np.array([0.2, 0.3, 0.1])
    Z = {"0": base, "A": base + gA, "B": base + gB, "AB": base + gA + gB}
    assert np.allclose(interaction_rows(Z), 0.0)
    k = kappa(Z, n_boot=200)
    assert k["kappa"] == pytest.approx(0.0, abs=1e-12) and k["label_under_fa001_rule"] == "ADDITIVE"
    Z["AB"] = base + gA + gB - 0.5 * gA  # sub-additive by half of A's gain
    assert kappa(Z, n_boot=200)["kappa"] == pytest.approx(-0.5)
    # vocabulary level: log-softmax of additive LOGIT shifts is not additive in general, but an
    # additive shift of log-probabilities is exactly zero interaction
    rng = np.random.default_rng(0)
    L0 = np.log(rng.dirichlet(np.ones(40), size=5))
    dA, dB = 0.1 * rng.standard_normal((5, 40)), 0.1 * rng.standard_normal((5, 40))
    vi = vocab_interaction({"0": L0, "A": L0 + dA, "B": L0 + dB, "AB": L0 + dA + dB}, topk=8)
    assert np.allclose(vi["I_full"], 0.0) and np.allclose(vi["I_topk"], 0.0)
    vi = vocab_interaction({"0": L0, "A": L0 + dA, "B": L0 + dB, "AB": L0 + 2 * dA + dB}, topk=8)
    s = summarise_vocab(vi)
    assert s["full"]["mean_relative"] > 0.1
    t = vocab_interaction({k: torch.from_numpy(v) for k, v in {"0": L0, "A": L0 + dA, "B": L0 + dB, "AB": L0 + 2 * dA + dB}.items()}, topk=8)
    assert np.allclose(t["I_full"], vi["I_full"])


def test_accuracy_effects_definitions():
    e = accuracy_effects({"0": 0.1, "A": 0.4, "B": 0.3, "AB": 0.5})
    assert e["composition_gain"] == pytest.approx(0.1) and e["composition_damage"] == 0.0
    assert e["additive_residual"] == pytest.approx(0.5 - 0.4 - 0.3 + 0.1)


def test_lambda_gate_falls_back_when_the_probe_is_too_sparse():
    model, fp, host, data, loss_fn = problem()
    spec = ArmSpec("E_learned_disjoint", ORDER, rho="0.8", l1_lambda=1e6, lambda_fallback=(0.0,))
    res = run_arm(spec, host, data, loss_fn, RECIPE)
    d = res.records[0].discovery
    assert [a["l1_lambda"] for a in d["lambda_attempts"]] == [1e6, 0.0]
    assert d["lambda_attempts"][0]["nonzero_free_after_probe"] == 0  # the huge threshold zeroed the probe
    assert d["lambda_gate_passed"] and d["selected_with_zero_importance"] == 0
    assert d["effective_checksum_before"] == d["effective_checksum_after_reset"]
    # without a fallback the gate failure is RECORDED, not hidden
    res = run_arm(ArmSpec("E_learned_disjoint", ORDER, rho="0.8", l1_lambda=1e6), host, data, loss_fn, RECIPE)
    assert res.records[0].discovery["lambda_gate_passed"] is False
