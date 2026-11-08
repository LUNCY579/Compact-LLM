import pytest
import torch
from experiments.rineya_hebbian_fact_memory_v1.memory import (
    BuildConfig, FrozenBilinear, build, ridge_readout, fit_projection,
    conjugate_gradient, swap_readout, effective_kernel, matrix,
)
from experiments.rineya_hebbian_fact_memory_v1.bench import synthetic_facts, study, ssfr
from experiments.rineya_hebbian_fact_memory_v1.metrics import recall, geometry, noisy_queries, paired, paired_bootstrap

torch.set_num_threads(1)


@pytest.mark.parametrize("field,value", [
    ("width", 0), ("width", True), ("width", 1.5), ("seed", -1), ("seed", 2**63),
    ("seed", True), ("variant", "mlp"), ("ridge", -1), ("ridge", float("nan")),
    ("ridge", float("inf")), ("ridge", True), ("ridge", "bad"),
    ("cg_tolerance", 0), ("cg_max_iterations", 0), ("max_elements", 0),
    ("projection_ridge", -1), ("ridge_convention", "ambiguous"), ("projection_objective", "bad"),
])
def test_bad_config(field, value):
    with pytest.raises(ValueError):
        BuildConfig(**{field: value})


@pytest.mark.parametrize("shape", [(3, 7), (9, 4), (5, 5)])
def test_primal_dual_mean_loss(shape):
    rng = torch.Generator().manual_seed(41)
    x = torch.randn(*shape, generator=rng, dtype=torch.float64)
    y = torch.randn(shape[0], 3, generator=rng, dtype=torch.float64)
    a, ra = ridge_readout(x, y, .07, route="primal")
    b, rb = ridge_readout(x, y, .07, route="dual")
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    assert ra["regularizer"] == rb["regularizer"] == shape[0]*.07


def test_paper_branch_exposes_difference():
    x = torch.arange(24, dtype=torch.float64).reshape(4, 6)/24
    y = torch.eye(4, dtype=torch.float64)
    a, ra = ridge_readout(x, y, .1, "paper_branch", "primal")
    b, rb = ridge_readout(x, y, .1, "paper_branch", "dual")
    assert ra["regularizer"] == .4 and rb["regularizer"] == .1
    assert not torch.allclose(a, b)
    matched, _ = ridge_readout(x, y, .4, "paper_branch", "dual")
    torch.testing.assert_close(a, matched)


def test_mlp_hebbian_identity_and_rank_boundary():
    rng = torch.Generator().manual_seed(12)
    phi = torch.randn(12, 4, generator=rng, dtype=torch.float64)
    native_b = torch.randn(3, 4, generator=rng, dtype=torch.float64)
    recovered, receipt = ridge_readout(phi, phi@native_b.T, 0)
    torch.testing.assert_close(recovered, native_b, atol=1e-12, rtol=1e-10)
    assert receipt["rank"] == 4
    deficient = phi.clone()
    deficient[:, 1] = deficient[:, 0]
    projected, rec = ridge_readout(deficient, deficient@native_b.T, 0)
    torch.testing.assert_close(deficient@projected.T, deficient@native_b.T)
    assert rec["rank"] == 3 and not torch.allclose(projected, native_b)


@pytest.mark.parametrize("variant", ["unwhitened", "diagonal", "whitened", "data_dependent"])
def test_construction_kernel_and_freezing(variant):
    k, v, labels = synthetic_facts(8, 6, 17)
    config = BuildConfig(width=12, variant=variant)
    rng_before = torch.random.get_rng_state().clone()
    model, rec = build(k, v[labels], config)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert not list(model.parameters()) and not model.b.requires_grad
    assert rec["construction_optimizer_steps"] == 0
    h = effective_kernel(model.features(k), config)
    torch.testing.assert_close(h@v[labels], model(k), atol=1e-8, rtol=1e-7)
    diag = geometry(h, v, labels)
    met = recall(model(k), v, labels)
    assert diag["decomposed_minimum_margin"] == pytest.approx(met["minimum_margin"], abs=1e-8)
    assert diag["cauchy_lower_bound"] <= met["minimum_margin"]+1e-8
    assert 0 <= diag["coupling"] <= 1+1e-12


@pytest.mark.parametrize("objective", ["vector", "upstream_score"])
def test_projection_cg_matches_dense_oracle(objective):
    rng = torch.Generator().manual_seed(8)
    k = torch.randn(5, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(5, 2, generator=rng, dtype=torch.float64)
    partner = torch.randn(4, 3, generator=rng, dtype=torch.float64)
    b0 = torch.randn(2, 4, generator=rng, dtype=torch.float64)
    config = BuildConfig(width=4, projection_objective=objective, projection_ridge=.1, cg_tolerance=1e-10)
    actual, rec = fit_projection(k, y, partner, b0, config)
    basis = torch.eye(12, dtype=torch.float64).reshape(12, 4, 3)
    def evaluate(w):
        out = ((k@w.T)*(k@partner.T))@b0.T
        return out if objective == "vector" else out@y.T
    design = torch.stack([evaluate(w).reshape(-1) for w in basis], 1)
    target = y if objective == "vector" else torch.eye(5, dtype=torch.float64)
    oracle = torch.linalg.solve(design.T@design+.1*torch.eye(12, dtype=torch.float64), design.T@target.reshape(-1))
    torch.testing.assert_close(actual.reshape(-1), oracle, atol=1e-8, rtol=1e-7)
    assert rec["converged"] and rec["relative_residual"] < 1e-8


def test_cg_rejects_nonconvergence_and_indefinite():
    x = torch.tensor([1., 1.], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="did not converge"):
        conjugate_gradient(lambda z: z*torch.tensor([1., 4.]), x, x, 1e-12, 1)
    with pytest.raises(RuntimeError, match="positive definiteness"):
        conjugate_gradient(lambda z: -z, x, x, 1e-12, 4)
    out, rec = conjugate_gradient(lambda z: z, x*0, x, 1e-12, 4)
    assert out.norm() == 0 and rec["iterations"] == 0


def test_swap_does_not_mutate_original_dtype_or_weights():
    k, v, labels = synthetic_facts(8, 6, 17)
    original, _ = build(k, v[labels], BuildConfig(width=16))
    original.float()
    before = {n: t.clone() for n, t in original.named_buffers()}
    updated, _ = swap_readout(original, k, v[labels.roll(1)])
    assert original.a.dtype == torch.float32 and updated.a.dtype == torch.float64
    for n, t in original.named_buffers():
        assert torch.equal(t, before[n])
    assert recall(updated(k), v, labels.roll(1))["strict_accuracy"] == 1


def test_antipodal_and_tie_limitations():
    k = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    v = torch.eye(2, dtype=torch.float64)
    model, _ = build(k, v, BuildConfig(width=8))
    torch.testing.assert_close(model(k)[0], model(k)[1])
    assert recall(model(k), v, torch.arange(2))["accuracy"] <= .5
    zero = recall(torch.zeros(2, 2), v, torch.zeros(2, dtype=torch.long))
    assert zero["accuracy"] == 1 and zero["strict_accuracy"] == 0


def test_many_to_one_geometry():
    k, v, _ = synthetic_facts(9, 6, 81)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2])
    model, _ = build(k, v[labels], BuildConfig(width=16))
    h = effective_kernel(model.features(k), BuildConfig(width=16))
    diag = geometry(h, v[:3], labels)
    assert recall(model(k), v[:3], labels)["strict_accuracy"] == 1
    assert diag["decomposed_minimum_margin"] > 0


@pytest.mark.parametrize("bad", [torch.empty(0, 2), torch.ones(2), torch.ones(2, 2, dtype=torch.long), torch.tensor([[float("nan")]])])
def test_bad_matrix(bad):
    with pytest.raises(ValueError):
        matrix(bad, "bad")


def test_invalid_dimensions_and_budget():
    with pytest.raises(ValueError):
        build(torch.zeros(2, 3), torch.eye(2))
    with pytest.raises(ValueError):
        build(torch.ones(3, 4), torch.eye(2))
    with pytest.raises(MemoryError):
        build(torch.ones(3, 4), torch.eye(3), BuildConfig(max_elements=5))
    with pytest.raises(MemoryError):
        geometry(torch.eye(10), torch.eye(10), torch.arange(10), max_elements=50)
    with pytest.raises(ValueError):
        recall(torch.ones(2, 3), torch.eye(3), torch.tensor([-1, 0]))


def test_noise_has_requested_norm_and_preserves_rng():
    k, _, _ = synthetic_facts(8, 6, 2)
    old = torch.random.get_rng_state().clone()
    noisy = noisy_queries(k, .2, 6)
    torch.testing.assert_close((noisy-k).norm(dim=1), k.norm(dim=1)*.2)
    assert torch.equal(old, torch.random.get_rng_state())
    assert torch.equal(noisy_queries(k, 0, 6), k)
    with pytest.raises(ValueError):
        noisy_queries(k, float("nan"), 1)


def test_paired_and_bootstrap():
    r = paired([True, False, True], [False, True, True])
    assert r["repairs"] == r["damages"] == 1 and r["accuracy_delta"] == 0
    b = paired_bootstrap([True]*4, [True]*4, repetitions=100)
    assert b["bootstrap_95_percent_interval"] == [0., 0.]
    with pytest.raises(ValueError):
        paired([1], [0])


def test_synthetic_runner_records_negative_controls():
    report = study(n=8, d=6, width=16, seed=17, gd_steps=4)
    assert report["native_qualified"] is False
    assert report["arms"]["whitened"]["noise"]["0.0"]["strict_accuracy"] == 1
    assert report["controls"]["shuffled_mapping"]["metrics"]["accuracy"] == 0
    assert report["controls"]["gd_bilinear"]["optimizer_steps"] == 4


def test_fact_adaptive_runner_is_not_train_only():
    report = ssfr(seed=17, steps=30, n=8, d=8, width=24)
    assert report["eval_rng_disjoint"] and not report["residual_bypass"]
    assert "swapped_mapping_b" in report and "stale_memory_negative_control" in report
    assert not report["native_qualified"]


@pytest.mark.parametrize("variant", ["unwhitened", "diagonal", "whitened"])
def test_edit_preserves_readout_algorithm(variant):
    k, v, labels = synthetic_facts(8, 6, 23)
    config = BuildConfig(width=12, variant=variant)
    base, _ = build(k, v[labels], config)
    swapped, _ = swap_readout(base, k, v[labels.roll(1)], config)
    fresh, _ = build(k, v[labels.roll(1)], config)
    for a, b in zip(swapped.buffers(), fresh.buffers()):
        torch.testing.assert_close(a, b)


def test_sequential_numeric_failure_is_recorded_not_hidden(monkeypatch):
    from experiments.rineya_hebbian_fact_memory_v1 import bench
    original = bench.build
    def injected(keys, targets, config):
        if len(keys) == 4 and config.variant == "data_dependent":
            raise RuntimeError("injected CG nonconvergence")
        return original(keys, targets, config)
    monkeypatch.setattr(bench, "build", injected)
    result = bench.study(n=8, d=6, width=16)
    arm = result["arms"]["data_dependent"]
    assert arm["status"] == "MEASURED"
    assert arm["sequential"]["status"] == "BLOCKED_NUMERICS"


def test_data_dependent_upstream_score_mode_constructs():
    k, v, labels = synthetic_facts(5, 4, 11)
    _, receipt = build(k, v[labels], BuildConfig(width=6, variant="data_dependent",
                         projection_objective="upstream_score", projection_ridge=.1))
    assert all(r["converged"] for r in receipt["factor_solves"])
