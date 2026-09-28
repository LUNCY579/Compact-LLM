import numpy as np
import pytest
from experiments.rineya_composable_subspace_stack_v1.core import (
    Arm, matrix, svd, truncate, protection, shared_private, ctm_basis,
    merge, compose, design, geometry, effective_lora)


@pytest.mark.parametrize('shape', [(5, 5), (7, 4), (4, 7)])
@pytest.mark.parametrize('rank', [0, 1, 3])
def test_projection_and_reconstruction(shape, rank):
    rng = np.random.default_rng(99)
    w, d = rng.normal(size=shape), rng.normal(size=shape)
    safe, removed, report = protection(w, d, rank)
    assert np.allclose(safe+removed, d)
    assert np.vdot(safe, removed) == pytest.approx(0, abs=1e-11)
    u, _, vt = svd(w, rank)
    assert np.linalg.norm(u.T @ safe) < 1e-11
    assert np.linalg.norm(safe @ vt.T) < 1e-11
    assert np.allclose((w+safe) @ vt.T, w @ vt.T)
    assert np.allclose((w+safe).T @ u, w.T @ u)
    assert report['removed_is_harmful'] is None


def test_protection_does_not_guarantee_singular_order():
    w = np.diag([5., 1.])
    safe, _, _ = protection(w, np.diag([3., 30.]), 1)
    assert (w+safe)[0, 0] == 5
    assert np.linalg.svd(w+safe, compute_uv=False)[0] == 31


@pytest.mark.parametrize('bad', [[], np.ones(4), np.ones((2, 2), int), np.array([[np.nan]]), np.array([[np.inf]]), np.zeros((0, 4))])
def test_invalid_matrices(bad):
    with pytest.raises(ValueError):
        matrix(bad)


@pytest.mark.parametrize('rank', [-1, 6, 0.5, True])
def test_invalid_rank(rank):
    with pytest.raises(ValueError):
        truncate(np.eye(4), rank)


def test_zero_deltas_have_no_fake_subspace_or_cosine():
    z = np.zeros((4, 4))
    shared, private, report = shared_private([z, z], 2)
    assert report['left_rank'] == report['right_rank'] == 0
    assert np.array_equal(shared[0], z) and np.array_equal(private[0], z)
    assert geometry([z, z])['pairs'][0]['cosine'] is None
    assert geometry([z])['spectra'][0]['effective_rank_energy'] == 0


def test_shared_private_reconstruction_and_orthogonality():
    a, b = np.diag([2., 1., 0., 0.]), np.diag([2., 0., 1., 0.])
    shared, private, report = shared_private([a, b], 2, .75)
    assert report['left_rank'] == report['right_rank'] == 1
    for d, s, p in zip([a, b], shared, private):
        assert np.allclose(d, s+p)
        assert abs(np.vdot(s, p)) < 1e-12
    assert np.allclose(shared[0], np.diag([2., 0., 0., 0.]))


@pytest.mark.parametrize('shape', [(6, 4), (4, 6), (5, 5)])
@pytest.mark.parametrize('rank', [0, 1, 3])
def test_ctm_tucker_monotonic_rank_and_original_coordinates(shape, rank):
    rng = np.random.default_rng(1729)
    ds = [rng.normal(size=shape), 0.1*rng.normal(size=shape)]
    u, v, history = ctm_basis(ds, rank, beta=.2, iterations=5)
    assert np.all(np.diff(history) >= -1e-9)
    out = u @ merge([u.T@d@v for d in ds], [.4, .6]) @ v.T if rank else np.zeros(shape)
    assert np.linalg.matrix_rank(out, tol=1e-9) <= rank
    # With a fixed common basis, linear merging and projection commute.
    assert np.allclose(out, u @ (u.T@merge(ds, [.4, .6])@v) @ v.T)


def test_ctm_full_rank_equals_raw_merge():
    rng = np.random.default_rng(20)
    ds = [rng.normal(size=(4, 4)), rng.normal(size=(4, 4))]
    out, _ = compose(np.eye(4), ds, [.5, .5], Arm('x', compression='ctm'), rank=4, protected_rank=0, shared_rank=2)
    assert np.allclose(out, merge(ds, [.5, .5]))


def test_share_identity_gain_one_is_exact_noop():
    ds = [np.diag([2., 1., 0., 0.]), np.diag([2., 0., 1., 0.])]
    out, _ = compose(np.eye(4), ds, [.5, .5], Arm('s', share=True), rank=2, shared_rank=2, protected_rank=0, private_gain=1.)
    assert np.allclose(out, merge(ds, [.5, .5]))


def test_share_reweighting_is_distinct_from_ctm_projection():
    ds = [np.diag([2., 1., 0., 0.]), np.diag([2., 0., 1., 0.])]
    out, _ = compose(np.eye(4), ds, [.5, .5], Arm('s', share=True), rank=2, shared_rank=2, protected_rank=0, private_gain=.5)
    assert np.allclose(out, np.diag([2., .25, .25, 0.]))
    assert not np.allclose(out, merge(ds, [.5, .5]))


@pytest.mark.parametrize('arm', design(), ids=lambda a: a.name)
def test_all_arms_finite_do_not_mutate_inputs(arm):
    rng = np.random.default_rng(555)
    w, a, b = [rng.normal(size=(6, 6)) for _ in range(3)]
    originals = [x.copy() for x in (w, a, b)]
    result, report = compose(w, [a, b], [.5, .5], arm, rank=2, protected_rank=1, shared_rank=2)
    assert np.isfinite(result).all()
    assert all(np.array_equal(x, y) for x, y in zip((w, a, b), originals))
    if arm.compression in ('ctm', 'mtc'):
        assert np.linalg.matrix_rank(result, tol=1e-8) <= 2
    if arm.norm_match:
        assert np.linalg.norm(result) == pytest.approx(np.linalg.norm(.5*a+.5*b))


def test_norm_match_refuses_erased_nonzero_update():
    with pytest.raises(ValueError, match='erased'):
        compose(np.eye(4), [np.eye(4), np.eye(4)], [.5, .5], Arm('erase', True, False, 'none', True), rank=2, protected_rank=4, shared_rank=2)


def test_lora_gauge_and_scale():
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, 4)), rng.normal(size=(5, 2))
    g = np.array([[2., .3], [.1, 1.]])
    assert np.allclose(effective_lora(g@a, b@np.linalg.inv(g), .4), .4*b@a)


def test_distinct_registered_cells():
    arms = design()
    assert len(arms) == 11
    assert len({(a.protect, a.share, a.compression, a.norm_match) for a in arms}) == 11


@pytest.mark.parametrize('bad', [[.5], [np.nan, .5], [True, .5]])
def test_bad_coefficients(bad):
    with pytest.raises(ValueError):
        merge([np.eye(3), np.eye(3)], bad)
