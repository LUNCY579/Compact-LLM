"""Focused mathematical falsifiers; no checkpoint or model loads."""
import unittest
import numpy as np
import geometry
from geometry import apply_rotation, construct, left_subspace, operator_metrics


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.u, _ = np.linalg.qr(self.rng.normal(size=(12, 4)))
        self.v, _ = np.linalg.qr(self.rng.normal(size=(12, 4)))
        self.b, self.r, self.k = construct(self.u, self.v)

    def test_rotation_norm_and_operator_bound(self):
        h = self.rng.normal(size=(3, 5, 12))
        changed = apply_rotation(h, self.b, self.r)
        np.testing.assert_allclose(np.linalg.norm(h, axis=-1), np.linalg.norm(changed, axis=-1), atol=1e-12)
        self.assertGreater(np.linalg.norm(changed-h), 1e-6)
        self.assertLessEqual(np.linalg.norm(self.r-np.eye(len(self.r)), 2), .05)
        self.assertLessEqual(np.linalg.norm(self.k, 2), .500001)

    def test_identity_and_commuting_subspaces(self):
        h = self.rng.normal(size=(5, 12))
        axes = np.eye(12)
        cases = {
            "identical": (self.u, self.u),
            "orthogonal": (axes[:, :4], axes[:, 4:8]),
            "nested": (axes[:, :2], axes[:, :4]),
            "nested_reverse": (axes[:, :4], axes[:, :2]),
        }
        for name, (u, v) in cases.items():
            with self.subTest(case=name):
                b, r, k = construct(u, v)
                np.testing.assert_allclose(k, 0, atol=1e-14)
                np.testing.assert_allclose(r, np.eye(len(r)), atol=1e-14)
                np.testing.assert_allclose(apply_rotation(h, b, r), h, atol=1e-14)
                self.assertFalse(operator_metrics(b, r, k)["noncommuting"])

    def test_zero_intervention_is_identity_despite_noncommuting_parents(self):
        h = self.rng.normal(size=(5, 12))
        b, r, k = construct(self.u, self.v, eta=0)
        self.assertGreater(np.linalg.norm(k), 1e-6)
        np.testing.assert_allclose(r, np.eye(len(r)), atol=1e-14)
        np.testing.assert_allclose(apply_rotation(h, b, r), h, atol=1e-14)

    def test_change_of_compact_basis_is_invariant(self):
        s,_ = np.linalg.qr(self.rng.normal(size=self.r.shape))
        h = self.rng.normal(size=(8,12))
        np.testing.assert_allclose(apply_rotation(h,self.b,self.r),
                                   apply_rotation(h,self.b@s,s.T@self.r@s),atol=1e-12)

    def test_random_control_matches_operator_but_not_displacement(self):
        random_b,_ = np.linalg.qr(self.rng.normal(size=self.b.shape))
        h = self.rng.normal(size=(10,12))
        for b in (self.b,random_b):
            qminus=(b@(self.r-np.eye(len(self.r))))@b.T  # tiny fixture only
            np.testing.assert_allclose(np.linalg.norm(qminus),np.linalg.norm(self.r-np.eye(len(self.r))),atol=1e-12)
        self.assertFalse(np.allclose(apply_rotation(h,self.b,self.r),apply_rotation(h,random_b,self.r)))

    def test_swapped_parents_negate_ambient_commutator_and_invert_rotation(self):
        # Compact bases differ, so compare ambient operators rather than R entries.
        bb, rr, kk = construct(self.v, self.u)
        np.testing.assert_allclose(bb @ kk @ bb.T, -(self.b @ self.k @ self.b.T), atol=1e-12)
        np.testing.assert_allclose(bb @ (rr-np.eye(len(rr))) @ bb.T,
                                   self.b @ (self.r.T-np.eye(len(self.r))) @ self.b.T, atol=1e-12)

    def test_parent_basis_changes_preserve_ambient_construction(self):
        su, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        sv, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        # Include a sign flip: union SVD coordinates need not agree entrywise.
        su[:, 0] *= -1
        bb, rr, kk = construct(self.u @ su, self.v @ sv)
        np.testing.assert_allclose(bb @ kk @ bb.T, self.b @ self.k @ self.b.T, atol=1e-12)
        h = self.rng.normal(size=(3, 5, 12))
        np.testing.assert_allclose(apply_rotation(h, bb, rr),
                                   apply_rotation(h, self.b, self.r), atol=1e-12)

    def test_rotation_then_reversed_parent_rotation_is_identity(self):
        h = self.rng.normal(size=(3, 5, 12))
        bb, rr, _ = construct(self.v, self.u)
        forward = apply_rotation(h, self.b, self.r)
        self.assertGreater(np.linalg.norm(forward-h), 1e-6)
        np.testing.assert_allclose(apply_rotation(forward, bb, rr), h, atol=1e-12)
        np.testing.assert_allclose(apply_rotation(apply_rotation(h, bb, rr), self.b, self.r),
                                   h, atol=1e-12)

    def test_compensated_change_of_basis_preserves_linear_decoder_output(self):
        # Independent row-vector toy: h -> hQ requires W -> Q.T W.
        angle = np.pi / 3
        q = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        h = np.array([[1., 0.], [.3, -.7]])
        w = np.array([[1., 2.], [0., -1.]])
        original = h @ w
        changed = h @ q
        np.testing.assert_allclose(changed @ (q.T @ w), original, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(changed, axis=-1),
                                   np.linalg.norm(h, axis=-1), atol=1e-14)
        self.assertFalse(np.allclose(changed @ w, original))
        # Fixed-decoder behavior can change despite the same hidden-state norm;
        # this establishes neither efficacy nor a model's natural mechanism.

    def test_inverse_rotations_can_produce_the_same_behavioral_label_change(self):
        # An inverse operator need not give an opposite task-level outcome.
        angle = np.pi / 3
        q = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        h, w = np.array([[1., 0.]]), np.array([[1.], [0.]])
        threshold = .75
        np.testing.assert_allclose(q @ q.T, np.eye(2), atol=1e-14)
        self.assertTrue(bool((h @ w > threshold).item()))
        forward = bool((h @ q @ w > threshold).item())
        reverse = bool((h @ q.T @ w > threshold).item())
        self.assertFalse(forward)
        self.assertEqual(forward, reverse)

    def test_invalid_input_canaries(self):
        with self.assertRaises(ValueError): construct(self.u*2,self.v)
        with self.assertRaises(ValueError): construct(self.u,self.v,.2)
        with self.assertRaises(ValueError): left_subspace(np.zeros((10,10)),4,1)


class SectionHGeometryTests(unittest.TestCase):
    def flat_spectrum_delta(self, gap=1.1):
        # Mimics the measured parents: sigma4/sigma5 ~ 1.1, a slowly decaying tail.
        rng = np.random.default_rng(11)
        left = np.linalg.qr(rng.standard_normal((256, 64)))[0]
        right = np.linalg.qr(rng.standard_normal((512, 64)))[0]
        values = np.concatenate(([4.0, 3.0, 2.2, 1.8 * gap / 1.1], 1.8 / 1.1 * 0.98 ** np.arange(60)))
        return (left * values) @ right.T, left

    def test_exact_subspace_is_the_true_top_singular_subspace(self):
        delta, left = self.flat_spectrum_delta()
        u, values = left_subspace(delta, 4)
        self.assertTrue(np.allclose(np.linalg.svd(u.T @ left[:, :4], compute_uv=False), 1, atol=1e-10))
        np.testing.assert_allclose(values, np.linalg.svd(delta, compute_uv=False)[:4], rtol=1e-12)
        self.assertTrue(np.array_equal(left_subspace(delta, 4, 1)[0], left_subspace(delta, 4, 2)[0]))

    def test_original_randomized_finder_drifts_on_a_flat_spectrum(self):
        # Positive control for the section H finding: one power step is not enough here.
        delta, left = self.flat_spectrum_delta()
        cosines = []
        for seed in range(5):
            u_v1, _ = geometry.left_subspace_randomized_v1(delta, 4, seed)
            cosines.append(np.linalg.svd(u_v1.T @ left[:, :4], compute_uv=False).min())
        self.assertLess(min(cosines), 0.99)

    def test_haar_frame_is_orthonormal_sign_fixed_and_seeded(self):
        frame = geometry.haar_frame((64, 8), 3)
        self.assertTrue(np.allclose(frame.T @ frame, np.eye(8), atol=1e-12))
        q, r = np.linalg.qr(np.random.default_rng(3).standard_normal((64, 8)))
        np.testing.assert_allclose(frame, q * np.sign(np.diag(r)))
        self.assertTrue(np.array_equal(frame, geometry.haar_frame((64, 8), 3)))
        self.assertFalse(np.array_equal(frame, geometry.haar_frame((64, 8), 4)))



if __name__ == "__main__": unittest.main()
