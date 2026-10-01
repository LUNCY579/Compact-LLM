"""Focused mathematical falsifiers; no checkpoint or model loads."""
import unittest
import numpy as np
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
        for u, v, eta in ((self.u,self.u,.1), (self.u,self.v,0)):
            b, r, k = construct(u, v, eta)
            np.testing.assert_allclose(apply_rotation(h,b,r), h, atol=1e-14)
        b,r,k = construct(np.eye(12)[:,:4],np.eye(12)[:,4:8])
        self.assertFalse(operator_metrics(b,r,k)["noncommuting"])

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

    def test_direction_and_invalid_input_canaries(self):
        _, reverse, reverse_k=construct(self.v,self.u)
        # Compact bases differ, so compare ambient operators rather than R entries.
        bb,rr,_=construct(self.v,self.u)
        np.testing.assert_allclose(bb@(rr-np.eye(len(rr)))@bb.T,
                                   self.b@(self.r.T-np.eye(len(self.r)))@self.b.T,atol=1e-12)
        with self.assertRaises(ValueError): construct(self.u*2,self.v)
        with self.assertRaises(ValueError): construct(self.u,self.v,.2)
        with self.assertRaises(ValueError): left_subspace(np.zeros((10,10)),4,1)


if __name__ == "__main__": unittest.main()
