import unittest
import torch

from experiments.continual_skill_audit_v1.adapter_geometry import compare_factors, effective_inner


class GeometryTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(12)
        self.a = torch.randn(3, 7, dtype=torch.float64, generator=generator)
        self.b = torch.randn(9, 3, dtype=torch.float64, generator=generator)
        self.c = torch.randn(2, 7, dtype=torch.float64, generator=generator)
        self.d = torch.randn(9, 2, dtype=torch.float64, generator=generator)

    def test_gram_identity_matches_dense_different_ranks(self):
        expected = float(((2 * self.b @ self.a) * (.5 * self.d @ self.c)).sum())
        self.assertAlmostEqual(effective_inner(self.a, self.b, self.c, self.d, 2, .5), expected, places=11)

    def test_coordinate_rescaling_does_not_change_effective_delta(self):
        result = compare_factors({'m': (self.a, self.b)}, {'m': (10 * self.a, self.b / 10)})
        self.assertAlmostEqual(result['cosine'], 1, places=12)
        self.assertAlmostEqual(result['best_scalar_left_to_right'], 1, places=12)
        self.assertLess(result['right_residual_energy_fraction'], 1e-12)

    def test_scalar_shrinkage_is_not_a_new_direction(self):
        result = compare_factors({'m': (self.a, self.b)}, {'m': (self.a, self.b)}, 2, .5)
        self.assertAlmostEqual(result['right_over_left_norm'], .25)
        self.assertAlmostEqual(result['best_scalar_left_to_right'], .25)
        self.assertLess(result['right_residual_energy_fraction'], 1e-12)

    def test_null_adapter_and_mismatch(self):
        result = compare_factors({'m': (self.a, self.b * 0)}, {'m': (self.a, self.b)})
        self.assertIsNone(result['cosine'])
        self.assertIsNone(result['best_scalar_left_to_right'])
        with self.assertRaises(ValueError):
            compare_factors({'x': (self.a, self.b)}, {'y': (self.a, self.b)})


if __name__ == '__main__':
    unittest.main()
