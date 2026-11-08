"""Tiny independent algebra invariants; no model, torch, training or bank access."""
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .mechanics import (
    activation_overlap, compose_deltas, delta_report, extract_delta, fit_transport,
    lora_delta, normalized_depth_map, norm_match, orthonormal_basis, parameter_overlap,
    project_delta, random_matched_delta, subspace_overlap, tensor_digest,
    transport_diagnostics,
)


class DepthTests(unittest.TestCase):
    def test_endpoints_monotonic_and_exact_ties(self):
        self.assertEqual(normalized_depth_map(3, 2), (0, 0, 1))
        self.assertEqual(normalized_depth_map(5, 3), (0, 0, 1, 1, 2))
        self.assertEqual(normalized_depth_map(1, 5), (0,))
        self.assertEqual(normalized_depth_map(5, 1), (0,) * 5)
        mapping = normalized_depth_map(36, 28)
        self.assertEqual((mapping[0], mapping[-1]), (0, 27))
        self.assertEqual(tuple(sorted(mapping)), mapping)

    def test_counts_refuse_invalid(self):
        for count in (0, -1, 1.0, True, "28"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                normalized_depth_map(count, 28)


class ExtractionTests(unittest.TestCase):
    def test_keyed_delta_reconstructs_and_does_not_mutate(self):
        base = {"z": np.ones((2, 3)), "a": np.array([1., 3.])}
        skilled = {"a": np.array([2., 1.]), "z": np.arange(6.).reshape(2, 3)}
        before = tensor_digest(base)
        delta = extract_delta(base, skilled)
        self.assertEqual(list(delta), ["a", "z"])
        for key in base:
            assert_array_equal(base[key] + delta[key], skilled[key])
            self.assertFalse(delta[key].flags.writeable)
        self.assertEqual(tensor_digest(base), before)

    def test_lora_matches_sequential_action(self):
        rng = np.random.default_rng(17)
        a, b, x = rng.normal(size=(2, 4)), rng.normal(size=(3, 2)), rng.normal(size=(7, 4))
        delta = lora_delta(a, b, 6)
        assert_allclose(x @ delta.T, ((x @ a.T) @ b.T) * 3, atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(delta), 2)

    def test_lora_rejects_invalid_rank_alpha_and_schema(self):
        invalid = ((np.ones((2, 3)), np.ones((4, 1)), 4),
                   (np.ones((2, 3)), np.ones((4, 2)), 0),
                   (np.ones((2, 3)), np.ones((4, 2)), float("nan")),
                   (np.ones(3), np.ones((4, 2)), 4))
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                lora_delta(*args)

    def test_refuses_nonfinite_nonnumeric_and_empty(self):
        for bad in (np.array([np.nan]), np.array([np.inf]), np.array([True]),
                    np.array([1j]), np.array(["1"]), np.empty(0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                extract_delta({"m": bad}, {"m": bad})

    def test_refuses_key_and_shape_drift(self):
        for bad in ({"n": np.ones((2, 2))}, {"m": np.ones((2, 3))}, {}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                extract_delta({"m": np.ones((2, 2))}, bad)

    def test_digest_binds_values_names_shapes_and_ignores_order(self):
        a = {"m": np.array([[1., 2.]], dtype=np.float32), "n": np.array([3.])}
        b = {"n": np.array([3.]), "m": np.array([[1., 2.]], dtype=np.float64)}
        self.assertEqual(tensor_digest(a), tensor_digest(b))
        for changed in ({"m": np.array([[1., 3.]]), "n": np.array([3.])},
                        {"m": np.array([1., 2.]), "n": np.array([3.])},
                        {"q": np.array([[1., 2.]]), "n": np.array([3.])}):
            self.assertNotEqual(tensor_digest(a), tensor_digest(changed))


class TransportTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260920)
        self.x = rng.normal(size=(30, 4))
        self.w = rng.normal(size=(4, 3))
        self.b = np.array([8., -3., 6.])
        self.y = self.x @ self.w + self.b
        self.fit = fit_transport(self.x[:20], self.y[:20])

    def test_recovers_affine_map_on_independent_rows(self):
        assert_allclose(self.fit.weight, self.w, atol=1e-12)
        assert_allclose(self.fit.bias, self.b, atol=1e-12)
        report = transport_diagnostics(self.fit, self.x[20:], self.y[20:])
        self.assertLess(report["mse"], 1e-25)
        self.assertAlmostEqual(report["r2"], 1)
        self.assertFalse(report["functional_transfer_established"])
        self.assertEqual(report["check_rows"], 10)

    def test_residual_cancels_intercept(self):
        change = np.full((10, 4), 0.5)
        assert_allclose(self.fit.residual(change),
                        self.fit.predict(self.x[20:] + change) - self.fit.predict(self.x[20:]), atol=1e-12)
        assert_array_equal(self.fit.residual(np.zeros((10, 4))), np.zeros((10, 3)))

    def test_frozen_fit_copies_inputs(self):
        saved = self.fit.predict(self.x[20:]).copy()
        self.x[:20] = 1e4
        self.y[:] = 1e4
        assert_array_equal(saved, self.fit.predict(self.x[20:]))
        with self.assertRaises(ValueError):
            self.fit.weight[0, 0] = 0

    def test_rank_deficient_and_constant_transport(self):
        x = np.column_stack([np.arange(6.), 2 * np.arange(6.), np.ones(6)])
        y = np.column_stack([3 * np.arange(6.) + 2, -np.arange(6.) + 7])
        fit = fit_transport(x, y)
        self.assertEqual(fit.rank, 1)
        assert_allclose(fit.predict(x), y, atol=1e-12)
        fit = fit_transport(np.ones((4, 3)), np.full((4, 2), 8.))
        self.assertEqual(fit.rank, 0)
        assert_array_equal(fit.residual(np.ones((2, 3))), np.zeros((2, 2)))
        self.assertIsNone(transport_diagnostics(fit, np.ones((2, 3)), np.full((2, 2), 8.))["r2"])

    def test_bad_fit_and_check_inputs_refused(self):
        for x, y in ((np.ones((3, 2)), np.ones((4, 1))),
                     (np.ones((1, 2)), np.ones((1, 1))),
                     (np.array([[1., np.nan], [2., 3.]]), np.ones((2, 1)))):
            with self.subTest(x=x), self.assertRaises(ValueError):
                fit_transport(x, y)
        for rcond in (0., -1., 1., float("inf")):
            with self.subTest(rcond=rcond), self.assertRaises(ValueError):
                fit_transport(self.x, self.y, rcond=rcond)
        with self.assertRaises(ValueError):
            self.fit.residual(np.ones((2, 5)))
        with self.assertRaises(ValueError):
            transport_diagnostics(self.fit, self.x[20:], self.y[20:, :2])


class ProjectionTests(unittest.TestCase):
    def test_rank_deficient_basis_and_empty_basis(self):
        columns = np.array([[1., 2., 0.], [2., 4., 0.], [0., 0., 0.]])
        q = orthonormal_basis(columns)
        self.assertEqual(q.shape, (3, 1))
        assert_allclose(q.T @ q, np.eye(1), atol=1e-12)
        assert_allclose(q @ q.T @ columns, columns, atol=1e-12)
        self.assertEqual(orthonormal_basis(np.zeros((3, 2))).shape, (3, 0))
        self.assertEqual(orthonormal_basis(np.empty((3, 0))).shape, (3, 0))

    def test_projection_reconstructs_pythagoras_and_is_idempotent(self):
        base = {"m": np.ones((2, 2))}
        skill = {"m": np.array([[1., 0.], [0., 0.]])}
        delta = {"m": np.array([[3., 4.], [0., 0.]])}
        p, removed, report = project_delta(delta, [skill, skill], base)
        assert_allclose(p["m"], [[0., 4.], [0., 0.]], atol=1e-12)
        assert_allclose(p["m"] + removed["m"], delta["m"], atol=1e-12)
        self.assertAlmostEqual(report["weighted_norm_before"] ** 2,
                               report["weighted_norm_after"] ** 2 + report["weighted_removed_norm"] ** 2)
        self.assertEqual(report["basis_rank"], 1)
        self.assertAlmostEqual(report["projection_energy_removed"], 9 / 25)
        p2, _, _ = project_delta(p, [skill], base)
        assert_allclose(p2["m"], p["m"], atol=1e-12)
        assert_array_equal(delta["m"], [[3., 4.], [0., 0.]])

    def test_fixed_base_weighting_and_layer_grouping(self):
        base = {"a": np.array([[2.]]), "b": np.array([[100.]]), "c": np.array([[1.]])}
        d = {"a": np.array([[4.]]), "b": np.array([[100.]]), "c": np.array([[2.]])}
        prior = {"a": np.array([[2.]]), "b": np.array([[100.]]), "c": np.array([[1.]])}
        p, _, report = project_delta(d, [prior], base, module_layers={"a": "0", "b": "0", "c": "1"})
        scales = report["module_scales"]
        self.assertAlmostEqual(scales["a"], 1 / (2 * np.sqrt(2)))
        self.assertAlmostEqual(scales["b"], 1 / (100 * np.sqrt(2)))
        self.assertEqual(scales["c"], 1)
        weighted_dot = sum(float(np.vdot(p[k] * scales[k], prior[k] * scales[k])) for k in p)
        self.assertAlmostEqual(weighted_dot, 0., places=12)
        # Weighted orthogonality need not imply raw parameter orthogonality.
        self.assertGreater(abs(parameter_overlap(p, prior)["global_cosine"]), 0.1)

    def test_reports_rank_growth_honestly(self):
        d = {"m": np.array([[1., 0.], [0., 0.]])}
        prior = {"m": np.array([[1., 1.], [1., 0.]])}
        p, _, report = project_delta(d, [prior], {"m": np.ones((2, 2))})
        self.assertEqual(report["rank_before"]["m"], 1)
        self.assertEqual(report["rank_after"]["m"], 2)
        self.assertEqual(report["rank_increased_modules"], ["m"])
        self.assertFalse(report["rank_preservation_guaranteed"])
        self.assertAlmostEqual(parameter_overlap(p, prior)["global_cosine"], 0., places=12)

    def test_zero_and_empty_protection_identity(self):
        d = {"m": np.eye(2)}
        for protected in ([], [{"m": np.zeros((2, 2))}]):
            p, r, report = project_delta(d, protected, {"m": np.ones((2, 2))})
            assert_array_equal(p["m"], d["m"])
            assert_array_equal(r["m"], np.zeros((2, 2)))
            self.assertEqual(report["basis_rank"], 0)

    def test_roundoff_only_projection_cannot_be_norm_amplified(self):
        d = {"m": np.array([[2., 3.], [4., 7.]])}
        p, r, report = project_delta(d, [d], {"m": np.ones((2, 2))})
        assert_array_equal(p["m"], np.zeros((2, 2)))
        assert_array_equal(r["m"], d["m"])
        self.assertTrue(report["projection_numerically_annihilated"])
        with self.assertRaises(ValueError):
            norm_match(p, 1.)

    def test_bad_basis_scope_and_metric_refused(self):
        d = {"m": np.eye(2)}
        with self.assertRaises(ValueError):
            project_delta(d, [{"n": np.eye(2)}], d)
        with self.assertRaises(ValueError):
            project_delta(d, [{"m": np.eye(3)}], d)
        with self.assertRaises(ValueError):
            project_delta(d, [], {"m": np.zeros((2, 2))})
        with self.assertRaises(ValueError):
            project_delta(d, [], d, module_layers={"n": "0"})
        with self.assertRaises(ValueError):
            orthonormal_basis(np.array([[np.nan]]))


class CompositionAndControlTests(unittest.TestCase):
    def test_composition_preserves_originals_and_reconstructs(self):
        a, b = {"m": np.eye(2)}, {"m": np.ones((2, 2))}
        da, db = tensor_digest(a), tensor_digest(b)
        ab = compose_deltas(a, b, scale_a=2, scale_b=-0.5)
        assert_array_equal(ab["m"], 2 * a["m"] - 0.5 * b["m"])
        self.assertEqual((tensor_digest(a), tensor_digest(b)), (da, db))
        self.assertFalse(ab["m"].flags.writeable)
        with self.assertRaises(ValueError):
            compose_deltas(a, {"m": np.eye(3)})
        with self.assertRaises(ValueError):
            compose_deltas(a, b, scale_b=float("nan"))

    def test_norm_control_preserves_direction_and_matches(self):
        original = {"a": np.array([[3., 4.]]), "b": np.array([[12.]])}
        controlled = norm_match(original, 2.6)
        self.assertAlmostEqual(delta_report(controlled)["l2"], 2.6)
        self.assertAlmostEqual(parameter_overlap(original, controlled)["global_cosine"], 1.)
        assert_allclose(controlled["a"], original["a"] * 0.2)
        self.assertEqual(delta_report(norm_match(original, 0))["l2"], 0)
        self.assertEqual(delta_report(norm_match({"m": np.zeros((2, 2))}, 0))["l2"], 0)

    def test_impossible_norm_control_refused(self):
        for target in (1., -1., float("inf"), True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                norm_match({"m": np.zeros((2, 2))}, target)

    def test_random_control_matches_local_spectra_ranks_and_norms(self):
        reference = {"a": np.diag([4., 2., 0.]), "b": np.array([[1., 2., 3.], [2., 4., 6.]]),
                     "zero": np.zeros((2, 3))}
        random = random_matched_delta(reference, 77)
        self.assertEqual(tensor_digest(random), tensor_digest(random_matched_delta(reference, 77)))
        self.assertNotEqual(tensor_digest(random), tensor_digest(random_matched_delta(reference, 78)))
        for key in reference:
            assert_allclose(np.linalg.svd(random[key], compute_uv=False),
                            np.linalg.svd(reference[key], compute_uv=False), atol=1e-12)
            self.assertEqual(np.linalg.matrix_rank(random[key]), np.linalg.matrix_rank(reference[key]))
            self.assertAlmostEqual(np.linalg.norm(random[key]), np.linalg.norm(reference[key]))
        self.assertEqual(delta_report(random)["target_parameter_count"], delta_report(reference)["target_parameter_count"])

    def test_random_control_invalid_inputs_refused(self):
        with self.assertRaises(ValueError):
            random_matched_delta({"m": np.ones(3)}, 3)
        for seed in (-1, True, 1.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                random_matched_delta({"m": np.eye(2)}, seed)


class OverlapTests(unittest.TestCase):
    def test_equal_module_overlap_not_dominated_by_big_module(self):
        a = {"big": np.ones((100, 100)), "small": np.ones((1, 1))}
        b = {"big": np.ones((100, 100)), "small": -np.ones((1, 1))}
        overlap = parameter_overlap(a, b)
        self.assertGreater(overlap["global_cosine"], 0.99)
        self.assertAlmostEqual(overlap["equal_module_mean_cosine"], 0)
        self.assertAlmostEqual(overlap["equal_module_mean_absolute_cosine"], 1)

    def test_subspace_angles_and_energy_known_answer(self):
        a = np.diag([2., 0., 0.])
        b = np.diag([0., 3., 0.])
        overlap = subspace_overlap(a, b)
        for space in ("input_space", "output_space"):
            self.assertAlmostEqual(overlap[space]["principal_angles_radians"][0], np.pi / 2)
            self.assertEqual(overlap[space]["a_energy_in_b"], 0)
            self.assertEqual(overlap[space]["rank_local_overlap"], 0)
        same = subspace_overlap(a, -7 * a)["output_space"]
        self.assertAlmostEqual(same["rank_local_overlap"], 1)
        self.assertAlmostEqual(same["a_energy_in_b"], 1)
        self.assertEqual(same["rank_a"], 1)

    def test_zero_overlap_is_undefined_not_false_success(self):
        zero = {"m": np.zeros((3, 2))}
        self.assertIsNone(parameter_overlap(zero, zero)["global_cosine"])
        self.assertEqual(parameter_overlap(zero, zero)["undefined_modules"], ["m"])
        self.assertIsNone(subspace_overlap(zero["m"], zero["m"])["input_space"]["rank_local_overlap"])
        self.assertIsNone(activation_overlap(zero["m"], zero["m"])["centered_linear_cka"])

    def test_activation_overlap_rotations_differ_from_raw_cosine(self):
        a = np.array([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]])
        b = a @ np.array([[0., -1.], [1., 0.]])
        report = activation_overlap(a, b)
        self.assertAlmostEqual(report["raw_residual_cosine"], 0)
        self.assertAlmostEqual(report["centered_linear_cka"], 1)
        self.assertFalse(report["functional_independence_established"])
        with self.assertRaises(ValueError):
            activation_overlap(a, b[:2])

    def test_norm_and_parameter_report(self):
        d = {"m": np.array([[3., 4.], [0., 0.]])}
        report = delta_report(d, {"m": 2 * d["m"]})
        self.assertEqual(report["l2"], 5)
        self.assertEqual(report["relative_l2_to_base"], 0.5)
        self.assertEqual(report["target_parameter_count"], 4)
        self.assertEqual(report["nonzero_parameter_count"], 2)
        self.assertEqual(report["modules"]["m"]["effective_rank"], 1)
        self.assertEqual(report["modules"]["m"]["minimal_rank_factor_parameter_count"], 4)


if __name__ == "__main__":
    unittest.main()
