"""Synthetic positive/adversarial tests; no Rineya checkpoints or benchmark rows."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from efe_reference import (
    audit_splits, fit_coefficients, implicit_adjoint, maximum_decode_only_saving,
    select_positions, smoke, solve_fixed_point,
)

HERE = Path(__file__).resolve().parent


class CacheTests(unittest.TestCase):
    def select(self, **changes):
        args = dict(positions=tuple(range(20)), prompt_length=4, next_position=20,
                    budget=10, recent=2, seed=17)
        args.update(changes)
        return select_positions(**args)

    def test_protects_prompt_and_recent(self):
        kept = self.select()
        self.assertTrue({0, 1, 2, 3, 18, 19}.issubset(kept))
        self.assertEqual(len(kept), 10)
        self.assertEqual(kept, tuple(sorted(set(kept))))

    def test_deterministic(self):
        self.assertEqual(self.select(), self.select())

    def test_head_streams_can_differ(self):
        self.assertGreater(len({self.select(kv_head=h) for h in range(8)}), 1)

    def test_recency_control(self):
        self.assertEqual(self.select(selector="recency"), (0, 1, 2, 3, 14, 15, 16, 17, 18, 19))

    def test_score_control(self):
        kept = self.select(selector="score", scores={p: -float(p) for p in range(20)})
        self.assertEqual(kept, (0, 1, 2, 3, 4, 5, 6, 7, 18, 19))

    def test_score_requires_finite_complete_map(self):
        for scores in (None, {}, {p: float("nan") for p in range(20)}):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                self.select(selector="score", scores=scores)

    def test_total_budget_includes_prompt(self):
        with self.assertRaises(ValueError):
            self.select(budget=5)

    def test_missing_protected_positions_rejected(self):
        for omitted in (0, 19):
            with self.subTest(omitted=omitted), self.assertRaises(ValueError):
                self.select(positions=tuple(p for p in range(20) if p != omitted))

    def test_does_not_restore_evicted_positions(self):
        resident = self.select()
        new = select_positions((*resident, 20), prompt_length=4, next_position=21,
                               budget=10, recent=2)
        self.assertTrue(set(new).issubset(set(resident) | {20}))

    def test_unprotected_ablation(self):
        self.assertEqual(self.select(selector="recency", protect_prompt=False), tuple(range(10, 20)))

    def test_no_eviction_with_large_budget(self):
        self.assertEqual(self.select(budget=20), tuple(range(20)))

    def test_invalid_inputs(self):
        cases = ({"budget": 0}, {"recent": -1}, {"seed": True}, {"selector": "oracle"},
                 {"positions": (0, 0)}, {"positions": (20,)}, {"next_position": 3})
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.select(**change)

    def test_decode_only_bound(self):
        self.assertAlmostEqual(maximum_decode_only_saving(4096, 256), 1 / 17)
        self.assertEqual(maximum_decode_only_saving(10, 0), 0)
        with self.assertRaises(ValueError):
            maximum_decode_only_saving(0, 0)


class FunctionalTests(unittest.TestCase):
    def test_exact_existing_basis(self):
        result = fit_coefficients(np.eye(2), [-0.6, 0.2], np.zeros((0, 2)), ridge=0)
        np.testing.assert_allclose(result["coefficients"], [0.6, -0.2])
        self.assertLess(result["relative_span_residual"], 1e-12)

    def test_missing_direction_residual(self):
        result = fit_coefficients([[1], [0]], [0, 1], np.zeros((0, 1)), ridge=0)
        self.assertAlmostEqual(result["relative_span_residual"], 1)

    def test_preservation_tradeoff_not_missing_capacity(self):
        result = fit_coefficients([[1]], [-1], [[1]], preservation_weight=9, ridge=0)
        self.assertAlmostEqual(result["coefficients"][0], 0.1)
        self.assertAlmostEqual(result["unconstrained_target_rmse"], 0)
        self.assertAlmostEqual(result["fitted_target_rmse"], 0.9)

    def test_rank_deficient_basis_supported(self):
        result = fit_coefficients([[1, 1], [2, 2]], [-1, -2], np.zeros((0, 2)), ridge=0)
        self.assertEqual(result["basis_rank"], 1)
        np.testing.assert_allclose(result["coefficients"], [0.5, 0.5])

    def test_zero_target_is_undefined_ratio(self):
        result = fit_coefficients([[1]], [0], [[0]])
        self.assertIsNone(result["relative_span_residual"])
        self.assertEqual(result["coefficients"], [0.0])

    def test_ridge_attenuates(self):
        result = fit_coefficients([[1]], [-1], [[0]], ridge=1)
        self.assertAlmostEqual(result["coefficients"][0], 0.5)

    def test_invalid_shapes_and_nonfinite_values(self):
        cases = (([], [], [[]]), ([[1]], [1, 2], [[1]]),
                 ([[1]], [1], [[1, 2]]), ([[float("nan")]], [1], [[1]]))
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                fit_coefficients(*args)

    def test_invalid_hyperparameters(self):
        for args in ({"ridge": -1}, {"preservation_weight": float("inf")}, {"step_size": 0}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                fit_coefficients([[1]], [-1], [[0]], **args)

    def test_weight_orthogonality_does_not_imply_output_orthogonality(self):
        jacobian = np.array([[1.0, 1.0]])
        d1, d2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        self.assertEqual(float(d1 @ d2), 0)
        self.assertEqual(float((jacobian @ d1) @ (jacobian @ d2)), 1)


class EquilibriumTests(unittest.TestCase):
    def test_affine_contraction(self):
        result = solve_fixed_point(lambda z: 0.5 * z + 1, [0.0], atol=1e-10, rtol=0)
        self.assertTrue(result.numerically_converged)
        np.testing.assert_allclose(result.state, [2], atol=3e-10)

    def test_small_damping_cannot_fake_convergence(self):
        result = solve_fixed_point(lambda z: z + 1, [0.0], damping=1e-12, max_steps=2)
        self.assertFalse(result.numerically_converged)
        self.assertAlmostEqual(result.undamped_residual, 1)
        self.assertLess(result.damped_residual, 1e-10)

    def test_last_residual_is_at_returned_state(self):
        result = solve_fixed_point(lambda z: 0.5 * z + 1, [0.0], max_steps=1)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.function_evaluations, 2)
        self.assertAlmostEqual(result.undamped_residual, 0.5)
        self.assertFalse(result.numerically_converged)

    def test_identity_is_not_valid_implicit_system(self):
        self.assertTrue(solve_fixed_point(lambda z: z, [0.0]).numerically_converged)
        with self.assertRaises(ValueError):
            implicit_adjoint([[1]], [1])

    def test_implicit_adjoint_matches_finite_difference(self):
        j = np.array([[0.2, 0.1], [0.0, 0.3]])
        b = np.array([0.4, -0.2])
        g = np.array([0.7, 0.9])
        def loss(offset):
            return float(g @ np.linalg.solve(np.eye(2) - j, offset))
        eps = 1e-6
        finite = [(loss(b + eps * e) - loss(b - eps * e)) / (2 * eps) for e in np.eye(2)]
        np.testing.assert_allclose(implicit_adjoint(j, g), finite, rtol=1e-7)

    def test_ill_conditioned_system_rejected(self):
        with self.assertRaises(ValueError):
            implicit_adjoint([[1 - 1e-12, 0], [0, 0]], [1, 1])

    def test_bad_map_rejected(self):
        for function in (lambda z: [float("nan")], lambda z: [1, 2]):
            with self.subTest(function=function), self.assertRaises(ValueError):
                solve_fixed_point(function, [0.0])

    def test_bad_solver_settings_rejected(self):
        for args in ({"damping": 0}, {"damping": 2}, {"max_steps": 0}, {"atol": 0, "rtol": 0}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                solve_fixed_point(lambda z: z, [0.0], **args)


class SplitTests(unittest.TestCase):
    def groups(self):
        return {role: [{"row_id": f"row-{i}", "parent_id": f"family-{i}",
                        "content_sha256": str(i) * 64, "spent": False}]
                for i, role in enumerate(("calibration", "preservation", "confirmation"))}

    def test_disjoint_declarations(self):
        audit_splits(self.groups())

    def test_cross_role_overlap_rejected(self):
        for key in ("row_id", "parent_id", "content_sha256"):
            groups = self.groups()
            groups["confirmation"][0][key] = groups["calibration"][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_splits(groups)

    def test_spent_or_unspecified_rejected(self):
        for spent in (True, None, 0):
            groups = self.groups()
            groups["calibration"][0]["spent"] = spent
            with self.subTest(spent=spent), self.assertRaises(ValueError):
                audit_splits(groups)

    def test_missing_role_or_empty_role_rejected(self):
        groups = self.groups()
        groups.pop("confirmation")
        with self.assertRaises(ValueError):
            audit_splits(groups)
        groups = self.groups()
        groups["confirmation"] = []
        with self.assertRaises(ValueError):
            audit_splits(groups)

    def test_duplicate_row_rejected(self):
        groups = self.groups()
        groups["calibration"].append(copy.deepcopy(groups["calibration"][0]))
        with self.assertRaises(ValueError):
            audit_splits(groups)

    def test_invalid_hash_rejected(self):
        groups = self.groups()
        groups["calibration"][0]["content_sha256"] = "not-a-hash"
        with self.assertRaises(ValueError):
            audit_splits(groups)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(HERE / "efe_reference.py"), *args],
                              capture_output=True, text=True, check=False)

    def test_smoke_is_explicitly_synthetic(self):
        result = smoke()
        self.assertEqual(result["native_model_loads"], 0)
        self.assertEqual(result["native_training_steps"], 0)
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["measurement_kind"], "synthetic_cpu_mechanics")

    def test_smoke_writes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            self.assertEqual(self.run_cli("smoke", "--out", str(output)).returncode, 0)
            original = output.read_bytes()
            self.assertEqual(json.loads(original)["experiment"], "EFE-001")
            self.assertEqual(self.run_cli("smoke", "--out", str(output)).returncode, 2)
            self.assertEqual(output.read_bytes(), original)

    def test_native_is_blocked(self):
        result = self.run_cli("native")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "BLOCKED")

    def test_plan_is_parseable(self):
        result = self.run_cli("plan")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["experiment"], "EFE-001")


if __name__ == "__main__":
    unittest.main()
