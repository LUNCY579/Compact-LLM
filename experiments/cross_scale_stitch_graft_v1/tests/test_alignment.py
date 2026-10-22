from dataclasses import replace
import unittest
import torch

from ..alignment import AffineFit, RidgeMoments, fit_bridge, metrics
from ..contracts import Capture, Identity, RowKey, Site, digest, disjoint, paired
from ..fixtures import identity


def bank(role="FIT", width=3, name="a", count=24):
    g = torch.Generator().manual_seed(5 if role == "FIT" else 9)
    rows = tuple(RowKey(f"{role}-{i}", f"{role}-{i // 4}", digest([role, i]), digest([role, i, "ids"]), 0, i)
                 for i in range(count))
    return Capture(identity(name), Site(1, "block_output_raw", width), role, digest("registered-fixture"),
                   rows, torch.randn(count, width, generator=g, dtype=torch.float64))


def affine_banks():
    a, c = bank(), bank("MAP_CHECK")
    w = torch.tensor([[2., 1.], [-1., 3.], [4., -2.]], dtype=torch.float64)
    bias = torch.tensor([0.4, -1.7], dtype=torch.float64)
    b = replace(a, identity=identity("b"), site=Site(1, "block_output_raw", 2), values=a.values @ w + bias)
    d = replace(c, identity=identity("b"), site=b.site, values=c.values @ w + bias)
    return a, b, c, d


class AlignmentTests(unittest.TestCase):
    def test_affine_recovery(self):
        a, b, c, d = affine_banks()
        fit, report = fit_bridge(a, b, c, d, ridge_relative=1e-9)
        self.assertGreater(report["map_check"]["r2"], 0.999999)
        self.assertFalse(report["functional_transfer_established"])
        self.assertEqual(fit.count, 24)

    def test_streamed_matches_full(self):
        a, b, c, d = affine_banks()
        x, _ = fit_bridge(a, b, c, d, chunk_rows=1)
        y, _ = fit_bridge(a, b, c, d, chunk_rows=24)
        torch.testing.assert_close(x.weight, y.weight, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(x.bias, y.bias, rtol=1e-12, atol=1e-12)

    def test_direct_centered_reference(self):
        a, b, c, d = affine_banks()
        fit, _ = fit_bridge(a, b, c, d)
        x, y = a.values - a.values.mean(0), b.values - b.values.mean(0)
        expected = torch.linalg.solve(x.T @ x + fit.ridge_absolute * torch.eye(3, dtype=torch.float64), x.T @ y)
        torch.testing.assert_close(fit.weight, expected.T, rtol=1e-12, atol=1e-12)

    def test_check_data_does_not_change_fit(self):
        a, b, c, d = affine_banks()
        fit, _ = fit_bridge(a, b, c, d)
        other, _ = fit_bridge(a, b, c, replace(d, values=d.values + 10))
        self.assertTrue(torch.equal(fit.weight, other.weight))
        self.assertTrue(torch.equal(fit.bias, other.bias))

    def test_constant_features_regularized(self):
        moments = RidgeMoments(3, 2)
        moments.update(torch.ones(4, 3), torch.full((4, 2), 7.0))
        fit = moments.solve()
        self.assertTrue(torch.equal(fit.weight, torch.zeros(2, 3, dtype=torch.float64)))
        self.assertTrue(torch.equal(fit.bias, torch.full((2,), 7., dtype=torch.float64)))

    def test_delta_drops_bias(self):
        a, b, c, d = affine_banks()
        fit, _ = fit_bridge(a, b, c, d)
        x, delta = a.values[:2], a.values[2:4]
        torch.testing.assert_close(fit(x + delta) - fit(x), fit.delta(delta))

    def test_budget_checked_before_allocate(self):
        with self.assertRaises(ValueError):
            RidgeMoments(4096, 1024, max_workspace_bytes=1024)

    def test_empty_moments_cannot_solve(self):
        with self.assertRaises(ValueError):
            RidgeMoments(2, 2).solve()

    def test_zero_variance_metrics_are_undefined(self):
        m = metrics(torch.zeros(3, 2), torch.zeros(3, 2))
        self.assertIsNone(m["r2"])
        self.assertIsNone(m["relative_rmse"])
        self.assertIsNone(m["mean_cosine"])
        self.assertEqual(m["undefined_cosine_rows"], 3)

    def test_identity_covers_logit_policy(self):
        a = identity()
        self.assertNotEqual(a.key, replace(a, logit_policy="custom_a11_unqualified").key)

    def test_same_role_repeated_capture_allowed(self):
        disjoint((bank(), bank(name="b")))

    def test_site_identity_drift_refused(self):
        a, b, c, d = affine_banks()
        with self.assertRaises(ValueError):
            fit_bridge(a, b, replace(c, site=Site(2, "block_output_raw", 3)), d)

    def test_gradient_capture_refused(self):
        a = bank()
        with self.assertRaises(ValueError):
            replace(a, values=a.values.requires_grad_()).validate()

    def test_double_update_large_offset(self):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(30, 3, generator=g, dtype=torch.float64) + 1e6
        y = x @ torch.ones(3, 2, dtype=torch.float64)
        a, b = RidgeMoments(3, 2), RidgeMoments(3, 2)
        a.update(x, y)
        for chunk in range(0, 30, 5):
            b.update(x[chunk:chunk+5], y[chunk:chunk+5])
        torch.testing.assert_close(a.solve().weight, b.solve().weight, rtol=1e-8, atol=1e-8)


# Separate named tests, not swallowed subtests: exact IDs are frozen in SOURCE_MANIFEST.
def invalid_capture_test(change):
    def test(self):
        with self.assertRaises((ValueError, TypeError)):
            change(bank()).validate()
    return test


for name, change in {
    "bad_role": lambda a: replace(a, role="TEST2"),
    "bad_registration": lambda a: replace(a, registration_sha256="bad"),
    "empty": lambda a: replace(a, rows=(), values=a.values[:0]),
    "shape": lambda a: replace(a, values=a.values[:, :2]),
    "nan": lambda a: replace(a, values=torch.full_like(a.values, float("nan"))),
    "integer": lambda a: replace(a, values=a.values.long()),
    "duplicate": lambda a: replace(a, rows=(a.rows[0],) * len(a.rows)),
    "duplicate_coordinate": lambda a: replace(a, rows=(a.rows[0], replace(a.rows[0], token_id=999)) + a.rows[2:]),
    "inconsistent_example": lambda a: replace(a, rows=(a.rows[0], replace(a.rows[1], example_id=a.rows[0].example_id)) + a.rows[2:]),
}.items():
    setattr(AlignmentTests, f"test_capture_rejects_{name}", invalid_capture_test(change))


def invalid_pair_test(change):
    def test(self):
        a = bank()
        with self.assertRaises(ValueError):
            paired(a, change(a))
    return test


for name, change in {
    "tokenizer": lambda a: replace(a, identity=replace(a.identity, tokenizer_sha256=digest("other"))),
    "template": lambda a: replace(a, identity=replace(a.identity, template_sha256=digest("other"))),
    "fixture_native_mix": lambda a: replace(a, identity=replace(a.identity, fixture=False)),
    "role": lambda a: replace(a, role="CHECK"),
    "registration": lambda a: replace(a, registration_sha256=digest("other")),
    "row_order": lambda a: replace(a, rows=tuple(reversed(a.rows))),
    "input_ids": lambda a: replace(a, rows=(replace(a.rows[0], input_ids_sha256=digest("changed")),) + a.rows[1:]),
}.items():
    setattr(AlignmentTests, f"test_pair_rejects_{name}", invalid_pair_test(change))


def leakage_test(field):
    def test(self):
        a, b = bank(), bank("MAP_CHECK")
        changed = replace(b.rows[0], **{field: getattr(a.rows[0], field)})
        with self.assertRaises(ValueError):
            disjoint((a, replace(b, rows=(changed,) + b.rows[1:])))
    return test


for field in ("example_id", "group_id", "prompt_sha256"):
    setattr(AlignmentTests, f"test_disjoint_rejects_{field}", leakage_test(field))


def penalty_test(value):
    def test(self):
        m = RidgeMoments(2, 2)
        m.update(torch.eye(2), torch.eye(2))
        with self.assertRaises(ValueError):
            m.solve(ridge_relative=value)
    return test


for name, value in (("zero", 0), ("negative", -1), ("nan", float("nan")), ("infinite", float("inf"))):
    setattr(AlignmentTests, "test_penalty_rejects_" + name, penalty_test(value))


class DataDigestTests(unittest.TestCase):
    def test_fit_report_binds_all_four_capture_values(self):
        from ..contracts import capture_digest
        a, b, c, d = affine_banks()
        _, report = fit_bridge(a, b, c, d)
        self.assertEqual(report["data_digests"]["source_fit"], capture_digest(a))
        self.assertNotEqual(capture_digest(a), capture_digest(replace(a, values=a.values + 1)))
        self.assertEqual(len(report["data_digests"]), 4)
