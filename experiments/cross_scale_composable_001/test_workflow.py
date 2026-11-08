"""Integration negatives bind the actual operated-on values to frozen receipts."""
import copy
import unittest

import numpy as np

from . import mechanics as m
from . import receipts as r
from . import workflow as w
from .test_receipts import arm


def capture_fixture():
    rng = np.random.default_rng(123)
    rows = [{"id": f"row-{i}", "parent_id": f"parent-{i}", "open": True,
             "split": "fit" if i < 6 else "map_check"} for i in range(8)]
    calibration = {"dataset_sha256": r.digest("fixture-open-dataset"),
                   "calibration_sha256": r.digest(rows), "rows": rows, "role": "open_calibration"}
    donor = rng.normal(size=(8, 3)).astype("float32")
    student = (donor @ rng.normal(size=(3, 2)) + 2).astype("float32")
    skilled = (donor + .01 * rng.normal(size=donor.shape)).astype("float32")
    captures, expected = {}, {}
    donor_hash = r.digest("fixture-donor-base")
    for role, values, layer in (("student_base", student, 13), ("donor_base", donor, 17),
                                ("donor_skilled", skilled, 17)):
        meta = {key: r.digest(key) for key in r.CAPTURE_KEYS}
        model_hash = donor_hash if role == "donor_base" else r.digest(role)
        meta.update(role=role, model_sha256=model_hash,
                    anchor_checkpoint_sha256=donor_hash if role == "donor_skilled" else model_hash,
                    calibration_sha256=calibration["calibration_sha256"],
                    dataset_sha256=calibration["dataset_sha256"], layer_ids=[layer],
                    pooling="last_nonpad_prompt_token", eval_mode=True, optimization_steps=0,
                    dtype="float32", padding_excluded=True, row_ids=[row["id"] for row in rows],
                    shape=list(values.shape), tensor_sha256=m.tensor_digest({"residual": values}))
        captures[role] = {"metadata": meta, "tensors": {"residual": values}}
        expected[role] = copy.deepcopy(meta)
    return calibration, captures, expected


def direction_fixture():
    rng = np.random.default_rng(1729)
    base = {"layer0.q": rng.normal(size=(3, 3)), "layer1.v": rng.normal(size=(3, 3))}
    a = {key: rng.normal(size=value.shape) * .02 for key, value in base.items()}
    b = {key: rng.normal(size=value.shape) * .02 for key, value in base.items()}
    def item(tensors, skill):
        metadata = {**arm(), "skill_delta_sha256": m.tensor_digest(tensors),
                    "target_modules": sorted(tensors), "skill_id": skill,
                    "training_run_id": "draw-" + skill, "independent_training": True}
        return {"metadata": metadata, "tensors": tensors}
    left, right = item(a, "A"), item(b, "B")
    basis = {"sha256": m.tensor_digest(b), "run_id": "draw-B", "rank": 1,
             "scope": sorted(b), "normalization": "base_module_relative_l2_equal_layer_v1",
             "source_kind": "independent_skill_delta"}
    right["basis"] = basis
    left["metadata"].update(basis_sha256=basis["sha256"], basis_run_id=basis["run_id"])
    left_expected, right_expected = copy.deepcopy(left["metadata"]), copy.deepcopy(right["metadata"])
    left_expected["base_tensors_sha256"] = right_expected["base_tensors_sha256"] = m.tensor_digest(base)
    right_expected["basis_receipt"] = copy.deepcopy(basis)
    return left, right, left_expected, right_expected, base


class WorkflowTests(unittest.TestCase):
    def fit(self, calibration, captures, expected, **kwargs):
        return w.fit_captured_residuals(calibration, captures, expected,
                                       forbidden_row_ids=kwargs.get("forbidden_row_ids", set()),
                                       forbidden_parent_ids=kwargs.get("forbidden_parent_ids", set()))

    def test_planted_transport_integration(self):
        _, target, receipt = self.fit(*capture_fixture())
        self.assertEqual(target.shape, (8, 2))
        self.assertTrue(receipt["stage_c_continuation_pass"])
        self.assertFalse(receipt["scientific_improvement_claim"])

    def test_runtime_and_anchor_must_match_frozen_expected(self):
        for roles, field in ((("donor_base", "donor_skilled"), "runtime_binding_sha256"),
                             (("student_base",), "runtime_binding_sha256"),
                             (("donor_skilled",), "anchor_checkpoint_sha256")):
            with self.subTest(roles=roles, field=field):
                calibration, captures, expected = capture_fixture()
                for role in roles:
                    captures[role]["metadata"][field] = r.digest("changed-together")
                with self.assertRaises(r.Blocked):
                    self.fit(calibration, captures, expected)

    def test_capture_payload_scope_precision_and_order(self):
        for mutation in ("precision", "order", "layer", "values"):
            with self.subTest(mutation=mutation):
                calibration, captures, expected = capture_fixture()
                item = captures["donor_skilled"]
                if mutation == "precision":
                    # Canonical numeric digest is intentionally unchanged; the dtype gate must catch it.
                    item["tensors"]["residual"] = item["tensors"]["residual"].astype("float64")
                elif mutation == "order":
                    item["metadata"]["row_ids"] = list(reversed(item["metadata"]["row_ids"]))
                elif mutation == "layer":
                    item["metadata"]["layer_ids"] = [35]
                    expected["donor_skilled"]["layer_ids"] = [35]
                else:
                    item["tensors"]["residual"][0, 0] += 1
                with self.assertRaises(r.Blocked):
                    self.fit(calibration, captures, expected)

    def test_zero_skill_and_sealed_parent_refused(self):
        calibration, captures, expected = capture_fixture()
        with self.assertRaises(r.Blocked):
            self.fit(calibration, captures, expected, forbidden_parent_ids={"parent-1"})
        captures["donor_skilled"]["tensors"]["residual"] = captures["donor_base"]["tensors"]["residual"].copy()
        expected["donor_skilled"]["tensor_sha256"] = m.tensor_digest(captures["donor_skilled"]["tensors"])
        with self.assertRaisesRegex(r.Blocked, "zero donor skill residual"):
            self.fit(calibration, captures, expected)

    def test_composition_binds_scope_skill_and_independence(self):
        left, right, le, re, _ = direction_fixture()
        result, receipt = w.compose_qualified(left, right, le, re)
        self.assertEqual(m.tensor_digest(result), receipt["sum_sha256"])
        for key, changed in (("skill_id", "unregistered"), ("independent_training", False),
                             ("target_modules", ["unregistered.module"])):
            with self.subTest(key=key):
                bad = copy.deepcopy(left)
                bad["metadata"][key] = changed
                with self.assertRaises(r.Blocked):
                    w.compose_qualified(bad, right, le, re)
        bad = copy.deepcopy(left)
        bad["tensors"]["outside.scope"] = np.ones((3, 3))
        with self.assertRaisesRegex(r.Blocked, "tensor keys"):
            w.compose_qualified(bad, right, le, re)

    def test_projection_bound_actual_tensors_and_receipts(self):
        delta, protected, de, pe, base = direction_fixture()
        projected, removed, receipt = w.project_qualified(delta, de, protected, pe, base)
        self.assertEqual(receipt["basis_rank"], 1)
        self.assertEqual(receipt["protected_basis_receipt_sha256"], r.digest(protected["basis"]))
        for key in projected:
            np.testing.assert_allclose(projected[key] + removed[key], delta["tensors"][key])
        # A different protected payload cannot become this basis by updating its delta sidecar.
        bad = copy.deepcopy(protected)
        bad["tensors"]["layer0.q"][0, 0] += 1
        bad_hash = m.tensor_digest(bad["tensors"])
        bad["metadata"]["skill_delta_sha256"] = bad_hash
        changed_pe = {**pe, "skill_delta_sha256": bad_hash}
        with self.assertRaisesRegex(r.Blocked, "protected basis tensor"):
            w.project_qualified(delta, de, bad, changed_pe, base)

    def test_projection_rejects_base_weighting_rank_and_other_run(self):
        for mutation in ("base", "weighting", "rank", "other_run"):
            with self.subTest(mutation=mutation):
                delta, protected, de, pe, base = direction_fixture()
                kwargs = {}
                if mutation == "base":
                    base["layer0.q"][0, 0] += .1
                elif mutation == "weighting":
                    kwargs["module_layers"] = {key: "all-one-layer" for key in base}
                elif mutation == "rank":
                    protected["basis"]["rank"] = 2
                    pe["basis_receipt"]["rank"] = 2
                else:
                    protected["basis"]["run_id"] = "other-run"
                    pe["basis_receipt"]["run_id"] = "other-run"
                with self.assertRaises(r.Blocked):
                    w.project_qualified(delta, de, protected, pe, base, **kwargs)

    def test_retention_delta_basis_does_not_require_distinct_skill(self):
        delta, protected, de, pe, base = direction_fixture()
        protected["metadata"]["skill_id"] = delta["metadata"]["skill_id"]
        protected["metadata"]["independent_training"] = False
        pe.update(skill_id=protected["metadata"]["skill_id"], independent_training=False)
        protected["basis"]["source_kind"] = "retention_sensitive_delta"
        pe["basis_receipt"] = copy.deepcopy(protected["basis"])
        _, _, receipt = w.project_qualified(delta, de, protected, pe, base)
        self.assertFalse(receipt["scientific_improvement_claim"])


if __name__ == "__main__":
    unittest.main()
