"""Receipt-gated integration of pure tensor mechanics; never loads a model."""
import numpy as np

from . import mechanics as m
from . import receipts as r


def fit_captured_residuals(calibration, captures, expected, *, forbidden_row_ids, forbidden_parent_ids):
    """One preregistered layer pair, all capture rows in calibration order.

    Capture tensors use one key, 'residual'; native readers must verify original
    file bytes before constructing this in-memory object. Numeric tensor digests
    then bind the exact values operated on. Caller supplies frozen expected receipts.
    """
    r.validate_calibration(calibration, forbidden_row_ids, forbidden_parent_ids)
    roles = ("student_base", "donor_base", "donor_skilled")
    r.require(set(captures) == set(expected) == set(roles), "capture roles differ")
    ids = [row["id"] for row in calibration["rows"]]
    for role in roles:
        item, exp = captures[role], expected[role]
        r.validate_capture(item["metadata"], exp)
        r.require(exp["role"] == role, "capture role binding differs")
        r.require(item["metadata"].get("row_ids") == ids, "capture row order differs")
        r.require(item["metadata"]["calibration_sha256"] == calibration["calibration_sha256"],
                  "capture uses another calibration")
        r.require(item["metadata"]["dataset_sha256"] == calibration["dataset_sha256"],
                  "capture uses another dataset")
        r.valid_hash(exp.get("tensor_sha256"), "capture tensor")
        r.require(m.tensor_digest(item["tensors"]) == exp["tensor_sha256"], "capture tensor bytes differ")
        r.require(set(item["tensors"]) == {"residual"}, "one capture layer per fit required")
        array = np.asarray(item["tensors"]["residual"])
        r.require(array.dtype == np.dtype("float32"), "capture payload precision differs from FP32 contract")
        r.require(array.ndim == 2 and array.shape[0] == len(ids), "capture dimensions differ")
        r.require(list(array.shape) == exp.get("shape"), "capture width differs")
        r.require(len(item["metadata"]["layer_ids"]) == 1, "one capture layer per fit required")
    for key in ("tokenizer_sha256", "dataset_sha256", "calibration_sha256", "layer_map_sha256",
                "prompt_positions_sha256", "implementation_digest"):
        r.require(len({captures[role]["metadata"][key] for role in roles}) == 1, "unaligned captures: " + key)
    donor_meta = captures["donor_base"]["metadata"]
    skilled_meta = captures["donor_skilled"]["metadata"]
    r.require(donor_meta.get("runtime_binding_sha256") == skilled_meta.get("runtime_binding_sha256"),
              "donor base/skilled numerical runtime differs")
    r.valid_hash(donor_meta.get("runtime_binding_sha256"), "donor numerical runtime")
    r.require(skilled_meta.get("anchor_checkpoint_sha256") == donor_meta["model_sha256"],
              "skilled donor has wrong base anchor")
    r.require(skilled_meta["layer_ids"] == donor_meta["layer_ids"], "donor base/skilled capture layer differs")
    r.require(skilled_meta["model_sha256"] != donor_meta["model_sha256"], "donor skilled identity equals base")
    x = captures["donor_base"]["tensors"]["residual"]
    y = captures["student_base"]["tensors"]["residual"]
    skilled = captures["donor_skilled"]["tensors"]["residual"]
    r.require(x.shape == skilled.shape, "donor skilled/base dimensions differ")
    r.require(bool(np.any(skilled != x)), "zero donor skill residual")
    fit_ids = np.array([row["split"] == "fit" for row in calibration["rows"]])
    model = m.fit_transport(x[fit_ids], y[fit_ids])
    r.require(model.rank > 0, "degenerate rank-zero transport")
    residual = model.residual(skilled - x)
    r.require(bool(np.any(residual != 0)), "transport erased all donor skill residual")
    diag = m.transport_diagnostics(model, x[~fit_ids], y[~fit_ids])
    mean_baseline_mse = float(np.mean((y[~fit_ids] - y[fit_ids].mean(0)) ** 2))
    receipt = {"status": "MECHANICALLY_VALIDATED", "calibration_sha256": calibration["calibration_sha256"],
               "capture_digests": {role: expected[role]["tensor_sha256"] for role in roles},
               "transport_sha256": m.tensor_digest({"weight": model.weight, "bias": model.bias}),
               "target_residual_sha256": m.tensor_digest({"residual": residual}),
               "fit_rank": model.rank, "fit_rows": int(fit_ids.sum()), "check_rows": int((~fit_ids).sum()),
               "heldout": diag, "frozen_fit_mean_baseline_mse": mean_baseline_mse,
               "stage_c_continuation_pass": diag["mse"] < mean_baseline_mse,
               "scientific_improvement_claim": False}
    return model, residual, receipt


def _bound_delta(item, expected):
    r.validate_arm(item["metadata"], expected)
    for key in r.COMPOSITION_KEYS:
        r.require(key in expected and item["metadata"].get(key) == expected[key],
                  "delta " + key + " differs from frozen binding")
    r.validate_modules(expected["target_modules"])
    r.require(sorted(item["tensors"]) == expected["target_modules"], "delta tensor keys differ from target modules")
    r.require(m.tensor_digest(item["tensors"]) == expected["skill_delta_sha256"], "wrong delta tensor receipt")


def compose_qualified(left, right, left_expected, right_expected):
    for item, expected in ((left, left_expected), (right, right_expected)):
        _bound_delta(item, expected)
    r.validate_composition(left["metadata"], right["metadata"])
    result = m.compose_deltas(left["tensors"], right["tensors"])
    return result, {"status": "MECHANICALLY_VALIDATED", "sum_sha256": m.tensor_digest(result),
                    "components": [m.delta_report(left["tensors"]), m.delta_report(right["tensors"])],
                    "total": m.delta_report(result), "overlap": m.parameter_overlap(left["tensors"], right["tensors"]),
                    "scientific_improvement_claim": False}


def project_qualified(delta, delta_expected, protected, protected_expected, base_tensors, *, module_layers=None):
    """Project one bound parameter delta against one measured protected direction.

    Retention-sensitive deltas need not be independent skills. Activation-space
    bases require a separately typed mechanism and are not silently reinterpreted
    as parameter tensors here. Expected receipts are trusted frozen registry data;
    this wrapper checks their consistency and the exact numeric inputs.
    """
    for item, expected in ((delta, delta_expected), (protected, protected_expected)):
        _bound_delta(item, expected)
    for key in ("base_checkpoint_sha256", "tokenizer_sha256", "model_binding_sha256", "target_modules"):
        r.require(delta["metadata"][key] == protected["metadata"][key], "incompatible projection " + key)
    for expected in (delta_expected, protected_expected):
        r.valid_hash(expected.get("base_tensors_sha256"), "projection base tensor digest")
        r.require(m.tensor_digest(base_tensors) == expected["base_tensors_sha256"], "wrong projection base tensors")
    basis = protected.get("basis")
    r.require(isinstance(basis, dict) and basis == protected_expected.get("basis_receipt"),
              "protected basis receipt differs from frozen binding")
    r.validate_basis(basis, delta_expected["basis_run_id"], delta_expected["basis_sha256"])
    r.require(basis["run_id"] == protected["metadata"]["training_run_id"], "basis source run differs")
    r.require(basis["sha256"] == m.tensor_digest(protected["tensors"]), "wrong protected basis tensor digest")
    r.require(basis["scope"] == delta_expected["target_modules"], "projection basis module scope differs")
    r.require(basis["source_kind"] in ("independent_skill_delta", "retention_sensitive_delta"),
              "parameter projection cannot consume activation basis")
    r.require(basis["normalization"] == "base_module_relative_l2_equal_layer_v1", "wrong basis metric")
    if basis["source_kind"] == "independent_skill_delta":
        r.require(protected["metadata"]["independent_training"] is True, "protected skill lacks independent training")
    groups = {key: key for key in delta_expected["target_modules"]} if module_layers is None else dict(module_layers)
    expected_groups = protected_expected.get("module_layers", {key: key for key in delta_expected["target_modules"]})
    r.require(groups == expected_groups, "projection layer weighting differs from frozen binding")
    projected, removed, receipt = m.project_delta(delta["tensors"], [protected["tensors"]], base_tensors,
                                                  module_layers=groups)
    r.require(receipt["basis_rank"] == basis["rank"], "actual protected basis rank differs")
    receipt.update(status="MECHANICALLY_VALIDATED", protected_basis_receipt_sha256=r.digest(basis),
                   base_tensors_sha256=delta_expected["base_tensors_sha256"],
                   original_delta_sha256=m.tensor_digest(delta["tensors"]),
                   projected_delta_sha256=m.tensor_digest(projected),
                   removed_delta_sha256=m.tensor_digest(removed), scientific_improvement_claim=False)
    return projected, removed, receipt
