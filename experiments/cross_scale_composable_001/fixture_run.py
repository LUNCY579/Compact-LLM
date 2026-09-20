"""Deterministic synthetic integration receipt. No pretrained execution path."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from . import mechanics as m
from . import receipts as r
from . import workflow as w


def process_memory_receipt():
    """Current process cumulative peak, not an incremental model-memory estimate."""
    receipt = {"available": False, "scope": "current_process_cumulative_peak_after_fixture_algebra",
               "native_stage_memory_estimate": False}
    if os.name != "nt":
        return {**receipt, "reason": "GetProcessMemoryInfo is only available on Windows"}
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {**receipt, "available": True, "provider": "Windows GetProcessMemoryInfo",
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "working_set_bytes": int(counters.WorkingSetSize),
                "peak_pagefile_bytes": int(counters.PeakPagefileUsage)}
    except (AttributeError, OSError) as exc:
        return {**receipt, "reason": str(exc)}


def transport_fixture():
    rng = np.random.default_rng(1729)
    rows = [{"id": f"synthetic-{i}", "parent_id": f"synthetic-parent-{i}", "open": True,
             "split": "fit" if i < 24 else "map_check"} for i in range(32)]
    binding = {"dataset_sha256": r.digest("synthetic-dataset-not-model-data"),
               "calibration_sha256": r.digest(rows), "rows": rows, "role": "open_calibration"}
    donor = rng.normal(size=(32, 8)).astype("float32")
    planted = rng.normal(size=(8, 4))
    student = (donor @ planted + np.arange(4)).astype("float32")
    shift = rng.normal(size=(32, 1)) @ rng.normal(size=(1, 8)) * .02
    skilled = (donor + shift).astype("float32")
    captures, expected = {}, {}
    base_hash = r.digest("synthetic-donor-base")
    for role, values, layer in (("student_base", student, 13), ("donor_base", donor, 17),
                                ("donor_skilled", skilled, 17)):
        model_hash = base_hash if role == "donor_base" else r.digest(role)
        metadata = {key: r.digest(key) for key in r.CAPTURE_KEYS}
        metadata.update(role=role, model_sha256=model_hash,
                        calibration_sha256=binding["calibration_sha256"], dataset_sha256=binding["dataset_sha256"],
                        layer_ids=[layer], pooling="last_nonpad_prompt_token", eval_mode=True,
                        optimization_steps=0, dtype="float32", padding_excluded=True,
                        row_ids=[row["id"] for row in rows], shape=list(values.shape),
                        tensor_sha256=m.tensor_digest({"residual": values}),
                        runtime_binding_sha256=r.digest("synthetic-f32-runtime"),
                        anchor_checkpoint_sha256=base_hash if role == "donor_skilled" else model_hash)
        captures[role] = {"metadata": metadata, "tensors": {"residual": values}}
        expected[role] = copy.deepcopy(metadata)
    return binding, captures, expected


def arm_metadata(tensors, skill):
    meta = {key: r.digest(key) for key in r.COMMON_KEYS + r.ARM_KEYS
            if key.endswith("sha256") or key == "implementation_digest"}
    meta.update(skill_delta_sha256=m.tensor_digest(tensors), basis_run_id="synthetic-protected-A",
                rank_budget=2, parameter_budget=48, optimizer_steps=8, training_run_id="synthetic-draw-" + skill,
                alpha=1., completed=True, target_modules=sorted(tensors), skill_id=skill, independent_training=True)
    return meta


def run():
    binding, captures, expected = transport_fixture()
    fit, target, transport = w.fit_captured_residuals(binding, captures, expected,
                                                   forbidden_row_ids=set(), forbidden_parent_ids=set())
    rng = np.random.default_rng(1729)
    base = {"layer0.q": rng.normal(size=(6, 6)), "layer1.v": rng.normal(size=(6, 6))}
    a = {key: m.lora_delta(rng.normal(size=(2, 6)), rng.normal(size=(6, 2)), 2) * .01 for key in base}
    b = {key: m.lora_delta(rng.normal(size=(2, 6)), rng.normal(size=(6, 2)), 2) * .01 for key in base}
    # Extraction is checked against its actual anchor before using the directions.
    a = m.extract_delta(base, {key: base[key] + a[key] for key in base})
    before = m.tensor_digest(a), m.tensor_digest(b)
    left = {"metadata": arm_metadata(a, "A"), "tensors": a}
    right = {"metadata": arm_metadata(b, "B"), "tensors": b}
    for item in (left, right):
        item["metadata"]["base_checkpoint_sha256"] = m.tensor_digest(base)
    naive, composition = w.compose_qualified(left, right, copy.deepcopy(left["metadata"]), copy.deepcopy(right["metadata"]))

    def registered_projection(item, protected):
        item, protected = copy.deepcopy(item), copy.deepcopy(protected)
        basis = {"sha256": m.tensor_digest(protected["tensors"]),
                 "run_id": protected["metadata"]["training_run_id"], "rank": 1,
                 "scope": sorted(protected["tensors"]),
                 "normalization": "base_module_relative_l2_equal_layer_v1",
                 "source_kind": "independent_skill_delta"}
        protected["basis"] = basis
        item["metadata"].update(basis_sha256=basis["sha256"], basis_run_id=basis["run_id"])
        item_expected, protected_expected = copy.deepcopy(item["metadata"]), copy.deepcopy(protected["metadata"])
        for metadata in (item_expected, protected_expected):
            metadata["base_tensors_sha256"] = m.tensor_digest(base)
        protected_expected["basis_receipt"] = copy.deepcopy(basis)
        return w.project_qualified(item, item_expected, protected, protected_expected, base)

    projected, removed, projection = registered_projection(right, left)
    reverse, reverse_removed, reverse_report = registered_projection(left, right)
    orth_sum = m.compose_deltas(a, projected)
    matched = m.norm_match(projected, m.delta_report(b)["l2"])
    random = m.random_matched_delta(b, seed=1729)
    for key in b:
        np.testing.assert_allclose(projected[key] + removed[key], b[key], atol=1e-12)
    r.require(before == (m.tensor_digest(a), m.tensor_digest(b)), "original deltas mutated")
    mse = float(np.mean((fit.predict(captures["donor_base"]["tensors"]["residual"][24:]) -
                         captures["student_base"]["tensors"]["residual"][24:]) ** 2))
    r.require(mse < 1e-10, "planted transport failed")
    # Dedicated RNGs make each registered control independent of fixture edits.
    permutation = np.random.default_rng(1729).permutation(24)
    shuffled = np.array(captures["donor_base"]["tensors"]["residual"][:24][permutation], copy=True)
    shuffled_fit = m.fit_transport(shuffled, captures["student_base"]["tensors"]["residual"][:24])
    transfer_control = m.transport_diagnostics(shuffled_fit, captures["donor_base"]["tensors"]["residual"][24:],
                                             captures["student_base"]["tensors"]["residual"][24:])
    transfer_control.update(seed=1729, fit_row_permutation=permutation.tolist(),
                            transport_sha256=m.tensor_digest({"weight": shuffled_fit.weight, "bias": shuffled_fit.bias}))
    donor_change = (captures["donor_skilled"]["tensors"]["residual"] -
                    captures["donor_base"]["tensors"]["residual"])
    donor_norm = m.delta_report({"residual": donor_change})["l2"]
    r.require(donor_norm > 0, "zero synthetic donor residual")
    random_change = m.norm_match({"residual": np.random.default_rng(1729).normal(size=donor_change.shape)},
                                 donor_norm)["residual"]
    random_target = fit.residual(random_change)
    residual_control = {
        "seed": 1729, "matching": "global donor-residual Frobenius norm, all calibration rows",
        "donor_residual_sha256": m.tensor_digest({"residual": donor_change}),
        "random_residual_sha256": m.tensor_digest({"residual": random_change}),
        "random_transported_sha256": m.tensor_digest({"residual": random_target}),
        "donor_l2": donor_norm, "random_donor_l2": m.delta_report({"residual": random_change})["l2"],
        "target_l2": m.delta_report({"residual": target})["l2"],
        "random_target_l2": m.delta_report({"residual": random_target})["l2"],
        "heldout_transported_overlap": m.activation_overlap(target[24:], random_target[24:]),
        "note": "Output norms are reported, not forced equal; transport can change direction-specific magnitude.",
    }
    np.testing.assert_allclose(residual_control["donor_l2"], residual_control["random_donor_l2"], rtol=1e-12)
    transport.update(fit_singular_values=fit.singular_values.tolist(),
                     effective_condition_number=float(fit.singular_values[0] / fit.singular_values[fit.rank - 1])
                     if fit.rank else None,
                     donor_width=fit.weight.shape[0], unidentified_donor_dimensions=fit.weight.shape[0] - fit.rank,
                     student_layer=13, donor_layer=17,
                     fixture_mse_below_fit_mean_baseline=bool(mse < transport["frozen_fit_mean_baseline_mse"]),
                     native_final_pair_continuation_qualified=False)
    r.require(transport["fixture_mse_below_fit_mean_baseline"], "synthetic fit failed frozen mean baseline criterion")
    named_deltas = {"A": a, "B": b, "B_projected": projected, "removed": removed,
                    "A_projected_reverse": reverse, "removed_reverse": reverse_removed,
                    "norm_restored": matched, "random": random}
    delta_receipts = {name: {**m.delta_report(values, base), "skill_delta_sha256": m.tensor_digest(values),
                             "base_checkpoint_sha256": m.tensor_digest(base), "synthetic_only": True}
                      for name, values in named_deltas.items()}
    def control_metadata(values):
        report = m.delta_report(values)
        return {**right["metadata"], "norm": report["l2"],
                "effective_ranks": {key: item["effective_rank"] for key, item in report["modules"].items()}}
    r.validate_norm_control(control_metadata(b), control_metadata(random))
    projected_ranks_match = control_metadata(b)["effective_ranks"] == control_metadata(matched)["effective_ranks"]
    norm_controls = {"random_reference": "B", "random_control_mechanically_matched": True,
                     "projected_norm_restored_matches_B": bool(np.isclose(m.delta_report(matched)["l2"],
                                                                          m.delta_report(b)["l2"], rtol=1e-12)),
                     "projected_effective_ranks_match_B": projected_ranks_match,
                     "projected_rank_comparison_status": "COMPARABLE" if projected_ranks_match else "BLOCKED_ON_COMPARABILITY",
                     "note": "Projection rank growth requires a separately registered equal-rank control."}
    composition.update(component_tensor_sha256=[m.tensor_digest(a), m.tensor_digest(b)],
                       base_checkpoint_sha256=m.tensor_digest(base))
    reverse_sum = m.compose_deltas(b, reverse)
    result = {"schema": 1, "status": "COMPLETE", "completed": True,
            "evidence_kind": "SYNTHETIC_CPU_FIXTURE_ONLY", "seed": 1729,
            "native_model_calls": 0, "optimizer_steps": 0, "scientific_improvement_claim": False,
            "layer_map_donor_to_student": list(m.normalized_depth_map(36, 28)),
            "transport": transport, "shuffle_control": transfer_control,
            "random_donor_residual_control": residual_control, "norm_controls": norm_controls,
            "deltas": delta_receipts,
            "projection": projection, "reverse_projection": reverse_report,
            "composition": composition,
            "orthogonal_composition": {**m.delta_report(orth_sum, base), "sum_sha256": m.tensor_digest(orth_sum),
                                        "component_tensor_sha256": [m.tensor_digest(a), m.tensor_digest(projected)]},
            "reverse_composition": {**m.delta_report(reverse_sum, base), "sum_sha256": m.tensor_digest(reverse_sum),
                                     "component_tensor_sha256": [m.tensor_digest(b), m.tensor_digest(reverse)]},
            "subspace_overlap": {key: m.subspace_overlap(a[key], b[key]) for key in a},
            "activation_overlap": m.activation_overlap(target, target[:, ::-1]),
            "process_memory": process_memory_receipt(),
            "behavioral_endpoints": None,
            "note": "Planted linear synthetic data; no scientific acquisition/retention/composability evidence."}
    # Refuse NaN/Infinity or unsupported types before any evidence write.
    r.canonical_bytes(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    r.write_once(args.out, result)
    print(json.dumps({"status": result["status"], "evidence_kind": result["evidence_kind"],
                      "path": str(args.out.resolve()), "sha256": r.file_digest(args.out)}))


if __name__ == "__main__":
    main()
