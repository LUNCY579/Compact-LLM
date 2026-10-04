"""Separate native observations, dense arithmetic simulations, and return-cast floors."""
from __future__ import annotations

import torch

from .backends import environment, native_vjp, precision_guard
from .core import Case, emulate, error_metrics, reference, witness


def audit_case(case: Case, backend: str = "math") -> dict:
    case.validate()
    with precision_guard(), torch.autocast(case.q.device.type, enabled=False):
        ref = reference(case)
        variants = {"native": native_vjp(case, backend),
                    "fp32_attention": native_vjp(case, "math", True)}
        for mode in ("cast_only", "gproj_q", "gproj"):
            result = emulate(case, mode=mode)
            variants["dense_" + mode] = dict(result, status="DENSE_EMULATION",
                                             backend_requested=None, backend_events=[])
        variants["dense_fp64_center_gproj"] = dict(
            emulate(case, center="exact", accumulation_dtype=torch.float64, round_key_operand=False),
            status="DENSE_IDEALIZED_CAST_EXPERIMENT")
    report = {"metadata": case.metadata, "shape_q": list(case.q.shape),
              "shape_k": list(case.k.shape), "dtype": str(case.q.dtype),
              "device": str(case.q.device), "causal_convention": "pytorch_upper_left",
              "scale": case.alpha, "variants": {}}
    for name, result in variants.items():
        entry = {key: value for key, value in result.items() if not isinstance(value, torch.Tensor)}
        if "dq" in result:
            entry["gradients"] = {}
            for grad in ("dq", "dk", "dv"):
                returned = result[grad].to(case.q.dtype)
                entry["gradients"][grad] = {
                    "returned_vs_fp64": error_metrics(returned, ref[grad]),
                    "returned_vs_rounded_reference": error_metrics(returned, ref[grad].to(case.q.dtype)),
                }
                if result["status"].startswith("DENSE"):
                    entry["gradients"][grad]["accumulator_vs_fp64"] = error_metrics(result[grad], ref[grad])
            entry["forward_vs_fp64"] = error_metrics(result["output"], ref["output"])
        if "rho" in result:
            entry["max_abs_row_mass"] = float(result["rho"].abs().max())
            entry["max_abs_projected_mass"] = float(result["projected_mass"].abs().max())
            entry["max_probability_mass_defect"] = float((result["mass"] - 1).abs().max())
        report["variants"][name] = entry
    return report


def metric_passes(metric: dict, relative: float, absolute: float) -> bool:
    if not metric.get("finite"):
        return False
    if metric.get("absolute_l2", float("inf")) <= absolute:
        return True
    rel = metric.get("relative_l2")
    return rel is not None and rel <= relative


def numerical_gate(audits: list[dict], relative: float, absolute: float) -> bool:
    if not audits:
        return False
    for audit in audits:
        if audit["variants"].get("native", {}).get("status") != "MEASURED":
            return False
        for arm in ("fp32_attention", "dense_gproj"):
            result = audit["variants"].get(arm, {})
            if "gradients" not in result:
                return False
            for grad in ("dq", "dk", "dv"):
                if not metric_passes(result["gradients"][grad]["returned_vs_fp64"], relative, absolute):
                    return False
    return True


def synthetic_report() -> dict:
    sweep = []
    for width in (2, 64, 128):
        for offset in (0.0, 64.0, 1024.0, 65536.0):
            case = witness(width, offset)
            ref, raw, fixed = reference(case), emulate(case, "cast_only"), emulate(case)
            sweep.append({"width": width, "offset": offset,
                          "reference_dq0": float(ref["dq"][0, 0, 0, 0]),
                          "cast_only_dq0": float(raw["dq"][0, 0, 0, 0]),
                          "gproj_dq0": float(fixed["dq"][0, 0, 0, 0])})
    return {"status": "SYNTHETIC_ONLY", "native_qualified": False, "runtime": environment(),
            "witness_sweep": sweep, "witness_audit": audit_case(witness()),
            "claims": {"rineya_trained": False, "retention_improved": False,
                       "fused_gproj_implemented": False, "speedup_established": False}}
