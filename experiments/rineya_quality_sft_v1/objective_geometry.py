"""Offline objective-direction mechanics; no model/bank/capture/native-run lane.

Mean whole-loss gradients are the target-count-weighted sum of final-response
and complementary-target mean gradients at the SAME state and intact group.
The complement includes prior history, headers, thinking stub and post-EOS
newline. A global supervision ratio is never substituted for group weights.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import torch

from . import EXPERIMENT
from .compiler import targets, validate_plan
from .contract import external_output
from . import integrity as I

MAX_BLOCK_ELEMENTS = 4_194_304  # Larger than the source tail's 3072x1024 matrices.
CHUNK_ELEMENTS = 262_144
TOLERANCES = {torch.float32: {"rtol": 1e-5, "atol": 1e-7}, torch.float64: {"rtol": 1e-9, "atol": 1e-12}}
GRAM_ROUNDOFF_RTOL = 1e-12


def _positive(value, name):
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ValueError("Positive finite " + name + " required")
    return value


def _weights(response_targets, prompt_targets):
    if (type(response_targets) is not int or type(prompt_targets) is not int
            or response_targets < 1 or prompt_targets < 1 or response_targets + prompt_targets > 10_000_000):
        raise ValueError("Explicit bounded positive response/complementary target counts required for both means")
    whole = response_targets + prompt_targets
    return whole, response_targets / whole, prompt_targets / whole


def describe_plan_geometry(sealedplan):
    """Validate and describe complete group metadata, without forwarding a model."""
    body = validate_plan(sealedplan)
    groups, totals = [], {"whole_targets": 0, "response_targets": 0, "prompt_targets": 0}
    for number, indices in enumerate(body["groups"], 1):
        whole_count = response_count = 0
        for index in indices:
            row = body["train"][index]
            whole, response = targets(row, "whole"), targets(row, "response")
            if (len(whole) != len(response) or any(r != -100 and r != w for w, r in zip(whole[1:], response[1:]))):
                raise ValueError("Response targets must be an exact subset of whole shifted positions")
            whole_count += sum(value != -100 for value in whole[1:])
            response_count += sum(value != -100 for value in response[1:])
        prompt_count = whole_count - response_count
        count, wr, wp = _weights(response_count, prompt_count)
        if count != whole_count or not math.isclose(wr + wp, 1., rel_tol=0., abs_tol=1e-15):
            raise ValueError("Target-count partition differs from whole objective")
        group = {"group": number, "indices": list(indices), "row_ids": [body["train"][i]["id"] for i in indices],
                 "whole_targets": whole_count, "response_targets": response_count, "prompt_targets": prompt_count,
                 "wr": wr, "wp": wp, "whole_to_response_target_ratio": whole_count / response_count}
        groups.append(group)
        for key in totals:
            totals[key] += group[key]
    ids = [row["id"] for row in body["train"]]
    consumed = [identifier for group in groups for identifier in group["row_ids"]]
    if ids != consumed or len(set(consumed)) != len(ids):
        raise ValueError("Every admitted world and frozen group must remain unchanged")
    return {"schema": "RQSFT_OBJECTIVE_PLAN_GEOMETRY_V1", "experiment": EXPERIMENT,
            "plan_sha256": sealedplan["sha256"], "fixture_only": body["kind"] == "fixture",
            "evidence": "PLAN_METADATA_ONLY", "groups": groups, "totals": totals,
            "worlds": len(ids), "ordered_ids_sha256": I.digest(ids), "groups_sha256": I.digest(body["groups"]),
            "all_worlds_preserved": True, "order_and_groups_preserved": True,
            "prompt_target_definition": "Whole-minus-final-response targets, including previous assistants/history/headers/stub/post-EOS newline.",
            "reduction": "Mean supervised-token loss within each intact optimizer group; never a mean of row means.",
            "native_gain": False, "native_model_calls": 0, "native_rqsft_recipe_changed": False,
            "training_authorized": False, "gradient_direction_observed": False}


def _tensor(value):
    tensor = torch.as_tensor(value)
    if (tensor.device.type == "meta" or tensor.layout != torch.strided or tensor.dtype not in TOLERANCES
            or tensor.numel() < 1 or tensor.numel() > MAX_BLOCK_ELEMENTS):
        raise ValueError("Nonempty bounded dense FP32/FP64 gradient blocks required")
    return tensor.detach()


def _cos(dot, left, right):
    return None if left == 0 or right == 0 else max(-1., min(1., dot / (left * right)))


@torch.no_grad()
def gradient_geometry(blocks, response_targets, prompt_targets, clip_norm=1.):
    """Stream detached matching blocks; never retain a full parameter bank.

    Each block contains name/response/prompt/whole. Gradients must be means over
    the stated targets at one identical state/group. Registered dtype tolerances
    fail closed; this function has no native gradient-capture interface.
    """
    clip_norm = _positive(clip_norm, "clip norm")
    whole_targets, wr, wp = _weights(response_targets, prompt_targets)
    seen, dtype, device = set(), None, None
    sums = {"rr": 0., "pp": 0., "ww": 0., "rp": 0., "rw": 0., "pw": 0., "error_squared": 0.}
    block_count = element_count = 0
    maximum_error = 0.
    for block in blocks:
        if not isinstance(block, dict) or set(block) != {"name", "response", "prompt", "whole"}:
            raise ValueError("Exactly one named response/prompt/whole gradient triple required")
        name = block["name"]
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("Unique nonempty block names required")
        seen.add(name)
        r, p, w = (_tensor(block[key]) for key in ("response", "prompt", "whole"))
        if r.shape != p.shape or r.shape != w.shape or r.dtype != p.dtype or r.dtype != w.dtype or r.device != p.device or r.device != w.device:
            raise ValueError("Gradient block shapes/dtypes/devices must match")
        if dtype is None:
            dtype, device = r.dtype, r.device
        elif r.dtype != dtype or r.device != device:
            raise ValueError("One dtype/device contract across all streamed blocks required")
        tolerance = TOLERANCES[dtype]
        rf, pf, wf = (x.reshape(-1) for x in (r, p, w))
        for start in range(0, rf.numel(), CHUNK_ELEMENTS):
            a, b, c = (x[start:start + CHUNK_ELEMENTS].to(dtype=torch.float64) for x in (rf, pf, wf))
            if any(not torch.isfinite(x).all() for x in (a, b, c)):
                raise ValueError("Nonfinite gradient block")
            expected = wr * a + wp * b
            error = c - expected
            # Separate FP32 reductions can differ most when the two signals
            # cancel. Relative tolerance follows the original signal sizes,
            # while a second global relative check prevents tiny bad mixtures
            # from passing solely through the component absolute tolerance.
            allowed = tolerance["atol"] + tolerance["rtol"] * (wr * a.abs() + wp * b.abs())
            if bool((error.abs() > allowed).any()):
                raise ValueError("Whole gradient violates registered weighted response/complement mixture")
            maximum_error = max(maximum_error, float(error.abs().max()))
            sums["rr"] += float(a.square().sum())
            sums["pp"] += float(b.square().sum())
            sums["ww"] += float(c.square().sum())
            sums["rp"] += float((a * b).sum())
            sums["rw"] += float((a * c).sum())
            sums["pw"] += float((b * c).sum())
            sums["error_squared"] += float(error.square().sum())
            if any(not math.isfinite(value) for value in sums.values()):
                raise ValueError("Float64 dot/norm accumulation overflow; no numerical fallback")
        block_count += 1
        element_count += r.numel()
        del r, p, w, rf, pf, wf, a, b, c, expected, error
    if block_count == 0:
        raise ValueError("At least one gradient triple required")
    norms = {"response": math.sqrt(sums["rr"]), "prompt": math.sqrt(sums["pp"]), "whole": math.sqrt(sums["ww"])}
    mixture_error = math.sqrt(sums["error_squared"])
    relative_scale = max(norms["whole"], wr * norms["response"] + wp * norms["prompt"])
    relative_error = mixture_error / relative_scale if relative_scale else None
    if ((relative_scale == 0 and mixture_error != 0)
            or relative_error is not None and relative_error > TOLERANCES[dtype]["rtol"]):
        raise ValueError("Whole gradient violates global signal-scaled relative mixture tolerance")
    gram_raw = wr * wr * sums["rr"] + wp * wp * sums["pp"] + 2 * wr * wp * sums["rp"]
    gram_scale = wr * wr * sums["rr"] + wp * wp * sums["pp"] + 2 * wr * wp * abs(sums["rp"])
    if gram_raw < -GRAM_ROUNDOFF_RTOL * gram_scale:
        raise ValueError("FP64 Gram mixture has an inconsistent negative squared norm")
    clipping = {key: {"scale": min(1., clip_norm / (value + 1e-6)),
                      "clipped_norm": value * min(1., clip_norm / (value + 1e-6))} for key, value in norms.items()}
    denominator = wr * norms["response"] + wp * norms["prompt"]
    return {"schema": "RQSFT_STREAMING_GRADIENT_GEOMETRY_V1", "experiment": EXPERIMENT,
            "evidence": "UNBOUND_GRADIENT_ALGEBRA_MECHANICS_ONLY", "whole_targets": whole_targets,
            "response_targets": response_targets, "prompt_targets": prompt_targets, "wr": wr, "wp": wp,
            "blocks": block_count, "elements": element_count, "dtype": str(dtype), "device": str(device),
            "maximum_block_elements": MAX_BLOCK_ELEMENTS, "chunk_elements": CHUNK_ELEMENTS,
            "mixture_verified": True, "mixture_tolerance": dict(TOLERANCES[dtype]),
            "mixture_maximum_abs_error": maximum_error, "mixture_error_l2": mixture_error,
            "mixture_relative_error": relative_error, "mixture_relative_scale": relative_scale,
            "mixture_tolerance_definition": "Component error<=atol+rtol*(wr*abs(gr)+wp*abs(gp)); global L2error/max(norm(gw),wr*norm(gr)+wp*norm(gp))<=rtol. Exact all-zero case is separately valid.",
            "gram_expected_whole_squared_norm": max(0., gram_raw), "gram_raw_squared_norm": gram_raw,
            "gram_observed_squared_norm_residual": sums["ww"] - gram_raw,
            "gram_roundoff_relative_tolerance": GRAM_ROUNDOFF_RTOL,
            "norms": norms, "squared_norms_and_dots": sums,
            "cosines": {"response_prompt": _cos(sums["rp"], norms["response"], norms["prompt"]),
                        "response_whole": _cos(sums["rw"], norms["response"], norms["whole"]),
                        "prompt_whole": _cos(sums["pw"], norms["prompt"], norms["whole"])},
            "weighted_signal_cancellation_ratio": norms["whole"] / denominator if denominator else None,
            "clipping": clipping, "clip_norm": clip_norm,
            "response_to_whole_gradient_norm_ratio": norms["response"] / norms["whole"] if norms["whole"] else None,
            "actual_parameter_update_inferred": False, "actual_parameter_update_l2": None,
            "optimizer_limit": "Unclipped gradients mix linearly; separately clipped gradients and Adam updates do not. Actual FP32 clipping/parameter rounding, theta-plus-delta ULP quantization and optimizer history are not modeled.",
            "native_gain": False, "native_model_calls": 0, "native_rqsft_recipe_changed": False,
            "training_authorized": False,
            "capture_requirement": "Caller must independently bind same-state, same-group mean gradients; target ratios alone do not prove direction or Adam updates."}


@torch.no_grad()
def first_step_adam_delta(gradient, lr, eps, clip_norm=1.):
    """Ideal first Adam delta with zero moments/decay, calculated in FP64.

    Bias corrections cancel beta1/beta2 on step1. After global clipping, delta_i
    = -lr*g_i/(abs(g_i)+eps). This is algebra, not an actual native optimizer step
    or a general claim about later Adam moments, weight decay or finite precision.
    The supplied gradient is the ENTIRE tiny toy vector: this local norm cannot
    simulate global clipping by calling the helper separately on native blocks.
    Native FP32 parameter rounding and all subsequent optimizer history are absent.
    """
    lr, eps, clip_norm = (_positive(value, name) for value, name in ((lr, "learning rate"), (eps, "epsilon"), (clip_norm, "clip norm")))
    g = _tensor(gradient).to(dtype=torch.float64)
    if not torch.isfinite(g).all():
        raise ValueError("Finite gradient required")
    norm = float(torch.linalg.vector_norm(g))
    if not math.isfinite(norm):
        raise ValueError("Nonfinite float64 gradient norm; no numerical fallback")
    clipped = g * min(1., clip_norm / (norm + 1e-6))
    return -lr * clipped / (clipped.abs() + eps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args(argv)
    source = I.check_input_path(args.plan)
    if any(any(word in part.casefold() for word in ("sealed", "confirmation", "protected")) for part in source.parts):
        raise ValueError("Protected input paths are forbidden")
    sealed = I.read_json(source)
    report = describe_plan_geometry(sealed)
    if report["fixture_only"] is not args.fixture_only:
        raise ValueError("Fixture plans require explicit --fixture-only; native metadata cannot masquerade as fixture")
    report["input_plan_path"] = str(source)
    report["input_file_sha256"] = I.file_hash(source)
    output = external_output(I.check_input_path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    I.write_new(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
