"""Token-balanced diagnostics of actual frozen donor transformations."""
from __future__ import annotations

import math
import time

import torch
from torch.nn import functional as F


def finite(tensor):
    if tensor is None or not torch.isfinite(tensor).all():
        raise FloatingPointError("nonfinite alignment tensor")
    return tensor


def frozen(module):
    if any(child.training for child in module.modules()) or any(p.requires_grad or p.grad is not None for p in module.parameters()):
        raise ValueError("backbone must be eval/frozen and gradient-free")


def deadline(started, seconds):
    # Forward/backward calls cannot safely be interrupted; check both boundaries.
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid alignment deadline")
    if time.perf_counter() - started > seconds:
        raise TimeoutError("alignment operation exceeded its registered elapsed budget")


def token_metrics(predicted, target):
    if predicted.ndim != 2 or predicted.shape != target.shape or not len(target):
        raise ValueError("expected nonempty equally shaped token matrices")
    # Accumulation in FP64 keeps diagnostics finite for finite FP32 activations.
    predicted, target = finite(predicted.double()), finite(target.double())
    energy = target.square().mean(-1)
    error = (predicted - target).square().mean(-1)
    cosine = F.cosine_similarity(predicted, target, dim=-1)
    return {"tokens": len(target), "mean_token_cosine": float(cosine.mean()),
            "mean_token_relative_mse": float((error / energy.clamp_min(1e-12)).mean()),
            "global_relative_mse": float(error.mean() / energy.mean().clamp_min(1e-12)),
            "predicted_rms": float(predicted.square().mean().sqrt()),
            "target_rms": float(energy.mean().sqrt()),
            "residual_magnitude_ratio": float(predicted.square().mean().sqrt() /
                                                energy.mean().sqrt().clamp_min(1e-12))}


def token_masks(input_ids, special_ids):
    if not input_ids or any(not x.numel() or x.dtype != torch.long for x in input_ids):
        raise ValueError("token strata require nonempty integer prompt IDs")
    ids = torch.cat([x.reshape(-1) for x in input_ids])
    special = torch.zeros_like(ids, dtype=torch.bool)
    for value in special_ids:
        special |= ids == value
    final = torch.cat([torch.arange(x.numel()) == x.numel()-1 for x in input_ids])
    return {"all_tokens": torch.ones_like(special), "ordinary_tokens": ~special,
            "special_tokens": special, "final_prompt_token": final}


def stratified_metrics(predicted, target, masks):
    if predicted.shape != target.shape or any(mask.dtype != torch.bool or mask.shape != (len(target),)
                                              for mask in masks.values()):
        raise ValueError("metric mask/activation shape mismatch")
    return {name: token_metrics(predicted[mask], target[mask]) if mask.any() else {"tokens": 0}
            for name, mask in masks.items()}


def normalized_loss(predicted, target, *, token_normalized):
    if predicted.shape != target.shape or predicted.ndim < 2 or not predicted.numel():
        raise ValueError("loss expects matching nonempty token states")
    finite(predicted); finite(target)
    error, energy = (predicted-target).square().mean(-1), target.square().mean(-1)
    if token_normalized:
        return finite((error / energy.clamp_min(1e-8)).mean())
    return finite(error.mean() / energy.mean().clamp_min(1e-8))


def donor_delta(layer, rope, projected, *, config=None):
    from shared.hidden_capture import positions_mask
    if projected.ndim != 3 or projected.shape[0] != 1 or not projected.numel():
        raise ValueError("donor transformation requires one complete prompt")
    frozen(layer)
    frozen(rope)
    finite(projected)
    started = time.perf_counter()
    positions, mask = positions_mask(projected)
    result = layer(projected, attention_mask=mask, position_ids=positions,
                   position_embeddings=rope(projected, positions), use_cache=False,
                   past_key_values=None)
    if not isinstance(result, torch.Tensor):
        raise TypeError("only qualified tensor-returning Qwen3 decoders are supported")
    if result.shape != projected.shape:
        raise ValueError("donor changed the residual stream dimensions")
    finite(result)
    if config is not None:
        if projected.device.type == "cuda":
            from shared.runtime import cuda_memory
            cuda_memory(config)  # Synchronizes before timing and checks reserved VRAM.
        deadline(started, config["max_prompt_seconds"])
        ratio = float(result.detach().double().square().mean().sqrt() /
                      projected.detach().double().square().mean().sqrt().clamp_min(1e-12))
        low, high = config["hidden_rms_ratio_bounds"]
        if not low <= ratio <= high:
            raise FloatingPointError("donor hidden RMS outside registered bounds")
    return finite(result-projected)
