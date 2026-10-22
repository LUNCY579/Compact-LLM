from __future__ import annotations

import math
import re
from typing import Any

import torch
from .direction_geometry import direction_metrics


LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def classify_tensor(key: str) -> dict[str, Any]:
    match = LAYER_KEY.match(key)
    layer = int(match.group(1)) if match else None
    suffix = match.group(2) if match else key
    if "self_attn.q_proj" in suffix:
        module = "attention.q"
    elif "self_attn.k_proj" in suffix:
        module = "attention.k"
    elif "self_attn.v_proj" in suffix:
        module = "attention.v"
    elif "self_attn.o_proj" in suffix:
        module = "attention.o"
    elif "mlp.gate_proj" in suffix:
        module = "mlp.gate"
    elif "mlp.up_proj" in suffix:
        module = "mlp.up"
    elif "mlp.down_proj" in suffix:
        module = "mlp.down"
    elif "layernorm" in suffix or suffix == "model.norm.weight":
        module = "norm"
    elif "embed_tokens" in suffix:
        module = "embedding"
    elif "lm_head" in suffix:
        module = "output_head"
    else:
        module = "other"
    return {"layer": layer, "module": module}


def _sample_flat(tensor: torch.Tensor, maximum: int = 1_000_000) -> torch.Tensor:
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= maximum:
        return flat.float()
    step = flat.numel() / maximum
    indices = torch.floor(
        torch.arange(maximum, dtype=torch.float64, device=flat.device) * step
    ).to(torch.long)
    return flat.index_select(0, indices).float()


def tensor_delta_stats(
    key: str,
    delta: torch.Tensor,
    *,
    base: torch.Tensor | None = None,
    other: torch.Tensor | None = None,
) -> dict[str, Any]:
    value = delta.detach().float()
    flat = value.reshape(-1)
    finite = torch.isfinite(flat)
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    finite_values = flat[finite]
    l2_sq = float(torch.dot(finite_values, finite_values).item())
    l2 = math.sqrt(max(0.0, l2_sq))
    count = int(flat.numel())
    result: dict[str, Any] = {
        "key": key,
        **classify_tensor(key),
        "shape": list(delta.shape),
        "dtype": str(delta.dtype).removeprefix("torch."),
        "count": count,
        "l2": l2,
        "rms": math.sqrt(l2_sq / max(1, finite_values.numel())),
        "maximum_absolute": (
            float(finite_values.abs().max().item()) if finite_values.numel() else None
        ),
        "sparsity": float((flat == 0).sum().item() / max(1, count)),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    if finite_values.numel():
        sample = _sample_flat(finite_values)
        quantiles = torch.quantile(
            sample,
            torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]),
        )
        result["quantiles"] = {
            label: float(value)
            for label, value in zip(
                ("q000", "q001", "q005", "q050", "q095", "q099", "q100"),
                quantiles.tolist(),
                strict=True,
            )
        }
        result["quantile_sample_count"] = int(sample.numel())
    if base is not None:
        base_value = base.detach().float().reshape(-1)
        base_l2 = float(torch.linalg.vector_norm(base_value).item())
        result["base_l2"] = base_l2
        result["relative_delta_norm"] = l2 / base_l2 if base_l2 > 0 else None
    if other is not None:
        other_value = other.detach().float().reshape(-1)
        if other_value.shape != flat.shape:
            raise ValueError(f"pair shape mismatch for {key}")
        other_l2 = float(torch.linalg.vector_norm(other_value).item())
        dot = float(torch.dot(flat, other_value).item())
        direction = direction_metrics(dot, l2, other_l2)
        result.update({"pair_" + name: value for name, value in direction.items()})
        signs = torch.sign(flat)
        other_signs = torch.sign(other_value)
        result["pair_sign_agreement"] = float(
            (signs == other_signs).sum().item() / max(1, count)
        )
        magnitude_x = flat.abs()
        magnitude_y = other_value.abs()
        magnitude_dot = float(torch.dot(magnitude_x, magnitude_y).item())
        sum_x = float(magnitude_x.sum().item())
        sum_y = float(magnitude_y.sum().item())
        sum_x2 = float(torch.dot(magnitude_x, magnitude_x).item())
        sum_y2 = float(torch.dot(magnitude_y, magnitude_y).item())
        numerator = magnitude_dot - sum_x * sum_y / max(1, count)
        denominator = math.sqrt(
            max(0.0, sum_x2 - sum_x * sum_x / max(1, count))
            * max(0.0, sum_y2 - sum_y * sum_y / max(1, count))
        )
        result["pair_magnitude_correlation"] = (
            numerator / denominator if denominator > 0 else None
        )
        result["pair_norm_ratio"] = l2 / other_l2 if other_l2 > 0 else None
    return result
