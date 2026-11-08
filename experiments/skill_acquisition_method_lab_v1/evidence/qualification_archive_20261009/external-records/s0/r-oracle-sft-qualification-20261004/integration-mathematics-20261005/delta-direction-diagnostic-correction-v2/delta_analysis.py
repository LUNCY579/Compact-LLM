from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import CheckpointError, CheckpointReader
from .diagnostics import classify_tensor, tensor_delta_stats
from .hashing import atomic_write_json, atomic_write_text, sha256_file
from .direction_geometry import direction_metrics, optional_number


@dataclass
class Accumulator:
    count: int = 0
    dot: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    base2: float = 0.0
    sign_agree: int = 0
    abs_x: float = 0.0
    abs_y: float = 0.0
    abs_x2: float = 0.0
    abs_y2: float = 0.0
    abs_xy: float = 0.0

    def add(self, x: torch.Tensor, y: torch.Tensor, base: torch.Tensor) -> None:
        x = x.float().reshape(-1)
        y = y.float().reshape(-1)
        base = base.float().reshape(-1)
        self.count += x.numel()
        self.dot += float(torch.dot(x, y).item())
        self.x2 += float(torch.dot(x, x).item())
        self.y2 += float(torch.dot(y, y).item())
        self.base2 += float(torch.dot(base, base).item())
        self.sign_agree += int((torch.sign(x) == torch.sign(y)).sum().item())
        ax, ay = x.abs(), y.abs()
        self.abs_x += float(ax.sum().item())
        self.abs_y += float(ay.sum().item())
        self.abs_x2 += float(torch.dot(ax, ax).item())
        self.abs_y2 += float(torch.dot(ay, ay).item())
        self.abs_xy += float(torch.dot(ax, ay).item())

    def result(self) -> dict[str, Any]:
        x_norm, y_norm = math.sqrt(self.x2), math.sqrt(self.y2)
        base_norm = math.sqrt(self.base2)
        centered_xy = self.abs_xy - self.abs_x * self.abs_y / max(1, self.count)
        centered_x2 = self.abs_x2 - self.abs_x**2 / max(1, self.count)
        centered_y2 = self.abs_y2 - self.abs_y**2 / max(1, self.count)
        correlation_denominator = math.sqrt(
            max(0.0, centered_x2) * max(0.0, centered_y2)
        )
        direction = direction_metrics(self.dot, x_norm, y_norm)
        return {
            "count": self.count,
            "delta4_l2": x_norm,
            "delta8_l2": y_norm,
            "base_l2": base_norm,
            "delta4_relative_norm": x_norm / base_norm if base_norm > 0 else None,
            "delta8_relative_norm": y_norm / base_norm if base_norm > 0 else None,
            **direction,
            "sign_agreement": self.sign_agree / max(1, self.count),
            "magnitude_correlation": (
                centered_xy / correlation_denominator
                if correlation_denominator > 0
                else None
            ),
            "norm_ratio_delta4_over_delta8": x_norm / y_norm if y_norm > 0 else None,
        }


def _head_views(
    key: str,
    tensor: torch.Tensor,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> list[torch.Tensor]:
    if ".self_attn.q_proj.weight" in key:
        return list(tensor.reshape(query_heads, head_dim, tensor.shape[1]))
    if ".self_attn.k_proj.weight" in key or ".self_attn.v_proj.weight" in key:
        return list(tensor.reshape(kv_heads, head_dim, tensor.shape[1]))
    if ".self_attn.o_proj.weight" in key:
        reshaped = tensor.reshape(tensor.shape[0], query_heads, head_dim)
        return [reshaped[:, index, :] for index in range(query_heads)]
    return []


def analyze_delta_pair(
    *,
    delta4_path: Path,
    delta8_path: Path,
    base_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    group_accumulators: dict[str, Accumulator] = defaultdict(Accumulator)
    tensor_rows = []
    head_rows = []
    with CheckpointReader(delta4_path) as delta4, CheckpointReader(
        delta8_path
    ) as delta8, CheckpointReader(base_path) as base:
        if set(delta4.keys) != set(delta8.keys):
            raise CheckpointError("delta key sets differ")
        config = base.config()
        query_heads = int(config["num_attention_heads"])
        kv_heads = int(config["num_key_value_heads"])
        head_dim = int(config["head_dim"])
        for key in delta4.keys:
            x = delta4.get_tensor(key)
            y = delta8.get_tensor(key)
            b = base.get_tensor(key)
            if x.shape != y.shape or x.shape != b.shape:
                raise CheckpointError(f"delta/base shape mismatch: {key}")
            metadata = classify_tensor(key)
            group_names = ["global", f"module:{metadata['module']}"]
            if metadata["layer"] is not None:
                group_names.append(f"layer:{metadata['layer']}")
            for group_name in group_names:
                group_accumulators[group_name].add(x, y, b)
            tensor_rows.append(tensor_delta_stats(key, x, base=b, other=y))
            x_heads = _head_views(
                key,
                x,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
            y_heads = _head_views(
                key,
                y,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
            for index, (xh, yh) in enumerate(zip(x_heads, y_heads, strict=True)):
                xh_flat, yh_flat = xh.float().reshape(-1), yh.float().reshape(-1)
                xnorm = float(torch.linalg.vector_norm(xh_flat).item())
                ynorm = float(torch.linalg.vector_norm(yh_flat).item())
                direction = direction_metrics(float(torch.dot(xh_flat, yh_flat).item()), xnorm, ynorm)
                head_rows.append(
                    {
                        "key": key,
                        "head": index,
                        **direction,
                        "sign_agreement": float(
                            (torch.sign(xh_flat) == torch.sign(yh_flat)).sum().item()
                            / max(1, xh_flat.numel())
                        ),
                        "delta4_l2": xnorm,
                        "delta8_l2": ynorm,
                    }
                )
    groups = {
        name: accumulator.result()
        for name, accumulator in sorted(group_accumulators.items())
    }
    report = {
        "schema_version": "1.1",
        "delta4": {
            "path": str(delta4_path.resolve()),
            "manifest_sha256": sha256_file(delta4_path / "delta_manifest.json"),
        },
        "delta8": {
            "path": str(delta8_path.resolve()),
            "manifest_sha256": sha256_file(delta8_path / "delta_manifest.json"),
        },
        "base_path": str(base_path.resolve()),
        "global": groups.pop("global"),
        "groups": groups,
        "tensor_count": len(tensor_rows),
        "head_count": len(head_rows),
        "tensors": tensor_rows,
        "heads": head_rows,
    }
    atomic_write_json(output_json, report)
    global_stats = report["global"]
    lines = [
        "# Δ4 / Δ8 Scale-Trajectory Analysis",
        "",
        f"- Global cosine: `{optional_number(global_stats['cosine'])}`",
        f"- Direction status: `{global_stats['direction_status']}`",
        f"- Global sign agreement: `{global_stats['sign_agreement']:.6f}`",
        f"- Magnitude correlation: `{optional_number(global_stats['magnitude_correlation'])}`",
        f"- Angular deviation (degrees): `{optional_number(global_stats['angular_degrees'], 3)}`",
        f"- Relative norm Δ4/base: `{optional_number(global_stats['delta4_relative_norm'])}`",
        f"- Relative norm Δ8/base: `{optional_number(global_stats['delta8_relative_norm'])}`",
        "",
        "## Modules",
        "",
        "| Module | Cosine | Sign agreement | Δ4/base | Δ8/base |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in groups.items():
        if not name.startswith("module:"):
            continue
        lines.append(
            f"| {name.removeprefix('module:')} | {optional_number(stats['cosine'], 4)} | "
            f"{stats['sign_agreement']:.4f} | {optional_number(stats['delta4_relative_norm'], 4)} | "
            f"{optional_number(stats['delta8_relative_norm'], 4)} |"
        )
    atomic_write_text(output_markdown, "\n".join(lines))
    return report
