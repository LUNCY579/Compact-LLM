"""Build auditable real, shuffled-peer, no-peer, and classical support plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from torch import Tensor

from .attention import PeerAttention, PeerAttentionResult
from .calibration import first_order_loss_change
from .descriptors import tensor_descriptor
from .generator import LOCAL_FEATURE_NAMES, PCSMGenerator
from .plan import PCSMPlan, TensorPlan, make_tensor_plan
from .property_maps import PropertyMaps, aggregate_property_maps, build_property_maps
from .randomness import derive_seed, seed_everything
from .topk import stable_topk_mask, topk_ste

PeerMode = Literal["real", "shuffled", "layer_shuffled", "none"]

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class PlanBuildResult:
    plan: PCSMPlan
    trace: tuple[dict[str, float], ...]
    attention: dict[str, object]


@dataclass(frozen=True)
class PreparedPeerContext:
    target_descriptor: Tensor
    gradient_descriptor: Tensor
    peer_descriptors: dict[str, Tensor]
    peer_maps: tuple[PropertyMaps, ...]
    attention_module: PeerAttention | None


def load_same_role_weights(
    model_path: Path, *, role_suffix: str = ".mlp.down_proj.weight"
) -> dict[str, Tensor]:
    """Load only one semantic role from a safetensors checkpoint."""

    checkpoint = model_path / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    weights: dict[str, Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            if name.endswith(role_suffix):
                tensor = handle.get_tensor(name)
                if tensor.ndim != 2:
                    raise RuntimeError(f"Same-role target is not a matrix: {name}")
                weights[name] = tensor.contiguous()
    if len(weights) < 2:
        raise RuntimeError(f"Need at least two same-role weights, found {len(weights)}")
    shapes = {tuple(value.shape) for value in weights.values()}
    if len(shapes) != 1:
        raise RuntimeError("PCSM v1 requires equal same-role shapes")
    return weights


def _shuffled_matrix(value: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(value.numel(), generator=generator)
    return value.reshape(-1).index_select(0, order).reshape(value.shape)


def _layer_depth(name: str) -> int:
    match = _LAYER_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Cannot derive layer depth from parameter name: {name}")
    return int(match.group(1))


def _zero_properties(shape: tuple[int, int]) -> PropertyMaps:
    rows, columns = shape
    return PropertyMaps(
        element=torch.zeros(shape, dtype=torch.float32),
        row=torch.zeros(rows, dtype=torch.float32),
        column=torch.zeros(columns, dtype=torch.float32),
        joint=torch.zeros(shape, dtype=torch.float32),
    )


def _normalized_energy_profiles(weight: Tensor) -> tuple[Tensor, Tensor]:
    values = weight.float()
    row = torch.sqrt(values.square().mean(dim=1) + 1e-12)
    column = torch.sqrt(values.square().mean(dim=0) + 1e-12)
    return (
        row / row.mean().clamp_min(1e-12),
        column / column.mean().clamp_min(1e-12),
    )


def _matrix_shape(weight: Tensor) -> tuple[int, int]:
    if weight.ndim != 2:
        raise ValueError("Expected a matrix")
    return int(weight.shape[0]), int(weight.shape[1])


def _prepare_peer_context(
    weights: dict[str, Tensor],
    *,
    target_name: str,
    task_gradient: Tensor,
    descriptor_dimension: int,
    mode: PeerMode,
    seed: int,
) -> PreparedPeerContext:
    if target_name not in weights:
        raise KeyError(target_name)
    target = weights[target_name].float()
    if task_gradient.shape != target.shape:
        raise ValueError("Task gradient does not match target weight")
    depths = {name: _layer_depth(name) for name in weights}
    maximum_depth = max(depths.values())
    if maximum_depth <= 0:
        raise ValueError("Layer-depth normalization requires repeated layers")
    target_descriptor = tensor_descriptor(
        target,
        dimension=descriptor_dimension,
        calibration_gradient=task_gradient,
        normalized_depth=depths[target_name] / maximum_depth,
        role="mlp.down_proj",
    )
    gradient_descriptor = tensor_descriptor(
        task_gradient,
        dimension=descriptor_dimension,
        calibration_gradient=task_gradient,
        normalized_depth=depths[target_name] / maximum_depth,
        role="mlp.down_proj.task_gradient",
    )
    if mode == "none":
        return PreparedPeerContext(
            target_descriptor=target_descriptor,
            gradient_descriptor=gradient_descriptor,
            peer_descriptors={},
            peer_maps=(),
            attention_module=None,
        )

    peer_weights = {
        name: value.float()
        if mode == "real" or mode == "layer_shuffled"
        else _shuffled_matrix(value.float(), derive_seed(seed, name, "shuffle"))
        for name, value in weights.items()
        if name != target_name
    }
    if target_name in peer_weights:
        raise AssertionError("Self advice escaped peer construction")
    peer_names = sorted(peer_weights)
    descriptor_depths = {name: depths[name] for name in peer_names}
    if mode == "layer_shuffled":
        generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(seed, "layer-depth-permutation")
        )
        permutation = torch.randperm(len(peer_names), generator=generator).tolist()
        permuted_depths = [depths[peer_names[index]] for index in permutation]
        descriptor_depths = dict(zip(peer_names, permuted_depths, strict=True))
    peer_descriptors = {
        name: tensor_descriptor(
            value,
            dimension=descriptor_dimension,
            normalized_depth=descriptor_depths[name] / maximum_depth,
            role="mlp.down_proj",
        )
        for name, value in peer_weights.items()
    }
    seed_everything(derive_seed(seed, "peer-attention"))
    attention_module = PeerAttention(descriptor_dimension)
    maps_by_name = {
        name: build_property_maps(value) for name, value in peer_weights.items()
    }
    names = tuple(sorted(peer_descriptors))
    return PreparedPeerContext(
        target_descriptor=target_descriptor,
        gradient_descriptor=gradient_descriptor,
        peer_descriptors=peer_descriptors,
        peer_maps=tuple(maps_by_name[name] for name in names),
        attention_module=attention_module,
    )


def peer_context(
    weights: dict[str, Tensor],
    *,
    target_name: str,
    task_gradient: Tensor,
    descriptor_dimension: int,
    mode: PeerMode,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, PropertyMaps, PeerAttentionResult | None]:
    prepared = _prepare_peer_context(
        weights,
        target_name=target_name,
        task_gradient=task_gradient,
        descriptor_dimension=descriptor_dimension,
        mode=mode,
        seed=seed,
    )
    if prepared.attention_module is None:
        return (
            prepared.target_descriptor,
            prepared.gradient_descriptor,
            torch.zeros(descriptor_dimension, dtype=torch.float32),
            _zero_properties(_matrix_shape(weights[target_name])),
            None,
        )
    attention = prepared.attention_module(
        target_name, prepared.target_descriptor, prepared.peer_descriptors
    )
    properties = aggregate_property_maps(
        prepared.peer_maps, attention.alphas, _matrix_shape(weights[target_name])
    )
    return (
        prepared.target_descriptor,
        prepared.gradient_descriptor,
        attention.peer_summary.detach(),
        properties,
        attention,
    )


def calibrate_pcsm_plan(
    weights: dict[str, Tensor],
    *,
    target_name: str,
    calibration_gradient: Tensor,
    meta_gradient: Tensor,
    base_fingerprint: str,
    k: int,
    seed: int,
    mode: PeerMode,
    steps: int = 20,
    generator_learning_rate: float = 0.01,
    descriptor_dimension: int = 24,
    structural_regularizer: float = 0.0,
    structural_proxy_learning_rate: float = 0.0001,
    support_fraction: float | None = None,
) -> PlanBuildResult:
    """Calibrate generator offline using held-out first-order alignment."""

    if steps < 0 or generator_learning_rate <= 0:
        raise ValueError("Calibration steps must be nonnegative and LR positive")
    if structural_regularizer < 0:
        raise ValueError("structural_regularizer must be nonnegative")
    if structural_proxy_learning_rate <= 0:
        raise ValueError("structural_proxy_learning_rate must be positive")
    target = weights[target_name].float()
    if (
        calibration_gradient.shape != target.shape
        or meta_gradient.shape != target.shape
    ):
        raise ValueError("Calibration gradients do not match target shape")
    prepared = _prepare_peer_context(
        weights,
        target_name=target_name,
        task_gradient=calibration_gradient,
        descriptor_dimension=descriptor_dimension,
        mode=mode,
        seed=seed,
    )
    seed_everything(derive_seed(seed, "pcsm-generator"))
    generator = PCSMGenerator(descriptor_dimension=descriptor_dimension)
    trainable = list(generator.parameters())
    if prepared.attention_module is not None:
        trainable.extend(prepared.attention_module.parameters())
    optimizer = torch.optim.Adam(trainable, lr=generator_learning_rate)
    meta_unit = meta_gradient.float() / torch.linalg.vector_norm(
        meta_gradient.float()
    ).clamp_min(1e-12)
    cal_unit = calibration_gradient.float() / torch.linalg.vector_norm(
        calibration_gradient.float()
    ).clamp_min(1e-12)
    trace: list[dict[str, float]] = []
    attention: PeerAttentionResult | None = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if prepared.attention_module is None:
            peer_descriptor = torch.zeros(descriptor_dimension, dtype=torch.float32)
            properties = _zero_properties(_matrix_shape(target))
            attention = None
        else:
            attention = prepared.attention_module(
                target_name,
                prepared.target_descriptor,
                prepared.peer_descriptors,
            )
            peer_descriptor = attention.peer_summary
            properties = aggregate_property_maps(
                prepared.peer_maps,
                attention.alphas,
                _matrix_shape(target),
                differentiable_attention=True,
            )
        output = generator(
            target,
            calibration_gradient,
            prepared.target_descriptor,
            prepared.gradient_descriptor,
            peer_descriptor,
            properties,
        )
        support_ste = topk_ste(output.scores, k)
        predicted = first_order_loss_change(
            meta_unit,
            cal_unit,
            support_ste,
            output.stabilizer,
            learning_rate=1.0,
        )
        objective = predicted * (target.numel() / k)
        regularizer = torch.zeros((), dtype=torch.float32)
        if structural_regularizer:
            proposal = target - (
                structural_proxy_learning_rate
                * support_ste
                * output.stabilizer
                * torch.sign(calibration_gradient.float())
            )
            proposal_row, proposal_column = _normalized_energy_profiles(proposal)
            regularizer = torch.mean((proposal_row - properties.row) ** 2) + torch.mean(
                (proposal_column - properties.column) ** 2
            )
            objective = objective + structural_regularizer * regularizer
        objective.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        trace.append(
            {
                "step": float(step),
                "objective": float(objective.detach()),
                "predicted_loss_change": float(predicted.detach()),
                "structural_regularizer": float(regularizer.detach()),
            }
        )
    with torch.no_grad():
        if prepared.attention_module is None:
            peer_descriptor = torch.zeros(descriptor_dimension, dtype=torch.float32)
            properties = _zero_properties(_matrix_shape(target))
            attention = None
        else:
            attention = prepared.attention_module(
                target_name,
                prepared.target_descriptor,
                prepared.peer_descriptors,
            )
            peer_descriptor = attention.peer_summary
            properties = aggregate_property_maps(
                prepared.peer_maps, attention.alphas, _matrix_shape(target)
            )
        output = generator(
            target,
            calibration_gradient,
            prepared.target_descriptor,
            prepared.gradient_descriptor,
            peer_descriptor,
            properties,
        )
    if attention is None:
        peer_names: tuple[str, ...] = ()
        alphas = torch.empty(0, dtype=torch.float32)
        attention_record: dict[str, object] = {"peer_names": [], "alphas": []}
    else:
        peer_names = attention.peer_names
        alphas = attention.alphas
        attention_record = attention.record()
    tensor_plan = make_tensor_plan(
        name=target_name,
        role="mlp.down_proj",
        scores=output.scores,
        stabilizer=output.stabilizer,
        k=k,
        peer_names=peer_names,
        attention_alphas=alphas,
        module_type="torch.nn.Linear",
        layer_depth=_layer_depth(target_name),
        support_fraction=support_fraction,
        feature_names=LOCAL_FEATURE_NAMES,
        feature_weights=output.feature_weights,
        generator_config=generator.configuration(),
    )
    plan = PCSMPlan(
        base_fingerprint=base_fingerprint,
        descriptor_dimension=descriptor_dimension,
        entries=(tensor_plan,),
        metadata={
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "generator_learning_rate": generator_learning_rate,
            "attention_residual_jointly_calibrated": bool(
                steps > 0 and prepared.attention_module is not None
            ),
            "structural_regularizer": structural_regularizer,
            "structural_regularizer_kind": (
                "updated_target_normalized_row_column_energy"
            ),
            "structural_proxy_learning_rate": structural_proxy_learning_rate,
            "support_frozen_once_after_calibration": True,
        },
    )
    return PlanBuildResult(
        plan=plan,
        trace=tuple(trace),
        attention=attention_record,
    )


def classical_tensor_plan(
    weight: Tensor,
    gradient: Tensor,
    *,
    target_name: str,
    base_fingerprint: str,
    k: int,
    seed: int,
    method: Literal["random", "magnitude", "gradient", "dense"],
    descriptor_dimension: int = 24,
    support_fraction: float | None = None,
) -> PCSMPlan:
    if weight.shape != gradient.shape:
        raise ValueError("Weight and gradient shapes differ")
    if method == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        scores = torch.rand(weight.shape, generator=generator)
    elif method == "magnitude":
        scores = weight.detach().float().abs()
    elif method == "gradient":
        scores = gradient.detach().float().abs()
    else:
        scores = torch.zeros_like(weight, dtype=torch.float32)
        k = weight.numel()
    mask = stable_topk_mask(scores, k)
    tensor_plan = TensorPlan(
        name=target_name,
        role="mlp.down_proj",
        shape=tuple(weight.shape),
        support_mask=mask,
        stabilizer=torch.ones_like(weight, dtype=torch.float32),
        peer_names=(),
        attention_alphas=torch.empty(0, dtype=torch.float32),
        module_type="torch.nn.Linear",
        layer_depth=(
            _layer_depth(target_name) if _LAYER_PATTERN.search(target_name) else None
        ),
        support_fraction=support_fraction,
        generator_config={"version": "classical-control-v1", "method": method},
    )
    return PCSMPlan(
        base_fingerprint=base_fingerprint,
        descriptor_dimension=descriptor_dimension,
        entries=(tensor_plan,),
        metadata={"method": method, "seed": seed},
    )


def without_stabilizer(plan: PCSMPlan) -> PCSMPlan:
    entries = tuple(
        replace(
            entry,
            stabilizer=torch.ones(entry.shape, dtype=torch.float32),
        )
        for entry in plan.entries
    )
    return PCSMPlan(
        base_fingerprint=plan.base_fingerprint,
        descriptor_dimension=plan.descriptor_dimension,
        entries=entries,
        metadata={**dict(plan.metadata), "stabilizer_ablation": True},
    )


__all__ = [
    "PlanBuildResult",
    "calibrate_pcsm_plan",
    "classical_tensor_plan",
    "load_same_role_weights",
    "peer_context",
    "without_stabilizer",
]
