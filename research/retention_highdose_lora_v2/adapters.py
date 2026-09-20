"""FP32 tail LoRA primitives; no model loading, execution permit, or CLI.

Production injection is fixed to the qualified 596M source, layers 8..27,
seven projections and four existing norms per layer. The private scope helper
exists only for tiny CPU fixtures; production callers use inject_lora().
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight",
         "self_attn.q_norm.weight", "self_attn.k_norm.weight")
PRODUCTION_SHAPES = {
    "self_attn.q_proj": (1024, 2048), "self_attn.k_proj": (1024, 1024),
    "self_attn.v_proj": (1024, 1024), "self_attn.o_proj": (2048, 1024),
    "mlp.gate_proj": (1024, 3072), "mlp.up_proj": (1024, 3072),
    "mlp.down_proj": (3072, 1024),
}
BASE_PARAMETERS = 596049920
ADAPTER_PARAMETERS = 7208960
NORM_PARAMETERS = 46080
TRAINABLE_PARAMETERS = 7255040
AUGMENTED_PARAMETERS = 603258880


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    in_features: int
    out_features: int


@dataclass(frozen=True)
class AdapterSpec:
    rank: int
    alpha: float
    scale: float
    layer_start: int
    layer_stop: int
    projections: tuple[ProjectionSpec, ...]
    norm_names: tuple[str, ...]
    adapter_names: tuple[str, ...]
    trainable_names: tuple[str, ...]
    frozen_base_names: tuple[str, ...]
    base_parameters: int
    adapter_parameters: int
    norm_parameters: int
    trainable_parameters: int
    augmented_parameters: int
    production: bool

    def as_dict(self) -> dict:
        # JSON round trip turns immutable tuples into ordinary receipt lists.
        return json.loads(json.dumps(asdict(self), allow_nan=False))


class LoRALinear(nn.Module):
    """Keep the original weight Parameter/name and add an unmerged FP32 2BA path."""

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        _require(type(linear) is nn.Linear and linear.bias is None, "LoRA requires a bias-free nn.Linear")
        _require(type(rank) is int and rank > 0 and math.isfinite(alpha) and alpha > 0,
                 "invalid LoRA rank/alpha")
        _require(linear.weight.dtype == torch.float32 and not linear.weight.is_meta,
                 "LoRA requires materialized FP32 base weights")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha) / rank
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.register_parameter("bias", None)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features,
                                               dtype=torch.float32, device=self.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank,
                                              dtype=torch.float32, device=self.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.train(linear.training)

    def forward(self, x):
        return F.linear(x, self.weight) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent, leaf = name.rsplit(".", 1)
    return model.get_submodule(parent), leaf


def _inject_scope(model: nn.Module, *, layer_start: int, layer_stop: int,
                  rank: int, alpha: float, strict: bool = False):
    """Private mechanical-fixture hook. Never expose this scope through a native CLI."""
    layers = model.model.layers
    _require(type(layer_start) is int and type(layer_stop) is int
             and 0 <= layer_start < layer_stop <= len(layers), "invalid layer scope")
    _require(type(rank) is int and rank > 0 and math.isfinite(alpha) and alpha > 0,
             "invalid LoRA rank/alpha")
    original = dict(model.named_parameters())
    _require(all(p.dtype == torch.float32 and not p.is_meta for p in original.values()),
             "all base parameters must be materialized FP32")
    _require(not any(isinstance(module, LoRALinear) for module in model.modules()), "model already has LoRA")
    base_count = sum(p.numel() for p in original.values())
    if strict:
        _require(len(layers) == 28 and layer_start == 8 and layer_stop == 28
                 and rank == 16 and alpha == 32 and base_count == BASE_PARAMETERS,
                 "production requires the exact 596M tail20 rank16/alpha32 geometry")
    projections, replacements, norm_names = [], [], []
    for index in range(layer_start, layer_stop):
        prefix = f"model.layers.{index}."
        expected_layer_names = {prefix + suffix + ".weight" for suffix in PROJECTIONS}
        expected_layer_names.update(prefix + suffix for suffix in NORMS)
        actual_layer_names = {name for name in original if name.startswith(prefix)}
        _require(actual_layer_names == expected_layer_names,
                 f"tail layer {index} must contain exactly seven matrices and four norms")
        for suffix in PROJECTIONS:
            name = prefix + suffix
            linear = model.get_submodule(name)
            _require(type(linear) is nn.Linear and linear.bias is None,
                     f"projection must be a bias-free nn.Linear: {name}")
            _require(tuple(linear.weight.shape) == (linear.out_features, linear.in_features),
                     f"projection shape mismatch: {name}")
            if strict:
                _require((linear.in_features, linear.out_features) == PRODUCTION_SHAPES[suffix],
                         f"projection differs from the registered source: {name}")
            projections.append(ProjectionSpec(name, linear.in_features, linear.out_features))
            replacements.append((name, linear))
        for suffix in NORMS:
            name = prefix + suffix
            _require(original[name].ndim == 1, f"norm must be a vector: {name}")
            if strict:
                width = 128 if suffix.startswith("self_attn.") else 1024
                _require(tuple(original[name].shape) == (width,), f"registered norm shape differs: {name}")
            norm_names.append(name)
    # Finish validating the whole inventory before changing modules/trainability.
    base_ids = [id(original[p.name + ".weight"]) for p in projections]
    _require(len(base_ids) == len(set(base_ids)), "projection weights may not alias")
    for parameter in original.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name, linear in replacements:
        parent, leaf = _parent(model, name)
        setattr(parent, leaf, LoRALinear(linear, rank, alpha))
    actual = dict(model.named_parameters())
    for name in norm_names:
        actual[name].requires_grad_(True)
    adapter_names = tuple(sorted(p.name + suffix for p in projections for suffix in (".lora_A", ".lora_B")))
    trainable_names = tuple(sorted((*adapter_names, *norm_names)))
    adapter_count = sum(actual[name].numel() for name in adapter_names)
    norm_count = sum(actual[name].numel() for name in norm_names)
    spec = AdapterSpec(rank, float(alpha), float(alpha) / rank, layer_start, layer_stop,
                       tuple(sorted(projections, key=lambda p: p.name)), tuple(sorted(norm_names)),
                       adapter_names, trainable_names, tuple(sorted(set(original) - set(norm_names))),
                       base_count, adapter_count, norm_count, adapter_count + norm_count,
                       base_count + adapter_count, strict)
    if strict:
        _require(len(projections) == 140 and len(norm_names) == 80 and len(trainable_names) == 360
                 and (adapter_count, norm_count, spec.trainable_parameters, spec.augmented_parameters)
                 == (ADAPTER_PARAMETERS, NORM_PARAMETERS, TRAINABLE_PARAMETERS, AUGMENTED_PARAMETERS),
                 "registered LoRA/norm parameter inventory differs")
    return named_trainables(model, spec), spec


def inject_lora(model: nn.Module) -> tuple[tuple[tuple[str, nn.Parameter], ...], AdapterSpec]:
    """Inject the one registered production scope; no scope/rank/precision overrides."""
    return _inject_scope(model, layer_start=8, layer_stop=28, rank=16, alpha=32.0, strict=True)


def named_trainables(model: nn.Module, spec: AdapterSpec) -> tuple[tuple[str, nn.Parameter], ...]:
    actual = dict(model.named_parameters())
    expected = set(spec.frozen_base_names) | set(spec.trainable_names)
    _require(set(actual) == expected, "model parameter names differ from immutable adapter spec")
    _require(tuple(sorted(name for name, p in actual.items() if p.requires_grad)) == spec.trainable_names,
             "trainable flags differ from immutable adapter spec")
    _require(all(p.dtype == torch.float32 and not p.is_meta for p in actual.values()),
             "adapter model is no longer materialized FP32")
    return tuple((name, actual[name]) for name in spec.trainable_names)


def frozen_base_digest(model: nn.Module, spec: AdapterSpec) -> str:
    """Hash every frozen original parameter, including all original projection W.

    Existing trainable norm vectors and new A/B factors are excluded. A change in
    any frozen matrix, embedding, head or out-of-scope norm changes this digest.
    """
    named_trainables(model, spec)
    parameters = dict(model.named_parameters())
    result = hashlib.sha256()
    for name in spec.frozen_base_names:
        parameter = parameters[name]
        metadata = json.dumps([name, list(parameter.shape), str(parameter.dtype)],
                              ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        result.update(metadata)
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 262144):
            part = flat[start:start + 262144].cpu().contiguous().view(torch.uint8).numpy()
            result.update(memoryview(part))
    return result.hexdigest()


def gradient_summary(model: nn.Module, spec: AdapterSpec, *, require_active: bool = True) -> dict:
    named = named_trainables(model, spec)
    parameters = dict(named)
    finite, active = {}, {}
    for name, parameter in named:
        gradient = parameter.grad
        finite[name] = gradient is not None and not gradient.is_sparse and bool(torch.isfinite(gradient).all())
        _require(finite[name], f"missing/sparse/nonfinite gradient: {name}")
        active[name] = bool(gradient.abs().sum() > 0)
    frozen = dict(model.named_parameters())
    _require(all(frozen[name].grad is None for name in spec.frozen_base_names),
             "frozen base parameter received a gradient")
    a_names = [name for name in spec.adapter_names if name.endswith(".lora_A")]
    b_names = [name for name in spec.adapter_names if name.endswith(".lora_B")]
    zero_b = all(not bool(parameters[name].detach().count_nonzero()) for name in b_names)
    if zero_b:
        _require(not any(active[name] for name in a_names), "A gradient must be zero at the zero-B initialization")
    if require_active:
        _require(all(active[name] for name in (*b_names, *spec.norm_names)),
                 "B/norm gradient participation is incomplete")
    return {"finite_tensors": sum(finite.values()), "active_adapter_A": sum(active[name] for name in a_names),
            "active_adapter_B": sum(active[name] for name in b_names),
            "active_norms": sum(active[name] for name in spec.norm_names),
            "zero_B_initialization": zero_b, "A_zero_at_zero_B": zero_b and not any(active[name] for name in a_names)}


def canonical_cpu_merge(weight: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                        scale: float = 2.0) -> torch.Tensor:
    """Exact export arithmetic: CPU FP32 W + (B @ A) * scale, one CPU thread.

    All three inputs must already be FP32. Independent contiguous CPU copies
    prevent export from modifying live weights or using a GPU-dependent matmul.
    """
    _require(torch.get_num_threads() == 1, "canonical CPU merge requires exactly one torch CPU thread")
    _require(not isinstance(scale, bool) and isinstance(scale, (int, float))
             and math.isfinite(scale) and scale > 0, "invalid merge scale")
    for label, value in (("weight", weight), ("A", A), ("B", B)):
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                 and not value.is_meta and value.ndim == 2, f"merge {label} must be materialized FP32 matrix")
    _require(A.shape[0] == B.shape[1] and tuple(weight.shape) == (B.shape[0], A.shape[1])
             and all(size > 0 for tensor in (weight, A, B) for size in tensor.shape),
             "merge projection/factor shapes differ")
    w, a, b = (value.detach().to(device="cpu", copy=True).contiguous() for value in (weight, A, B))
    _require(all(bool(torch.isfinite(value).all()) for value in (w, a, b)), "merge input is nonfinite")
    merged = (w + (b @ a) * scale).contiguous()
    _require(bool(torch.isfinite(merged).all()), "canonical CPU merge produced nonfinite weights")
    return merged


def dense_tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    """SHA256(canonical([name, shape, 'torch.float32']) + contiguous CPU bytes)."""
    _require(isinstance(name, str) and bool(name), "dense tensor hash requires its original name")
    _require(tensor.device.type == "cpu" and tensor.dtype == torch.float32
             and tensor.is_contiguous() and bool(torch.isfinite(tensor).all()),
             "dense hash requires finite contiguous CPU FP32 data")
    metadata = json.dumps([name, list(tensor.shape), "torch.float32"], ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    result = hashlib.sha256(metadata)
    result.update(memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy()))
    return result.hexdigest()


def iter_dense_updates(model: nn.Module, spec: AdapterSpec) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield 220 full dense theta tensors in production, not deltas; never mutate W.

    Every result is canonical CPU FP32, even when the live model is on CUDA.
    Consume one tensor at a time for bounded merge/export memory. All unlisted
    base tensors must be copied exactly from the verified source by the exporter.
    """
    named_trainables(model, spec)
    parameters = dict(model.named_parameters())
    projections = {p.name + ".weight": p for p in spec.projections}
    for name in sorted((*projections, *spec.norm_names)):
        if name in projections:
            module = model.get_submodule(projections[name].name)
            _require(isinstance(module, LoRALinear) and module.rank == spec.rank and module.scale == spec.scale,
                     "projection adapter differs from immutable spec")
            value = canonical_cpu_merge(module.weight, module.lora_A, module.lora_B, spec.scale)
        else:
            value = parameters[name].detach().to(device="cpu", copy=True).contiguous()
        _require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                 f"nonfinite or non-FP32 dense export: {name}")
        yield name, value.contiguous()


@contextmanager
def temporary_dense_merge(model: nn.Module, spec: AdapterSpec) -> Iterator[dict[str, str]]:
    """Temporarily install canonical merged matrices and return all dense hashes.

    Original wrapper objects and their frozen W/A/B Parameters remain untouched.
    Existing norm Parameters stay installed. Every successfully swapped module
    is restored in finally, including failures partway through construction.
    Production retains at most 140 additional merged matrices on the live device;
    CPU merged tensors are streamed rather than accumulated.
    """
    named_trainables(model, spec)
    originals = {projection.name: model.get_submodule(projection.name) for projection in spec.projections}
    installed, hashes = [], {}
    try:
        for name, value in iter_dense_updates(model, spec):
            hashes[name] = dense_tensor_sha256(name, value)
            module_name = name.removesuffix(".weight")
            if module_name not in originals:
                continue
            original = originals[module_name]
            # A normal CPU Linear avoids meta construction. Its throwaway random
            # initialization must not advance either training or qualification RNG.
            with torch.random.fork_rng(devices=[]):
                dense = nn.Linear(original.in_features, original.out_features, bias=False,
                                  device="cpu", dtype=torch.float32)
            dense.weight = nn.Parameter(value.to(device=original.weight.device), requires_grad=False)
            dense.train(original.training)
            parent, leaf = _parent(model, module_name)
            installed.append((parent, leaf, original))
            setattr(parent, leaf, dense)
        _require(len(hashes) == len(spec.projections) + len(spec.norm_names), "incomplete dense merge hash inventory")
        yield hashes
    finally:
        for parent, leaf, original in reversed(installed):
            setattr(parent, leaf, original)


# Both names describe the same narrow qualification context; no alternate path.
temporary_dense_model = temporary_dense_merge


def trainable_state(model: nn.Module, spec: AdapterSpec) -> dict[str, torch.Tensor]:
    """Small cloned CPU parameter-only export; native checkpoints also need moments/RNG."""
    return {name: parameter.detach().cpu().clone() for name, parameter in named_trainables(model, spec)}
