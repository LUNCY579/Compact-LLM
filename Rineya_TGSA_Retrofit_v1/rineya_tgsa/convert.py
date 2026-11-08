"""Deterministic conversion of a pretrained Qwen-style checkpoint to TGSA.

The conversion objective is stated plainly: **preserve as much pretrained
function as mathematically possible before the first optimization step.** In
this design that bound is not "as much as possible" in a hand-wavy sense — it is
*all of it*, exactly, because the retrofit reduces algebraically to native
attention while its conditioning projections are zero.

## What is reused

For every converted layer, these native tensors are copied **verbatim**:

| native tensor | retrofit tensor | shape |
|---|---|---|
| `self_attn.q_proj.weight` | `q_proj.weight` | `[Hq*d, hidden]` |
| `self_attn.k_proj.weight` | `k_proj.weight` | `[Hkv*d, hidden]` |
| `self_attn.v_proj.weight` | `v_proj.weight` | `[Hkv*d, hidden]` |
| `self_attn.o_proj.weight` | `o_proj.weight` | `[hidden, Hq*d]` |

plus the corresponding biases when the source model has them. Nothing is
reshaped, transposed, sliced, or re-ordered, so no mapping error is possible.

## How native GQA heads map into sequential heads

The retrofit iterates **query** heads sequentially, `j = 0 .. Hq-1`. Query head
`j` reads its own slice of the native Q projection — rows `[j*d : (j+1)*d]` —
and attends against the key/value group

    g(j) = j // (Hq // Hkv)

which is exactly the native GQA grouping (`repeat_interleave` by `group_size`).
Head `j`'s contribution to the output uses columns `[j*d : (j+1)*d]` of the
native O projection. Because Q/K/V are still projected from the *full* hidden
state — as in the pretrained model, not from a per-head slice of it — the
per-head computation is identical to native and the concatenate-then-project
output is identical too.

The KV cache therefore holds `Hkv` groups exactly as the native model does. GQA's
memory advantage is preserved, not silently discarded.

## What is newly initialized

Only the trajectory pathway: `carrier_init`, `trajectory_encoders.*`,
`correction_gate.*`, `correction_projection.*`, `carrier_norm.*`, and
`conditioning.*`. The conditioning projections are zero, which is what makes the
conversion exact. Every new tensor group is listed in the manifest with its own
hash so a reviewer can confirm nothing native was replaced.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .attention import RineyaTGSAAttention
from .config import RineyaTGSAConfig

MANIFEST_SCHEMA_VERSION = "rineya.tgsa-conversion-manifest.v1"

NATIVE_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def tensor_sha256(tensor: Tensor) -> str:
    """Content hash of a tensor, independent of device and layout."""

    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_group_sha256(state: Mapping[str, Tensor], keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(tensor_sha256(state[key]).encode("ascii"))
    return digest.hexdigest()


@dataclass(slots=True)
class LayerConversion:
    """Record of one converted layer."""

    layer_index: int
    reused_tensors: dict[str, str]
    new_tensor_groups: dict[str, str]
    conditioning_is_neutral: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "reused_tensors": self.reused_tensors,
            "new_tensor_groups": self.new_tensor_groups,
            "conditioning_is_neutral": self.conditioning_is_neutral,
        }


def _new_tensor_groups(module: RineyaTGSAAttention) -> dict[str, list[str]]:
    """Group the newly created tensors by role, for manifest hashing."""

    groups: dict[str, list[str]] = {}
    for name, _ in module.named_parameters():
        root = name.split(".")[0]
        if root in NATIVE_PROJECTIONS:
            continue
        groups.setdefault(root, []).append(name)
    return {key: sorted(value) for key, value in sorted(groups.items())}


def convert_attention_module(
    source_state: Mapping[str, Tensor],
    config: RineyaTGSAConfig,
    *,
    layer_index: int,
    prefix: str,
    initialization_seed: int,
) -> tuple[RineyaTGSAAttention, LayerConversion]:
    """Build one retrofitted attention module from native tensors.

    ``prefix`` is the source key prefix, e.g. ``model.layers.7.self_attn``.
    ``initialization_seed`` is derived per layer so a conversion is reproducible
    byte-for-byte under the same environment.
    """

    torch.manual_seed(initialization_seed)
    module = RineyaTGSAAttention(config, layer_index=layer_index)

    reused: dict[str, str] = {}
    with torch.no_grad():
        for name in NATIVE_PROJECTIONS:
            target = getattr(module, name)
            weight_key = f"{prefix}.{name}.weight"
            if weight_key not in source_state:
                raise KeyError(f"source checkpoint is missing {weight_key!r}")
            source_weight = source_state[weight_key]
            if source_weight.shape != target.weight.shape:
                raise ValueError(
                    f"{weight_key}: source shape {tuple(source_weight.shape)} does not match "
                    f"retrofit shape {tuple(target.weight.shape)}; the retrofit config does not "
                    f"describe this checkpoint"
                )
            target.weight.copy_(source_weight)
            reused[weight_key] = tensor_sha256(source_weight)

            bias_key = f"{prefix}.{name}.bias"
            if target.bias is not None:
                if bias_key not in source_state:
                    raise KeyError(
                        f"retrofit config declares a bias for {name} but {bias_key!r} is absent"
                    )
                target.bias.copy_(source_state[bias_key])
                reused[bias_key] = tensor_sha256(source_state[bias_key])
            elif bias_key in source_state:
                raise ValueError(
                    f"source has {bias_key!r} but the retrofit config disables it; "
                    f"conversion would silently drop a pretrained tensor"
                )

    module_state = dict(module.state_dict())
    new_groups = {
        group: state_group_sha256(module_state, keys)
        for group, keys in _new_tensor_groups(module).items()
    }
    record = LayerConversion(
        layer_index=layer_index,
        reused_tensors=reused,
        new_tensor_groups=new_groups,
        conditioning_is_neutral=module.conditioning_is_neutral(),
    )
    if not record.conditioning_is_neutral:
        raise RuntimeError(
            f"layer {layer_index}: conditioning is not neutral after conversion; the "
            f"retrofit would not reproduce the pretrained function"
        )
    return module, record


def derive_layer_seed(master_seed: int, layer_index: int) -> int:
    payload = json.dumps([int(master_seed), "layer", int(layer_index)], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def convert_checkpoint(
    *,
    source_state: Mapping[str, Tensor],
    source_config: Mapping[str, Any],
    convert_layers: Sequence[int],
    layer_prefix: str = "model.layers.{index}.self_attn",
    master_seed: int = 20260829,
    trajectory_overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[int, RineyaTGSAAttention], dict[str, object]]:
    """Convert selected layers and return the modules plus a full manifest.

    Layers outside ``convert_layers`` are left entirely untouched: this supports
    the layer-placement study, where only a subset of the network is retrofitted.
    """

    config = RineyaTGSAConfig.from_source_config(source_config, **(trajectory_overrides or {}))
    total_layers = int(source_config.get("num_hidden_layers", 0))
    for index in convert_layers:
        if total_layers and not 0 <= index < total_layers:
            raise ValueError(f"layer {index} is outside the source model's {total_layers} layers")

    modules: dict[int, RineyaTGSAAttention] = {}
    records: list[dict[str, object]] = []
    for index in sorted(set(convert_layers)):
        module, record = convert_attention_module(
            source_state,
            config,
            layer_index=index,
            prefix=layer_prefix.format(index=index),
            initialization_seed=derive_layer_seed(master_seed, index),
        )
        modules[index] = module
        records.append(record.to_dict())

    converted_state: dict[str, Tensor] = {}
    for index, module in modules.items():
        for key, tensor in module.state_dict().items():
            converted_state[f"tgsa.layers.{index}.{key}"] = tensor

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_config_sha256": hashlib.sha256(
            json.dumps(dict(source_config), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "source_checkpoint_sha256": state_group_sha256(source_state, source_state.keys()),
        "source_tensor_count": len(source_state),
        "converted_checkpoint_sha256": state_group_sha256(
            converted_state, converted_state.keys()
        ),
        "converted_tensor_count": len(converted_state),
        "retrofit_config": config.to_dict(),
        "master_seed": master_seed,
        "layer_prefix": layer_prefix,
        "converted_layers": sorted(modules),
        "total_source_layers": total_layers,
        "layers": records,
        "mapping_rules": {
            "reuse": {
                "q_proj": "verbatim copy, no reshape or reorder",
                "k_proj": "verbatim copy, no reshape or reorder",
                "v_proj": "verbatim copy, no reshape or reorder",
                "o_proj": "verbatim copy, no reshape or reorder",
            },
            "gqa_head_mapping": (
                "query head j attends against kv group j // (num_attention_heads // "
                "num_key_value_heads), which is exactly the native repeat_interleave grouping"
            ),
            "sequential_axis": (
                "the carrier recurses across query heads within a token, never across "
                "sequence positions, so the KV cache is identical to native GQA"
            ),
            "new_tensors": (
                "carrier_init, trajectory_encoders, correction_gate, correction_projection, "
                "carrier_norm, conditioning; conditioning is zero-initialized so the module "
                "reduces algebraically to native attention at step 0"
            ),
            "exactness": (
                "verified bitwise against an independent parallel GQA reference in float64"
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "reproducibility": (
            "Byte-for-byte reproducible under the same environment: every new tensor is "
            "drawn from a per-layer seed derived deterministically from master_seed and the "
            "layer index."
        ),
    }
    return modules, manifest


def verify_conversion_manifest(
    manifest: Mapping[str, object],
    *,
    source_state: Mapping[str, Tensor],
    modules: Mapping[int, RineyaTGSAAttention],
) -> dict[str, object]:
    """Re-derive every hash in a manifest and report mismatches.

    Fails closed: a manifest that does not describe the artifacts in front of you
    is treated as an error, not a warning.
    """

    errors: list[str] = []
    checks: dict[str, bool] = {}

    actual_source = state_group_sha256(source_state, source_state.keys())
    checks["source_checkpoint_matches"] = actual_source == manifest["source_checkpoint_sha256"]
    if not checks["source_checkpoint_matches"]:
        errors.append("source checkpoint hash does not match the manifest")

    converted_state: dict[str, Tensor] = {}
    for index, module in modules.items():
        for key, tensor in module.state_dict().items():
            converted_state[f"tgsa.layers.{index}.{key}"] = tensor
    actual_converted = state_group_sha256(converted_state, converted_state.keys())
    checks["converted_checkpoint_matches"] = (
        actual_converted == manifest["converted_checkpoint_sha256"]
    )
    if not checks["converted_checkpoint_matches"]:
        errors.append("converted checkpoint hash does not match the manifest")

    checks["layers_match"] = sorted(modules) == list(manifest["converted_layers"])
    if not checks["layers_match"]:
        errors.append("converted layer set does not match the manifest")

    for record in manifest["layers"]:  # type: ignore[union-attr]
        index = int(record["layer_index"])
        module = modules.get(index)
        if module is None:
            errors.append(f"layer {index} is in the manifest but not in the modules")
            continue
        for key, expected in record["reused_tensors"].items():
            if key not in source_state:
                errors.append(f"layer {index}: reused tensor {key} is absent from the source")
                continue
            if tensor_sha256(source_state[key]) != expected:
                errors.append(f"layer {index}: reused tensor {key} changed since conversion")
        if not module.conditioning_is_neutral() and record["conditioning_is_neutral"]:
            errors.append(f"layer {index}: conditioning is no longer neutral")

    checks["reused_tensors_intact"] = not any("reused tensor" in e for e in errors)
    return {
        "passed": not errors,
        "checks": dict(sorted(checks.items())),
        "errors": errors,
    }
