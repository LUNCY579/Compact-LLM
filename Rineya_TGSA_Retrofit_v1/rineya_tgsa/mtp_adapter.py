"""Multi-token prediction integration for the pretrained Rineya TGSA retrofit.

This module composes two deliberately separate mechanisms:

* the existing Qwen3/Rineya TGSA retrofit changes selected attention modules but
  is zero-impact at conversion; and
* MTP reads the final causal hidden state through auxiliary offset heads without
  changing the ordinary LM logits.

The wrapper is correctness-first.  It keeps the base model's output and cache
objects intact, exposes ``mtp_logits`` for offsets 2..K, supports exact greedy
block verification through ``tgsa.phase2_decode``, and saves adapter-only state
without duplicating native pretrained tensors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

try:
    from tgsa.mtp import MTPConfig, MultiTokenPredictionHeads, joint_causal_mtp_loss
except Exception as error:  # pragma: no cover - actionable import failure
    raise RuntimeError(
        "The combined retrofit requires the Phase-2 TGSA package. Install "
        "Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 in editable mode first."
    ) from error

COMBINED_MANIFEST_SCHEMA = "rineya.tgsa-mtp-retrofit.v1"
COMBINED_STAGE_ORDER = ("adapters", "attention", "converted_layers", "all")
TGSA_ADAPTER_ROOTS = (
    "carrier_init",
    "trajectory_encoders",
    "correction_gate",
    "correction_projection",
    "carrier_norm",
    "conditioning",
)
NATIVE_ATTENTION_ROOTS = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(state[name]).encode("ascii"))
    return digest.hexdigest()


@dataclass(slots=True)
class RineyaMTPOutput:
    """Minimal output contract consumed by training and verified decoding."""

    logits: Tensor
    mtp_logits: dict[int, Tensor]
    hidden_states: Tensor
    base_output: Any
    loss: Tensor | None = None

    @property
    def past_key_values(self) -> Any:
        return getattr(self.base_output, "past_key_values", None)


class RineyaTGSAWithMTP(nn.Module):
    """Wrap a converted or native causal LM with output-neutral MTP heads."""

    def __init__(
        self,
        base_model: nn.Module,
        mtp_config: MTPConfig,
        *,
        initialization_seed: int = 20260830,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)
        self.mtp_config = mtp_config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.mtp_heads = MultiTokenPredictionHeads(mtp_config)
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ValueError("base model must expose output embeddings with a weight tensor")
        weight = output_embeddings.weight
        if tuple(weight.shape) != (mtp_config.vocab_size, mtp_config.hidden_size):
            raise ValueError(
                "output embedding geometry does not match MTP config: "
                f"weight={tuple(weight.shape)} config="
                f"({mtp_config.vocab_size}, {mtp_config.hidden_size})"
            )

    def get_output_embeddings(self) -> nn.Module | None:
        getter = getattr(self.base_model, "get_output_embeddings", None)
        if callable(getter):
            return getter()
        return getattr(self.base_model, "lm_head", None)

    def prepare_inputs_for_generation(self, *args: object, **kwargs: object) -> object:
        function = getattr(self.base_model, "prepare_inputs_for_generation", None)
        if not callable(function):
            raise AttributeError("base model has no prepare_inputs_for_generation")
        return function(*args, **kwargs)

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        labels: Tensor | None = None,
        compute_joint_loss: bool = False,
        **kwargs: object,
    ) -> RineyaMTPOutput:
        call_kwargs = dict(kwargs)
        # Reference decoders pass this to native TGSA modules; it is not a Hugging
        # Face causal-LM argument and must not leak into the base model.
        call_kwargs.pop("telemetry_level", None)
        call_kwargs["output_hidden_states"] = True
        call_kwargs["return_dict"] = True
        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask
        # The wrapper owns loss construction so the base model cannot apply a
        # different prompt mask or shift convention behind our back.
        call_kwargs.pop("labels", None)
        base_output = self.base_model(input_ids=input_ids, **call_kwargs)
        logits = getattr(base_output, "logits", None)
        hidden_stack = getattr(base_output, "hidden_states", None)
        if not isinstance(logits, Tensor):
            raise TypeError("base model output has no tensor logits")
        if not hidden_stack:
            raise TypeError("base model did not return hidden states")
        final_hidden = hidden_stack[-1]
        if not isinstance(final_hidden, Tensor):
            raise TypeError("base model final hidden state is not a tensor")
        output_embeddings = self.get_output_embeddings()
        assert output_embeddings is not None and hasattr(output_embeddings, "weight")
        output_bias = getattr(output_embeddings, "bias", None)
        mtp_logits = self.mtp_heads(
            final_hidden,
            output_weight=output_embeddings.weight,
            output_bias=output_bias,
        )
        loss = None
        if compute_joint_loss:
            if input_ids is None:
                raise ValueError("compute_joint_loss requires input_ids")
            loss = joint_causal_mtp_loss(
                logits,
                mtp_logits,
                input_ids,
                self.mtp_config,
                attention_mask=attention_mask,
                target_mask=target_mask,
            ).total
        elif labels is not None:
            raise ValueError(
                "labels were supplied without compute_joint_loss=True; use target_mask to "
                "make the causal supervision boundary explicit"
            )
        return RineyaMTPOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            hidden_states=final_hidden,
            base_output=base_output,
            loss=loss,
        )

    def mtp_state_dict(self) -> dict[str, Tensor]:
        return {
            name.removeprefix("mtp_heads."): tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name.startswith("mtp_heads.")
        }

    def mtp_state_sha256(self) -> str:
        return state_sha256(self.mtp_state_dict())


@dataclass(frozen=True, slots=True)
class CombinedRetrofitConfig:
    max_offset: int = 4
    mtp_rank: int = 64
    mtp_loss_weight: float = 0.30
    mtp_offset_decay: float = 0.75
    mtp_detach_backbone: bool = False
    mtp_min_proposal_confidence: float = 0.0
    initialization_seed: int = 20260830

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def mtp_config_from_model(
    model: nn.Module,
    config: CombinedRetrofitConfig,
) -> MTPConfig:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise ValueError("model has no config")
    hidden_size = int(getattr(model_config, "hidden_size"))
    vocab_size = int(getattr(model_config, "vocab_size"))
    return MTPConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_offset=config.max_offset,
        rank=config.mtp_rank,
        loss_weight=config.mtp_loss_weight,
        offset_decay=config.mtp_offset_decay,
        detach_backbone=config.mtp_detach_backbone,
        tie_output_embeddings=True,
        min_proposal_confidence=config.mtp_min_proposal_confidence,
    )


def attach_mtp(
    model: nn.Module,
    config: CombinedRetrofitConfig = CombinedRetrofitConfig(),
) -> tuple[RineyaTGSAWithMTP, dict[str, object]]:
    mtp_config = mtp_config_from_model(model, config)
    wrapper = RineyaTGSAWithMTP(
        model,
        mtp_config,
        initialization_seed=config.initialization_seed,
    )
    manifest = {
        "schema_version": COMBINED_MANIFEST_SCHEMA,
        "component": "mtp_attachment",
        "config": config.to_dict(),
        "mtp_config": mtp_config.to_dict(),
        "mtp_state_sha256": wrapper.mtp_state_sha256(),
        "output_embedding_sha256": tensor_sha256(wrapper.get_output_embeddings().weight),  # type: ignore[union-attr]
        "main_logits_modified": False,
        "claim_scope": (
            "MTP heads were attached without replacing the main LM head; this records "
            "initialization only and is not evidence of future-token accuracy"
        ),
    }
    return wrapper, manifest


def convert_and_attach_tgsa_mtp(
    model: nn.Module,
    *,
    tgsa_config: Any,
    layers: Sequence[int],
    combined_config: CombinedRetrofitConfig = CombinedRetrofitConfig(),
    tgsa_master_seed: int = 20260829,
) -> tuple[RineyaTGSAWithMTP, dict[str, object]]:
    """Apply the exact Qwen3 TGSA conversion, then attach output-neutral MTP."""

    from .qwen3_adapter import convert_qwen3_model

    conversion = convert_qwen3_model(
        model,
        tgsa_config,
        layers=list(layers),
        master_seed=tgsa_master_seed,
    )
    wrapper, mtp_manifest = attach_mtp(model, combined_config)
    manifest = {
        "schema_version": COMBINED_MANIFEST_SCHEMA,
        "component": "tgsa_plus_mtp",
        "tgsa_conversion": conversion,
        "mtp_attachment": mtp_manifest,
        "converted_layers": sorted(set(int(layer) for layer in layers)),
        "neutrality_contract": {
            "tgsa": "trajectory conditioning is zero-initialized",
            "mtp": "auxiliary heads do not feed the main logits",
        },
        "claim_scope": (
            "combined conversion structure and tensor provenance; real-model logits must "
            "still be measured before and after conversion in the deployment dtype/path"
        ),
    }
    return wrapper, manifest


def _is_tgsa_adapter_name(name: str) -> bool:
    return any(f".{root}." in f".{name}." or name.endswith(f".{root}") for root in TGSA_ADAPTER_ROOTS)


def adapter_only_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return MTP and newly introduced TGSA tensors, never native weights."""

    selected: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("mtp_heads.") or _is_tgsa_adapter_name(name):
            selected[name] = tensor.detach().cpu()
    if not selected:
        raise ValueError("no TGSA/MTP adapter tensors were found")
    return selected


def apply_combined_stage(
    model: nn.Module,
    stage: Literal["adapters", "attention", "converted_layers", "all"],
    *,
    converted_layers: Sequence[int],
) -> dict[str, object]:
    """Progressively release adapters, native attention, converted blocks, or all."""

    if stage not in COMBINED_STAGE_ORDER:
        raise ValueError(f"unknown combined stage: {stage}")
    prefixes = tuple(f"base_model.model.layers.{int(index)}." for index in converted_layers)
    for name, parameter in model.named_parameters():
        in_mtp = name.startswith("mtp_heads.")
        in_converted = any(name.startswith(prefix) for prefix in prefixes)
        tgsa_adapter = in_converted and _is_tgsa_adapter_name(name)
        native_attention = in_converted and any(
            f".self_attn.{root}." in f".{name}." or name.endswith(f".self_attn.{root}.weight")
            for root in NATIVE_ATTENTION_ROOTS
        )
        if stage == "all":
            trainable = True
        elif stage == "adapters":
            trainable = in_mtp or tgsa_adapter
        elif stage == "attention":
            trainable = in_mtp or tgsa_adapter or native_attention
        else:
            trainable = in_mtp or in_converted
        parameter.requires_grad_(trainable)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {
        "stage": stage,
        "converted_layers": sorted(set(int(index) for index in converted_layers)),
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_tensors": sum(1 for parameter in model.parameters() if parameter.requires_grad),
    }


def audit_main_logit_neutrality(
    native_model: nn.Module,
    wrapped_model: RineyaTGSAWithMTP,
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> dict[str, object]:
    """Measure whether attaching MTP alone changes the main model logits."""

    native_training = native_model.training
    wrapped_training = wrapped_model.training
    native_model.eval()
    wrapped_model.eval()
    try:
        with torch.no_grad():
            native = native_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits
            wrapped = wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
    finally:
        wrapped_model.train(wrapped_training)
        native_model.train(native_training)
    difference = (native.float() - wrapped.float()).abs()
    return {
        "bitwise_identical": bool(torch.equal(native, wrapped)),
        "max_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
        "native_logits_sha256": tensor_sha256(native),
        "wrapped_logits_sha256": tensor_sha256(wrapped),
        "main_logits_modified_by_mtp": False,
    }
