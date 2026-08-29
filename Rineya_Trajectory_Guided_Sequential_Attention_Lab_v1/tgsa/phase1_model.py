"""Small causal language model and intervention tooling for TGSA Phase 1.

This module intentionally uses the existing auditable TGSA reference operator.
It supplies enough surrounding model structure to test learned capability rather
than only kernel execution, while remaining small enough for paired multi-seed
experiments on commodity hardware and CPU CI smoke tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import TGSAConfig
from .metrics import count_parameters
from .module import (
    ParallelCausalSelfAttention,
    RMSNorm,
    TrajectoryEncoder,
    TrajectoryGuidedSequentialAttention,
)
from .telemetry import TGSATelemetry

ModelArm = Literal[
    "parallel_mha",
    "cascade",
    "history_control",
    "raw_delta",
    "learned_delta",
    "qkv_conditioned",
    "predictor_corrector",
    "full",
]
InterventionMode = Literal["identity", "zero", "negate", "roll_batch", "detach"]


@dataclass(frozen=True, slots=True)
class TinyLMConfig:
    vocab_size: int
    max_sequence_length: int
    embed_dim: int = 96
    num_heads: int = 6
    num_layers: int = 2
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    bias: bool = False
    trajectory_rank: int = 16
    arm: ModelArm = "learned_delta"
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_sequence_length <= 1:
            raise ValueError("max_sequence_length must exceed one")
        if self.embed_dim <= 0 or self.num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.trajectory_rank <= 0:
            raise ValueError("trajectory_rank must be positive")
        if self.arm not in {
            "parallel_mha",
            "cascade",
            "history_control",
            "raw_delta",
            "learned_delta",
            "qkv_conditioned",
            "predictor_corrector",
            "full",
        }:
            raise ValueError(f"unsupported model arm: {self.arm}")

    def with_arm(self, arm: ModelArm) -> "TinyLMConfig":
        values = asdict(self)
        values["arm"] = arm
        return TinyLMConfig(**values)


@dataclass(slots=True)
class TinyLMOutput:
    logits: Tensor
    block_telemetry: tuple[TGSATelemetry | None, ...]


class TinyDecoderBlock(nn.Module):
    def __init__(self, config: TinyLMConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.arm = config.arm
        self.attention_norm = RMSNorm(config.embed_dim)
        if config.arm == "parallel_mha":
            self.attention: nn.Module = ParallelCausalSelfAttention(
                config.embed_dim,
                config.num_heads,
                dropout=config.dropout,
                bias=config.bias,
            )
        else:
            attention_config = TGSAConfig.for_ablation(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                arm=config.arm,  # type: ignore[arg-type]
                trajectory_rank=config.trajectory_rank,
                dropout=config.dropout,
                bias=config.bias,
                telemetry_level="none",
            )
            self.attention = TrajectoryGuidedSequentialAttention(attention_config)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = RMSNorm(config.embed_dim)
        hidden_dim = max(1, int(round(config.embed_dim * config.ffn_multiplier)))
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, config.embed_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        key_padding_mask: Tensor | None,
        telemetry_level: Literal["none", "summary", "full"] = "none",
    ) -> tuple[Tensor, TGSATelemetry | None]:
        normalized = self.attention_norm(hidden_states)
        telemetry: TGSATelemetry | None = None
        if isinstance(self.attention, TrajectoryGuidedSequentialAttention):
            if telemetry_level == "none":
                attention_output = self.attention(
                    normalized,
                    is_causal=True,
                    key_padding_mask=key_padding_mask,
                    telemetry_level="none",
                )
            else:
                attention_output, telemetry = self.attention(
                    normalized,
                    is_causal=True,
                    key_padding_mask=key_padding_mask,
                    telemetry_level=telemetry_level,
                )
        else:
            attention_output = self.attention(
                normalized,
                is_causal=True,
                key_padding_mask=key_padding_mask,
            )
        hidden_states = hidden_states + self.residual_dropout(attention_output)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, telemetry


class TinyTGSAForCausalLM(nn.Module):
    """Decoder-only model used for the controlled model-backed study."""

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TinyDecoderBlock(config, layer_index=index) for index in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        telemetry_level: Literal["none", "summary", "full"] = "none",
    ) -> TinyLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence = input_ids.shape
        if sequence > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence} exceeds configured maximum "
                f"{self.config.max_sequence_length}"
            )
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != input_ids.shape:
                raise ValueError("key_padding_mask must be bool and match input_ids")
        positions = torch.arange(sequence, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden_states = self.embedding_dropout(hidden_states)
        telemetry: list[TGSATelemetry | None] = []
        for block in self.blocks:
            hidden_states, block_telemetry = block(
                hidden_states,
                key_padding_mask=key_padding_mask,
                telemetry_level=telemetry_level,
            )
            telemetry.append(block_telemetry)
        logits = self.lm_head(self.final_norm(hidden_states))
        return TinyLMOutput(logits=logits, block_telemetry=tuple(telemetry))


def masked_answer_loss(logits: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or loss_mask.shape != labels.shape:
        raise ValueError("logits, labels, and loss_mask shapes are inconsistent")
    selected = loss_mask & labels.ne(-100)
    if not selected.any():
        raise ValueError("loss mask selects no labels")
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(labels)
    return token_losses[selected].mean()


def answer_logits(logits: Tensor, answer_input_indices: Tensor) -> Tensor:
    if logits.ndim != 3 or answer_input_indices.ndim != 1:
        raise ValueError("invalid answer-logit inputs")
    if logits.shape[0] != answer_input_indices.shape[0]:
        raise ValueError("batch dimension mismatch")
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, answer_input_indices]


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def state_dict_sha256(module: nn.Module, *, keys: Sequence[str] | None = None) -> str:
    state = module.state_dict()
    selected = sorted(state) if keys is None else sorted(keys)
    digest = hashlib.sha256()
    for key in selected:
        if key not in state:
            raise KeyError(key)
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def state_key_shape_signature(module: nn.Module) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return tuple(
        (name, str(tensor.dtype), tuple(tensor.shape))
        for name, tensor in sorted(module.state_dict().items())
    )


def shared_state_fingerprint(left: nn.Module, right: nn.Module) -> dict[str, object]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    common = sorted(
        key
        for key in left_state.keys() & right_state.keys()
        if left_state[key].shape == right_state[key].shape
        and left_state[key].dtype == right_state[key].dtype
    )
    return {
        "common_tensor_count": len(common),
        "left_common_sha256": state_dict_sha256(left, keys=common),
        "right_common_sha256": state_dict_sha256(right, keys=common),
        "identical": all(torch.equal(left_state[key], right_state[key]) for key in common),
        "left_only": sorted(left_state.keys() - right_state.keys()),
        "right_only": sorted(right_state.keys() - left_state.keys()),
    }


def build_model(config: TinyLMConfig, *, initialization_seed: int) -> TinyTGSAForCausalLM:
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    return TinyTGSAForCausalLM(config)


def paired_initialization_audit(
    *,
    base_config: TinyLMConfig,
    initialization_seed: int,
) -> dict[str, object]:
    """Prove the finite-difference and history controls start identically."""

    history = build_model(base_config.with_arm("history_control"), initialization_seed=initialization_seed)
    delta = build_model(base_config.with_arm("learned_delta"), initialization_seed=initialization_seed)
    if state_key_shape_signature(history) != state_key_shape_signature(delta):
        raise RuntimeError("history_control and learned_delta state structures differ")
    history_hash = state_dict_sha256(history)
    delta_hash = state_dict_sha256(delta)
    if history_hash != delta_hash:
        raise RuntimeError("history_control and learned_delta did not receive identical initialization")

    cascade = build_model(base_config.with_arm("cascade"), initialization_seed=initialization_seed)
    shared = shared_state_fingerprint(cascade, delta)
    if not bool(shared["identical"]):
        raise RuntimeError("cascade and learned_delta shared parameters differ at initialization")
    return {
        "history_delta_exact_state_structure": True,
        "history_delta_exact_initialization": True,
        "history_delta_state_sha256": history_hash,
        "history_trainable_parameters": count_parameters(history),
        "learned_delta_trainable_parameters": count_parameters(delta),
        "history_delta_parameter_count_equal": count_parameters(history) == count_parameters(delta),
        "cascade_trainable_parameters": count_parameters(cascade),
        "cascade_delta_shared_initialization": shared,
    }


def parameter_audit(module: nn.Module) -> dict[str, object]:
    groups: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        root = name.split(".", 1)[0]
        groups[root] = groups.get(root, 0) + parameter.numel()
    trajectory = sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if "trajectory_encoders" in name
    )
    return {
        "total_parameters": count_parameters(module, trainable_only=False),
        "trainable_parameters": count_parameters(module, trainable_only=True),
        "trajectory_encoder_parameters": trajectory,
        "top_level_parameter_groups": dict(sorted(groups.items())),
    }


def gradient_group_norms(module: nn.Module) -> dict[str, float]:
    accumulators: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        if "trajectory_encoders" in name:
            group = "trajectory_encoder"
        elif ".attention." in name:
            group = "attention_other"
        elif "ffn" in name:
            group = "ffn"
        elif "embedding" in name:
            group = "embedding"
        elif "lm_head" in name:
            group = "lm_head"
        else:
            group = "other"
        norm = float(parameter.grad.detach().float().norm().item())
        accumulators[group] = accumulators.get(group, 0.0) + norm * norm
    return {name: math.sqrt(value) for name, value in sorted(accumulators.items())}


class ControlIntervention(AbstractContextManager["ControlIntervention"]):
    """Intercept raw control signals at every trajectory encoder.

    `roll_batch` is the primary alignment falsifier: it preserves the empirical
    distribution and tensor shape of the control signal while assigning each
    example another example's trajectory. It never rolls across sequence
    positions, so it cannot create future-token leakage.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: InterventionMode,
        *,
        capture_records: bool = False,
        max_records: int = 256,
    ) -> None:
        if mode not in {"identity", "zero", "negate", "roll_batch", "detach"}:
            raise ValueError(f"unsupported intervention mode: {mode}")
        if max_records < 0:
            raise ValueError("max_records must be non-negative")
        self.model = model
        self.mode = mode
        self.capture_records = capture_records
        self.max_records = max_records
        self.records: list[dict[str, object]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._module_names = {id(module): name for name, module in model.named_modules()}

    def _transform(self, value: Tensor) -> Tensor:
        if self.mode == "identity":
            return value
        if self.mode == "zero":
            return torch.zeros_like(value)
        if self.mode == "negate":
            return -value
        if self.mode == "detach":
            return value.detach()
        if value.shape[0] < 2:
            raise ValueError("roll_batch intervention requires evaluation batch size >= 2")
        return value.roll(shifts=1, dims=0)

    def _hook(self, module: nn.Module, inputs: tuple[object, ...]) -> tuple[object, ...]:
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise RuntimeError("unexpected TrajectoryEncoder input contract")
        original = inputs[0]
        transformed = self._transform(original)
        if self.capture_records and len(self.records) < self.max_records:
            original_float = original.detach().float()
            transformed_float = transformed.detach().float()
            flat_original = original_float.flatten()
            flat_transformed = transformed_float.flatten()
            denominator = flat_original.norm() * flat_transformed.norm()
            cosine = (
                float((flat_original @ flat_transformed / denominator.clamp_min(1.0e-12)).item())
                if flat_original.numel()
                else float("nan")
            )
            self.records.append(
                {
                    "module": self._module_names.get(id(module), "<unknown>"),
                    "mode": self.mode,
                    "shape": list(original.shape),
                    "input_rms": float(original_float.square().mean().sqrt().item()),
                    "output_rms": float(transformed_float.square().mean().sqrt().item()),
                    "input_output_cosine": cosine,
                }
            )
        return (transformed,)

    def __enter__(self) -> "ControlIntervention":
        if self._handles:
            raise RuntimeError("ControlIntervention cannot be entered twice")
        for module in self.model.modules():
            if isinstance(module, TrajectoryEncoder):
                self._handles.append(module.register_forward_pre_hook(self._hook))
        if not self._handles and self.mode != "identity":
            raise ValueError("model has no learned/raw trajectory encoders to intervene on")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return None


def model_debug_snapshot(model: TinyTGSAForCausalLM) -> dict[str, object]:
    return {
        "config": asdict(model.config),
        "parameter_audit": parameter_audit(model),
        "state_dict_sha256": state_dict_sha256(model),
        "training": model.training,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
    }
