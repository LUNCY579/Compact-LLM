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
from typing import Literal, Mapping, Sequence, cast

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
        # `apply` walks every nn.Linear, including the trajectory-encoder factors,
        # so the operator's registered initialization must be restored afterwards.
        # Without this the zero-initialized residual is replaced by a generic
        # N(0, 0.02) draw and the encoder is no longer the identity at step 0.
        for block in self.blocks:
            if isinstance(block.attention, TrajectoryGuidedSequentialAttention):
                block.attention.reset_registered_parameters()
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


CANONICAL_SUPERNET_ARM: ModelArm = "learned_delta"


def derive_initialization_seed(initialization_seed: int, *parts: object) -> int:
    """Derive a recorded, arm-specific seed for tensors the supernet lacks."""

    payload = json.dumps(
        [int(initialization_seed), *[str(part) for part in parts]],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def _seeded(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_canonical_supernet(
    config: TinyLMConfig,
    *,
    initialization_seed: int,
) -> TinyTGSAForCausalLM:
    """Build the single canonical parameter source for one seed.

    Every arm in a seed is initialized by copying from this model, so shared
    tensors are identical by construction rather than by coincident RNG order.
    """

    _seeded(initialization_seed)
    return TinyTGSAForCausalLM(config.with_arm(CANONICAL_SUPERNET_ARM))


def copy_shared_tensors(
    target: nn.Module,
    canonical_state: Mapping[str, Tensor],
) -> tuple[str, ...]:
    """Copy every exactly-compatible tensor from ``canonical_state``.

    A key is copied only when it exists on both sides with the same dtype and
    the same shape. Returns the sorted keys that were **not** copied, i.e. the
    arm-only tensors that keep their derived-seed initialization.
    """

    target_state = target.state_dict()
    copied: set[str] = set()
    with torch.no_grad():
        for key, tensor in target_state.items():
            source = canonical_state.get(key)
            if source is None:
                continue
            if source.shape != tensor.shape or source.dtype != tensor.dtype:
                continue
            tensor.copy_(source)
            copied.add(key)
    return tuple(sorted(set(target_state) - copied))


def build_model(
    config: TinyLMConfig,
    *,
    initialization_seed: int,
    canonical_state: Mapping[str, Tensor] | None = None,
) -> TinyTGSAForCausalLM:
    """Construct one arm from the seed's canonical supernet.

    ``build_model`` never relies on two arms consuming the RNG stream in the
    same order. It builds (or accepts) the canonical ``learned_delta`` supernet
    for the seed and copies every exactly-compatible tensor by key. Arm-only
    tensors are constructed under an explicitly derived, recorded seed.

    See ``PHASE1_LAUNCH_AUDIT.md`` finding P0-2: the pre-remediation path merely
    reset the global seed before building structurally different graphs, so the
    learned trajectory factors consumed random numbers that the cascade arm did
    not, and shared tensors silently diverged.
    """

    if canonical_state is None:
        canonical = build_canonical_supernet(config, initialization_seed=initialization_seed)
        if config.arm == CANONICAL_SUPERNET_ARM:
            return canonical
        canonical_state = canonical.state_dict()
    elif config.arm == CANONICAL_SUPERNET_ARM:
        _seeded(initialization_seed)
        model = TinyTGSAForCausalLM(config)
        copy_shared_tensors(model, canonical_state)
        return model

    _seeded(derive_initialization_seed(initialization_seed, "arm", config.arm))
    model = TinyTGSAForCausalLM(config)
    copy_shared_tensors(model, canonical_state)
    return model


def paired_initialization_audit(
    *,
    base_config: TinyLMConfig,
    initialization_seed: int,
) -> dict[str, object]:
    """Prove every registered arm starts from one canonical parameter source."""

    canonical = build_canonical_supernet(base_config, initialization_seed=initialization_seed)
    canonical_state = {key: value.clone() for key, value in canonical.state_dict().items()}
    canonical_hash = state_dict_sha256(canonical)

    arms: dict[str, TinyTGSAForCausalLM] = {
        arm: build_model(
            base_config.with_arm(cast(ModelArm, arm)),
            initialization_seed=initialization_seed,
            canonical_state=canonical_state,
        )
        for arm in ("cascade", "history_control", "learned_delta")
    }
    history = arms["history_control"]
    delta = arms["learned_delta"]
    cascade = arms["cascade"]

    if state_key_shape_signature(history) != state_key_shape_signature(delta):
        raise RuntimeError("history_control and learned_delta state structures differ")
    history_hash = state_dict_sha256(history)
    delta_hash = state_dict_sha256(delta)
    if history_hash != delta_hash:
        raise RuntimeError("history_control and learned_delta did not receive identical initialization")
    if delta_hash != canonical_hash:
        raise RuntimeError("learned_delta did not reproduce the canonical supernet state")

    shared = shared_state_fingerprint(cascade, delta)
    if not bool(shared["identical"]):
        raise RuntimeError("cascade and learned_delta shared parameters differ at initialization")

    # Every registered arm must be a byte-exact restriction of the canonical
    # supernet on its shared keys. Anything else means an arm was initialized
    # from an independent RNG draw.
    per_arm: dict[str, object] = {}
    for arm, model in sorted(arms.items()):
        fingerprint = shared_state_fingerprint(model, canonical)
        if not bool(fingerprint["identical"]):
            raise RuntimeError(f"{arm} shared tensors differ from the canonical supernet")
        arm_only = sorted(set(model.state_dict()) - set(canonical_state))
        per_arm[arm] = {
            "shared_tensor_count": fingerprint["common_tensor_count"],
            "shared_sha256": fingerprint["left_common_sha256"],
            "identical_to_canonical": True,
            "arm_only_tensors": arm_only,
            "arm_only_initialization_seed": derive_initialization_seed(
                initialization_seed, "arm", arm
            ),
            "trainable_parameters": count_parameters(model),
            "state_sha256": state_dict_sha256(model),
        }

    return {
        "canonical_supernet_arm": CANONICAL_SUPERNET_ARM,
        "canonical_state_sha256": canonical_hash,
        "canonical_initialization_seed": initialization_seed,
        "all_arms_match_canonical_shared_tensors": True,
        "history_delta_exact_state_structure": True,
        "history_delta_exact_initialization": True,
        "history_delta_state_sha256": history_hash,
        "history_trainable_parameters": count_parameters(history),
        "learned_delta_trainable_parameters": count_parameters(delta),
        "history_delta_parameter_count_equal": count_parameters(history) == count_parameters(delta),
        "cascade_trainable_parameters": count_parameters(cascade),
        "cascade_delta_shared_initialization": shared,
        "per_arm": per_arm,
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


def length_matched_derangement(
    group_keys: Sequence[object],
    *,
    derangement_index: int = 0,
) -> tuple[list[int], list[bool]]:
    """Build a fixed, fixed-point-free donor assignment inside equal groups.

    Examples are grouped by ``group_keys`` — the registered stratum paired with
    the exact valid sequence length — and each group of size ``k >= 2`` receives
    a cyclic shift by ``1 + (derangement_index % (k - 1))``. Donor and receiver
    therefore always share a valid length, so no valid position can ever be
    handed a donor's padded-position control.

    ``derangement_index`` selects one of several deterministic permutations so
    the alignment contrast can be shown robust to the choice rather than resting
    on a single arbitrary permutation.

    Returns ``(donor_index, applied)``. A group of size one has no valid donor:
    those rows keep their own control and are reported as ``applied=False`` so
    coverage is measured rather than silently assumed. Leaving them unperturbed
    contributes a zero paired difference, which biases the aligned-minus-rolled
    contrast toward zero and therefore cannot manufacture a positive result.

    See ``PHASE1_LAUNCH_AUDIT.md`` finding P1-4: the pre-remediation
    ``roll_batch`` shifted controls by one row inside heterogeneous padded
    batches, so a valid position could receive a donor's padded-position value.
    """

    if derangement_index < 0:
        raise ValueError("derangement_index must be non-negative")
    members: dict[object, list[int]] = {}
    for position, key in enumerate(group_keys):
        members.setdefault(key, []).append(position)
    donor = list(range(len(group_keys)))
    applied = [False] * len(group_keys)
    for key in sorted(members, key=repr):
        rows = members[key]
        size = len(rows)
        if size < 2:
            continue
        shift = 1 + (derangement_index % (size - 1))
        for offset, row in enumerate(rows):
            donor[row] = rows[(offset + shift) % size]
            applied[row] = True
    return donor, applied


class ControlCapture(AbstractContextManager["ControlCapture"]):
    """Record every trajectory-encoder input from one unperturbed forward pass.

    Each ``TrajectoryEncoder`` is invoked exactly once per forward pass, so the
    module identity is a sufficient key. The captured tensors are the aligned
    control signals that :class:`ReplayIntervention` later hands to other
    examples, which is what makes the falsifier a clean counterfactual instead
    of a compounding perturbation.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.captured: dict[int, Tensor] = {}
        self.order: list[int] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module: nn.Module, inputs: tuple[object, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise RuntimeError("unexpected TrajectoryEncoder input contract")
        key = id(module)
        if key in self.captured:
            raise RuntimeError("TrajectoryEncoder was invoked twice in one capture pass")
        self.captured[key] = inputs[0].detach().clone()
        self.order.append(key)

    def __enter__(self) -> "ControlCapture":
        if self._handles:
            raise RuntimeError("ControlCapture cannot be entered twice")
        for module in self.model.modules():
            if isinstance(module, TrajectoryEncoder):
                self._handles.append(module.register_forward_pre_hook(self._hook))
        if not self._handles:
            raise ValueError("model has no trajectory encoders to capture")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return None


class ReplayIntervention(AbstractContextManager["ReplayIntervention"]):
    """Replace each example's control with a donor's *captured aligned* control.

    Every donor signal comes from an unperturbed forward pass, so head ``j``'s
    substituted control is not itself a product of the intervention at heads
    ``< j``. Rows with ``applied=False`` keep their own control unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        captured: Mapping[int, Tensor],
        donor_index: Sequence[int],
        applied: Sequence[bool],
    ) -> None:
        if len(donor_index) != len(applied):
            raise ValueError("donor_index and applied must have equal length")
        self.model = model
        self.captured = dict(captured)
        self.donor_index = list(int(value) for value in donor_index)
        self.applied = [bool(value) for value in applied]
        self.records: list[dict[str, object]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._module_names = {id(module): name for name, module in model.named_modules()}

    def _hook(self, module: nn.Module, inputs: tuple[object, ...]) -> tuple[object, ...]:
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise RuntimeError("unexpected TrajectoryEncoder input contract")
        original = inputs[0]
        donor_source = self.captured.get(id(module))
        if donor_source is None:
            raise RuntimeError("no captured aligned control for this trajectory encoder")
        if donor_source.shape != original.shape:
            raise RuntimeError("captured control shape does not match the replay pass")
        if len(self.donor_index) != original.shape[0]:
            raise ValueError("donor assignment does not match the evaluation batch size")
        index = torch.tensor(self.donor_index, device=original.device, dtype=torch.long)
        donated = donor_source.to(device=original.device, dtype=original.dtype).index_select(0, index)
        keep = torch.tensor(
            [not value for value in self.applied],
            device=original.device,
            dtype=torch.bool,
        )[:, None, None]
        transformed = torch.where(keep, original, donated)
        self.records.append(
            {
                "module": self._module_names.get(id(module), "<unknown>"),
                "mode": "replay_derangement",
                "shape": list(original.shape),
                "rows_intervened": int(sum(self.applied)),
                "rows_total": len(self.applied),
                "input_rms": float(original.detach().float().square().mean().sqrt().item()),
                "output_rms": float(transformed.detach().float().square().mean().sqrt().item()),
            }
        )
        return (transformed,)

    def __enter__(self) -> "ReplayIntervention":
        if self._handles:
            raise RuntimeError("ReplayIntervention cannot be entered twice")
        for module in self.model.modules():
            if isinstance(module, TrajectoryEncoder):
                self._handles.append(module.register_forward_pre_hook(self._hook))
        if not self._handles:
            raise ValueError("model has no trajectory encoders to intervene on")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return None


class ControlIntervention(AbstractContextManager["ControlIntervention"]):
    """Intercept raw control signals at every trajectory encoder.

    These are the *descriptive* interventions. ``zero`` and ``negate`` probe
    dependence and directional sensitivity but introduce out-of-distribution
    control values.

    ``roll_batch`` is retained as a diagnostic with an explicitly narrowed
    interpretation. It shifts controls by one row inside a heterogeneous padded
    batch, so a valid position can receive a donor's padded-position control,
    and later heads see controls that are already products of the intervention
    at earlier heads. The registered alignment falsifier is instead
    :class:`ReplayIntervention` driven by :func:`length_matched_derangement`.
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
