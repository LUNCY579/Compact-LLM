"""Tiny-model integration and paired initialization for TGSA + MTP Phase 2.

Phase 1 identifies the trajectory mechanism.  Phase 2 is intentionally a
separate factorial experiment: attention family (parallel vs TGSA) x decoding
training objective (AR vs MTP).  Keeping these axes separate prevents an MTP
optimization effect from being mistaken for a TGSA mechanism effect.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Literal, Mapping, cast

import torch
from torch import Tensor

from .mtp import MTPConfig, MultiTokenPredictionHeads
from .phase1_model import (
    ModelArm,
    TinyLMConfig,
    TinyTGSAForCausalLM,
    copy_shared_tensors,
    derive_initialization_seed,
    shared_state_fingerprint,
    state_dict_sha256,
)
from .telemetry import TGSATelemetry

AttentionFamily = Literal["parallel", "tgsa"]


@dataclass(frozen=True, slots=True)
class FactorialArmSpec:
    """One cell in the predeclared 2x2 complementarity design."""

    name: str
    attention_family: AttentionFamily
    attention_arm: ModelArm
    mtp_enabled: bool
    mtp_target_mode: Literal["future", "duplicate_next", "shuffled_batch"] = "future"
    primary: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def primary_factorial_arms(*, tgsa_arm: ModelArm = "full") -> tuple[FactorialArmSpec, ...]:
    if tgsa_arm == "parallel_mha":
        raise ValueError("tgsa_arm must select a TGSA mechanism, not parallel_mha")
    return (
        FactorialArmSpec("parallel_ar", "parallel", "parallel_mha", False),
        FactorialArmSpec("parallel_mtp", "parallel", "parallel_mha", True),
        FactorialArmSpec("tgsa_ar", "tgsa", tgsa_arm, False),
        FactorialArmSpec("tgsa_mtp", "tgsa", tgsa_arm, True),
    )


def auxiliary_controls(*, tgsa_arm: ModelArm = "full") -> tuple[FactorialArmSpec, ...]:
    """Non-gating controls that match auxiliary capacity without future targets."""

    return (
        FactorialArmSpec(
            "parallel_aux_next",
            "parallel",
            "parallel_mha",
            True,
            mtp_target_mode="duplicate_next",
            primary=False,
        ),
        FactorialArmSpec(
            "tgsa_aux_next",
            "tgsa",
            tgsa_arm,
            True,
            mtp_target_mode="duplicate_next",
            primary=False,
        ),
    )


@dataclass(slots=True)
class Phase2LMOutput:
    logits: Tensor
    mtp_logits: dict[int, Tensor]
    hidden_states: Tensor
    block_telemetry: tuple[TGSATelemetry | None, ...]


class TinyTGSAForCausalLMWithMTP(TinyTGSAForCausalLM):
    """The Phase-1 tiny causal LM plus always-present auxiliary proposal heads.

    AR and MTP cells instantiate the *same state structure*.  In AR cells the
    proposal heads are present but excluded from the objective and optimizer.
    This makes the within-attention-family initialization byte-identical and
    avoids treating extra parameter allocation as evidence for future-token
    supervision.
    """

    def __init__(
        self,
        config: TinyLMConfig,
        mtp_config: MTPConfig,
        *,
        mtp_initialization_seed: int,
    ) -> None:
        if mtp_config.hidden_size != config.embed_dim:
            raise ValueError("MTP hidden_size must equal the tiny model embed_dim")
        if mtp_config.vocab_size != config.vocab_size:
            raise ValueError("MTP vocab_size must equal the tiny model vocab_size")
        super().__init__(config)
        self.mtp_config = mtp_config
        # The auxiliary seed is independent of graph construction order and does
        # not advance the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(mtp_initialization_seed))
            self.mtp_heads = MultiTokenPredictionHeads(mtp_config)

    def forward(
        self,
        input_ids: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        telemetry_level: Literal["none", "summary", "full"] = "none",
    ) -> Phase2LMOutput:
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
        final_hidden = self.final_norm(hidden_states)
        logits = self.lm_head(final_hidden)
        mtp_logits = self.mtp_heads(
            final_hidden,
            output_weight=self.lm_head.weight,
            output_bias=self.lm_head.bias,
        )
        return Phase2LMOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            hidden_states=final_hidden,
            block_telemetry=tuple(telemetry),
        )

    def set_mtp_trainable(self, trainable: bool) -> None:
        for parameter in self.mtp_heads.parameters():
            parameter.requires_grad_(trainable)

    def mtp_state_sha256(self) -> str:
        keys = [key for key in self.state_dict() if key.startswith("mtp_heads.")]
        return state_dict_sha256(self, keys=keys)


@dataclass(slots=True)
class FactorialModelBundle:
    models: dict[str, TinyTGSAForCausalLMWithMTP]
    specs: dict[str, FactorialArmSpec]
    initialization_audit: dict[str, object]


def _build_one(
    config: TinyLMConfig,
    mtp_config: MTPConfig,
    *,
    model_seed: int,
    mtp_seed: int,
) -> TinyTGSAForCausalLMWithMTP:
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    return TinyTGSAForCausalLMWithMTP(
        config,
        mtp_config,
        mtp_initialization_seed=mtp_seed,
    )


def build_factorial_models(
    base_config: TinyLMConfig,
    mtp_config: MTPConfig,
    *,
    initialization_seed: int,
    tgsa_arm: ModelArm = "full",
    include_auxiliary_controls: bool = False,
) -> FactorialModelBundle:
    """Build all cells from two canonical family states and audit every copy.

    * Within an attention family, AR and MTP begin as deep copies: every byte is
      identical, including the dormant MTP heads.
    * Across attention families, every shape-compatible tensor is copied from
      the TGSA canonical source, so embeddings, FFNs, norms, LM head, and MTP
      heads are paired.  Only structurally incompatible attention tensors keep a
      derived-seed initialization.
    * Every MTP head has the same state in every arm.
    """

    specs = list(primary_factorial_arms(tgsa_arm=tgsa_arm))
    if include_auxiliary_controls:
        specs.extend(auxiliary_controls(tgsa_arm=tgsa_arm))
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise RuntimeError("factorial arm names are not unique")

    mtp_seed = derive_initialization_seed(initialization_seed, "phase2", "mtp")
    tgsa_seed = derive_initialization_seed(initialization_seed, "phase2", "tgsa")
    parallel_seed = derive_initialization_seed(initialization_seed, "phase2", "parallel")

    tgsa_canonical = _build_one(
        base_config.with_arm(tgsa_arm),
        mtp_config,
        model_seed=tgsa_seed,
        mtp_seed=mtp_seed,
    )
    parallel_canonical = _build_one(
        base_config.with_arm("parallel_mha"),
        mtp_config,
        model_seed=parallel_seed,
        mtp_seed=mtp_seed,
    )
    tgsa_state = {key: value.detach().clone() for key, value in tgsa_canonical.state_dict().items()}
    parallel_uncopied = copy_shared_tensors(parallel_canonical, tgsa_state)

    family_templates = {
        "parallel": parallel_canonical,
        "tgsa": tgsa_canonical,
    }
    models: dict[str, TinyTGSAForCausalLMWithMTP] = {
        spec.name: copy.deepcopy(family_templates[spec.attention_family]) for spec in specs
    }

    # AR cells keep dormant heads in the state dict but do not optimize them.
    for name, model in models.items():
        model.set_mtp_trainable(by_name[name].mtp_enabled)

    audit = factorial_initialization_audit(
        models,
        by_name,
        initialization_seed=initialization_seed,
        mtp_seed=mtp_seed,
        tgsa_seed=tgsa_seed,
        parallel_seed=parallel_seed,
        parallel_uncopied=parallel_uncopied,
    )
    return FactorialModelBundle(models=models, specs=by_name, initialization_audit=audit)


def _all_equal(values: list[str]) -> bool:
    return bool(values) and len(set(values)) == 1


def factorial_initialization_audit(
    models: Mapping[str, TinyTGSAForCausalLMWithMTP],
    specs: Mapping[str, FactorialArmSpec],
    *,
    initialization_seed: int,
    mtp_seed: int,
    tgsa_seed: int,
    parallel_seed: int,
    parallel_uncopied: tuple[str, ...] = (),
) -> dict[str, object]:
    required = {"parallel_ar", "parallel_mtp", "tgsa_ar", "tgsa_mtp"}
    if not required.issubset(models):
        raise ValueError(f"missing primary factorial cells: {sorted(required - set(models))}")

    full_hash = {name: state_dict_sha256(model) for name, model in sorted(models.items())}
    mtp_hash = {name: model.mtp_state_sha256() for name, model in sorted(models.items())}
    within_parallel = full_hash["parallel_ar"] == full_hash["parallel_mtp"]
    within_tgsa = full_hash["tgsa_ar"] == full_hash["tgsa_mtp"]
    mtp_identical = _all_equal(list(mtp_hash.values()))
    cross = shared_state_fingerprint(models["parallel_ar"], models["tgsa_ar"])
    cross_shared_identical = bool(cross["identical"])

    checks = {
        "parallel_ar_mtp_exact_state": within_parallel,
        "tgsa_ar_mtp_exact_state": within_tgsa,
        "mtp_heads_identical_across_all_cells": mtp_identical,
        "cross_family_shape_compatible_tensors_identical": cross_shared_identical,
        "all_specs_recorded": set(models) == set(specs),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"factorial initialization audit failed: {failed}")

    trainable = {
        name: sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        for name, model in sorted(models.items())
    }
    total = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in sorted(models.items())
    }
    return {
        "schema_version": "tgsa.mtp-factorial-initialization.v1",
        "checks": checks,
        "initialization_seed": initialization_seed,
        "derived_seeds": {
            "mtp": mtp_seed,
            "tgsa_family": tgsa_seed,
            "parallel_family": parallel_seed,
        },
        "state_sha256": full_hash,
        "mtp_state_sha256": mtp_hash,
        "cross_family_shared_state": cross,
        "parallel_structurally_unpaired_tensors": list(parallel_uncopied),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "specs": {name: spec.to_dict() for name, spec in sorted(specs.items())},
        "claim_scope": (
            "proves paired initialization and capacity accounting only; it is not "
            "evidence that TGSA, MTP, or their combination improves capability"
        ),
    }
