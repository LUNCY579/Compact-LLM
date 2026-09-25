"""Exact parameter/storage accounting for FMA and its matched control."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import FMAConfig, ModelShape


@dataclass(frozen=True)
class Budget:
    vocab_size: int
    hidden_size: int
    value_width: int
    head_dim: int
    modified_layers: int
    rank: int
    shared_token_table_params: int
    layer_projection_params: int
    normalization_params: int
    added_fma_params: int
    transition_model_total_params: int | None
    hard_replacement_model_total_params: int | None
    removed_standard_value_params: int
    full_ma_params: int
    full_ma_bf16_bytes: int
    factorized_bf16_bytes: int
    control_rank: int
    control_added_params: int
    control_budget_delta_params: int
    control_budget_delta_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def nearest_control_rank(shape: ModelShape, layers: int, target: int) -> int:
    per_rank = layers * (shape.hidden_size + shape.value_width)
    return max(1, round(target / per_rank))


def parameter_budget(
    shape: ModelShape,
    cfg: FMAConfig,
    *,
    baseline_params: int | None = None,
    bytes_per_scalar: int = 2,
) -> Budget:
    modified = len(cfg.resolved_layers(shape))
    dv = shape.value_width
    shared = shape.vocab_size * cfg.rank
    projection = modified * cfg.rank * dv
    # Paper: RMSNorm independently within each KV head. The Rineya implementation
    # uses one learned d_h scale vector per layer, broadcast across KV heads.
    norm = modified * shape.head_dim if cfg.affine_head_rmsnorm else 0
    added = shared + projection + norm
    removed_wv = modified * shape.hidden_size * dv
    full_ma = modified * (shape.vocab_size * dv + (shape.head_dim if cfg.affine_head_rmsnorm else 0))
    control_rank = nearest_control_rank(shape, modified, added)
    control_added = modified * control_rank * (shape.hidden_size + dv)
    transition_total = baseline_params + added if baseline_params is not None else None
    hard_total = baseline_params - removed_wv + added if baseline_params is not None else None
    return Budget(
        vocab_size=shape.vocab_size,
        hidden_size=shape.hidden_size,
        value_width=dv,
        head_dim=shape.head_dim,
        modified_layers=modified,
        rank=cfg.rank,
        shared_token_table_params=shared,
        layer_projection_params=projection,
        normalization_params=norm,
        added_fma_params=added,
        transition_model_total_params=transition_total,
        hard_replacement_model_total_params=hard_total,
        removed_standard_value_params=removed_wv,
        full_ma_params=full_ma,
        full_ma_bf16_bytes=full_ma * bytes_per_scalar,
        factorized_bf16_bytes=added * bytes_per_scalar,
        control_rank=control_rank,
        control_added_params=control_added,
        control_budget_delta_params=control_added - added,
        control_budget_delta_fraction=(control_added - added) / added,
    )
