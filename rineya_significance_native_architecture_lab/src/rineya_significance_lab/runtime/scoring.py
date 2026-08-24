"""Post-RoPE grouped-query significance scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .masks import bottom_right_causal_allowed


def _reshape_attention_term(
    term: torch.Tensor,
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    groups: int,
    query_length: int,
    window: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize an attention mask/bias to ``[b, kv, groups|1, w, n]``."""

    if term.ndim == 2:
        term = term.view(1, 1, *term.shape)
    elif term.ndim == 3:
        term = term.unsqueeze(1)
    elif term.ndim != 4:
        raise ValueError(f"attention mask/bias must have rank 2, 3, or 4; got {term.ndim}")
    if term.shape[0] not in (1, batch):
        raise ValueError(f"attention mask batch {term.shape[0]} does not broadcast to {batch}")
    if term.shape[-2] < query_length or term.shape[-1] != key_length:
        raise ValueError(
            f"attention mask tail {tuple(term.shape[-2:])} is incompatible with "
            f"query/key lengths {(query_length, key_length)}"
        )
    term = term[..., -window:, :].to(device=device).expand(batch, -1, -1, -1)
    mask_heads = term.shape[1]
    if mask_heads == 1:
        return term.unsqueeze(2)
    if mask_heads == kv_heads:
        return term.unsqueeze(2)
    if mask_heads == query_heads:
        return term.reshape(batch, kv_heads, groups, window, key_length)
    raise ValueError(
        f"attention mask head dimension {mask_heads} must be 1, {kv_heads} KV heads, "
        f"or {query_heads} query heads"
    )


@dataclass(frozen=True)
class PostRopeGQASignificanceDiagnostics:
    """Auxiliary attention statistics from the exact scoring distribution."""

    contribution: torch.Tensor
    attention_entropy_nats_mean: float
    attention_entropy_normalized_mean: float
    attention_entropy_nats_minimum: float
    attention_entropy_nats_maximum: float


def _post_rope_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    window: int = 16,
    scaling: float | None = None,
    attention_mask: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return scoring attention as ``[batch, KV heads, groups, window, positions]``.

    ``query`` must be post-RoPE ``[b, Hq, q, d]`` and ``key`` must be post-RoPE,
    pre-``repeat_kv`` ``[b, Hkv, n, d]``. Attention received is summed over the
    query heads sharing a KV head and over the trailing observation rows.
    Computation is deliberately float32, matching the audited implementation.
    """

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(
            f"query/key must have shape [batch, heads, positions, dim], got "
            f"{tuple(query.shape)} and {tuple(key.shape)}"
        )
    batch, query_heads, query_length, head_dim = query.shape
    key_batch, kv_heads, key_length, key_dim = key.shape
    if batch != key_batch or head_dim != key_dim:
        raise ValueError("query/key batch and head dimension must match")
    if query_length <= 0 or key_length <= 0:
        raise ValueError("query and key sequences must be non-empty")
    if query_heads % kv_heads:
        raise ValueError(f"{query_heads} query heads are not divisible by {kv_heads} KV heads")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if query.device != key.device:
        raise ValueError(f"query and key must share a device, got {query.device} and {key.device}")
    groups = query_heads // kv_heads
    observed = min(int(window), query_length)
    scale = 1.0 / math.sqrt(head_dim) if scaling is None else float(scaling)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scaling must be finite and positive, got {scale}")

    q = query[:, :, -observed:, :].reshape(batch, kv_heads, groups, observed, head_dim).float()
    k = key.float()
    logits = torch.einsum("bgcwd,bgnd->bgcwn", q, k) * scale

    if attention_mask is None:
        allowed = bottom_right_causal_allowed(observed, key_length, device=query.device)
        logits = logits.masked_fill(~allowed.view(1, 1, 1, observed, key_length), float("-inf"))
    else:
        mask = _reshape_attention_term(
            attention_mask,
            batch=batch,
            query_heads=query_heads,
            kv_heads=kv_heads,
            groups=groups,
            query_length=query_length,
            window=observed,
            key_length=key_length,
            device=query.device,
        )
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        elif mask.is_floating_point():
            logits = logits + mask.float()
        else:
            raise TypeError(f"attention mask must be boolean or floating point, got {mask.dtype}")

    if position_bias is not None:
        bias = _reshape_attention_term(
            position_bias,
            batch=batch,
            query_heads=query_heads,
            kv_heads=kv_heads,
            groups=groups,
            query_length=query_length,
            window=observed,
            key_length=key_length,
            device=query.device,
        )
        if not bias.is_floating_point():
            raise TypeError(f"position bias must be floating point, got {bias.dtype}")
        logits = logits + bias.float()

    if torch.isneginf(logits).all(dim=-1).any():
        raise RuntimeError("at least one observed query row has no attendable key")
    attention = logits.softmax(dim=-1)
    if not torch.isfinite(attention).all():
        raise RuntimeError("auxiliary attention softmax produced non-finite values")
    return attention


def post_rope_gqa_significance(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    window: int = 16,
    scaling: float | None = None,
    attention_mask: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute one significance contribution ``[batch, KV heads, positions]``."""

    attention = _post_rope_gqa_attention(
        query,
        key,
        window=window,
        scaling=scaling,
        attention_mask=attention_mask,
        position_bias=position_bias,
    )
    return attention.sum(dim=(2, 3)).contiguous()


def post_rope_gqa_significance_diagnostics(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    window: int = 16,
    scaling: float | None = None,
    attention_mask: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
) -> PostRopeGQASignificanceDiagnostics:
    """Return the exact score contribution plus attention-entropy diagnostics."""

    attention = _post_rope_gqa_attention(
        query,
        key,
        window=window,
        scaling=scaling,
        attention_mask=attention_mask,
        position_bias=position_bias,
    )
    entropy = -(
        attention * attention.clamp_min(torch.finfo(attention.dtype).tiny).log()
    ).sum(dim=-1)
    attendable = (attention > 0).sum(dim=-1)
    denominator = attendable.clamp_min(2).to(torch.float32).log()
    normalized = torch.where(
        attendable > 1,
        entropy / denominator,
        torch.zeros_like(entropy),
    )
    if not torch.isfinite(entropy).all() or not torch.isfinite(normalized).all():
        raise RuntimeError("attention entropy diagnostics produced non-finite values")
    return PostRopeGQASignificanceDiagnostics(
        contribution=attention.sum(dim=(2, 3)).contiguous(),
        attention_entropy_nats_mean=float(entropy.mean().item()),
        attention_entropy_normalized_mean=float(normalized.mean().item()),
        attention_entropy_nats_minimum=float(entropy.min().item()),
        attention_entropy_nats_maximum=float(entropy.max().item()),
    )
