"""Mask helpers for compressed-cache chunked prefill."""

from __future__ import annotations

import torch


def bottom_right_causal_allowed(
    query_length: int,
    key_value_length: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a bottom-right-aligned boolean causal mask ``[q, kv]``.

    For an existing compressed cache followed by a new chunk, query row ``i``
    may attend to every resident old entry and chunk entries through ``i``.
    PyTorch SDPA's bare ``is_causal=True`` is top-left aligned when ``q != kv``
    and therefore cannot express this case.
    """

    if query_length < 0 or key_value_length < 0:
        raise ValueError("query_length and key_value_length must be non-negative")
    if key_value_length < query_length:
        raise ValueError(
            f"key/value length {key_value_length} cannot be shorter than query length {query_length}"
        )
    device = torch.device("cpu") if device is None else torch.device(device)
    offset = key_value_length - query_length
    q_idx = torch.arange(query_length, device=device).unsqueeze(-1)
    kv_idx = torch.arange(key_value_length, device=device).unsqueeze(0)
    return kv_idx <= (q_idx + offset)


def bottom_right_causal_mask(
    query_length: int,
    key_value_length: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive SDPA mask with shape ``[1, 1, q, kv]``."""

    if not dtype.is_floating_point:
        raise TypeError(f"additive attention masks require a floating dtype, got {dtype}")
    allowed = bottom_right_causal_allowed(query_length, key_value_length, device=device)
    mask = torch.zeros((query_length, key_value_length), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)
