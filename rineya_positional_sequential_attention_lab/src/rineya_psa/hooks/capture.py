"""Chunked attention statistics (SPEC 5.4).

The registry hands one query-row chunk at a time to :meth:`CaptureAccumulator.observe`
and immediately drops it, so the peak footprint is ``[B, H_q, chunk, T]`` and never
``[B, H_q, T, T]``. Every reduction is folded into a float64 numpy array the moment
the chunk is reduced; float64 because a 32k-token prompt contributes 32 768 row
terms per layer and float32 accumulation loses the tail.

**Normalisation.** Each *valid query row* carries weight exactly 1, pooled over
chunks, over batch elements, and over successive forward passes until
``PSAContext.reset_capture()``. Rows masked out by ``ctx.valid_mask`` (padding)
contribute nothing and are not counted. So:

``attn_by_bin[g, b]``
    mean over valid rows of the attention mass that row puts on absolute source
    bin ``b``. Rows sum to 1 across bins up to the mass on masked/padded columns,
    which is ~0 by construction.
``attn_entropy[g]``
    mean over valid rows of ``-sum_j p log p`` in nats.
``gold_span_attn[g]``
    mean over valid rows of the mass on ``ctx.evidence_spans``.
``qk_logit_mean[g, b]``
    mean over valid rows of the *within-row* mean pre-softmax logit in bin ``b``,
    counting only unmasked sources. Its denominator is per-bin: a row that has no
    unmasked source in bin ``b`` (the whole bin is in the row's future) is excluded
    from that bin's average instead of being scored as zero. This is the only
    reduction whose row count varies by bin, and it is why early query rows do not
    drag the late-bin means toward the masking floor.
``head_out_norm[g]``
    mean over valid rows of ``||O[g, i, :]||_2``, measured after Mode 3 because that
    is the vector ``o_proj`` actually consumes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rineya_psa.hooks.binning import PositionBinning
from rineya_psa.hooks.specs import CaptureSpec

#: Guards ``log(0)`` in the entropy reduction without biasing a real probability.
_LOG_FLOOR = 1e-30


def span_mask(
    spans: tuple[tuple[int, int], ...],
    kv_length: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[kv_length]`` 0/1 indicator of the union of ``spans`` (half-open, clipped)."""
    m = torch.zeros(kv_length, device=device, dtype=dtype)
    for start, end in spans:
        s = max(0, int(start))
        e = min(int(kv_length), int(end))
        if e > s:
            m[s:e] = 1.0
    return m


class CaptureAccumulator:
    """Running float64 sums keyed ``(layer, tensor_name)``."""

    def __init__(self, spec: CaptureSpec, binning: PositionBinning) -> None:
        self.spec = spec
        self.binning = binning
        self._sums: dict[tuple[int, str], np.ndarray] = {}
        self._rows: dict[tuple[int, str], np.ndarray] = {}
        # Cached per (kv_length, device) so a 32k prompt builds the indicator once.
        self._onehot: dict[tuple[int, str, str], torch.Tensor] = {}
        self._span: dict[tuple[int, str, int], torch.Tensor] = {}

    # -- internals ------------------------------------------------------------

    def _add(self, key: tuple[int, str], value: np.ndarray, rows: np.ndarray) -> None:
        if key in self._sums:
            self._sums[key] += value
            self._rows[key] += rows
        else:
            self._sums[key] = value.astype(np.float64, copy=True)
            self._rows[key] = np.asarray(rows, dtype=np.float64).copy()

    def _one_hot(self, kv_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (kv_length, str(device), str(dtype))
        cached = self._onehot.get(key)
        if cached is None:
            cached = self.binning.one_hot(kv_length, device=device, dtype=dtype)
            self._onehot[key] = cached
        return cached

    def _span_vec(
        self,
        spans: tuple[tuple[int, int], ...],
        kv_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (kv_length, str(device), hash((spans, str(dtype))))
        cached = self._span.get(key)
        if cached is None:
            cached = span_mask(spans, kv_length, device=device, dtype=dtype)
            self._span[key] = cached
        return cached

    # -- the per-chunk entry point -------------------------------------------

    def observe(
        self,
        layer: int,
        *,
        probs: torch.Tensor,
        logits: torch.Tensor,
        keep: torch.Tensor | None,
        head_out: torch.Tensor,
        row_valid: torch.Tensor | None,
        evidence_spans: tuple[tuple[int, int], ...] = (),
    ) -> None:
        """Reduce one query-row chunk.

        ``probs``/``logits`` are ``[B, H_q, q, T]`` post-softmax weights and the raw
        scaled ``q.k`` logits *before* the causal mask was added (so masked cells are
        finite and can simply be filtered by ``keep``). ``keep`` is a broadcastable
        bool of unmasked cells, ``head_out`` is ``[B, H_q, q, D]``, ``row_valid`` is
        ``[B, q]`` bool or None.
        """
        spec = self.spec
        if not spec.applies_to(layer):
            return
        B, n_q, q_rows, kv_len = probs.shape
        device, work_dtype = probs.device, torch.float32

        if row_valid is None:
            row_w = torch.ones((B, 1, q_rows, 1), device=device, dtype=work_dtype)
            n_rows = float(B * q_rows)
        else:
            row_w = row_valid.to(device=device, dtype=work_dtype).view(B, 1, q_rows, 1)
            n_rows = float(row_w.sum().item())
        if n_rows == 0.0:
            return

        p = probs.to(work_dtype)

        if spec.wants("attn_by_bin"):
            onehot = self._one_hot(kv_len, device, work_dtype)
            mass = torch.matmul(p, onehot) * row_w  # [B, H_q, q, n_bins]
            self._add(
                (layer, "attn_by_bin"),
                mass.sum(dim=(0, 2)).double().cpu().numpy(),
                np.float64(n_rows),
            )

        if spec.wants("attn_entropy"):
            ent = -(p * torch.log(p.clamp_min(_LOG_FLOOR))).sum(dim=-1, keepdim=True) * row_w
            self._add(
                (layer, "attn_entropy"),
                ent.sum(dim=(0, 2)).squeeze(-1).double().cpu().numpy(),
                np.float64(n_rows),
            )

        if spec.wants("gold_span_attn"):
            vec = self._span_vec(evidence_spans, kv_len, device, work_dtype)
            mass = torch.matmul(p, vec.view(kv_len, 1)) * row_w  # [B, H_q, q, 1]
            self._add(
                (layer, "gold_span_attn"),
                mass.sum(dim=(0, 2)).squeeze(-1).double().cpu().numpy(),
                np.float64(n_rows),
            )

        if spec.wants("qk_logit_mean"):
            onehot = self._one_hot(kv_len, device, work_dtype)
            if keep is None:
                allowed = torch.ones((1, 1, q_rows, kv_len), device=device, dtype=work_dtype)
            else:
                allowed = keep.to(work_dtype).expand(-1, -1, q_rows, kv_len)
            per_bin_count = torch.matmul(allowed, onehot)  # [B|1, 1, q, n_bins]
            z = logits.to(work_dtype) * allowed
            per_bin_sum = torch.matmul(z, onehot)  # [B, H_q, q, n_bins]
            row_mean = per_bin_sum / per_bin_count.clamp_min(1.0)
            contributes = (per_bin_count > 0).to(work_dtype) * row_w  # [B|1, 1, q, n_bins]
            self._add(
                (layer, "qk_logit_mean"),
                (row_mean * contributes).sum(dim=(0, 2)).double().cpu().numpy(),
                (contributes.expand(B, 1, q_rows, -1).sum(dim=(0, 1, 2)))
                .double()
                .cpu()
                .numpy(),
            )

        if spec.wants("head_out_norm"):
            norms = head_out.to(work_dtype).pow(2).sum(dim=-1, keepdim=True).sqrt() * row_w
            self._add(
                (layer, "head_out_norm"),
                norms.sum(dim=(0, 2)).squeeze(-1).double().cpu().numpy(),
                np.float64(n_rows),
            )

    # -- finalisation ---------------------------------------------------------

    def finalize(self) -> dict[tuple[int, str], np.ndarray]:
        out: dict[tuple[int, str], np.ndarray] = {}
        for key, total in self._sums.items():
            rows = self._rows[key]
            denom = np.maximum(np.asarray(rows, dtype=np.float64), 1.0)
            out[key] = np.asarray(total, dtype=np.float64) / denom
        return out

    def row_counts(self) -> dict[tuple[int, str], np.ndarray]:
        """Denominators actually used, for provenance and for spotting empty bins."""
        return {k: np.asarray(v, dtype=np.float64).copy() for k, v in self._rows.items()}


def ensure_accumulator(ctx: Any) -> CaptureAccumulator | None:
    """Lazily attach an accumulator to ``ctx`` on the first capturing forward."""
    if not ctx.capture_active():
        return None
    acc = ctx._acc
    if acc is None:
        acc = CaptureAccumulator(ctx.capture, ctx.capture_binning())
        ctx._acc = acc
    return acc


def finalize_captures(ctx: Any) -> dict[tuple[int, str], np.ndarray]:
    """Publish means onto ``ctx.captured``. Idempotent; safe with no accumulator."""
    acc = ctx._acc
    if acc is None:
        return ctx.captured
    ctx.captured = acc.finalize()
    return ctx.captured
