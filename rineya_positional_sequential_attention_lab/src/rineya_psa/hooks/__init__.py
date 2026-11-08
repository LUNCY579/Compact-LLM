"""Attention hook framework (SPEC section 5).

Public surface::

    from rineya_psa.hooks import (
        AbsoluteBinning, DistanceBinning, PositionBinning,
        LogitBiasSpec, ValueScaleSpec, HeadScaleSpec, CaptureSpec, PSAContext,
        psa_session, register_psa,
    )

Importing this package pulls in torch and transformers, which SPEC section 2 permits
only under ``hooks/``. Analysis modules that must stay torch-free import
``rineya_psa.types`` instead.
"""

from __future__ import annotations

from rineya_psa.hooks.binning import (
    CHUNK_BUDGET_BYTES,
    DEFAULT_CHUNK_ROWS,
    AbsoluteBinning,
    DistanceBinning,
    PositionBinning,
    binning_from_describe,
)
from rineya_psa.hooks.capture import CaptureAccumulator, finalize_captures, span_mask
from rineya_psa.hooks.registry import (
    PSA_KEY,
    attention_modules,
    is_registered,
    psa_attention_forward,
    psa_session,
    register_psa,
)
from rineya_psa.hooks.specs import (
    CAPTURE_KINDS,
    DEFAULT_ATTENUATION_EPS,
    CaptureSpec,
    HeadScaleSpec,
    LogitBiasSpec,
    PSAContext,
    ValueScaleSpec,
)

__all__ = [
    "CAPTURE_KINDS",
    "CHUNK_BUDGET_BYTES",
    "DEFAULT_ATTENUATION_EPS",
    "DEFAULT_CHUNK_ROWS",
    "PSA_KEY",
    "AbsoluteBinning",
    "CaptureAccumulator",
    "CaptureSpec",
    "DistanceBinning",
    "HeadScaleSpec",
    "LogitBiasSpec",
    "PSAContext",
    "PositionBinning",
    "ValueScaleSpec",
    "attention_modules",
    "binning_from_describe",
    "finalize_captures",
    "is_registered",
    "psa_attention_forward",
    "psa_session",
    "register_psa",
    "span_mask",
]
