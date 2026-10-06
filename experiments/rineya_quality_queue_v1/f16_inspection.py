"""Inspect explicit opaque F16 row bytes without model, file or floating I/O.

This optional check is separate from lossless archiving: rejected nonfinite rows
can still be archived unchanged. A successful check attests only the supplied
bytes, never their native origin, fixed input binding or model quality.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any

MAX_VOCABULARY = 1_000_000


def inspect_finite_argmax_row(payload: bytes, *, vocabulary_size: int,
                              expected_argmax_token_id: int | None = None) -> dict[str, Any]:
    """Require every finite little-endian binary16 value and its first maximum.

    Finite binary16 bit patterns admit an exact integer ordering. Both signed
    zeros share a rank, so equal numerical maxima retain the lowest token ID.
    No rounding, float conversion, token decoding, asset read or model call is
    performed. The caller must independently authenticate the row's provenance,
    case/arm/bias/prefix binding and runtime before accepting a native receipt.
    """
    if type(vocabulary_size) is not int or not 1 <= vocabulary_size <= MAX_VOCABULARY:
        raise ValueError("bounded positive integer vocabulary_size required")
    if type(payload) is not bytes or len(payload) != 2 * vocabulary_size:
        raise ValueError("exact bounded F16 row bytes required")
    if expected_argmax_token_id is not None and (
        type(expected_argmax_token_id) is not int
        or not 0 <= expected_argmax_token_id < vocabulary_size
    ):
        raise ValueError("expected token ID outside row")
    best_rank, best_index = -1, -1
    for index, (word,) in enumerate(struct.iter_unpack("<H", payload)):
        if word & 0x7C00 == 0x7C00:
            raise ValueError("nonfinite F16 value at token ID " + str(index))
        if word & 0x7FFF == 0:
            rank = 0x8000
        elif word & 0x8000:
            rank = 0xFFFF - word
        else:
            rank = 0x8000 + word
        if rank > best_rank:
            best_rank, best_index = rank, index
    if expected_argmax_token_id is not None and best_index != expected_argmax_token_id:
        raise ValueError("lowest-ID argmax differs from supplied token ID")
    return {
        "schema": "rineya.f16_row_inspection/1",
        "dtype": "F16", "byte_order": "little", "elements": vocabulary_size,
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "argmax_token_id": best_index, "all_values_finite": True,
        "scope": "SUPPLIED_BYTES_ONLY",
        "native_execution_authorized": False, "quality_claim": False,
    }
