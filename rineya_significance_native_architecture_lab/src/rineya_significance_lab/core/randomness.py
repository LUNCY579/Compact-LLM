"""Stable random-number derivation independent of Python's hash randomization."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a reproducible unsigned 64-bit seed from structured identifiers."""

    digest = hashlib.blake2b(digest_size=8, person=b"rineya-s")
    digest.update(str(int(base_seed)).encode("ascii"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big", signed=False)


def stable_rng(base_seed: int, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *parts))
