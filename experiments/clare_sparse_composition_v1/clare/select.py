"""Exact-budget coordinate selection with deterministic tie-breaking.

Every selection rule in the experiment reduces to one primitive:

    choose exactly ``budget`` eligible coordinates with the SMALLEST uint64 keys;
    exactly-equal keys are broken by ascending global index.

* learned (CLARE) importance: high 32 bits = inverted float32 bits of |delta| (larger importance
  -> smaller key), low 32 bits = a seeded hash of the coordinate. Equal importances (notably the
  exact zeros left by the L1 probe) are therefore broken pseudo-randomly, not by position, so a
  budget larger than the non-zero support cannot silently favour early layers.
* random allocation: key = splitmix64(global_index XOR seed_key). splitmix64 is a bijection on
  uint64, so for a fixed seed every key is distinct and the chosen set is a pseudo-random subset
  of exactly ``budget`` eligible coordinates.
* magnitude pruning: same as learned importance, over all coordinates of one delta.

The implementation is a 4-pass radix select over 16-bit digits followed by one materialising
pass. It never holds more than one slot's keys in memory, so it scales to 10^8-10^9 coordinates
on a small host. Keys are recomputed per pass from a provider callback.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .footprint import Footprint, Slot

U64 = np.uint64
_MASK32 = U64(0xFFFFFFFF)

KeyProvider = Callable[[Slot], "tuple[np.ndarray, np.ndarray]"]
"""slot -> (keys uint64[numel], eligible bool[numel]); flat, in the slot's C order."""


def seed_key(*parts: object) -> np.uint64:
    """Derive a 64-bit key from arbitrary labelled parts (study, arm, seed, skill, ...)."""
    text = "|".join(str(p) for p in parts).encode("utf-8")
    return U64(int.from_bytes(hashlib.sha256(text).digest()[:8], "little"))


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser (bijective on uint64); wraps modulo 2**64."""
    x = np.asarray(x, dtype=U64)
    with np.errstate(over="ignore"):
        z = x + U64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        z = z ^ (z >> U64(31))
    return z


def global_indices(slot: Slot) -> np.ndarray:
    return np.arange(slot.offset, slot.offset + slot.numel, dtype=U64)


def random_keys(slot: Slot, key: np.uint64) -> np.ndarray:
    return splitmix64(global_indices(slot) ^ U64(key))


def importance_keys(slot: Slot, importance: np.ndarray, tiebreak_key: np.uint64) -> np.ndarray:
    """Keys whose ascending order is descending importance, ties broken by a seeded hash."""
    q = np.ascontiguousarray(importance, dtype=np.float32).reshape(-1)
    if q.size != slot.numel:
        raise ValueError(f"{slot.name}: importance has {q.size} values, slot has {slot.numel}")
    if not np.isfinite(q).all():
        raise ValueError(f"{slot.name}: non-finite importance")
    if np.signbit(q).any():
        raise ValueError(f"{slot.name}: importance must be non-negative (use |delta|, not delta)")
    bits = q.view(np.uint32).astype(U64)
    hi = (_MASK32 - bits) << U64(32)
    lo = random_keys(slot, tiebreak_key) & _MASK32
    return hi | lo


@dataclass(frozen=True)
class SelectionReport:
    budget: int
    eligible: int
    threshold_key: int
    taken_below_threshold: int
    taken_at_threshold: int
    at_threshold_available: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _eligible_count(fp: Footprint, provider: KeyProvider) -> int:
    n = 0
    for slot in fp:
        _, elig = provider(slot)
        n += int(np.count_nonzero(elig))
    return n


def select_smallest(fp: Footprint, provider: KeyProvider, budget: int,
                    eligible_total: int | None = None) -> tuple[dict[str, np.ndarray], SelectionReport]:
    """Select exactly ``budget`` eligible coordinates with the smallest keys.

    Returns per-slot boolean masks (flat, slot C order) and a report. Deterministic: the result
    depends only on the keys, the eligibility and the footprint order.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError(f"budget must be a non-negative int, got {budget!r}")
    n_elig = _eligible_count(fp, provider) if eligible_total is None else eligible_total
    if budget > n_elig:
        raise ValueError(f"budget {budget} exceeds eligible coordinates {n_elig}")
    if budget == 0:
        empty = {s.name: np.zeros(s.numel, dtype=bool) for s in fp}
        return empty, SelectionReport(0, n_elig, 0, 0, 0, 0)

    prefix = 0          # value of the high digits fixed so far
    fixed_bits = 0      # number of high bits fixed so far
    remaining = budget  # rank (1-based) of the threshold element among those sharing the prefix
    for shift in (48, 32, 16, 0):
        hist = np.zeros(65536, dtype=np.int64)
        for slot in fp:
            keys, elig = provider(slot)
            if fixed_bits:
                cand = elig & ((keys >> U64(64 - fixed_bits)) == U64(prefix))
            else:
                cand = elig
            if not cand.any():
                continue
            digits = ((keys[cand] >> U64(shift)) & U64(0xFFFF)).astype(np.int64)
            hist += np.bincount(digits, minlength=65536)
        cum = np.cumsum(hist)
        d = int(np.searchsorted(cum, remaining, side="left"))
        if d >= 65536:  # pragma: no cover - guarded by the budget check
            raise AssertionError("radix select ran past the histogram")
        if d > 0:
            remaining -= int(cum[d - 1])
        prefix = (prefix << 16) | d
        fixed_bits += 16

    tau = U64(prefix)
    below = budget - remaining
    out: dict[str, np.ndarray] = {}
    need_at = remaining
    at_available = 0
    taken_below = 0
    for slot in fp:
        keys, elig = provider(slot)
        sel = elig & (keys < tau)
        taken_below += int(np.count_nonzero(sel))
        eq = np.flatnonzero(elig & (keys == tau))
        at_available += int(eq.size)
        if need_at and eq.size:
            take = eq[:need_at]
            sel[take] = True
            need_at -= int(take.size)
        out[slot.name] = sel
    if taken_below != below or need_at != 0:
        raise AssertionError(f"selection accounting failed: below {taken_below} vs {below}, "
                             f"unfilled {need_at}")
    return out, SelectionReport(budget, n_elig, int(tau), taken_below, remaining, at_available)


# -- convenience wrappers -----------------------------------------------------------------

def _eligible(slot: Slot, arr) -> np.ndarray:
    """Eligibility must be an explicit boolean array of the slot's size (no silent casting)."""
    a = np.asarray(arr)
    if a.dtype != np.bool_:
        raise ValueError(f"{slot.name}: eligibility must be boolean, got {a.dtype}")
    a = a.reshape(-1)
    if a.size != slot.numel:
        raise ValueError(f"{slot.name}: eligibility has {a.size} entries, slot has {slot.numel}")
    return a


def select_random(fp: Footprint, eligible: Mapping[str, np.ndarray], budget: int,
                  key: np.uint64) -> tuple[dict[str, np.ndarray], SelectionReport]:
    def provider(slot: Slot):
        return random_keys(slot, key), _eligible(slot, eligible[slot.name])
    return select_smallest(fp, provider, budget)


def select_top_importance(fp: Footprint, importance: Mapping[str, np.ndarray] | Callable[[Slot], np.ndarray],
                          eligible: Mapping[str, np.ndarray] | None, budget: int,
                          tiebreak_key: np.uint64) -> tuple[dict[str, np.ndarray], SelectionReport]:
    """Top-``budget`` eligible coordinates by importance (ties: seeded hash, then index)."""
    get_q = importance if callable(importance) else (lambda s: importance[s.name])

    def provider(slot: Slot):
        keys = importance_keys(slot, get_q(slot), tiebreak_key)
        if eligible is None:
            elig = np.ones(slot.numel, dtype=bool)
        else:
            elig = _eligible(slot, eligible[slot.name])
        return keys, elig
    return select_smallest(fp, provider, budget)
