"""Binary support masks and the strict CLARE allocation state.

A ``SkillMask`` stores one boolean per footprint coordinate, bit-packed per slot (little bit
order). Masks are saved EXPLICITLY; nothing in this experiment reconstructs a support by
thresholding a floating-point delta.

``AllocationState`` records which skill owns which coordinate. In strict mode a claimed
coordinate can never be claimed again, so supports are pairwise disjoint by construction and
every pair of skill deltas has a zero parameter-space dot product. That is a mechanical
consequence of the allocation, not an empirical finding.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .footprint import Footprint
from .io import read_json, sha256_file, write_bytes, write_json

MASK_FORMAT = "clare-mask-v1"
STATE_FORMAT = "clare-allocation-v1"


def _pack(flat_bool: np.ndarray) -> np.ndarray:
    return np.packbits(np.ascontiguousarray(flat_bool, dtype=bool).reshape(-1), bitorder="little")


def _unpack(packed: np.ndarray, numel: int) -> np.ndarray:
    return np.unpackbits(packed, count=numel, bitorder="little").astype(bool)


class SkillMask:
    """Immutable per-slot packed boolean mask over a footprint."""

    def __init__(self, footprint: Footprint, packed: Mapping[str, np.ndarray]):
        if set(packed) != set(footprint.names()):
            raise ValueError("mask slots do not match the footprint")
        self.footprint = footprint
        self._packed: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        for slot in footprint:
            p = np.ascontiguousarray(packed[slot.name], dtype=np.uint8).reshape(-1)
            if p.size != (slot.numel + 7) // 8:
                raise ValueError(f"{slot.name}: packed size {p.size} != {(slot.numel + 7) // 8}")
            tail = slot.numel % 8
            if tail and int(p[-1]) >> tail:
                raise ValueError(f"{slot.name}: padding bits set beyond the last coordinate")
            p.setflags(write=False)
            self._packed[slot.name] = p
            self._counts[slot.name] = int(np.unpackbits(p, bitorder="little").sum())

    # -- construction -------------------------------------------------------------------
    @classmethod
    def from_bool(cls, footprint: Footprint, masks: Mapping[str, np.ndarray]) -> "SkillMask":
        packed = {}
        for slot in footprint:
            m = np.asarray(masks[slot.name])
            if m.dtype != np.bool_:
                raise TypeError(f"{slot.name}: mask must be boolean, got {m.dtype}")
            if m.size != slot.numel:
                raise ValueError(f"{slot.name}: mask has {m.size} entries, slot has {slot.numel}")
            packed[slot.name] = _pack(m)
        return cls(footprint, packed)

    @classmethod
    def empty(cls, footprint: Footprint) -> "SkillMask":
        return cls(footprint, {s.name: np.zeros((s.numel + 7) // 8, dtype=np.uint8) for s in footprint})

    @classmethod
    def full(cls, footprint: Footprint) -> "SkillMask":
        return cls.from_bool(footprint, {s.name: np.ones(s.numel, dtype=bool) for s in footprint})

    # -- access -------------------------------------------------------------------------
    def bool(self, name: str) -> np.ndarray:
        return _unpack(self._packed[name], self.footprint.slot(name).numel)

    def packed(self, name: str) -> np.ndarray:
        return self._packed[name]

    def count(self, name: str | None = None) -> int:
        return self._counts[name] if name is not None else sum(self._counts.values())

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def indices(self, name: str) -> np.ndarray:
        """Local flat indices (int64, ascending) of the set coordinates of one slot."""
        return np.flatnonzero(self.bool(name)).astype(np.int64)

    def checksum(self) -> str:
        h = hashlib.sha256()
        h.update(MASK_FORMAT.encode("ascii"))
        h.update(bytes.fromhex(self.footprint.digest()))
        for slot in self.footprint:
            h.update(self._packed[slot.name].tobytes())
        return h.hexdigest()

    def overlap(self, other: "SkillMask") -> int:
        self._same_space(other)
        n = 0
        for slot in self.footprint:
            both = np.bitwise_and(self._packed[slot.name], other._packed[slot.name])
            n += int(np.unpackbits(both, bitorder="little").sum())
        return n

    def union(self, other: "SkillMask") -> "SkillMask":
        self._same_space(other)
        return SkillMask(self.footprint, {s.name: np.bitwise_or(self._packed[s.name], other._packed[s.name])
                                          for s in self.footprint})

    def invert(self) -> "SkillMask":
        out = {}
        for slot in self.footprint:
            p = np.bitwise_not(self._packed[slot.name])
            tail = slot.numel % 8
            if tail:
                p = p.copy()
                p[-1] &= np.uint8((1 << tail) - 1)
            out[slot.name] = p
        return SkillMask(self.footprint, out)

    def _same_space(self, other: "SkillMask") -> None:
        if other.footprint.digest() != self.footprint.digest():
            raise ValueError("masks live in different footprints")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkillMask) and self.checksum() == other.checksum()

    def __hash__(self) -> int:
        return hash(self.checksum())

    # -- persistence --------------------------------------------------------------------
    def save(self, path_stem: str | Path, meta: Mapping | None = None) -> dict:
        """Write ``<stem>.bin`` (packed bits, footprint order) and ``<stem>.json`` (sidecar)."""
        stem = Path(path_stem)
        blob = b"".join(self._packed[s.name].tobytes() for s in self.footprint)
        bin_sha = write_bytes(stem.with_suffix(".bin"), blob)
        sidecar = {
            "format": MASK_FORMAT,
            "footprint_digest": self.footprint.digest(),
            "footprint": self.footprint.spec(),
            "bitorder": "little",
            "total_coordinates": self.footprint.total,
            "selected_coordinates": self.count(),
            "selected_per_slot": [self._counts[s.name] for s in self.footprint],
            "mask_checksum": self.checksum(),
            "bin_sha256": bin_sha,
            "meta": dict(meta or {}),
        }
        write_json(stem.with_suffix(".json"), sidecar)
        return sidecar

    @classmethod
    def load(cls, path_stem: str | Path, footprint: Footprint | None = None) -> "SkillMask":
        stem = Path(path_stem)
        side = read_json(stem.with_suffix(".json"))
        if side.get("format") != MASK_FORMAT:
            raise ValueError(f"unknown mask format {side.get('format')!r}")
        fp = Footprint.from_spec(side["footprint"])
        if fp.digest() != side["footprint_digest"]:
            raise ValueError("sidecar footprint spec does not match its digest")
        if footprint is not None and footprint.digest() != fp.digest():
            raise ValueError("mask was saved for a different footprint")
        if sha256_file(stem.with_suffix(".bin")) != side["bin_sha256"]:
            raise ValueError("mask payload hash mismatch")
        blob = np.fromfile(stem.with_suffix(".bin"), dtype=np.uint8)
        packed, pos = {}, 0
        for slot in fp:
            n = (slot.numel + 7) // 8
            packed[slot.name] = blob[pos:pos + n].copy()
            pos += n
        if pos != blob.size:
            raise ValueError("mask payload has trailing bytes")
        mask = cls(footprint or fp, packed)
        if mask.checksum() != side["mask_checksum"] or mask.count() != side["selected_coordinates"]:
            raise ValueError("mask content checksum mismatch")
        return mask


class AllocationState:
    """Which skill owns which footprint coordinate. Strict mode forbids any re-claim."""

    def __init__(self, footprint: Footprint, strict: bool = True):
        self.footprint = footprint
        self.strict = strict
        self._order: list[str] = []
        self._masks: dict[str, SkillMask] = {}
        self._claimed = SkillMask.empty(footprint)

    # -- queries ------------------------------------------------------------------------
    @property
    def skills(self) -> list[str]:
        return list(self._order)

    def mask(self, skill: str) -> SkillMask:
        return self._masks[skill]

    def claimed(self) -> SkillMask:
        return self._claimed

    def free(self) -> SkillMask:
        return self._claimed.invert()

    def free_bool(self, name: str) -> np.ndarray:
        return ~self._claimed.bool(name)

    def free_count(self) -> int:
        return self.footprint.total - self._claimed.count()

    def owner(self, name: str) -> np.ndarray:
        """int16 owner per coordinate of one slot: -1 free, k = position in claim order."""
        slot = self.footprint.slot(name)
        own = np.full(slot.numel, -1, dtype=np.int16)
        for k, skill in enumerate(self._order):
            m = self._masks[skill].bool(name)
            if self.strict and (own[m] != -1).any():  # pragma: no cover - claim() prevents this
                raise AssertionError("strict allocation holds overlapping owners")
            own[m] = k
        return own

    # -- mutation -----------------------------------------------------------------------
    def claim(self, skill: str, mask: SkillMask, expected_budget: int | None = None) -> None:
        if skill in self._masks:
            raise ValueError(f"skill {skill!r} already holds an allocation")
        if mask.footprint.digest() != self.footprint.digest():
            raise ValueError("mask footprint differs from the allocation footprint")
        if expected_budget is not None and mask.count() != expected_budget:
            raise ValueError(f"{skill}: mask has {mask.count()} coordinates, budget is {expected_budget}")
        clash = mask.overlap(self._claimed)
        if self.strict and clash:
            raise ValueError(f"{skill}: {clash} coordinates are already claimed (strict allocation)")
        self._order.append(skill)
        self._masks[skill] = mask
        self._claimed = self._claimed.union(mask)

    # -- persistence --------------------------------------------------------------------
    def checksum(self) -> str:
        h = hashlib.sha256()
        h.update(STATE_FORMAT.encode("ascii"))
        h.update(bytes.fromhex(self.footprint.digest()))
        h.update(b"strict" if self.strict else b"loose")
        for skill in self._order:
            h.update(skill.encode("utf-8") + b"\0")
            h.update(bytes.fromhex(self._masks[skill].checksum()))
        return h.hexdigest()

    def save(self, directory: str | Path) -> dict:
        d = Path(directory)
        entries = []
        for k, skill in enumerate(self._order):
            side = self._masks[skill].save(d / f"mask_{k:02d}_{skill}", meta={"skill": skill, "position": k})
            entries.append({"skill": skill, "position": k, "stem": f"mask_{k:02d}_{skill}",
                            "mask_checksum": side["mask_checksum"],
                            "selected_coordinates": side["selected_coordinates"]})
        state = {"format": STATE_FORMAT, "footprint_digest": self.footprint.digest(),
                 "strict": self.strict, "order": list(self._order), "masks": entries,
                 "free_coordinates": self.free_count(), "state_checksum": self.checksum()}
        write_json(d / "allocation_state.json", state)
        return state

    @classmethod
    def load(cls, directory: str | Path, footprint: Footprint) -> "AllocationState":
        d = Path(directory)
        state = read_json(d / "allocation_state.json")
        if state.get("format") != STATE_FORMAT or state["footprint_digest"] != footprint.digest():
            raise ValueError("allocation state format/footprint mismatch")
        out = cls(footprint, strict=bool(state["strict"]))
        for entry in state["masks"]:
            m = SkillMask.load(d / entry["stem"], footprint)
            if m.checksum() != entry["mask_checksum"]:
                raise ValueError(f"{entry['skill']}: mask checksum mismatch on resume")
            out.claim(entry["skill"], m)
        if out.checksum() != state["state_checksum"] or out.free_count() != state["free_coordinates"]:
            raise ValueError("allocation state checksum mismatch on resume")
        return out


def pairwise_overlaps(masks: Mapping[str, SkillMask]) -> dict[tuple[str, str], dict]:
    names = list(masks)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = masks[a].overlap(masks[b])
            union = masks[a].count() + masks[b].count() - inter
            out[(a, b)] = {"intersection": inter, "union": union,
                           "jaccard": inter / union if union else 0.0}
    return out


def layer_table(state: AllocationState) -> list[dict]:
    """Per slot and skill: available before the skill, selected, allocation fraction."""
    rows = []
    fp = state.footprint
    for slot in fp:
        claimed_so_far = np.zeros(slot.numel, dtype=bool)
        for k, skill in enumerate(state.skills):
            m = state.mask(skill).bool(slot.name)
            available = int(slot.numel - claimed_so_far.sum())
            sel = int(m.sum())
            rows.append({"slot": slot.name, "layer": slot.layer, "module": slot.module,
                         "skill": skill, "position": k, "slot_coordinates": slot.numel,
                         "available_before": available, "selected": sel,
                         "selected_fraction_of_available": sel / available if available else 0.0,
                         "selected_fraction_of_slot": sel / slot.numel})
            claimed_so_far |= m
    return rows


def iter_pairs(items: Iterable[str]) -> list[tuple[str, str]]:
    items = list(items)
    return [(a, b) for i, a in enumerate(items) for b in items[i + 1:]]
