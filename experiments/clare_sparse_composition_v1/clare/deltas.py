"""Skill deltas over a footprint: explicit storage, representation-independent checksums, and
deterministic composition.

Two storage forms:

* ``SparseDelta``: an explicit ``SkillMask`` plus FP32 values in mask order (slot order, then
  ascending local index). Used by the disjoint arms (D, E, F-private).
* ``DenseDelta``: FP32 values for every footprint coordinate. Used by the dense arms (A, B) and
  by magnitude pruning (C) before it is converted to a sparse delta.

``delta_checksum`` hashes the DENSE-EQUIVALENT float32 bytes slot by slot, so the same delta
has the same checksum in either form. Composition sums deltas in a registered order into an FP32
accumulator; for pairwise-disjoint supports the sum is exact and therefore order-independent.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .footprint import Footprint
from .io import read_json, sha256_file, write_json
from .masks import SkillMask

DELTA_FORMAT = "clare-delta-v1"
F32 = np.dtype("<f4")


def _checksum(fp: Footprint, dense_slots: Iterable[tuple[str, np.ndarray]]) -> str:
    h = hashlib.sha256()
    h.update(DELTA_FORMAT.encode("ascii"))
    h.update(bytes.fromhex(fp.digest()))
    names = []
    for name, arr in dense_slots:
        a = np.ascontiguousarray(arr, dtype=F32).reshape(-1)
        h.update(a.tobytes())
        names.append(name)
    if names != fp.names():
        raise ValueError("checksum stream does not cover the footprint in order")
    return h.hexdigest()


class Delta:
    """Common interface. Subclasses provide ``dense(name)`` and ``support(name)``."""

    footprint: Footprint

    def dense(self, name: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def support(self, name: str) -> np.ndarray:
        """Boolean support = coordinates with a non-zero value (NOT the allocated mask)."""
        return self.dense(name).reshape(-1) != 0

    def iter_dense(self) -> Iterator[tuple[str, np.ndarray]]:
        for slot in self.footprint:
            yield slot.name, self.dense(slot.name)

    def checksum(self) -> str:
        return _checksum(self.footprint, self.iter_dense())

    def sq_norm(self) -> float:
        total = 0.0
        for _, a in self.iter_dense():
            a64 = a.astype(np.float64).reshape(-1)
            total += float(a64 @ a64)
        return total

    def norm(self) -> float:
        return math.sqrt(self.sq_norm())

    def nonzero(self) -> int:
        return sum(int(np.count_nonzero(a)) for _, a in self.iter_dense())


class DenseDelta(Delta):
    def __init__(self, footprint: Footprint, values: Mapping[str, np.ndarray]):
        if set(values) != set(footprint.names()):
            raise ValueError("dense delta slots do not match the footprint")
        self.footprint = footprint
        self._v: dict[str, np.ndarray] = {}
        for slot in footprint:
            a = np.ascontiguousarray(values[slot.name], dtype=F32).reshape(-1)
            if a.size != slot.numel:
                raise ValueError(f"{slot.name}: {a.size} values for {slot.numel} coordinates")
            if not np.isfinite(a).all():
                raise ValueError(f"{slot.name}: non-finite delta values")
            a = a + np.float32(0.0)  # canonical zeros: -0.0 -> +0.0, so composition is bit-stable
            a.setflags(write=False)
            self._v[slot.name] = a

    @classmethod
    def zeros(cls, footprint: Footprint) -> "DenseDelta":
        return cls(footprint, {s.name: np.zeros(s.numel, dtype=F32) for s in footprint})

    def dense(self, name: str) -> np.ndarray:
        return self._v[name]

    def restrict(self, mask: SkillMask) -> "SparseDelta":
        """Keep the values on ``mask`` (explicit support), drop the rest."""
        return SparseDelta(mask, {s.name: self._v[s.name][mask.bool(s.name)] for s in self.footprint})

    def save(self, path_stem: str | Path, meta: Mapping | None = None) -> dict:
        stem = Path(path_stem)
        payload = stem.with_suffix(".dense.f32")
        if payload.exists():
            raise FileExistsError(f"refusing to overwrite {payload}")
        payload.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with open(payload, "xb") as f:
            for slot in self.footprint:
                b = self._v[slot.name].astype(F32, copy=False).tobytes()
                h.update(b)
                f.write(b)
        side = {"format": DELTA_FORMAT, "storage": "dense", "footprint_digest": self.footprint.digest(),
                "footprint": self.footprint.spec(), "payload": payload.name,
                "payload_sha256": h.hexdigest(), "delta_checksum": self.checksum(),
                "l2_norm": self.norm(), "nonzero": self.nonzero(), "meta": dict(meta or {})}
        write_json(stem.with_suffix(".json"), side)
        return side


class SparseDelta(Delta):
    def __init__(self, mask: SkillMask, values: Mapping[str, np.ndarray]):
        self.footprint = mask.footprint
        self.mask = mask
        self._v: dict[str, np.ndarray] = {}
        for slot in self.footprint:
            a = np.ascontiguousarray(values[slot.name], dtype=F32).reshape(-1)
            if a.size != mask.count(slot.name):
                raise ValueError(f"{slot.name}: {a.size} values for {mask.count(slot.name)} masked coordinates")
            if not np.isfinite(a).all():
                raise ValueError(f"{slot.name}: non-finite delta values")
            a = a + np.float32(0.0)  # canonical zeros: -0.0 -> +0.0, so composition is bit-stable
            a.setflags(write=False)
            self._v[slot.name] = a

    @classmethod
    def zeros(cls, mask: SkillMask) -> "SparseDelta":
        return cls(mask, {s.name: np.zeros(mask.count(s.name), dtype=F32) for s in mask.footprint})

    def values(self, name: str) -> np.ndarray:
        return self._v[name]

    def dense(self, name: str) -> np.ndarray:
        slot = self.footprint.slot(name)
        out = np.zeros(slot.numel, dtype=F32)
        out[self.mask.bool(name)] = self._v[name]
        return out

    def save(self, path_stem: str | Path, meta: Mapping | None = None) -> dict:
        stem = Path(path_stem)
        mask_side = self.mask.save(stem.with_name(stem.name + "_mask"), meta=meta)
        payload = stem.with_suffix(".values.f32")
        if payload.exists():
            raise FileExistsError(f"refusing to overwrite {payload}")
        h = hashlib.sha256()
        with open(payload, "xb") as f:
            for slot in self.footprint:
                b = self._v[slot.name].tobytes()
                h.update(b)
                f.write(b)
        side = {"format": DELTA_FORMAT, "storage": "sparse", "footprint_digest": self.footprint.digest(),
                "footprint": self.footprint.spec(), "payload": payload.name,
                "payload_sha256": h.hexdigest(), "mask_stem": stem.name + "_mask",
                "mask_checksum": mask_side["mask_checksum"], "allocated": self.mask.count(),
                "delta_checksum": self.checksum(), "l2_norm": self.norm(),
                "nonzero": self.nonzero(), "meta": dict(meta or {})}
        write_json(stem.with_suffix(".json"), side)
        return side


def load_delta(path_stem: str | Path, footprint: Footprint | None = None) -> Delta:
    stem = Path(path_stem)
    side = read_json(stem.with_suffix(".json"))
    if side.get("format") != DELTA_FORMAT:
        raise ValueError(f"unknown delta format {side.get('format')!r}")
    fp = Footprint.from_spec(side["footprint"])
    if fp.digest() != side["footprint_digest"] or (footprint is not None and footprint.digest() != fp.digest()):
        raise ValueError("delta footprint mismatch")
    fp = footprint or fp
    payload = stem.parent / side["payload"]
    if sha256_file(payload) != side["payload_sha256"]:
        raise ValueError("delta payload hash mismatch")
    raw = np.fromfile(payload, dtype=F32)
    if side["storage"] == "dense":
        vals, pos = {}, 0
        for slot in fp:
            vals[slot.name] = raw[pos:pos + slot.numel]
            pos += slot.numel
        if pos != raw.size:
            raise ValueError("dense payload size mismatch")
        out: Delta = DenseDelta(fp, vals)
    elif side["storage"] == "sparse":
        mask = SkillMask.load(stem.parent / side["mask_stem"], fp)
        if mask.checksum() != side["mask_checksum"]:
            raise ValueError("sparse delta mask checksum mismatch")
        vals, pos = {}, 0
        for slot in fp:
            n = mask.count(slot.name)
            vals[slot.name] = raw[pos:pos + n]
            pos += n
        if pos != raw.size:
            raise ValueError("sparse payload size mismatch")
        out = SparseDelta(mask, vals)
    else:
        raise ValueError(f"unknown storage {side['storage']!r}")
    if out.checksum() != side["delta_checksum"]:
        raise ValueError("delta content checksum mismatch")
    return out


def compose(deltas: Sequence[Delta], coefficients: Sequence[float] | None = None) -> DenseDelta:
    """FP32 sum ``sum_i c_i * delta_i`` accumulated in the given order.

    Coefficients default to 1. With coefficient 1 and pairwise-disjoint supports every output
    coordinate receives at most one non-zero term, so the result is exact and order-invariant.
    """
    if not deltas:
        raise ValueError("nothing to compose")
    fp = deltas[0].footprint
    if any(d.footprint.digest() != fp.digest() for d in deltas):
        raise ValueError("deltas live in different footprints")
    coefs = [1.0] * len(deltas) if coefficients is None else [float(c) for c in coefficients]
    if len(coefs) != len(deltas):
        raise ValueError("one coefficient per delta")
    out = {}
    for slot in fp:
        acc = np.zeros(slot.numel, dtype=F32)
        for c, d in zip(coefs, deltas):
            v = d.dense(slot.name)
            acc += v if c == 1.0 else (np.float32(c) * v).astype(F32)
        out[slot.name] = acc
    return DenseDelta(fp, out)


def difference(after: Delta, before: Delta) -> DenseDelta:
    """``after - before`` in FP32 (per-stage delta of a dense sequential arm)."""
    if after.footprint.digest() != before.footprint.digest():
        raise ValueError("deltas live in different footprints")
    return DenseDelta(after.footprint, {s.name: (after.dense(s.name) - before.dense(s.name)).astype(F32)
                                        for s in after.footprint})


def magnitude_prune(delta: Delta, budget: int, tiebreak_key) -> SparseDelta:
    """Arm C: keep the ``budget`` largest-|value| coordinates of an independently trained delta."""
    from .select import select_top_importance

    sel, _ = select_top_importance(delta.footprint, lambda s: np.abs(delta.dense(s.name)), None,
                                   budget, tiebreak_key)
    mask = SkillMask.from_bool(delta.footprint, sel)
    return SparseDelta(mask, {s.name: delta.dense(s.name)[sel[s.name]] for s in delta.footprint})


# ---------------------------------------------------------------------------------------
# Streaming persistence and disk-backed (slot-at-a-time) deltas
# ---------------------------------------------------------------------------------------

class _StreamStats:
    def __init__(self, fp: Footprint):
        self.fp = fp
        self.payload = hashlib.sha256()
        self.content = hashlib.sha256()
        self.content.update(DELTA_FORMAT.encode("ascii"))
        self.content.update(bytes.fromhex(fp.digest()))
        self.sq = 0.0
        self.nonzero = 0
        self.names: list[str] = []

    def add_dense(self, name: str, dense: np.ndarray) -> None:
        self.names.append(name)
        self.content.update(dense.tobytes())
        d64 = dense.astype(np.float64)
        self.sq += float(d64 @ d64)
        self.nonzero += int(np.count_nonzero(dense))

    def finish(self) -> None:
        if self.names != self.fp.names():
            raise ValueError("stream did not cover the footprint in order")


def _canon(a: np.ndarray, n: int, name: str) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=F32).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name}: {a.size} values, expected {n}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite delta values")
    return a + np.float32(0.0)


def write_dense_stream(path_stem: str | Path, footprint: Footprint,
                       slots: Iterable[tuple[str, np.ndarray]], meta: Mapping | None = None) -> dict:
    """Write a dense delta slot by slot (never holding more than one slot)."""
    stem = Path(path_stem)
    payload = stem.with_suffix(".dense.f32")
    if payload.exists() or stem.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite {stem}")
    payload.parent.mkdir(parents=True, exist_ok=True)
    st = _StreamStats(footprint)
    with open(payload, "xb") as f:
        for name, arr in slots:
            a = _canon(arr, footprint.slot(name).numel, name)
            b = a.tobytes()
            st.payload.update(b)
            f.write(b)
            st.add_dense(name, a)
    st.finish()
    side = {"format": DELTA_FORMAT, "storage": "dense", "footprint_digest": footprint.digest(),
            "footprint": footprint.spec(), "payload": payload.name, "payload_sha256": st.payload.hexdigest(),
            "delta_checksum": st.content.hexdigest(), "l2_norm": math.sqrt(st.sq), "nonzero": st.nonzero,
            "meta": dict(meta or {})}
    write_json(stem.with_suffix(".json"), side)
    return side


def write_sparse_stream(path_stem: str | Path, mask: SkillMask,
                        values: Iterable[tuple[str, np.ndarray]], meta: Mapping | None = None) -> dict:
    stem = Path(path_stem)
    payload = stem.with_suffix(".values.f32")
    if payload.exists() or stem.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite {stem}")
    mask_side = mask.save(stem.with_name(stem.name + "_mask"), meta=meta)
    fp = mask.footprint
    st = _StreamStats(fp)
    with open(payload, "xb") as f:
        for name, arr in values:
            v = _canon(arr, mask.count(name), name)
            b = v.tobytes()
            st.payload.update(b)
            f.write(b)
            dense = np.zeros(fp.slot(name).numel, dtype=F32)
            dense[mask.bool(name)] = v
            st.add_dense(name, dense)
    st.finish()
    side = {"format": DELTA_FORMAT, "storage": "sparse", "footprint_digest": fp.digest(),
            "footprint": fp.spec(), "payload": payload.name, "payload_sha256": st.payload.hexdigest(),
            "mask_stem": stem.name + "_mask", "mask_checksum": mask_side["mask_checksum"],
            "allocated": mask.count(), "delta_checksum": st.content.hexdigest(),
            "l2_norm": math.sqrt(st.sq), "nonzero": st.nonzero, "meta": dict(meta or {})}
    write_json(stem.with_suffix(".json"), side)
    return side


class DiskDelta(Delta):
    """Read-only delta backed by a saved payload; reads ONE slot per call (seek/read, no mmap)."""

    def __init__(self, path_stem: str | Path, footprint: Footprint | None = None, verify: bool = True):
        stem = Path(path_stem)
        side = read_json(stem.with_suffix(".json"))
        if side.get("format") != DELTA_FORMAT:
            raise ValueError(f"unknown delta format {side.get('format')!r}")
        fp = Footprint.from_spec(side["footprint"])
        if fp.digest() != side["footprint_digest"] or (footprint is not None and footprint.digest() != fp.digest()):
            raise ValueError("delta footprint mismatch")
        self.footprint = footprint or fp
        self.side = side
        self.path = stem.parent / side["payload"]
        self.storage = side["storage"]
        self.mask: SkillMask | None = None
        self._offsets: dict[str, tuple[int, int]] = {}
        pos = 0
        if self.storage == "sparse":
            self.mask = SkillMask.load(stem.parent / side["mask_stem"], self.footprint)
            if self.mask.checksum() != side["mask_checksum"]:
                raise ValueError("mask checksum mismatch")
        for slot in self.footprint:
            n = slot.numel if self.storage == "dense" else self.mask.count(slot.name)
            self._offsets[slot.name] = (pos, n)
            pos += n * 4
        if self.path.stat().st_size != pos:
            raise ValueError("delta payload size mismatch")
        if verify:
            if sha256_file(self.path) != side["payload_sha256"]:
                raise ValueError("delta payload hash mismatch")
            if self.checksum() != side["delta_checksum"]:
                raise ValueError("delta content checksum mismatch")

    def _read(self, name: str) -> np.ndarray:
        off, n = self._offsets[name]
        with open(self.path, "rb") as f:
            f.seek(off)
            buf = f.read(n * 4)
        if len(buf) != n * 4:
            raise IOError(f"short read on {self.path}")
        return np.frombuffer(buf, dtype=F32).copy()

    def values(self, name: str) -> np.ndarray:
        return self._read(name)

    def dense(self, name: str) -> np.ndarray:
        v = self._read(name)
        if self.storage == "dense":
            return v
        out = np.zeros(self.footprint.slot(name).numel, dtype=F32)
        out[self.mask.bool(name)] = v
        return out

    @property
    def l2_norm(self) -> float:
        return float(self.side["l2_norm"])
