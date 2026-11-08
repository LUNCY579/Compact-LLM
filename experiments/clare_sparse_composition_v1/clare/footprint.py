"""The eligible coordinate space: an ordered list of tensors with global flat offsets.

Every mask, delta and allocation in this experiment is addressed in one global coordinate
space ``[0, total)``. A coordinate is ``(slot, local flat index)``; its global index is
``slot.offset + local``. The footprint digest binds the exact tensor names, shapes and order,
so a mask or delta can never be applied to a different footprint by accident.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class Slot:
    name: str
    shape: tuple[int, ...]
    offset: int
    numel: int
    layer: int | None
    module: str

    @property
    def end(self) -> int:
        return self.offset + self.numel


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        if type(d) is not int or d <= 0:
            raise ValueError(f"invalid dimension {d!r} in shape {tuple(shape)!r}")
        n *= d
    return n


def _module_of(name: str) -> str:
    m = _LAYER_RE.search(name)
    tail = name[m.end():] if m else name
    return tail[:-len(".weight")] if tail.endswith(".weight") else tail


class Footprint:
    """Immutable ordered tensor inventory. Order is the order given; it is never re-sorted."""

    def __init__(self, named_shapes: Iterable[tuple[str, Sequence[int]]]):
        slots: list[Slot] = []
        seen: set[str] = set()
        offset = 0
        for name, shape in named_shapes:
            if type(name) is not str or not name:
                raise ValueError(f"invalid tensor name {name!r}")
            if name in seen:
                raise ValueError(f"duplicate tensor name {name!r}")
            seen.add(name)
            shape = tuple(int(d) for d in shape)
            n = _numel(shape)
            m = _LAYER_RE.search(name)
            slots.append(Slot(name, shape, offset, n, int(m.group(1)) if m else None, _module_of(name)))
            offset += n
        if not slots:
            raise ValueError("empty footprint")
        self._slots = tuple(slots)
        self._by_name = {s.name: s for s in self._slots}
        self.total = offset

    # -- identity -----------------------------------------------------------------------
    def spec(self) -> list[list]:
        return [[s.name, list(s.shape)] for s in self._slots]

    def digest(self) -> str:
        payload = json.dumps(self.spec(), sort_keys=False, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @classmethod
    def from_spec(cls, spec: Sequence[Sequence]) -> "Footprint":
        return cls((str(n), tuple(s)) for n, s in spec)

    # -- access -------------------------------------------------------------------------
    @property
    def slots(self) -> tuple[Slot, ...]:
        return self._slots

    def __iter__(self) -> Iterator[Slot]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def slot(self, name: str) -> Slot:
        return self._by_name[name]

    def names(self) -> list[str]:
        return [s.name for s in self._slots]

    def locate(self, global_index: int) -> tuple[Slot, int]:
        """Map a global coordinate to (slot, local flat index). Binary search over offsets."""
        if not 0 <= global_index < self.total:
            raise IndexError(global_index)
        lo, hi = 0, len(self._slots) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._slots[mid].offset <= global_index:
                lo = mid
            else:
                hi = mid - 1
        s = self._slots[lo]
        return s, global_index - s.offset

    def layers(self) -> list[int | None]:
        out: list[int | None] = []
        for s in self._slots:
            if s.layer not in out:
                out.append(s.layer)
        return out

    def by_layer(self) -> dict[int | None, list[Slot]]:
        out: dict[int | None, list[Slot]] = {}
        for s in self._slots:
            out.setdefault(s.layer, []).append(s)
        return out

    def summary(self) -> dict:
        per_module: dict[str, int] = {}
        for s in self._slots:
            per_module[s.module] = per_module.get(s.module, 0) + s.numel
        return {
            "digest": self.digest(),
            "tensors": len(self._slots),
            "total_coordinates": self.total,
            "layers": [l for l in self.layers()],
            "coordinates_per_module": per_module,
        }


def qwen3_mlp_footprint(layers: Iterable[int], hidden: int, intermediate: int,
                        modules: Sequence[str] = ("gate_proj", "up_proj", "down_proj"),
                        prefix: str = "model.layers") -> Footprint:
    """Footprint of Qwen3-style MLP weights (nn.Linear stores [out_features, in_features])."""
    shapes = {"gate_proj": (intermediate, hidden), "up_proj": (intermediate, hidden),
              "down_proj": (hidden, intermediate)}
    items = []
    for layer in layers:
        for mod in modules:
            items.append((f"{prefix}.{layer}.mlp.{mod}.weight", shapes[mod]))
    return Footprint(items)
