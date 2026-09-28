"""Seek/read safetensors reader with the interface of titan_model_soup_mini_v1/local-runs/verify_independent.ST.

verify_independent.ST reads through a memory map, and a map keeps every page it touches in this process's
working set (CONFIRM-002's pre-build measured 1.30 GiB resident after one embedding). Plain reads leave the
bytes in the OS file cache, which counts as available memory, and this process holds only the current chunk.
Same keys() order, same info(), and rows() returns the same float32 (hi - lo, per_row) array, so a builder
that is given this class writes byte-identical output (checked by weight hash, and by test_o_side_offline.py).
"""
from __future__ import annotations

import json, math, struct
from pathlib import Path

import numpy as np

ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}


class SeekST:
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n).decode("utf-8"))
        self.meta = self.header.pop("__metadata__", None)
        self.data_start = 8 + n

    def keys(self):
        return sorted(self.header)

    def info(self, k):
        d = self.header[k]
        return d["dtype"], list(d["shape"]), d["data_offsets"]

    def raw(self, k, lo, hi) -> bytes:
        dt, shape, (o0, _) = self.info(k)
        per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
        self.f.seek(self.data_start + o0 + lo * per_row * ITEMSIZE[dt])
        want = (hi - lo) * per_row * ITEMSIZE[dt]
        buf = self.f.read(want)
        if len(buf) != want:
            raise IOError(f"{self.path.name}:{k}: short read {len(buf)} of {want}")
        return buf

    def rows(self, k, lo, hi):
        """Rows [lo:hi) of tensor k as float32, shape (hi - lo, per_row); BF16/F16 upcast exactly."""
        dt, shape, _ = self.info(k)
        per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
        count = (hi - lo) * per_row
        buf = self.raw(k, lo, hi)
        if dt == "F32":
            a = np.frombuffer(buf, dtype="<f4", count=count)
        elif dt == "F16":
            a = np.frombuffer(buf, dtype="<f2", count=count).astype(np.float32)
        else:                                   # BF16: the upper 16 bits of the float32 pattern
            a = (np.frombuffer(buf, dtype="<u2", count=count).astype(np.uint32) << 16).view(np.float32)
        return a.reshape((hi - lo, per_row))

    def close(self):
        self.f.close()
