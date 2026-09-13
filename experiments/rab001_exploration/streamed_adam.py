"""Adam storage with bounded host buffers and no persistent disk-state mappings.

StreamedAdam preserves OffloadedAdam's parameter order, GPU placement and fp32
per-chunk update operations. Only disk transport changes: read/write bounded
buffers, then fsync and close each parameter's moment files. This is storage
engineering, not a new optimizer recipe. No model is loaded by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

# arm_dose loads this module after setting the legacy module's B2. Reuse that
# exact cached module. Standalone tests need safe argv because legacy parses it
# at import time; the constructor still receives the caller's actual learning rate.
_LEGACY_NAME = 'experiments.rab001_exploration.retention_dose_gpu'
if _LEGACY_NAME in sys.modules:
    _legacy = sys.modules[_LEGACY_NAME]
else:
    _saved_argv = sys.argv
    sys.argv = ['retention_dose_gpu', '1e-5', '5200', '17']
    try:
        _legacy = importlib.import_module(_LEGACY_NAME)
    finally:
        sys.argv = _saved_argv


@dataclass(frozen=True)
class DiskState:
    path: Path
    elements: int


def _read_exact(handle, target: memoryview) -> None:
    """Read into an existing byte buffer; reject truncated moment files."""
    cursor = 0
    while cursor < len(target):
        count = handle.readinto(target[cursor:])
        if not count:
            raise EOFError(f'truncated Adam moment file: {handle.name}')
        cursor += count


def _write_exact(handle, source: memoryview) -> None:
    cursor = 0
    while cursor < len(source):
        count = handle.write(source[cursor:])
        if not count:
            raise OSError(f'short Adam moment write: {handle.name}')
        cursor += count


class StreamedAdam(_legacy.OffloadedAdam):
    storage_backend = 'streamed_file_buffers_fsync_v1'

    def __init__(self, params, lr: float, gpu_budget_bytes: int, spill_dir: Path):
        super().__init__(params, lr, gpu_budget_bytes, spill_dir)
        # No tensor view has been constructed over these arrays. They are owned
        # solely by the inherited constructor, so closing their backing mappings
        # here cannot invalidate an optimizer tensor or outstanding CUDA copy.
        for store in (self.m, self.v):
            for index, value in enumerate(store):
                if isinstance(value, np.memmap):
                    state = DiskState(Path(value.filename), value.size)
                    value.flush()
                    value._mmap.close()
                    store[index] = state
        self._buffers = None
        self._buffer_tensors = None
        self.host_chunk_bytes = 0
        if any(isinstance(value, DiskState) for value in self.m):
            self._ensure_buffers()
        self.disk_read_bytes = 0
        self.disk_write_bytes = 0
        self.disk_fsync_calls = 0

    def _ensure_buffers(self):
        if self._buffers is None:
            elements = min(_legacy.CHUNK, max((p.numel() for p in self.params), default=0))
            if not elements:
                return
            self._buffers = tuple(np.empty(elements, dtype=np.float32) for _ in range(2))
            self._buffer_tensors = tuple(torch.from_numpy(value) for value in self._buffers)
            self.host_chunk_bytes = elements * 4 * 2

    @staticmethod
    def _check_size(state: DiskState):
        if state.path.stat().st_size != state.elements * 4:
            raise ValueError(f'Adam moment file has wrong size: {state.path}')

    def _read(self, handle, buffer_index: int, start: int, count: int):
        handle.seek(start * 4)
        _read_exact(handle, memoryview(self._buffers[buffer_index][:count]).cast('B'))
        self.disk_read_bytes += count * 4
        return self._buffer_tensors[buffer_index][:count]

    def _write(self, handle, buffer_index: int, start: int, count: int):
        handle.seek(start * 4)
        _write_exact(handle, memoryview(self._buffers[buffer_index][:count]).cast('B'))
        self.disk_write_bytes += count * 4

    def _sync(self, handle):
        handle.flush()
        os.fsync(handle.fileno())
        self.disk_fsync_calls += 1

    @torch.no_grad()
    def step(self, lr: float) -> None:
        # Keep the legacy global step even when an individual parameter has no
        # gradient. Changing that would change bias correction and the trajectory.
        self.t += 1
        bc1 = 1 - _legacy.B1 ** self.t
        bc2 = 1 - _legacy.B2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad.view(-1)
            flat = p.view(-1)
            resident = isinstance(m, torch.Tensor)
            if resident:
                if not isinstance(v, torch.Tensor):
                    raise ValueError('mixed storage within an Adam moment pair')
                for start in range(0, flat.numel(), _legacy.CHUNK):
                    stop = min(start + _legacy.CHUNK, flat.numel())
                    gg = g[start:stop]
                    gm, gv = m[start:stop], v[start:stop]
                    gm.mul_(_legacy.B1).add_(gg, alpha=1 - _legacy.B1)
                    gv.mul_(_legacy.B2).addcmul_(gg, gg, value=1 - _legacy.B2)
                    denom = (gv / bc2).sqrt_().add_(_legacy.EPS)
                    flat[start:stop].addcdiv_(gm, denom, value=-lr / bc1)
                    del denom
                continue
            if not isinstance(m, DiskState) or not isinstance(v, DiskState):
                raise ValueError('disk moments must use bounded file storage')
            self._ensure_buffers()
            self._check_size(m)
            self._check_size(v)
            with m.path.open('r+b', buffering=0) as fm, v.path.open('r+b', buffering=0) as fv:
                for start in range(0, flat.numel(), _legacy.CHUNK):
                    stop = min(start + _legacy.CHUNK, flat.numel())
                    count = stop - start
                    gg = g[start:stop]
                    gm = self._read(fm, 0, start, count).to(p.device)
                    gv = self._read(fv, 1, start, count).to(p.device)
                    # Same order, dtypes and operators as legacy OffloadedAdam.
                    gm.mul_(_legacy.B1).add_(gg, alpha=1 - _legacy.B1)
                    gv.mul_(_legacy.B2).addcmul_(gg, gg, value=1 - _legacy.B2)
                    denom = (gv / bc2).sqrt_().add_(_legacy.EPS)
                    flat[start:stop].addcdiv_(gm, denom, value=-lr / bc1)
                    # Blocking device-to-host copies reuse existing buffers.
                    # There is no second gm.cpu()/gv.cpu() host allocation.
                    self._buffer_tensors[0][:count].copy_(gm)
                    self._buffer_tensors[1][:count].copy_(gv)
                    self._write(fm, 0, start, count)
                    self._write(fv, 1, start, count)
                    del gm, gv, denom
                # Unbuffered Python I/O still uses the OS cache. fsync makes
                # written pages clean before these file handles are closed.
                self._sync(fm)
                self._sync(fv)

    @torch.no_grad()
    def release_gpu(self) -> None:
        """Preserve the legacy optional parking API, with bounded host copies."""
        self._ensure_buffers()
        for index, home in enumerate(self.home):
            if home != 'cuda':
                continue
            for store, name in ((self.m, 'm'), (self.v, 'v')):
                value = store[index]
                if isinstance(value, DiskState):
                    continue
                state = DiskState(self.spill_dir / f'park_{name}{index}.f32', value.numel())
                with state.path.open('w+b', buffering=0) as handle:
                    for start in range(0, state.elements, _legacy.CHUNK):
                        count = min(_legacy.CHUNK, state.elements - start)
                        self._buffer_tensors[0][:count].copy_(value[start:start + count])
                        self._write(handle, 0, start, count)
                    self._sync(handle)
                store[index] = state
                del value
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reacquire_gpu(self) -> None:
        """Restore the constructor's moment placement without full host copies."""
        self._ensure_buffers()
        for index, home in enumerate(self.home):
            if home != 'cuda':
                continue
            for store in (self.m, self.v):
                state = store[index]
                if isinstance(state, torch.Tensor):
                    continue
                self._check_size(state)
                value = torch.empty(state.elements, dtype=torch.float32, device='cuda')
                with state.path.open('rb', buffering=0) as handle:
                    for start in range(0, state.elements, _legacy.CHUNK):
                        count = min(_legacy.CHUNK, state.elements - start)
                        value[start:start + count].copy_(self._read(handle, 0, start, count))
                store[index] = value
        gc.collect()
        torch.cuda.empty_cache()
