"""Tiny optimizer transport equivalence checks; no source model is loaded.

Run with the CUDA runtime and --require-cuda to exercise mixed GPU/disk state.
"""
import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

streamed = importlib.import_module('experiments.rab001_exploration.streamed_adam')
legacy = streamed._legacy


@contextmanager
def recipe(chunk=4):
    names = ('B1', 'B2', 'EPS', 'CHUNK')
    saved = {name: getattr(legacy, name) for name in names}
    legacy.B1, legacy.B2, legacy.EPS, legacy.CHUNK = 0.73, 0.91, 1e-6, chunk
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(legacy, name, value)


def state_tensor(value):
    if isinstance(value, streamed.DiskState):
        return torch.from_numpy(np.fromfile(value.path, dtype=np.float32))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return torch.tensor(np.asarray(value), dtype=torch.float32)


def close_legacy(optimizer):
    for store in (optimizer.m, optimizer.v):
        for value in store:
            if isinstance(value, np.memmap):
                value.flush()
                value._mmap.close()


class StreamedAdamTests(unittest.TestCase):
    def compare_backends(self, device, gpu_budget, *, direct=False, sizes=(15, 7, 11), chunk=4):
        with recipe(chunk), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'RAB001_ADAM_DIRECT_IO': '1' if direct else '0'}):
            initial = [torch.linspace(-0.7, 0.9, n, dtype=torch.float32) for n in sizes]
            left = [torch.nn.Parameter(p.clone().to(device)) for p in initial]
            right = [torch.nn.Parameter(p.clone().to(device)) for p in initial]
            old = legacy.OffloadedAdam(left, 1e-5, gpu_budget, Path(tmp) / 'legacy')
            new = streamed.StreamedAdam(right, 1e-5, gpu_budget, Path(tmp) / 'streamed')
            try:
                self.assertEqual(old.home, new.home)
                self.assertEqual(old.gpu_bytes, new.gpu_bytes)
                self.assertEqual(old.bytes(), new.bytes())
                self.assertEqual(old.disk_bytes(), new.disk_bytes())
                self.assertEqual(new.host_chunk_bytes, min(chunk, max(sizes)) * 4 * 2)
                self.assertFalse(any(isinstance(x, np.memmap) for store in (new.m, new.v) for x in store))
                for step in range(5):
                    for index, (a, b) in enumerate(zip(left, right)):
                        if step == 2 or (step == 1 and index == 1):
                            a.grad = b.grad = None
                        else:
                            gradient = (torch.arange(a.numel(), dtype=torch.float32) - 3) * (step + 1) / 100
                            a.grad = gradient.clone().to(device)
                            b.grad = gradient.clone().to(device)
                    rate = 1e-5 / (step + 1)
                    old.step(rate)
                    new.step(rate)
                    self.assertEqual(old.t, new.t)
                    for a, b in zip(left, right):
                        self.assertTrue(torch.equal(a, b), f'parameter mismatch on {device}, step {step}')
                    for old_store, new_store in ((old.m, new.m), (old.v, new.v)):
                        for a, b in zip(old_store, new_store):
                            self.assertTrue(torch.equal(state_tensor(a), state_tensor(b)),
                                            f'moment mismatch on {device}, step {step}')
                    old.zero_grad()
                    new.zero_grad()
                    self.assertTrue(all(p.grad is None for p in left + right))
                self.assertGreater(new.disk_read_bytes, 0)
                self.assertEqual(new.disk_read_bytes, new.disk_write_bytes)
                self.assertGreater(new.disk_fsync_calls, 0)
                if direct:
                    self.assertGreater(new.direct_read_bytes, 0)
                    self.assertEqual(new.direct_read_bytes, new.disk_read_bytes)
                    self.assertEqual(new.direct_write_bytes, new.disk_write_bytes)
                    self.assertEqual(new.cached_fallback_disk_bytes, 0)
                    self.assertTrue(all(buffer.ctypes.data % new.direct_io_alignment == 0
                                        for buffer in new._buffers))
                if device == 'cuda':
                    snapshot = [[state_tensor(x) for x in store] for store in (new.m, new.v)]
                    new.release_gpu()
                    self.assertTrue(all(isinstance(x, streamed.DiskState) for store in (new.m, new.v) for x in store))
                    new.reacquire_gpu()
                    for expected_store, actual_store in zip(snapshot, (new.m, new.v)):
                        for expected, actual in zip(expected_store, actual_store):
                            self.assertTrue(torch.equal(expected, state_tensor(actual)))
                    self.assertEqual([isinstance(x, torch.Tensor) for x in new.m],
                                     [home == 'cuda' for home in new.home])
            finally:
                close_legacy(old)

    def test_cpu_all_disk_exact_equality(self):
        self.compare_backends('cpu', 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA runtime required for mixed-residency check')
    def test_cuda_mixed_residency_exact_equality(self):
        # Only the seven-element parameter fits the inherited smallest-first placement.
        self.compare_backends('cuda', 7 * 4 * 2)

    def test_truncated_state_fails_closed(self):
        with recipe(), tempfile.TemporaryDirectory() as tmp:
            parameter = torch.nn.Parameter(torch.ones(5, dtype=torch.float32))
            optimizer = streamed.StreamedAdam([parameter], 1e-5, 0, Path(tmp))
            optimizer.m[0].path.write_bytes(b'\0' * 4)
            parameter.grad = torch.ones_like(parameter)
            with self.assertRaises(ValueError):
                optimizer.step(1e-5)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows direct I/O required')
    def test_windows_direct_cpu_exact_equality(self):
        self.compare_backends('cpu', 0, direct=True, sizes=(8192, 4096, 12288), chunk=2048)

    @unittest.skipUnless(sys.platform == 'win32' and torch.cuda.is_available(), 'Windows CUDA required')
    def test_windows_direct_cuda_mixed_exact_equality(self):
        self.compare_backends('cuda', 4096 * 4 * 2, direct=True,
                              sizes=(8192, 4096, 12288), chunk=2048)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows direct I/O required')
    def test_large_unaligned_state_does_not_fall_back(self):
        with recipe(2048), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'RAB001_ADAM_DIRECT_IO': '1'}):
            parameter = torch.nn.Parameter(torch.ones(16385, dtype=torch.float32))
            with self.assertRaisesRegex(ValueError, 'no cached fallback'):
                streamed.StreamedAdam([parameter], 1e-5, 0, Path(tmp))
            self.assertTrue((Path(tmp) / 'm0.f32').is_file())

    @unittest.skipUnless(sys.platform == 'win32', 'Windows direct I/O required')
    def test_tiny_unaligned_fallback_is_explicit(self):
        with recipe(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'RAB001_ADAM_DIRECT_IO': '1'}):
            parameter = torch.nn.Parameter(torch.ones(5, dtype=torch.float32))
            optimizer = streamed.StreamedAdam([parameter], 1e-5, 0, Path(tmp))
            self.assertEqual(optimizer.cached_fallback_disk_bytes, 5 * 4 * 2)
            self.assertEqual(optimizer.direct_io_disk_bytes, 0)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step(1e-5)
            self.assertEqual(optimizer.direct_read_bytes, 0)


@unittest.skipUnless(sys.platform == 'win32', 'Windows direct I/O required')
class WindowsDirectFileTests(unittest.TestCase):
    def test_roundtrip_and_alignment_range_guards(self):
        from experiments.rab001_exploration import windows_direct_io as direct
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'moment.f32'
            path.touch()
            alignment = direct.storage_geometry(path)['io_alignment']
            with path.open('r+b') as f:
                f.truncate(2 * alignment)
            buffer = direct.aligned_float32(alignment // 2, alignment)
            buffer[:] = np.arange(buffer.size, dtype=np.float32)
            expected = buffer.copy()
            raw = memoryview(buffer).cast('B')
            with direct.DirectFile(path, 2 * alignment) as handle:
                self.assertEqual(handle.write(raw), 2 * alignment)
                handle.flush()
                buffer.fill(0)
                handle.seek(0)
                self.assertEqual(handle.readinto(raw), 2 * alignment)
                np.testing.assert_array_equal(buffer, expected)
                handle.seek(0)
                for bad in (raw[:alignment - 1], raw[1:alignment + 1], bytes(alignment)):
                    with self.assertRaises(ValueError):
                        handle.write(bad)
                with self.assertRaises(ValueError):
                    handle.seek(1)
                handle.seek(2 * alignment)
                with self.assertRaises(ValueError):
                    handle.readinto(raw[:alignment])

    def test_short_write_preserves_partial_file_and_raises(self):
        from experiments.rab001_exploration import windows_direct_io as direct
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partial.f32'
            path.touch()
            alignment = direct.storage_geometry(path)['io_alignment']
            with path.open('r+b') as f:
                f.truncate(2 * alignment)
            buffer = direct.aligned_float32(alignment // 2, alignment)
            buffer.fill(7.5)
            with direct.DirectFile(path, 2 * alignment) as handle:
                original = handle._perform
                def short(write, address, size, transferred):
                    return original(write, address, size // 2, transferred)
                with mock.patch.object(handle, '_perform', short):
                    with self.assertRaisesRegex(OSError, 'short direct write'):
                        handle.write(memoryview(buffer).cast('B'))
            data = path.read_bytes()
            self.assertEqual(data[:alignment], memoryview(buffer).cast('B')[:alignment])
            self.assertEqual(data[alignment:], bytes(alignment))

    def test_short_read_and_size_mismatch_raise(self):
        from experiments.rab001_exploration import windows_direct_io as direct
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partial.f32'
            path.touch()
            alignment = direct.storage_geometry(path)['io_alignment']
            with path.open('r+b') as f:
                f.truncate(2 * alignment)
            buffer = direct.aligned_float32(alignment // 2, alignment)
            with direct.DirectFile(path, 2 * alignment) as handle:
                original = handle._perform
                def short(write, address, size, transferred):
                    return original(write, address, size // 2, transferred)
                with mock.patch.object(handle, '_perform', short):
                    with self.assertRaisesRegex(OSError, 'short direct read'):
                        handle.readinto(memoryview(buffer).cast('B'))
            with self.assertRaises(ValueError):
                direct.DirectFile(path, alignment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--require-cuda', action='store_true')
    options, remaining = parser.parse_known_args()
    if options.require_cuda and not torch.cuda.is_available():
        raise SystemExit('CUDA equivalence is required; use the CUDA runtime')
    unittest.main(argv=[sys.argv[0], *remaining])
