"""Bounded file-reader tests. MetadataTests is stdlib-only; TensorTests imports torch."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from experiments.rab001_exploration import tensor_stream as stream


def fixture(path, header, data, *, raw_header=None):
    raw = json.dumps(header, separators=(',', ':')).encode() if raw_header is None else raw_header
    raw += b' ' * ((-len(raw)) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + data)


def mixed_header():
    return {'matrix': {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]},
            'vector': {'dtype': 'BF16', 'shape': [3], 'data_offsets': [16, 22]}}


class MetadataTests(unittest.TestCase):
    def test_mixed_dtype_flat_reads_and_independent_buffers(self):
        data = struct.pack('<4f3H', 1.0, 2.0, 3.0, 4.0, 0x3f80, 0x4000, 0x4040)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.safetensors'
            fixture(path, {**mixed_header(), '__metadata__': {'fixture': 'true'}}, data)
            with stream.TensorFile(path) as reader:
                self.assertEqual(reader.header, mixed_header())
                with patch.object(stream, 'MAX_ELEMENTS', 2):
                    first = reader.read_bytes('matrix', 1, 2)  # crosses the matrix row boundary
                    self.assertEqual(struct.unpack('<2f', first), (2.0, 3.0))
                    second = reader.read_bytes('vector', 1, 2)
                    self.assertEqual(struct.unpack('<2H', second), (0x4000, 0x4040))
                    second[0] = 255
                    self.assertEqual(struct.unpack('<2f', first), (2.0, 3.0))
                self.assertEqual(reader.read_bytes('vector', 2, 1), struct.pack('<H', 0x4040))
            with self.assertRaisesRegex(ValueError, 'closed'):
                reader.read_bytes('matrix', 0, 1)

    def test_ranges_are_bounded_before_reading_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.safetensors'
            fixture(path, mixed_header(), bytes(22))
            with stream.TensorFile(path) as reader:
                for start, count in ((-1, 1), (0, 0), (0, stream.MAX_ELEMENTS + 1),
                                     (2, 3), (True, 1), (0, False)):
                    with self.subTest(start=start, count=count), self.assertRaises(ValueError):
                        reader.read_bytes('matrix', start, count)
                with self.assertRaisesRegex(ValueError, 'absent'):
                    reader.read_bytes('missing', 0, 1)

    def test_header_rejects_overlap_gap_bad_dtype_and_bad_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.safetensors'
            for changed in ({'data_offsets': [14, 20]}, {'data_offsets': [18, 24]},
                            {'dtype': 'F64'}, {'shape': [True]}, {'shape': [-3]},
                            {'data_offsets': [-2, 4]}, {'data_offsets': [16, 24]}):
                header = mixed_header()
                header['vector'].update(changed)
                fixture(path, header, bytes(22))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    stream.TensorFile(path)

    def test_header_rejects_duplicate_keys_truncation_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.safetensors'
            entry = b'{"dtype":"F32","shape":[1],"data_offsets":[0,4]}'
            fixture(path, {}, bytes(4), raw_header=b'{"x":' + entry + b',"x":' + entry + b'}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                stream.TensorFile(path)
            for data in (bytes(21), bytes(23)):
                fixture(path, mixed_header(), data)
                with self.assertRaisesRegex(ValueError, 'file size'):
                    stream.TensorFile(path)
            path.write_bytes(struct.pack('<Q', stream.MAX_HEADER_BYTES + 1))
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                stream.TensorFile(path)
            path.write_bytes(struct.pack('<Q', 16) + b'{}')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                stream.TensorFile(path)

    def test_mid_read_truncation_is_not_zero_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.safetensors'
            fixture(path, mixed_header(), bytes(22))
            with stream.TensorFile(path) as reader:
                with path.open('r+b') as mutation:
                    mutation.truncate(path.stat().st_size - 1)
                with self.assertRaisesRegex(ValueError, 'truncated during read'):
                    reader.read_bytes('vector', 2, 1)


class TensorTests(unittest.TestCase):
    def test_bf16_f32_dtype_values_and_lifetime_without_mmap(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.safetensors'
            fixture(path, mixed_header(), struct.pack('<4f3H', 1, -2, 3, -4,
                                                     0x3f80, 0xc000, 0x4040))
            with patch('mmap.mmap', side_effect=AssertionError('mmap forbidden')):
                with stream.TensorFile(path) as reader:
                    first = reader.read('matrix', 1, 3)
                    second = reader.read('vector', 0, 3)
                self.assertEqual(first.dtype, torch.float32)
                self.assertEqual(second.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(first, torch.tensor([-2.0, 3.0, -4.0])))
                self.assertTrue(torch.equal(second, torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16)))
                second.zero_()
                self.assertTrue(torch.equal(first, torch.tensor([-2.0, 3.0, -4.0])))

    def _writer_roundtrip(self, device):
        import torch
        from safetensors import safe_open
        parameters = {'matrix': torch.nn.Parameter(torch.arange(15, dtype=torch.float32,
                                                                device=device).reshape(3, 5) / 7),
                      'scalar': torch.tensor(-0.0, dtype=torch.float32, device=device),
                      'vector': torch.tensor([1e-30, -1e30, 0.125, -0.375, 19.25], device=device),
                      'empty': torch.empty((0, 3), dtype=torch.float32, device=device)}
        original = {name: value.detach().cpu().clone() for name, value in parameters.items()}
        seen, guards = [], []
        original_cpu_chunk = stream._cpu_chunk
        def observed_chunk(parameter, start, count):
            seen.append((start, count, parameter.device.type))
            result = original_cpu_chunk(parameter, start, count)
            self.assertEqual(result.device.type, 'cpu')
            self.assertEqual(result.numel(), count)
            self.assertLessEqual(result.numel(), 4)
            return result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'streamed.safetensors'
            with patch.object(stream, 'MAX_ELEMENTS', 4), patch.object(stream, '_cpu_chunk', observed_chunk):
                record = stream.write_parameters(path, parameters, guard=lambda: guards.append(1))
            self.assertEqual(record['parameters'], 21)
            self.assertEqual(record['data_bytes'], 84)
            self.assertEqual(record['max_chunk_elements'], 4)
            self.assertEqual(record['bytes'], path.stat().st_size)
            self.assertEqual(len(guards), len(seen))
            self.assertTrue(all(count <= 4 and kind == device for _, count, kind in seen))
            self.assertIn((12, 3, device), seen)  # final matrix chunk crosses logical rows
            with safe_open(str(path), framework='pt', device='cpu') as saved:
                self.assertEqual(set(saved.keys()), set(parameters))
                for name, expected in original.items():
                    actual = saved.get_tensor(name)
                    self.assertEqual(actual.shape, expected.shape)
                    # Integer views also verify the sign bit of zero exactly.
                    self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)), name)
                    self.assertTrue(torch.equal(parameters[name].detach().cpu().view(torch.int32),
                                                expected.view(torch.int32)), name)
            unchanged = path.read_bytes()
            with self.assertRaises(FileExistsError):
                stream.write_parameters(path, parameters)
            self.assertEqual(path.read_bytes(), unchanged)

    def test_writer_cpu_roundtrip_chunk_bound_and_source_immutability(self):
        self._writer_roundtrip('cpu')

    def test_writer_cuda_roundtrip_chunk_bound_and_source_immutability(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA runtime required for device-to-host writer coverage')
        self._writer_roundtrip('cuda')

    def test_writer_preserves_partial_file_on_guard_or_nonfinite_failure(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial.safetensors'
            calls = []
            def guard():
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('simulated memory guard')
            with patch.object(stream, 'MAX_ELEMENTS', 3):
                with self.assertRaisesRegex(RuntimeError, 'simulated memory guard'):
                    stream.write_parameters(path, {'weight': torch.arange(7, dtype=torch.float32)}, guard)
            self.assertTrue(path.is_file())
            raw = path.read_bytes()
            size = struct.unpack('<Q', raw[:8])[0]
            self.assertEqual(len(raw) - 8 - size, 3 * 4)
            with self.assertRaisesRegex(ValueError, 'file size'):
                stream.TensorFile(path)
            bad = Path(directory) / 'nonfinite.safetensors'
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                stream.write_parameters(bad, {'bad': torch.tensor([float('nan')])})
            self.assertTrue(bad.is_file())

    def test_writer_rejects_dtype_and_noncontiguous_without_copying(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.safetensors'
            for tensor in (torch.ones(2, dtype=torch.float64), torch.ones(2, 3).t()):
                with self.assertRaisesRegex(ValueError, 'contiguous CPU/CUDA float32'):
                    stream.write_parameters(path, {'weight': tensor})
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
