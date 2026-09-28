"""Bounded ordinary-file safetensors IO, without memory-mapped tensor storage.

Only BF16 and F32 are supported. Header, inventory and range validation use the
standard library; tensor IO imports torch lazily. Each read owns a
new writable buffer of at most MAX_ELEMENTS * 4 bytes. Returned CPU tensors keep
that buffer alive and cannot alias another read. Callers should release each
chunk before advancing; this helper never caches tensor bytes or opens a mmap.

One TensorFile instance is a sequential reader, not a thread-safe shared cursor.
``write_parameters`` also imports torch lazily and streams contiguous CPU/CUDA
float32 tensors to a new file, copying only one bounded chunk to the CPU at a time.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import sys

MAX_ELEMENTS = 1 << 18
MAX_HEADER_BYTES = 32 * 1024 * 1024
ITEM_BYTES = {'BF16': 2, 'F32': 4}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate safetensors JSON key: {key}')
        result[key] = value
    return result


def _header(handle, path: Path) -> tuple[dict, int]:
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError(f'truncated safetensors length prefix: {path}')
    size = struct.unpack('<Q', prefix)[0]
    if not 2 <= size <= MAX_HEADER_BYTES:
        raise ValueError(f'unsafe safetensors header length: {path}')
    raw = handle.read(size)
    if len(raw) != size:
        raise ValueError(f'truncated safetensors header: {path}')
    header = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError(f'safetensors header must be an object: {path}')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError(f'invalid safetensors metadata: {path}')
    if not header:
        raise ValueError(f'empty safetensors tensor inventory: {path}')
    intervals = []
    for name, info in header.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            raise ValueError(f'invalid safetensors tensor entry: {name!r}')
        if set(info) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError(f'invalid safetensors tensor fields: {name}')
        shape, dtype, offsets = info['shape'], info['dtype'], info['data_offsets']
        if (not isinstance(dtype, str) or dtype not in ITEM_BYTES or not isinstance(shape, list)
                or any(type(dimension) is not int or dimension < 0 for dimension in shape)):
            raise ValueError(f'unsupported dtype or invalid tensor shape: {name}')
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)):
            raise ValueError(f'invalid safetensors tensor offsets: {name}')
        start, end = offsets
        if end < start or end - start != math.prod(shape) * ITEM_BYTES[dtype]:
            raise ValueError(f'tensor byte range does not match shape/dtype: {name}')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f'overlap or gap in safetensors data: {path}')
        cursor = end
    data_start = 8 + size
    if data_start + cursor != os.fstat(handle.fileno()).st_size:
        raise ValueError(f'safetensors file size does not match declared data: {path}')
    return header, data_start


class TensorFile:
    """A validated safetensors file with bounded flat-element reads.

    ``read(name, start, count)`` returns one flat CPU tensor in its stored dtype.
    ``read_bytes`` performs the same range checks and returns a bytearray without
    importing torch, useful for header/IO tests and non-tensor consumers.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open('rb', buffering=0)
        try:
            self.header, self.data_start = _header(self._file, self.path)
        except BaseException:
            self._file.close()
            raise

    def __enter__(self):
        if self._file.closed:
            raise ValueError('cannot reopen a closed TensorFile')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        self._file.close()

    def _range(self, name: str, start: int, count: int) -> tuple[int, int]:
        if self._file.closed:
            raise ValueError('TensorFile is closed')
        if name not in self.header:
            raise ValueError(f'tensor absent from file: {name}')
        if type(start) is not int or type(count) is not int or start < 0 or not 1 <= count <= MAX_ELEMENTS:
            raise ValueError(f'read must contain 1..{MAX_ELEMENTS} elements with a nonnegative start')
        info = self.header[name]
        if start + count > math.prod(info['shape']):
            raise ValueError(f'read exceeds tensor element range: {name}')
        item_bytes = ITEM_BYTES[info['dtype']]
        return self.data_start + info['data_offsets'][0] + start * item_bytes, count * item_bytes

    def read_bytes(self, name: str, start: int, count: int) -> bytearray:
        offset, length = self._range(name, start, count)
        self._file.seek(offset)
        buffer = bytearray(length)
        view = memoryview(buffer)
        consumed = 0
        while consumed < length:
            read = self._file.readinto(view[consumed:])
            if not read:
                raise ValueError(f'tensor data truncated during read: {name}')
            consumed += read
        return buffer

    def read(self, name: str, start: int, count: int):
        if sys.byteorder != 'little':
            raise ValueError('tensor_stream requires a little-endian host')
        buffer = self.read_bytes(name, start, count)
        import torch
        dtype = torch.bfloat16 if self.header[name]['dtype'] == 'BF16' else torch.float32
        return torch.frombuffer(buffer, dtype=dtype, count=count)


def _write_all(handle, payload) -> None:
    view = memoryview(payload).cast('B')
    written = 0
    while written < len(view):
        count = handle.write(view[written:])
        if not count:
            raise OSError('safetensors write made no progress')
        written += count


def _cpu_chunk(parameter, start: int, count: int):
    """The only device-to-host copy in the writer; never flatten by copying."""
    return parameter.detach().view(-1)[start:start + count].to(device='cpu', non_blocking=False)


def write_parameters(path: str | Path, parameters: dict, guard=None) -> dict:
    """Write contiguous CPU/CUDA F32 parameters without a whole CPU checkpoint copy.

    The caller owns the parameters and must not mutate them concurrently. Every
    tensor is serialized in logical flat order; inputs are never modified. The
    directory must already exist and the output file is create-only. ``guard()``
    runs before each data chunk, after the header has been written. Guard, finite
    value or IO failures leave the incomplete file in place and propagate.

    Returns tensor/byte inventory only; callers can hash the final file with their
    streaming SHA helper. A write buffer is at most MAX_ELEMENTS * 4 bytes (1 MiB
    at the default), plus a smaller temporary finite-check mask.
    """
    import torch
    if sys.byteorder != 'little':
        raise ValueError('tensor_stream requires a little-endian host')
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError('writer requires a nonempty parameter dictionary')
    if any(not isinstance(name, str) or not name or name == '__metadata__' for name in parameters):
        raise ValueError('writer requires nonempty tensor names distinct from __metadata__')
    if type(MAX_ELEMENTS) is not int or MAX_ELEMENTS <= 0:
        raise ValueError('writer chunk bound must be a positive integer')
    path = Path(path)
    header, offset = {}, 0
    for name, parameter in sorted(parameters.items()):
        if (not isinstance(parameter, torch.Tensor) or parameter.dtype != torch.float32
                or parameter.device.type not in ('cpu', 'cuda') or not parameter.is_contiguous()
                or parameter.layout != torch.strided):
            raise ValueError(f'writer requires contiguous CPU/CUDA float32 tensors: {name}')
        end = offset + parameter.numel() * 4
        header[name] = {'dtype': 'F32', 'shape': list(parameter.shape), 'data_offsets': [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('utf-8')
    raw += b' ' * ((-len(raw)) % 8)
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError('writer safetensors header exceeds reader bound')
    with path.open('xb', buffering=0) as handle, torch.no_grad():
        _write_all(handle, struct.pack('<Q', len(raw)))
        _write_all(handle, raw)
        for name in header:
            parameter = parameters[name]
            for start in range(0, parameter.numel(), MAX_ELEMENTS):
                if guard is not None:
                    guard()
                count = min(MAX_ELEMENTS, parameter.numel() - start)
                chunk = _cpu_chunk(parameter, start, count)
                if not bool(torch.isfinite(chunk).all()):
                    raise ValueError(f'nonfinite tensor chunk while saving: {name} at {start}')
                _write_all(handle, memoryview(chunk.numpy()))
                del chunk
        handle.flush()
        os.fsync(handle.fileno())
    # Header/size validation is bounded and uses no tensor-data mmap.
    with TensorFile(path) as saved:
        if saved.header != header:
            raise ValueError('written safetensors inventory failed verification')
    return {'path': str(path), 'tensors': len(header), 'parameters': offset // 4,
            'tensor_shapes': {name: info['shape'] for name, info in header.items()},
            'dtype': 'float32', 'bytes': path.stat().st_size, 'data_bytes': offset,
            'max_chunk_elements': MAX_ELEMENTS}
