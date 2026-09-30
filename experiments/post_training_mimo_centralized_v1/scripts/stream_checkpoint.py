"""Strict Titan BF16 loading through one bounded CPU read buffer.

No framework import, memory mapping, random draw, CPU model or full tensor copy.
The caller binds the entire checkpoint SHA before invoking this loader.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct


LOADER_MODE = "stream_read_2MiB_v1"
CHUNK_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024
SERIALIZED_TENSORS = 310


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate safetensors header key")
        result[key] = value
    return result


def _invalid_number(value):
    raise RuntimeError("noninteger/nonfinite safetensors header number")


def _read_exact_into(stream, view):
    position = 0
    while position < len(view):
        count = stream.readinto(view[position:])
        _require(type(count) is int and count > 0, "truncated safetensors payload/header")
        position += count


def _header(stream, file_bytes, states, torch):
    prefix = bytearray(8)
    _read_exact_into(stream, memoryview(prefix))
    header_bytes = struct.unpack("<Q", prefix)[0]
    _require(2 <= header_bytes <= MAX_HEADER_BYTES and header_bytes <= file_bytes - 8,
             "invalid or oversized safetensors header length")
    raw = bytearray(header_bytes)
    _read_exact_into(stream, memoryview(raw))
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_float=_invalid_number, parse_constant=_invalid_number)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise RuntimeError("invalid safetensors header JSON") from error
    _require(type(header) is dict, "safetensors header must be an object")
    if "__metadata__" in header:
        metadata = header.pop("__metadata__")
        _require(type(metadata) is dict and all(type(k) is str and type(v) is str for k, v in metadata.items()),
                 "invalid safetensors metadata")
    keys = set(header)
    _require(len(keys) == SERIALIZED_TENSORS and set(states) - keys == {"lm_head.weight"}
             and not keys - set(states), "checkpoint key mismatch")
    payload_bytes = file_bytes - 8 - header_bytes
    intervals = []
    for name, entry in header.items():
        _require(type(entry) is dict and set(entry) == {"dtype", "shape", "data_offsets"},
                 "invalid tensor header fields: " + name)
        shape, offsets = entry["shape"], entry["data_offsets"]
        _require(entry["dtype"] == "BF16" and states[name].dtype == torch.bfloat16,
                 "tensor dtype mismatch: " + name)
        _require(type(shape) is list and all(type(dim) is int and dim > 0 for dim in shape)
                 and shape == list(states[name].shape), "tensor shape mismatch: " + name)
        _require(states[name].is_contiguous(), "noncontiguous checkpoint target: " + name)
        _require(type(offsets) is list and len(offsets) == 2
                 and all(type(value) is int and value >= 0 for value in offsets),
                 "invalid tensor offsets: " + name)
        begin, end = offsets
        _require(end <= payload_bytes and end - begin == math.prod(shape) * 2,
                 "tensor byte span mismatch: " + name)
        intervals.append((begin, end, name))
    end = 0
    for begin, next_end, name in sorted(intervals):
        _require(begin == end, "overlapping or gapped safetensors payload: " + name)
        end = next_end
    _require(end == payload_bytes, "unclaimed safetensors payload bytes")
    return header, header_bytes, payload_bytes


def load_streamed(torch, model, model_path, accounting, tensor_digest):
    """Return source tensor hashes and diagnostics after exact GPU-copy verification.

    tensor_copy is one composite read/copy span per serialized tensor, preserving
    the 310 original journal labels. Each CPU-to-device chunk copy is blocking so
    the single writable buffer can be safely reused. Digest verification follows
    the completed copy span. The caller retains tie/parameter-count assertions.
    """
    _require(callable(tensor_digest), "tensor digest callback required")
    states = model.state_dict()
    path = Path(model_path)
    file_bytes = path.stat().st_size
    hashes, chunk_count, maximum_read = {}, 0, 0
    buffer = bytearray(CHUNK_BYTES)
    with path.open("rb", buffering=0) as stream:
        header, header_bytes, payload_bytes = _header(stream, file_bytes, states, torch)
        for name in sorted(header):
            begin, end = header[name]["data_offsets"]
            target = states[name].detach().reshape(-1)

            def copy_tensor():
                nonlocal chunk_count, maximum_read
                stream.seek(8 + header_bytes + begin)
                source_hash = hashlib.sha256()
                offset = 0
                while offset < end - begin:
                    count = min(CHUNK_BYTES, end - begin - offset)
                    view = memoryview(buffer)[:count]
                    _read_exact_into(stream, view)
                    source_hash.update(view)
                    source = torch.frombuffer(view, dtype=torch.bfloat16)
                    target[offset // 2:(offset + count) // 2].copy_(source, non_blocking=False)
                    del source, view
                    chunk_count += 1
                    maximum_read = max(maximum_read, count)
                    offset += count
                return source_hash.hexdigest()

            expected = accounting.call("tensor_copy", copy_tensor, stage="tensor_copy:" + name)
            hashes[name] = expected
            _require(tensor_digest(torch, states[name]) == expected, "loaded tensor mismatch: " + name)
    return hashes, dict(loader_mode=LOADER_MODE, header_bytes=header_bytes, file_bytes=file_bytes,
                        payload_bytes=payload_bytes, serialized_tensors=len(hashes),
                        read_buffer_bytes=CHUNK_BYTES, maximum_payload_read_bytes=maximum_read,
                        payload_chunk_count=chunk_count, memory_mapped=False,
                        cpu_model_allocated=False, blocking_chunk_copies=True)
