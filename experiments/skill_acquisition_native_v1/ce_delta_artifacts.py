"""Streamed framework-free saved FP32 delta verification; never execution proof."""
from __future__ import annotations

import hashlib
import math
import struct

from . import ce_qualification_artifacts as raw

require = raw.require


def read_delta(reader, reference, binding, inventory):
    require(type(reader) is raw.RawEvidenceReader and type(reference) is dict
            and raw._int(reference.get('bytes'), 9), 'bounded paid saved delta reader/reference')
    require(type(inventory) is list and 0 < len(inventory) <= 1024, 'bounded delta inventory count')
    with reader._overhead('saved_delta_inventory_and_numeric_work', work_bytes=32 * reference['bytes'] + raw.MAX_BINDING_BYTES,
                          hash_attempts=2 + len(inventory), hash_bytes=2 * reference['bytes'] + raw.MAX_BINDING_BYTES):
        require(type(inventory) is list and inventory and all(type(entry) is dict
                and set(entry) == {'name', 'shape', 'dtype'} and type(entry['name']) is str and entry['name']
                and '\0' not in entry['name'] and '\n' not in entry['name'] and entry['dtype'] == 'F32'
                and type(entry['shape']) is list and len(entry['shape']) <= 8
                and all(type(size) is int and 0 < size <= 2 ** 31 for size in entry['shape'])
                for entry in inventory), 'exact complete FP32 delta inventory')
        expected = {entry['name']: entry['shape'] for entry in inventory}
        require(len(expected) == len(inventory), 'unique saved delta tensor names')
        stream = reader._blocks(reference, binding, 'qualification_delta')
        buffer, header_end, header, entries = b'', None, None, None
        index, position, values_count, data_bytes = 0, 0, 0, 0
        hashes, tails = {}, {}
        square_total, square_correction = 0.0, 0.0
        data_hash = hashlib.sha256(); primary = None

        def consume(block):
            nonlocal index, position, values_count, data_bytes, square_total, square_correction
            data_hash.update(block); data_bytes += len(block)
            offset = 0
            while offset < len(block):
                require(index < len(entries), 'saved delta trailing bytes')
                name, entry = entries[index]
                end = entry['data_offsets'][1]
                length = min(end - position, len(block) - offset)
                require(length > 0, 'saved delta invalid zero tensor span')
                part = block[offset:offset + length]; hashes[name].update(part)
                joined = tails[name] + part; aligned = len(joined) // 4 * 4
                values = struct.iter_unpack('<f', joined[:aligned])
                terms = []
                for (value,) in values:
                    require(math.isfinite(value), 'saved delta contains nonfinite FP32 values')
                    terms.append(value * value); values_count += 1
                addition = math.fsum(terms); combined = square_total + addition
                square_correction += (square_total - combined) + addition if abs(square_total) >= abs(addition) else (addition - combined) + square_total
                square_total = combined; tails[name] = joined[aligned:]
                offset += length; position += length
                if position == end:
                    require(not tails[name], 'saved delta incomplete FP32 value')
                    index += 1

        try:
            for block in stream:
                if header is not None:
                    consume(block); continue
                buffer += block
                if header_end is None and len(buffer) >= 8:
                    length = int.from_bytes(buffer[:8], 'little')
                    require(0 < length <= raw.CHUNK, 'bounded saved delta header')
                    header_end = 8 + length
                if header_end is None or len(buffer) < header_end: continue
                header = raw._json(buffer[8:header_end])
                require(type(header) is dict and set(header) == set(expected), 'saved delta tensor inventory differs')
                entries = sorted(header.items(), key=lambda pair: pair[1]['data_offsets'][0])
                end = 0
                for name, entry in entries:
                    size = 4 * math.prod(expected[name])
                    require(entry == {'dtype': 'F32', 'shape': expected[name], 'data_offsets': [end, end + size]},
                            'saved delta dtype/shape/contiguous offsets differ')
                    end += size
                    hashes[name] = hashlib.sha256(raw.canonical({'dtype': 'torch.float32', 'shape': expected[name]}) + b'\n')
                    tails[name] = b''
                require(reference['bytes'] == header_end + end, 'saved delta exact payload length')
                payload = buffer[header_end:]; buffer = b''
                if payload: consume(payload)
            require(header is not None and index == len(entries)
                    and data_bytes == 4 * sum(math.prod(shape) for shape in expected.values()), 'truncated saved FP32 delta')
            digests = {name: value.hexdigest() for name, value in hashes.items()}
            state = hashlib.sha256()
            for name in sorted(digests): state.update(name.encode('utf8') + b'\0' + digests[name].encode('ascii') + b'\n')
            return {'data_section_sha256': data_hash.hexdigest(), 'state_sha256': state.hexdigest(),
                    'delta_norm': math.sqrt(square_total + square_correction), 'numel': values_count, 'tensors': len(expected),
                    'dtype': 'float32', 'all_values_finite': True, 'execution_verified': False, 'native_qualified': False}
        except BaseException as error:
            primary = error; raise
        finally:
            try: stream.close()
            except BaseException as error:
                if primary is not None: raw._accounting(primary, error); raise primary from error
                raise
