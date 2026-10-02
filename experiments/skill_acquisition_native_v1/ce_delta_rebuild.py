"""Paid pure saved-delta FP32 rebuild from pinned base/warm/checkpoint bytes.

Full state names/aliases and tiny non-base buffers must be retained from the
actual model. Producer scalar hashes or residuals never substitute for bytes.
This verifies arithmetic/state linkage, not native execution or qualification.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct

from . import ce_qualification_artifacts as raw, ce_execution_verifier as linkage

SCHEMA = 'CE8-DELTA-FULLSTATE-SOURCE-1'
DTYPES = {'F32': (4, 'torch.float32'), 'BF16': (2, 'torch.bfloat16'), 'U8': (1, 'torch.uint8'), 'I64': (8, 'torch.int64')}
require = raw.require


def combine(values):
    result = hashlib.sha256()
    for name in sorted(values): result.update(name.encode('utf8') + b'\0' + values[name].encode('ascii') + b'\n')
    return result.hexdigest()


def tensor_hash(dtype, shape): return hashlib.sha256(raw.canonical({'dtype': DTYPES[dtype][1], 'shape': shape}) + b'\n')


def f32(value):
    require(math.isfinite(value), 'nonfinite FP32 rebuild arithmetic')
    try: encoded = struct.pack('<f', value)
    except OverflowError as error: raise raw.EvidenceRejected('FP32 rebuild overflow') from error
    result = struct.unpack('<f', encoded)[0]; require(math.isfinite(result), 'nonfinite rounded FP32 rebuild')
    return result


class TensorBytes:
    """Whole-file exact hash first; bounded paid reads for named tensor ranges."""
    def __init__(self, reader, reference, binding, role):
        require(type(reader) is raw.RawEvidenceReader, 'explicit independently prepaid tensor reader')
        self.reader, self.reference, self.binding, self.role = reader, reference, binding, role
        wrapped = linkage._decorated(reader, reference, binding, role)
        prefix, target, data_count = b'', None, 0
        self.header = None; payload_hash = hashlib.sha256()
        with reader._overhead('delta_tensor_header_layout', work_bytes=32 * raw.CHUNK,
                             hash_attempts=1, hash_bytes=reference['bytes']):
            for block in reader._blocks(wrapped, binding, role):
                if self.header is not None: payload_hash.update(block); data_count += len(block); continue
                prefix += block
                if target is None and len(prefix) >= 8:
                    size = int.from_bytes(prefix[:8], 'little'); require(0 < size <= raw.CHUNK, 'bounded source tensor header')
                    target = 8 + size
                if target is None or len(prefix) < target: continue
                observed = raw._json(prefix[8:target]); require(type(observed) is dict, 'source tensor header object')
                self.header = {name: entry for name, entry in observed.items() if name != '__metadata__'}
                end = 0
                for name, entry in sorted(self.header.items(), key=lambda item: item[1]['data_offsets'][0]):
                    require(type(name) is str and name and '\0' not in name and '\n' not in name
                            and type(entry) is dict and set(entry) == {'dtype', 'shape', 'data_offsets'}
                            and entry['dtype'] in DTYPES and type(entry['shape']) is list
                            and all(type(n) is int and n > 0 for n in entry['shape']), 'source tensor dtype/shape/name')
                    width = DTYPES[entry['dtype']][0]; size = width * math.prod(entry['shape'])
                    require(entry['data_offsets'] == [end, end + size], 'contiguous exact source tensor layout')
                    end += size
                require(reference['bytes'] == target + end, 'exact source tensor file byte count')
                self.start = target
                part = prefix[target:]; payload_hash.update(part); data_count += len(part); prefix = b''
            require(self.header is not None and data_count == reference['bytes'] - self.start, 'truncated source tensor payload')
        self.data_sha256 = payload_hash.hexdigest()

    def chunks(self, name):
        require(name in self.header, 'bound state tensor missing from exact source file')
        entry = self.header[name]; count = entry['data_offsets'][1] - entry['data_offsets'][0]
        # Local permanent reader bounds precede seek/open. Rehashing the whole
        # source after arithmetic detects mutation of these random read ranges.
        with self.reader._overhead('named_source_tensor_read', work_bytes=0):
            require(self.reader.remaining['io_read_bytes_upper'] >= count + 1, 'named tensor read allowance exhausted')
            self.reader.remaining['io_read_bytes_upper'] -= count + 1
            identifier = __import__('uuid').uuid4().hex
            self.reader.operation('qualification_artifact_readback', 'started', operation_id=identifier,
                io_read_bytes_upper=count + 1, identity_hash_attempts_upper=0, identity_hash_bytes_upper=0,
                path=self.reference['path'], tensor_name=name)
            read, primary = 0, None
            try:
                path = Path(self.reference['path']); require(path.is_file() and not path.is_symlink()
                    and path.stat().st_size == self.reference['bytes'], 'source tensor path/size changed')
                with path.open('rb') as stream:
                    stream.seek(self.start + entry['data_offsets'][0])
                    while read < count:
                        block = stream.read(min(raw.CHUNK, count - read)); require(block, 'truncated named source tensor')
                        read += len(block); yield block
                require(read == count, 'exact named source tensor byte count')
            except BaseException as error:
                primary = error
                try: self.reader.operation('qualification_artifact_readback', 'failed', operation_id=identifier, bytes_read=read, error_type=type(error).__name__)
                except BaseException as accounting: raw._accounting(error, accounting); raise error from accounting
                raise
            if primary is None:
                self.reader.operation('qualification_artifact_readback', 'returned', operation_id=identifier, bytes_read=read)

    def values(self, name):
        dtype = self.header[name]['dtype']; require(dtype in ('F32', 'BF16'), 'arithmetic source must be F32/BF16')
        width = DTYPES[dtype][0]; tail = b''
        for block in self.chunks(name):
            joined = tail + block; length = len(joined) // width * width; tail = joined[length:]
            for (value,) in struct.iter_unpack('<f' if dtype == 'F32' else '<H', joined[:length]):
                if dtype == 'BF16': value = struct.unpack('<f', struct.pack('<I', value << 16))[0]
                require(math.isfinite(value), 'nonfinite raw source tensor value'); yield value
        require(not tail, 'partial source scalar')

    def unchanged(self):
        wrapped = linkage._decorated(self.reader, self.reference, self.binding, self.role)
        for _ in self.reader._blocks(wrapped, self.binding, self.role): pass


def verify_rebuild(contract, readers, binding, *, checkpoint_binding):
    """Recompute complete common/master/deployed identities and all residuals.

    readers are exact independently priced RawEvidenceReader objects for base,
    warm, delta, checkpoint and optional buffers roots. External files are never
    implicitly permitted by an output-root path or a producer boolean.
    """
    require(type(contract) is dict and contract.get('schema') == SCHEMA
            and type(readers) is dict and {'base', 'warm', 'delta', 'checkpoint'} <= set(readers), 'complete raw saved-delta source contract/readers')
    coordinator = readers['delta']; require(type(coordinator) is raw.RawEvidenceReader, 'paid rebuild arithmetic owner')
    refs = contract['refs']; state = contract['full_state_inventory']; selected = contract['selected_inventory']
    total = sum(ref['bytes'] for ref in refs.values() if ref is not None)
    with coordinator._overhead('full_saved_delta_FP32_rebuild', work_bytes=64 * total,
            hash_attempts=16 + 8 * len(state), hash_bytes=16 * total + raw.MAX_BINDING_BYTES):
        require(type(state) is list and 0 < len(state) <= 4096 and len({row['name'] for row in state}) == len(state), 'complete distinct runtime state names')
        files = {role: TensorBytes(readers[role], ref, checkpoint_binding if role == 'checkpoint' else binding, 'delta_' + role)
            for role, ref in refs.items() if ref is not None}
        base, warm, saved, master = (files[name] for name in ('base', 'warm', 'delta', 'checkpoint'))
        chosen = {row['name']: row['shape'] for row in selected}
        require(set(saved.header) == set(chosen) and set(warm.header) <= set(base.header)
                and all(entry['dtype'] == 'F32' for entry in warm.header.values()), 'exact saved/warm delta support')
        require(warm.data_sha256 == contract['warm_data_sha256'], 'actual pinned warm data section digest')
        warm_hashes = {}
        for name, entry in warm.header.items():
            hashed = tensor_hash('F32', entry['shape'])
            for block in warm.chunks(name): hashed.update(block)
            warm_hashes[name] = hashed.hexdigest()
        require(combine(warm_hashes) == contract['warm_state_sha256'], 'actual pinned warm state digest')
        common_hashes, final_hashes, rebuilt_hashes, checkpoint_frozen, delta_hashes = {}, {}, {}, {}, {}
        nonzero, changed_tensors, residual_max, residual_total, residual_correction, count_all = 0, 0, 0.0, 0.0, 0.0, 0
        for row in sorted(state, key=lambda item: item['name']):
            name, shape, dtype = row['name'], row['shape'], row['dtype']
            require(row.get('kind') in ('parameter', 'buffer') and dtype in DTYPES and type(shape) is list,
                    'exact runtime state kind/dtype/shape')
            source = row['source_name']
            common_digest, final_digest, deployed_digest = tensor_hash(dtype, shape), tensor_hash(dtype, shape), tensor_hash(dtype, shape)
            if source is None:
                require(name not in chosen and 'buffers' in files and files['buffers'].header.get(name) == {
                    'dtype': dtype, 'shape': shape, 'data_offsets': files['buffers'].header[name]['data_offsets']}, 'exact tiny non-base state bytes')
                for block in files['buffers'].chunks(name): common_digest.update(block)
                final_digest = deployed_digest = common_digest
            else:
                require(dtype == 'F32' and source in base.header and base.header[source]['dtype'] == 'BF16'
                        and base.header[source]['shape'] == shape, 'actual deployed BF16 base converted to runtime FP32')
                if source in warm.header: require(warm.header[source]['shape'] == shape, 'warm/base shape binding')
                common_values = base.values(source); warm_values = warm.values(source) if source in warm.header else None
                trainable = name in chosen
                if trainable:
                    require(chosen[name] == shape and saved.header[name]['dtype'] == 'F32' and saved.header[name]['shape'] == shape
                            and master.header.get('param::' + name, {}).get('dtype') == 'F32'
                            and master.header['param::' + name]['shape'] == shape, 'selected exact parent master/delta byte inventory')
                    delta_values, master_values = saved.values(name), master.values('param::' + name)
                    delta_digest = tensor_hash('F32', shape); had_change = False
                terms, common_chunk, final_chunk, deployed_chunk, delta_chunk = [], bytearray(), bytearray(), bytearray(), bytearray()
                for base_value in common_values:
                    common = f32(base_value + next(warm_values)) if warm_values is not None else base_value
                    if trainable:
                        delta, actual_master = next(delta_values), next(master_values)
                        require(struct.pack('<f', f32(actual_master - common)) == struct.pack('<f', delta), 'saved delta differs from exact FP32 master-common subtraction')
                        deployed = f32(common + delta); difference = deployed - actual_master
                        nonzero += difference != 0; had_change |= difference != 0; residual_max = max(residual_max, abs(difference))
                        terms.append(difference * difference); count_all += 1
                        delta_chunk.extend(struct.pack('<f', delta))
                    else: actual_master = deployed = common
                    common_chunk.extend(struct.pack('<f', common)); final_chunk.extend(struct.pack('<f', actual_master)); deployed_chunk.extend(struct.pack('<f', deployed))
                    if len(common_chunk) >= raw.CHUNK:
                        common_digest.update(common_chunk); final_digest.update(final_chunk); deployed_digest.update(deployed_chunk)
                        if trainable: delta_digest.update(delta_chunk)
                        addition = math.fsum(terms); total_next = residual_total + addition
                        residual_correction += (residual_total - total_next) + addition if abs(residual_total) >= abs(addition) else (addition - total_next) + residual_total
                        residual_total = total_next; terms.clear(); common_chunk.clear(); final_chunk.clear(); deployed_chunk.clear(); delta_chunk.clear()
                common_digest.update(common_chunk); final_digest.update(final_chunk); deployed_digest.update(deployed_chunk)
                if trainable:
                    delta_digest.update(delta_chunk); delta_hashes[name] = delta_digest.hexdigest(); changed_tensors += had_change
                addition = math.fsum(terms); total_next = residual_total + addition
                residual_correction += (residual_total - total_next) + addition if abs(residual_total) >= abs(addition) else (addition - total_next) + residual_total
                residual_total = total_next
                for values in (warm_values, delta_values if trainable else None, master_values if trainable else None):
                    if values is not None: require(next(values, None) is None, 'source tensor length differs')
            common_hashes[name], final_hashes[name], rebuilt_hashes[name] = common_digest.hexdigest(), final_digest.hexdigest(), deployed_digest.hexdigest()
            if name not in chosen: checkpoint_frozen[row['kind'] + ':' + name] = final_hashes[name]
        require(set(chosen) <= set(common_hashes) and count_all == sum(math.prod(shape) for shape in chosen.values()), 'complete selected rebuild count')
        result = {'common_anchor_sha256': combine(common_hashes), 'final_master_sha256': combine(final_hashes),
            'deployed_arm_sha256': combine(rebuilt_hashes), 'delta_state_sha256': combine(delta_hashes),
            'checkpoint_frozen_state_sha256': combine(checkpoint_frozen),
            'runtime_frozen_state_sha256': combine({name: value for name, value in final_hashes.items() if name not in chosen}),
            'residual_max_abs': residual_max, 'residual_l2': math.sqrt(residual_total + residual_correction),
            'residual_nonzero_elements': nonzero, 'residual_nonzero_tensors': changed_tensors, 'selected_numel': count_all,
            'roundtrip_independently_rebuilt': True, 'native_qualified': False, 'execution_verified': False}
        for key in ('common_anchor_sha256', 'final_master_sha256', 'deployed_arm_sha256', 'checkpoint_frozen_state_sha256'):
            require(result[key] == contract[key], 'raw full-state rebuild identity mismatch: ' + key)
        for file in files.values(): file.unchanged()
        return result


def verify_stage_sources(context, payload, delta_ref, binding, *, readers):
    """Bind actual stage/full-state/parent manifest bytes, then derive rebuild."""
    output = readers['delta']
    with output._overhead('saved_delta_raw_source_linkage', work_bytes=64 * raw.MAX_JSON,
            hash_attempts=16, hash_bytes=64 * raw.MAX_JSON):
        source = output._envelope(payload['full_state_source_ref'], binding, 'delta_full_state', 'delta_full_state')
        require(source.get('schema') == SCHEMA and source['selected_inventory'] == context['bindings']['scope']['native_backend']['parameter_inventory'],
                'actual full-state writer inventory/source schema')
        parent_binding = source['checkpoint_binding']
        raw.binding(parent_binding)
        parent_ref = source['parent_manifest_ref']
        wrapped = linkage._decorated(output, parent_ref, parent_binding, 'checkpoint_manifest')
        require(parent_ref['bytes'] <= raw.MAX_JSON, 'bounded actual parent checkpoint manifest')
        blob = b''.join(output._blocks(wrapped, parent_binding, 'checkpoint_manifest'))
        manifest = raw._json(blob)
        require(raw.canonical(manifest) + b'\n' == blob and manifest.get('schema') == 'SKILL-ACQUISITION-NATIVE-CHECKPOINT-1'
                and raw.digest({key: value for key, value in manifest.items() if key != 'manifest_sha256'}) == manifest.get('manifest_sha256')
                and manifest['bindings']['qualification_evidence_binding'] == parent_binding
                and manifest['bindings']['inventory'] == source['selected_inventory']
                and manifest['progress']['C'] == manifest['progress']['U'] == 2
                and source['parent_restore_record']['manifest_sha256'] == manifest['manifest_sha256'],
                'actual source parent U2 master checkpoint/fresh restore linkage')
        donor = payload['donor']; anchors = context['bindings']['anchors']; spec = context['bindings']['scope']['native_backend']
        require(source['base_ref']['path'] == str(Path(spec['model_dir']) / 'model.safetensors')
                and source['base_ref']['sha256'] == anchors['base_weights_sha256']
                and source['warm_ref'] == {key: spec['warm_delta_ref'][key] for key in ('path', 'sha256', 'bytes')}
                and source['warm_data_sha256'] == spec['warm_delta_ref']['data_sha256']
                and source['warm_state_sha256'] == spec['warm_delta_ref']['state_sha256']
                and source['common_anchor_sha256'] == anchors['common_anchor_sha256'] == donor['common_anchor_sha256'],
                'full rebuild exact pinned external base/warm/common source bytes')
        saved = manifest['payload']
        contract = {'schema': SCHEMA, 'full_state_inventory': source['full_state_inventory'], 'selected_inventory': source['selected_inventory'],
            'refs': {'base': source['base_ref'], 'warm': source['warm_ref'], 'delta': {key: delta_ref[key] for key in ('path', 'sha256', 'bytes')},
                'checkpoint': {'path': saved['path'], 'sha256': saved['file_sha256'], 'bytes': saved['bytes']}, 'buffers': source['nonbase_ref']},
            'warm_data_sha256': source['warm_data_sha256'], 'warm_state_sha256': source['warm_state_sha256'],
            'common_anchor_sha256': anchors['common_anchor_sha256'], 'final_master_sha256': donor['final_master_sha256'],
            'deployed_arm_sha256': donor['deployed_arm_sha256'], 'checkpoint_frozen_state_sha256': manifest['frozen_state_sha256']}
    result = verify_rebuild(contract, readers, binding, checkpoint_binding=parent_binding)
    require(result['runtime_frozen_state_sha256'] == source['frozen_runtime_sha256']
            and result['delta_state_sha256'] == donor['state_sha256'], 'actual full frozen/delta state hash conventions differ')
    observed = payload['observations']; report = donor['roundtrip']
    require(observed['final_master_sha256'] == result['final_master_sha256'] and observed['rebuilt_endpoint_sha256'] == result['deployed_arm_sha256']
            and report['residual_nonzero_elements'] == result['residual_nonzero_elements']
            and report['residual_nonzero_tensors'] == result['residual_nonzero_tensors'], 'producer report differs from byte-derived endpoint/counts')
    for actual, reported in ((result['residual_max_abs'], observed['rounding_residual_max_abs']), (result['residual_l2'], observed['rounding_residual_l2'])):
        require(raw._finite(reported) and abs(actual - reported) <= 1e-12 + 1e-10 * actual, 'producer residual differs from independent saved-byte rebuild')
    return result
