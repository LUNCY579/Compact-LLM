"""Tiny real safetensor arithmetic fixtures; no framework/model/OS execution."""
import copy
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_delta_rebuild as d
from experiments.skill_acquisition_native_v1 import ce_delta_state_writer as writer


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='delta-rebuild-pure-'); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.events = []
        self.binding = {'stage': 'delta_roundtrip', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
            'static_recipe_sha256': 'c' * 64, 'code_sha256': {'fixture.py': 'd' * 64}, 'worker_identity_sha256': 'e' * 64, 'attempt_id': 'delta'}
    def file(self, name, tensors):
        header, body = {}, b''
        for key, (dtype, shape, values) in sorted(tensors.items()):
            if dtype == 'BF16': payload = b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', value))[0] >> 16) for value in values)
            elif dtype == 'F32': payload = struct.pack('<%df' % len(values), *values)
            else: payload = bytes(values)
            header[key] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(body), len(body) + len(payload)]}; body += payload
        encoded = d.raw.canonical(header); path = self.root / (name + '.safetensors')
        blob = struct.pack('<Q', len(encoded)) + encoded + body; path.write_bytes(blob)
        return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}, hashlib.sha256(body).hexdigest()
    def digest(self, values):
        out = {}
        for name, values in values.items():
            shape = [len(values)]; h = hashlib.sha256(d.raw.canonical({'dtype': 'torch.float32', 'shape': shape}) + b'\n')
            h.update(struct.pack('<%df' % len(values), *values)); out[name] = h.hexdigest()
        result = hashlib.sha256()
        for name in sorted(out): result.update(name.encode() + b'\0' + out[name].encode() + b'\n')
        return result.hexdigest(), out
    def fixture(self):
        base_values = [100000000.0, 1.0, -2.0]
        bf = [struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', value))[0] & 0xffff0000))[0] for value in base_values]
        warm_values = [0.0, .25, 0.0]; common = [struct.unpack('<f', struct.pack('<f', x + y))[0] for x, y in zip(bf, warm_values)]
        master = [1.0, 1.125, -1.9999998807907104]
        delta = [struct.unpack('<f', struct.pack('<f', x - y))[0] for x, y in zip(master, common)]
        deployed = [struct.unpack('<f', struct.pack('<f', x + y))[0] for x, y in zip(common, delta)]
        frozen, extra = [.5], [1.0, .125]
        refs = {}; refs['base'], _ = self.file('base', {'w': ('BF16', [3], base_values), 'frozen': ('BF16', [1], frozen)})
        refs['warm'], warm_data = self.file('warm', {'w': ('F32', [3], warm_values)})
        refs['delta'], _ = self.file('delta', {'w': ('F32', [3], delta)})
        refs['checkpoint'], _ = self.file('checkpoint', {'param::w': ('F32', [3], master), 'rng::cpu': ('U8', [1], [1])})
        refs['buffers'], _ = self.file('buffers', {'rope': ('F32', [2], extra)})
        common_map = {'w': common, 'frozen': frozen, 'rope': extra}; final_map = dict(common_map, w=master); deployed_map = dict(common_map, w=deployed)
        common_hash, _ = self.digest(common_map); final_hash, final_digests = self.digest(final_map); deployed_hash, _ = self.digest(deployed_map)
        frozen_sha = hashlib.sha256()
        for name, digest in sorted({'parameter:frozen': final_digests['frozen'], 'buffer:rope': final_digests['rope']}.items()):
            frozen_sha.update(name.encode() + b'\0' + digest.encode() + b'\n')
        warm_state, _ = self.digest({'w': warm_values})
        contract = {'schema': d.SCHEMA, 'refs': refs, 'warm_data_sha256': warm_data, 'warm_state_sha256': warm_state,
            'selected_inventory': [{'name': 'w', 'shape': [3], 'dtype': 'F32'}],
            'full_state_inventory': [{'name': 'w', 'shape': [3], 'dtype': 'F32', 'kind': 'parameter', 'source_name': 'w'},
                {'name': 'frozen', 'shape': [1], 'dtype': 'F32', 'kind': 'parameter', 'source_name': 'frozen'},
                {'name': 'rope', 'shape': [2], 'dtype': 'F32', 'kind': 'buffer', 'source_name': None}],
            'common_anchor_sha256': common_hash, 'final_master_sha256': final_hash, 'deployed_arm_sha256': deployed_hash,
            'checkpoint_frozen_state_sha256': frozen_sha.hexdigest()}
        differences = [x - y for x, y in zip(deployed, master)]
        return contract, differences
    def readers(self):
        return {role: d.raw.RawEvidenceReader(lambda *args, **kwargs: self.events.append((args, kwargs)),
            {key: 10 ** 12 for key in d.raw.BUDGET_FIELDS}, self.root) for role in ('base', 'warm', 'delta', 'checkpoint', 'buffers')}
    def test_saved_bytes_recompute_full_anchor_master_deployed_and_nonzero_residual(self):
        contract, differences = self.fixture()
        result = d.verify_rebuild(contract, self.readers(), self.binding, checkpoint_binding=self.binding)
        self.assertEqual(result['selected_numel'], 3); self.assertEqual(result['residual_nonzero_elements'], 1)
        self.assertEqual(result['residual_max_abs'], max(abs(x) for x in differences))
        self.assertEqual(result['residual_l2'], math.sqrt(math.fsum(x * x for x in differences)))
        self.assertTrue(result['roundtrip_independently_rebuilt']); self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])
    def test_rehashed_wrong_saved_delta_or_selected_master_still_fails_arithmetic(self):
        contract, _ = self.fixture()
        for role, tensors in (('delta', {'w': ('F32', [3], [1.0, 2.0, 3.0])}), ('checkpoint', {'param::w': ('F32', [3], [2.0, 1.125, -2.0])})):
            changed = copy.deepcopy(contract); changed['refs'][role], _ = self.file('changed-' + role, tensors)
            with self.assertRaisesRegex(ValueError, 'master-common'): d.verify_rebuild(changed, self.readers(), self.binding, checkpoint_binding=self.binding)
    def test_nonfinite_or_unrelated_nonbase_buffer_cannot_match_full_anchor(self):
        contract, _ = self.fixture(); changed = copy.deepcopy(contract)
        changed['refs']['buffers'], _ = self.file('changed-buffer', {'rope': ('F32', [2], [2.0, .125])})
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): d.verify_rebuild(changed, self.readers(), self.binding, checkpoint_binding=self.binding)
        changed = copy.deepcopy(contract); changed['refs']['base'], _ = self.file('nonfinite', {'w': ('BF16', [3], [float('inf'), 1, -2]), 'frozen': ('BF16', [1], [.5])})
        with self.assertRaisesRegex(ValueError, 'nonfinite'): d.verify_rebuild(changed, self.readers(), self.binding, checkpoint_binding=self.binding)
    def test_missing_fullstate_or_forged_complete_hash_is_refused(self):
        contract, _ = self.fixture()
        for change in (lambda value: value.update(final_master_sha256='f' * 64),
                       lambda value: value['full_state_inventory'].pop()):
            changed = copy.deepcopy(contract); change(changed)
            with self.assertRaises(ValueError): d.verify_rebuild(changed, self.readers(), self.binding, checkpoint_binding=self.binding)
    def test_reader_budget_denies_before_any_source_bytes_and_writer_native_entry_closed(self):
        contract, _ = self.fixture(); readers = self.readers(); readers['delta'].remaining['reducer_work_bytes_upper'] = 0
        with self.assertRaisesRegex(ValueError, 'budget exhausted'): d.verify_rebuild(contract, readers, self.binding, checkpoint_binding=self.binding)
        class Trap:
            def __getattribute__(self, name): raise AssertionError('unadmitted model/framework')
        with self.assertRaises(ValueError): writer.retain_full_state(Trap())


if __name__ == '__main__': unittest.main()
