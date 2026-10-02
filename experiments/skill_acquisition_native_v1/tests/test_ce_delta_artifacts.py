"""Actual tiny saved delta bytes/finite/hash controls; no ML framework."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_delta_artifacts as d


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.binding = {'stage': 'delta_roundtrip', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
            'static_recipe_sha256': 'c' * 64, 'code_sha256': {'fixture': 'd' * 64},
            'worker_identity_sha256': 'e' * 64, 'attempt_id': 'delta'}
        self.inventory = [{'name': 'w', 'shape': [2], 'dtype': 'F32'}, {'name': 'z', 'shape': [1], 'dtype': 'F32'}]
    def save(self, values=(3, 4, 0), table=None):
        header = {'w': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]},
                  'z': {'dtype': 'F32', 'shape': [1], 'data_offsets': [8, 12]}}
        if table: table(header)
        encoded = d.raw.canonical(header); body = struct.pack('<3f', *values)
        blob = struct.pack('<Q', len(encoded)) + encoded + body; path = self.root / 'delta.safetensors'; path.write_bytes(blob)
        return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob),
                'role': 'qualification_delta', 'binding_sha256': d.raw.digest(self.binding)}, body
    def read(self, reference, inventory=None):
        reader = d.raw.RawEvidenceReader(lambda *args, **fields: None,
            {key: 10 ** 9 for key in d.raw.BUDGET_FIELDS}, self.root, chunk_bytes=4)
        return d.read_delta(reader, reference, self.binding, self.inventory if inventory is None else inventory)
    def test_small_stream_chunks_verify_payload_state_digest_and_norm(self):
        reference, body = self.save(); result = self.read(reference)
        self.assertEqual(result['data_section_sha256'], hashlib.sha256(body).hexdigest())
        self.assertEqual(result['delta_norm'], 5); self.assertEqual(result['numel'], 3)
        state = hashlib.sha256()
        for name, shape, payload in (('w', [2], body[:8]), ('z', [1], body[8:])):
            tensor = hashlib.sha256(d.raw.canonical({'dtype': 'torch.float32', 'shape': shape}) + b'\n' + payload).hexdigest()
            state.update(name.encode() + b'\0' + tensor.encode() + b'\n')
        self.assertEqual(result['state_sha256'], state.hexdigest()); self.assertFalse(result['execution_verified'])
    def test_nonfinite_dtype_shape_or_offset_cannot_pass_valid_file_hash(self):
        for values, mutate in (((float('nan'), 4, 0), None), ((3, 4, 0), lambda h: h['w'].update(dtype='BF16')),
                ((3, 4, 0), lambda h: h['w'].update(shape=[1, 2])), ((3, 4, 0), lambda h: h['z'].update(data_offsets=[9, 13]))):
            reference, _ = self.save(values, mutate)
            with self.assertRaises(ValueError): self.read(reference)
    def test_missing_inventory_or_changed_raw_bytes_are_rejected(self):
        reference, _ = self.save()
        with self.assertRaises(ValueError): self.read(reference, self.inventory[:1])
        path = Path(reference['path']); changed = bytearray(path.read_bytes()); changed[-1] ^= 1; path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, 'raw artifact bytes/hash mismatch'):
            self.read(reference)


if __name__ == '__main__': unittest.main()
