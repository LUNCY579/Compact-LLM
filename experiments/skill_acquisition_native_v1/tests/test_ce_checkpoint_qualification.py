"""Pure branch/fence/actual byte-file contracts; no native model or process."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_checkpoint_qualification as c
from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as r


class Contracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.serial = 0
        self.context = {'phase': 'qualification', 'bindings': {'scope': {'qualifier_execution': {
            'adam_batches': [['r%d' % i for i in range(8)], ['r%d' % i for i in range(8, 16)]]}}}}
        self.left = {'stage': 'checkpoint_resume', 'request_sha256': '1' * 64,
            'scope_sha256': '2' * 64, 'static_recipe_sha256': '3' * 64,
            'code_sha256': {'worker.py': '4' * 64}, 'worker_identity_sha256': '5' * 64, 'attempt_id': 'uninterrupted'}
        self.right = dict(self.left, worker_identity_sha256='6' * 64, attempt_id='resumed')
        self.inventory = [{'name': 'w', 'shape': [2], 'dtype': 'F32'}]

    def reader(self):
        return r.RawEvidenceReader(lambda *args, **kwargs: None,
            {'operation_attempts_upper': 1000, 'io_read_bytes_upper': 10 ** 7,
             'identity_hash_attempts_upper': 1000, 'identity_hash_bytes_upper': 10 ** 8,
             'reducer_work_bytes_upper': 5 * 10 ** 8}, self.root, chunk_bytes=64)

    def state(self, binding, change=None):
        self.serial += 1; stem = self.root / ('branch-%d' % self.serial)
        tensors = {'param::w': ('F32', [2], struct.pack('<2f', 3, 4)),
            'exp_avg::w': ('F32', [2], struct.pack('<2f', .5, .25)),
            'exp_avg_sq::w': ('F32', [2], struct.pack('<2f', .2, .1)),
            'step::w': ('F32', [], struct.pack('<f', 2)),
            'rng::cpu': ('U8', [3], b'abc'), 'rng::cuda::0': ('U8', [4], b'defg')}
        if change: change(tensors)
        header, body, offset = {}, b'', 0
        for name, (dtype, shape, value) in sorted(tensors.items()):
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(value)]}
            body += value; offset += len(value)
        h = r.canonical(header); payload = struct.pack('<Q', len(h)) + h + body
        payload_path = stem.with_suffix('.safetensors'); payload_path.write_bytes(payload)
        manifest = {'schema': c.CHECKPOINT_SCHEMA,
            'bindings': {'qualification_evidence_binding': binding}, 'progress': {'C': 2, 'U': 2},
            'specifications': {name: [dtype, shape] for name, (dtype, shape, _) in tensors.items()},
            'module_training': {'model': False}, 'optimizer_groups': [{'lr': 1e-5}],
            'Adam_initialized': ['w'], 'Adam_scalar_steps': 2,
            'rng': {'python': [3, [1, 2], None], 'numpy': ['MT19937', [3, 4], 0, 0, 0.0],
                    'numpy_version': 'fixture', 'cuda_count': 1}, 'frozen_state_sha256': 'f' * 64,
            'payload': {'path': str(payload_path), 'bytes': len(payload),
                        'file_sha256': hashlib.sha256(payload).hexdigest()}}
        manifest['manifest_sha256'] = r.digest(manifest)
        raw = r.canonical(manifest) + b'\n'; path = stem.with_suffix('.json'); path.write_bytes(raw)
        return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                'role': 'checkpoint_manifest', 'binding_sha256': r.digest(binding)}

    def compare(self, left=None, right=None):
        left = self.state(self.left) if left is None else left
        right = self.state(self.right) if right is None else right
        return c.compare_checkpoint_states(self.reader(), left, self.left, right, self.right, parameter_inventory=self.inventory)

    def test_branch_plan_is_independent_two_update_qualification(self):
        plans = c.branch_plan(self.context)
        self.assertEqual([p['updates'] for p in plans], [2, 1, 1])
        self.assertEqual([p['final_U'] for p in plans], [2, 1, 2])
        self.assertEqual(plans[2]['restore_from'], 'prefix')
        self.assertEqual(plans[2]['row_batches'], self.context['bindings']['scope']['qualifier_execution']['adam_batches'][1:])

    def test_production_or_wrong_batch_scope_cannot_use_adapter(self):
        for change in (lambda x: x.update(phase='training'),
                       lambda x: x['bindings']['scope']['qualifier_execution'].update(adam_batches=[['r'] * 8] * 2),
                       lambda x: x['bindings']['scope']['qualifier_execution'].update(adam_batches=[['r'] * 8] * 375)):
            ctx = copy.deepcopy(self.context); change(ctx)
            with self.assertRaises(ValueError): c.branch_plan(ctx)

    def test_native_entries_refuse_before_backend_or_context_access(self):
        class Trap:
            def __getattribute__(self, name): raise AssertionError('unadmitted access')
        for work in (lambda: c.release_parent_model(Trap(), None, Trap()),
                     lambda: c.reload_parent_model(Trap(), None, Trap()),
                     lambda: c.checkpoint_adapter(Trap(), None, Trap(), Trap(), Trap(), Trap(), Trap())):
            with self.assertRaises(ValueError): work()
        self.assertFalse(c._REGISTERED_GPU_HANDOFF_SERVICES)
        self.assertFalse(c._REGISTERED_CHECKPOINT_PROCESS_SERVICES)

    def test_actual_payload_bytes_and_all_rng_metadata_equal_remain_unqualified(self):
        result = self.compare()
        self.assertTrue(all(result[k] for k in ('parameters_bitwise', 'adam_bitwise', 'module_modes_bitwise',
            'python_rng_bitwise', 'numpy_rng_bitwise', 'torch_cpu_rng_bitwise', 'torch_cuda_rng_bitwise')))
        self.assertFalse(result['execution_verified']); self.assertFalse(result['native_qualified'])

    def test_distinct_raw_child_request_compares_under_exact_study_lineage(self):
        child = dict(self.right)
        request_ref = {'path': str(self.root / 'raw-child-request.json'), 'sha256': '7' * 64, 'bytes': 100}
        lineage = {'schema': 'CE8-CHECKPOINT-CHILD-PREDECESSOR-1',
            'worker_identity_sha256': child['worker_identity_sha256'], 'study_request_sha256': child['request_sha256'],
            'child_request_sha256': request_ref['sha256'], 'child_request_ref': request_ref,
            'child_proof': {**{key: child[key] for key in ('worker_identity_sha256', 'request_sha256', 'scope_sha256')},
                'sources': {'request_ref': request_ref}},
            'parent_request_sha256': self.left['request_sha256'], 'parent_worker_identity_sha256': self.left['worker_identity_sha256'],
            **{key: self.left[key] for key in ('scope_sha256', 'static_recipe_sha256', 'code_sha256')},
            'parent_proof': {key: self.left[key] for key in ('worker_identity_sha256', 'request_sha256', 'scope_sha256')}}
        left, right = self.state(self.left), self.state(child)
        result = c.compare_checkpoint_states(self.reader(), left, self.left, right, child, parameter_inventory=self.inventory,
            parent_binding=self.left, child_lineages={child['worker_identity_sha256']: lineage})
        self.assertTrue(result['parameters_bitwise']); self.assertFalse(result['execution_verified'])
        changed = copy.deepcopy(lineage); changed['parent_request_sha256'] = '8' * 64
        with self.assertRaisesRegex(ValueError, 'exact observed child'):
            c.compare_checkpoint_states(self.reader(), left, self.left, right, child, parameter_inventory=self.inventory,
                parent_binding=self.left, child_lineages={child['worker_identity_sha256']: changed})
        changed = copy.deepcopy(lineage); changed['child_request_sha256'] = child['request_sha256']
        with self.assertRaisesRegex(ValueError, 'raw child request hash'):
            c.compare_checkpoint_states(self.reader(), left, self.left, right, child, parameter_inventory=self.inventory,
                parent_binding=self.left, child_lineages={child['worker_identity_sha256']: changed})

    def test_changed_parameter_or_cuda_rng_bytes_fail_despite_valid_hashes(self):
        for name in ('param::w', 'rng::cuda::0'):
            def mutate(tensors, name=name):
                dtype, shape, data = tensors[name]; tensors[name] = (dtype, shape, bytes([data[0] ^ 1]) + data[1:])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'bytes differ'):
                self.compare(right=self.state(self.right, mutate))

    def test_missing_adam_state_or_truncated_payload_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, 'keys/shapes'):
            self.compare(right=self.state(self.right, lambda tensors: tensors.pop('step::w')))
        ref = self.state(self.right); manifest = r._json(Path(ref['path']).read_bytes())
        path = Path(manifest['payload']['path']); path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'path/size/ownership'):
            self.compare(right=ref)

    def test_same_worker_or_foreign_source_cannot_claim_fresh_process_comparison(self):
        left, right = self.state(self.left), self.state(self.right)
        wrong = dict(self.right, worker_identity_sha256=self.left['worker_identity_sha256'])
        with self.assertRaisesRegex(ValueError, 'fresh checkpoint workers'):
            c.compare_checkpoint_states(self.reader(), left, self.left, right, wrong, parameter_inventory=self.inventory)
        wrong = dict(self.right, code_sha256={'foreign.py': '9' * 64})
        with self.assertRaisesRegex(ValueError, 'study/source/recipe'):
            c.compare_checkpoint_states(self.reader(), left, self.left, right, wrong, parameter_inventory=self.inventory)

    def test_oversized_manifest_rejected_before_read_or_hash_reservation(self):
        events = []
        reader = self.reader(); reader.operation = lambda *args, **kwargs: events.append((args, kwargs))
        ref = self.state(self.left); ref['bytes'] = r.MAX_JSON + 1
        with self.assertRaisesRegex(ValueError, 'bounded checkpoint JSON'):
            c._read_manifest(reader, ref, self.left)
        self.assertEqual(events, [])

    def test_child_failure_uses_owned_stop_terminal_then_close_and_keeps_primary(self):
        class FakeOwned:
            def __init__(self): self.calls = []
            def _stop_owned(self, reason, primary=None): self.calls.append(('stop', primary))
            def terminal(self, seconds): self.calls.append(('terminal', seconds)); return {'fixture': True}
            def close(self): self.calls.append(('close', None)); raise RuntimeError('close failed')
        owned, primary = FakeOwned(), ValueError('original RPC failure')
        with self.assertRaisesRegex(ValueError, 'original RPC failure') as caught:
            c._close_child(owned, 3, primary)
        self.assertIs(caught.exception, primary)
        self.assertEqual([name for name, _ in owned.calls], ['stop', 'terminal', 'close'])
        self.assertTrue(primary.checkpoint_child_terminal_verified)
        self.assertIn('close failed', repr(primary.checkpoint_child_cleanup_failure))

    def test_live_child_terminal_failure_is_not_quiescence(self):
        class FakeOwned:
            def _stop_owned(self, reason, primary=None): raise RuntimeError('owned stop failed')
            def terminal(self, seconds): raise RuntimeError('no terminal proof')
            def close(self): raise RuntimeError('still live')
        primary = ValueError('original verification failure')
        with self.assertRaisesRegex(ValueError, 'original verification failure') as caught:
            c._close_child(FakeOwned(), 3, primary)
        self.assertFalse(caught.exception.checkpoint_child_terminal_verified)
        self.assertEqual(len(caught.exception.checkpoint_child_cleanup_failure), 3)

    def test_restore_fence_exact_ancestry_operation_and_generation(self):
        before = {'C': 2, 'U': 2, 'generation': 0, 'history': [{'i': 0}, {'i': 1}]}
        p = {'C': 2, 'U': 2, 'generation': 1, 'history': before['history'], 'snapshot': 'a' * 64, 'symbolic_snapshot': 'b' * 64}
        record = dict(p, schema=c.RESTORE_SCHEMA, source_worker_binding=self.left,
            progress=p, manifest_sha256='c' * 64, install_fence_required=True)
        operation = {'phase': 'returned', 'kind': 'checkpoint_restore', 'result': record,
            'worker_id': self.left['worker_identity_sha256'], 'attempt_id': self.left['attempt_id'],
            'stage': self.left['stage'], 'source_worker_binding': self.left}
        self.assertEqual(c.validate_restore_fence(record, before, operation, current_binding=self.left), p)
        stale = copy.deepcopy(record); stale['generation'] = stale['progress']['generation'] = 0
        with self.assertRaisesRegex(ValueError, 'new restore generation'):
            c.validate_restore_fence(stale, before, operation, current_binding=self.left)
        wrong = copy.deepcopy(operation); wrong['result']['manifest_sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'returned restore operation'):
            c.validate_restore_fence(record, before, wrong, current_binding=self.left)
        for key in ('C', 'U', 'generation'):
            wrong_before = copy.deepcopy(before); wrong_before[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'strict before'):
                c.validate_restore_fence(record, wrong_before, operation, current_binding=self.left)
        bad = copy.deepcopy(record); bad['U'] = bad['progress']['U'] = True
        with self.assertRaisesRegex(ValueError, 'C/U ancestor'):
            c.validate_restore_fence(bad, before, operation, current_binding=self.left)
        foreign = copy.deepcopy(operation); foreign['worker_id'] = self.right['worker_identity_sha256']
        with self.assertRaisesRegex(ValueError, 'returned restore operation'):
            c.validate_restore_fence(record, before, foreign, current_binding=self.left)


if __name__ == '__main__': unittest.main()
