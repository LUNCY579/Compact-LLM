"""Stdlib metadata checks plus resource-gated tiny actual CPU checkpoint tests.

Set COMP_RS_RUN_NATIVE_CPU_TESTS=1 only after root resource clearance. Importing
or collecting this file does not import Torch/NumPy or create a model.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from experiments.skill_acquisition_native_v1 import checkpoint as cp


def progress(C, U, updated, generation=0):
    history = [{'boundary': index, 'row_ids': ['r%d' % index, 's%d' % index], 'updated': flag}
               for index, flag in enumerate(updated)]
    return {'C': C, 'U': U, 'history': history, 'generation': generation,
            'snapshot': cp.digest({'C': C, 'U': U, 'history': history, 'generation': generation})}


class MetadataTests(unittest.TestCase):
    def test_U_counts_updated_history_independently_of_C(self):
        rows = [['r0', 's0'], ['r1', 's1']]
        value = progress(2, 1, [True, False])
        self.assertIs(cp.validate_progress(value, rows), value)
        for change in ({'U': 2}, {'C': True}, {'generation': False}, {'snapshot': 'not-a-hash'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cp.validate_progress(dict(value, **change), rows)
        changed = copy.deepcopy(value)
        changed['history'][0]['row_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'ordered plan'):
            cp.validate_progress(changed, rows)

    def test_exact_plan_export_and_optional_bound_history_fields(self):
        rows = cp._plan_rows({'plan': {'boundaries': [
            {'logical_boundary': 0, 'prompts': [{'row_id': 'r0'}, {'row_id': 's0'}]}]}})
        self.assertEqual(rows, [['r0', 's0']])
        value = progress(1, 0, [False])
        value['history'][0]['operation_id'] = 'bound-attempt'
        value['policy_id'] = 'a' * 64
        cp.validate_progress(value, rows)

    def test_parent_start_admission_precedes_child_journal_io(self):
        adapter = cp.NativeCheckpoints.__new__(cp.NativeCheckpoints)
        observed = []
        adapter.ledger = lambda kind, phase, **fields: observed.append(('parent', kind, phase, fields))
        adapter._journal = lambda kind, **fields: (observed.append(('child', kind, fields)) or {'sha256': 'a' * 64})
        adapter._emit('checkpoint_restore', 'started', 'restore-1', read_bytes_upper=7, write_bytes_upper=3)
        self.assertEqual([item[0] for item in observed], ['parent', 'child'])
        self.assertEqual(observed[0][3]['write_bytes_upper'], 3)
        adapter._emit('checkpoint_restore', 'returned', 'restore-1', manifest_sha256='b' * 64)
        self.assertEqual([item[0] for item in observed[2:]], ['child', 'parent'])
        self.assertEqual(observed[-1][3]['checkpoint_event_sha256'], 'a' * 64)

    def test_rejected_parent_start_has_no_child_write(self):
        adapter = cp.NativeCheckpoints.__new__(cp.NativeCheckpoints)
        observed = []
        def reject(kind, phase, **fields):
            observed.append('parent')
            raise ValueError('cumulative hash/IO ceiling')
        adapter.ledger = reject
        adapter._journal = lambda kind, **fields: observed.append('child')
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            adapter._emit('checkpoint_save', 'started', 'denied')
        self.assertEqual(observed, ['parent'])

    @staticmethod
    def hash_adapter():
        adapter = cp.NativeCheckpoints.__new__(cp.NativeCheckpoints)
        events = []
        adapter.ledger = lambda kind, phase, **fields: events.append(dict(fields, kind=kind, phase=phase))
        adapter._journal = lambda kind, **fields: {'sha256': cp.digest(dict(fields, kind=kind))}
        return adapter, events

    def test_streamed_three_pass_hash_span_matches_actual_stdlib_payload(self):
        adapter, events = self.hash_adapter()
        specifications = {'param::weight': ['F32', [2]], 'rng::cpu': ['U8', [3]]}
        chunks = {'param::weight': b'\0' * 8, 'rng::cpu': b'abc'}
        data_bytes, file_bytes = cp._payload_sizes(specifications)
        with tempfile.TemporaryDirectory(prefix='comp-rs-hash-schema-') as directory:
            with adapter._hash_span(2 * data_bytes + file_bytes, 'checkpoint_streamed_publication', attempts=3):
                saved = cp.streams.stream_tensors(Path(directory) / 'tiny.safetensors', specifications,
                                                lambda name: iter([memoryview(chunks[name])]))
            self.assertEqual(saved['bytes'], file_bytes)
        start, returned = events
        self.assertEqual(start['kind'], 'fingerprint')
        self.assertEqual((start['identity_hash_attempts_upper'], start['identity_hash_bytes_upper']), (2, 0))
        self.assertEqual(returned['identity_hash_attempts_actual'], 3)
        self.assertEqual(returned['identity_hash_bytes_actual'], 2 * data_bytes + file_bytes)
        self.assertEqual(start['operation_id'], returned['operation_id'])

    def test_partial_hash_pass_keeps_admitted_cost_explicitly_uncertain(self):
        adapter, events = self.hash_adapter()
        with self.assertRaisesRegex(OSError, 'partial'):
            with adapter._hash_span(113, 'checkpoint_payload_file'):
                raise OSError('partial fixture read')
        self.assertEqual([event['phase'] for event in events], ['started', 'failed'])
        self.assertEqual(events[0]['state_bytes'], 113)
        self.assertNotIn('identity_hash_bytes_actual', events[1])
        self.assertNotIn('identity_hash_attempts_actual', events[1])
        self.assertFalse(events[1]['identity_hash_actual_known'])
        self.assertIn('remains charged', events[1]['uncertainty_rule'])

    def test_constructor_and_metadata_hash_minima_are_nonzero_without_io(self):
        bindings = {'inventory': [{'name': 'weight', 'shape': [2], 'dtype': 'F32'}], 'data': 'fixed'}
        lower = cp.constructor_hash_lower_bound(bindings)
        self.assertEqual(lower, {'identity_hash_attempts_lower': 1,
                                 'identity_hash_bytes_lower': len(cp.canonical(bindings))})
        adapter = cp.NativeCheckpoints.__new__(cp.NativeCheckpoints)
        adapter.bindings = bindings
        for kind in ('save', 'restore'):
            lower = adapter._metadata_hash_lower_bound(kind, progress(0, 0, []) if kind == 'save' else None)
            self.assertEqual(lower['identity_hash_attempts_lower'], 4)
            self.assertGreater(lower['identity_hash_bytes_lower'], len(cp.canonical(bindings)))


@unittest.skipUnless(os.environ.get('COMP_RS_RUN_NATIVE_CPU_TESTS') == '1',
                     'tiny Torch CPU tests await root resource clearance')
class NativeCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, np
        import torch
        import numpy as np
        cls.torch = torch
        if torch.cuda.is_initialized():
            raise AssertionError('CPU checkpoint tests must not initialize CUDA')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='comp-rs-native-checkpoint-')
        self.root = Path(self.tmp.name)
        self.gates = []

    def tearDown(self):
        self.tmp.cleanup()
        self.assertFalse(torch.cuda.is_initialized())

    @staticmethod
    def seed():
        random.seed(1701); np.random.seed(1702); torch.manual_seed(1703)

    @staticmethod
    def model_optimizer():
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.25), torch.nn.Linear(5, 2))
        model.register_buffer('frozen_buffer', torch.tensor([3., 7.]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999),
                                     eps=1e-8, weight_decay=0, foreach=False)
        return model, optimizer

    @staticmethod
    def bindings(model, boundaries=6):
        inventory = [{'name': name, 'shape': list(value.shape), 'dtype': 'float32'}
                     for name, value in model.named_parameters() if value.requires_grad]
        return {'ordered_plan': [{'row_ids': ['r%d' % index, 's%d' % index]} for index in range(boundaries)],
                'inventory': inventory, 'code': {'checkpoint': 'fixture-v1'}, 'data': 'fixture-TRAIN',
                'runtime': 'tiny-cpu-fp32', 'recipe': {'name': 'fixture-AdamW'},
                'own_start_sha256': 'a' * 64, 'common_anchor_sha256': 'b' * 64}

    def adapter(self, directory, model, ledger, bindings=None):
        return cp.NativeCheckpoints(directory, bindings or self.bindings(model),
                                    sum(value.numel() for value in model.parameters()),
                                    self.gates.append).attach_ledger(ledger)

    class Ledger:
        def __init__(self):
            self.events, self.generation = [], 0
        def operation(self, kind, phase, **fields):
            self.events.append(dict(fields, kind=kind, phase=phase))
        def snapshot(self):
            return {'generation': self.generation, 'physical_events': len(self.events),
                    'head': cp.digest(self.events)}

    @staticmethod
    def boundary(model, optimizer, updated):
        # Uniform exposures consume RNG but do not initialize or advance Adam.
        values = torch.randn(4, 3) + random.random() + float(np.random.random())
        if updated:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            model(values).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model.eval()

    @staticmethod
    def snapshot(model, optimizer):
        tensors = {name: value.detach().clone() for name, value in model.named_parameters()}
        moments = [{name: value.detach().clone() for name, value in optimizer.state.get(parameter, {}).items()}
                   for parameter in model.parameters()]
        rng, rng_tensors = cp.streams.rng_state()
        modes = {name: module.training for name, module in model.named_modules()}
        return tensors, moments, json.loads(cp.canonical(rng)), rng_tensors, modes

    def assert_snapshot_equal(self, left, right):
        self.assertEqual(left[2], right[2]); self.assertEqual(left[4], right[4])
        self.assertTrue(all(torch.equal(left[0][key], right[0][key]) for key in left[0]))
        self.assertTrue(all(set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)
                            for a, b in zip(left[1], right[1])))
        self.assertTrue(all(torch.equal(left[3][key], right[3][key]) for key in left[3]))

    def test_empty_adam_and_uniform_C_greater_than_U_roundtrip(self):
        self.seed(); model, optimizer = self.model_optimizer(); ledger = self.Ledger()
        adapter = self.adapter(self.root / 'empty', model, ledger)
        first = adapter.save(model, optimizer, progress(0, 0, []), ledger)
        self.assertTrue(all(not value for value in first['Adam_initialized'].values()))
        self.boundary(model, optimizer, False)
        second = adapter.save(model, optimizer, progress(1, 0, [False]), ledger)
        self.assertEqual(second['Adam_scalar_steps'], 0)
        saved = self.snapshot(model, optimizer)
        self.boundary(model, optimizer, True)
        result = adapter.restore(model, optimizer)
        self.assertEqual((result['C'], result['U']), (1, 0))
        self.assertFalse(optimizer.state)
        self.assert_snapshot_equal(saved, self.snapshot(model, optimizer))
        starts = {event['operation_id']: event for event in ledger.events if event['phase'] == 'started'}
        for event in ledger.events:
            if event['kind'] == 'fingerprint' and event['phase'] == 'returned':
                reserved = starts[event['operation_id']]
                self.assertEqual(event['identity_hash_attempts_actual'], 1 + reserved['identity_hash_attempts_upper'])
                self.assertEqual(event['identity_hash_bytes_actual'], reserved['state_bytes'])
        restore_start = next(event for event in ledger.events
                             if event['kind'] == 'checkpoint_restore' and event['phase'] == 'started')
        self.assertGreater(restore_start['write_bytes_upper'], 0)

    def test_returned_restore_callback_is_a_complete_consumable_manifest_proof(self):
        self.seed(); model, optimizer = self.model_optimizer(); ledger = self.Ledger()
        adapter = self.adapter(self.root / 'proof', model, ledger)
        manifest = adapter.save(model, optimizer, progress(0, 0, []), ledger)
        restored = adapter.restore(model, optimizer)
        proofs = [event for event in ledger.events
                  if event['kind'] == 'checkpoint_restore' and event['phase'] == 'returned']
        self.assertEqual(len(proofs), 1)
        proof = proofs[0]
        # A journal consumer needs only the durable returned callback and public
        # restore record, never unrecorded state inside the checkpoint adapter.
        def consume(record, event):
            cp.require(event['manifest_sha256'] == record['manifest_sha256']
                       and all(event[source] == record['progress'][target] for source, target in
                               (('C', 'C'), ('U', 'U'), ('generation', 'generation'),
                                ('symbolic_snapshot', 'snapshot'))), 'returned restore proof mismatch')
            return event['operation_id']
        self.assertEqual(proof['manifest_sha256'], manifest['manifest_sha256'])
        self.assertEqual(consume(restored, proof), proof['operation_id'])
        with self.assertRaisesRegex(ValueError, 'proof mismatch'):
            consume(dict(restored, manifest_sha256='f' * 64), proof)

    def test_actual_params_Adam_rng_bitwise_interrupted_vs_uninterrupted(self):
        updated = [True, False, True, True, False, True]
        self.seed(); model, optimizer = self.model_optimizer()
        for flag in updated: self.boundary(model, optimizer, flag)
        expected = self.snapshot(model, optimizer)
        self.seed(); model, optimizer = self.model_optimizer(); ledger = self.Ledger()
        adapter = self.adapter(self.root / 'split', model, ledger)
        adapter.save(model, optimizer, progress(0, 0, []), ledger)
        for index in range(2):
            self.boundary(model, optimizer, updated[index])
            adapter.save(model, optimizer, progress(index + 1, sum(updated[:index + 1]), updated[:index + 1]), ledger)
        self.assertEqual(len(adapter._latest()), 2)
        # Returned later update is discarded; its root-ledger evidence remains.
        self.boundary(model, optimizer, True)
        ledger.operation('optimizer_step', 'returned', operation_id='discarded', generation=0)
        ledger.generation = 5
        restarted_model, restarted_optimizer = self.model_optimizer()
        reopened = self.adapter(self.root / 'split', restarted_model, ledger)
        restored = reopened.restore(restarted_model, restarted_optimizer)
        self.assertEqual((restored['C'], restored['U']), (2, 1))
        self.assertEqual(restored['generation'], 6)
        self.assertNotEqual(restored['snapshot'], restored['restored_snapshot'])
        self.assertTrue(any(event.get('operation_id') == 'discarded' for event in ledger.events))
        for index in range(restored['C'], len(updated)):
            self.boundary(restarted_model, restarted_optimizer, updated[index])
        self.assert_snapshot_equal(expected, self.snapshot(restarted_model, restarted_optimizer))
        self.assertGreater(restored['cumulative_storage_costs']['counts']['checkpoint_save_returned'], 0)

    def test_payload_corruption_fails_latest_without_mutating_live_state(self):
        self.seed(); model, optimizer = self.model_optimizer(); ledger = self.Ledger()
        adapter = self.adapter(self.root / 'corrupt', model, ledger)
        adapter.save(model, optimizer, progress(0, 0, []), ledger)
        self.boundary(model, optimizer, True)
        adapter.save(model, optimizer, progress(1, 1, [True]), ledger)
        latest = adapter._latest()[-1].with_suffix('.safetensors')
        with latest.open('r+b') as stream:
            stream.seek(-1, 2); value = stream.read(1); stream.seek(-1, 2); stream.write(bytes([value[0] ^ 1]))
        live_before = self.snapshot(model, optimizer)
        with self.assertRaisesRegex(ValueError, 'payload bytes corrupt'):
            adapter.restore(model, optimizer)
        self.assert_snapshot_equal(live_before, self.snapshot(model, optimizer))
        self.assertEqual(len(adapter._latest()), 2)

    def test_recomputed_metadata_and_wrong_Adam_U_rejected(self):
        self.seed(); model, optimizer = self.model_optimizer(); ledger = self.Ledger()
        adapter = self.adapter(self.root / 'wrong', model, ledger)
        self.boundary(model, optimizer, True)
        with self.assertRaisesRegex(ValueError, 'equal U, not C'):
            adapter.save(model, optimizer, progress(2, 2, [True, True]), ledger)
        self.assertFalse(adapter._latest())
        good = adapter.save(model, optimizer, progress(2, 1, [True, False]), ledger)
        path = adapter._latest()[-1]
        changed = copy.deepcopy(good)
        changed['progress']['history'][0]['row_ids'].reverse()
        changed['manifest_sha256'] = cp.digest({key: value for key, value in changed.items() if key != 'manifest_sha256'})
        path.write_bytes(cp.canonical(changed))  # malicious fixture; adapter never rewrites checkpoints
        before = self.snapshot(model, optimizer)
        with self.assertRaisesRegex(ValueError, 'publication intent'):
            adapter.restore(model, optimizer)
        self.assert_snapshot_equal(before, self.snapshot(model, optimizer))

    def test_payload_reads_writes_are_bounded_and_failed_save_keeps_cost(self):
        self.seed(); model, optimizer = self.model_optimizer(); ledger = self.Ledger()
        adapter = self.adapter(self.root / 'failure', model, ledger)
        adapter.save(model, optimizer, progress(0, 0, []), ledger)
        original = cp.streams.stream_tensors
        def failure(*args, **kwargs): raise OSError('fixture interrupted payload write')
        cp.streams.stream_tensors = failure
        try:
            with self.assertRaisesRegex(OSError, 'interrupted payload'):
                adapter.save(model, optimizer, progress(1, 0, [False]), ledger)
        finally:
            cp.streams.stream_tensors = original
        reopened = self.adapter(self.root / 'failure', model, ledger)
        report = reopened.cost_snapshot()
        self.assertEqual(report['counts']['checkpoint_save_started'], 2)
        self.assertEqual(report['counts']['checkpoint_save_failed'], 1)
        self.assertEqual(report['counts']['checkpoint_save_returned'], 1)
        restored = reopened.restore(model, optimizer)
        self.assertEqual((restored['C'], restored['U']), (0, 0))
        self.assertIn('write_bytes_upper', next(event for event in reopened.events
                                               if event.get('phase') == 'failed'))
        self.assertEqual(cp.CHUNK, 1024 ** 2)

    def test_actual_large_tensor_io_calls_never_exceed_one_MiB(self):
        self.seed()
        model = torch.nn.Module()
        model.register_parameter('weight', torch.nn.Parameter(torch.zeros(300000, dtype=torch.float32)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, foreach=False)
        ledger = self.Ledger()
        adapter = self.adapter(self.root / 'bounded', model, ledger)
        original_open = Path.open
        sizes = []
        class ObservedFile:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): self.stream.__enter__(); return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def __getattr__(self, name): return getattr(self.stream, name)
            def read(self, size=-1):
                self_case.assertTrue(0 <= size <= cp.CHUNK, 'unbounded checkpoint read')
                sizes.append(('read', size))
                return self.stream.read(size)
            def write(self, block):
                self_case.assertLessEqual(len(block), cp.CHUNK, 'unbounded checkpoint write')
                sizes.append(('write', len(block)))
                return self.stream.write(block)
        self_case = self
        def observed(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return ObservedFile(stream) if Path(path).resolve().parent == adapter.directory else stream
        with mock.patch.object(Path, 'open', observed):
            adapter.save(model, optimizer, progress(0, 0, []), ledger)
            with torch.no_grad(): model.weight.add_(1)
            adapter.restore(model, optimizer)
        self.assertTrue(torch.equal(model.weight, torch.zeros_like(model.weight)))
        self.assertIn(('write', cp.CHUNK), sizes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
