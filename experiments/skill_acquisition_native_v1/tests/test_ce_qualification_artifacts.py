"""Actual temporary-file readback fixtures; no framework or native model."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as a


class RawArtifactControls(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root, self.events, self.serial = Path(temp.name), [], 0
        self.expected = {'stage': 'longest_masks', 'request_sha256': '1' * 64,
            'scope_sha256': '2' * 64, 'static_recipe_sha256': '3' * 64,
            'code_sha256': {'worker.py': '4' * 64}, 'worker_identity_sha256': '5' * 64,
            'attempt_id': 'actual-fixture-stage-attempt'}
        self.rows, self.bindings = [], []
        for i in range(90):
            rid, prompt, completion = 'row-%03d' % i, [10, 11], [20, 151645]
            prepared = {'id': rid, 'prompt_ids': prompt, 'completion_ids': completion}
            self.bindings.append({'id': rid, 'prompt_tokens': 2, 'labeled_tokens': 2,
                'full_sequence_tokens': 4, 'prompt_ids_sha256': a.digest(prompt),
                'completion_ids_sha256': a.digest(completion), 'prepared_row_sha256': a.digest(prepared)})
            L = i % 2
            self.rows.append(dict(prepared, padding_tokens=L, padding_side='left' if L else 'none',
                input_ids=[151643] * L + prompt + completion,
                attention_mask=[0] * L + [1] * 4, labels=[-100] * (L + 2) + completion,
                predictor_positions=[L + 1, L + 2], target_positions=[L + 2, L + 3]))

    def reader(self, **overrides):
        bounds = dict(operation_attempts_upper=1000, io_read_bytes_upper=10 ** 7,
                      identity_hash_attempts_upper=1000, identity_hash_bytes_upper=10 ** 8,
                      reducer_work_bytes_upper=10 ** 8)
        bounds.update(overrides)
        return a.RawEvidenceReader(lambda kind, phase, **fields: self.events.append((kind, phase, fields)),
                                   bounds, self.root, chunk_bytes=4)

    def artifact(self, content, role, expected=None):
        expected = self.expected if expected is None else expected
        self.serial += 1; path = self.root / ('artifact-%03d' % self.serial)
        raw = content if type(content) is bytes else a.canonical(content)
        path.write_bytes(raw)
        return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                'role': role, 'binding_sha256': a.digest(expected)}

    def envelope(self, kind, payload, expected=None):
        return {'schema': a.SCHEMA, **(self.expected if expected is None else expected),
                'evidence_kind': kind, 'payload': payload}

    def masks(self, rows=None, envelope=None):
        return self.artifact(envelope or self.envelope('masks', {'rows': self.rows if rows is None else rows}), 'masks')

    def likelihood(self, mutate=None):
        self.expected['stage'] = 'likelihood_gradient_parity'
        rows = [{'id': b['id'], 'prepared_row_sha256': b['prepared_row_sha256'],
            'completion_ids': [20, 151645], 'cached_logprobs': [-1.0, -2.0],
            'bulk_logprobs': [-1.125, -2.0]} for b in self.bindings]
        if mutate: mutate(rows)
        return self.artifact(self.envelope('likelihood', {'rows': rows}), 'likelihood')

    def gradient(self, reference=(3.0, 4.0, 0.0, 12.0), candidate=(3.0, 0.0, 0.0, 12.0), mutate=None):
        self.expected['stage'] = 'likelihood_gradient_parity'
        left = self.artifact(struct.pack('<%df' % len(reference), *reference), 'gradient_reference')
        right = self.artifact(struct.pack('<%df' % len(candidate), *candidate), 'gradient_candidate')
        inventory = [{'name': 'layer.a', 'shape': [2], 'dtype': 'F32'},
                     {'name': 'layer.b', 'shape': [2], 'dtype': 'F32'}]
        payload = {'fixed_loss_denominator': 8, 'loss_reduction': a.LOSS, 'dtype': 'F32',
            'start_anchor_sha256': '6' * 64, 'rows': self.bindings[:8],
            'tensor_inventory': [{'name': 'layer.a', 'shape': [2], 'offset_elements': 0, 'numel': 2},
                                 {'name': 'layer.b', 'shape': [2], 'offset_elements': 2, 'numel': 2}],
            'reference_ref': left, 'candidate_ref': right}
        if mutate: mutate(payload)
        manifest = self.artifact(self.envelope('gradient', payload), 'gradient_manifest')
        return manifest, inventory

    def test_all90_masks_recompute_left_and_unpadded_positions(self):
        result = self.reader().read_masks(self.masks(), self.expected, self.bindings)
        self.assertEqual(len(result['observations']['rows']), 90)
        self.assertEqual(result['observations']['rows'][1]['predictor_positions'], [1, 2])
        self.assertEqual(result['observations']['rows'][1]['raw_padding_tokens'], 1)
        self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])

    def test_attention_mask_requires_exact_integer_zero_one_values(self):
        for value in (True, 1.0):
            rows = copy.deepcopy(self.rows); rows[0]['attention_mask'][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(a.EvidenceRejected, 'integer raw'):
                self.reader().read_masks(self.masks(rows), self.expected, self.bindings)

    def test_witness_subset_and_duplicate_cannot_replace_all90(self):
        with self.assertRaisesRegex(a.EvidenceRejected, 'all 90'):
            self.reader().read_masks(self.masks(self.rows[:8]), self.expected, self.bindings[:8])
        rows = copy.deepcopy(self.rows); rows[-1] = rows[0]
        with self.assertRaisesRegex(a.EvidenceRejected, 'witness rows'):
            self.reader().read_masks(self.masks(rows), self.expected, self.bindings)

    def test_wrong_rendered_witness_content_rehash_is_rejected(self):
        rows = copy.deepcopy(self.rows); rows[0]['prompt_ids'][0] = 12
        with self.assertRaisesRegex(a.EvidenceRejected, 'actual witness'):
            self.reader().read_masks(self.masks(rows), self.expected, self.bindings)

    def test_EOS_shift_and_prompt_or_padding_labels_rejected(self):
        for key, value in [('predictor_positions', [2, 3]), ('labels', [-100, 11, 20, 151645]),
                           ('labels', [-100, -100, 20, -100])]:
            rows = copy.deepcopy(self.rows); rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(a.EvidenceRejected, 'one-shift'):
                self.reader().read_masks(self.masks(rows), self.expected, self.bindings)
        rows = copy.deepcopy(self.rows); rows[0]['completion_ids'] = [151645, 20]
        with self.assertRaisesRegex(a.EvidenceRejected, 'terminal target EOS'):
            self.reader().read_masks(self.masks(rows), self.expected, self.bindings)

    def test_actual_readback_recomputes_all_likelihood_errors(self):
        ref = self.likelihood(); result = self.reader().read_likelihood(ref, self.expected, self.bindings)
        self.assertEqual(len(result['observations']['per_token_comparisons']), 90)
        self.assertTrue(all(r['max_abs_nats'] == .125 for r in result['observations']['per_token_comparisons']))
        self.assertFalse(result['native_qualified'])

    def test_likelihood_count_target_and_nonfinite_rejected(self):
        changes = [lambda rows: rows[0].update(cached_logprobs=[-1.0]),
                   lambda rows: rows[0].update(completion_ids=[21, 151645])]
        for change in changes:
            ref = self.likelihood(change)
            with self.assertRaises(a.EvidenceRejected):
                self.reader().read_likelihood(ref, self.expected, self.bindings)
        ref = self.likelihood(); raw = Path(ref['path']).read_bytes().replace(b'-1.0', b'NaN', 1)
        ref = self.artifact(raw, 'likelihood')
        with self.assertRaisesRegex(a.EvidenceRejected, 'nonfinite'):
            self.reader().read_likelihood(ref, self.expected, self.bindings)

    def test_bounded_streamed_gradient_matches_independent_analytic_norm(self):
        ref, inventory = self.gradient(); result = self.reader().read_gradient(ref, self.expected,
            self.bindings[:8], inventory, start_anchor_sha256='6' * 64)
        observed = result['observations']['gradient_reference']
        self.assertEqual(observed['reference_norm'], 13.0)
        self.assertEqual(observed['difference_norm'], 4.0)
        self.assertEqual(observed['max_abs_error'], 4.0)
        self.assertEqual(result['observations']['numel'], 4)
        self.assertFalse(result['native_qualified'])

    def test_gradient_rejects_nonfinite_raw_values_wrong_count_layout_and_rows(self):
        ref, inventory = self.gradient(reference=(3.0, float('nan'), 0.0, 12.0))
        with self.assertRaisesRegex(a.EvidenceRejected, 'nonfinite raw gradient'):
            self.reader().read_gradient(ref, self.expected, self.bindings[:8], inventory, start_anchor_sha256='6' * 64)
        mutations = [lambda p: p['tensor_inventory'][1].update(offset_elements=1),
                     lambda p: p['reference_ref'].update(bytes=12),
                     lambda p: p.update(rows=list(reversed(p['rows']))),
                     lambda p: p.update(fixed_loss_denominator=16)]
        for mutate in mutations:
            ref, inventory = self.gradient(mutate=mutate)
            with self.assertRaises(a.EvidenceRejected):
                self.reader().read_gradient(ref, self.expected, self.bindings[:8], inventory, start_anchor_sha256='6' * 64)

    def test_unrelated_artifact_and_source_worker_stage_attempt_mismatch_rejected(self):
        for key, value in [('worker_identity_sha256', '7' * 64), ('stage', 'runtime_anchor'),
                           ('code_sha256', {'different.py': '8' * 64}), ('attempt_id', 'other-attempt')]:
            envelope = self.envelope('masks', {'rows': self.rows}); envelope[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(a.EvidenceRejected, 'envelope'):
                self.reader().read_masks(self.masks(envelope=envelope), self.expected, self.bindings)
        envelope = self.envelope('likelihood', {'rows': self.rows})
        with self.assertRaisesRegex(a.EvidenceRejected, 'envelope'):
            self.reader().read_masks(self.masks(envelope=envelope), self.expected, self.bindings)

    def test_pre_reservation_before_IO_and_failure_never_refunds(self):
        ref = self.masks(); reader = self.reader(io_read_bytes_upper=ref['bytes'])
        with self.assertRaisesRegex(a.EvidenceRejected, 'budget exhausted'):
            reader.read_masks(ref, self.expected, self.bindings)
        self.assertEqual(self.events, [])
        reader = self.reader(); ref['sha256'] = '0' * 64
        before = dict(reader.remaining)
        with self.assertRaisesRegex(a.EvidenceRejected, 'bytes/hash'):
            reader.read_masks(ref, self.expected, self.bindings)
        self.assertEqual(self.events[0][1], 'started'); self.assertEqual(self.events[-1][1], 'failed')
        started = sum(phase == 'started' for _, phase, _ in self.events)
        self.assertEqual(reader.remaining['operation_attempts_upper'], before['operation_attempts_upper'] - started)
        self.assertLess(reader.remaining['identity_hash_attempts_upper'], before['identity_hash_attempts_upper'] - 1)
        self.assertLess(reader.remaining['reducer_work_bytes_upper'], before['reducer_work_bytes_upper'])
        self.assertEqual(reader.remaining['io_read_bytes_upper'], before['io_read_bytes_upper'] - ref['bytes'] - 1)

    def failing_observer_reader(self):
        ordinary = self.reader()
        def operation(kind, phase, **fields):
            self.events.append((kind, phase, fields))
            if phase == 'failed': raise RuntimeError('ledger-failed-marker')
        ordinary.operation = operation
        return ordinary

    def test_hash_error_preserved_when_failure_observer_also_raises(self):
        ref = self.masks(); ref['sha256'] = '0' * 64
        with self.assertRaisesRegex(a.EvidenceRejected, 'bytes/hash') as caught:
            self.failing_observer_reader().read_masks(ref, self.expected, self.bindings)
        self.assertIn('ledger-failed-marker', repr(caught.exception.accounting_failure))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_truncated_artifact_is_failed_not_returned_and_keeps_primary(self):
        ref = self.masks(); path = Path(ref['path']); path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(a.EvidenceRejected, 'path/size/ownership') as caught:
            self.failing_observer_reader().read_masks(ref, self.expected, self.bindings)
        self.assertIn('ledger-failed-marker', repr(caught.exception.accounting_failure))
        self.assertNotIn('returned', [phase for kind, phase, _ in self.events if kind == 'qualification_artifact_readback'])

    def test_nonfinite_consumer_error_survives_both_stream_close_accounting_failures(self):
        ref, inventory = self.gradient(reference=(3.0, float('nan'), 0.0, 12.0))
        with self.assertRaisesRegex(a.EvidenceRejected, 'nonfinite raw gradient') as caught:
            self.failing_observer_reader().read_gradient(ref, self.expected, self.bindings[:8],
                inventory, start_anchor_sha256='6' * 64)
        details = caught.exception.accounting_failure['previous']['stream_close_failures']
        self.assertEqual(len(details), 2)
        self.assertTrue(all('ledger-failed-marker' in repr(value) for value in details))

    def test_partial_consumer_read_never_records_returned(self):
        ref = self.masks(); stream = self.reader()._blocks(ref, self.expected, 'masks')
        self.assertEqual(len(next(stream)), 4); stream.close()
        phases = [phase for kind, phase, _ in self.events if kind == 'qualification_artifact_readback']
        self.assertEqual(phases, ['started', 'failed'])

    def test_arithmetic_verification_budget_is_admitted_before_work(self):
        ref = self.likelihood()
        with self.assertRaisesRegex(a.EvidenceRejected, 'verification budget exhausted'):
            self.reader(reducer_work_bytes_upper=0).read_likelihood(ref, self.expected, self.bindings)
        self.assertEqual(self.events, [])

    def test_maximum_astral_source_names_fit_the_prepaid_binding_hash_bound(self):
        expected = copy.deepcopy(self.expected)
        expected['code_sha256'] = {'\U0001f600' * 508 + '%04d' % index: '4' * 64 for index in range(1024)}
        self.assertTrue(all(len(name) == 512 for name in expected['code_sha256']))
        actual_bytes = len(a.canonical(expected))
        self.assertGreater(actual_bytes, 4 * a.CHUNK)
        self.assertLessEqual(actual_bytes, a.MAX_BINDING_BYTES)
        # Only one actual artifact byte is read; large source metadata stays
        # in the expected binding and its hash is explicitly reserved first.
        ref = self.artifact(b'x', 'gradient_reference', expected)
        reader = self.reader()
        self.assertEqual(b''.join(reader._blocks(ref, expected, 'gradient_reference')), b'x')
        started = [fields for kind, phase, fields in self.events
                   if kind == 'qualification_reducer_overhead' and phase == 'started']
        self.assertEqual(started[0]['identity_hash_bytes_upper'], a.MAX_BINDING_BYTES)
        self.assertGreaterEqual(started[0]['identity_hash_bytes_upper'], actual_bytes)


if __name__ == '__main__': unittest.main()
