"""Small reversible filesystem fixtures; no models, native authority or PASS."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_publication as p, ce_qualification_accounting as a
from .test_ce_qualification_accounting import fixture


def reference(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='publication-pure-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.context, self.proof, self.prepared = fixture(self.root)
        ledger = a.QualificationAccounting(self.context, None, self.proof, self.prepared)
        self.artifacts = {}
        for stage in a.q.STAGES:
            path = self.root / (stage + '.json'); path.write_bytes(b'{"fixture":true}\n')
            ref = reference(path); self.artifacts[stage] = [ref]
            if stage != 'ingestion_tokenizer':
                ledger.begin_stage(stage, 'fixture/' + stage); ledger.finish_stage([ref])
        ledger.end()
        self.snapshot = ledger.snapshot()
        self.refs = {'header': reference(ledger.header_path), 'journal': reference(ledger.path)}
        self.terminal = {'worker_identity_sha256': self.proof['worker_identity_sha256'],
            'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'], 'exit_code': 0}
        bounds = {key: 10 ** 12 for key in qfields()}; bounds['wall_seconds_upper'] = 1000
        self.context['bindings']['scope']['publication'] = {'schema': p.SPEC, 'owned_directory': str(self.root / 'published'),
            'max_receipt_bytes': p.MAX_JSON, 'bounds': bounds,
            'reader_bounds': {key: 10 ** 12 for key in p.raw.BUDGET_FIELDS}, 'external_input_refs': []}
    def publisher(self): return p.CompletionPublisher(self.context, None, self.proof, self.prepared)
    def publish(self, publisher):
        # Keep native Windows ABI calls outside pure suite; root separately
        # qualifies the small platform primitive. This uses create-only links.
        def exclusive(source, target, root):
            self.assertEqual(source.parent.resolve(), root.resolve())
            self.assertEqual(target.parent.resolve(), root.resolve())
            os.link(source, target)
            return {'method': 'fixture_exclusive_hardlink', 'replacement_allowed': False}
        with patch.object(p, '_atomic_no_replace', exclusive):
            return publisher.publish(self.snapshot, self.artifacts, self.refs, self.terminal)

    def test_closed_native_issuer_refuses_before_source_objects_or_io(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unadmitted native object')
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved IO')):
            with self.assertRaisesRegex(ValueError, 'issuer is unregistered'):
                p.CompletionPublisher(Trap(), Trap(), Trap(), Trap(), native_issuer=lambda *args: True)
        self.assertFalse(p._REGISTERED_PUBLICATION_ISSUERS)
        self.assertFalse(p._REGISTERED_EXECUTION_VERIFIERS)

    def test_create_only_receipt_binds_raw_artifacts_costs_and_terminal_but_is_unqualified(self):
        publisher = self.publisher(); result = self.publish(publisher)
        path = Path(result['receipt_ref']['path']); raw = path.read_bytes(); value = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result['receipt_ref']['sha256'])
        self.assertEqual(value['cumulative'], self.snapshot['cumulative'])
        self.assertEqual(value['ledger_refs'], self.refs)
        self.assertEqual(value['stage_artifacts'], self.artifacts)
        self.assertEqual(value['terminal_worker'], self.terminal)
        self.assertEqual(value['publication_prepayment']['bounds'], publisher.spec['bounds'])
        self.assertEqual(value['publication_prepayment']['reservation_sha256'], publisher.handle.evidence['reservation_sha256'])
        self.assertFalse(value['native_qualified']); self.assertFalse(value['execution_verified'])
        self.assertEqual(value['status'], 'UNQUALIFIED_COMPLETION')
        self.assertEqual(result['prepayment']['status'], 'RETURNED')

    def test_existing_final_never_overwritten(self):
        publisher = self.publisher(); publisher.directory.mkdir()
        path = publisher.directory / 'QUALIFICATION_COMPLETION.json'; path.write_bytes(b'original')
        with self.assertRaisesRegex(ValueError, 'never overwrites'):
            self.publish(publisher)
        self.assertEqual(path.read_bytes(), b'original')
        self.assertEqual(publisher.handle.evidence['status'], 'FAILED')

    def test_refund_or_pending_cleanup_claim_is_rejected_by_source_replay(self):
        for mutation in ('refund', 'pending'):
            publisher = self.publisher()
            original = copy.deepcopy(self.snapshot)
            if mutation == 'refund': self.snapshot['cumulative']['io_write_bytes_upper'] -= 1
            else: self.snapshot['pending_operations'] = [{'kind': 'cleanup', 'phase': 'started'}]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.publish(publisher)
            self.assertFalse((publisher.directory / 'QUALIFICATION_COMPLETION.json').exists())
            self.snapshot = original

    def test_foreign_or_changed_stage_artifact_is_refused(self):
        publisher = self.publisher()
        with tempfile.TemporaryDirectory(prefix='foreign-publication-') as directory:
            path = Path(directory) / 'foreign.json'; path.write_bytes(b'{"foreign":true}\n')
            with self.assertRaisesRegex(ValueError, 'foreign publication'):
                publisher._owned_ref(reference(path))
        self.artifacts['runtime_anchor'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.publish(publisher)

    def test_stale_worker_or_failed_terminal_exit_cannot_publish(self):
        for change in ({'worker_identity_sha256': '0' * 64}, {'exit_code': 1}):
            publisher = self.publisher(); original = dict(self.terminal); self.terminal.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'same terminal worker'):
                self.publish(publisher)
            self.terminal = original

    def test_short_write_retains_partial_and_full_prepaid_charge(self):
        publisher = self.publisher(); original_open = Path.open
        class ShortWriter:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): self.stream.__enter__(); return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, block): self.stream.write(block[:5]); return 5
        def opened(path, mode='r', *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            return ShortWriter(stream) if mode == 'xb' and path.suffix == '.partial' else stream
        with patch.object(Path, 'open', opened), self.assertRaisesRegex(ValueError, 'short publication write'):
            self.publish(publisher)
        partials = list(publisher.directory.glob('*.partial'))
        self.assertEqual(len(partials), 1); self.assertEqual(partials[0].stat().st_size, 5)
        self.assertEqual(publisher.handle.evidence['status'], 'FAILED')
        self.assertGreater(publisher.counts['io_write_bytes_upper'], 5)
        self.assertFalse((publisher.directory / 'QUALIFICATION_COMPLETION.json').exists())

    def test_large_declared_artifact_is_streamed_without_join_or_large_read(self):
        publisher = self.publisher()
        reference = {'path': str(self.root / 'large.f32'), 'bytes': 9 * p.CHUNK,
                     'sha256': 'a' * 64}
        yielded = []
        def blocks(*args):
            # One immutable chunk reused: total size exceeds JSON ceiling,
            # while the fixture allocates only one bounded chunk at a time.
            chunk = b'0' * p.CHUNK
            for index in range(9):
                yielded.append(index)
                yield chunk
        with patch.object(publisher, '_owned_ref', lambda ref: ref), patch.object(publisher.reader, '_blocks', blocks):
            self.assertEqual(publisher._stream_ref(reference, 'large_raw'), reference['bytes'])
            with self.assertRaisesRegex(ValueError, 'bounded JSON'):
                publisher._read_ref(reference, 'large_raw')
        self.assertEqual(yielded, list(range(9)))


def qfields(): return p.q.COST_FIELDS


if __name__ == '__main__': unittest.main()
