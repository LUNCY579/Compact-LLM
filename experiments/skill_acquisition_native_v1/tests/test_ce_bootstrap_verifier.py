"""Raw real-file replay and tamper controls; no framework or worker execution."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_bootstrap_verifier as v
from . import test_ce_bootstrap_accounting as fixtures
from . import test_ce_qualification_accounting as qa


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='bootstrap-verifier-pure-')
        self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.context, self.proof, self.receipt = qa.fixture(self.root)
        self.context['verification_binding'] = {'stage': 'bootstrap_verification',
            'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
            'static_recipe_sha256': self.context['static_recipe_sha256'], 'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'attempt_id': 'independent-bootstrap-replay'}
        self.clock = fixtures.Clock()
        self.ledger = v.bootstrap.BootstrapAccounting(self.context, None, self.proof, clock=self.clock)
        self.owner = {'accounting_spec_sha256': 'a' * 64, 'prepared_receipt_sha256': 'b' * 64}
        self.bounds = {key: 1000 for key in v.bootstrap.COST_FIELDS}
        self.calls = []
    def ref(self, path):
        blob = path.read_bytes(); return {'path': str(path), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
    def seal(self, snapshot=None):
        path = self.root / 'seal.json'; path.write_bytes(v.canonical(self.ledger.snapshot() if snapshot is None else snapshot) + b'\n')
        return self.ref(path)
    def reader(self):
        return v.raw.RawEvidenceReader(lambda *args, **fields: self.calls.append((args, fields)),
            {key: 10 ** 12 for key in v.raw.BUDGET_FIELDS}, self.root)
    def verify(self, *, header=None, journal=None, seal=True, evidence=(), contracts=None, workers=None):
        return v.verify_bootstrap(self.context, self.ref(self.ledger.header_path) if header is None else header,
            self.ref(self.ledger.path) if journal is None else journal, self.seal() if seal is True else None if seal is False else seal,
            reader=self.reader(), outer_preflight=lambda *args: self.calls.append(('preflight', args)),
            expected_workers={self.proof['worker_identity_sha256']: self.proof} if workers is None else workers,
            expected_owners=[{'owner': self.owner, 'bounds': self.bounds}] if contracts is None else contracts,
            outer_evidence=evidence)
    def rewrite(self, mutate):
        rows = [v.raw._json(line) for line in self.ledger.path.read_bytes().splitlines()]
        mutate(rows); previous = None
        for seq, row in enumerate(rows):
            row.update(seq=seq, previous=previous)
            row['sha256'] = v.digest({key: val for key, val in row.items() if key != 'sha256'}); previous = row['sha256']
        self.ledger.path.write_bytes(b''.join(v.canonical(row) + b'\n' for row in rows))
    def test_original_header_chain_offsets_tickets_and_failed_costs_recompute(self):
        completed = self.ledger.reserve_setup(self.owner, self.bounds); completed.begin()
        self.clock.advance(1.5); completed.complete()
        failed = self.ledger.reserve_setup(self.owner, self.bounds); failed.begin(); self.clock.advance(2); failed.fail(OSError('file write failed'))
        pending = self.ledger.reserve_setup(self.owner, self.bounds); pending.begin()
        evidence = self.ledger.inspect_evidence()
        result = self.verify(evidence=[completed.evidence, failed.evidence, pending.evidence])
        self.assertEqual(result['derived_totals'], self.ledger.cumulative)
        self.assertEqual(result['derived_tickets'], self.ledger.tickets)
        self.assertEqual(result['recorded_wall_seconds'], 3.5)
        self.assertEqual(result['ticket_statuses']['STARTED'], [pending.evidence['ticket_id']])
        self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified']); self.assertFalse(result['clock_coverage_complete'])
        self.assertEqual(sum(row['bytes'] for row in result['original_offsets']), self.ledger.path.stat().st_size)
        self.assertEqual(result['original_offsets'][-1]['offset'] + result['original_offsets'][-1]['bytes'], evidence['journal_ref']['bytes'])
        self.assertTrue(all(row['full_upper_reservation_retained'] for row in result['matched_outer_evidence']))
    def test_external_preflight_denial_precedes_all_io_and_hashing(self):
        refs = self.ref(self.ledger.header_path), self.ref(self.ledger.path)
        def denied(*args): raise ValueError('external prepaid verification denied')
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('unpriced file read')), patch.object(v, 'digest', side_effect=AssertionError('unpriced arithmetic hash')):
            with self.assertRaisesRegex(ValueError, 'external prepaid'):
                v.verify_bootstrap(self.context, *refs, reader=self.reader(), outer_preflight=denied,
                    expected_workers={}, expected_owners=[])
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})
    def test_refund_forged_ticket_status_or_outer_owner_rejects(self):
        handle = self.ledger.reserve_setup(self.owner, self.bounds); handle.begin(); handle.fail(ValueError('fixture'))
        snapshot = self.ledger.snapshot(); snapshot['cumulative']['io_read_bytes_upper'] -= 1
        with self.assertRaisesRegex(ValueError, 'refund/ticket'): self.verify(seal=self.seal(snapshot))
        snapshot = self.ledger.snapshot(); snapshot['tickets'][handle.evidence['ticket_id']]['status'] = 'RETURNED'
        with self.assertRaisesRegex(ValueError, 'refund/ticket'): self.verify(seal=self.seal(snapshot))
        bad = copy.deepcopy(handle.evidence); bad['owner']['accounting_spec_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'owner/reservation/status'): self.verify(evidence=[bad])
    def test_rehashed_foreign_owner_repricing_and_wrong_lifecycle_still_reject(self):
        handle = self.ledger.reserve_setup(self.owner, self.bounds); handle.begin(); handle.complete()
        original = self.ledger.path.read_bytes()
        for mutate in (lambda rows: next(row for row in rows if row['kind'] == 'reserve')['payload']['owner'].update(accounting_spec_sha256='f' * 64),
                       lambda rows: next(row for row in rows if row['kind'] == 'reserve')['payload']['bounds'].update(io_read_bytes_upper=999),
                       lambda rows: next(row for row in rows if row['kind'] == 'setup_started').update(kind='setup_returned')):
            self.ledger.path.write_bytes(original); self.rewrite(mutate)
            with self.assertRaises(ValueError): self.verify(seal=False)
    def test_rehashed_foreign_worker_clock_jump_boot_or_missing_failure_rejects(self):
        handle = self.ledger.reserve_setup(self.owner, self.bounds); handle.begin(); self.clock.advance(1); handle.fail(ValueError('specific failure'))
        original = self.ledger.path.read_bytes()
        for mutate in (lambda rows: next(row for row in rows if row['kind'] == 'worker')['payload']['proof'].update(worker_identity_sha256='f' * 64),
                       lambda rows: next(row for row in rows if row['kind'] == 'wall')['payload'].update(elapsed_delta=0),
                       lambda rows: next(row for row in rows if row['kind'] == 'wall')['payload'].update(boot_identity_sha256='f' * 64),
                       lambda rows: next(row for row in rows if row['kind'] == 'setup_failed')['payload'].pop('error')):
            self.ledger.path.write_bytes(original); self.rewrite(mutate)
            with self.assertRaises(ValueError): self.verify(seal=False)
    def test_duplicate_ticket_rehashed_chain_partial_tail_and_hash_mismatch_are_refused(self):
        self.ledger.reserve_setup(self.owner, self.bounds)
        original = self.ledger.path.read_bytes()
        self.rewrite(lambda rows: rows.append(copy.deepcopy(next(row for row in rows if row['kind'] == 'reserve'))))
        with self.assertRaisesRegex(ValueError, 'unique bootstrap ticket'): self.verify(seal=False)
        self.ledger.path.write_bytes(original + b'{partial')
        with self.assertRaisesRegex(ValueError, 'partial bootstrap tail'): self.verify(seal=False)
        self.ledger.path.write_bytes(original)
        wrong = self.ref(self.ledger.path); wrong['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'bytes/hash mismatch'): self.verify(journal=wrong, seal=False)
    def test_reopen_clock_is_bridged_and_original_deadline_cannot_reset(self):
        self.clock.advance(2); self.ledger.tick_wall(); self.clock.advance(3)
        reopened = v.bootstrap.BootstrapAccounting(self.context, None, self.proof, clock=self.clock)
        self.ledger = reopened
        result = self.verify()
        self.assertEqual(result['worker_origin_gaps'], [])
        self.assertEqual(result['recorded_wall_seconds'], 5)
        self.assertEqual(result['first_monotonic'], 100)
        self.assertFalse(result['clock_coverage_complete'])
        self.rewrite(lambda rows: rows[-1]['payload'].update(monotonic=10101))
        with self.assertRaisesRegex(ValueError, 'physical deadline'): self.verify(seal=False)
    def test_foreign_source_header_and_unrelated_outer_schema_have_explicit_refusal(self):
        header = v.raw._json(self.ledger.header_path.read_bytes()); header['source_code_sha256']['code_sha256'] = {'foreign': 'a' * 64}
        path = self.root / 'foreign-header.json'; path.write_bytes(v.canonical(header) + b'\n')
        with self.assertRaisesRegex(ValueError, 'header/request/scope/source'): self.verify(header=self.ref(path))
        with self.assertRaisesRegex(ValueError, 'unsupported outer evidence'):
            self.verify(evidence=[{'schema': 'CE8-SERVICES-BOOTSTRAP-1', 'native_qualified': True}])

    def test_publisher_started_evidence_links_to_later_return_without_inventing_body_order(self):
        handle = self.ledger.reserve_setup(self.owner, self.bounds); handle.begin()
        retained_during_write = copy.deepcopy(handle.evidence)
        handle.complete()
        result = self.verify(evidence=[retained_during_write])
        matched = result['matched_outer_evidence'][0]
        self.assertEqual(matched['observed_status'], 'STARTED'); self.assertEqual(matched['terminal_status'], 'RETURNED')
        self.assertFalse(matched['artifact_birth_order_verified']); self.assertFalse(matched['body_operation_actuals_verified'])
        bad = dict(retained_during_write, status='FAILED')
        with self.assertRaisesRegex(ValueError, 'never occurred'): self.verify(evidence=[bad])

    def test_closed_normal_and_failed_body_clocks_replay_without_native_final_claim(self):
        handle = self.ledger.reserve_setup(self.owner, self.bounds); handle.begin(); self.clock.advance(2); handle.complete()
        self.ledger.end(); evidence = self.ledger.inspect_evidence()
        result = self.verify(evidence=[handle.evidence])
        self.assertTrue(result['clock_coverage_complete']); self.assertFalse(result['physical_final_verified'])
        self.assertEqual(result['recorded_wall_seconds'], 2)
        self.assertIsNotNone(result['end_reservation']); self.assertFalse(result['native_qualified'])
        self.assertEqual(evidence['snapshot']['failed_ticket_ids'], [])
        self.rewrite(lambda rows: next(row for row in rows if row['kind'] == 'end')['payload'].update(reservation_sha256='f' * 64))
        with self.assertRaisesRegex(ValueError, 'exact source end'): self.verify(seal=False)

    def test_pending_interruption_is_durable_and_never_becomes_ended_coverage(self):
        handle = self.ledger.reserve_setup(self.owner, self.bounds); handle.begin(); self.clock.advance(3)
        self.ledger = v.bootstrap.BootstrapAccounting(self.context, None, self.proof, clock=self.clock)
        result = self.verify()
        self.assertFalse(result['clock_coverage_complete']); self.assertEqual(result['ticket_statuses']['STARTED'], [handle.evidence['ticket_id']])
        self.assertTrue(self.ledger.snapshot()['unknown_wall']); self.assertEqual(self.ledger.events[-1]['kind'], 'interrupted')
        with self.assertRaises(ValueError): self.ledger.end()


if __name__ == '__main__': unittest.main()
