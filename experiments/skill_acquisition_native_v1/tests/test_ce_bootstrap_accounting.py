"""Actual local append/readback with explicit pure fixtures; no native grants."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_bootstrap_accounting as b
from .test_ce_qualification_accounting import fixture


class Clock:
    def __init__(self): self.value = 100.0
    def __call__(self): return self.value
    def advance(self, amount): self.value += amount


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='bootstrap-accounting-pure-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.context, self.proof, self.receipt = fixture(self.root)
        self.clock = Clock()
        self.owner = {'accounting_spec_sha256': 'a' * 64, 'prepared_receipt_sha256': 'b' * 64}
        self.bounds = {key: 1000 for key in b.COST_FIELDS}
    def make(self): return b.BootstrapAccounting(self.context, None, self.proof, clock=self.clock)

    def test_native_registry_refuses_before_clock_context_or_files(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unapproved access')
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved IO')):
            with self.assertRaisesRegex(ValueError, 'issuer unregistered'):
                b.BootstrapAccounting(Trap(), Trap(), Trap(), clock=Trap(), native_issuer=lambda *args: True)
            with self.assertRaisesRegex(ValueError, 'verifier is unregistered'):
                b.verify_prepayment(Trap(), Trap(), Trap(), Trap(), verifier=lambda *args: True)
        self.assertFalse(b._REGISTERED_NATIVE_ISSUERS); self.assertFalse(b._REGISTERED_PREPAYMENT_VERIFIERS)

    def test_child_bootstrap_has_independent_directory_and_cannot_reprice_common_scope(self):
        scope = self.context['bindings']['scope']
        scope['checkpoint_qualification'] = {'owned_directory': str(self.root / 'checkpoints')}
        self.context['checkpoint_child_declaration'] = {'branch': 'prefix'}
        before = copy.deepcopy(scope)
        common = scope['qualifier_bootstrap']
        directory = self.root / 'checkpoints' / 'children' / 'prefix' / 'bootstrap'
        self.context['checkpoint_child_bootstrap'] = dict(copy.deepcopy(common), owned_directory=str(directory))
        child = self.make()
        self.assertEqual(child.directory, directory)
        self.assertEqual(scope, before)
        handle = child.reserve_setup(self.owner, self.bounds); handle.begin(); handle.fail(ValueError('fixture'))
        self.assertEqual(handle.evidence['status'], 'FAILED')
        self.assertFalse((self.root / 'bootstrap' / b.JOURNAL).exists())
        self.context['checkpoint_child_bootstrap']['self_bounds']['io_read_bytes_upper'] += 1
        with patch.object(Path, 'open', side_effect=AssertionError('IO before exact child derivation')):
            with self.assertRaisesRegex(ValueError, 'only exact directory'):
                self.make()
        self.assertEqual(scope, before)

    def test_reservation_is_durable_before_dependent_body_and_failed_work_keeps_charge(self):
        ledger = self.make(); handle = ledger.reserve_setup(self.owner, self.bounds)
        last = json.loads(ledger.path.read_bytes().splitlines()[-1])
        self.assertEqual(last['kind'], 'reserve'); self.assertEqual(handle.evidence['status'], 'RESERVED')
        b.verify_prepayment(handle, self.context, self.proof, self.owner)
        handle.begin(); costs = copy.deepcopy(ledger.cumulative)
        handle.fail(OSError('fixture header failed'))
        self.assertEqual(handle.evidence['status'], 'FAILED')
        self.assertTrue(all(ledger.cumulative[key] >= costs[key] for key in b.COST_FIELDS))
        with self.assertRaisesRegex(ValueError, 'status drift'):
            b.verify_prepayment(handle, self.context, self.proof, self.owner)

    def test_owner_worker_and_fixture_handle_cannot_be_substituted(self):
        ledger = self.make(); handle = ledger.reserve_setup(self.owner, self.bounds)
        changed = dict(self.owner, accounting_spec_sha256='c' * 64)
        with self.assertRaisesRegex(ValueError, 'drift'):
            b.verify_prepayment(handle, self.context, self.proof, changed)
        with self.assertRaisesRegex(ValueError, 'immutable'): handle._native = True
        with self.assertRaises(ValueError): b.verify_prepayment(handle.evidence, self.context, self.proof, self.owner)

    def test_duplicate_and_wrong_phase_ticket_denials_do_not_persist_invalid_rows(self):
        ledger = self.make(); handle = ledger.reserve_setup(self.owner, self.bounds)
        def unchanged(call):
            raw, head, tickets = ledger.path.read_bytes(), ledger._head(), copy.deepcopy(ledger.tickets)
            with self.assertRaises(ValueError): call()
            self.assertEqual(ledger.path.read_bytes(), raw)
            self.assertEqual(ledger._head(), head)
            self.assertEqual(ledger.tickets, tickets)
            self.assertFalse(ledger.fatal)
        unchanged(handle.complete)
        handle.begin(); unchanged(handle.begin)
        unchanged(lambda: ledger._event('setup_returned', {'ticket_id': handle.evidence['ticket_id'],
            'monotonic': ledger.last_clock, 'worker_identity_sha256': '0' * 64}))
        handle.complete(); unchanged(handle.complete)
        unchanged(lambda: handle.fail(OSError('late error')))

    def test_idle_wall_and_physical_deadline_are_recorded_separately(self):
        ledger = self.make(); before = ledger.cumulative['wall_seconds_upper']
        self.clock.advance(7.5); ledger.tick_wall()
        self.assertEqual(ledger.actual_wall_seconds, 7.5)
        self.assertEqual(ledger.cumulative['wall_seconds_upper'], before + 7.5)
        self.clock.advance(10001)
        with self.assertRaisesRegex(ValueError, 'physical deadline'): ledger.tick_wall()

    def test_completed_replay_keeps_all_reservations_and_header_overheads(self):
        ledger = self.make(); handle = ledger.reserve_setup(self.owner, self.bounds); handle.begin(); handle.complete()
        old = copy.deepcopy(ledger.cumulative)
        reopened = self.make()
        self.assertTrue(all(reopened.cumulative[key] >= old[key] for key in b.COST_FIELDS))
        self.assertEqual(reopened.tickets[handle.evidence['ticket_id']]['status'], 'RETURNED')
        evidence = reopened.inspect_evidence()
        self.assertEqual(evidence['journal_ref']['bytes'], reopened.path.stat().st_size)
        self.assertFalse(evidence['native_qualified'])

    def test_pending_ticket_replay_is_unknown_and_cannot_repay_or_hide_corruption(self):
        ledger = self.make(); ledger.reserve_setup(self.owner, self.bounds).begin()
        reopened = self.make(); self.assertTrue(reopened.unknown_wall)
        with self.assertRaisesRegex(ValueError, 'unknown wall'): reopened.reserve_setup(self.owner, self.bounds)
        with reopened.path.open('ab') as stream: stream.write(b'{')
        original = reopened.path.read_bytes()
        with self.assertRaises(ValueError): self.make()
        self.assertEqual(reopened.path.read_bytes(), original)

    def test_normal_end_prepays_final_record_and_allows_only_one_bounded_readback(self):
        ledger = self.make(); handle = ledger.reserve_setup(self.owner, self.bounds); handle.begin(); self.clock.advance(2); handle.complete()
        ended = ledger.end(); self.assertTrue(ended['clock_recorded_end']); self.assertTrue(ended['body_completed'])
        self.assertEqual(ended['original_origin'], 100); self.assertEqual(ended['last_monotonic'], 102)
        rows = [json.loads(line) for line in ledger.path.read_bytes().splitlines()]
        self.assertEqual(rows[-2]['kind'], 'end_reservation'); self.assertEqual(rows[-1]['kind'], 'end')
        self.assertGreaterEqual(ended['cumulative']['io_write_bytes_upper'], rows[-2]['payload']['bounds']['io_write_bytes_upper'])
        evidence = ledger.inspect_evidence(); self.assertEqual(evidence['journal_ref']['bytes'], ledger.path.stat().st_size)
        for work in (ledger.end, ledger.tick_wall, lambda: ledger.reserve_setup(self.owner, self.bounds), ledger.inspect_evidence):
            with self.assertRaises(ValueError): work()
        before = ledger.path.read_bytes(); reopened = self.make()
        self.assertTrue(reopened.closed); self.assertEqual(reopened.started_clock, 100); self.assertEqual(reopened.path.read_bytes(), before)

    def test_same_worker_reopen_bridges_every_second_and_preserves_original_origin(self):
        ledger = self.make(); self.clock.advance(2); ledger.tick_wall(); self.clock.advance(3)
        reopened = self.make(); self.assertEqual(reopened.started_clock, 100)
        self.assertEqual(reopened.actual_wall_seconds, 5); self.assertEqual(reopened.last_clock, 105)
        self.assertEqual(reopened.events[-1]['kind'], 'resume')
        self.clock.advance(10000)
        with self.assertRaisesRegex(ValueError, 'physical deadline'): reopened.reserve_setup(self.owner, self.bounds)

    def test_failed_ticket_can_close_clock_but_cannot_be_promoted_to_body_success(self):
        ledger = self.make(); handle = ledger.reserve_setup(self.owner, self.bounds); handle.begin(); handle.fail(OSError('retained failure'))
        costs = copy.deepcopy(ledger.cumulative); ended = ledger.end()
        self.assertEqual(ended['failed_ticket_ids'], [handle.evidence['ticket_id']]); self.assertEqual(ended['outcome'], 'FAILED')
        self.assertTrue(ended['clock_recorded_end']); self.assertFalse(ended['body_completed']); self.assertFalse(ended['native_qualified'])
        self.assertTrue(all(ended['cumulative'][key] >= costs[key] for key in b.COST_FIELDS))
        with self.assertRaises(ValueError): handle.complete()

    def test_pending_reopen_records_unknown_once_and_reopening_cannot_reset_it(self):
        ledger = self.make(); handle = ledger.reserve_setup(self.owner, self.bounds); handle.begin(); self.clock.advance(3)
        with self.assertRaisesRegex(ValueError, 'terminal'): ledger.end()
        reopened = self.make(); self.assertTrue(reopened.unknown_wall); self.assertEqual(reopened.events[-1]['kind'], 'interrupted')
        original = reopened.path.read_bytes(); again = self.make()
        self.assertTrue(again.unknown_wall); self.assertEqual(again.path.read_bytes(), original)
        self.assertEqual(again.tickets[handle.evidence['ticket_id']]['status'], 'STARTED')
        with self.assertRaises(ValueError): again.end()

    def test_final_write_suffix_is_prepaid_and_late_durability_is_not_clock_success(self):
        ledger = self.make(); original_sync = b.os.fsync; calls = []
        def delayed_sync(descriptor):
            calls.append(descriptor)
            if len(calls) == 2:
                self.assertIsNotNone(ledger._end_reserved)
                self.assertGreaterEqual(ledger.cumulative['wall_seconds_upper'], ledger._end_reserved['bounds']['wall_seconds_upper'])
                self.clock.advance(ledger._end_reserved['bounds']['wall_seconds_upper'] + 1)
            return original_sync(descriptor)
        with patch.object(b.os, 'fsync', side_effect=delayed_sync):
            with self.assertRaisesRegex(ValueError, 'suffix exceeds prepaid'): ledger.end()
        self.assertTrue(ledger.unknown_wall); self.assertTrue(ledger.fatal)
        self.assertFalse(ledger.snapshot()['complete_recorded_clock']); self.assertFalse(ledger.snapshot()['body_completed'])
        with self.assertRaises(ValueError): ledger.reserve_setup(self.owner, self.bounds)

    def test_interrupted_final_record_retains_reservation_and_cannot_restart_work(self):
        ledger = self.make(); original_event = ledger._event
        def interrupted(kind, payload, **kwargs):
            if kind == 'end': raise OSError('fixture interruption before final record')
            return original_event(kind, payload, **kwargs)
        with patch.object(ledger, '_event', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'before final record'): ledger.end()
        reserved = copy.deepcopy(ledger._end_reserved)
        reopened = self.make(); self.assertTrue(reopened.unknown_wall)
        self.assertEqual(reopened.events[-1]['kind'], 'interrupted')
        self.assertEqual(reopened._end_reserved, reserved)
        self.assertEqual(reopened.events[-1]['payload']['pending_end_reservation_sha256'], b.digest(reserved))
        with self.assertRaises(ValueError): reopened.end()


if __name__ == '__main__': unittest.main()
