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


if __name__ == '__main__': unittest.main()
