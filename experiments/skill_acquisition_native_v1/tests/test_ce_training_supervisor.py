"""Original durable CE byte replay and closed supervisor; no model/OS jobs."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_training_supervisor as s
from . import test_ce_journal as original


class Trap:
    def __getattribute__(self, name): raise AssertionError('before source admission')


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='training-supervisor-pure-'); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan, self.bindings, self.caps = original.fixture()
        self.ledger = original.cj.CeJournal(self.root / 'ledger', self.plan, self.bindings, self.caps,
                                          synthetic_cpu=True, clock=lambda: 1.)
        self.ledger.start_invocation(boot_id='fixture-boot')
        self.binding = {'stage': 'training_supervisor', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
            'static_recipe_sha256': s.digest(self.bindings['recipe']), 'code_sha256': self.bindings['source'],
            'worker_identity_sha256': 'c' * 64, 'attempt_id': 'pure-raw-replay'}
        self.artifact = {'owned_directory': str(self.ledger.directory), 'plan': self.plan, 'bindings': self.bindings, 'caps': self.caps}
        self.calls = []
    def ref(self, path):
        blob = path.read_bytes(); return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    def replay(self, snapshot=None, preflight=None):
        reader = s.raw.RawEvidenceReader(lambda *args, **fields: self.calls.append((args, fields)),
            {key: 10 ** 12 for key in s.raw.BUDGET_FIELDS}, self.root)
        return s.replay_journal(self.artifact, self.ref(self.ledger.header_path), self.ref(self.ledger.path),
            self.ledger.snapshot() if snapshot is None else snapshot, self.binding, reader=reader,
            preflight=preflight or (lambda *args: self.calls.append(('preflight', args))), fixture=True)
    def test_raw_causal_commit_and_ended_invocation_replay_no_execution_claim(self):
        helper = object.__new__(original.Controls); helper.bindings = self.bindings
        self.ledger.commit_boundary(helper.successful_result(self.ledger))
        self.ledger.end_invocation('INTERRUPTED')
        state = self.replay()
        self.assertEqual(state.progress_record()['C'], 1)
        self.assertEqual(state.snapshot(), self.ledger.snapshot())
        self.assertFalse(state.snapshot()['native_qualified'])
    def test_failed_and_pending_costs_cannot_refund_or_claim_cleared_progress(self):
        self.ledger.operation('input_io', 'started', 'failed'); self.ledger.operation('input_io', 'failed', 'failed')
        self.ledger.operation('input_io', 'started', 'pending')
        state = self.replay(); self.assertEqual(state.counters, self.ledger.counters)
        self.assertEqual(state.snapshot()['pending_operations'], 1)
        wrong = copy.deepcopy(self.ledger.snapshot()); wrong['cumulative']['io_read_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'snapshot refund'): self.replay(wrong)
    def test_foreign_header_chain_partial_tail_and_census_are_rejected(self):
        before = self.ledger.path.read_bytes(); self.ledger.path.write_bytes(before[:-1])
        with self.assertRaisesRegex(ValueError, 'partial CE tail'): self.replay()
        self.ledger.path.write_bytes(before.replace(b'"seq":0', b'"seq":9', 1))
        with self.assertRaisesRegex(ValueError, 'hash/order'): self.replay()
        self.ledger.path.write_bytes(before)
        header = s.raw._json(self.ledger.header_path.read_bytes()); first = next(iter(header['token_census']))
        header['token_census'][first]['prompt_tokens'] += 1
        self.ledger.header_path.write_bytes(s.canonical(header) + b'\n')
        with self.assertRaisesRegex(ValueError, 'token census'): self.replay()
    def test_preflight_denial_and_counterfeit_native_supervisor_before_body(self):
        def deny(*args): raise RuntimeError('replay prepaid scope denied')
        with self.assertRaisesRegex(RuntimeError, 'scope denied'): self.replay(preflight=deny)
        self.assertEqual(self.calls, [])
        with patch.object(s.launch, 'launch_native', side_effect=AssertionError('unadmitted child launch')):
            with self.assertRaises(ValueError): s.supervise_training(Trap(), Trap(), Trap())


if __name__ == '__main__': unittest.main()
