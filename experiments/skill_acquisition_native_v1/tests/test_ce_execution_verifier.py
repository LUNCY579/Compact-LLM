"""Retained real temporary-file accounting fixtures, without models/frameworks."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.skill_acquisition_native_v1 import ce_execution_verifier as v
from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as a
from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as raw
from .test_ce_qualification_accounting import fixture


class ReplayControls(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.context, self.proof, self.receipt = fixture(self.root)
        self.ledger = a.QualificationAccounting(self.context, None, self.proof, self.receipt)
        self.context['prepared_receipt'] = self.receipt
        self.context['allowed_worker_identities'] = [self.proof['worker_identity_sha256']]
        self.context['verification_binding'] = {'stage': 'qualification_verifier',
            'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
            'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': 'f' * 64, 'attempt_id': 'metadata-verifier-fixture'}
        self.preflights, self.operations = [], []

    def ref(self, path):
        content = Path(path).read_bytes()
        return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}

    def seal(self, change=None):
        value = self.ledger.snapshot()
        if change: change(value)
        path = self.root / 'seal.json'; path.write_bytes(raw.canonical(value) + b'\n')
        return self.ref(path)

    def verify(self, seal_change=None, handles=None, preflight=None):
        reader = raw.RawEvidenceReader(lambda kind, phase, **fields: self.operations.append((kind, phase, fields)),
            {'operation_attempts_upper': 10000, 'io_read_bytes_upper': 1024 ** 3,
             'identity_hash_attempts_upper': 1000000, 'identity_hash_bytes_upper': 8 * 1024 ** 3,
             'reducer_work_bytes_upper': 8 * 1024 ** 3}, self.root)
        return v.verify_qualification(self.context, self.ref(self.ledger.header_path), self.ref(self.ledger.path),
            self.seal(seal_change), reader=reader, worker_handles=handles,
            outer_preflight=preflight or (lambda kind, fields: self.preflights.append((kind, fields))))

    def rewrite(self, mutate):
        entries = [raw._json(line) for line in self.ledger.path.read_bytes().splitlines()]
        mutate(entries)
        previous = None; lines = []
        for seq, entry in enumerate(entries):
            body = {key: value for key, value in entry.items() if key != 'sha256'}
            body['seq'], body['previous'] = seq, previous
            changed = dict(body, sha256=raw.digest(body)); previous = changed['sha256']
            lines.append(raw.canonical(changed) + b'\n')
        self.ledger.path.write_bytes(b''.join(lines))

    def test_setup_worker_replay_derives_totals_and_never_native_pass(self):
        result = self.verify()
        self.assertEqual(result['derived_totals'], self.ledger.cumulative)
        self.assertEqual(result['events'], len(self.ledger.events))
        self.assertEqual(result['status'], 'UNQUALIFIED')
        self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])
        self.assertFalse(result['all_stages_complete_by_replay'])
        self.assertEqual(self.preflights[0][0], 'qualification_execution_verifier')

    def test_failed_operation_reservations_are_not_refunded(self):
        self.ledger.begin_stage('runtime_anchor', 'failed-stage')
        self.ledger.operation('model_load', 'started', 'load')
        self.ledger.operation('model_load', 'failed', 'load', error_type='fixture')
        self.ledger.fail_stage(RuntimeError('fixture failed'))
        result = self.verify()
        self.assertEqual(result['derived_totals'], self.ledger.cumulative)
        self.assertEqual(result['derived_attempts'][0]['status'], 'FAILED')
        self.assertGreaterEqual(result['derived_totals']['io_read_bytes_upper'], 10 ** 9)
        with self.assertRaisesRegex(raw.EvidenceRejected, 'totals/refunds'):
            self.verify(lambda seal: seal['cumulative'].update(io_read_bytes_upper=0))

    def test_pending_lifetime_cleanup_is_charged_and_cannot_be_sealed_complete(self):
        self.ledger.begin_stage('runtime_anchor', 'pending-cleanup')
        self.ledger.operation('cleanup', 'started', 'lifetime', deferred_model_cleanup=True)
        result = self.verify()
        self.assertEqual(len(result['pending_operations']), 1)
        self.assertTrue(result['pending_operations'][0]['lifetime_cleanup'])
        self.assertGreaterEqual(result['derived_totals']['wall_seconds_upper'], 1000)
        with self.assertRaisesRegex(raw.EvidenceRejected, 'claims completion'):
            self.verify(lambda seal: seal.update(completed=True))

    def test_tampered_hash_and_partial_tail_fail_without_repair(self):
        content = self.ledger.path.read_bytes()
        changed = content.replace(b'"seq":0', b'"seq":9', 1); self.ledger.path.write_bytes(changed)
        with self.assertRaisesRegex(raw.EvidenceRejected, 'hash/body/order'):
            self.verify()
        self.ledger.path.write_bytes(content[:-1])
        with self.assertRaisesRegex(raw.EvidenceRejected, 'partial ledger tail'):
            self.verify()
        self.assertEqual(self.ledger.path.read_bytes(), content[:-1])

    def test_rehashed_setup_repricing_and_foreign_worker_still_fail(self):
        original = self.ledger.path.read_bytes()
        self.rewrite(lambda entries: entries[0]['payload']['profile'].update(read_bytes_upper=0))
        with self.assertRaisesRegex(raw.EvidenceRejected, 'refund/repricing'):
            self.verify()
        self.ledger.path.write_bytes(original)
        self.rewrite(lambda entries: entries[1]['payload']['proof'].update(worker_identity_sha256='7' * 64))
        with self.assertRaisesRegex(raw.EvidenceRejected, 'foreign worker'):
            self.verify()

    def test_rehashed_worker_source_and_header_recipe_mismatch_rejected(self):
        original = self.ledger.path.read_bytes()
        self.rewrite(lambda entries: entries[1]['payload']['proof'].update(code_sha256={'foreign.py': '8' * 64}))
        with self.assertRaisesRegex(raw.EvidenceRejected, 'worker source'):
            self.verify()
        self.ledger.path.write_bytes(original)
        header = raw._json(self.ledger.header_path.read_bytes()); header['recipe']['optimizer']['lr'] *= 2
        self.ledger.header_path.write_bytes(raw.canonical(header) + b'\n')
        with self.assertRaisesRegex(raw.EvidenceRejected, 'original header'):
            self.verify()

    def test_stage_predecessor_and_attempt_reuse_are_replayed_not_trusted(self):
        self.ledger.begin_stage('runtime_anchor', 'attempt')
        self.ledger.fail_stage(RuntimeError('fixture'))
        self.ledger.begin_stage('runtime_anchor', 'second')
        original = self.ledger.path.read_bytes()
        self.rewrite(lambda entries: entries[-1]['payload'].update(stage='longest_masks'))
        with self.assertRaisesRegex(ValueError, 'predecessor'):
            self.verify()
        self.ledger.path.write_bytes(original)
        self.rewrite(lambda entries: entries[-1]['payload'].update(attempt_id='attempt'))
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.verify()

    def test_preflight_denial_precedes_all_reader_work(self):
        def deny(*args): raise RuntimeError('verification scope denied')
        with self.assertRaisesRegex(RuntimeError, 'scope denied'):
            self.verify(preflight=deny)
        self.assertEqual(self.operations, [])

    def test_checkpoint_child_artifact_binds_separate_raw_request_to_same_study(self):
        attempt = {'stage': 'checkpoint_resume', 'attempt_id': 'parent-cp', 'status': 'RETURNED'}
        parent = v._binding_for(self.context, attempt, self.proof['worker_identity_sha256'])
        child = dict(parent, worker_identity_sha256='6' * 64, attempt_id='child-cp')
        request_path = self.root / 'child-request.json'; request_path.write_bytes(b'{"branch":"uninterrupted"}\n')
        request_ref = self.ref(request_path)
        lineage = {'schema': a.CHILD_PREFIX, 'worker_identity_sha256': child['worker_identity_sha256'],
            'study_request_sha256': child['request_sha256'], 'child_request_sha256': request_ref['sha256'],
            'child_request_ref': request_ref, 'parent_request_sha256': parent['request_sha256'],
            'parent_worker_identity_sha256': parent['worker_identity_sha256'],
            **{key: parent[key] for key in ('scope_sha256', 'static_recipe_sha256', 'code_sha256')},
            'parent_proof': {key: parent[key] for key in ('request_sha256', 'scope_sha256', 'worker_identity_sha256')},
            'child_proof': {**{key: child[key] for key in ('request_sha256', 'scope_sha256', 'worker_identity_sha256')},
                'sources': {'request_ref': request_ref}}}
        artifact = dict(request_ref, role='checkpoint_branch', binding_sha256=raw.digest(child))
        attempt['evidence'] = {'raw_refs': [artifact]}
        state = SimpleNamespace(attempts=[attempt])
        reductions, gaps = v._reductions(self.context, state, {'parent-cp': parent['worker_identity_sha256']},
            None, {artifact['path']: child}, {child['worker_identity_sha256']: lineage})
        self.assertEqual(reductions, {})
        self.assertTrue(any('OS handle' in gap for gap in gaps))
        wrong = copy.deepcopy(lineage); wrong['child_request_sha256'] = child['request_sha256']
        with self.assertRaisesRegex(ValueError, 'raw child request hash'):
            v._reductions(self.context, state, {'parent-cp': parent['worker_identity_sha256']}, None,
                {artifact['path']: child}, {child['worker_identity_sha256']: wrong})

    def test_supplied_worker_flags_cannot_replace_actual_execution_handles(self):
        with self.assertRaisesRegex(raw.EvidenceRejected, 'source-issued handles'):
            self.verify(handles={self.proof['worker_identity_sha256']: {'native_actual': True, 'terminal': True}})


if __name__ == '__main__': unittest.main()
