"""Counterfeit native hooks must refuse before model, IO or framework work."""
import sys
import unittest
from pathlib import Path
import tempfile

from experiments.skill_acquisition_native_v1 import ce_native_authority as authority
from experiments.skill_acquisition_native_v1 import ce_journal, ce_update, update
from .test_ce_update import recipe, rows


class Trap:
    def __getattribute__(self, name):
        raise AssertionError('unadmitted object accessed: ' + name)
    def __fspath__(self):
        raise AssertionError('unadmitted path accessed')


class NativeHookTests(unittest.TestCase):
    def test_counterfeit_permits_cannot_enter_core_journal_or_donor(self):
        class Forged:
            def require(self, *args, **kwargs): return True
        before = {name for name in ('torch', 'numpy', 'transformers', 'safetensors') if name in sys.modules}
        for permit in (True, {'approved': True}, Forged(), object.__new__(authority.NativeExecutionPermit)):
            with self.subTest(permit_type=type(permit).__name__):
                native = recipe()
                native.update(synthetic_cpu=False, approved=True, training_authorized=True)
                native['optimizer']['capturable'] = True
                with self.assertRaises(ValueError):
                    ce_update.boundary_update(Trap(), Trap(), rows(), native, Trap(), native_permit=permit)
                with self.assertRaises(ValueError):
                    ce_journal.CeJournal(Trap(), Trap(), Trap(), Trap(), synthetic_cpu=False, native_permit=permit)
                bindings = {'schema': update.DONOR_SCHEMA, 'synthetic_cpu': False,
                            'approved': True, 'training_authorized': True}
                with self.assertRaises(ValueError):
                    update.extract_donor(Trap(), Trap(), Trap(), Trap(), Trap(), bindings, native_permit=permit)
        self.assertEqual({name for name in ('torch', 'numpy', 'transformers', 'safetensors') if name in sys.modules}, before)
        self.assertFalse(authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)


class NativeJournalStateRuleTests(unittest.TestCase):
    """Pure unconstructed state-machine fixtures; no permit or native constructor.

    Only protected planning seams run. They perform no work/callback dispatch,
    ledger writes, framework imports or native authority registration.
    """
    def fixture(self, *, native_branch=True):
        journal = object.__new__(ce_journal.CeJournal)
        journal.synthetic_cpu = not native_branch
        journal.allowed_kinds = ce_journal.NATIVE_KINDS
        journal.current_invocation = 'fixture-worker'
        journal.unknown_wall = journal.fatal = journal.restore_pending = False
        journal.requires_restore = True
        journal.active_attempt = None
        journal.operations, journal.events, journal.metadata_ids = {}, [], set()
        journal.progress = {'C': 0, 'U': 0, 'history': [], 'generation': 0, 'snapshot': 'a' * 64}
        journal.plan_rows = [['r%d' % i for i in range(8)]]
        journal.static_recipe_sha256 = 'd' * 64
        journal.bindings = {'anchors': {'own_start_sha256': 'a' * 64}}
        journal.profiles = {kind: {'read_bytes_upper': 0, 'write_bytes_upper': 0,
            'identity_hash_attempts_upper': 0, 'identity_hash_bytes_upper': 0, 'wall_s_upper': 10}
            for kind in journal.allowed_kinds}
        return journal
    def operation(self, journal, kind, phase='started', oid=None, **fields):
        payload = {'operation': kind, 'phase': phase, 'id': oid or 'fixture/' + kind, 'fields': fields}
        changes, action = journal._prepare_operation(payload)
        action()
        return changes

    def test_restore_required_allows_only_native_bootstrap_and_existing_restore_work(self):
        for pending in (False, True):
            for kind in ce_journal.NATIVE_RESUME_BOOTSTRAP:
                with self.subTest(pending=pending, operation=kind):
                    journal = self.fixture(); journal.restore_pending = pending
                    changes = self.operation(journal, kind)
                    self.assertEqual(changes['start:' + kind], 1)
                    self.assertTrue(journal.requires_restore)
                    self.assertIsNone(journal.active_attempt)
        for kind in ce_journal.NATIVE_RESUME_BOOTSTRAP:
            with self.subTest(cpu_operation=kind):
                with self.assertRaisesRegex(ValueError, 'restore required'):
                    self.operation(self.fixture(native_branch=False), kind)

    def test_bootstrap_exception_never_admits_ce_save_or_donor_before_restore(self):
        for kind in ('checkpoint_save', 'donor_reference', 'donor_write', 'donor_roundtrip'):
            with self.subTest(operation=kind), self.assertRaises(ValueError):
                self.operation(self.fixture(), kind)
        for kind, fields in (('fingerprint', {'reason': 'ce_pre_boundary_identity'}),
                             ('loss_forward', {'boundary_id': 0}), ('optimizer', {'boundary_id': 0})):
            with self.subTest(operation=kind), self.assertRaises(ValueError):
                self.operation(self.fixture(), kind, **fields)

    def test_lifetime_cleanup_returns_after_commit_or_restore_fence_with_same_worker(self):
        for generation in (0, 1):
            with self.subTest(generation=generation):
                journal = self.fixture()
                self.operation(journal, 'cleanup')
                journal.progress.update(C=1, U=1, generation=generation, snapshot='b' * 64)
                self.operation(journal, 'cleanup', 'returned', wall_s=0.25)
                self.assertEqual(journal.operations['fixture/cleanup']['phase'], 'returned')
                self.assertTrue(journal.requires_restore)

    def test_lifetime_cleanup_exception_keeps_worker_identity_and_profile_guards(self):
        journal = self.fixture(); self.operation(journal, 'cleanup')
        journal.progress.update(generation=1, snapshot='b' * 64)
        journal.current_invocation = 'other-worker'
        with self.assertRaisesRegex(ValueError, 'current-worker'):
            self.operation(journal, 'cleanup', 'returned', wall_s=0.25)
        journal.current_invocation = 'fixture-worker'
        with self.assertRaisesRegex(ValueError, 'exceeds reservation'):
            self.operation(journal, 'cleanup', 'returned', bytes_read=1)
        self.assertEqual(journal.operations['fixture/cleanup']['phase'], 'started')

    def test_scoped_gradient_cleanup_still_rejects_changed_restore_fence(self):
        journal = self.fixture()
        journal.active_attempt = {'id': 'ce-attempt', 'failed': True, 'cleanup_started': False}
        self.operation(journal, 'gradient_cleanup', boundary_id=0, generation=0,
                       symbolic_snapshot='a' * 64, start_policy_id='a' * 64, static_recipe_sha256='d' * 64)
        journal.progress.update(generation=1, snapshot='b' * 64)
        with self.assertRaisesRegex(ValueError, 'stale operation return'):
            self.operation(journal, 'gradient_cleanup', 'returned', wall_s=0.25)

    def test_native_complete_waits_for_current_lifetime_cleanup_but_ignores_old_worker_pending(self):
        journal = self.fixture()
        self.operation(journal, 'cleanup')
        journal.progress.update(C=1, U=1, generation=1, snapshot='b' * 64)
        journal.requires_restore = False
        journal.invocations = {'fixture-worker': {'ended': False}, 'old-worker': {'ended': True}}
        journal.operations['old/pending'] = {'phase': 'started', 'invocation': 'old-worker',
                                           'generation': 0, 'snapshot': 'a' * 64, 'attempt': None}
        ending = {'id': 'fixture-worker', 'outcome': 'COMPLETE'}
        with self.assertRaisesRegex(ValueError, 'complete consistent'):
            journal._prepare('invocation_end', ending)
        self.assertFalse(journal.invocations['fixture-worker']['ended'])
        self.assertEqual(journal.current_invocation, 'fixture-worker')
        self.operation(journal, 'cleanup', 'returned', wall_s=0.25)
        changes, action = journal._prepare('invocation_end', ending)
        self.assertEqual(changes, {})
        action()
        self.assertTrue(journal.invocations['fixture-worker']['ended'])
        self.assertEqual(journal.invocations['fixture-worker']['outcome'], 'COMPLETE')
        self.assertIsNone(journal.current_invocation)
        self.assertEqual(journal.operations['old/pending']['phase'], 'started')

    def test_metadata_roots_preserve_cpu_owner_and_explicit_native_checkpoint_sibling(self):
        root = Path(tempfile.gettempdir()).resolve() / 'ce-native-state-fixture-output'
        journal = self.fixture()
        journal.directory = root / 'ledger'
        payload = {'class': 'checkpoint_json', 'bytes_upper': 100,
                   'write_id': 'fixture/checkpoint', 'path': str(root / 'checkpoints' / 'binding.json')}
        journal.metadata_roots = [journal.directory]
        with self.assertRaisesRegex(ValueError, 'owned metadata'):
            journal._prepare('metadata', payload)
        journal.metadata_roots = [root]
        changes, action = journal._prepare('metadata', payload); action()
        self.assertEqual(changes, {'metadata_reserved_bytes': 100, 'io_write_bytes': 100})
        self.assertIn(payload['write_id'], journal.metadata_ids)
        for path in (root, root.parent / 'outside-native-owner' / 'binding.json'):
            with self.subTest(path=str(path)), self.assertRaisesRegex(ValueError, 'owned metadata'):
                journal._prepare('metadata', dict(payload, write_id='fixture/outside', path=str(path)))


if __name__ == '__main__': unittest.main()
