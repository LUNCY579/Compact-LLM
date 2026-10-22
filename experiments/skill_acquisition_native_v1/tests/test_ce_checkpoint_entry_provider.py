"""Actual paid small-file metadata fixtures; no OS, frameworks, permits or grants."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_entry_provider as entry
from . import test_ce_checkpoint_local_issuer as local_fixture, test_ce_controller as controller_fixture
from . import test_ce_process_launch as launch_fixture


def fixture(root):
    common, request, reference, proof, _, _ = local_fixture.fixture(root)
    controller_request = controller_fixture.request()
    controller_request.update(mode='registered_scope', approved=True, qualification_authorized=True,
                              training_authorized=False, status='AUTHORIZED_QUALIFICATION_ONLY')
    study = entry.digest(entry.controller.request_preimage(controller_request))
    common['request_sha256'] = request['study_request_sha256'] = study
    proof['request_sha256'] = study
    branch = root / 'checkpoint' / 'children' / 'uninterrupted'; branch.mkdir(parents=True)
    raw_request_path = branch / 'REQUEST.json'
    names = [entry.child.__name__.replace('.', '/') + '.py', entry.__name__.replace('.', '/') + '.py', 'fixture_factories.py']
    refs = {}
    for name in names:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); content = b'# metadata fixture\n'; path.write_bytes(content)
        refs[name] = {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    code = {name: ref['sha256'] for name, ref in refs.items()}
    common['bindings']['source']['code_sha256'] = request['code_sha256'] = code
    common['prepared_receipt'] = local_fixture.qa.fixture(root)[2]
    common['bindings']['scope']['checkpoint_qualification']['local_issuer'] = local_fixture.policy()
    common_content = entry.canonical(common) + b'\n'; common_path = root / 'COMMON_STUDY_CONTEXT.json'; common_path.write_bytes(common_content)
    common_ref = {'path': str(common_path), 'sha256': hashlib.sha256(common_content).hexdigest(), 'bytes': len(common_content)}
    request['common_context_ref'] = common_ref
    content = entry.canonical(request) + b'\n'; raw_request_path.write_bytes(content)
    reference = {'path': str(raw_request_path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    for name in ('python.exe', 'image.exe'):
        (root / name).write_bytes(b'pinned fixture executable')
    def ref(name):
        content = (root / name).read_bytes()
        return {'path': str(root / name), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    launch = launch_fixture.spec()
    launch.update(mode='registered_native_worker', request_sha256=study, scope_sha256=common['scope_sha256'],
        cwd=str(root), source_checkout=str(root), worker_module=entry.child.__name__, request_ref=reference,
        source_files=refs, approved_code_sha256=code, python_ref=ref('python.exe'), process_image_ref=ref('image.exe'))
    launch['argv'] = [launch['python_ref']['path'], '-m', entry.child.__name__, '--request', str(raw_request_path)]
    profile = {'read_bytes_upper': 8 * 1024 ** 2, 'write_bytes_upper': 8 * 1024 ** 2,
        'identity_hash_attempts_upper': 1000, 'identity_hash_bytes_upper': 64 * entry.local.raw.MAX_JSON, 'wall_s_upper': 1000}
    bootstrap = {'owned_directory': str(branch / 'service-bootstrap'),
        'caps': dict(read_bytes=10**10, write_bytes=10**10, hash_attempts=100000, hash_bytes=10**11,
                     metadata_bytes=10**10, operation_attempts=10000, wall_seconds=100000),
        'setup_bounds': profile, 'operation_profiles': {kind: copy.deepcopy(profile) for kind in
            ('input_io', 'entry_lifetime', 'entry_worker_binding', 'worker_file_read')}}
    configuration = {'schema': entry.SPEC, 'launch_spec': launch, 'controller_request': controller_request,
        'service_spec': {'approved_code_sha256': code, 'bootstrap': bootstrap}, 'common_context_ref': common_ref,
        'observation_bounds': {'operation_attempts_upper': 1000, 'io_read_bytes_upper': 10**8,
            'identity_hash_attempts_upper': 1000, 'identity_hash_bytes_upper': 10**8, 'wall_seconds_upper': 1000},
        'entry_factories': {name: {'module': 'fixture_factories', 'symbol': name, 'source_sha256': code['fixture_factories.py']}
                            for name in ('prepared_rows', 'prefix_terminal_handle')}}
    proof.update(sources=dict(proof['sources'], request_ref=reference, source_files=refs))
    ledger = entry.services.BootstrapLedger(bootstrap, study, synthetic_fixture=True)
    return configuration, common, request, proof, ledger


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='checkpoint-entry-metadata-pure-'); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.configuration, self.common, self.request, self.proof, self.ledger = fixture(self.root)

    def test_native_provider_and_adapter_deny_before_path_configuration_os_or_permit_access(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('pre-authority traversal')
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('pre-authority IO')):
            with self.assertRaisesRegex(ValueError, 'provider is not selected'):
                entry.CheckpointEntryProvider().open_prepaid_entry(Trap())
            with self.assertRaisesRegex(ValueError, 'metadata services owner'):
                entry.MetadataEntryPayment(Trap(), Trap())
        self.assertIsNone(entry.local._SOURCE_ENTRY_PROVIDER)
        self.assertIsNone(entry._SOURCE_CONFIGURATION)
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})

    def test_real_prepaid_request_common_reads_keep_costs_and_no_native_authority(self):
        entry.validate_configuration(self.configuration)
        result = entry.fixture_read_inputs({'schema': entry.FIXTURE, 'native_qualified': False, 'execution_verified': False}, self.configuration, self.ledger)
        self.assertEqual(result['request'], self.request); self.assertEqual(result['common'], self.common)
        self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])
        self.assertGreater(result['bootstrap_snapshot']['cumulative']['read_bytes'], 0)
        self.assertEqual([op['phase'] for op in self.ledger.operations.values()], ['returned', 'returned'])

    def test_hash_drift_foreign_common_owner_and_census_refuse_without_refunds(self):
        path = Path(self.configuration['launch_spec']['request_ref']['path'])
        content = path.read_bytes(); path.write_bytes(content.replace(b'uninterrupted', b'uninterrupteX', 1))
        before = dict(self.ledger.counters)
        with self.assertRaisesRegex(ValueError, 'bytes/hash drift'):
            entry.read_static_inputs(self.configuration, self.ledger.operation)
        self.assertGreater(self.ledger.counters['read_bytes'], before['read_bytes'])
        self.assertEqual(next(iter(self.ledger.operations.values()))['phase'], 'failed')
        for change in ('source', 'receipt', 'owner'):
            config, request, common = (copy.deepcopy(value) for value in (self.configuration, self.request, self.common))
            if change == 'source': common['bindings']['source']['code_sha256']['fixture_factories.py'] = '0' * 64
            elif change == 'receipt': common['prepared_receipt']['model_loads'] = 10
            else: config['service_spec']['bootstrap']['owned_directory'] = str(self.root / 'foreign')
            with self.subTest(change=change), self.assertRaises(ValueError): entry.validate_common(config, request, common)

    def test_finite_lifetime_binding_failure_and_completion_are_durable_without_native_grant(self):
        payment = entry.MetadataEntryPayment.fixture(self.ledger, self.configuration)
        payment.bind_fixture(self.common, self.proof); payment.begin()
        charged = dict(self.ledger.counters)
        payment.fail(ValueError('missing actual preparation issuer'))
        self.assertEqual(payment.status, 'FAILED')
        self.assertEqual(self.ledger.operations[payment.identifier]['phase'], 'failed')
        self.assertGreaterEqual(self.ledger.counters['hash_bytes'], charged['hash_bytes'])
        self.assertFalse(payment.evidence['native']); self.assertFalse(payment.evidence['native_qualified'])
        with self.assertRaisesRegex(ValueError, 'fixture entry payment'): payment.verify({}, {}, {})
        with self.assertRaises(ValueError): payment.complete()

    def test_duplicate_begin_completion_and_binding_denied_before_invalid_append(self):
        payment = entry.MetadataEntryPayment.fixture(self.ledger, self.configuration)
        payment.bind_fixture(self.common, self.proof); payment.begin()
        before = self.ledger.path.read_bytes()
        with self.assertRaises(ValueError): payment.begin()
        self.assertEqual(self.ledger.path.read_bytes(), before)
        payment.complete(); before = self.ledger.path.read_bytes()
        with self.assertRaises(ValueError): payment.complete()
        self.assertEqual(self.ledger.path.read_bytes(), before)
        self.assertEqual(self.ledger.operations[payment.identifier]['phase'], 'returned')

    def test_entry_lifetime_cap_denies_before_source_read_and_keeps_earlier_setup(self):
        self.ledger.spec['caps']['wall_seconds'] = self.ledger.counters['wall_seconds']
        self.configuration['service_spec']['bootstrap']['caps']['wall_seconds'] = self.ledger.counters['wall_seconds']
        setup = self.ledger.path.read_bytes()
        with patch.object(Path, 'open', side_effect=AssertionError('unpaid file work')):
            with self.assertRaisesRegex(ValueError, 'cap exhausted'):
                entry.MetadataEntryPayment.fixture(self.ledger, self.configuration)
        self.assertEqual(self.ledger.path.read_bytes(), setup)

    def test_static_callback_descriptors_and_branch_scope_changes_cannot_select_other_authority(self):
        for change in ('callback', 'hash', 'request'):
            config = copy.deepcopy(self.configuration)
            if change == 'callback': config['entry_factories']['prepared_rows'] = lambda: True
            elif change == 'hash': config['entry_factories']['prepared_rows']['source_sha256'] = '0' * 64
            else: config['controller_request']['arm'] = 'permuted'
            with self.subTest(change=change), self.assertRaises(ValueError): entry.validate_configuration(config)
        with self.assertRaises(entry.services.MissingServiceRole):
            entry.services.resolve_role(self.configuration['entry_factories']['prepared_rows'], self.configuration['launch_spec']['approved_code_sha256'])

    def test_original_metadata_owner_prices_ref_and_fixture_truthiness_cannot_change(self):
        payment = entry.MetadataEntryPayment.fixture(self.ledger, self.configuration)
        payment.bind_fixture(self.common, self.proof)
        original = self.ledger.path.read_bytes()
        self.ledger.spec['caps']['read_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'payment/bootstrap owner'): payment.begin()
        self.assertEqual(self.ledger.path.read_bytes(), original)
        class Trap:
            def __getattribute__(self, key): raise AssertionError('fixture evidence traversed as native')
        with self.assertRaisesRegex(ValueError, 'fixture entry payment'): payment.verify(Trap(), Trap(), Trap())

    def test_owned_metadata_close_attempts_physical_release_when_price_denied(self):
        calls = []
        original = ValueError('close price denied')
        class API:
            def close(self, handle): calls.append(handle)
        class Budget:
            def run(self, kind, work): raise original
        with self.assertRaises(ValueError) as caught: entry._close_observed(API(), 'owned fixture handle', Budget())
        self.assertIs(caught.exception, original)
        self.assertEqual(calls, ['owned fixture handle'])

    def test_current_and_redirector_handles_are_owned_before_post_open_observer_failure(self):
        for failed_pid in (41, 40):
            calls = []; original = RuntimeError('open returned observer failed')
            spec = self.configuration['launch_spec']
            class API:
                just_opened = None
                def open(self, pid): self.just_opened = pid; calls.append(('open', pid)); return pid
                def close(self, handle): calls.append(('close', handle))
                def boot(self): return spec['parent_identity']['boot_identity']
                def snapshot(self, handle):
                    return {'pid': handle, 'signaled': False, 'exit_code': 259, 'creation_filetime_100ns': 140,
                        'image_path': spec['process_image_ref']['path'], 'argv': [spec['process_image_ref']['path'], *spec['argv'][1:]]}
            api = API()
            def observe(kind, phase, **fields):
                if phase == 'returned' and api.just_opened is not None:
                    pid = api.just_opened; api.just_opened = None
                    if pid == failed_pid: raise original
            with self.subTest(pid=failed_pid), patch.object(entry.worker, '_WindowsOS', return_value=api), \
                    patch.object(entry.os, 'getpid', return_value=41), patch.object(entry.launch, '_parent_pid', return_value=40):
                with self.assertRaises(RuntimeError) as caught: entry.current_observation_spec(self.configuration, observe)
            self.assertIs(caught.exception, original)
            self.assertEqual([pid for action, pid in calls if action == 'close'], [41] if failed_pid == 41 else [40, 41])

    def test_redirector_query_error_survives_close_observer_failure(self):
        original, cleanup = ValueError('redirector query failed'), RuntimeError('redirector close receipt failed')
        spec = self.configuration['launch_spec']; closed = []
        class API:
            def open(self, pid): return pid
            def close(self, handle): closed.append(handle)
            def boot(self): return spec['parent_identity']['boot_identity']
            def snapshot(self, handle):
                if handle == 40: raise original
                return {'pid': 41, 'signaled': False, 'exit_code': 259, 'creation_filetime_100ns': 140,
                    'image_path': spec['process_image_ref']['path'], 'argv': [spec['process_image_ref']['path'], *spec['argv'][1:]]}
        def observe(kind, phase, **fields):
            if kind == 'worker_handle_close' and phase == 'returned' and closed == [40]: raise cleanup
        with patch.object(entry.worker, '_WindowsOS', return_value=API()), patch.object(entry.os, 'getpid', return_value=41), \
                patch.object(entry.launch, '_parent_pid', return_value=40):
            with self.assertRaises(ValueError) as caught: entry.current_observation_spec(self.configuration, observe)
        self.assertIs(caught.exception, original); self.assertIs(original.__cause__, cleanup)
        self.assertEqual(closed, [40, 41])
        self.assertEqual(original.entry_redirector_close_failure['message'], 'redirector close receipt failed')

    def test_post_payment_cli_validation_fails_lifetime_before_os_or_source_work(self):
        payment = entry.MetadataEntryPayment.fixture(self.ledger, self.configuration)
        runtime = SimpleNamespace(bootstrap=self.ledger)
        with patch.object(entry.worker, '_WindowsOS', side_effect=AssertionError('unapproved OS after invalid CLI')):
            with self.assertRaisesRegex(ValueError, 'CLI path cannot select'):
                entry.CheckpointEntryProvider()._open_after_payment(self.configuration, runtime, payment, str(self.root / 'foreign.json'))
        self.assertEqual(payment.status, 'FAILED')
        self.assertEqual(self.ledger.operations[payment.identifier]['phase'], 'failed')

    def test_post_provider_validation_failure_closes_owned_handles_and_fails_payment(self):
        payment = entry.MetadataEntryPayment.fixture(self.ledger, self.configuration)
        payment.bind_fixture(self.common, self.proof)
        handle = object.__new__(entry.worker.WinWorkerHandle)
        observed = {'entry_prepayment': payment, 'worker_handle': handle, 'observation_permit': None}
        with patch.object(entry.worker.WinWorkerHandle, 'close') as close:
            with self.assertRaisesRegex(ValueError, 'current OS observation entry'):
                entry.local.LocalCheckpointIssuer()._consume_entry(observed, 'unused', entry.child.CheckpointChildHandler, entry.child.validate_command)
        close.assert_called_once_with()
        self.assertEqual(payment.status, 'FAILED')
        self.assertEqual(self.ledger.operations[payment.identifier]['phase'], 'failed')


if __name__ == '__main__': unittest.main()
