"""Framework/OS-free source adapter contracts and actual tiny metadata files."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_scope_adapters as a


class Trap:
    def __getattribute__(self, key): raise AssertionError('unadmitted service/source access')


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='scope-adapters-pure-'); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.events = []
        self.operation = lambda *args, **fields: self.events.append((args, fields))
    def ref(self, path):
        blob = path.read_bytes(); return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    def write(self, path, value): path.write_bytes(a.raw.canonical(value) + b'\n'); return self.ref(path)
    def test_native_source_roles_deny_before_os_files_or_opaque_object_traversal(self):
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        actions = (lambda: a.bootstrap_outer(Trap(), Trap(), Trap()), lambda: a.outer_issuer(Trap(), Trap(), Trap()),
            lambda: a.worker_observation(Trap(), Trap()), lambda: a.outer_verifier(Trap(), 'a' * 64, 'b' * 64, 'qualification'),
            lambda: a.outer_reservation(Trap(), 'qualification_metadata', {}),
            lambda: a.worker_metadata_verifier(Trap(), 'a' * 64, 'b' * 64, 'c' * 64),
            lambda: a.native_authority_verifier(Trap(), 'a' * 64, 'b' * 64, 'c' * 64, 'qualification', 'canonical'),
            lambda: a.bootstrap_native_issuer(Trap(), None, Trap()),
            lambda: a.bootstrap_prepayment_verifier(Trap(), Trap(), Trap(), Trap()))
        with patch.object(Path, 'open', side_effect=AssertionError('unadmitted file')), patch.object(a.worker, '_WindowsOS', side_effect=AssertionError('unadmitted OS')):
            for action in actions:
                with self.assertRaises(ValueError): action()
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})
        self.assertFalse(a.services._REGISTERED_SERVICE_VERIFIERS); self.assertFalse(a.controller._REGISTERED_OUTER_AUTHORITY_VERIFIERS)
        self.assertFalse(a.authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS); self.assertFalse(a.worker._REGISTERED_METADATA_VERIFIERS)
    def test_raw_request_hash_differs_from_auth_excluding_logical_preimage(self):
        request = {'schema': a.controller.SCHEMA, 'phase': 'qualification', 'arm': 'canonical',
            'scope_authorization_ref': {'path': str(self.root / 'auth.json'), 'sha256': 'b' * 64}}
        reference = self.write(self.root / 'request.json', request)
        result = a.read_request_metadata(request, reference, self.operation)
        self.assertEqual(result['request_preimage_sha256'], a.digest(a.controller.request_preimage(request)))
        self.assertNotEqual(result['request_preimage_sha256'], reference['sha256']); self.assertFalse(result['native_qualified'])
        wrong = copy.deepcopy(request); wrong['arm'] = 'permuted'
        with self.assertRaisesRegex(ValueError, 'differs from actual service'): a.read_request_metadata(wrong, reference, self.operation)
    def policy(self):
        module = 'experiments.skill_acquisition_native_v1.ce_parent_controller'; name = module.replace('.', '/') + '.py'
        ref = lambda path: {'path': str(path), 'sha256': 'a' * 64, 'bytes': 10}
        image = ref(self.root / 'base-python.exe'); launcher = ref(self.root / 'venv-python.exe')
        source = {name: ref(self.root / name)}
        worker = {'python_ref': launcher, 'process_image_ref': image, 'request_ref': ref(self.root / 'request.json'),
            'source_checkout': str(self.root), 'source_files': source, 'approved_code_sha256': {name: 'a' * 64},
            'entry_module': module, 'argv': [image['path'], '-m', module, '--request', str(self.root / 'request.json')],
            'launch_not_before_filetime_100ns': 100, 'launch_not_after_filetime_100ns': 120,
            'bounds': {'operation_attempts_upper': 100, 'io_read_bytes_upper': 100000,
                'identity_hash_attempts_upper': 100, 'identity_hash_bytes_upper': 100000, 'wall_seconds_upper': 60}}
        context = {'request_sha256': 'b' * 64, 'scope_sha256': 'c' * 64, 'bindings': {'source': {'code_sha256': {name: 'a' * 64}},
            'scope': {'native_backend': {'source_checkout': str(self.root)}}}}
        facts = {'pid': 123, 'creation_filetime_100ns': 110, 'signaled': False, 'exit_code': 259,
            'image_path': image['path'], 'argv': worker['argv']}
        return context, {'worker': worker}, facts
    def test_current_worker_scope_binds_dual_python_whole_argv_creation_boot_and_code(self):
        context, policy, facts = self.policy()
        value = a.worker_specification(context, policy, facts, 'windows-boot-guid-fixture')
        self.assertEqual(value['expected_pid'], 123); self.assertEqual(value['expected_creation_filetime_100ns'], 110)
        self.assertEqual(value['python_ref'], policy['worker']['python_ref']); self.assertEqual(value['process_image_ref'], policy['worker']['process_image_ref'])
        self.assertEqual(value['request_sha256'], context['request_sha256']); self.assertEqual(value['scope_sha256'], context['scope_sha256'])
        self.assertNotIn('native_execution_permit', value)
        for change in (lambda f: f.update(creation_filetime_100ns=121), lambda f: f.update(pid=True),
                       lambda f: f.update(signaled=True), lambda f: f.update(argv=['wrong']), lambda f: f.update(image_path=str(self.root / 'foreign.exe'))):
            bad = copy.deepcopy(facts); change(bad)
            with self.assertRaises(ValueError): a.worker_specification(context, policy, bad, 'windows-boot-guid-fixture')
    def test_source_origin_or_derived_identity_cannot_be_supplied_by_worker_policy(self):
        context, policy, facts = self.policy(); policy['worker']['expected_pid'] = 999
        with self.assertRaisesRegex(ValueError, 'excludes derived'): a.worker_specification(context, policy, facts, 'windows-boot-guid-fixture')
        context, policy, facts = self.policy(); context['bindings']['source']['code_sha256'] = {'foreign.py': 'f' * 64}
        with self.assertRaisesRegex(ValueError, 'source origins'): a.worker_specification(context, policy, facts, 'windows-boot-guid-fixture')
    def capsule_fixture(self):
        context = {'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64, 'static_recipe_sha256': 'c' * 64,
            'worker_proof_sha256': 'd' * 64, 'bindings': {'source': {'code_sha256': {'fixture.py': 'e' * 64}}}}
        common = {key: context[key] for key in ('request_sha256', 'scope_sha256', 'static_recipe_sha256', 'worker_proof_sha256')}
        common['code_sha256'] = context['bindings']['source']['code_sha256']
        bounds = {key: 100 for key in a.services.COUNTERS}
        reserved = {'schema': 'CE8-SERVICE-CONSTRUCTOR-PREPAYMENT-1', 'request_sha256': context['request_sha256'],
            'service_spec_sha256': '1' * 64, 'bootstrap_spec_sha256': '2' * 64, 'bounds': bounds,
            'status': 'PERMANENTLY_RESERVED', 'metadata_only': True, 'native_qualified': False, 'execution_verified': False}
        payment = {key: reserved[key] for key in ('schema', 'request_sha256', 'service_spec_sha256', 'bootstrap_spec_sha256', 'bounds',
            'metadata_only', 'native_qualified', 'execution_verified')}
        payment.update(consumed=True, reservation_ref=self.write(self.root / 'original-reservation.json', reserved))
        common['constructor_prepayment'] = payment
        snapshot = {'schema': a.services.BOOTSTRAP, 'request_sha256': context['request_sha256'], 'fatal': False,
            'native_qualified': False, 'head': '3' * 64, 'cumulative': {key: 100 for key in a.services.COUNTERS}}
        source = dict(common, schema=a.CAPSULE, service_bootstrap_schema=a.services.BOOTSTRAP, service_bootstrap_snapshot=snapshot)
        setup = dict(common, schema=a.SETUP)
        refs = {'ingestion_bootstrap_ref': a._write_capsule(self.root / 'source.json', source, self.operation),
            'outer_setup_reservation_ref': a._write_capsule(self.root / 'setup.json', setup, self.operation)}
        context['qualifier_setup_refs'] = refs
        return context, source
    def test_actual_setup_capsules_read_back_and_reject_foreign_rehashed_linkage(self):
        context, source = self.capsule_fixture()
        self.assertEqual(a.read_setup_capsules(context, self.operation)['ingestion_bootstrap_ref'], source)
        source['worker_proof_sha256'] = 'f' * 64
        context['qualifier_setup_refs']['ingestion_bootstrap_ref'] = self.write(self.root / 'source.json', source)
        with self.assertRaisesRegex(ValueError, 'scope/source/worker linkage'): a.read_setup_capsules(context, self.operation)
    def test_capsule_publication_admission_and_create_only_failure_retains_original(self):
        path = self.root / 'capsule.json'
        def denied(*args, **kwargs): raise ValueError('metadata prepayment rejected')
        with self.assertRaisesRegex(ValueError, 'prepayment rejected'): a._write_capsule(path, {'fixture': True}, denied)
        self.assertFalse(path.exists())
        a._write_capsule(path, {'fixture': True}, self.operation); original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'create-only'): a._write_capsule(path, {'replace': True}, self.operation)
        self.assertEqual(path.read_bytes(), original)
    def test_native_permit_revalidation_denies_before_any_priced_os_refresh(self):
        owner = object.__new__(a.services.NativeControllerServices)
        owner.permit, owner.request, owner.spec = None, {}, {}
        with patch.object(a.worker, 'verify_worker_handle', side_effect=AssertionError('recursive OS refresh')):
            with self.assertRaises(ValueError): a.native_authority_verifier(owner, 'a' * 64, 'b' * 64, 'c' * 64, 'qualification', 'canonical')

    def test_runtime_setup_refs_derive_owner_without_scope_selfreference_and_survive_child(self):
        from . import test_ce_qualification_accounting as qa
        context, proof, receipt = qa.fixture(self.root)
        context['bindings']['scope']['qualifier_accounting'].pop('ingestion_bootstrap_ref')
        context['bindings']['scope']['checkpoint_qualification'] = {'owned_directory': str(self.root / 'checkpoints')}
        before = copy.deepcopy(context['bindings']['scope'])
        source, _ = self.capsule_fixture(); refs = source['qualifier_setup_refs']
        context['qualifier_setup_refs'] = refs
        derived = a.accounting.accounting_specification(context)
        self.assertEqual(context['bindings']['scope'], before)
        self.assertEqual(derived['ingestion_bootstrap_ref'], refs['ingestion_bootstrap_ref'])
        self.assertEqual(derived['outer_setup_reservation_ref'], refs['outer_setup_reservation_ref'])
        context['checkpoint_child_declaration'] = {'branch': 'prefix'}
        child = dict(derived, owned_directory=str(self.root / 'checkpoints' / 'children' / 'prefix' / 'ledger'))
        context['checkpoint_child_accounting'] = {'specification': child, 'owner': {
            'accounting_spec_sha256': a.digest(child), 'prepared_receipt_sha256': receipt['receipt_sha256']}}
        self.assertEqual(a.accounting.accounting_specification(context), child)
        context['qualifier_setup_refs']['extra'] = refs['ingestion_bootstrap_ref']
        with self.assertRaisesRegex(ValueError, 'exact two'): a.accounting.accounting_specification(context)


if __name__ == '__main__': unittest.main()
