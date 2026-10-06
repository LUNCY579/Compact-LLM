"""Trusted main/installer/pipeline handoff, entirely inert and CPU-only.

Selected documents/source hashes/rosters are read. Historical host/Git/layout
and library observations are injected. Already-loaded stdlib exports have no
fresh-bootstrap or namespace-execution qualification. Same-process arbitrary
memory mutation is outside this approved control-flow test's claim.
"""
from pathlib import Path
import builtins, copy, contextlib, sys, time, unittest
from unittest.mock import patch
import test_library_boundary_manifest as manifest
from test_full_native_cpu_workflow import compiled
from test_worker_bindings import InertSuppliers


class OwnedLibraryEntryControls(unittest.TestCase):
    def setUp(self):
        self.fixture_case = manifest.LibraryBoundaryManifestControls()
        self.fixture_case.setUp(); self.addCleanup(self.fixture_case.doCleanups)
        m = self.fixture_case
        self.fixture = m.fixture; self.C = m.C; self.W = m.worker; self.L = m.loader
        self.cap = m.selector_capability
        self.output = self.fixture.package/'artifacts/prefix_input_owned_run_CPU'; self.output.mkdir()
        self.owner = {'worker_pid':12345,'root_worker_pid':12345,'supervisor_pid':12344,
            'job_name':'Local\\SYNTH_NQ_JOB_'+'1'*32,'supervisor_create_time':1000.25,
            'config_sha256':self.fixture.anchors['expected_config_file']}
        self.inert = InertSuppliers(self.fixture, self.owner)
        support = compiled(manifest.HERE/'fixture_support.py')
        fake = support.F.FakeTorch(support.F.Recorder())
        fake.__version__ = self.fixture.config['torch_version']; fake.backends = self.inert.torch.backends
        for name in ('set_float32_matmul_precision','use_deterministic_algorithms','set_num_threads'):
            setattr(fake, name, getattr(self.inert.torch, name))
        self.inert.torch = fake; self.host = support.FakeHost(self.output)
        self.host.offset = 0; self.host.started = time.monotonic(); self.host.assigned = self.host.resumed = True
        self.host.job_name = self.owner['job_name']; self.host.supervisor_pid = self.owner['supervisor_pid']
        self.environment = {}; self.installed = None
        original = self.inert.framework
        def framework():
            self.assertEqual(self.installed.library_guard._depth, 1)
            self.library_supplier.add(m.extension)
            return original()
        self.inert.framework = framework
        self.attempted_native = []; original_import = builtins.__import__
        def imports(name, *args, **kwargs):
            if name.split('.', 1)[0] in {'native_worker','native_process','worker_ram_probe'}:
                self.attempted_native.append(name); raise AssertionError('actual native helper import forbidden')
            return original_import(name, *args, **kwargs)
        self.import_patch = patch.object(builtins, '__import__', side_effect=imports)
        self.import_patch.start(); self.addCleanup(self.import_patch.stop)
        self.addCleanup(lambda:self.assertEqual(self.attempted_native, []))

    def prepare(self):
        self.context = self.fixture.context()
        self.library_supplier = manifest.Supplier(self.fixture_case.selected())
        for name, value in (('supervisor_ownership.json', self.owner),
                ('OWNED_EXTERNAL_ANCHORS.json', {'config_path':str(self.fixture.config_path),'anchors':self.fixture.anchors})):
            (self.output/name).write_bytes(self.fixture._raw(value))

    def install(self):
        finder = self.L._CPU_install_fixture(self.context, self.C, _capability=self.cap,
            library_supplier=self.library_supplier)
        self.installed = finder; return finder

    def execution(self, loader=None, capability=None):
        return self.W._owned_execution_factory(self.L if loader is None else loader,
            _CPU_capability=self.cap if capability is None else capability,
            suppliers=self.inert.mapping(), host=self.host)

    def run_main(self):
        original = self.L._CPU_install_fixture
        def installed(*args, **kwargs):
            self.installed = original(*args, **kwargs); return self.installed
        with patch.object(self.L, '_CPU_install_fixture', side_effect=installed):
            return self.W._CPU_registered_main_fixture(self.context,self.owner,self.output,
                _capability=self.cap,loader=self.L,suppliers=self.inert.mapping(),host=self.host,
                environment=self.environment,library_supplier=self.library_supplier)

    def test_actual_main_installer_pipeline_retains_original_contract_context_and_worker_functions(self):
        with self.fixture_case.isolated():
            self.prepare(); baseline = list(sys.meta_path); proof = self.run_main()
            self.assertEqual(sys.meta_path, baseline)
        pipeline = proof['pipeline_proof']; worker = pipeline['worker']
        self.assertIs(pipeline['context'], proof['original_context'])
        self.assertIs(pipeline['contract'], self.C); self.assertIs(worker.C, self.C)
        self.assertIs(worker.source_core, self.W.source_core)
        self.assertIs(worker.original_guard, self.W.original_guard)
        self.assertIs(worker._CPU_real_torch_bindings, self.W._CPU_real_torch_bindings)
        self.assertTrue(pipeline['owned_entry']['original_contract_retained'])
        self.assertTrue(pipeline['owned_entry']['opaque_installer_authority_consumed'])
        self.assertTrue(pipeline['owned_entry']['library_boundary_consumed'])
        self.assertEqual(self.inert.events[:2], ['native_worker_supplier','ownership_verify'])
        self.assertEqual(self.inert.events.count('framework_supplier'), 1)
        self.assertLess(self.inert.events.index('disk_free'), self.inert.events.index('framework_supplier'))
        finder = proof['finder']; self.assertIs(finder, self.installed)
        self.assertTrue(finder.library_cleanup['fixture_only'])
        self.assertEqual(finder.library_cleanup['observed_boundaries'][-1]['boundary'], 'framework_import')
        self.assertTrue(self.library_supplier.closed)
        self.assertFalse(proof['native_PASS']); self.assertFalse(proof['native_authority_issued'])
        self.assertFalse(proof['library_terminal_evidence_join_complete'])
        container = self.C.parse((self.output/'OWNED_WORKER_RETURN.json').read_bytes())
        self.assertEqual(set(container), {'report','runtime_execution_modules','owned_membership_verified','IO','native_PASS'})
        self.assertFalse(container['native_PASS'])

    def test_byte_identical_second_loader_token_refused_by_captured_first_instance(self):
        second = compiled(self.fixture.package/'prefix_registered_loader.py')
        self.assertEqual(second.__compiled_source_descriptor__, self.L.__compiled_source_descriptor__)
        with self.fixture_case.isolated():
            self.prepare(); first = self.install(); execute = self.execution()
            self.L.remove(first)
            supplier = manifest.Supplier(self.fixture_case.selected())
            other = second._CPU_install_fixture(self.context,self.C,_capability=self.cap,library_supplier=supplier)
            try:
                with self.assertRaisesRegex(ValueError, 'actual selected installer opaque authority'):
                    execute(self.context,self.owner,self.output,other)
                self.assertEqual(self.inert.events, [])
            finally: second.remove(other)

    def test_forged_opaque_token_refused_before_suppliers(self):
        with self.fixture_case.isolated():
            self.prepare(); finder = self.install()
            try:
                with patch.object(finder, 'installed_authority', object()):
                    with self.assertRaisesRegex(ValueError, 'actual selected installer opaque authority'):
                        self.execution()(self.context,self.owner,self.output,finder)
                self.assertEqual(self.inert.events, [])
            finally:self.L.remove(finder)

    def test_wrong_context_refused_before_revalidation_or_suppliers(self):
        with self.fixture_case.isolated():
            self.prepare(); finder = self.install()
            value = self.C.context_value(self.context); value['expected_head'] = 'b'*40
            changed = self.C._mint_native_context(value)
            try:
                with self.assertRaisesRegex(ValueError, 'original installed context projection'):
                    self.execution()(changed,self.owner,self.output,finder)
                self.assertEqual(self.inert.events, [])
            finally:self.L.remove(finder)

    def test_CPU_installation_mode_cannot_be_promoted_by_flags_or_private_native_factory(self):
        with self.fixture_case.isolated():
            self.prepare(); finder = self.install(); value = self.C.context_value(self.context)
            promoted = copy.deepcopy(value); promoted['fixture_only'] = False
            changed = self.C._mint_native_context(promoted)
            try:
                with patch.object(finder, 'fixture_only', False):
                    with self.assertRaisesRegex(ValueError, 'explicit CPU installed-selector fixture only'):
                        self.execution()(changed,self.owner,self.output,finder)
                    native_execute = self.W._owned_execution_factory(self.L)
                    with patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('native suppliers forbidden')):
                        with self.assertRaisesRegex(ValueError, 'CPU installation record has no native selector authority'):
                            native_execute(self.context,self.owner,self.output,finder)
                self.assertEqual(self.inert.events, [])
            finally:self.L.remove(finder)

    def test_removed_reinserted_finder_token_remains_revoked(self):
        with self.fixture_case.isolated():
            self.prepare(); finder = self.install(); execute = self.execution()
            self.L.remove(finder); sys.meta_path.insert(0,finder)
            try:
                with self.assertRaisesRegex(ValueError, 'actual selected installer opaque authority'):
                    execute(self.context,self.owner,self.output,finder)
                self.assertEqual(self.inert.events, [])
            finally:sys.meta_path.remove(finder)

    def test_inactive_failed_closed_guard_refused_before_suppliers(self):
        with self.fixture_case.isolated():
            self.prepare(); finder = self.install()
            try:
                for name, value in (('active',False),('failed',True),('closed',True)):
                    with self.subTest(state=name), patch.object(finder.library_guard,name,value):
                        with self.assertRaisesRegex(ValueError,'active unfailed unclosed installed guard'):
                            self.execution()(self.context,self.owner,self.output,finder)
                self.assertEqual(self.inert.events, [])
            finally:self.L.remove(finder)

    def test_original_guard_or_method_swap_refused_by_record_before_suppliers(self):
        with self.fixture_case.isolated():
            self.prepare(); finder = self.install()
            try:
                with patch.object(finder,'library_guard',object()):
                    with self.assertRaisesRegex(ValueError,'original installed guard class and method'):
                        self.execution()(self.context,self.owner,self.output,finder)
                method = self.fixture_case.noop_bound_method(finder, str(self.fixture.package/'prefix_registered_loader.py'))
                with patch.object(finder,'framework_import',method):
                    with self.assertRaisesRegex(ValueError,'original installed finder class and method'):
                        self.execution()(self.context,self.owner,self.output,finder)
                self.assertEqual(self.inert.events, [])
            finally:self.L.remove(finder)

    def test_public_legacy_callbacks_and_checker_refused_before_all_suppliers(self):
        pipeline = self.W.exact_module(self.fixture.package/'prefix_native_pipeline.py')
        callback = unittest.mock.Mock(side_effect=AssertionError('untrusted callback must not execute'))
        with patch.object(self.C,'stage_native_gate'), patch.object(self.C,'context_value',return_value={'fixture_only':False}), \
                patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('native suppliers forbidden')) as suppliers, \
                patch.object(pipeline,'source_module',side_effect=AssertionError('worker/C recompile forbidden')) as sources:
            with self.assertRaisesRegex(ValueError,'native bindings require actual owned main entry'):
                self.W.real_torch_bindings(object(),{},'unused',framework_boundary=callback,checker=callback)
            with self.assertRaisesRegex(ValueError,'registered pipeline requires actual owned main entry'):
                pipeline.run_registered('unused',{},'unused',framework_boundary=callback,checker=callback)
        callback.assert_not_called(); suppliers.assert_not_called(); sources.assert_not_called()

    def test_source_drift_refused_by_actual_fresh_context_before_suppliers(self):
        with self.fixture_case.isolated():
            self.prepare(); finder = self.install()
            path = self.fixture.package/'prefix_native_pipeline.py'; raw = path.read_bytes(); path.write_bytes(b'#'+raw[1:])
            try:
                with self.assertRaisesRegex(ValueError,'registered runtime exact stable bytes'):
                    self.execution()(self.context,self.owner,self.output,finder)
                self.assertEqual(self.inert.events, [])
            finally:self.L.remove(finder)

    def test_cleanup_failure_keeps_partial_owned_return_and_revokes_handle(self):
        with self.fixture_case.isolated():
            self.prepare(); self.library_supplier.close_error = True
            with self.assertRaisesRegex(ValueError,'DLL search-scope cleanup failed'):self.run_main()
            self.assertNotIn(self.installed,sys.meta_path); self.assertTrue(self.library_supplier.closed)
            with self.assertRaisesRegex(ValueError,'actual selected installer opaque authority'):
                self.L._check_installation_authority(self.installed.installed_authority,
                    self.C.context_value(self.context),_CPU_capability=self.cap)
        value = self.C.parse((self.output/'OWNED_WORKER_RETURN.json').read_bytes())
        self.assertFalse(value['native_PASS']); self.assertIsNone(self.installed.library_cleanup)

    def test_framework_failure_is_preserved_and_failed_guard_prevents_completion(self):
        def fail():raise ValueError('injected inert framework failure')
        self.inert.framework = fail
        with self.fixture_case.isolated():
            self.prepare()
            with self.assertRaisesRegex(ValueError,'cannot publish completion'):self.run_main()
            self.assertNotIn(self.installed,sys.meta_path); self.assertTrue(self.library_supplier.closed)
        value = self.C.parse((self.output/'OWNED_WORKER_RETURN.json').read_bytes())
        self.assertEqual(value['report']['status'],'NATIVE_INPUT_CHECKS_FAILED')
        self.assertFalse(value['native_PASS']); self.assertTrue(self.installed.library_guard.failed)

    def test_primary_ownership_exception_identity_preserved_with_cleanup_failure_cause(self):
        primary=RuntimeError('injected inert primary ownership failure'); cleanup_seen=[]
        def verify(owner,config):
            self.inert.events.append('ownership_verify');raise primary
        self.inert.verify=verify;original_remove=self.L.remove
        def remove(finder):
            try:return original_remove(finder)
            except BaseException as cleanup:cleanup_seen.append(cleanup);raise
        with self.fixture_case.isolated():
            self.prepare();self.library_supplier.close_error=True
            with patch.object(self.L,'remove',side_effect=remove):
                with self.assertRaisesRegex(RuntimeError,'injected inert primary ownership failure') as caught:
                    self.run_main()
            self.assertIs(caught.exception,primary);self.assertEqual(len(cleanup_seen),1)
            self.assertIs(primary.__cause__,cleanup_seen[0])
            self.assertIsInstance(primary.__cause__,ValueError)
            self.assertIn('DLL search-scope cleanup failed',str(primary.__cause__))
            self.assertEqual(self.inert.events,['native_worker_supplier','ownership_verify'])
            self.assertEqual(self.environment,{})
            self.assertNotIn(self.installed,sys.meta_path);self.assertTrue(self.library_supplier.closed)
            with self.assertRaisesRegex(ValueError,'actual selected installer opaque authority'):
                self.L._check_installation_authority(self.installed.installed_authority,
                    self.C.context_value(self.context),_CPU_capability=self.cap)
        self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists())
        self.assertFalse((self.output/'PHASE_OBSERVATIONS.jsonl').exists())


if __name__ == '__main__':unittest.main()
