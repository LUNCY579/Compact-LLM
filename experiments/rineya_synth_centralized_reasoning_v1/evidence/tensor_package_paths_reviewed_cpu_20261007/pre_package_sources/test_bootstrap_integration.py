"""Actual registered-source CPU install/import/remove with authored startup."""
import builtins,contextlib,copy,hashlib,importlib,importlib.machinery,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'tensor_bootstrap_bridge_work'
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(BASE/'test_finite_bootstrap_helper.py','CPU_prior_fixture');H=load(HERE/'finite_bootstrap_helper.py','CPU_integrated_helper');B.H=H
class Controls(B.Controls):
    def setUp(self):
        super().setUp();self.C=load(HERE/'prefix_tensor_contract.py','CPU_contract');self.L=load(HERE/'prefix_registered_loader.py','CPU_loader');self.capC=self.C._mint_CPU_native_context_fixture()
        self.layout={**self.C.QUALIFIED_LAYOUT,'logical_base_prefix':str(self.site/'logical'),'physical_base_prefix':str(self.site/'physical')};self.C.QUALIFIED_LAYOUT=self.layout
        self.target=self.site/'CPU_inert.py';self.target.write_bytes(b'VALUE=37\n');self.files[str(self.target)]=B.desc(self.target,'python_source')
        assumptions={name:{'path':row['path'],'source_sha256':row['source_sha256'],'cache_path':None,'cache_sha256':None} for name,row in self.policy['modules'].items()}
        for m in self.modules.values():m.__cached__=None
        self.value={'fixture_only':True,'native_PASS':False,'native_authority_issued':False,'config':{'source_files':{}},'runtime':{'files':self.files,'stdlib_layout':self.layout,'cache_policy':{'Torch_root':str(self.site/'torch'),'namespace_roots':{'pywin32_system32':[str(self.ns)]},'preloaded_bootstrap_modules':assumptions,'preloaded_module_cache_pins':{}}}}
        self.context=self.C._mint_native_context(self.value);self.overlay=self.C._CPU_prepare_startup_overlay(self.value,self.policy,_capability=self.capC);self.handle=self.enroll()
    @contextlib.contextmanager
    def active(self):
        with patch.object(sys,'modules',self.surface.modules),patch.object(sys,'meta_path',self.surface.meta_path),patch.object(sys,'path_hooks',self.surface.path_hooks),patch.object(sys,'path',self.surface.path),patch.object(sys,'path_importer_cache',self.surface.path_importer_cache):yield
    def install(self,**kwargs):return self.L._CPU_install_bootstrap_fixture(self.context,self.C,kwargs.get('overlay',self.overlay),H,self.handle,self.surface,_capability=kwargs.get('cap',self.capC),_helper_capability=self.cap)
    def test_actual_registered_loader_import_installation_checker_and_restore(self):
        before=(list(self.surface.meta_path),list(self.surface.path_hooks),list(self.surface.path),dict(self.surface.path_importer_cache));containers=(self.surface.meta_path,self.surface.path_hooks,self.surface.path,self.surface.path_importer_cache)
        with self.active():
            finder=self.install();token=finder.installed_authority
            self.assertIsNone(self.L._check_installed_record(finder,self.value,_CPU_capability=self.capC));self.assertIsNone(self.L._check_installation_authority(token,self.value,finder=finder,_CPU_capability=self.capC))
            module=importlib.import_module('CPU_inert');self.assertEqual(module.VALUE,37);self.assertIsInstance(module.__loader__,self.L.RegisteredLoader);self.assertTrue(finder.observed['CPU_inert']['current_module_execution_observed']);self.assertFalse(finder.observed['pywin32_system32']['current_module_execution_observed'])
            self.assertTrue(finder.CPU_startup_lease[4]);self.L._CPU_remove_bootstrap_fixture(finder)
            with self.assertRaises(ValueError):self.L._check_installation_authority(token,self.value,_CPU_capability=self.capC)
        self.assertEqual((self.surface.meta_path,self.surface.path_hooks,self.surface.path,self.surface.path_importer_cache),before)
        for now,old in zip((self.surface.meta_path,self.surface.path_hooks,self.surface.path,self.surface.path_importer_cache),containers):self.assertIs(now,old)
    def test_forged_overlay_rebound_context_and_capability_refused(self):
        with self.active():
            for kwargs in [{'overlay':object()},{'cap':object()}]:
                with self.assertRaises(ValueError):self.install(**kwargs)
            changed=copy.deepcopy(self.value);changed['config']['rebound']=True;self.context=self.C._mint_native_context(changed)
            with self.assertRaises(ValueError):self.install()
        self.assertEqual(len(self.surface.meta_path),8)
    def test_cache_callable_mutation_refuses_before_execution(self):
        with self.active():
            finder=self.install();importlib.import_module('CPU_inert');cached=next(v for v in self.surface.path_importer_cache.values() if v is not None);called=[];cached.find_spec=lambda *args:called.append(True)
            try:
                with self.assertRaises(ValueError):finder.find_spec('another')
                self.assertEqual(called,[])
                with self.assertRaises(ValueError):self.L._CPU_remove_bootstrap_fixture(finder)
                self.assertIn(cached,self.surface.path_importer_cache.values())
            finally:pass
    def test_source_change_refuses_exec_and_normal_failure_cleanup_restores(self):
        with self.active():
            finder=self.install();self.target.write_bytes(b'VALUE=99\n')
            with self.assertRaises(ValueError):importlib.import_module('CPU_inert')
            self.assertNotIn('CPU_inert',self.surface.modules);self.L._CPU_remove_bootstrap_fixture(finder)
        self.assertEqual(len(self.surface.meta_path),8)
    def test_bootstrap_cache_source_mismatch_and_namespace_failure_restore(self):
        namespace=self.modules['pywin32_system32'];namespace.__path__.append(str(self.site))
        with self.active():
            with self.assertRaises(ValueError):self.install()
        self.assertEqual(len(self.surface.meta_path),8);namespace.__path__.pop()
        name='_virtualenv';self.modules[name].__cached__=str(self.site/'missing.pyc')
        with self.active():
            with self.assertRaises(ValueError):self.install()
        self.assertEqual(len(self.surface.meta_path),8)
    def test_callback_mutation_after_resolution_is_not_enrolled(self):
        with self.active():
            finder=self.install();original=self.C.file_descriptor;mutated=[]
            def alter(path,d,**kw):
                result=original(path,d,**kw)
                if path==str(self.target):
                    cached=next(v for v in self.surface.path_importer_cache.values() if v is not None);cached._loaders=[];mutated.append(cached)
                return result
            self.C.file_descriptor=alter
            importlib.import_module('CPU_inert');self.assertTrue(mutated)
            with self.assertRaises(ValueError):self.L._CPU_remove_bootstrap_fixture(finder)
            self.assertIn(mutated[0],self.surface.path_importer_cache.values())
    def test_installer_rejects_resolver_and_tracker_replacement(self):
        with self.active():
            finder=self.install();original=finder._find_spec_body;finder._find_spec_body=lambda *a:None
            with self.assertRaises(ValueError):self.L._check_installed_record(finder,self.value,_CPU_capability=self.capC)
            finder._find_spec_body=original;before=finder.CPU_cache_tracker.before;finder.CPU_cache_tracker.before=lambda:None
            with self.assertRaises(ValueError):finder.find_spec('CPU_inert')
            finder.CPU_cache_tracker.before=before;self.L._CPU_remove_bootstrap_fixture(finder)
    def test_native_contract_and_native_loader_refuse(self):
        with self.assertRaises(ValueError):self.C.native_entry(None)
        with self.assertRaises(ValueError):self.L.install(self.context,self.C)
if __name__=='__main__':
    names=[n for n in vars(Controls) if n.startswith('test_')];suite=unittest.TestSuite(Controls(n) for n in names);result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'kind':'CPU_TENSOR_BOOTSTRAP_INTEGRATION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'site_disabled':sys.flags.no_site==1,'sources':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['finite_bootstrap_helper.py','prefix_tensor_contract.py','prefix_registered_loader.py','test_bootstrap_integration.py']},'actual_inert_registered_source_imported':True,'fixture_source_context_injected':True,'native_provider_implemented':False,'native_authority_issued':False,'public_source_changed':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
