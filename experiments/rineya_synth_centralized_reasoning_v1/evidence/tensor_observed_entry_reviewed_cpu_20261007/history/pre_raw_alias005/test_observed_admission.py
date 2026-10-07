"""Linked observed bootstrap assumptions and common inert import tracker."""
from pathlib import Path
import builtins,copy,hashlib,importlib,json,sys,types,typing,unittest
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'tensor_bootstrap_integration_work'
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(BASE/'test_bootstrap_integration.py','CPU_fixture_base')
class Controls(B.Controls):
    def setUp(self):
        super().setUp();self.L=load(HERE/'prefix_registered_loader.py','CPU_common_loader')
        pkg=self.site/'CPU_observed_pkg';pkg.mkdir();self.pkg=pkg
        for name,raw in [('__init__.py',b'from . import child\nVALUE=child.VALUE\n'),('child.py',b'VALUE=19\n')]:
            p=pkg/name;p.write_bytes(raw);self.files[str(p)]=B.B.desc(p,'python_source')
        # This importer fixture intentionally has only authored ordinary source
        # bootstrap modules plus an authenticated namespace-free surface.
        self.surface.modules.pop('pywin32_system32');self.surface.meta_path[:]=self.standard;self.surface.path_hooks[:]=self.standardhooks;self.surface.path[:]=[str(self.site)];self.surface.path_importer_cache.clear();self.value['runtime']['cache_policy']['namespace_roots']={}
        self.context=self.C._mint_native_context(self.value)
    def install(self,**kwargs):return self.L._CPU_install_observed_fixture(self.context,self.C,_capability=kwargs.get('cap',self.capC))
    def test_common_installer_actual_package_init_child_original_checker_public_remove(self):
        before=dict(self.surface.path_importer_cache)
        with self.active():
            finder=self.install();token=finder.installed_authority;module=importlib.import_module('CPU_observed_pkg');self.assertEqual(module.VALUE,19);self.assertIsInstance(module.__loader__,self.L.RegisteredLoader);self.assertTrue(finder.observed['CPU_observed_pkg.child']['current_module_execution_observed']);self.assertIsNone(self.L._check_installation_authority(token,self.value,finder=finder,_CPU_capability=self.capC));self.L.remove(finder);self.assertEqual(self.surface.path_importer_cache,before)
            with self.assertRaises(ValueError):self.L._check_installation_authority(token,self.value,_CPU_capability=self.capC)
    def test_owned_cache_mutation_refuses_before_callback_and_cleanup_preserves_it(self):
        with self.active():
            finder=self.install();importlib.import_module('CPU_observed_pkg');cached=self.surface.path_importer_cache[str(self.site)];called=[];cached.find_spec=lambda *a:called.append(True)
            with self.assertRaises(ValueError):finder.find_spec('CPU_other')
            self.assertEqual(called,[])
            with self.assertRaises(ValueError):self.L.remove(finder)
            self.assertIs(self.surface.path_importer_cache[str(self.site)],cached)
    def test_actual009_cache_alias_projection_excludes_main_and_preserves_ctypes(self):
        record=json.loads((HERE/'OBSERVED_ADMISSION_001.json').read_text());self.assertNotIn('__main__',record['preloaded_bootstrap_modules']);self.assertIsNone(record['tensor_main']);self.assertEqual(set(record['preloaded_class_aliases']),{'typing.io','typing.re'});self.assertEqual(len(record['preloaded_module_cache_pins']),59)
        for name in ['ctypes','ctypes.wintypes','ctypes._endian']:self.assertIsInstance(record['preloaded_bootstrap_modules'][name]['cache_path'],str);self.assertEqual(record['preloaded_bootstrap_modules'][name]['cached_value_kind'],'str')
        self.assertEqual(record['preloaded_bootstrap_modules']['_ctypes']['cached_value_kind'],'missing');self.assertFalse(record['native_launch_authorized'])
    def test_exact_observed_maps_derive_from_current_original_not_retagged_main(self):
        C=load(HERE/'prefix_tensor_contract.py','CPU_observed_contract');record=json.loads((HERE/'OBSERVED_ADMISSION_001.json').read_text());original=json.loads(Path(record['observation_path']).read_text());modules,pins,aliases=C.observed_bootstrap_maps(original);self.assertEqual(modules,record['preloaded_bootstrap_modules']);self.assertEqual(pins,record['preloaded_module_cache_pins']);self.assertEqual(aliases,record['preloaded_class_aliases']);self.assertNotIn('__main__',modules)
    def test_version2_actual_aliases_separate_inert_main_common_source_install(self):
        helper=load(HERE/'typing_alias_checks.py','CPU_alias_helper');runtime=json.loads((self.C.PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json').read_text());layout=runtime['stdlib_layout'];self.value['runtime']['files'].update(runtime['files']);self.value['runtime']['stdlib_layout']=layout;self.C.QUALIFIED_LAYOUT=layout;policy=self.value['runtime']['cache_policy'];record=json.loads((HERE/'OBSERVED_ADMISSION_001.json').read_text());policy.update(bootstrap_admission_version=2,preloaded_class_aliases=record['preloaded_class_aliases'])
        alias_path=str(HERE/'typing_alias_checks.py');raw=Path(alias_path).read_bytes();desc={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};self.value['config']['source_files'][alias_path]=desc;policy['typing_alias_helper']={'path':alias_path,'descriptor':desc}
        self.surface.modules.update({'typing':typing,'typing.io':vars(typing)['io'],'typing.re':vars(typing)['re']});policy['preloaded_bootstrap_modules']['typing']=record['preloaded_bootstrap_modules']['typing'];policy['preloaded_module_cache_pins']=record['preloaded_module_cache_pins']
        for name,row in policy['preloaded_bootstrap_modules'].items():
            if name!='typing':row.update(cached_attribute_present=True,cached_value_kind='none')
        main_path=self.site/'CPU_tensor_main.py';main_path.write_bytes(b'# inert source-bound main metadata\n');main_desc=B.B.desc(main_path,'python_source');self.files[str(main_path)]=main_desc;main=types.ModuleType('__main__');main.__file__=str(main_path);main.__cached__=None;self.surface.modules['__main__']=main;policy['tensor_main_source']={'path':str(main_path),'descriptor':main_desc,'cached_attribute_present':True,'cached_value_kind':'none','cache_path':None,'cache_sha256':None}
        self.context=self.C._mint_native_context(self.value)
        with self.active():
            finder=self.install();self.assertEqual(finder.observed['typing.io']['loader_kind'],'externally_observed_stdlib_class_alias');self.assertIn('__main__',finder.observed);module=importlib.import_module('CPU_observed_pkg');self.assertEqual(module.VALUE,19);self.L.remove(finder)
            main.__cached__='erased_or_fake_cache.pyc'
            with self.assertRaises(ValueError):self.install()
    def test_preloaded_regular_parent_child_is_source_bound_without_completed_claim(self):
        parent=self.site/'CPU_preloaded';parent.mkdir();init=parent/'__init__.py';init.write_bytes(b'# authored bootstrap package\n');child=parent/'child.py';child.write_bytes(b'VALUE=23\n');self.files[str(init)]=B.B.desc(init,'python_source');self.files[str(child)]=B.B.desc(child,'python_source');module=types.ModuleType('CPU_preloaded');module.__file__=str(init);module.__cached__=None;loader=importlib.machinery.SourceFileLoader('CPU_preloaded',str(init));module.__loader__=loader;module.__path__=[str(parent)];module.__spec__=importlib.machinery.ModuleSpec('CPU_preloaded',loader,origin=str(init),is_package=True);module.__spec__.submodule_search_locations=module.__path__;self.surface.modules['CPU_preloaded']=module;self.value['runtime']['cache_policy']['preloaded_bootstrap_modules']['CPU_preloaded']={'path':str(init),'source_sha256':self.files[str(init)]['sha256'],'cache_path':None,'cache_sha256':None};self.context=self.C._mint_native_context(self.value)
        with self.active():
            finder=self.install();childmodule=importlib.import_module('CPU_preloaded.child');self.assertEqual(childmodule.VALUE,23);self.assertFalse(finder.observed['CPU_preloaded']['current_module_execution_observed']);self.assertNotIn('CPU_preloaded',finder.CPU_module_registry);self.L.remove(finder)
    def test_replaced_importer_cache_container_refuses_before_foreign_callback(self):
        with self.active():
            finder=self.install();original=sys.path_importer_cache;called=[];foreign=importlib.machinery.FileFinder(str(self.site));foreign.find_spec=lambda *a:called.append(True);sys.path_importer_cache={str(self.site):foreign}
            try:
                with self.assertRaises(ValueError):finder.find_spec('CPU_other')
                self.assertEqual(called,[])
                with self.assertRaises(ValueError):self.L.remove(finder)
                self.assertIs(sys.path_importer_cache[str(self.site)],foreign)
            finally:sys.path_importer_cache=original
    def test_false_capability_context_source_and_native_route_refuse(self):
        with self.active():
            with self.assertRaises(ValueError):self.install(cap=object())
            self.target.write_bytes(b'VALUE=99\n');finder=self.install()
            try:
                with self.assertRaises(ValueError):importlib.import_module('CPU_inert')
            finally:self.L.remove(finder)
        with self.assertRaises(ValueError):self.L.install(self.context,self.C)
if __name__=='__main__':
    names=[n for n in vars(Controls) if n.startswith('test_')];result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));r={'kind':'CPU_LINKED_OBSERVED_ADMISSION_COMMON_IMPORTER_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_inert_package_initializer_child_import':True,'native_gate_opened':False,'ctypes_Torch_API_called':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['prefix_registered_loader.py','prefix_tensor_contract.py','test_observed_admission.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
