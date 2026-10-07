"""Actual package/submodule/nested import controls on CPU only."""
from pathlib import Path
import hashlib,importlib,json,sys,types,unittest
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'tensor_bootstrap_integration_work'
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(BASE/'test_bootstrap_integration.py','CPU_previous_integration')
class Controls(B.Controls):
    def setUp(self):
        super().setUp();self.L=load(HERE/'prefix_registered_loader.py','CPU_package_loader');self.modules['pywin32_system32'].__spec__.name='pywin32_system32'
        self.pkg=self.site/'CPU_pkg';(self.pkg/'nested').mkdir(parents=True)
        for rel,raw in [('__init__.py',b'VALUE=1\n'),('child.py',b'VALUE=2\n'),('nested/__init__.py',b'VALUE=3\n'),('nested/deep.py',b'VALUE=4\n')]:
            p=self.pkg/rel;p.write_bytes(raw);self.files[str(p)]=B.B.desc(p,'python_source')
        self.context=self.C._mint_native_context(self.value);self.overlay=self.C._CPU_prepare_startup_overlay(self.value,self.policy,_capability=self.capC)
    def test_actual_package_child_nested_import_and_original_tokens(self):
        before=list(self.surface.meta_path)
        with self.active():
            finder=self.install();module=importlib.import_module('CPU_pkg.nested.deep');self.assertEqual(module.VALUE,4);child=importlib.import_module('CPU_pkg.child');self.assertEqual(child.VALUE,2)
            for name in ['CPU_pkg','CPU_pkg.child','CPU_pkg.nested','CPU_pkg.nested.deep']:
                self.assertTrue(finder.observed[name]['current_module_execution_observed']);self.assertIsInstance(self.surface.modules[name].__loader__,self.L.RegisteredLoader)
            self.assertIn(str(self.pkg),finder.CPU_startup_lease[4]);self.assertIn(str(self.pkg/'nested'),finder.CPU_startup_lease[4]);self.assertIsNone(self.L._check_installed_record(finder,self.value,_CPU_capability=self.capC));self.L._CPU_remove_bootstrap_fixture(finder)
        self.assertEqual(self.surface.meta_path,before)
    def test_package_module_substitution_or_path_rebinding_refused(self):
        with self.active():
            finder=self.install();module=importlib.import_module('CPU_pkg');copy=types.ModuleType('CPU_pkg');copy.__dict__.update(module.__dict__);self.surface.modules['CPU_pkg']=copy
            with self.assertRaises(ValueError):finder.find_spec('CPU_pkg.child',[str(self.pkg)])
            self.surface.modules['CPU_pkg']=module;module.__path__.append(str(self.site))
            with self.assertRaises(ValueError):finder.find_spec('CPU_pkg.child',module.__path__)
            module.__path__.pop()
            with self.assertRaises(ValueError):finder.find_spec('CPU_pkg.child',None)
            loader=module.__loader__;old=(loader.path,loader.descriptor,module.__file__,module.__spec__.origin,module.__path__[:],module.__spec__.submodule_search_locations[:]);replacement=self.pkg/'nested/__init__.py'
            loader.path=str(replacement);loader.descriptor=self.files[str(replacement)];module.__file__=str(replacement);module.__spec__.origin=str(replacement);module.__path__[:]=[str(replacement.parent)];module.__spec__.submodule_search_locations[:]=module.__path__
            with self.assertRaises(ValueError):finder.find_spec('CPU_pkg.child',module.__path__)
            loader.path,loader.descriptor,module.__file__,module.__spec__.origin=old[:4];module.__path__[:]=old[4];module.__spec__.submodule_search_locations[:]=old[5];self.L._CPU_remove_bootstrap_fixture(finder)
    def test_unregistered_package_source_and_namespace_extra_path_refused(self):
        (self.site/'CPU_extra_namespace').mkdir()
        with self.active():
            finder=self.install()
            with self.assertRaises(ValueError):importlib.import_module('CPU_extra_namespace')
            importlib.import_module('CPU_pkg');foreign=self.pkg/'foreign.py';foreign.write_bytes(b'raise AssertionError("unregistered must not execute")\n')
            with self.assertRaises(ValueError):importlib.import_module('CPU_pkg.foreign')
            ns=self.modules['pywin32_system32'];ns.__path__.append(str(self.site))
            with self.assertRaises(ValueError):finder.find_spec('pywin32_system32.child',ns.__path__)
            ns.__path__.pop();extra=self.ns/'foreign.dll';extra.write_bytes(b'inert never loaded')
            with self.assertRaises(ValueError):finder.find_spec('pywin32_system32.child',ns.__path__)
            extra.unlink();self.L._CPU_remove_bootstrap_fixture(finder)
    def test_parent_source_change_and_owned_package_cache_mutation_refuse(self):
        with self.active():
            finder=self.install();importlib.import_module('CPU_pkg.child');init=self.pkg/'__init__.py';original=init.read_bytes();init.write_bytes(b'VALUE=9\n')
            with self.assertRaises(ValueError):finder.find_spec('CPU_pkg.next',[str(self.pkg)])
            init.write_bytes(original);cached=self.surface.path_importer_cache[str(self.pkg)];cached._loaders=[]
            with self.assertRaises(ValueError):finder.find_spec('CPU_pkg.next',[str(self.pkg)])
            with self.assertRaises(ValueError):self.L._CPU_remove_bootstrap_fixture(finder)
            self.assertIs(self.surface.path_importer_cache[str(self.pkg)],cached)
if __name__=='__main__':
    names=[n for n in vars(Controls) if n.startswith('test_')];result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));r={'kind':'CPU_REGISTERED_PACKAGE_PATH_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'sources':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['prefix_registered_loader.py','test_package_paths.py']},'actual_inert_package_and_nested_import':True,'native_authority_issued':False,'model_native_api_work':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
