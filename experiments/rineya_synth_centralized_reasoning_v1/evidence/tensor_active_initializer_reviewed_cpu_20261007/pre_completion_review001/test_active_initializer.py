"""Selected package initializer imports children while exact code frame is active."""
from pathlib import Path
import hashlib,importlib,json,sys,types,unittest
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'tensor_bootstrap_package_work'
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(BASE/'test_package_paths.py','CPU_previous_packages')
class Controls(B.Controls):
    def setUp(self):
        super().setUp();self.L=load(HERE/'prefix_registered_loader.py','CPU_active_loader')
    def set_source(self,relative,raw):
        p=self.pkg/relative;p.write_bytes(raw);self.files[str(p)]=B.B.B.desc(p,'python_source');self.context=self.C._mint_native_context(self.value);self.overlay=self.C._CPU_prepare_startup_overlay(self.value,self.policy,_capability=self.capC)
    def test_actual_init_imports_child_and_nested_active_then_completed(self):
        self.set_source('__init__.py',b'from . import child\nfrom . import nested\nVALUE=child.VALUE+nested.VALUE\n');self.set_source('nested/__init__.py',b'from . import deep\nVALUE=deep.VALUE\n')
        with self.active():
            finder=self.install();seen=[];original=self.C.file_descriptor
            def observe(path,d,**kw):
                if path==str(self.pkg/'child.py'):seen.append(('CPU_pkg' in finder.CPU_active_registry,'CPU_pkg' in finder.observed))
                return original(path,d,**kw)
            self.C.file_descriptor=observe;module=importlib.import_module('CPU_pkg');self.assertEqual(module.VALUE,6);self.assertTrue(seen);self.assertTrue(all(active and not completed for active,completed in seen));self.assertEqual(finder.CPU_active_registry,{})
            for name in ['CPU_pkg','CPU_pkg.child','CPU_pkg.nested','CPU_pkg.nested.deep']:self.assertTrue(finder.observed[name]['current_module_execution_observed'])
            self.L._CPU_remove_bootstrap_fixture(finder)
        self.assertEqual(len(self.surface.meta_path),8)
    def test_initializer_failure_revokes_active_without_completed_parent(self):
        self.set_source('__init__.py',b'from . import child\nraise RuntimeError("authored initializer failure")\n')
        with self.active():
            finder=self.install()
            with self.assertRaisesRegex(RuntimeError,'authored initializer failure'):importlib.import_module('CPU_pkg')
            self.assertEqual(finder.CPU_active_registry,{});self.assertNotIn('CPU_pkg',finder.CPU_module_registry);self.assertNotIn('CPU_pkg',finder.observed);self.assertNotIn('CPU_pkg',self.surface.modules);self.L._CPU_remove_bootstrap_fixture(finder)
        self.assertEqual(len(self.surface.meta_path),8)
    def test_active_parent_path_mutation_and_child_source_drift_refuse(self):
        self.set_source('__init__.py',b'__path__.append("unregistered")\nfrom . import child\n')
        with self.active():
            finder=self.install()
            with self.assertRaises(ValueError):importlib.import_module('CPU_pkg')
            self.assertEqual(finder.CPU_active_registry,{});self.assertNotIn('CPU_pkg.child',self.surface.modules);self.L._CPU_remove_bootstrap_fixture(finder)
        self.set_source('__init__.py',b'from . import child\n');child=self.pkg/'child.py';child.write_bytes(b'VALUE=99\n')
        with self.active():
            finder=self.install()
            with self.assertRaises(ValueError):importlib.import_module('CPU_pkg')
            self.assertEqual(finder.CPU_active_registry,{});self.assertNotIn('CPU_pkg',finder.observed);self.L._CPU_remove_bootstrap_fixture(finder)
    def test_forged_active_record_without_original_frame_refuses(self):
        with self.active():
            finder=self.install();module=importlib.import_module('CPU_pkg');loader=module.__loader__;binding=finder.CPU_module_registry['CPU_pkg'][2];finder.CPU_module_registry.pop('CPU_pkg');finder.observed.pop('CPU_pkg');module.__spec__._initializing=True
            code=compile((self.pkg/'__init__.py').read_bytes(),str(self.pkg/'__init__.py'),'exec');finder.CPU_active_registry['CPU_pkg']=(module,loader,binding,code,module.__dict__)
            with self.assertRaises(ValueError):finder.find_spec('CPU_pkg.child',[str(self.pkg)])
            finder.CPU_active_registry.clear();self.L._CPU_remove_bootstrap_fixture(finder)
if __name__=='__main__':
    names=[n for n in vars(Controls) if n.startswith('test_')];result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));r={'kind':'CPU_ACTIVE_PACKAGE_INITIALIZATION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['prefix_registered_loader.py','test_active_initializer.py']},'actual_inert_initializer_and_child_frames':True,'native_authority_issued':False,'model_native_api_work':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
