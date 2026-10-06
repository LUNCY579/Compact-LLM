"""Actual finder installation, inert source import and removal on CPU."""
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
import importlib,importlib.machinery,sys,tempfile,unittest
from unittest.mock import patch
from outer_context_fixture import OuterContextFixture,HERE,source

class CPULoaderIntegrationControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_actual_loader_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.fixture=OuterContextFixture(self.temp.name);self.C=self.fixture.C;self.context=self.fixture.context();self.L=source('prefix_registered_loader.py')
        self.cap=self.C._mint_CPU_native_context_fixture()
    @contextmanager
    def isolated(self):
        bootstrap=ModuleType('CPU_bootstrap_inert');bootstrap.__file__=str(self.fixture.site/'typing_extensions.py');bootstrap.__cached__=None
        allowed=[importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder]
        with self.fixture.activate(),patch.dict(sys.modules,{'CPU_bootstrap_inert':bootstrap},clear=True),patch.object(sys,'meta_path',allowed),patch.object(sys,'path',[str(self.fixture.site)]):
            yield
    def install(self):return self.L._CPU_install_fixture(self.context,self.C,_capability=self.cap)
    def test_actual_installed_finder_imports_selected_inert_source_and_removes(self):
        with self.isolated():
            baseline=list(sys.meta_path);finder=self.install()
            try:
                module=importlib.import_module('psutil')
                expected=str(self.fixture.site/'psutil/__init__.py')
                self.assertEqual(module.__file__,expected);self.assertIsInstance(module.__loader__,self.L.RegisteredLoader)
                # ModuleSpec may compute a documentary cache filename. Prove
                # that our source-only loader neither writes nor executes it.
                self.assertFalse(Path(module.__spec__.cached).exists())
                self.assertEqual(finder.observed['psutil']['path'],expected)
                self.assertTrue(finder.observed['psutil']['current_module_execution_observed'])
                self.assertEqual(finder.observed['psutil']['loader_kind'],'registered_source')
                self.assertFalse(finder.observed['psutil']['bootstrap_execution_inferred'])
                self.assertFalse(finder.observed['CPU_bootstrap_inert']['current_module_execution_observed'])
                self.assertTrue(finder.fixture_only);self.assertFalse(finder.native_PASS);self.assertFalse(finder.native_authority_issued)
            finally:self.L.remove(finder)
            self.assertEqual(sys.meta_path,baseline)
    def test_source_mutation_rejected_before_actual_import_exec_and_removal_survives(self):
        with self.isolated():
            baseline=list(sys.meta_path);finder=self.install();p=self.fixture.site/'psutil/__init__.py';raw=p.read_bytes();p.write_bytes(b'X'+raw[1:])
            try:
                with self.assertRaisesRegex(ValueError,'registered runtime exact stable bytes'):importlib.import_module('psutil')
                self.assertNotIn('psutil',sys.modules);self.assertNotIn('psutil',finder.observed)
            finally:self.L.remove(finder)
            self.assertEqual(sys.meta_path,baseline)
    def test_unregistered_source_import_refused_without_fallback(self):
        p=self.fixture.site/'CPU_unregistered.py';p.write_bytes(b'raise AssertionError("must not execute")\n')
        with self.isolated():
            finder=self.install()
            try:
                with self.assertRaisesRegex(ValueError,'unregistered source'):importlib.import_module('CPU_unregistered')
                self.assertNotIn('CPU_unregistered',sys.modules)
            finally:self.L.remove(finder)
    def test_private_capability_and_context_layout_required(self):
        with self.isolated():
            before=list(sys.meta_path)
            with self.assertRaisesRegex(ValueError,'private CPU'):self.L._CPU_install_fixture(self.context,self.C,_capability=object())
            with patch.object(self.C,'QUALIFIED_LAYOUT',{**self.fixture.layout,'python_version':'unqualified'}):
                with self.assertRaisesRegex(ValueError,'CPU resolver exact selected context layout'):self.install()
            self.assertEqual(sys.meta_path,before)
    def test_native_physical_policy_and_installed_resolver_do_not_accept_other_layout(self):
        with self.assertRaisesRegex(ValueError,'exact reviewed interpreter alias layout'):
            self.L.physical(str(self.fixture.site/'typing_extensions.py'),self.fixture.layout)
        with self.isolated():
            finder=self.install()
            try:
                changed={**self.fixture.layout,'physical_base_prefix':str(self.fixture.root/'other')}
                with self.assertRaisesRegex(ValueError,'exact reviewed interpreter alias layout'):
                    finder.resolver(str(self.fixture.site/'typing_extensions.py'),changed)
            finally:self.L.remove(finder)

if __name__=='__main__':unittest.main()
