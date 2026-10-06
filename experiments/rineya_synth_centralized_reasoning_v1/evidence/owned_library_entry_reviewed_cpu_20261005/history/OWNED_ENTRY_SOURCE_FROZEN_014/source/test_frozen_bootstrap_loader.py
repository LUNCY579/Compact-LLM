"""Inert frozen/builtin bootstrap classification; no runtime byte reads."""
from types import ModuleType,SimpleNamespace
import importlib.machinery
import sys,unittest
from unittest.mock import patch
from outer_context_fixture import source

L=source('prefix_registered_loader.py')
C=source('prefix_tensor_contract.py')

class FrozenBootstrapLoaderControls(unittest.TestCase):
    def module(self,name,origin,loader,documentary=True):
        value=ModuleType(name)
        value.__spec__=importlib.machinery.ModuleSpec(name,loader,origin=origin)
        if documentary:value.__file__='C:/CPU_DOCUMENTARY_ONLY/must_not_read.py'
        return value
    def install(self,modules):
        value={'runtime':{'files':{},'stdlib_layout':C.QUALIFIED_LAYOUT,
            'cache_policy':{'namespace_roots':{},'Torch_root':'C:/CPU_NEVER_TORCH',
                            'preloaded_bootstrap_modules':{}}},'config':{'source_files':{}}}
        with patch.dict(sys.modules,modules,clear=True),patch.object(sys,'meta_path',[importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder]),patch.object(L,'physical',side_effect=AssertionError('documentary path must not be resolved')),patch.object(C,'file_descriptor',side_effect=AssertionError('must not hash')),patch.object(C,'read_bytes',side_effect=AssertionError('must not read')):
            finder=L._install_body(value,C,fixture_only=True)
            self.assertIs(sys.meta_path[0],finder)
            self.assertEqual(finder.observed,{})
            L.remove(finder)
            self.assertNotIn(finder,sys.meta_path)
            return finder
    def test_real_interpreter_recognized_frozen_module_documentary_path_ignored(self):
        name='_frozen_importlib';self.assertIsNotNone(importlib.machinery.FrozenImporter.find_spec(name))
        self.install({name:self.module(name,'frozen',importlib.machinery.FrozenImporter)})
    def test_real_interpreter_recognized_builtin_module_documentary_path_ignored(self):
        self.install({'sys':self.module('sys','built-in',importlib.machinery.BuiltinImporter)})
    def test_counterfeit_frozen_origin_and_loader_rejected_without_bytes(self):
        cases=[('CPU_nonexistent_frozen',self.module('CPU_nonexistent_frozen','frozen',importlib.machinery.FrozenImporter)),
               ('sys',self.module('sys','frozen',importlib.machinery.FrozenImporter)),
               ('_frozen_importlib',self.module('_frozen_importlib','frozen',object())),
               ('_frozen_importlib',self.module('wrong_name','frozen',importlib.machinery.FrozenImporter))]
        for name,module in cases:
            with self.subTest(name=name,loader=module.__spec__.loader):
                with self.assertRaisesRegex(ValueError,'builtin/frozen'):
                    self.install({name:module})
    def test_fileless_source_module_stays_rejected(self):
        value=self.module('CPU_source','CPU_source.py',object(),documentary=False)
        with self.assertRaisesRegex(ValueError,'fileless preloaded module'):
            self.install({'CPU_source':value})
    def test_preloaded_torch_cannot_claim_interpreter_origin(self):
        value=self.module('torch','frozen',importlib.machinery.FrozenImporter)
        with self.assertRaisesRegex(ValueError,'Torch must be fresh'):
            self.install({'torch':value})

if __name__=='__main__':unittest.main()
