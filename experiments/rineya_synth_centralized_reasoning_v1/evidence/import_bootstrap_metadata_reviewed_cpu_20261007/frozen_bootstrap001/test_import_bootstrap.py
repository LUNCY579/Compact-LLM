"""Own-process additive module/cache and main checks; no real dependency import."""
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import hashlib,importlib.machinery,importlib.util,json,marshal,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent

def load(p,name):m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
C=load(HERE/'import_observation_contract.py','CPU_bootstrap_contract');H=load(HERE/'import_bootstrap.py','CPU_bootstrap')
def desc(p,role):raw=p.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_additive_preload_');self.root=Path(self.tmp.name);self.path=self.root/'inert.py';self.path.write_bytes(b'VALUE=7\n');self.cache=self.root/'__pycache__/inert.cpython-311.pyc';self.cache.parent.mkdir();code=compile(self.path.read_bytes(),str(self.path),'exec');self.cache.write_bytes(importlib.util.MAGIC_NUMBER+b'\0'*12+marshal.dumps(code));self.module=ModuleType('CPU_additional');self.module.__file__=str(self.path);self.module.__cached__=str(self.cache);loader=importlib.machinery.SourceFileLoader('CPU_additional',str(self.path));self.module.__loader__=loader;self.module.__spec__=importlib.machinery.ModuleSpec('CPU_additional',loader,origin=str(self.path));self.row={'path':str(self.path),'descriptor':desc(self.path,'python_source'),'cached_attribute_present':True,'cached_value_kind':'str','cache_path':str(self.cache),'cache_descriptor':desc(self.cache,'qualified_bytecode_cache'),'source_execution_history_proven':False};self.files={str(self.path):self.row['descriptor'],str(self.cache):self.row['cache_descriptor']};self.historical={'old009':{'path':'old','source_sha256':'a'*64,'cache_path':None}};self.value={'config':{'source_files':{}},'runtime':{'files':self.files,'stdlib_layout':C.QUALIFIED_LAYOUT,'cache_policy':{'preloaded_bootstrap_modules':self.historical,'additional_preload_sources':{'CPU_additional':self.row}}}}
    def tearDown(self):self.tmp.cleanup()
    def test_actual_additive_source_cache_code_equivalence_historical_map_unchanged(self):
        before=C.canonical(self.historical);result=H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module});self.assertEqual(result['CPU_additional']['cached_value_kind'],'str');self.assertTrue(result['CPU_additional']['current_state_source_verified']);self.assertFalse(result['CPU_additional']['source_execution_history_proven']);self.assertEqual(C.canonical(self.historical),before)
    def test_wrong_cache_source_code_unknown_and_history_rejected(self):
        old=self.cache.read_bytes();self.cache.write_bytes(importlib.util.MAGIC_NUMBER+b'\0'*12+marshal.dumps(compile(b'VALUE=9',str(self.path),'exec')));self.row['cache_descriptor']=desc(self.cache,'qualified_bytecode_cache');self.files[str(self.cache)]=self.row['cache_descriptor']
        with self.assertRaises(ValueError):H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})
        self.cache.write_bytes(old);self.row['cache_descriptor']=desc(self.cache,'qualified_bytecode_cache');self.files[str(self.cache)]=self.row['cache_descriptor'];self.row['source_execution_history_proven']=True
        with self.assertRaises(ValueError):H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})
        self.row['source_execution_history_proven']=False
        with self.assertRaises(ValueError):H.verify_additional_preloads(self.value,C,{})
    def test_own_current_main_frame_source_equivalence_and_wrong_globals(self):
        path=self.root/'own_main.py';source=b"def probe(helper,contract,descriptor,path):\n import sys\n return helper.verify_current_main(globals(),sys.modules['__main__'],sys._getframe(),descriptor,contract,path)\n";path.write_bytes(source);module=ModuleType('__main__');module.__file__=str(path);module.__spec__=None;module.__cached__=None;exec(compile(source,str(path),'exec'),module.__dict__)
        with patch.dict(sys.modules,{'__main__':module}):record=module.probe(H,C,desc(path,'python_source'),path)
        self.assertTrue(record['current_frame_source_equivalent']);self.assertFalse(record['source_execution_history_proven']);self.assertEqual(record['loader_kind'],'direct_import_worker_current_frame_source_equivalence')
        with self.assertRaises(ValueError):module.probe(H,C,desc(path,'python_source'),path)
    def test_no_native_preload_from_CPU_context_and_same_source_reader_fullqualifier_seam(self):
        with self.assertRaises(ValueError):H.preload_ownership_dependencies(self.value,C)
        source=(HERE/'import_observation_contract.py').read_text();self.assertIn('qualifier.qualify_import_value(value,budget)',source);self.assertLess(source.index('qualifier.qualify_import_value(value,budget)'),source.index('return _mint_import_native(value)'));self.assertEqual(C.PERMISSION['tensor_allocations'],0);self.assertFalse(C.PERMISSION['user_CUDA_context'])
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_OWN_PROCESS_ADDITIVE_PRELOAD_MAIN_SOURCE_CONTROLS','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'actual_inert_cache_source_code_metadata_checked':True,'actual_own_main_function_frame_source_checked':True,'real_psutil_or_native_extension_imported':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'import_bootstrap.py',HERE/'import_observation_contract.py',HERE/'import_observation_worker.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
