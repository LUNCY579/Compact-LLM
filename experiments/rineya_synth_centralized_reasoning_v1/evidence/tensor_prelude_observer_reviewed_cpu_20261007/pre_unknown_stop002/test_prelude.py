"""Actual frozen definitions/cache metadata on CPU; no ctypes/Win32/observer."""
from pathlib import Path
import builtins,copy,hashlib,json,sys,types,unittest
HERE=Path(__file__).resolve().parent
ORIGINAL=builtins.__import__
def trap(name,*args,**kwargs):
    if name.split('.',1)[0] in {'ctypes','_ctypes','torch','psutil','native_worker','native_supervisor','native_process','native_nvml','pywin32_bootstrap'} or name.startswith('__editable__'):raise AssertionError('CPU source prelude cannot import native/framework/site modules')
    return ORIGINAL(name,*args,**kwargs)
builtins.__import__=trap
def load(name):
    p=HERE/name;m=types.ModuleType('CPU_'+p.stem);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
C=load('contract.py');P=load('prelude.py');W=load('worker.py')
class Controls(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((HERE/'PRELUDE_PLAN_003.json').read_text());sources={row['path']:row['descriptor'] for row in self.plan['sources'].values()};sources.update(self.plan['helper_sources']);runtime=json.loads((C.PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_006.json').read_text());caches=json.loads((HERE/'ROOT_FINITE_CACHE_CANDIDATE_001.json').read_text())['files'];self.value={'config':{'source_files':sources},'runtime':{'files':{**runtime['files'],**caches},'stdlib_layout':{'logical_base_prefix':str(Path(sys.base_prefix)),'physical_base_prefix':str(Path(sys.base_prefix).resolve())}},'fixture_only':True}
    def test_actual_selected_definitions_contract_identity_cache_metadata_and_restore(self):
        before=list(sys.path);record,transition,modules=P.source_prelude(self.plan,C,self.value,cpu_only=True)
        try:
            self.assertEqual([r['file'] for r in record['compiled_helpers']],[self.plan['sources']['prefix_input_owned_worker.py']['path'],next(iter(self.plan['helper_sources'])),self.plan['sources']['prefix_registered_loader.py']['path']]);self.assertFalse(record['tensor_main_executed']);self.assertFalse(record['Torch_called']);self.assertEqual(len(record['source_costs']),3);self.assertTrue(record['modules_after_prelude']);self.assertIn('importlib.abc',[r['name'] for r in record['modules_after_prelude']]);self.assertFalse(record['registered_root_transition']['hook_invoked'])
            self.assertEqual(modules['prefix_input_owned_worker.py'].C.__file__,self.plan['sources']['prefix_tensor_contract.py']['path']);self.assertNotIn('ctypes',sys.modules);self.assertNotIn('torch',sys.modules)
        finally:P.restore_roots(transition)
        self.assertEqual(sys.path,before)
    def test_source_descriptor_rebound_refuses_before_compile(self):
        bad=copy.deepcopy(self.plan);bad['sources']['prefix_input_owned_worker.py']['descriptor']['sha256']='0'*64
        with self.assertRaises(ValueError):P.source_prelude(bad,C,self.value,cpu_only=True)
    def test_cache_equivalence_and_exceptiontable_fingerprint(self):
        path=str(Path(hashlib.__file__).resolve());files=self.value['runtime']['files'];row=P.cache_metadata(hashlib,path,self.value['runtime']['stdlib_layout'],files,{p:d for p,d in files.items() if d['role']=='qualified_bytecode_cache'})
        if row['path'] is not None and Path(row['path']).exists():self.assertTrue(row['source_equivalence']);self.assertIsNone(row['error']);self.assertIsNotNone(row['sha256'])
        code=compile('try:\n    x=1\nexcept Exception:\n    x=2\n','CPU_source','exec');self.assertNotEqual(P.fingerprint(code),P.fingerprint(code.replace(co_exceptiontable=b'')))
    def test_owned_root_mutation_refusal_does_not_clobber(self):
        record,transition,modules=P.source_prelude(self.plan,C,self.value,cpu_only=True);sys.path.append('CPU_FOREIGN')
        try:
            with self.assertRaises(ValueError):P.restore_roots(transition)
            self.assertEqual(sys.path[-1],'CPU_FOREIGN')
        finally:sys.path.pop();P.restore_roots(transition)
    def test_unselected_source_cache_reads_and_snapshot_failure_cleanup(self):
        from unittest.mock import patch
        m=types.ModuleType('CPU_unknown');m.__file__='C:/UNSELECTED/protected.py';m.__cached__='C:/UNSELECTED/protected.pyc'
        with patch.object(P.sys,'modules',{'CPU_unknown':m}),patch.object(P,'bounded_bytes',side_effect=AssertionError('unselected bytes must not be read')):
            row=P.module_snapshot(self.value['runtime']['stdlib_layout'],{},{});self.assertIsNotNone(row[0]['error']);self.assertIsNone(row[0]['source_sha256'])
            row=P.cache_metadata(m,'C:/UNSELECTED/protected.py',self.value['runtime']['stdlib_layout'],{},{});self.assertIsNotNone(row['error']);self.assertIsNone(row['sha256'])
        before=list(sys.path)
        with patch.object(P,'module_snapshot',side_effect=RuntimeError('authored snapshot failure')):
            with self.assertRaisesRegex(RuntimeError,'authored snapshot failure'):P.source_prelude(self.plan,C,self.value,cpu_only=True)
        self.assertEqual(sys.path,before)
    def test_existing_twelve_operations_and_before_ctypes_order_remain(self):
        self.assertEqual(len(C.OPS),12);self.assertEqual(C.OPS,W.OPS);self.assertEqual(C.ABI['row_bytes'],2600);self.assertEqual(C.RESOURCES['GPU_free_at_admission_GiB'],4.0)
        source=(HERE/'worker.py').read_text();body=source[source.index('def _native_bootstrap('):source.index('def main():')];self.assertLess(body.index('source_prelude'),body.index("__import__('ctypes')"));self.assertIn('modules_at_guard_boundary',source)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_FROZEN_TENSOR_PRELUDE_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'ctypes_Win32_observer_framework_tensor_called':False,'exact_frozen_definition_execution':True,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['contract.py','worker.py','prelude.py','audit.py','supervisor.py','launch_driver.py','test_prelude.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
