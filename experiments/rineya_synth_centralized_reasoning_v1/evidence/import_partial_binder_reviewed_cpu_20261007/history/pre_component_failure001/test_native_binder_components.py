"""Finite partial native binder components; source/caps only, no native calls."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import hashlib,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent

def load(p,name):m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);m.__compiled_source_descriptor__={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'python_source'};return m
W=load(HERE/'import_observation_worker.py','CPU_partial_import_worker');C=W.C
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='CPU_native_partial_');self.output=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_separate_context_actual_observer_registry_and_crosscontext_refusal(self):
        path=C.PACKAGE/'prefix_library_observer.py';adapter=load(path,'CPU_partial_adapter');cap=C._CPU_capability();desc=adapter.__compiled_source_descriptor__;binding={'path':'CPU_observer.dll','descriptor':{'bytes':1,'sha256':'1'*64,'role':'observer_binary'},'abi':C.LIBRARY_OBSERVER_ABI,'dll_handle':17};selected={'adapter_source':{'path':str(path),'descriptor':desc},'binary':{'path':binding['path'],'descriptor':binding['descriptor']}}
        value={'fixture_only':True,'scope':C.SCOPE,'config':{'source_files':{str(path):desc}},'runtime':{}};context=C._CPU_context(value,cap);observer=object.__new__(adapter.NativeLibraryObserver);observer.contract=C;observer._binding=binding
        # Only metadata selection is injected; original five registry definitions,
        # actual selected class/method identities and ImportContext are executed.
        original=C.library_event_observer_registration;C.library_event_observer_registration=lambda *a,**k:selected
        try:
            C._CPU_register_library_event_observer_fixture(context,adapter,observer,binding,_capability=cap);proof=C._CPU_verify_library_event_observer_fixture(observer,_capability=cap);self.assertTrue(proof['fixture_only']);self.assertFalse(proof['native_authority_issued'])
            with self.assertRaises(ValueError):C.verify_native_library_event_observer(observer)
            with self.assertRaises(ValueError):C.context_value(C.B._mint_native_context({'fixture_only':True}))
        finally:C.library_event_observer_registration=original
    def test_original_ownership_source_slice_avoids_module_path_mutation(self):
        path=C.PACKAGE/'native_worker.py';raw=path.read_bytes();value={'config':{'source_files':{str(path):{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}}}};before=list(sys.path)
        with patch.dict(sys.modules,{'ctypes':SimpleNamespace()}):module=W._ownership_source_slice(value)
        self.assertEqual(sys.path,before);self.assertEqual(module.verify_worker_ownership.__code__.co_filename,str(path));self.assertEqual(set(vars(module)),{'_job_query_error','_job_kernel','owned_job_pids','owned_job_resources','in_owned_job','verify_worker_ownership','read_heartbeat'})
    def test_acquisition_failure_always_restores_roots_and_retains_cleanup_priority(self):
        calls=[];state={'measurement':None,'finder':None,'transition':object(),'utilities':SimpleNamespace(_restore_registered_roots=lambda v:calls.append('restore'))};self.assertTrue(W._cleanup_acquired_state(state)['owned_roots_restored']);self.assertEqual(calls,['restore'])
        class Measurement:
            closed=False;provider=None
            def close(self):calls.append('close');raise RuntimeError('close primary')
        state={'measurement':Measurement(),'finder':None,'transition':object(),'utilities':SimpleNamespace(_restore_registered_roots=lambda v:(_ for _ in ()).throw(OSError('roots secondary')))}
        with self.assertRaisesRegex(RuntimeError,'close primary') as caught:W._cleanup_acquired_state(state)
        self.assertIsInstance(caught.exception.__cause__,OSError)
    def test_constructor_partial_packet_actual_writer_and_no_false_measurement(self):
        ledger=W.original_ledger(self.output);packet={'kind':'actual_inert_partial_record','identity':None,'native_PASS':False};packet['sha256']=C.digest(packet)
        try:
            returned=W._publish_measurement(ledger,None,{},partial_raw=packet);saved=json.loads((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').read_text());self.assertEqual(saved,packet);self.assertEqual(returned['body_sha256'],packet['sha256'])
            with self.assertRaises(ValueError):W._publish_measurement(ledger,None,{})
        finally:ledger.close()
    def test_selected_measurement_loader_forged_object_and_context_refused(self):
        path=HERE/'predecessors/prefix_registered_loader.py';raw=path.read_bytes();g=HERE/'measurement_guard.py';gd={'bytes':g.stat().st_size,'sha256':hashlib.sha256(g.read_bytes()).hexdigest(),'role':'python_source'};value={'config':{'source_files':{str(path):{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'},str(g):gd}}};module=load(g,'CPU_partial_measurement_module');loader=W._measurement_loader(value,object(),module)
        with self.assertRaises(ValueError):loader._selected_library_guard(value,C,None,fixture_only=True,library_supplier=None,library_bootstrap_verifier=None)
        with self.assertRaises(ValueError):loader._selected_library_guard({'rebound':True},C,None,fixture_only=True,library_supplier=None,library_bootstrap_verifier=None)
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_PARTIAL_IMPORT_NATIVE_BINDER_COMPONENT_CONTROLS','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'actual_observer_registry_definitions_and_ImportContext':True,'registry_metadata_selector_injected':True,'original_helper_function_slice_executed_no_Win32_calls':True,'actual_original_writer_partial_packet':True,'full_native_reader_cache_role_bootstrap_qualification_complete':False,'psutil_extra_bootstrap_assumptions_qualified':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'import_observation_contract.py',HERE/'import_observation_worker.py',HERE/'measurement_guard.py',HERE/'measurement_replay.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
