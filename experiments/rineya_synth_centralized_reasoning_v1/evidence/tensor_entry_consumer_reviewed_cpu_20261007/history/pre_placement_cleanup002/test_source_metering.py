"""Actual selected source/observer/provider boundaries; inert backend, no APIs."""
from pathlib import Path
import hashlib,json,tempfile,types,unittest,sys
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
PROVIDER=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/history/WORKER_LIBRARY_PUBLICATION_SOURCE_FROZEN_017/source/prefix_library_policy.py'
OBSERVER=PACKAGE/'prefix_library_observer.py'

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);m.__compiled_source_descriptor__={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'role':'python_source'};exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
W=load(HERE/'prefix_input_owned_worker.py','CPU_meter_worker')
class NativeTrap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='CPU_source_meter_');self.root=Path(self.tmp.name);self.meter=W._SelectedSourceMeter();self.costs={};self.meter.enter('startup_source_qualification',self.costs);self.trap=NativeTrap();sys.meta_path.insert(0,self.trap)
    def tearDown(self):
        self.meter.leave();self.meter.restore();sys.meta_path.remove(self.trap);self.tmp.cleanup()
    def test_actual_contract_read_hash_sealed_roster_and_cache_bytes(self):
        C=load(HERE/'prefix_tensor_contract.py','CPU_meter_contract');self.meter.bind_contract(C)
        p=self.root/'inert.py';p.write_bytes(b'VALUE=7\n');d={'bytes':8,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'python_source'}
        self.assertEqual(C.read_bytes(p),b'VALUE=7\n');C.file_descriptor(p,d);v={'kind':'authored'};v['sha256']=C.digest(v);j=self.root/'sealed.json';j.write_text(C.canonical(v));raw=j.read_bytes();self.assertEqual(C.sealed_json(j,hashlib.sha256(raw).hexdigest(),v['sha256']),v);self.assertIn(str(p),C.physical_roster(self.root,['.py'],['__pycache__']))
        c=self.root/'authored.pyc';c.write_bytes(b'cpu-metadata');C.read_bytes(c)
        rows=self.meter.snapshot()['startup_source_qualification'];self.assertEqual(rows['contract.sealed_json']['returned'],1);self.assertEqual(rows['contract.read_bytes.source']['returned'],2);self.assertEqual(self.costs['source_bytes'],16+len(raw));self.assertEqual(self.costs['cache_bytes'],12);self.assertEqual(rows['contract.read_bytes.cache']['returned'],1);self.assertEqual(sum(row['bytes'] for row in rows.values() if row['category']=='source'),self.costs['source_bytes']);self.assertEqual(sum(row['bytes'] for row in rows.values() if row['category']=='cache'),self.costs['cache_bytes']);self.assertEqual(rows['contract.physical_roster']['bytes'],0)
    def test_real_failed_reads_reconcile_and_mutated_binding_is_not_clobbered(self):
        C=load(HERE/'prefix_tensor_contract.py','CPU_meter_contract_failure');self.meter.bind_contract(C)
        with self.assertRaises(FileNotFoundError):C.read_bytes(self.root/'missing.py')
        row=self.meter.snapshot()['startup_source_qualification']['contract.read_bytes.source'];self.assertEqual((row['attempted'],row['failed'],row['pending']),(1,1,0))
        owned=C.read_bytes;foreign=lambda *a:None;C.read_bytes=foreign
        with self.assertRaisesRegex(ValueError,'no clobber'):self.meter.restore()
        self.assertIs(C.read_bytes,foreign);C.read_bytes=owned
    def test_actual_exact_module_reader_and_immutable_Git_blob(self):
        G=load(HERE/'registration_git_join.py','CPU_meter_git');self.meter.bind_git(G);raw=G.read(PROVIDER);self.assertEqual(hashlib.sha256(raw).hexdigest(),hashlib.sha256(PROVIDER.read_bytes()).hexdigest())
        relative=PROVIDER.relative_to(PACKAGE.parents[1]).as_posix();blob='HEAD:'+relative
        result=G.subprocess.run(['git','cat-file','blob',blob],cwd=PACKAGE.parents[1],capture_output=True,check=True)
        self.assertEqual(result.stdout,raw);W.exact_module(OBSERVER,hashlib.sha256(OBSERVER.read_bytes()).hexdigest(),_meter=self.meter)
        rows=self.meter.snapshot()['startup_source_qualification'];self.assertEqual(rows['git.immutable_blob.source']['bytes'],len(raw));self.assertEqual(rows['worker.exact_module_read.source']['bytes'],OBSERVER.stat().st_size)
    def test_selected_observer_and_provider_methods_execute_inert_backends(self):
        O=load(OBSERVER,'CPU_meter_observer');B=load(PROVIDER,'CPU_meter_provider')
        observer=object.__new__(O.NativeLibraryObserver);observer.check=lambda:None;observer.value={};observer.context=object();observer.contract=types.SimpleNamespace(context_value=lambda ctx:{});observer._binding={'selected':'CPU'};observer.birth=None;observer.owner={}
        observer.transport=types.SimpleNamespace(abi_binding=lambda:{'selected':'CPU'},self_identity=lambda:{'pid':1,'creation_filetime_100ns':2},verify_preloaded_bootstrap=lambda v:None,verify_owned_birth=lambda *a:None)
        original=O.NativeLibraryObserver.bootstrap;result=self.meter.api(O,observer,'bootstrap','inert');self.assertEqual(result['stage'],'inert');self.assertIs(O.NativeLibraryObserver.bootstrap,original)
        provider=object.__new__(B.ManagedWin32LibraryProvider);provider.directories=['CPU'];provider.system_directory='CPU';provider.closed=False;provider.bootstrap_verifier=lambda stage:{'kind':'VERIFIED_CURRENT_LIBRARY_BOOTSTRAP_PROVENANCE','stage':stage,'fresh_process_identity_verified':True,'provider_source_verified':True,'preloaded_sources_verified':True,'module_event_scope_provenance_verified':True}
        provider.ctypes=types.SimpleNamespace(set_last_error=lambda n:None,create_unicode_buffer=lambda n:types.SimpleNamespace(value='',__len__=lambda:32768),get_last_error=lambda:0)
        class Buffer:
            value=''
            def __len__(self):return 32768
        provider.ctypes.create_unicode_buffer=lambda n:Buffer();provider.kernel=types.SimpleNamespace(GetDllDirectoryW=lambda n,b:0)
        self.assertEqual(self.meter.api(B,provider,'scope')['kind'],'OBSERVED_MANAGED_DLL_DIRECTORY_SCOPE');self.assertEqual(self.costs['API_calls'],2)
        observer.transport.self_identity=lambda:(_ for _ in ()).throw(RuntimeError('inert backend failed'))
        with self.assertRaisesRegex(RuntimeError,'inert backend failed'):self.meter.api(O,observer,'bootstrap','failed')
        row=self.meter.snapshot()['startup_source_qualification']['API.NativeLibraryObserver.bootstrap'];self.assertEqual((row['attempted'],row['returned'],row['failed'],row['pending']),(2,1,1,0));self.assertFalse(hasattr(provider,'native_runtime_qualified'))
    def test_same_journal_persists_actual_boundary_counters(self):
        self.meter.leave();ledger,phases=W._entry_journal_bootstrap(self.root);saved={}
        def source(state,costs):
            C=load(HERE/'prefix_tensor_contract.py','CPU_counter_contract');state['source_meter'].bind_contract(C);p=self.root/'real.py';p.write_bytes(b'x=1');C.read_bytes(p)
        actions={name:lambda state,costs:None for name in phases};actions[phases[0]]=source;actions[phases[5]]=lambda state,costs:state['source_meter'].restore()
        report=W._entry_phase_body(ledger,phases,actions,lambda *a:{'status':'CPU_PREFIX_INPUT_CHECKS_FAILED'});final=json.loads((self.root/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertEqual(final['selected_boundary_counters'][phases[0]]['contract.read_bytes.source']['bytes'],3);self.assertEqual(final['startup_subcosts'][phases[0]]['source_bytes'],3);self.assertFalse(final['whole_scope_success_validated'])
    def test_exact_core_multiplicity_and_native_failure_mode_without_permit(self):
        self.meter.leave();ledger,phases=W._entry_journal_bootstrap(self.root);P=load(HERE/'prefix_native_pipeline.py','CPU_meter_pipeline');core=W.source_core();F=load(W.C.CORE/'fixture_support.py','CPU_meter_core_fixture')
        for phase in phases[:5]:ledger.startup(phase,lambda:None)
        execute,namespace,proof=P._bind_shared_core_executor(W,ledger);api=F.InjectedAPI();report=execute(api,api.guard,self.root);ledger.call(core['C'].OPS[0],lambda:None);ledger.startup(phases[5],lambda:None);summary=ledger.finish_owned(report);self.assertFalse(summary['whole_scope_success_validated']);self.assertEqual(report['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED')
        output=self.root/'native_mode';output.mkdir();other,phases=W._entry_journal_bootstrap(output);other.entry_scope='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY';actions={n:lambda *a:None for n in phases};actions[phases[0]]=lambda *a:(_ for _ in ()).throw(ValueError('closed native source'))
        with self.assertRaisesRegex(ValueError,'closed native source'):W._entry_phase_body(other,phases,actions,lambda *a:self.fail(),_native_mode=True)
        saved=json.loads((output/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertEqual(saved['status'],'NATIVE_INPUT_CHECKS_FAILED');self.assertEqual(saved['scope'],'ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY');self.assertFalse(saved['native_PASS'])
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));files=[HERE/n for n in ('prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_tensor_contract.py','prefix_registered_loader.py','test_source_metering.py')]+[OBSERVER,PROVIDER]
    receipt={'kind':'CPU_SELECTED_BOUNDARY_METERING_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'actual_selected_source_functions_attempted':True,'backend_mode':'inert','native_API_or_framework_called':False,'native_authority_issued':False,'native_provider_phase_placement_fixed':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
