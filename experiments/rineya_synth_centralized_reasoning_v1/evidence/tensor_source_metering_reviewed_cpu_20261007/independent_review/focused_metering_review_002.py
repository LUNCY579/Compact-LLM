"""Independent frozen boundary probes; original methods, inert backend only."""
from pathlib import Path
import ast, hashlib, json, sys, tempfile, types, unittest

REVIEW=Path(__file__).resolve().parent
WORK=REVIEW.parent/'pre_placement_metering002'
PACKAGE=next(p for p in WORK.parents if p.name=='rineya_synth_centralized_reasoning_v1')
REPOSITORY=PACKAGE.parents[1]
EXPECTED=json.loads((WORK/'ROOT_PRESERVATION.json').read_text(encoding='utf-8'))
PINS_BEFORE={p:{'bytes':(WORK/p).stat().st_size,'sha256':hashlib.sha256((WORK/p).read_bytes()).hexdigest()} for p in EXPECTED}
if PINS_BEFORE!=EXPECTED:raise ValueError('frozen file pins changed before review')
PIN_HASHES={'prefix_input_owned_worker.py':'e34ae4290804e4d013527efb769c55cd4003ce653a17516d09c80cf70f325480','prefix_native_pipeline.py':'0f972a06651aaf028710f5df22931a42ac7ccd54e427bb2e294d1b79ff7df9ea','prefix_tensor_contract.py':'0d12fd01bb1f06c29054943a3a114efc6eac9619c33989e5ce030d97a3cf4263','prefix_registered_loader.py':'62c25f6cf5f843a08463dbb7c3c573f6a038f036917fc4130da0aab5d4b7e0c5'}
for name,pin in PIN_HASHES.items():
    if PINS_BEFORE[name]['sha256']!=pin:raise ValueError('required candidate hash changed')

def load(path,name):
    raw=path.read_bytes();module=types.ModuleType(name);module.__file__=str(path)
    module.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
    exec(compile(raw,str(path),'exec'),module.__dict__);return module

class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'ctypes','torch','psutil','native_worker'}:raise AssertionError('native import forbidden: '+fullname)

TRAP=Trap();sys.meta_path.insert(0,TRAP)
W=load(WORK/'prefix_input_owned_worker.py','independent_meter_worker')
# Only the fixture-relative local package expression is substituted. Raw
# candidate bytes remain pinned and unchanged; all writer code/pins are retained.
tree=ast.parse((WORK/'prefix_input_owned_worker.py').read_bytes())
bootstrap=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_entry_journal_bootstrap')
substitutions=0
for node in bootstrap.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='package':
        if ast.unparse(node.value)!="HERE.parents[1] if HERE.parent.name == 'artifacts' else HERE":raise ValueError('unexpected fixture package expression')
        node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(PACKAGE))],keywords=[]);substitutions+=1
if substitutions!=1:raise ValueError('exactly one fixture-only package substitution')
exec(compile(ast.fix_missing_locations(ast.Module(body=[bootstrap],type_ignores=[])),str(WORK/'prefix_input_owned_worker.py'),'exec'),W.__dict__)
OBS=WORK/'metering_selected_source_predecessors/prefix_library_observer.py'
PROV=WORK/'metering_selected_source_predecessors/prefix_library_policy.py'
TRACKED_PROVIDER=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/history/WORKER_LIBRARY_PUBLICATION_SOURCE_FROZEN_017/source/prefix_library_policy.py'
DETAILS={}

class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='independent_frozen_meter002_');self.root=Path(self.temp.name)
        self.m=W._SelectedSourceMeter();self.costs={};self.m.enter('review_boundary',self.costs)
    def tearDown(self):
        self.m.leave();self.m.restore();self.temp.cleanup()
    def test_actual_source_hash_sealed_roster_cache_leaf_counts(self):
        C=load(WORK/'prefix_tensor_contract.py','independent_contract_reads');self.m.bind_contract(C)
        p=self.root/'selected.py';raw=b'answer=13\n';p.write_bytes(raw)
        descriptor={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
        self.assertEqual(C.read_bytes(p),raw);C.file_descriptor(p,descriptor)
        body={'kind':'independent_inert_json','value':13};body['sha256']=C.digest(body);j=self.root/'selected.json';jr=C.canonical(body).encode('ascii');j.write_bytes(jr)
        self.assertEqual(C.sealed_json(j,hashlib.sha256(jr).hexdigest(),body['sha256']),body)
        self.assertIn(str(p),C.physical_roster(self.root,['.py'],['__pycache__']))
        cache=self.root/'selected.pyc';cr=b'independent-cache-fixture';cache.write_bytes(cr);self.assertEqual(C.read_bytes(cache),cr)
        rows=self.m.snapshot()['review_boundary']
        self.assertEqual(rows['contract.read_bytes'],{'attempted':3,'returned':3,'failed':0,'pending':0,'bytes':len(raw)+len(jr)+len(cr)})
        self.assertEqual(rows['contract.file_descriptor']['bytes'],len(raw));self.assertEqual(rows['contract.physical_roster']['bytes'],0)
        self.assertEqual(self.costs['source_bytes'],2*len(raw)+len(jr));self.assertEqual(self.costs['cache_bytes'],len(cr));DETAILS['leaf_counts']=rows
    def test_failed_hash_and_read_and_no_clobber(self):
        C=load(WORK/'prefix_tensor_contract.py','independent_contract_failed');self.m.bind_contract(C)
        p=self.root/'bad.py';p.write_bytes(b'x=3')
        with self.assertRaisesRegex(ValueError,'registered runtime exact stable bytes'):C.file_descriptor(p,{'bytes':3,'sha256':'0'*64,'role':'python_source'})
        with self.assertRaises(FileNotFoundError):C.read_bytes(self.root/'missing.py')
        rows=self.m.snapshot()['review_boundary']
        for name in ('contract.file_descriptor','contract.read_bytes'):
            self.assertEqual(rows[name],{'attempted':1,'returned':0,'failed':1,'pending':0,'bytes':0})
        owned=C.read_bytes;foreign=lambda *args,**kw:b'foreign';C.read_bytes=foreign
        with self.assertRaisesRegex(ValueError,'no clobber'):self.m.restore()
        self.assertIs(C.read_bytes,foreign);C.read_bytes=owned;DETAILS['failed_counts']=rows
    def test_actual_git_blob_and_exact_reader_and_restoration(self):
        G=load(WORK/'registration_git_join.py','independent_meter_git');original=G.subprocess;read=G.read;self.m.bind_git(G)
        expected=TRACKED_PROVIDER.read_bytes();self.assertEqual(G.read(TRACKED_PROVIDER),expected)
        blob='HEAD:'+TRACKED_PROVIDER.relative_to(REPOSITORY).as_posix()
        actual=G.subprocess.run(['git','cat-file','blob',blob],cwd=REPOSITORY,capture_output=True,check=True)
        self.assertEqual(actual.stdout,expected)
        O=W.exact_module(OBS,PINS_BEFORE['metering_selected_source_predecessors/prefix_library_observer.py']['sha256'],_meter=self.m)
        self.assertEqual(O.__compiled_source_descriptor__['sha256'],PINS_BEFORE['metering_selected_source_predecessors/prefix_library_observer.py']['sha256'])
        rows=self.m.snapshot()['review_boundary'];self.assertEqual(rows['git.immutable_blob']['bytes'],len(expected));self.assertEqual(rows['worker.exact_module_read']['bytes'],OBS.stat().st_size)
        self.m.restore();self.assertIs(G.subprocess,original);self.assertIs(G.read,read);DETAILS['git_reader_counts']=rows
    def test_original_observer_provider_methods_and_inert_failure(self):
        O=load(OBS,'independent_exact_observer');B=load(PROV,'independent_exact_provider')
        o=object.__new__(O.NativeLibraryObserver);o.check=lambda:None;o.value={};o.context=object();o.contract=types.SimpleNamespace(context_value=lambda ctx:{});o._binding={'inert':1};o.birth=None;o.owner={}
        o.transport=types.SimpleNamespace(abi_binding=lambda:{'inert':1},self_identity=lambda:{'pid':2,'creation_filetime_100ns':3},verify_preloaded_bootstrap=lambda v:None,verify_owned_birth=lambda *args:None)
        oboot=O.NativeLibraryObserver.bootstrap;self.assertEqual(self.m.api(O,o,'bootstrap','independent')['stage'],'independent');self.assertIs(O.NativeLibraryObserver.bootstrap,oboot)
        provider=object.__new__(B.ManagedWin32LibraryProvider);provider.directories=['inert'];provider.system_directory='inert';provider.closed=False
        provider.bootstrap_verifier=lambda stage:{'kind':'VERIFIED_CURRENT_LIBRARY_BOOTSTRAP_PROVENANCE','stage':stage,'fresh_process_identity_verified':True,'provider_source_verified':True,'preloaded_sources_verified':True,'module_event_scope_provenance_verified':True}
        class Buffer:
            value=''
            def __len__(self):return 32768
        provider.ctypes=types.SimpleNamespace(set_last_error=lambda n:None,get_last_error=lambda:0,create_unicode_buffer=lambda n:Buffer());provider.kernel=types.SimpleNamespace(GetDllDirectoryW=lambda *args:0)
        scope=B.ManagedWin32LibraryProvider.scope;self.assertEqual(self.m.api(B,provider,'scope')['kind'],'OBSERVED_MANAGED_DLL_DIRECTORY_SCOPE');self.assertIs(B.ManagedWin32LibraryProvider.scope,scope)
        o.transport.self_identity=lambda:(_ for _ in ()).throw(RuntimeError('independent inert failure'))
        with self.assertRaisesRegex(RuntimeError,'independent inert failure'):self.m.api(O,o,'bootstrap','failed')
        self.assertEqual(self.costs['API_calls'],3);rows=self.m.snapshot()['review_boundary'];self.assertEqual(rows['API.NativeLibraryObserver.bootstrap'],{'attempted':2,'returned':1,'failed':1,'pending':0,'bytes':0})
        self.assertNotIn('bootstrap',vars(o));self.assertNotIn('scope',vars(provider));DETAILS['method_counts']=rows
    def test_original_core_repeat_dose_refuses_at_final_seal(self):
        self.m.leave();ledger,phases=W._entry_journal_bootstrap(self.root);P=load(WORK/'prefix_native_pipeline.py','independent_repeat_pipeline');F=load(W.C.CORE/'fixture_support.py','independent_repeat_fixture')
        for phase in phases[:5]:ledger.startup(phase,lambda:None)
        run,core,proof=P._bind_shared_core_executor(W,ledger);api=F.InjectedAPI();report=run(api,api.guard,self.root)
        self.assertEqual(report['status'],'CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED')
        ledger.call(core['C'].OPS[0],lambda:None);ledger.startup(phases[5],lambda:None);saved=ledger.finish_owned(report)
        self.assertEqual(saved['operations'][core['C'].OPS[0]],{'attempted':2,'returned':2,'failed':0,'pending':0});self.assertFalse(saved['whole_scope_success_validated']);self.assertEqual(report['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED');self.assertFalse(saved['native_PASS'])
        DETAILS['repeated_core_dose']={'operation':core['C'].OPS[0],'counters':saved['operations'][core['C'].OPS[0]],'markers':saved['markers'],'status':saved['status']}
    def test_same_original_writer_native_mode_failure_shape_without_permit(self):
        self.m.leave();ledger,phases=W._entry_journal_bootstrap(self.root);ledger.entry_scope='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'
        def source(state,costs):
            C=load(WORK/'prefix_tensor_contract.py','independent_entry_contract');state['source_meter'].bind_contract(C);p=self.root/'actual.py';p.write_bytes(b'y=4');C.read_bytes(p);raise ValueError('closed independent native-mode fixture')
        actions={p:lambda *args:None for p in phases};actions[phases[0]]=source;actions[phases[5]]=lambda state,costs:state['source_meter'].restore()
        with self.assertRaisesRegex(ValueError,'closed independent native-mode fixture'):W._entry_phase_body(ledger,phases,actions,lambda *args:self.fail('must not execute'),_native_mode=True)
        saved=json.loads((self.root/'PHASE_ACCOUNTING_FINAL.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['scope'],'ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY');self.assertEqual(saved['status'],'NATIVE_INPUT_CHECKS_FAILED');self.assertFalse(saved['native_PASS']);self.assertFalse(saved['whole_scope_success_validated']);self.assertEqual(saved['selected_boundary_counters'][phases[0]]['contract.read_bytes']['bytes'],3)
        with self.assertRaisesRegex(ValueError,'no native entry'):W.C.stage_native_gate()
        DETAILS['native_mode_failure']={'scope':saved['scope'],'status':saved['status'],'native_PASS':saved['native_PASS'],'whole_scope_success_validated':saved['whole_scope_success_validated']}

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    sys.meta_path.remove(TRAP)
    after={p:{'bytes':(WORK/p).stat().st_size,'sha256':hashlib.sha256((WORK/p).read_bytes()).hexdigest()} for p in EXPECTED}
    receipt={'kind':'INDEPENDENT_FROZEN_SELECTED_BOUNDARY_METER_REVIEW_002','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'frozen_source_files_unchanged':after==PINS_BEFORE,'source_pins_before':PINS_BEFORE,'source_pins_after':after,'details':DETAILS,'native_import_trap_active_during_all_controls':True,'native_API_or_framework_called':False,'native_authority_issued':False,'whole_native_program_accounting_proven':False,'provider_install_phase_placement_reviewed':False,'fixture_package_path_adaptation':{'function':'_entry_journal_bootstrap','one_AST_local_package_expression_replaced':True,'selected_PACKAGE':str(PACKAGE),'candidate_bytes_modified':False},'initial_failed_fixture_review_preserved':'metering002_initial_fixture_failure','scope':'actual selected reader/hash/Git/original method/writer boundaries with inert file/backend fixtures; no original native closures executed'}
    receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (REVIEW/'INDEPENDENT_METERING_REVIEW002_SUCCESSOR.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() and after==PINS_BEFORE else 1)
