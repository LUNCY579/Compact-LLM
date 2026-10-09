"""Small stdlib controls; fake CUDA only, no model/runtime binaries or APIs."""
from copy import deepcopy
from contextlib import contextmanager
import importlib.util,json,os,subprocess
from pathlib import Path
import sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='short_phase_collection_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(CODE))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
contract=load('short_phase_contract',CODE/'short_phase_contract.py')
obs=load('short_phase_observation',CODE/'short_phase_observation.py')
worker=load('short_phase_worker',CODE/'short_phase_worker.py')
supervisor=load('short_phase_supervisor',CODE/'short_phase_supervisor.py')
from inference_qual_worker import make_guard
from native_worker import preserve_terminal,Journal
from rs_contract import digest
GIB=1024**3

class CUDA:
    def __init__(self):self.calls=[];self.initialized=False;self.error=None;self.free=5*GIB
    def is_initialized(self):self.calls.append('is_initialized');return self.initialized
    def init(self):self.calls.append('init');self.initialized=True
    def mem_get_info(self):
        self.calls.append('mem_get_info')
        if self.error is not None:raise self.error
        return (self.free,8*GIB)
    def max_memory_allocated(self):self.calls.append('max_allocated');return 0
    def max_memory_reserved(self):self.calls.append('max_reserved');return 0
    def manual_seed_all(self,value):self.calls.append(('CUDA_seed',value))
    def reset_peak_memory_stats(self):self.calls.append('reset_peaks')

def fixture(directory):
    sink=obs.PhaseSink(directory);rec=obs.Recorder(sink);cuda=CUDA();torch=SimpleNamespace(cuda=cuda)
    proxy=obs.GuardTorchProxy(torch,rec)
    actual=make_guard(directory,{'job_name':'CPU_FAKE_JOB','supervisor_pid':7},0.,lambda:proxy,
        owned_resources=lambda name:{'tree_rss_bytes':GIB},available=lambda:4*GIB,monotonic=lambda:1.,wall=lambda:1.,
        heartbeat_reader=lambda path:{'supervisor_pid':7,'unix_seconds':1.})
    return sink,rec,cuda,torch,obs.observed_guard(actual,rec)

sys.path.insert(0,str(PACKAGE/'tests'))
legacy=load('legacy_short_phase_fixture',PACKAGE/'tests/test_short_collection.py')

def collection_fixture(directory,*,observed=True):
    sink=obs.PhaseSink(directory);rec=obs.Recorder(sink)
    with patch.object(legacy,'Journal',lambda path:obs.PhaseJournal(path,rec) if observed else Journal(path)):
        f=list(legacy.WorkerTests().fixture(directory))
    torch,model,decoder,config,journal,dose,report,helpers=f
    cuda=CUDA();torch.cuda=cuda;torch.manual_seed=lambda value:None;holder=[None]
    p=sum(32*r['prompt_tokens']+496 for r in config['rows']);config['dose'].update(maximum_capture_input_positions=p,
        maximum_forward_input_positions=p+2*config['rows'][0]['prompt_tokens'])
    config['runtime_versions']=deepcopy(contract.native.RUNTIME);config['seed']=101
    actual=make_guard(directory,{'job_name':'CPU_FAKE_JOB','supervisor_pid':7},0.,
        lambda:None if holder[0] is None else obs.GuardTorchProxy(torch,rec) if observed else torch,
        owned_resources=lambda name:{'tree_rss_bytes':GIB},available=lambda:4*GIB,monotonic=lambda:1.,wall=lambda:1.,
        heartbeat_reader=lambda path:{'supervisor_pid':7,'unix_seconds':1.})
    guard=obs.observed_guard(actual,rec) if observed else actual;journal.guard=guard
    api={'disk_free':lambda:4*GIB,'torch_import':lambda:torch,'Qwen3Config_symbol_import':lambda:'CPU_CONFIG_SYMBOL',
         'Qwen3ForCausalLM_symbol_import':lambda:'CPU_MODEL_SYMBOL','framework_version_read':lambda:{k:contract.native.RUNTIME[k] for k in ('torch','transformers')},
         'growth_observation':lambda:{'available_bytes':4*GIB,'worker_RSS_bytes':GIB},'configure':lambda torch:None,
         'Python_seed':lambda seed:None,'load_titan':lambda *args:(journal.call('model_construct',lambda:model),{'scope':'CPU_FIXTURE'},{}),
         'decoder_construct':lambda config:decoder,'final_preflight':lambda:None}
    return f,sink,rec,cuda,holder,guard,api

def original_sequence(f,holder,guard,api):
    """Original caller's query/guard order, independent of the staged pipeline."""
    torch,model,decoder,config,journal,dose,report,helpers=f
    guard('pre_import');assert api['disk_free']()>=4*GIB;holder[0]=api['torch_import']()
    c=api['Qwen3Config_symbol_import']();m=api['Qwen3ForCausalLM_symbol_import']();api['framework_version_read']()
    growth=api['growth_observation']();assert growth['available_bytes']>=1.25*GIB+max(0,2*GIB-growth['worker_RSS_bytes'])
    journal.call('cuda_context_init',torch.cuda.init);assert torch.cuda.mem_get_info()[0]>=4*GIB
    api['configure'](torch);api['Python_seed'](101);torch.manual_seed(101);torch.cuda.manual_seed_all(101);torch.cuda.reset_peak_memory_stats()
    model,loader,hashes=api['load_titan'](torch,c,m,config,journal.call,guard);report['loader']=loader;journal.evidence('loader',loader)
    decoder=journal.call('generated_decoder_construct',lambda:api['decoder_construct'](config));dose=worker.ForwardDose(torch,config)
    worker.original.collect_loaded(torch,model,decoder,config,hashes,journal,guard,report,dose,device='cpu',**helpers)
    api['final_preflight']();guard('completed')

def complete_collection(directory):
    f,sink,rec,cuda,holder,guard,api=collection_fixture(directory)
    torch,model,decoder,config,journal,dose,report,helpers=f
    worker.pipeline(config,api,holder,journal,rec,guard,report,collection_kwargs={'device':'cpu',**helpers})
    identity={'execution_scope':'CPU_FIXTURE_PHASE_COLLECTION_PROTOCOL','purpose':contract.PURPOSE,'config_sha256':'f'*64,
              'source_commit':'CPU_FIXTURE_COMMIT','batch_selector':{'source_kind':'atomic_RS_DEV','ordinal':4}}
    report.update(status='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED',error=None,scope=contract.PURPOSE,
        config_sha256=identity['config_sha256'],source_commit=identity['source_commit'],operation_accounting=journal.snapshot())
    obs.span(rec,'main_preserve_terminal',lambda:preserve_terminal(Path(directory),report,journal,obs.TerminalTorchProxy(torch,rec)),before_health=False)
    final=obs.finish(directory,rec,sink,identity)
    events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
    return f,cuda,obs.read_events(sink.path),final,report,events,identity

def scope_fixture():
    value={key:None for key in contract.BASE_FIELDS};rows=[legacy.row(i) for i in range(64)]
    prior=contract.native.read_json(PACKAGE/'configs/SYNTH_NQ001_PROC7.json');ram=contract.base.successor_profile()
    p=sum(32*r['prompt_tokens']+496 for r in rows);parity=2*rows[0]['prompt_tokens']
    value.update(schema_version=1,id='CPU_FIXTURE_ORIGINAL',contract=contract.base.VERSION,authorization=deepcopy(contract.base.AUTHORIZATION),
        profile=contract.base.qualified.PROFILE,resources=deepcopy(contract.native.RESOURCE_BOUNDS),tolerances=deepcopy(contract.native.TOLERANCES),
        runtime_versions={**contract.native.RUNTIME,'tokenizers':'0.22.2'},runtime_pins=contract.base.merge_runtime_inventory(prior['runtime_pins'],ram),
        decoder_runtime_pins={'CPU_DECODER_%02d'%i:'b'*64 for i in range(17)},data_pins={'CPU_DATA_%02d'%i:'c'*64 for i in range(5)},
        model_pins=dict(contract.base.ASSET_PINS),model_dir=str(contract.base.MODEL_DIR),offline_environment=deepcopy(contract.native.OFFLINE_ENV),
        precision=prior['precision'],seed=101,model_role='fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint',
        batch_selector={'source_kind':'atomic_RS_DEV','ordinal':4},batch={'rows':deepcopy(rows),'source_packet_sha256':digest(rows)},
        rows=rows,source_packet_sha256=digest(rows),short_RAM_monitor_profile=ram,
        dose={'context_cap':512,'maximum_generated_tokens':32,'rows':64,'maximum_capture_forwards':2048,
            'maximum_capture_input_positions':p,'maximum_parity_forwards':2,'maximum_parity_input_positions':parity,
            'maximum_forward_calls':2050,'maximum_forward_input_positions':p+parity,'optimizer_steps':0,'backwards':0})
    for key in ('correctness_certified','retention_certified','training_authorized','scoring_authorized','language_qualification_established',
        'long_context_qualification_established','prior_correct_enrollment_performed','protected_bank_access','paid_services','confirmation_eligible','quality_gain_established'):value[key]=False
    return contract.amend_scope(value)

class Controls(unittest.TestCase):
    def test_exact_imported_collector_decoder_math_and_observer_identity(self):
        for name in ('ForwardDose','LocalGeneratedDecoder','DurableRawCollection','collect_loaded'):
            self.assertIs(getattr(worker,name),getattr(worker.original,name))
        import context_phase_worker
        self.assertIs(obs.Recorder,context_phase_worker.Recorder);self.assertIs(obs.GuardTorchProxy,context_phase_worker.GuardTorchProxy)
        self.assertEqual(Path(worker.original.__file__).resolve().parent,PACKAGE.resolve())
        for module in (contract,obs,worker,supervisor):self.assertEqual(Path(module.__file__).resolve().parent,CODE.resolve())
    def test_stage_entries_close_before_paths_artifacts_or_native(self):
        with patch.object(contract,'STAGED',True):
            for fn in (lambda:contract.write_scope('UNUSED','atomic_RS_DEV',4),lambda:contract.preflight('UNUSED'),
                       lambda:worker.run('UNUSED','UNUSED'),lambda:supervisor.dispatch('UNUSED','UNUSED')):
                with self.assertRaisesRegex(ValueError,'staged phase-observed'):fn()
        self.assertFalse(any(n in sys.modules for n in ('torch','transformers','tokenizers')))
    def test_guard_false_true_exact_once_and_complete_nested_replay(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink,rec,cuda,torch,guard=fixture(d);guard('false');cuda.initialized=True;guard('true')
            self.assertEqual(['is_initialized','is_initialized','mem_get_info'],cuda.calls)
            events=obs.read_events(sink.path);proof=obs.validate_events(events,rec.snapshot())
            self.assertEqual((2,2,1),(proof['original_guards'],proof['is_initialized_queries'],proof['guard_memory_queries']))
            final=obs.finish(d,rec,sink,{'execution_scope':'CPU_FIXTURE_ONLY'})
            self.assertEqual('COMPLETE_PHASE_OBSERVATION',final['status']);self.assertTrue(final['phase_IO']['closed'])
            self.assertEqual(final['phase_IO'],sink.snapshot())
    def test_same_query_result_exception_and_IO_error_priority(self):
        result=object();error=OSError('EXACT_ACTUAL_QUERY');calls=[]
        rec=obs.Recorder(lambda *args:(_ for _ in ()).throw(OSError('RECORDING_IO')))
        proxy=obs.GuardTorchProxy(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:calls.append(1) or result,
            mem_get_info=lambda:(_ for _ in ()).throw(error))),rec)
        self.assertIs(proxy.cuda.is_initialized(),result)
        with self.assertRaises(OSError) as got:proxy.cuda.mem_get_info()
        self.assertIs(got.exception,error);self.assertEqual([1],calls);self.assertTrue(rec.errors)
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink,rec,cuda,torch,guard=fixture(d);cuda.initialized=True;cuda.error=error
            rec.preserve=lambda *args:(_ for _ in ()).throw(OSError('RECORDING_IO'))
            with self.assertRaises(OSError) as got:guard('error')
            self.assertIs(got.exception,error);self.assertEqual(['is_initialized','mem_get_info'],cuda.calls);sink.close()
    def test_original_floors_deadline_heartbeat_unchanged(self):
        for rss,available,now,heartbeat in ((2*GIB+1,4*GIB,1.,1.),(GIB,int(1.25*GIB)-1,1.,1.),(GIB,4*GIB,601.,1.),(GIB,4*GIB,1.,-2.)):
            rec=obs.Recorder(lambda *args:None)
            actual=make_guard(HERE,{'job_name':'FAKE','supervisor_pid':7},0.,lambda:None,
                owned_resources=lambda name,rss=rss:{'tree_rss_bytes':rss},available=lambda available=available:available,
                monotonic=lambda now=now:now,wall=lambda:1.,heartbeat_reader=lambda path,h=heartbeat:{'supervisor_pid':7,'unix_seconds':h})
            with self.assertRaises(ValueError):obs.observed_guard(actual,rec)('denied')
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink,rec,cuda,torch,guard=fixture(d);cuda.initialized=True;cuda.free=int(.5*GIB)-1
            with self.assertRaisesRegex(ValueError,'GPU free floor'):guard('floor')
            sink.close()
    def test_primitive_return_before_post_return_guard_failure(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink,rec,cuda,torch,guard=fixture(d);journal=obs.PhaseJournal(d,rec);journal.guard=guard
            error=OSError('POST_RETURN_GUARD')
            def init():cuda.init();cuda.error=error
            with self.assertRaises(OSError) as got:journal.call('cuda_context_init',init)
            self.assertIs(got.exception,error)
            self.assertEqual(1,rec.operations['primitive_cuda_context_init']['returned'])
            self.assertEqual(1,rec.operations['api_cuda_context_init']['failed'])
            self.assertEqual(1,journal.snapshot()['operations']['cuda_context_init']['returned'])
            self.assertEqual(['is_initialized','init','is_initialized','mem_get_info'],cuda.calls)
            journal.stream.close();sink.close()
    def test_marker_partial_write_and_fsync_failures_retained_closed(self):
        for failure in ('partial','fsync'):
            with tempfile.TemporaryDirectory(dir=HERE) as d:
                sink=obs.PhaseSink(d);rec=obs.Recorder(sink)
                if failure=='partial':sink.writer=lambda raw:sink.stream.write(raw[:7])
                else:sink.fsync=lambda fd:(_ for _ in ()).throw(OSError('FSYNC_FAILED'))
                value=obs.span
                with self.assertRaisesRegex(RuntimeError,'publication failed'):value(rec,'main_probe',lambda:None)
                self.assertGreater(sink.operations['phase_marker_write']['failed'],0)
                final=obs.finish(d,rec,sink,{'execution_scope':'CPU_FIXTURE_ONLY'})
                self.assertEqual('FAILED_PHASE_OBSERVATION',final['status'])
                with self.assertRaises(ValueError):obs.validate_events(obs.read_events(sink.path),rec.snapshot())
    def test_close_and_final_write_failures_never_complete(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink=obs.PhaseSink(d);rec=obs.Recorder(sink);obs.span(rec,'main_probe',lambda:None);real=sink.stream
            sink.stream=SimpleNamespace(closed=False,close=lambda:(_ for _ in ()).throw(OSError('CLOSE_FAILED')))
            final=obs.finish(d,rec,sink,{'execution_scope':'CPU_FIXTURE_ONLY'});real.close()
            self.assertEqual('FAILED_PHASE_OBSERVATION',final['status']);self.assertFalse(final['phase_IO']['closed'])
            self.assertEqual(1,final['phase_IO']['operations']['phase_ledger_close']['failed'])
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink=obs.PhaseSink(d);rec=obs.Recorder(sink);obs.span(rec,'main_probe',lambda:None);opened=Path.open
            def fail(path,*args,**kwargs):
                if path.name==obs.FINAL_NAME:raise OSError('FINAL_RECEIPT_FAILED')
                return opened(path,*args,**kwargs)
            with patch.object(Path,'open',fail):
                with self.assertRaisesRegex(OSError,'FINAL_RECEIPT_FAILED'):obs.finish(d,rec,sink,{'execution_scope':'CPU_FIXTURE_ONLY'})
            self.assertFalse((Path(d)/obs.FINAL_NAME).exists());self.assertEqual(1,sink.operations['phase_final_summary_write']['failed'])
    def test_pending_duplicate_reordered_nonfinite_and_altered_phase_denied(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink,rec,cuda,torch,guard=fixture(d);cuda.initialized=True;guard('source')
            good=obs.read_events(sink.path);obs.validate_events(good,rec.snapshot())
            bads=[good+[deepcopy(good[0])],list(reversed(good)),deepcopy(good),deepcopy(good)]
            bads[2][0]['phase']['monotonic']=float('nan');bads[3][0]['phase']['operation']='ALTERED'
            for bad in bads:
                with self.assertRaises((ValueError,KeyError)):obs.validate_events(bad,rec.snapshot())
            rec.begin('pending')
            with self.assertRaises(ValueError):obs.validate_events(obs.read_events(sink.path),rec.snapshot())
            final=obs.finish(d,rec,sink,{'execution_scope':'CPU_FIXTURE_ONLY'});self.assertEqual('FAILED_PHASE_OBSERVATION',final['status'])
    def test_terminal_helper_swallowed_query_error_still_fails_final_accounting(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            sink,rec,cuda,torch,guard=fixture(d);cuda.initialized=True;cuda.error=OSError('TERMINAL_QUERY_FAILED')
            journal=obs.PhaseJournal(d,rec);report={'status':'COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED'}
            obs.span(rec,'main_preserve_terminal',lambda:preserve_terminal(Path(d),report,journal,obs.TerminalTorchProxy(torch,rec)),before_health=False)
            self.assertEqual('COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED',json.loads((Path(d)/'worker_result.json').read_text())['status'])
            self.assertIn('diagnostic_error',json.loads((Path(d)/'worker_GPU_final.json').read_text()))
            self.assertEqual(1,rec.operations['terminal_cuda_mem_get_info']['failed'])
            self.assertEqual(['is_initialized','max_allocated','max_reserved','mem_get_info'],cuda.calls)
            final=obs.finish(d,rec,sink,{'execution_scope':'CPU_FIXTURE_ONLY'});self.assertEqual('FAILED_PHASE_OBSERVATION',final['status'])

    def test_full64_original_math_phase_main_roster_and_journal_publication_joins(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            f,cuda,entries,final,report,events,identity=complete_collection(d)
            proof=obs.validate_collection_events(entries,final,report,events,identity,fixture=True)
            self.assertTrue(proof['fixture_only']);self.assertFalse(proof['full_raw_decoder_model_resource_audit_complete'])
            self.assertEqual(64,len(report['collection']['rows']));self.assertEqual(194,len(f[1].calls))
            self.assertEqual(194,report['forward_dose']['attempted_calls']);self.assertEqual(580,report['forward_dose']['input_positions'])
            self.assertEqual([[9,10,151645]]*64,f[2].calls);self.assertFalse(f[1].prehooks)
            self.assertEqual(192,len(list(Path(d).glob('stage_raw_row*_token*.json'))))
            self.assertEqual(64,len(list(Path(d).glob('stage_inference_row_*.json'))))
            self.assertEqual(proof['is_initialized_queries']+1,cuda.calls.count('is_initialized'))
            self.assertEqual(proof['guard_memory_queries']+2,cuda.calls.count('mem_get_info'))
            with self.assertRaisesRegex(ValueError,'fixture/native'):obs.validate_collection_events(entries,final,report,events,identity,fixture=False)
    def test_exact_guard_and_all_CUDA_calls_match_original_unobserved_Journal_caller(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d1,tempfile.TemporaryDirectory(dir=HERE) as d2:
            f,cuda,entries,final,report,events,identity=complete_collection(d1)
            old,sink,rec,old_cuda,holder,guard,api=collection_fixture(d2,observed=False)
            original_sequence(old,holder,guard,api)
            preserve_terminal(Path(d2),{'status':'CPU_FIXTURE'},old[4],old[0]);sink.close()
            self.assertEqual(old_cuda.calls,cuda.calls)
            self.assertEqual(old[4].snapshot(),report['operation_accounting'])
            self.assertEqual(old[6]['collection'],report['collection'])
    def test_main_roster_Journal_cost_and_final_status_closed_mutations_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            f,cuda,entries,final,report,events,identity=complete_collection(d)
            for mutate in (lambda x:x['phase_IO'].update(closed=False),lambda x:x.update(status='FAILED_PHASE_OBSERVATION'),
                lambda x:x['phase_accounting']['operations']['main_load_titan'].update(returned=0)):
                bad=deepcopy(final);mutate(bad);bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
                with self.assertRaises(ValueError):obs.validate_collection_events(entries,bad,report,events,identity,fixture=True)
            bad_report=deepcopy(report);bad_report['operation_accounting']['operations']['inference_forward']['returned']-=1
            with self.assertRaisesRegex(ValueError,'accounting'):obs.validate_collection_events(entries,final,bad_report,events,identity,fixture=True)
            bad_events=deepcopy(events);next(v for v in bad_events if v['event']=='attempt')['operation']='HIDDEN_OPERATION'
            with self.assertRaisesRegex(ValueError,'order joins'):obs.validate_collection_events(entries,final,report,bad_events,identity,fixture=True)
            # Erase an API guard and nested queries, then re-number, re-account
            # bytes/counts and re-seal. Denial must be source-order admission.
            stack=[];remove=None
            for i,marker in enumerate(entries):
                p=marker['phase']
                if p['event']=='attempt':
                    if p['operation']=='guard_before_inference_forward' and stack and stack[-1][1]=='api_inference_forward':
                        remove=(i,p['ordinal']);break
                    stack.append((p['ordinal'],p['operation']))
                else:stack.pop()
            begin,ordinal=remove;end=next(i for i in range(begin+1,len(entries)) if entries[i]['phase']['ordinal']==ordinal and entries[i]['phase']['event']=='returned')
            erased=deepcopy(entries[:begin]+entries[end+1:]);mapping={};costs={}
            for marker in erased:
                p=marker['phase'];old=p['ordinal'];name=p['operation']
                if p['event']=='attempt':mapping[old]=len(mapping)+1;costs.setdefault(name,obs.counts());costs[name]['attempted']+=1;costs[name]['returned']+=1
                p['ordinal']=mapping[old];marker['name']='phase_%04d_%s'%(p['ordinal'],'start' if p['event']=='attempt' else 'end')
            sealed=deepcopy(final);sealed['phase_accounting'].update(operations=costs,last_ordinal=len(mapping),last_phase=erased[-1]['phase'])
            size=sum(len((obs.canonical(v)+'\n').encode()) for v in erased);sealed['phase_IO'].update(markers_returned=len(erased),marker_bytes_attempted=size,marker_bytes_returned=size)
            sealed['phase_IO']['operations']['phase_marker_write'].update(attempted=len(erased),returned=len(erased))
            sealed['sha256']=digest({k:v for k,v in sealed.items() if k!='sha256'})
            obs.validate_events(erased,sealed['phase_accounting'])
            with self.assertRaisesRegex(ValueError,'every original API'):obs.validate_collection_events(erased,sealed,report,events,identity,fixture=True)
    def test_marker_failure_before_and_after_actual_forward_preserves_cost_and_stops(self):
        for event in ('attempt','returned'):
            with tempfile.TemporaryDirectory(dir=HERE) as d:
                f,sink,rec,cuda,holder,guard,api=collection_fixture(d);writer=sink.writer
                def fail(raw):
                    marker=json.loads(raw)['phase']
                    if marker['operation']=='primitive_inference_forward' and marker['event']==event:raise OSError('FORWARD_MARKER_IO')
                    return writer(raw)
                sink.writer=fail
                with self.assertRaisesRegex(RuntimeError,'publication failed'):
                    worker.pipeline(f[3],api,holder,f[4],rec,guard,f[6],collection_kwargs={'device':'cpu',**f[-1]})
                self.assertEqual(2 if event=='attempt' else 3,len(f[1].calls))
                self.assertEqual(2 if event=='attempt' else 3,f[6]['forward_dose']['attempted_calls'])
                self.assertEqual(0 if event=='attempt' else 1,f[6]['collection_costs']['input_positions']['returned']//2)
                self.assertFalse(f[1].prehooks);f[4].stream.close();sink.close()
    def test_four_GiB_preload_and_growth_reserve_before_loader_unchanged(self):
        for failed in ('preload','growth'):
            with tempfile.TemporaryDirectory(dir=HERE) as d:
                f,sink,rec,cuda,holder,guard,api=collection_fixture(d)
                if failed=='preload':cuda.free=3*GIB
                else:api['growth_observation']=lambda:{'available_bytes':2*GIB,'worker_RSS_bytes':GIB}
                with self.assertRaises(ValueError):worker.pipeline(f[3],api,holder,f[4],rec,guard,f[6],collection_kwargs={'device':'cpu',**f[-1]})
                self.assertEqual([],f[1].calls);self.assertNotIn('main_load_titan',rec.operations)
                self.assertEqual(1 if failed=='preload' else 0,cuda.calls.count('init'));f[4].stream.close();sink.close()
    def test_budget_cross_process_hashseed_equality_and_predecessor_negative(self):
        script="import sys,json;sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[1]);import short_phase_contract as c;print(json.dumps(c.observation_budget(),sort_keys=True))"
        budgets=[]
        for seed in ('1','999'):
            # -I intentionally ignores PYTHONHASHSEED; -B with an explicitly
            # supplied module path lets this control change the hash seed.
            process=subprocess.run([sys.executable,'-B','-c',script,str(CODE),str(PACKAGE)],env={**os.environ,'PYTHONHASHSEED':seed},
                capture_output=True,text=True,timeout=30)
            self.assertEqual(0,process.returncode,process.stderr);budgets.append(json.loads(process.stdout))
        self.assertEqual(budgets[0],budgets[1]);self.assertEqual(contract.observation_budget(),budgets[0])
        # Meaningful negative: reproduce the preserved predecessor's unsorted
        # tie selection in memory, never editing source or native authority.
        previous="import sys,inspect,json;sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[1]);import short_phase_contract as c;s=inspect.getsource(c.observation_budget).replace('key=lambda name:(len(name),name)','key=len');exec(s,c.__dict__);print(json.dumps(c.observation_budget(),sort_keys=True))"
        old=[]
        for seed in ('1','999'):
            process=subprocess.run([sys.executable,'-B','-c',previous,str(CODE),str(PACKAGE)],env={**os.environ,'PYTHONHASHSEED':seed},
                capture_output=True,text=True,timeout=30)
            self.assertEqual(0,process.returncode,process.stderr);old.append(json.loads(process.stdout))
        self.assertNotEqual(old[0]['longest_registered_operation'],old[1]['longest_registered_operation'])
    def test_new_scope_whole64_source_runtime_model_and_authority_denials(self):
        scope=scope_fixture();contract.validate_scope(scope)
        mutations=(lambda b:b['rows'].pop(),lambda b:b['rows'].reverse(),lambda b:b['rows'][-1].update(target='PRIVATE'),
            lambda b:b['authorization'].update(scoring=True),lambda b:b['dose'].update(maximum_forward_calls=2051),
            lambda b:b['runtime_pins'].pop(next(k for k in b['runtime_pins'] if Path(k).name=='ast.py')),
            lambda b:b['model_pins'].update({'model.safetensors':'0'*64}),lambda b:b['batch_selector'].update(ordinal=True),
            lambda b:b['short_phase_observation_profile'].update(additional_CUDA_queries=1))
        for mutate in mutations:
            bad=deepcopy(scope);mutate(bad)
            with self.assertRaises(ValueError):contract.validate_scope(bad)
        for name in ('short_phase_worker.py','short_phase_observation.py'):
            bad=deepcopy(scope);bad['source_pins'].pop(str((CODE/name).resolve()))
            with self.assertRaisesRegex(ValueError,'roster'):contract.validate_scope(bad)
    def test_independent_inventory_all_paths_hashes_and_old_unknown_profiles_rejected(self):
        scope=scope_fixture();sections={'source':'source_pins','runtime':'runtime_pins','decoder':'decoder_runtime_pins','data':'data_pins','model':'model_pins'}
        inventory={'id':contract.REGISTRATION_ID,'contract':contract.VERSION,'source_commit':'CPU_FIXED_COMMIT','config_file_sha256':'f'*64,
            'profile_body_sha256':digest(scope['short_phase_observation_profile']),
            'inventories':{section:deepcopy(scope[key]) for section,key in sections.items()},
            'counts':{section:len(scope[key]) for section,key in sections.items()}}
        inventory['sha256']=digest(inventory)
        arguments={'expected_profile_sha':inventory['profile_body_sha256'],'expected_config_sha':'f'*64,'expected_commit':'CPU_FIXED_COMMIT'}
        self.assertTrue(contract.validate_inventory(inventory,scope,**arguments))
        for mutate in (lambda b:b.update(id='OLD_RAM2'),lambda b:b['inventories']['source'].pop(next(iter(b['inventories']['source']))),
            lambda b:b['inventories']['runtime'].update({next(iter(b['inventories']['runtime'])):'0'*64}),lambda b:b.update(profile_body_sha256='0'*64)):
            bad=deepcopy(inventory);mutate(bad);bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
            with self.assertRaises(ValueError):contract.validate_inventory(bad,scope,**arguments)
    def test_relevant_existing_short_worker_controls_against_staged_alias_and_observed_Journal(self):
        sinks=[]
        def journal(path):
            sink=obs.PhaseSink(path);sinks.append(sink);return obs.PhaseJournal(path,obs.Recorder(sink))
        methods=('test_full64_mock_collection_decode_generated_only_durable_tokens_no_scoring',
            'test_token_partial_before_actual32cap_is_not_marked_cap_truncated',
            'test_all64_prevalidate_last_oracle_or_overflow_before_any_forward',
            'test_forward_failure_preserves_row_tokens_costs_and_no_next_row',
            'test_return_sink_failure_keeps_actual_model_return_costs_and_stops',
            'test_partial_sink_failure_sticky_halt_with_no_next_forward',
            'test_zero_B_or_base_byte_failure_cannot_complete_collection','test_actual_perhook2050_globalcaps_refuse_before_next_model_attempt')
        temporary=tempfile.TemporaryDirectory
        @contextmanager
        def directory(*args,**kwargs):
            with temporary(*args,**kwargs) as path:
                try:yield path
                finally:
                    # Legacy tests close Journal before their own temporary
                    # directory exits. This added independent ledger must also
                    # close there; production keeps it through terminal queries.
                    for sink in sinks:
                        if sink.path.parent==Path(path):sink.close()
        try:
            with patch.object(legacy,'worker',worker),patch.object(legacy,'Journal',journal),patch.object(legacy,'WORK',HERE),patch.object(legacy.tempfile,'TemporaryDirectory',directory):
                for name in methods:getattr(legacy.WorkerTests(),name)()
        finally:
            for sink in sinks:sink.close()

    def test_full64_32step_dose_and_prospective_native_marker_byte_bound(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            f,sink,rec,cuda,holder,guard,api=collection_fixture(d)
            # Storage timing is not being benchmarked: other controls exercise
            # real fsync/errors. Every marker still has actual encoded bytes.
            sink.fsync=lambda fd:None;f[1].tokens=[9]
            worker.pipeline(f[3],api,holder,f[4],rec,guard,f[6],collection_kwargs={'device':'cpu',**f[-1]})
            identity={'execution_scope':'CPU_FIXTURE_PHASE_COLLECTION_PROTOCOL','purpose':contract.PURPOSE,
                'config_sha256':'f'*64,'source_commit':'CPU_FIXTURE_COMMIT','batch_selector':{'source_kind':'atomic_RS_DEV','ordinal':4}}
            report=f[6];report.update(status='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED',error=None,scope=contract.PURPOSE,
                config_sha256=identity['config_sha256'],source_commit=identity['source_commit'],operation_accounting=f[4].snapshot())
            obs.span(rec,'main_preserve_terminal',lambda:preserve_terminal(Path(d),report,f[4],obs.TerminalTorchProxy(f[0],rec)),before_health=False)
            final=obs.finish(d,rec,sink,identity);entries=obs.read_events(sink.path)
            events=[json.loads(line) for line in (Path(d)/'events.jsonl').read_text().splitlines()]
            proof=obs.validate_collection_events(entries,final,report,events,identity,fixture=True)
            self.assertEqual(2050,report['forward_dose']['attempted_calls']);self.assertEqual(35844,report['forward_dose']['input_positions'])
            self.assertTrue(all(len(row['raw_generated_ids'])==32 and row['finish_reason']=='generation_cap' and row['truncated']
                for row in report['collection']['rows']))
            bound=contract.observation_budget()
            self.assertEqual(98442,bound['maximum_markers']);self.assertLessEqual(bound['conservative_success_ledger_bytes'],obs.LIMITS['maximum_ledger_bytes'])
            self.assertLessEqual(len(entries),bound['maximum_markers']);self.assertLessEqual(sink.bytes_returned,bound['conservative_success_ledger_bytes'])
            print(json.dumps({'CPU_FULL32_FIXTURE':{'physical_calls':2050,'input_positions':35844,'marker_count':len(entries),
                'marker_bytes':sink.bytes_returned,'maximum_actual_marker_bytes':max(len((obs.canonical(e)+'\n').encode()) for e in entries),
                'original_guards':proof['original_guards'],'real_fsync_timing_not_measured':True},'SOURCE_NATIVE_BUDGET':bound}))

if __name__=='__main__':unittest.main()
