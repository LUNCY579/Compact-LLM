"""Independent phase/IO accounting; no guards, model APIs or recursive Journal."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json,math,os,sys
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='short_phase_collection_work' and HERE.parent.name=='artifacts' else HERE
sys.path.insert(0,str(PACKAGE))
from context_phase_worker import Recorder,GuardTorchProxy,observed_guard
from native_worker import Journal
from rs_contract import canonical,digest,require

KIND='SYNTH_SHORT_PHASE_ACCOUNTING_V1'
LEDGER_NAME='PHASE_OBSERVATIONS.jsonl'
FINAL_NAME='PHASE_ACCOUNTING_FINAL.json'
LIMITS={'maximum_markers':200000,'maximum_marker_bytes':65536,'maximum_ledger_bytes':64*1024**2}
MAIN=('disk_admission','torch_import','Qwen3Config_symbol_import','Qwen3ForCausalLM_symbol_import',
      'framework_version_read','post_import_growth_observation','direct_CUDA_preload_mem_get_info',
      'precision_and_thread_configuration','Python_seed_101','Torch_CPU_seed_101','Torch_CUDA_seed_all_101',
      'CUDA_reset_peak_memory_stats','load_titan','forward_dose_construct','collect_loaded',
      'final_config_preflight','preserve_terminal')
API_FIXED={'checkpoint_load':1,'cuda_context_init':1,'generated_decoder_construct':1,'install_activation_hooks':1,
    'install_fp32_compute':1,'install_qv_lora':1,'model_construct':1,'tensor_copy':310,'verify_frozen_base':1,
    'zero_lora_query_after':1,'zero_lora_query_before':1,'zero_parity_adapter_forward':1,'zero_parity_base_forward':1}

def counts():return dict(attempted=0,returned=0,failed=0,pending=0)

class PhaseSink:
    """One exclusive append-only ledger; each complete marker is fsynced."""
    def __init__(self,output,*,write=None,fsync=os.fsync):
        self.path=Path(output)/LEDGER_NAME;self.operations={};self.markers=0
        self.bytes_attempted=self.bytes_returned=0;self.closed=False
        self.stream=self.io('phase_ledger_create',lambda:self.path.open('xb',buffering=0))
        self.writer=write or self.stream.write;self.fsync=fsync
    def io(self,name,function):
        item=self.operations.setdefault(name,counts());item['attempted']+=1;item['pending']+=1
        try:value=function()
        except BaseException:item['pending']-=1;item['failed']+=1;raise
        item['pending']-=1;item['returned']+=1;return value
    def __call__(self,name,value):
        raw=(canonical({'name':name,'phase':value})+'\n').encode('ascii');self.bytes_attempted+=len(raw)
        def write():
            require(not self.closed and self.markers<LIMITS['maximum_markers'] and len(raw)<=LIMITS['maximum_marker_bytes']
                and self.bytes_returned+len(raw)<=LIMITS['maximum_ledger_bytes'],'bounded open phase ledger')
            require(self.writer(raw)==len(raw),'complete phase marker write');self.stream.flush();self.fsync(self.stream.fileno())
            self.markers+=1;self.bytes_returned+=len(raw)
        return self.io('phase_marker_write',write)
    def close(self):
        if self.closed:return
        try:value=self.io('phase_ledger_close',self.stream.close)
        except BaseException:
            self.closed=bool(getattr(self.stream,'closed',False));raise
        self.closed=True;return value
    def snapshot(self):return deepcopy({'operations':self.operations,'markers_returned':self.markers,
        'marker_bytes_attempted':self.bytes_attempted,'marker_bytes_returned':self.bytes_returned,'closed':self.closed})

def span(recorder,name,function,*,result=False,before_health=True):
    """Same callable/result/error; no resource guard is added here."""
    ordinal=recorder.begin(name)
    try:
        if before_health:recorder.healthy()
        value=function()
    except BaseException as error:recorder.end(ordinal,name,error=error);raise
    recorder.end(ordinal,name,value=value if result else None);recorder.healthy();return value

class PhaseJournal(Journal):
    def __init__(self,output,recorder):super().__init__(Path(output));self.recorder=recorder
    def call(self,name,function):
        # The actual primitive has its own return before Journal's post-return
        # guard, so a later guard failure cannot erase that observation.
        original=super().call
        return span(self.recorder,'api_'+name,lambda:original(name,
            lambda:span(self.recorder,'primitive_'+name,function)),before_health=False)
    def evidence(self,name,value):
        original=super().evidence
        return span(self.recorder,'publication_'+name,lambda:original(name,value))

class TerminalCUDAProxy:
    """Observe only preserve_terminal's existing four queries, exactly once."""
    def __init__(self,actual,recorder):self.actual,self.recorder=actual,recorder
    def is_initialized(self):return self.recorder.call('terminal_cuda_is_initialized',self.actual.is_initialized)
    def max_memory_allocated(self):return self.recorder.call('terminal_cuda_max_memory_allocated',self.actual.max_memory_allocated)
    def max_memory_reserved(self):return self.recorder.call('terminal_cuda_max_memory_reserved',self.actual.max_memory_reserved)
    def mem_get_info(self):return self.recorder.call('terminal_cuda_mem_get_info',self.actual.mem_get_info)

class TerminalTorchProxy:
    def __init__(self,torch,recorder):self.cuda=TerminalCUDAProxy(torch.cuda,recorder)

def complete_counts(operations):
    return all(set(v)==set(counts()) and all(type(x) is int and x>=0 for x in v.values())
        and v['attempted']==v['returned'] and v['failed']==v['pending']==0 for v in operations.values())

def finish(output,recorder,sink,identity,*,primary_error=None,terminal_error=None):
    """After original terminal diagnostics; reconcile the final write itself."""
    close_error=None
    try:sink.close()
    except BaseException as error:close_error={'type':type(error).__name__,'message':str(error)}
    phase=recorder.snapshot();before=sink.snapshot();projected=deepcopy(before)
    projected['operations']['phase_final_summary_write']={'attempted':1,'returned':1,'failed':0,'pending':0}
    good=primary_error is terminal_error is close_error is None and not phase['publication_errors']
    good=good and complete_counts(phase['operations']) and complete_counts(before['operations'])
    body={'kind':KIND,'identity':deepcopy(identity),'status':'COMPLETE_PHASE_OBSERVATION' if good else 'FAILED_PHASE_OBSERVATION',
        'primary_error':primary_error,'terminal_error':terminal_error,'close_error':close_error,
        'phase_accounting':phase,'phase_IO':projected,'limits':deepcopy(LIMITS),
        'publication_scope':'Final summary write counted once; summary bytes/hash are independently inventoried externally.',
        'duration_scope':'Start/end envelopes include marker fsync, callable work and scheduling; not isolated OS service times.',
        'scoring_authorized':False,'retention_certified':False,'historical_cause_established':False,'native_fit_established':False}
    body['sha256']=digest(body);raw=(canonical(body)+'\n').encode('ascii');path=Path(output)/FINAL_NAME
    def write():
        with path.open('xb',buffering=0) as stream:
            require(stream.write(raw)==len(raw),'complete final phase summary');stream.flush();os.fsync(stream.fileno())
    sink.io('phase_final_summary_write',write)
    require(sink.snapshot()==projected,'actual final phase publication costs reconcile')
    return body

def read_events(path):
    entries=[];total=0
    with Path(path).open('rb') as stream:
        for raw in stream:
            total+=len(raw);require(raw.endswith(b'\n') and len(raw)<=LIMITS['maximum_marker_bytes']
                and total<=LIMITS['maximum_ledger_bytes'] and len(entries)<LIMITS['maximum_markers'],'bounded complete phase JSONL')
            def pairs(items):
                value={}
                for key,item in items:require(key not in value,'duplicate phase JSON key');value[key]=item
                return value
            entries.append(json.loads(raw.decode('ascii'),object_pairs_hook=pairs,
                parse_constant=lambda text:(_ for _ in ()).throw(ValueError('nonfinite phase JSON'))))
    return entries

def validate_events(entries,account):
    """Pure complete/pair/nesting/query admission; no model/decoder reads."""
    stack=[];starts={};operations={};children={};last_time=None;last_utc=None
    for item in entries:
        require(type(item) is dict and set(item)=={'name','phase'},'exact named phase marker')
        value=item['phase'];ordinal=value['ordinal'];name=value['operation'];event=value['event'];now=value['monotonic']
        require(type(ordinal) is int and ordinal>0 and type(name) is str and 0<len(name)<=1024
            and type(now) in (int,float) and math.isfinite(now),'typed finite phase identity/time')
        require(value['historical_cause_established'] is False,'no historical attribution')
        utc=datetime.fromisoformat(value['utc']);require(utc.tzinfo is not None,'aware phase UTC')
        require((last_time is None or now>=last_time) and (last_utc is None or utc>=last_utc),'phase chronological order')
        last_time,last_utc=now,utc
        require(item['name']=='phase_%04d_%s'%(ordinal,'start' if event=='attempt' else 'end'),'exact ordinal marker name')
        counter=operations.setdefault(name,counts())
        if event=='attempt':
            require(ordinal==len(starts)+1 and ordinal not in starts,'contiguous unique phase start')
            if stack:children.setdefault(stack[-1],[]).append(ordinal)
            starts[ordinal]=value;stack.append(ordinal);counter['attempted']+=1;counter['pending']+=1
        else:
            require(event=='returned' and value['error'] is None and stack and stack.pop()==ordinal
                and starts[ordinal]['operation']==name,'complete exact nested phase return; failures/pending denied')
            counter['pending']-=1;counter['returned']+=1
    require(not stack and complete_counts(operations) and account['operations']==operations
        and account['last_ordinal']==len(starts) and account['publication_errors']==[],'complete phase ledger conservation')
    require(entries and account['last_phase']==entries[-1]['phase'],'final phase snapshot matches ledger')
    ends={v['phase']['ordinal']:v['phase'] for v in entries if v['phase']['event']=='returned'}
    guard_count=init_count=memory_count=0
    for ordinal,start in starts.items():
        name=start['operation']
        if not name.startswith('guard_') or name in ('guard_cuda_is_initialized','guard_cuda_mem_get_info'):continue
        guard_count+=1;direct=children.get(ordinal,[]);names=[starts[x]['operation'] for x in direct]
        require(names in ([],['guard_cuda_is_initialized'],['guard_cuda_is_initialized','guard_cuda_mem_get_info']),
            'original guard query path; no additional/recursive query')
        if direct:
            initialized=ends[direct[0]]['result'];require(type(initialized) is bool,'typed observed guard initialization')
            require(len(direct)==(2 if initialized else 1),'false initialization skips memory; true queries once')
            init_count+=1;memory_count+=len(direct)-1
    require(operations.get('guard_cuda_is_initialized',counts())['returned']==init_count
        and operations.get('guard_cuda_mem_get_info',counts())['returned']==memory_count,'every guard CUDA query belongs to one original guard')
    return {'phases':len(starts),'markers':len(entries),'original_guards':guard_count,'is_initialized_queries':init_count,'guard_memory_queries':memory_count}

def validate_collection_events(entries,final,worker,journal_events,expected_identity,*,fixture=False):
    """Early phase admission only; independent full raw/model/resource audit remains required."""
    require(final['kind']==KIND and final['sha256']==digest({k:v for k,v in final.items() if k!='sha256'}),'final phase body checksum')
    require(final['identity']==expected_identity and final['status']=='COMPLETE_PHASE_OBSERVATION'
        and worker['status']=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED' and worker.get('error') is None,
        'same externally selected identity and paired complete raw/final statuses')
    require(expected_identity['execution_scope']==('CPU_FIXTURE_PHASE_COLLECTION_PROTOCOL' if fixture else 'OWNED_SHORT_M0_PHASE_OBSERVED_RAW_COLLECTION'),
        'explicit fixture/native phase schema; no authority conversion')
    require(worker['scope']==expected_identity['purpose'] and worker['config_sha256']==expected_identity['config_sha256']
        and worker['source_commit']==expected_identity['source_commit'],'raw and final external scope/config/commit joins')
    require(final['limits']==LIMITS and final['primary_error'] is final['terminal_error'] is final['close_error'] is None,
        'closed phase errors/limits')
    io=final['phase_IO'];require(io['closed'] is True and complete_counts(io['operations'])
        and set(io['operations'])=={'phase_ledger_create','phase_marker_write','phase_ledger_close','phase_final_summary_write'},
        'closed complete exact phase IO roster')
    require(io['markers_returned']==len(entries)==io['operations']['phase_marker_write']['returned']
        and all(io['operations'][name]['returned']==1 for name in ('phase_ledger_create','phase_ledger_close','phase_final_summary_write')),
        'all markers and single create/close/final writes retained')
    byte_count=sum(len((canonical(item)+'\n').encode('ascii')) for item in entries)
    require(io['marker_bytes_attempted']==io['marker_bytes_returned']==byte_count,'complete marker byte conservation')
    proof=validate_events(entries,final['phase_accounting'])
    begins=[v['phase'] for v in entries if v['phase']['event']=='attempt']
    mains=[v['operation'] for v in begins if v['operation'].startswith('main_')]
    require(mains==(['main_initial_config_preflight'] if not fixture else [])+['main_'+name for name in MAIN],
        'exact source-defined main operation order, no missing/extra/reordered phase')
    apis=[v['operation'][4:] for v in begins if v['operation'].startswith('api_')]
    primitives=[v['operation'][10:] for v in begins if v['operation'].startswith('primitive_')]
    attempts=[v for v in journal_events if v['event']=='attempt'];returned=[v for v in journal_events if v['event']=='returned']
    require(apis==primitives==[v['operation'] for v in attempts],'actual primitive/API/Journal attempt order joins')
    require([v['id'] for v in attempts]==list(range(1,len(attempts)+1)) and len(returned)==len(attempts)
        and not any(v['event']=='failed' for v in journal_events),'complete actual Journal attempt/return roster')
    by_id={v['id']:v for v in returned};require(len(by_id)==len(returned)
        and all(by_id[v['id']]['operation']==v['operation'] for v in attempts),'exact Journal same-id returned operation')
    actual={}
    for v in attempts:
        name=v['operation'];actual.setdefault(name,counts());actual[name]['attempted']+=1;actual[name]['returned']+=1
    require(worker['operation_accounting']['operations']==actual and worker['operation_accounting']['last_call']==len(attempts),
        'worker actual API accounting joins Journal')
    require(set(actual)<=set(API_FIXED)|{'inference_forward'},'closed original collector API names')
    if not fixture:
        expected={**API_FIXED,'inference_forward':worker['collection_costs']['attempted_forwards']}
        require(actual=={name:{'attempted':n,'returned':n,'failed':0,'pending':0} for name,n in expected.items()},
            'exact native capture plus original322 API calls; fixture constructors never native proof')
    for name,cost in actual.items():
        require(final['phase_accounting']['operations']['api_'+name]==cost
            and final['phase_accounting']['operations']['primitive_'+name]==cost,'exact phase and actual primitive costs '+name)
    publications=[v['operation'][12:] for v in begins if v['operation'].startswith('publication_')]
    require(['stage_'+name+'.json' for name in publications]==[v['path'] for v in journal_events if v['event']=='measurement_preserved'],
        'phase publication order matches every original durable raw fragment')
    return {**proof,'actual_API_calls':len(attempts),'fixture_only':fixture,
        'full_raw_decoder_model_resource_audit_complete':False,'scoring_authorized':False}
