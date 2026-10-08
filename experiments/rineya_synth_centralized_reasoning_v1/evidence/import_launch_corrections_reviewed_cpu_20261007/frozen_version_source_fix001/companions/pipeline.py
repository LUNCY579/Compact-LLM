"""Injected CPU-only one-prefix core; no native execution interfaces."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import importlib.util,os,time
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('prefix_input_exact_contract',HERE/'contract.py')
C=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C)
C.require(Path(C.__file__).resolve()==(HERE/'contract.py').resolve(),'exact contract module origin')
def counts():return {'attempted':0,'returned':0,'failed':0,'pending':0}
def projected(v):
    nodes=[0]
    def inner(v,depth=0):
        nodes[0]+=1
        if depth>4 or nodes[0]>1024:return {'projection_truncated':True}
        if v is None or type(v) in (bool,int):return v
        if type(v) is str:return v[:1024]
        if type(v) is float:
            import math
            return v if math.isfinite(v) else {'unsupported_nonfinite':True}
        if type(v) in (tuple,list) and len(v)<=512:return [inner(x,depth+1) for x in v]
        if type(v) is dict and len(v)<=32:return {k:inner(x,depth+1) for k,x in v.items() if type(k) is str and len(k)<=128}
        return {'opaque_return_not_inspected':True}
    result=inner(v)
    if len(C.canonical(result).encode('ascii'))>8192:return {'bounded_projection_truncated':True}
    return result
def error_value(error):
    try:message=str(error)
    except BaseException:return {'type':type(error).__name__,'message':'<original-message-stringification-failed>','message_unavailable':True}
    if len(message)>4096:return {'type':type(error).__name__,'message':message[:4096],'message_truncated':True}
    return {'type':type(error).__name__,'message':message}
class Ledger:
    def __init__(self,output):
        self.output=Path(output);self.ops={};self.markers=0;self.bytes=0;self.serial=0;self.io={};self.closed=False;self.errors=[]
        self.stream=self.action('ledger_create',lambda:(self.output/'PHASE_OBSERVATIONS.jsonl').open('xb'))
    def action(self,name,fn):
        item=self.io.setdefault(name,counts());item['attempted']+=1;item['pending']+=1
        try:v=fn()
        except BaseException:item['pending']-=1;item['failed']+=1;raise
        item['pending']-=1;item['returned']+=1;return v
    def marker(self,name,event,ordinal,value=None,error=None):
        data={'operation':name,'event':event,'ordinal':ordinal,'monotonic':time.monotonic(),'utc':datetime.now(timezone.utc).isoformat()}
        if event!='attempt':data.update(result=projected(value),error=None if error is None else error_value(error))
        raw=(C.canonical(data)+'\n').encode('ascii')
        def write():
            C.require(not self.closed and self.markers<C.LIMITS['maximum_markers'] and len(raw)<=C.LIMITS['maximum_marker_bytes']
                and self.bytes+len(raw)<=C.LIMITS['maximum_ledger_bytes'],'bounded phase writer')
            C.require(self.stream.write(raw)==len(raw),'complete marker write');self.stream.flush();os.fsync(self.stream.fileno());self.markers+=1;self.bytes+=len(raw)
        self.action('marker_write',write)
    def call(self,name,fn):
        item=self.ops.setdefault(name,counts());item['attempted']+=1;item['pending']+=1;self.serial+=1;oid=self.serial
        try:self.marker(name,'attempt',oid)
        except BaseException as error:item['pending']-=1;item['failed']+=1;self.errors.append(error_value(error));raise
        try:v=fn()
        except BaseException as error:
            item['pending']-=1;item['failed']+=1
            try:self.marker(name,'failed',oid,error=error)
            except BaseException as publication_error:self.errors.append(error_value(publication_error))
            raise
        item['pending']-=1;item['returned']+=1
        try:self.marker(name,'returned',oid,value=v)
        except BaseException as error:self.errors.append(error_value(error));raise
        return v
    def durable(self,name,value):
        raw=(C.canonical(value)+'\n').encode('ascii');C.require(len(raw)<=C.LIMITS['maximum_worker_json_bytes'],'bounded report JSON')
        with (self.output/name).open('xb') as stream:C.require(stream.write(raw)==len(raw),'complete report write');stream.flush();os.fsync(stream.fileno())
    def publish(self,name,value):return self.call('publication_'+name,lambda:self.durable(name+'.json',value))
    def close(self):
        if not self.closed:self.action('ledger_close',self.stream.close);self.closed=True
    def finish(self,report):
        self.close();projected_io=deepcopy(self.io);projected_io['final_summary_write']={'attempted':1,'returned':1,'failed':0,'pending':0}
        summary={'scope':C.CPU_SCOPE,'status':report['status'],'operations':deepcopy(self.ops),'IO':projected_io,'markers':self.markers,'marker_bytes':self.bytes,'closed':self.closed,'publication_errors':self.errors,'native_PASS':False}
        self.action('final_summary_write',lambda:self.durable('PHASE_ACCOUNTING_FINAL.json',summary))
        C.require(self.io==projected_io,'final summary projected and actual IO reconcile')
        return summary
class GuardCUDA:
    def __init__(self,actual,ledger):self.actual=actual;self.ledger=ledger
    def query(self,name,fn):
        C.require(self.ledger.ops.get(name,counts())['attempted']<7,'declared original guard query count before actual call')
        return self.ledger.call(name,fn)
    def is_initialized(self):return self.query('guard_cuda_is_initialized',self.actual.is_initialized)
    def mem_get_info(self):return self.query('guard_cuda_mem_get_info',self.actual.mem_get_info)
class GuardView:
    def __init__(self,torch,ledger):self.cuda=GuardCUDA(torch.cuda,ledger)
def run_native(*args,**kwargs):return C.native_entry(*args,**kwargs)
def dispatch(*args,**kwargs):return C.native_entry(*args,**kwargs)
def run_cpu(api,guard,output,*,fixture_identity=C.CPU_SCOPE,source_pins=None,runtime_pins=None):
    C.require(fixture_identity==C.CPU_SCOPE,'explicit CPU injected identity; no native authority conversion')
    prefix=C.load_input()['prefix_ids'];ledger=Ledger(output);torch=None;tensors={'ids':None,'mask':None};primary=None;secondary=[];report={'predicate_metadata':{}};initial=None
    def call(name,fn):return ledger.call(name,fn)
    def gated(name):return call('guard_'+name,lambda:guard(name,None if torch is None else GuardView(torch,ledger)))
    def check(name,fn):
        result=call(name,fn);report['predicate_metadata'][name]=projected(result);C.require(result,'exact unaltered input tensor' if name.startswith('input_') else 'exact tensor attention mask')
    def shape_ids():
        v=tuple(tensors['ids'].shape);report['predicate_metadata']['ids_shape']=projected(v);return v==(1,len(prefix))
    def shape_mask():
        a=tuple(tensors['mask'].shape);b=tuple(tensors['ids'].shape);report['predicate_metadata'].update(mask_shape=projected(a),ids_shape_second_read=projected(b));return a==b
    def dtype(which):
        v=tensors[which].dtype==torch.long;report['predicate_metadata'][which+'_dtype_long']=projected(v);return v
    def device(which):
        v=str(tensors[which].device);report['predicate_metadata'][which+'_device']=v;return v=='cuda:0'
    def values(which):
        v=tensors[which].tolist();report['predicate_metadata'][which+'_host_values_sha256']=C.digest(v);return v==([prefix] if which=='ids' else [[1]*len(prefix)])
    try:
        supplied=call('initial_config_preflight',api['config_preflight']);C.require(type(supplied) is dict and supplied,'exact initial config metadata required');initial=deepcopy(supplied);initial_hash=C.digest(initial)
        call('worker_ownership_verify',api['ownership_verify']);ledger.publish('ownership_verified',{'scope':C.CPU_SCOPE})
        gated('pre_import');C.require(call('disk_admission',api['disk_free'])>=4*C.GIB,'unchanged4GiB disk admission')
        torch=call('torch_import',api['torch_import']);gated('after_import');C.require(call('torch_version_read',api['torch_version_read'])==api['runtime_version'],'registered Torch version')
        growth=call('post_import_growth_observation',api['growth_observation']);C.require(set(growth)=={'available_bytes','worker_RSS_bytes'} and all(type(v)is int and v>=0 for v in growth.values()) and growth['worker_RSS_bytes']<=2*C.GIB and growth['available_bytes']>=1.25*C.GIB+max(0,2*C.GIB-growth['worker_RSS_bytes']),'unchanged full post-import growth reserve')
        gated('after_growth');gated('before_context_init');call('cuda_context_init',torch.cuda.init);gated('after_context_init')
        mem=call('pre_input_direct_CUDA_mem_get_info',torch.cuda.mem_get_info);C.require(type(mem)is tuple and len(mem)==2 and all(type(v)is int for v in mem) and 4*C.GIB<=mem[0]<=mem[1],'unchanged4GiB preload GPU admission')
        call('original_precision_thread_configuration',api['configure']);call('Python_seed_101',lambda:api['python_seed'](101));call('Torch_CPU_seed_101',lambda:torch.manual_seed(101));call('Torch_CUDA_seed_all_101',lambda:torch.cuda.manual_seed_all(101));call('CUDA_reset_peak_memory_stats',torch.cuda.reset_peak_memory_stats)
        gated('before_input_construction');tensors['ids']=call('input_tensor_create',lambda:torch.tensor([list(prefix)],dtype=torch.long,device='cuda:0'));tensors['mask']=call('mask_ones_like_create',lambda:torch.ones_like(tensors['ids']))
        check('input_is_tensor',lambda:torch.is_tensor(tensors['ids']));check('input_shape_matches',shape_ids);check('input_dtype_long',lambda:dtype('ids'));check('input_device_cuda0',lambda:device('ids'));check('input_exact_tolist',lambda:values('ids'))
        check('mask_is_tensor',lambda:torch.is_tensor(tensors['mask']));check('mask_shape_matches',shape_mask);check('mask_dtype_long',lambda:dtype('mask'));check('mask_device_cuda0',lambda:device('mask'));check('mask_exact_tolist',lambda:values('mask'))
        gated('after_input_validation');meta=report['predicate_metadata'];report['input_summary']={'prefix_ids_sha256':C.digest(prefix),'observed_ids_host_values_sha256':meta['ids_host_values_sha256'],'mask_values_sha256':meta['mask_host_values_sha256'],'shape':meta['ids_shape'],'mask_shape':meta['mask_shape'],'second_ids_shape':meta['ids_shape_second_read'],'dtype_long_observation':meta['ids_dtype_long'],'mask_dtype_long_observation':meta['mask_dtype_long'],'device':meta['ids_device'],'mask_device':meta['mask_device'],'logical_tensor_bytes':6752,'logical_bytes_not_allocator_reservation':True}
    except BaseException as error:
        primary=error_value(error)
        try:report['exception_frames']=call('bounded_original_exception_frames_capture',lambda:C.frame_helper().capture_exception_frames(error,source_pins or {},runtime_pins or {}))
        except BaseException:report['exception_frame_diagnostic_failed']={'status':'BOUNDED_FRAME_DIAGNOSTIC_FAILED_PRIMARY_ERROR_PRESERVED'}
    finally:
        for which,name in (('ids','input_reference_release'),('mask','mask_reference_release')):
            try:
                if tensors[which] is not None:call(name,lambda which=which:tensors.__setitem__(which,None))
            except BaseException as error:secondary.append({'operation':name,'error':error_value(error)})
        try:
            if initial is not None:C.require(C.digest(call('final_config_preflight',api['config_preflight']))==initial_hash,'immutable original final source/config identity')
            if torch is not None:
                initialized=call('terminal_cuda_is_initialized',torch.cuda.is_initialized);C.require(type(initialized)is bool,'typed terminal init');terminal={'is_initialized':initialized}
                if initialized:
                    a=call('terminal_cuda_max_memory_allocated',torch.cuda.max_memory_allocated);b=call('terminal_cuda_max_memory_reserved',torch.cuda.max_memory_reserved);mem=call('terminal_cuda_mem_get_info',torch.cuda.mem_get_info)
                    C.require(type(a)is int and type(b)is int and 0<=a<=b,'typed terminal allocator amounts')
                    C.require(type(mem)is tuple and len(mem)==2 and all(type(v)is int for v in mem) and .5*C.GIB<=mem[0]<=mem[1],'typed terminal free/total and unchangedfloor');terminal.update(max_memory_allocated=a,max_memory_reserved=b,mem_get_info=list(mem))
                report['terminal_allocator']=terminal
            if primary is None and not secondary:gated('complete')
        except BaseException as error:secondary.append({'operation':'final_config_or_terminal','error':error_value(error)})
        message_exact=primary is None or not(primary.get('message_truncated',False) or primary.get('message_unavailable',False))
        report.update(scope=C.CPU_SCOPE,status='CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED' if primary is None and not secondary and not ledger.errors else 'CPU_PREFIX_INPUT_CHECKS_FAILED',error=primary,primary_message_exact=message_exact,diagnostic_evidence_incomplete=not message_exact,terminal_error=secondary[0]['error'] if secondary else None,secondary_errors=secondary,logical_tensor_bytes=6752,requires_complete_final_phase_accounting=True,model_loaded=False,model_constructed=False,forwards=0,backwards=0,optimizer_steps=0,decodes=0,grades=0,M0_membership=False,native_launch_authorized=False,scoring_authorized=False,native_fit_established=False,cause_established=False)
        try:ledger.publish('input_check_summary',report.get('input_summary',{'partial_failure':True}));ledger.publish('worker_result',report)
        except BaseException as error:report['status']='CPU_PREFIX_INPUT_CHECKS_FAILED';report['publication_error']=error_value(error)
        try:
            if report['status']=='CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED':C.require([n for n in ledger.ops if not n.startswith('guard_')]==list(C.OPS),'exact code-driven declared success operation order')
            summary=ledger.finish(report);report['final_phase_summary']=summary
        except BaseException as error:report['status']='CPU_PREFIX_INPUT_CHECKS_FAILED';report['final_summary_error']=error_value(error)
        report['operation_accounting']=deepcopy(ledger.ops);report['IO_accounting']=deepcopy(ledger.io);report['publication_errors']=deepcopy(ledger.errors)
    return report
